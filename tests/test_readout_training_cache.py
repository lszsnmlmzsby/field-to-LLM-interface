"""Tiny local cache-contract checks, not remote Qwen/A800 performance evidence."""
from __future__ import annotations

import copy
import pytest
import torch

from test_multisource_readout import config
from test_point_readout import CharacterTokenizer
from scripts import train_point_readout as trainer
from tensor_compression.downstream import multisource_readout_data as multi
from tensor_compression.downstream import point_readout_data
from tensor_compression.downstream.field_query_program import PROGRAM_TASKS
from tensor_compression.downstream.readout_training_cache import (
    PreparedTrainingDataset, prepare_training_cache, training_token_identity, validate_training_cache,
)


@pytest.fixture
def cache_inputs(config, tmp_path):
    recipe = copy.deepcopy(config)
    recipe["data"]["sources"] = recipe["data"]["sources"][:1]  # Complete tiny INS fixture only.
    recipe["generation"].update(train_states=12, eval_states=24, tasks=list(PROGRAM_TASKS),
        query_program={"enabled": True, "tasks_per_state": 14, "diagnostic_states_per_source": 0,
                       "read_count_max": 8, "line_count_max": 5, "region_value_max_cells": 8,
                       "statistic_max_cells": 16})
    qa = tmp_path / "qa"
    multi.build_multisource_dataset(recipe, qa)
    tokenizer = CharacterTokenizer()
    training = {"max_prompt_tokens": 4096, "max_target_tokens": 256}
    identity = training_token_identity(tokenizer, text_encoding_version=trainer.TEXT_ENCODING_VERSION,
                                       system_message=trainer.SYSTEM_MESSAGE)
    return qa, tmp_path / "prepared", tokenizer, training, identity


def prepare(inputs, **kwargs):
    qa, output, tokenizer, training, identity = inputs
    return prepare_training_cache(qa, output, tokenizer, training,
        encode_record=kwargs.pop("encode_record", trainer.encode_training_record), token_identity=identity, **kwargs)


def test_streaming_cache_matches_every_field_token_and_qa_order(cache_inputs, monkeypatch):
    qa, output, tokenizer, training, identity = cache_inputs
    # The new path must not call the eager million-dictionary JSONL loader.
    with monkeypatch.context() as patch:
        patch.setattr(point_readout_data, "read_jsonl", lambda *_: pytest.fail("eager QA JSONL load"))
        manifest = prepare(cache_inputs)
    original = multi.MultiSourcePointReadoutDataset(qa, "train", cache_size=2)
    prepared = PreparedTrainingDataset(output, qa_dir=qa, token_identity=identity, training=training)
    try:
        assert prepared.identity == original.identity
        assert prepared.metadata == original.metadata
        assert not isinstance(prepared.records, list)
        assert len(prepared) == len(original)
        assert manifest["token_report"]["status"] == "ok"
        assert manifest["source_audit"]["all_training_questions_replayed"]
        for i in range(len(original)):
            original_row, original_field = original[i]
            row, field = prepared[i]
            assert all(row[k] == original_row[k] for k in ("qa_id", "state_ref", "grid_shape", "task_type"))
            assert row["_prepared_index"] == i
            assert torch.equal(field, original_field)
            assert field.dtype == torch.float32
            expected = trainer.encode_training_record(original_row, tokenizer, training)
            prompt, answer = prepared.token_pair(i)
            assert prompt.tolist() == expected[0]
            assert answer.tolist() == expected[1]
            assert prepared.token_lengths[i] == len(prompt) + len(answer)
            assert not prompt.flags.writeable
            assert prepared.field_array(i).flags.writeable
        assert prepared.records[-1]["qa_id"] == original.records[-1]["qa_id"]
        assert [r["qa_id"] for r in prepared.records[:3]] == [r["qa_id"] for r in original.records[:3]]
    finally:
        prepared.close()
        original.close()


def test_cache_identity_and_budget_changes_fail(cache_inputs):
    qa, output, tokenizer, training, identity = cache_inputs
    prepare(cache_inputs)
    changed = dict(identity, text_encoding_version="changed")
    with pytest.raises(ValueError, match="tokenizer/encoding"):
        validate_training_cache(output, token_identity=changed)
    with pytest.raises(ValueError, match="token budgets"):
        validate_training_cache(output, training=dict(training, max_target_tokens=257))
    with pytest.raises(FileExistsError, match="nonempty"):
        prepare(cache_inputs)
    with pytest.raises(ValueError, match="separate"):
        prepare_training_cache(qa, qa / "derived", tokenizer, training,
            encode_record=trainer.encode_training_record, token_identity=identity)


@pytest.mark.parametrize("target", ["tokens.bin", "index.npy", "fields.npy"])
def test_tampered_derived_arrays_are_rejected(cache_inputs, target):
    _, output, _, _, _ = cache_inputs
    prepare(cache_inputs)
    path = output / target
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        previous = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([previous[0] ^ 1]))
    with pytest.raises(ValueError, match="cache file changed"):
        validate_training_cache(output)


def test_failed_preparation_is_never_trainable_and_preserves_sources(cache_inputs):
    qa, output, _, _, _ = cache_inputs
    original_metadata = (qa / "metadata.json").read_bytes()
    def broken(*_):
        raise RuntimeError("intentional encoder failure")
    with pytest.raises(RuntimeError, match="encoder failure"):
        prepare(cache_inputs, encode_record=broken)
    assert (output / ".build_in_progress").is_file()
    assert not (output / "manifest.json").exists()
    assert (qa / "metadata.json").read_bytes() == original_metadata
    with pytest.raises(ValueError, match="incomplete"):
        PreparedTrainingDataset(output, validate=False)
    with pytest.raises(FileExistsError, match="nonempty"):
        prepare(cache_inputs)


def test_changed_qa_and_original_source_are_rejected(cache_inputs):
    qa, output, _, _, _ = cache_inputs
    manifest = prepare(cache_inputs)
    source = next(iter(manifest["metadata"]["source"]["sources"].values()))
    from pathlib import Path
    source_path = Path(source["path_hint"])
    source_path.touch()
    with pytest.raises(ValueError, match="Source file changed"):
        validate_training_cache(output)
    with (qa / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="Dataset file changed"):
        validate_training_cache(output)


def test_token_identity_detects_template_and_system_changes():
    tokenizer = CharacterTokenizer()
    original = training_token_identity(tokenizer, text_encoding_version="v1", system_message="system")
    tokenizer.chat_template += "-modified"
    assert training_token_identity(tokenizer, text_encoding_version="v1", system_message="system") != original
    assert training_token_identity(CharacterTokenizer(), text_encoding_version="v1", system_message="other") != original
