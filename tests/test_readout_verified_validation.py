"""Launch-local verification reuse keeps validation semantics and compact state RAM."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from test_multisource_readout import config
from test_readout_training_cache import cache_inputs, prepare
from tensor_compression.downstream import multisource_readout_data as multi
from tensor_compression.downstream import point_readout_data
from tensor_compression.downstream.readout_training_cache import PreparedTrainingDataset, validate_training_cache


@pytest.mark.parametrize("split", ["val", "test"])
def test_verified_validation_keeps_all_rows_audit_and_identity(cache_inputs, monkeypatch, split):
    qa, output, _, _, _ = cache_inputs
    prepare(cache_inputs)
    manifest = validate_training_cache(output)
    original = multi.MultiSourcePointReadoutDataset(qa, split)
    eager = point_readout_data.read_jsonl
    def guarded_read(path):
        assert Path(path).name != "states.jsonl", "states must be streamed"
        return eager(path)
    with monkeypatch.context() as patch:
        patch.setattr(multi, "sha256_file", lambda *_: pytest.fail("duplicate whole-file hashing"))
        patch.setattr(point_readout_data, "read_jsonl", guarded_read)
        verified = multi.MultiSourcePointReadoutDataset(qa, split, verified_metadata=manifest["metadata"])
    try:
        assert verified.identity == original.identity
        assert verified.records == original.records
        assert set(verified.states) == {row["state_ref"] for row in verified.records}
        assert len(verified.states) < len(original.states)
        assert verified.audit() == original.audit()
        for i in range(len(original)):
            row, field = original[i]
            new_row, new_field = verified[i]
            assert row == new_row and torch.equal(field, new_field)
    finally:
        verified.close()
        original.close()


def test_verified_validation_rejects_metadata_or_source_change(cache_inputs):
    qa, output, _, _, _ = cache_inputs
    manifest = prepare(cache_inputs)
    changed = copy.deepcopy(manifest["metadata"])
    changed["counts"]["train"] += 1
    with pytest.raises(ValueError, match="verified launch metadata"):
        multi.MultiSourcePointReadoutDataset(qa, "val", verified_metadata=changed)
    source = next(iter(manifest["metadata"]["source"]["sources"].values()))
    Path(source["path_hint"]).touch()
    with pytest.raises(ValueError, match="Source file changed"):
        multi.MultiSourcePointReadoutDataset(qa, "val", verified_metadata=manifest["metadata"])


@pytest.mark.parametrize("change", ["trajectory", "duplicate"])
def test_streamed_validation_checks_even_unreferenced_training_states(cache_inputs, change):
    qa, output, _, _, _ = cache_inputs
    manifest = prepare(cache_inputs)
    states = [json.loads(line) for line in (qa / "states.jsonl").read_text(encoding="utf-8").splitlines()]
    train = next(state for state in states if state["split"] == "train")
    if change == "trajectory":
        train["sample_index"] = -100
        error = "trajectory partition"
    else:
        states.append(train)
        error = "Duplicate state references"
    (qa / "states.jsonl").write_text("".join(json.dumps(row) + "\n" for row in states), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        multi.MultiSourcePointReadoutDataset(qa, "val", verified_metadata=manifest["metadata"])
    # The unchanged public/default path still performs complete file verification.
    with pytest.raises(ValueError, match="Dataset file changed"):
        multi.MultiSourcePointReadoutDataset(qa, "val")


def test_prepared_shape_columns_are_exact_readonly_views(cache_inputs):
    _, output, _, _, _ = cache_inputs
    prepare(cache_inputs)
    data = PreparedTrainingDataset(output, validate=False)
    try:
        heights, widths = data.records.shape_columns
        prompt, answer = data.token_pair(0)
        lengths = data.token_lengths
        expected_prompt, expected_answer = prompt.tolist(), answer.tolist()
        expected_shapes = list(zip(heights.tolist(), widths.tolist()))
        assert not heights.flags.writeable and not widths.flags.writeable
        assert expected_shapes == [tuple(row["grid_shape"]) for row in data.records]
    finally:
        data.close()
    # Error tracebacks and the trainer's length table can outlive the dataset.
    # Their views must keep the mapping alive instead of accessing freed memory.
    assert prompt.tolist() == expected_prompt and answer.tolist() == expected_answer
    assert list(zip(heights.tolist(), widths.tolist())) == expected_shapes
    assert int(lengths[0]) == len(prompt) + len(answer)
    data.close()  # idempotent
