from __future__ import annotations

import copy
import json
import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from tensor_compression.downstream.point_readout import (
    TASKS, LOCATION_TASKS, build_prompt, make_record, normalize_field, oracle_answer, parse_answer, query_points,
    sample_spec, score_answer, summarize_scores, validate_record,
)
from tensor_compression.downstream.point_readout_data import (
    PointReadoutDataset, build_dataset, load_config, write_json,
)
from scripts.score_point_readout import score_predictions


@pytest.fixture
def z():
    return normalize_field(torch.arange(120).reshape(10, 12).float())[0]


@pytest.mark.parametrize("task", TASKS)
def test_truth_replay_and_reference_scoring(z, task):
    spec = sample_spec(task, z.shape, random.Random(2), z=z)
    row = make_record("state", z, task, spec)
    validate_record(row, z)
    assert score_answer(row, row["answer"])["all_correct"]
    row["oracle"]["values"][0] += 1
    with pytest.raises(ValueError, match="replay|canonical"):
        validate_record(row, z)


def test_region_statistics_use_subregion_of_whole_grid():
    field = torch.tensor([[90., 90., 90.], [90., 1., 3.], [90., 5., 7.]])
    spec = {"start": [2, 2], "size": [2, 2]}
    expected = {"region_mean": 4., "region_std": 5. ** .5, "region_min": 1., "region_max": 7.}
    for task, value in expected.items():
        values, coordinates = oracle_answer(field, task, spec)
        assert values == pytest.approx([value])
        assert coordinates == []
        row = make_record("state", field, task, spec)
        validate_record(row, field)


@pytest.mark.parametrize("task", LOCATION_TASKS)
def test_coordinate_ties_are_complete_pairs_not_fuzzy_values(task):
    field = torch.tensor([[9., 9., 9.], [9., 1., 4.], [9., 4., 1.]])
    spec = {"start": [2, 2], "size": [2, 2], "target": 1.}
    row = make_record("state", field, task, spec)
    correct = [[2, 3], [3, 2]] if task == "region_argmax" else [[2, 2], [3, 3]]
    assert row["oracle"]["valid_coordinates"] == correct
    for pair in correct:
        score = score_answer(row, json.dumps(pair))
        assert score["all_correct"] and score["scoring_units"] == 1
        assert score["absolute_errors"] == []
    wrong = [correct[0][0], correct[1][1]]
    assert score_answer(row, json.dumps(wrong))["correct_values"] == 0
    for pair in ([2.1, 2], [0, 2], [4, 2]):
        assert score_answer(row, json.dumps(pair))["error"] == "invalid_coordinate"
    assert score_answer(row, "[2, 2, 3, 3]")["error"] == "wrong_value_count"


def test_nearest_value_uses_visible_rounded_target():
    field = torch.tensor([[.14, .11], [.9, .8]])
    spec = {"start": [1, 1], "size": [2, 2], "target": .1}
    row = make_record("state", field, "nearest_value_location", spec)
    assert row["oracle"]["values"] == [1, 2]
    assert "0.1" in row["question"]
    sampled = sample_spec("nearest_value_location", [2, 2], random.Random(1), z=field)
    assert sampled["target"] == round(sampled["target"], 1)


def test_coordinate_prompt_uses_no_oracle(z):
    row = make_record("state", z, "region_argmax", {"start": [1, 1], "size": [2, 2]})
    prompt = build_prompt(row)
    row["oracle"] = {"values": "SECRET", "valid_coordinates": "SECRET"}
    row["answer"] = "SECRET"
    assert build_prompt(row) == prompt


def test_region_and_vertical_line_order():
    assert query_points("region_values", {"start": [2, 3], "size": [2, 2]}, [5, 7]) == [
        [2, 3], [2, 4], [3, 3], [3, 4]]
    assert query_points("line_profile", {"start": [1, 7], "step": [1, 0], "count": 3}, [5, 7]) == [
        [1, 7], [2, 7], [3, 7]]


@pytest.mark.parametrize("points", [[[0, 1]], [[11, 1]], [[1, 13]], [[1.0, 2]], [[True, 1]]])
def test_coordinates_are_strictly_one_based(z, points):
    with pytest.raises(ValueError):
        make_record("state", z, "single_point", {"points": points})


def test_odd_shapes_and_boundary_reads():
    z = normalize_field(torch.arange(17 * 90).reshape(17, 90).float())[0]
    row = make_record("odd", z, "single_point", {"points": [[17, 90]]})
    validate_record(row, z)
    assert "17 by 90" in build_prompt(row)
    assert row["oracle"]["values"] == [float(z[-1, -1])]


def test_prompt_allowlist(z):
    row = make_record("state", z, "single_point", {"points": [[2, 3]]})
    prompt = build_prompt(row)
    poisoned = copy.deepcopy(row)
    poisoned.update(answer="LEAK", state_ref="LEAK", task_type="LEAK", secret="LEAK")
    poisoned["oracle"] = {"values": [999], "query_spec": "LEAK"}
    assert build_prompt(poisoned) == prompt
    assert "LEAK" not in build_prompt(poisoned)
    assert "Choices:" not in prompt and "Options:" not in prompt


@pytest.mark.parametrize("text", ["[NaN]", "[Infinity]", "[true]", "[[0.2]]", "[]", "0.2",
                                    "Answer: [0.2]", "[0.2] or [0.3]", "[0.2,]", "[1e999]"])
def test_reject_ambiguous_or_nonfinite_arrays(text):
    assert parse_answer(text)[1] is not None


def test_numeric_equivalence_and_order(z):
    row = make_record("state", z, "multi_point", {"points": [[1, 1], [10, 12]]})
    values = row["oracle"]["values"]
    assert score_answer(row, f"[ {values[0]:.2e}, {values[1]:.2e} ]")["all_correct"]
    assert not score_answer(row, json.dumps(list(reversed(values))))["all_correct"]
    missing = score_answer(row, json.dumps(values[:1]))
    assert not missing["valid"] and missing["correct_values"] == 0 and missing["missing_count"] == 1
    extra = score_answer(row, json.dumps(values + [3]))
    assert extra["error"] == "wrong_value_count" and extra["extra_count"] == 1
    assert score_answer(row, row["answer"], terminated=False)["error"] == "generation_truncated"


def test_tolerance_is_not_relative_to_target(z):
    row = make_record("state", z, "single_point", {"points": [[1, 1]]})
    value = row["oracle"]["values"][0]
    assert score_answer(row, json.dumps([value + .19]))["all_correct"]
    assert not score_answer(row, json.dumps([value + .21]))["all_correct"]
    with pytest.raises(ValueError, match="rounding"):
        make_record("state", z, "single_point", {"points": [[1, 1]]}, decimals=0, tolerance=.2)


def test_aggregate_keeps_invalid_answers_in_denominator(z):
    row = make_record("state", z, "single_point", {"points": [[1, 1]]})
    scores = [{"task_type": "single_point", "shape": "10x12", "field": "Vx",
               "score": score_answer(row, text)} for text in (row["answer"], "garbage")]
    metrics = summarize_scores(scores)
    assert metrics["valid_rate"] == .5
    assert metrics["point_accuracy"] == .5
    assert metrics["macro_task_point_accuracy"] == .5


def test_mixed_task_macro_gives_each_task_equal_weight(z):
    read = make_record("state", z, "region_values", {"start": [1, 1], "size": [2, 2]})
    locate = make_record("state", z, "region_argmax", {"start": [1, 1], "size": [2, 2]})
    rows = [{"task_type": row["task_type"], "shape": "10x12", "field": "Vx",
             "score": score_answer(row, answer)} for row, answer in ((read, read["answer"]), (locate, "[1, 1]"))]
    metrics = summarize_scores(rows)
    assert metrics["scoring_units"] == 5
    assert metrics["unit_accuracy"] == .8
    assert metrics["macro_task_answer_accuracy"] == .5
    assert metrics["by_answer_kind"]["coordinate"]["answer_accuracy"] == 0


@pytest.fixture
def data_fixture(tmp_path):
    source = tmp_path / "source.h5"
    # Generated fixture only, never a source family offered by the experiment.
    array = np.random.default_rng(5).normal(size=(20, 3, 24, 28)).astype("float32")
    with h5py.File(source, "w") as handle:
        handle.create_dataset("Vx", data=array)
        handle.create_dataset("Vy", data=array * 3 + 7)
    config = load_config(ROOT / "configs/field_to_llm_point_readout.yaml", "smoke")
    config["data"].update(fields=["Vx", "Vy"], train_shapes=["8x8", "8x12"], heldout_shapes=["9x11"])
    config["generation"].update(train_states=8, eval_states_per_shape=2)
    output = tmp_path / "qa"
    metadata = build_dataset(config, source, output)
    return source, output, config, metadata


def test_dataset_splits_replay_and_reproducibility(data_fixture, tmp_path):
    source, qa, config, metadata = data_fixture
    sets = [set(ids) for ids in metadata["trajectory_splits"].values()]
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i+1:])
    for split in ("train", "val", "test"):
        dataset = PointReadoutDataset(qa, source, split)
        try:
            assert dataset.audit()["questions"] == metadata["counts"][split]
        finally:
            dataset.close()
    repeated = build_dataset(config, source, tmp_path / "repeat")
    assert repeated["files"] == metadata["files"]
    with pytest.raises(FileExistsError):
        build_dataset(config, source, qa)


def test_selected_tasks_and_invalid_statistic_region(data_fixture, tmp_path):
    source, _, config, _ = data_fixture
    config = copy.deepcopy(config)
    config["generation"]["tasks"] = ["single_point", "region_mean", "region_argmax"]
    metadata = build_dataset(config, source, tmp_path / "selected")
    assert metadata["counts"]["train"] == 24
    assert metadata["task_counts"]["train"] == {name: 8 for name in config["generation"]["tasks"]}
    config["generation"]["statistic_region_shapes"] = [[30, 30]]
    with pytest.raises(ValueError, match="region must fit"):
        build_dataset(config, source, tmp_path / "bad_region")


def test_source_mutation_rejected(data_fixture):
    source, qa, _, _ = data_fixture
    dataset = PointReadoutDataset(qa, source, "val")
    try:
        state = dataset.states[dataset.records[0]["state_ref"]]
        r, c = state["top_left"]
        with h5py.File(source, "r+") as handle:
            handle[state["field"]][state["sample_index"], state["time_index"], r, c] += 1
        with pytest.raises(ValueError, match="changed"):
            dataset[0]
    finally:
        dataset.close()


def test_manifest_mutation_and_unfinished_build_rejected(data_fixture):
    source, qa, _, _ = data_fixture
    path = qa / "val.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        PointReadoutDataset(qa, source, "val")
    (qa / ".build_in_progress").touch()
    with pytest.raises(ValueError, match="did not finish"):
        PointReadoutDataset(qa, source, "val")


def test_independent_scorer_missing_duplicates_and_unknown_ids(data_fixture):
    source, qa, _, _ = data_fixture
    dataset = PointReadoutDataset(qa, source, "test")
    try:
        predictions = [{"qa_id": r["qa_id"], "prediction": r["answer"]} for r in dataset.records]
        assert score_predictions(dataset, predictions)["point_accuracy"] == 1
        assert score_predictions(dataset, predictions[:-1])["all_correct_rate"] < 1
        with pytest.raises(ValueError, match="duplicate"):
            score_predictions(dataset, predictions + predictions[:1])
        with pytest.raises(ValueError, match="Unknown"):
            score_predictions(dataset, [{"qa_id": "unknown", "prediction": "[0]"}])
    finally:
        dataset.close()


class CharacterTokenizer:
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2
    chat_template = "test-native-chat-v1"

    def __call__(self, text, add_special_tokens=True, truncation=False):
        return {"input_ids": ([1] if add_special_tokens else []) + [ord(c) + 3 for c in text]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(n - 3) if n >= 3 else f"<{n}>" for n in ids)

    def get_vocab(self):
        return {f"token_{i}": i for i in range(256)}

    def get_chat_template(self):
        return self.chat_template

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        ids = [self.bos_token_id]
        for message in messages:
            ids.extend(self(f"{message['role']}:\n{message['content']}", add_special_tokens=False)["input_ids"])
            ids.extend([self.eos_token_id, ord("\n") + 3])
        if add_generation_prompt:
            ids.extend(self("assistant:\n", add_special_tokens=False)["input_ids"])
        return ids


def tiny_components(config=None):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from scripts import train_point_readout as trainer
    torch.set_num_threads(1)
    torch.manual_seed(123)
    if config is None:
        config = load_config(ROOT / "configs/field_to_llm_point_readout.yaml", "smoke")
    config = copy.deepcopy(config)
    config["model"].update(torch_dtype="float32", gradient_checkpointing=False)
    config["spatial_adapter"].update(adapter_dim=16, adapter_layers=1, adapter_heads=2)
    config["cross_attention"].update(layers_1based=[1, 2], bridge_dim=16, heads=2, gate_init=.2)
    config["memory"].update(value_fourier_bands=1, value_hidden_dim=8)
    config["field_encoder"]["model"]["base_channels"] = 8
    config["training"].update(max_prompt_tokens=2048, max_target_tokens=96)
    llm = Qwen2ForCausalLM(Qwen2Config(vocab_size=256, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=4096, bos_token_id=1, eos_token_id=2, pad_token_id=0))
    args = trainer.model_args(config)
    encoder, spatial, _, _ = trainer.core.build_scratch_memory_components(args, (1, 16, 16))
    sidecar, _ = trainer.core.build_sidecar(llm, spatial, args, torch.device("cpu"), encoder)
    return trainer, llm, sidecar, CharacterTokenizer(), config


def test_answer_masking_and_no_truncation(z):
    from scripts.train_point_readout import training_tensors
    row = make_record("state", z, "multi_point", {"points": [[1, 1], [10, 12]]})
    tokenizer = CharacterTokenizer()
    training = {"max_prompt_tokens": 2048, "max_target_tokens": 96}
    ids, mask, labels = training_tensors([row], tokenizer, training)
    first = int(labels.ne(-100).nonzero()[0, 1])
    assert bool(labels[0, :first].eq(-100).all())
    assert int(labels[0, -1]) == tokenizer.eos_token_id
    supervised = labels[0, labels[0].ne(-100)].tolist()
    assert tokenizer.decode(supervised[:-1]) == row["answer"]
    assert "assistant:\n" in tokenizer.decode(ids[0, :first].tolist())
    assert torch.equal(ids[labels.ne(-100)], labels[labels.ne(-100)])
    assert bool(mask.all())
    with pytest.raises(ValueError, match="Prompt"):
        training_tensors([row], tokenizer, {**training, "max_prompt_tokens": 4})
    with pytest.raises(ValueError, match="Answer"):
        training_tensors([row], tokenizer, {**training, "max_target_tokens": 1})


def test_native_chat_is_required_and_prefix_must_match(z):
    from scripts.train_point_readout import encode_prompt, training_tensors
    row = make_record("state", z, "single_point", {"points": [[1, 1]]})
    tokenizer = CharacterTokenizer()
    tokenizer.chat_template = None
    with pytest.raises(ValueError, match="native chat template"):
        encode_prompt(row, tokenizer, 2048)
    class BrokenPrefix(CharacterTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            result = super().apply_chat_template(*args, **kwargs)
            if not kwargs["add_generation_prompt"]:
                result[0] = 99
            return result
    with pytest.raises(ValueError, match="prefix"):
        training_tensors([row], BrokenPrefix(), {"max_prompt_tokens": 2048, "max_target_tokens": 96})


def test_huggingface_native_template_and_supervision(z):
    """Exercise the real Transformers chat API, including special-token boundaries."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from scripts.train_point_readout import training_tensors, encode_prompt
    backend = Tokenizer(WordLevel({"[UNK]": 0, "<|endoftext|>": 1, "<|im_start|>": 2,
                                  "<|im_end|>": 3, "[": 4, "]": 5, "0": 6, ".": 7, "3": 8}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
        eos_token="<|im_end|>", pad_token="<|endoftext|>", additional_special_tokens=["<|im_start|>"])
    tokenizer.chat_template = ("{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}"
                               "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
    row = make_record("state", z, "single_point", {"points": [[1, 1]]})
    row["answer"] = "[0.3]"
    prompt = encode_prompt(row, tokenizer, 2048)
    ids, _, labels = training_tensors([row], tokenizer, {"max_prompt_tokens": 2048, "max_target_tokens": 96})
    assert ids[0, :len(prompt)].tolist() == prompt
    assert labels[0, :len(prompt)].eq(-100).all()
    assert ids[0, len(prompt):].tolist() == tokenizer("[0.3]", add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]


@pytest.mark.skipif(not os.environ.get("FIELD_TO_LLM_MODEL_DIR"), reason="Set FIELD_TO_LLM_MODEL_DIR for actual local tokenizer checks")
def test_local_qwen_chat_template_for_all_tasks(z):
    """Uses only local tokenizer/config assets, never downloads or loads model weights."""
    from types import SimpleNamespace
    from transformers import AutoTokenizer, GenerationConfig
    from scripts.train_point_readout import encode_prompt, training_tensors, generation_stop_ids
    path = os.environ["FIELD_TO_LLM_MODEL_DIR"]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    generation = GenerationConfig.from_pretrained(path, local_files_only=True)
    llm = SimpleNamespace(generation_config=generation)
    assert set(generation.eos_token_id) <= set(generation_stop_ids(llm, tokenizer))
    for task in TASKS:
        row = make_record("state", z, task, sample_spec(task, z.shape, random.Random(2), z=z))
        prompt = encode_prompt(row, tokenizer, 384)
        assert tokenizer.decode(prompt).endswith("<|im_start|>assistant\n")
        ids, _, labels = training_tensors([row], tokenizer, {"max_prompt_tokens": 384, "max_target_tokens": 96})
        assert ids[0, :len(prompt)].tolist() == prompt
        answer = labels[0, labels[0].ne(-100)].tolist()
        assert answer[-1] == tokenizer.eos_token_id
        assert tokenizer.decode(answer[:-1]) == row["answer"]


@pytest.mark.parametrize("eos_source", ["generation_config", "config"])
def test_generation_recognizes_alternative_eos_without_emitting_it(z, monkeypatch, eos_source):
    trainer, llm, sidecar, tokenizer, _ = tiny_components()
    llm.eval()
    sidecar.eval()
    alternate = 5  # A second EOS, different from tokenizer.eos_token_id.
    getattr(llm, eos_source).eos_token_id = [tokenizer.eos_token_id, alternate]
    row = make_record("state", z, "single_point", {"points": [[1, 1]]})
    sequence = iter(tokenizer("[0.3]", add_special_tokens=False)["input_ids"] + [alternate])
    def next_logits(hidden):
        logits = hidden.new_full((hidden.shape[0], 256), -100)
        logits[:, next(sequence)] = 100
        return logits
    monkeypatch.setattr(llm.get_output_embeddings(), "forward", next_logits)
    result = trainer.generate_answer(llm, sidecar, tokenizer, row, z[None], torch.device("cpu"),
                                      torch.float32, max_prompt_tokens=2048, max_new_tokens=12)
    assert result == {"prediction": "[0.3]", "terminated": True, "stop_token_id": alternate, "generated_tokens": 6}
    assert sidecar._bound_state is None


def test_backward_reaches_field_and_bridges_but_not_qwen(z):
    trainer, llm, sidecar, tokenizer, config = tiny_components()
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer.core.set_frozen_llm_execution_mode(llm, checkpoint_training=True)
    sidecar.train()
    row = make_record("state", z, "single_point", {"points": [[1, 1]]})
    try:
        loss, _ = trainer.training_loss(llm, sidecar, tokenizer, [row], z[None, None],
                                        torch.device("cpu"), torch.float32, config["training"])
        loss.backward()
        assert sidecar.bridges[0].gate.grad.abs() > 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in sidecar.memory.field_encoder.parameters())
        side_ids = {id(p) for p in sidecar.parameters()}
        assert all(p.grad is None and not p.requires_grad for p in llm.parameters() if id(p) not in side_ids)
    finally:
        sidecar.clear()


def test_cached_generation_matches_uncached_and_clears_memory(z):
    trainer, llm, sidecar, tokenizer, _ = tiny_components()
    llm.eval()
    sidecar.eval()
    row = make_record("state", z, "single_point", {"points": [[1, 1]]})
    kwargs = dict(max_prompt_tokens=2048, max_new_tokens=5)
    cached = trainer.generate_answer(llm, sidecar, tokenizer, row, z[None], torch.device("cpu"),
                                      torch.float32, use_cache=True, **kwargs)
    uncached = trainer.generate_answer(llm, sidecar, tokenizer, row, z[None], torch.device("cpu"),
                                        torch.float32, use_cache=False, **kwargs)
    assert cached == uncached
    assert sidecar._bound_state is None
    assert all(bridge._memory is None for bridge in sidecar.bridges)
    # A second, different-shaped field must also bind cleanly after cached decoding.
    other = normalize_field(torch.arange(7 * 9).reshape(7, 9).float())[0]
    other_row = make_record("other", other, "single_point", {"points": [[7, 9]]})
    trainer.generate_answer(llm, sidecar, tokenizer, other_row, other[None], torch.device("cpu"),
                            torch.float32, **kwargs)
    assert sidecar._bound_state is None


def test_generation_exception_clears_bound_memory(z, monkeypatch):
    trainer, llm, sidecar, tokenizer, _ = tiny_components()
    row = make_record("state", z, "single_point", {"points": [[1, 1]]})
    def fail(*args, **kwargs):
        raise RuntimeError("injected decoder failure")
    monkeypatch.setattr(llm.model, "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        trainer.generate_answer(llm, sidecar, tokenizer, row, z[None], torch.device("cpu"),
                                torch.float32, max_prompt_tokens=2048)
    assert sidecar._bound_state is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_bfloat16_cuda_training_and_generation(z):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")
    trainer, llm, sidecar, tokenizer, config = tiny_components()
    device = torch.device("cuda")
    llm.to(device=device, dtype=torch.bfloat16)
    sidecar.to(device=device, dtype=torch.float32)
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer.core.set_frozen_llm_execution_mode(llm, checkpoint_training=True)
    sidecar.train()
    row = make_record("state", z, "multi_point", {"points": [[1, 1], [10, 12]]})
    try:
        loss, _ = trainer.training_loss(llm, sidecar, tokenizer, [row], z[None, None],
                                        device, torch.bfloat16, config["training"])
        loss.backward()
        assert torch.isfinite(sidecar.bridges[0].gate.grad)
    finally:
        sidecar.clear()
    trainer.core.set_frozen_llm_execution_mode(llm, checkpoint_training=False)
    sidecar.eval()
    result = trainer.generate_answer(llm, sidecar, tokenizer, row, z[None], device, torch.bfloat16,
                                      max_prompt_tokens=2048, max_new_tokens=3)
    assert result["generated_tokens"] <= 3
    assert sidecar._bound_state is None


def test_batching_has_no_duplication_and_is_deterministic(data_fixture):
    from scripts.train_point_readout import shape_batches
    source, qa, _, _ = data_fixture
    dataset = PointReadoutDataset(qa, source, "train")
    try:
        batches = shape_batches(dataset.records, 3, 42, 0)
        assert sorted(i for batch in batches for i in batch) == list(range(len(dataset)))
        assert all(len({tuple(dataset.records[i]["grid_shape"]) for i in b}) == 1 for b in batches)
        assert batches == shape_batches(dataset.records, 3, 42, 0)
        assert batches != shape_batches(dataset.records, 3, 42, 1)
    finally:
        dataset.close()


def test_automatic_training_budget_respects_selected_tasks(data_fixture):
    from scripts.train_point_readout import planned_updates
    source, qa, config, _ = data_fixture
    dataset = PointReadoutDataset(qa, source, "train")
    try:
        training = {**config["training"], "max_updates": None, "epochs": 2}
        assert planned_updates(dataset.records, training, 42) == 44
        subset = [r for r in dataset.records if r["task_type"] == "single_point"]
        assert planned_updates(subset, training, 42) == 4
        assert planned_updates(subset, {**training, "max_updates": 7}, 42) == 7
    finally:
        dataset.close()


def test_resume_rejects_wrong_protocol_or_changed_contract():
    from scripts.train_point_readout import CHECKPOINT_TYPE, validate_resume
    with pytest.raises(ValueError, match="multiple-choice"):
        validate_resume({"checkpoint_type": "old"}, {})
    with pytest.raises(ValueError, match="differs"):
        validate_resume({"checkpoint_type": CHECKPOINT_TYPE, "identity": {"dataset": "old"}}, {"dataset": "new"})


def test_resume_binds_native_template_and_generation_stops(data_fixture):
    source, qa, initial, _ = data_fixture
    trainer, llm, _, tokenizer, config = tiny_components(initial)
    dataset = PointReadoutDataset(qa, source, "train")
    try:
        identity = trainer.run_identity(config, dataset, llm, tokenizer)
        old = {k: v for k, v in identity.items() if k != "text_encoding"}
        with pytest.raises(ValueError, match="chat template/stop protocol"):
            trainer.validate_resume({"checkpoint_type": trainer.CHECKPOINT_TYPE, "identity": old}, identity)
        llm.generation_config.eos_token_id = [tokenizer.eos_token_id, 5]
        changed = trainer.run_identity(config, dataset, llm, tokenizer)
        assert changed["text_encoding"]["stop_token_ids"] == [2, 5]
        with pytest.raises(ValueError, match="chat template/stop protocol"):
            trainer.validate_resume({"checkpoint_type": trainer.CHECKPOINT_TYPE, "identity": identity}, changed)
        tokenizer.chat_template = "another-native-template"
        assert trainer.run_identity(config, dataset, llm, tokenizer)["text_encoding"] != changed["text_encoding"]
    finally:
        dataset.close()


def test_local_model_identity_tracks_weight_contents_and_allows_relocation(tmp_path):
    from scripts.train_point_readout import model_asset_identity
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for path in (first, second):
        (path / "model.safetensors").write_bytes(b"weight fixture")
    expected = model_asset_identity(str(first), None)
    assert model_asset_identity(str(second), None) == expected
    (second / "model.safetensors").write_bytes(b"different weights")
    assert model_asset_identity(str(second), None) != expected


def test_training_resume_and_independent_test_evaluation(data_fixture, tmp_path, monkeypatch):
    """Exercise the actual CLI loop with a tiny real Qwen, including interrupted resume."""
    import yaml
    from transformers import Qwen2ForCausalLM
    from scripts import train_point_readout as trainer
    source, qa, initial, _ = data_fixture
    _, example_llm, _, tokenizer, config = tiny_components(initial)
    llm_config = copy.deepcopy(example_llm.config)
    config["model"]["gradient_checkpointing"] = True
    config["training"].update(max_updates=2, log_interval=1, save_every_updates=1,
                              eval_every_updates=1, gradient_accumulation_steps=2)
    config["experiment_profile"] = "test"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"profiles": {"smoke": config}}), encoding="utf-8")

    def load_model(args, device):
        torch.manual_seed(123)
        llm = Qwen2ForCausalLM(copy.deepcopy(llm_config)).to(device)
        llm.config.use_cache = False
        if args.llm_gradient_checkpointing:
            llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return llm, torch.float32

    monkeypatch.setattr(trainer.core, "load_tokenizer", lambda args: tokenizer)
    monkeypatch.setattr(trainer.core, "load_llm_with_bounded_host_memory", load_model)
    base = ["train_point_readout.py", "--config", str(config_path), "--profile", "smoke",
            "--qa-dir", str(qa), "--hdf5-path", str(source), "--device", "cpu"]

    def run(out, *extra):
        monkeypatch.setattr(sys, "argv", base + ["--output-dir", str(out), *extra])
        trainer.main()

    uninterrupted = tmp_path / "complete"
    run(uninterrupted)
    assert (uninterrupted / "best.pt").exists()
    assert not list(uninterrupted.glob("test*"))
    expected = torch.load(uninterrupted / "last.pt", weights_only=True)
    interrupted = tmp_path / "interrupted"
    run(interrupted, "--stop-after-updates", "1")
    summary = json.loads((interrupted / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "stopped" and summary["step"] == 1 and summary["planned_updates"] == 2
    assert torch.load(interrupted / "last.pt", weights_only=True)["validation_pending"]
    run(interrupted, "--resume", str(interrupted / "last.pt"))
    resumed = torch.load(interrupted / "last.pt", weights_only=True)
    assert json.loads((interrupted / "run_summary.json").read_text(encoding="utf-8"))["status"] == "completed"
    assert resumed["step"] == expected["step"] == 2
    assert not resumed["validation_pending"]
    assert [json.loads(line)["step"] for line in (interrupted / "validation.jsonl").read_text().splitlines()] == [1, 2]
    assert (resumed["epoch"], resumed["cursor"]) == (expected["epoch"], expected["cursor"])
    for name in expected["sidecar"]:
        torch.testing.assert_close(resumed["sidecar"][name], expected["sidecar"][name], rtol=0, atol=0)
    # Recreate a final-step checkpoint saved before validation; resume must finish
    # validation even though no optimizer updates remain, and produce best.pt.
    pending = tmp_path / "pending_final.pt"
    torch.save({**expected, "validation_pending": True, "best": -1.0}, pending)
    recovered = tmp_path / "recovered_final"
    run(recovered, "--resume", str(pending))
    assert (recovered / "best.pt").exists()
    assert torch.load(recovered / "last.pt", weights_only=True)["step"] == 2
    evaluation = tmp_path / "test_eval"
    run(evaluation, "--resume", str(interrupted / "best.pt"), "--evaluate-only", "--split", "test")
    online = json.loads((evaluation / "test_metrics.json").read_text(encoding="utf-8"))
    dataset = PointReadoutDataset(qa, source, "test")
    try:
        predictions = [json.loads(line) for line in (evaluation / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        offline = score_predictions(dataset, predictions)
        assert online["point_accuracy"] == offline["point_accuracy"]
        assert online["questions"] == len(dataset)
    finally:
        dataset.close()


def test_server_field_inspection_and_download_checksum(data_fixture, tmp_path):
    import hashlib
    from scripts.check_field_qa_environment import inspect_fields
    from scripts.download_pdebench_field import verify
    source, _, _, _ = data_fixture
    result = inspect_fields(source, ["Vx", "Vy"])
    assert result["Vx"] == [20, 3, 24, 28]
    with pytest.raises(ValueError):
        inspect_fields(source, ["missing"])
    artifact = tmp_path / "download.part"
    artifact.write_bytes(b"complete fixture")
    verify(artifact, hashlib.md5(b"complete fixture").hexdigest())
    with pytest.raises(ValueError, match="MD5 mismatch"):
        verify(artifact, hashlib.md5(b"incomplete fixture").hexdigest())


def test_download_restarts_partial_transfer_without_discarding_bytes(tmp_path, monkeypatch):
    import subprocess
    from scripts import download_pdebench_field as downloader
    partial = tmp_path / "source.hdf5.part"
    partial.write_bytes(b"existing")
    calls, sleeps = [], []

    def transfer(command, check):
        assert check and command[command.index("--continue-at") + 1] == "-"
        assert command[command.index("--output") + 1] == str(partial)
        calls.append(partial.read_bytes())
        with partial.open("ab") as handle:
            handle.write(b"more")
        if len(calls) == 1:
            raise subprocess.CalledProcessError(18, command)

    monkeypatch.setattr(downloader.subprocess, "run", transfer)
    monkeypatch.setattr(downloader.time, "sleep", sleeps.append)
    downloader.download_partial(partial, resume_retries=1)
    assert calls == [b"existing", b"existingmore"]
    assert partial.read_bytes() == b"existingmoremore"
    assert sleeps == [10]


@pytest.mark.parametrize("exit_code,attempts", [(18, 3), (22, 1), (23, 1)])
def test_download_retries_are_bounded_and_preserve_failed_file(tmp_path, monkeypatch, exit_code, attempts):
    import subprocess
    from scripts import download_pdebench_field as downloader
    partial = tmp_path / "source.hdf5.part"
    partial.write_bytes(b"keep")
    calls = []

    def fail(command, check):
        calls.append(command)
        raise subprocess.CalledProcessError(exit_code, command)

    monkeypatch.setattr(downloader.subprocess, "run", fail)
    monkeypatch.setattr(downloader.time, "sleep", lambda seconds: None)
    with pytest.raises(subprocess.CalledProcessError):
        downloader.download_partial(partial, resume_retries=2)
    assert len(calls) == attempts
    assert partial.read_bytes() == b"keep"
