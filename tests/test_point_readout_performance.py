"""Exact cache parity and lifecycle checks using local tiny models only."""
from __future__ import annotations

import copy
import json
import sys

import pytest
import torch

from test_multisource_readout import config as multisource_fixture
from test_point_readout import CharacterTokenizer, tiny_components
from tensor_compression.downstream.point_readout import make_record, normalize_field
from scripts import train_point_readout as trainer
from scripts.train_tensor_qwen_cross_attention import DenseMemoryState, GatedTensorCrossAttention


def sample_rows():
    a = normalize_field(torch.arange(12).reshape(3, 4).float())[0]
    b = normalize_field(torch.arange(10).reshape(2, 5).float().square())[0]
    records = [make_record(state, field, "single_point", {"points": [point]})
               for state, field, point in (("a", a, [1, 1]), ("a", a, [1, 2]),
                                           ("b", b, [2, 5]), ("a", a, [2, 1]))]
    return records, {"a": a[None], "b": b[None]}


def test_training_token_cache_exact_tensors_bounded_and_content_sensitive():
    tokenizer = CharacterTokenizer()
    training = {"max_prompt_tokens": 2048, "max_target_tokens": 96}
    cache = trainer.TrainingTokenCache(tokenizer, training, max_bytes=100000, max_entries=2)
    rows, _ = sample_rows()
    expected = trainer.training_tensors(rows[:2], tokenizer, training)
    for _ in range(2):
        actual = trainer.training_tensors(rows[:2], tokenizer, training, token_cache=cache)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert cache.hits == 2 and cache.misses == 2
    changed = copy.deepcopy(rows[0])
    changed["answer"] = "[0.0]"  # Same QA ID must not reuse different target tokens.
    actual = trainer.training_tensors([changed], tokenizer, training, token_cache=cache)
    expected = trainer.training_tensors([changed], tokenizer, training)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert cache.misses == 3 and len(cache.entries) == 2
    assert cache.bytes <= cache.max_bytes
    zero = trainer.TrainingTokenCache(tokenizer, training, max_bytes=0)
    zero.get(rows[0], tokenizer, training)
    assert not zero.entries and zero.bytes == 0
    with pytest.raises(ValueError, match="different tokenizer or token budget"):
        cache.get(rows[0], tokenizer, {**training, "max_target_tokens": 1})
    tokenizer.chat_template = "changed"
    with pytest.raises(ValueError, match="different tokenizer or token budget"):
        cache.get(rows[0], tokenizer, training)


def test_training_token_cache_preserves_loss_and_gradients():
    _, llm, sidecar, tokenizer, config = tiny_components()
    rows, fields = sample_rows()
    rows, field = rows[:2], torch.stack([fields["a"]] * 2)
    cache = trainer.TrainingTokenCache(tokenizer, config["training"])
    results = []
    for active_cache in (None, cache, cache):
        sidecar.zero_grad(set_to_none=True)
        sidecar.train()
        torch.manual_seed(10)
        try:
            loss, _ = trainer.training_loss(llm, sidecar, tokenizer, rows, field,
                torch.device("cpu"), torch.float32, config["training"], token_cache=active_cache)
            loss.backward()
            results.append((loss.detach(), {name: parameter.grad.clone() for name, parameter in
                            sidecar.named_parameters() if parameter.grad is not None}))
            assert all(bridge._projected_memory is None for bridge in sidecar.bridges)
        finally:
            sidecar.clear()
    for loss, gradients in results[1:]:
        assert torch.equal(loss, results[0][0])
        assert gradients.keys() == results[0][1].keys()
        assert all(torch.equal(value, results[0][1][name]) for name, value in gradients.items())


def test_projection_cache_exact_and_invalidated_after_parameter_update_or_rebind():
    torch.manual_seed(14)
    bridge = GatedTensorCrossAttention(8, 6, 8, 2, 0., .2).eval()
    memory = DenseMemoryState(torch.randn(1, 5, 6), torch.randn(1, 5, 6), torch.tensor(0.))
    hidden = torch.randn(1, 3, 8)
    calls = []
    hook = bridge.k_proj.register_forward_hook(lambda *args: calls.append(True))
    try:
        bridge.bind_memory(memory)
        with torch.no_grad():
            first = bridge(hidden)
            assert torch.equal(first, bridge(hidden))
            assert len(calls) == 1
            bridge.k_proj.weight.add_(.1)
            updated = bridge(hidden)
            assert len(calls) == 2
            bridge.bind_memory(memory)
            assert torch.equal(updated, bridge(hidden))
            assert len(calls) == 3
        # eval() alone must not cache a tensor participating in autograd.
        bridge(hidden).sum().backward()
        assert bridge._projected_memory is None
        assert bridge.k_proj.weight.grad is not None
        with torch.no_grad():
            bridge(hidden)
        bridge.train()
        assert bridge._projected_memory is None
        bridge.clear_memory()
        assert bridge._memory is None and bridge._projection_signature is None
    finally:
        hook.remove()


class TinyDataset:
    def __init__(self):
        self.records, self.fields = sample_rows()
        self.metadata = {"train_shapes": ["3x4"]}
        self.states = {name: {"field": "velocity", "source_id": name, "family": "fixture"}
                       for name in self.fields}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return row, self.fields[row["state_ref"]]


@pytest.mark.parametrize("use_text_cache", [False, True])
def test_evaluation_field_and_projection_reuse_preserves_predictions_order_and_scores(tmp_path, use_text_cache):
    _, llm, sidecar, tokenizer, config = tiny_components()
    config["evaluation"].update(max_new_tokens=4, use_cache=use_text_cache)
    dataset, calls = TinyDataset(), []
    hook = sidecar.memory.register_forward_hook(lambda *args: calls.append(True))
    fast_path, reference_path = tmp_path / "fast.jsonl", tmp_path / "reference.jsonl"
    try:
        fast = trainer.evaluate(llm, sidecar, tokenizer, dataset, torch.device("cpu"), torch.float32,
                                config, fast_path)
        assert len(calls) == 3  # a,a,b,a: preserve order, never retain two field memories.
        assert fast["field_memory_encodes"] == 3 and fast["field_memory_reuses"] == 1
        assert sidecar._bound_state is None
        assert all(bridge._projected_memory is None for bridge in sidecar.bridges)
        calls.clear()
        # Recreate the prior path: recompute each field and every memory K/V.
        config["evaluation"]["reuse_field_memory"] = False
        def clear_projections(module, inputs):
            module._projection_signature = module._projected_memory = None
        projection_hooks = [bridge.register_forward_pre_hook(clear_projections) for bridge in sidecar.bridges]
        try:
            reference = trainer.evaluate(llm, sidecar, tokenizer, dataset, torch.device("cpu"), torch.float32,
                                         config, reference_path)
        finally:
            for projection_hook in projection_hooks:
                projection_hook.remove()
        assert len(calls) == len(dataset)
        assert fast_path.read_bytes() == reference_path.read_bytes()
        predictions = [json.loads(line) for line in fast_path.read_text().splitlines()]
        assert [row["qa_id"] for row in predictions] == [row["qa_id"] for row in dataset.records]
        for key in ("elapsed_seconds", "peak_allocated_gib", "field_memory_encodes", "field_memory_reuses"):
            fast.pop(key)
            reference.pop(key)
        assert fast == reference
    finally:
        hook.remove()


def test_evaluation_failure_clears_memory_and_does_not_skip_question(tmp_path, monkeypatch):
    _, llm, sidecar, tokenizer, config = tiny_components()
    calls = []
    def fail(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("injected decoding failure")
    monkeypatch.setattr(trainer, "generate_from_prompt", fail)
    with pytest.raises(RuntimeError, match="injected decoding failure"):
        trainer.evaluate(llm, sidecar, tokenizer, TinyDataset(), torch.device("cpu"), torch.float32,
                         config, tmp_path / "failed.jsonl")
    assert len(calls) == 1
    assert sidecar._bound_state is None
    assert all(bridge._projected_memory is None for bridge in sidecar.bridges)


def test_multisource_cli_training_resume_test_and_independent_provenance_scores(
        multisource_fixture, tmp_path, monkeypatch):
    """Real HDF5 factory/audit, tiny Qwen training, strict resume and held-out test.

    Only pretrained-model/tokenizer loading is replaced. All source receipt
    validation, crop replay, tokenization, model execution and scoring are real.
    The INS receipt is the explicitly tiny fixture defined by the data tests.
    """
    from transformers import Qwen2ForCausalLM
    from scripts.score_point_readout import score_predictions
    from tensor_compression.downstream.point_readout_data import PointReadoutDataset, build_dataset
    from tensor_compression.downstream.multisource_readout_data import MultiSourcePointReadoutDataset

    _, example_llm, _, tokenizer, config = tiny_components()
    config["data"].update(copy.deepcopy(multisource_fixture["data"]))
    # Keep every source and split. One field per source makes this a small
    # integration fixture; static acoustic density exercises the new reporting.
    for source in config["data"]["sources"]:
        source["fields"] = (["t0_fields/density"] if source["family"] == "acoustic"
                            else source["fields"][:1])
    config["generation"].update(copy.deepcopy(multisource_fixture["generation"]))
    config["generation"].update(tasks=["single_point"], train_states=32, eval_states=64)
    qa = tmp_path / "multisource_qa"
    metadata = build_dataset(config, None, qa)
    assert metadata["state_counts"] == {"train": 32, "val": 64, "test": 64}
    config["data"]["qa_dir"] = str(qa)
    config["model"]["gradient_checkpointing"] = True
    # Eight channels/eight GN groups in the generic tiny fixture cannot process
    # a batch-one 1x1 anchor. Match production's four channels per GN group.
    config["field_encoder"]["model"]["base_channels"] = 32
    config["memory"]["block_shape"] = [2, 2]
    config["training"].update(max_updates=2, batch_size=3, gradient_accumulation_steps=2,
                              log_interval=1, save_every_updates=1, eval_every_updates=2,
                              max_target_tokens=16)
    config["evaluation"]["max_new_tokens"] = 16
    config["experiment_profile"] = "smoke"
    config_path = tmp_path / "training_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    llm_config = copy.deepcopy(example_llm.config)

    def load_model(args, device):
        torch.manual_seed(123)
        model = Qwen2ForCausalLM(copy.deepcopy(llm_config)).to(device)
        model.config.use_cache = False
        if args.llm_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return model, torch.float32

    monkeypatch.setattr(trainer.core, "load_tokenizer", lambda args: tokenizer)
    monkeypatch.setattr(trainer.core, "load_llm_with_bounded_host_memory", load_model)
    base = ["train_point_readout.py", "--config", str(config_path), "--profile", "smoke", "--device", "cpu"]

    def run(path, *options):
        monkeypatch.setattr(sys, "argv", base + ["--output-dir", str(path), *options])
        trainer.main()

    complete = tmp_path / "complete"
    run(complete)
    expected = torch.load(complete / "last.pt", weights_only=True)
    interrupted = tmp_path / "interrupted"
    run(interrupted, "--stop-after-updates", "1")
    assert torch.load(interrupted / "last.pt", weights_only=True)["step"] == 1
    run(interrupted, "--resume", str(interrupted / "last.pt"))
    resumed = torch.load(interrupted / "last.pt", weights_only=True)
    assert resumed["step"] == expected["step"] == 2
    assert resumed["identity"] == expected["identity"]
    assert (resumed["epoch"], resumed["cursor"]) == (expected["epoch"], expected["cursor"])
    for name, parameter in expected["sidecar"].items():
        torch.testing.assert_close(parameter, resumed["sidecar"][name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="another experiment|differs"):
        run(interrupted, "--resume", str(interrupted / "last.pt"), "--batch-size", "2")

    output = tmp_path / "test_evaluation"
    run(output, "--resume", str(interrupted / "best.pt"), "--evaluate-only", "--split", "test")
    online = json.loads((output / "test_metrics.json").read_text(encoding="utf-8"))
    predictions = [json.loads(line) for line in (output / "test_predictions.jsonl").read_text().splitlines()]
    dataset = PointReadoutDataset(qa, None, "test")
    try:
        assert isinstance(dataset, MultiSourcePointReadoutDataset)
        assert dataset.audit()["questions"] == len(predictions) == 64
        assert [p["qa_id"] for p in predictions] == [r["qa_id"] for r in dataset.records]
        offline = score_predictions(dataset, predictions)
        for key in ("by_source_id", "by_family", "by_is_static", "by_is_constant"):
            assert online[key] == offline[key]
            assert sum(group["questions"] for group in online[key].values()) == len(dataset)
        assert set(online["by_source_id"]) == {source["id"] for source in config["data"]["sources"]}
        assert set(online["by_is_static"]) == {"True", "False"}
        assert set(online["by_is_constant"]) == {"True", "False"}
        assert online["macro_task_answer_accuracy"] == offline["macro_task_answer_accuracy"]
        assert online["seen_shapes"] == offline["seen_shapes"]
        assert online["heldout_shapes"] == offline["heldout_shapes"]
    finally:
        dataset.close()
