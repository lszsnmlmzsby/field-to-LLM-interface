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


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("block", [1, 2])
def test_batched_greedy_matches_individual_variable_length_prompts(use_cache, block):
    _, _, _, _, config = tiny_components()
    config["memory"]["block_shape"] = [block, block]
    _, llm, sidecar, tokenizer, _ = tiny_components(config)
    trainer.core.set_frozen_llm_execution_mode(llm, checkpoint_training=False)
    sidecar.eval()
    prompts = [[50, 61, 72], [82], [50, 61, 72, 83, 94]]
    with torch.inference_mode():
        sidecar.bind(TinyDataset().fields["a"][None])
        try:
            expected = [trainer.generate_from_prompt(llm, tokenizer, p, torch.device("cpu"),
                        torch.float32, max_new_tokens=8, use_cache=use_cache) for p in prompts]
            actual = trainer.generate_from_prompts(llm, tokenizer, prompts, torch.device("cpu"),
                        torch.float32, max_new_tokens=8, use_cache=use_cache)
            assert actual == expected
            # Projections retain batch ONE; shared field memory is not duplicated.
            assert all(bridge._projected_memory[0].shape[0] == 1 for bridge in sidecar.bridges)
        finally:
            sidecar.clear()


def test_batched_decoder_independent_eos_padding_positions_and_token_budget(monkeypatch):
    from types import SimpleNamespace
    calls = []
    class ScriptedModel:
        config = SimpleNamespace(eos_token_id=[2, 3])

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(last_hidden_state=torch.zeros(3, 1, 4), past_key_values="cache")

        def get_output_embeddings(self):
            def project(hidden):
                logits = torch.full((3, 256), -100.)
                tokens = ([2, 81, 82], [99, 3, 83], [99, 99, 84])[len(calls) - 1]
                logits[torch.arange(3), torch.tensor(tokens)] = 100
                return logits
            return project
    model, tokenizer = ScriptedModel(), CharacterTokenizer()
    monkeypatch.setattr(trainer.core, "decoder_backbone", lambda model: model)
    predictions = trainer.generate_from_prompts(model, tokenizer, [[51], [61, 62, 63], [71, 72]],
        torch.device("cpu"), torch.float32, max_new_tokens=3)
    assert [p["generated_tokens"] for p in predictions] == [1, 2, 3]
    assert [p["stop_token_id"] for p in predictions] == [2, 3, None]
    assert [p["terminated"] for p in predictions] == [True, True, False]
    assert predictions[0]["prediction"] == ""
    assert predictions[1]["prediction"] == tokenizer.decode([81])
    assert predictions[2]["prediction"] == tokenizer.decode([82, 83, 84])
    assert calls[0]["attention_mask"].tolist() == [[0, 0, 1], [1, 1, 1], [0, 1, 1]]
    assert calls[0]["position_ids"].tolist() == [[0, 0, 0], [0, 1, 2], [0, 0, 1]]
    assert calls[1]["input_ids"].shape == (3, 1)
    assert calls[1]["position_ids"][1:].tolist() == [[3], [2]]
    assert calls[1]["attention_mask"][:, -1].tolist() == [0, 1, 1]
    assert calls[2]["attention_mask"][:, -1].tolist() == [0, 0, 1]


@pytest.mark.parametrize("input_mode", ["correct", "disabled"])
def test_batched_evaluation_preserves_all_rows_order_metrics_and_field_boundaries(tmp_path, input_mode):
    _, llm, sidecar, tokenizer, config = tiny_components()
    config["evaluation"]["max_new_tokens"] = 8
    dataset = TinyDataset()
    serial = tmp_path / "serial.jsonl"
    expected = trainer.evaluate(llm, sidecar, tokenizer, dataset, torch.device("cpu"), torch.float32,
                                config, serial, input_mode=input_mode)
    config["runtime"].update(eval_batch_size=8, eval_parity_check_questions=32)
    batched = tmp_path / "batched.jsonl"
    actual = trainer.evaluate(llm, sidecar, tokenizer, dataset, torch.device("cpu"), torch.float32,
                              config, batched, input_mode=input_mode)
    assert serial.read_bytes() == batched.read_bytes()
    for key in ("questions", "macro_task_answer_accuracy", "valid_rate", "errors", "seen_shapes",
                "heldout_shapes", "by_family", "generated_tokens", "zero_value_baseline"):
        assert actual[key] == expected[key]
    assert actual["generation_batches"] == 3 < expected["generation_batches"]
    assert actual["batch_parity_checked"] == 2 and actual["batch_parity_mismatches"] == 0
    assert actual["field_memory_encodes"] == (3 if input_mode == "correct" else 0)
    assert sidecar._bound_state is None


@pytest.mark.parametrize("failure", ["oom", "parity"])
def test_batched_evaluation_fallback_never_drops_questions(tmp_path, monkeypatch, failure):
    _, llm, sidecar, tokenizer, config = tiny_components()
    config["evaluation"]["max_new_tokens"] = 4
    dataset = TinyDataset()
    reference = tmp_path / "reference.jsonl"
    trainer.evaluate(llm, sidecar, tokenizer, dataset, torch.device("cpu"), torch.float32, config, reference)
    config["runtime"].update(eval_batch_size=8, eval_parity_check_questions=32)
    real_generate = trainer.generate_from_prompts
    injected = []
    def generate(*args, **kwargs):
        if len(args[2]) > 1 and not injected:
            injected.append(True)
            if failure == "oom":
                raise torch.cuda.OutOfMemoryError("injected, not actual GPU OOM")
            result = real_generate(*args, **kwargs)
            result[0]["prediction"] += "different"
            return result
        return real_generate(*args, **kwargs)
    monkeypatch.setattr(trainer, "generate_from_prompts", generate)
    output = tmp_path / "fallback.jsonl"
    metrics = trainer.evaluate(llm, sidecar, tokenizer, dataset, torch.device("cpu"), torch.float32, config, output)
    assert output.read_bytes() == reference.read_bytes()
    assert metrics["questions"] == len(dataset) and metrics["evaluation_batch_limit"] == 1
    assert metrics["evaluation_oom_retries"] == int(failure == "oom")
    assert metrics["batch_parity_mismatches"] == int(failure == "parity")
    assert sidecar._bound_state is None


def test_adaptive_micro_batch_cost_limits_persist_without_penalizing_small_fields():
    _, _, _, tokenizer, config = tiny_components()
    dataset = TinyDataset()
    dataset.records[0] = {**dataset.records[0], "grid_shape": [128, 128]}
    controller = trainer.AdaptiveMicroBatch(16)
    big, limit = controller.select(dataset, [[0]], tokenizer, config["training"], None)
    assert limit == 16
    controller.record_oom(big, 4)
    assert controller.select(dataset, [[0]], tokenizer, config["training"], None)[1] == 4
    assert controller.select(dataset, [[1]], tokenizer, config["training"], None)[1] == 16
    restored = trainer.AdaptiveMicroBatch(8, controller.state_dict())
    assert restored.select(dataset, [[0]], tokenizer, config["training"], None)[1] == 4
    assert restored.select(dataset, [[1]], tokenizer, config["training"], None)[1] == 8


def test_adaptive_chunks_preserve_global_weighted_gradients():
    from scripts.readout_distributed import ParallelContext
    _, llm, sidecar, tokenizer, config = tiny_components()
    dataset, results = TinyDataset(), []
    for controller in (None, trainer.AdaptiveMicroBatch(2)):
        sidecar.train()
        sidecar.zero_grad(set_to_none=True)
        values, _, execution = trainer.backward_update(llm, sidecar, tokenizer, dataset, [[0, 1, 3], [2]],
            torch.device("cpu"), torch.float32, config["training"], ParallelContext("cpu"), 1,
            adaptive_micro_batch=controller)
        results.append((values, {n: p.grad.clone() for n, p in sidecar.named_parameters() if p.grad is not None}))
        assert execution[0]["questions"] == 4
    assert results[0][0] == pytest.approx(results[1][0], rel=1e-5, abs=1e-6)
    for name, gradient in results[0][1].items():
        torch.testing.assert_close(gradient, results[1][1][name], rtol=1e-4, atol=1e-6)


def test_adaptive_oom_learning_survives_restore_and_retries_whole_update(monkeypatch):
    from scripts.readout_distributed import ParallelContext, capture_rng, restore_rng
    from test_readout_distributed import ToySidecar
    _, _, _, tokenizer, config = tiny_components()
    dataset = TinyDataset()
    dataset.records = [copy.deepcopy(dataset.records[0]) for _ in range(7)]
    for row in dataset.records:
        row["grid_shape"] = [128, 128]  # Cost metadata; the loss stub uses no encoder.
    controller = trainer.AdaptiveMicroBatch(4)
    injected = []
    def loss_fn(llm, module, tokenizer, records, fields, *args, **kwargs):
        module.calls.add_(1)
        torch.rand(1)
        if len(records) == 3 and not injected:
            injected.append(True)
            module.weight.grad.add_(123)
            raise torch.cuda.OutOfMemoryError("injected after a successful chunk")
        loss = (module.weight * fields).square().mean()
        return loss, {"answer_ce": float(loss.detach()), "reconstruction": 0.}
    monkeypatch.setattr(trainer, "training_loss", loss_fn)
    device, context = torch.device("cpu"), ParallelContext("cpu")
    initial_rng, actual = capture_rng(device), ToySidecar()
    values, cap, reports = trainer.backward_update(None, actual, tokenizer, dataset, [list(range(7))],
        device, torch.float32, config["training"], context, 4, adaptive_micro_batch=controller)
    final_rng = capture_rng(device)
    assert cap == 2 and reports[0]["oom_retries"] == 1
    restored = trainer.AdaptiveMicroBatch(4, controller.state_dict())
    assert restored.select(dataset, [list(range(7))], tokenizer, config["training"], None)[1] == 2
    restore_rng(initial_rng, device)
    reference = ToySidecar()
    expected, _, _ = trainer.backward_update(None, reference, tokenizer, dataset, [list(range(7))],
        device, torch.float32, config["training"], context, 2)
    assert values == expected
    torch.testing.assert_close(actual.weight.grad, reference.weight.grad, rtol=0, atol=0)
    assert actual.calls == reference.calls == 4
    assert torch.equal(final_rng["torch_rng"], torch.get_rng_state())


def test_length_scheduling_and_token_budget_preserve_gradients_and_reduce_padding():
    from scripts.readout_distributed import ParallelContext
    from array import array
    _, llm, sidecar, tokenizer, config = tiny_components()
    dataset = TinyDataset()
    base = dataset.records[0]
    dataset.records = [copy.deepcopy(base) for _ in range(8)]
    for i, extra in enumerate([0, 50, 2, 40, 5, 30, 8, 20]):
        dataset.records[i]["question"] += " extra" * extra
    lengths = array("I")
    trainer.preflight_token_budgets(dataset.records, tokenizer, config["training"], token_lengths=lengths)
    assert list(lengths) == [len(p) + len(a) for p, a in
        (trainer.encode_training_record(row, tokenizer, config["training"]) for row in dataset.records)]
    results = []
    budget = max(lengths) * 2
    for balanced in (False, True):
        sidecar.train()
        values, _, execution = trainer.backward_update(llm, sidecar, tokenizer, dataset, [list(range(8))],
            torch.device("cpu"), torch.float32, config["training"], ParallelContext("cpu"), 4,
            token_lengths=lengths, balance_by_length=balanced,
            micro_batch_token_budget=budget if balanced else 0)
        results.append((values, {n: p.grad.clone() for n, p in sidecar.named_parameters()
                                if p.grad is not None}, execution[0]))
    assert results[0][0] == pytest.approx(results[1][0], rel=1e-5, abs=1e-6)
    for name, gradient in results[0][1].items():
        torch.testing.assert_close(gradient, results[1][1][name], rtol=2e-4, atol=2e-6)
    assert results[1][2]["true_text_tokens"] == results[0][2]["true_text_tokens"] == sum(lengths)
    assert results[1][2]["padded_text_tokens"] < results[0][2]["padded_text_tokens"]
    assert results[1][2]["questions"] == 8
    assert max(results[1][2]["micro_batch_sizes"]) <= 4


def test_one_long_question_is_never_truncated_by_execution_token_budget():
    from scripts.readout_distributed import ParallelContext
    _, llm, sidecar, tokenizer, config = tiny_components()
    dataset = TinyDataset()
    lengths = [sum(map(len, trainer.encode_training_record(row, tokenizer, config["training"])))
               for row in dataset.records]
    _, _, execution = trainer.backward_update(llm, sidecar, tokenizer, dataset, [[0, 1, 3]],
        torch.device("cpu"), torch.float32, config["training"], ParallelContext("cpu"), 8,
        token_lengths=lengths, balance_by_length=True, micro_batch_token_budget=1)
    assert execution[0]["micro_batch_sizes"] == [1, 1, 1]
    assert execution[0]["true_text_tokens"] == sum(lengths[i] for i in [0, 1, 3])


def test_adaptive_limits_apply_per_chunk_not_to_all_short_questions(monkeypatch):
    from scripts.readout_distributed import ParallelContext
    from test_readout_distributed import ToySidecar
    dataset = TinyDataset()
    dataset.records = [copy.deepcopy(dataset.records[0]) for _ in range(6)]
    lengths = [900, 800, 200, 190, 180, 170]
    controller = trainer.AdaptiveMicroBatch(4)
    controller.record_oom((4, 8), 2)
    def loss_fn(llm, module, tokenizer, records, fields, *args, **kwargs):
        loss = (module.weight * fields).square().mean()
        return loss, {"answer_ce": float(loss.detach()), "reconstruction": 0.}
    monkeypatch.setattr(trainer, "training_loss", loss_fn)
    _, _, execution = trainer.backward_update(None, ToySidecar(), None, dataset, [list(range(6))],
        torch.device("cpu"), torch.float32, {}, ParallelContext("cpu"), 4,
        token_lengths=lengths, balance_by_length=True, adaptive_micro_batch=controller)
    assert execution[0]["micro_batch_sizes"] == [2, 4]
    assert execution[0]["questions"] == 6


def test_fixed_checkpoint_benchmark_restores_state_and_compares_identical_questions():
    from scripts.readout_distributed import ParallelContext, capture_rng
    from array import array
    _, llm, sidecar, tokenizer, config = tiny_components()
    config["training"].update(batch_size=3, gradient_accumulation_steps=1)
    dataset, lengths = TinyDataset(), array("I")
    trainer.preflight_token_budgets(dataset.records, tokenizer, config["training"], token_lengths=lengths)
    initial_state = {name: value.clone() for name, value in sidecar.state_dict().items()}
    rng = capture_rng(torch.device("cpu"))
    checkpoint = {"epoch": 0, "cursor": 0, "rank_states": [rng]}
    result = trainer.benchmark_execution(llm, sidecar, tokenizer, dataset, torch.device("cpu"),
        torch.float32, config, ParallelContext("cpu"), trainer.TrainingTokenCache(tokenizer, config["training"]),
        lengths, checkpoint, updates=2, warmup=1)
    assert result["optimizer_updates"] == 0 and result["checkpoint_written"] is False
    assert len(result["results"]) == 3
    for policy in result["results"]:
        assert policy["questions"] == 4 and policy["updates"] == 2
        assert policy["true_text_tokens"] == sum(lengths)
        assert policy["max_loss_abs_diff_vs_original"] < 1e-5
        assert policy["first_update_gradient_vs_original"]["relative_l2"] < 1e-4
    for name, value in sidecar.state_dict().items():
        torch.testing.assert_close(value, initial_state[name], rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng["torch_rng"])


def test_execution_settings_do_not_change_strict_checkpoint_identity():
    from types import SimpleNamespace
    _, llm, _, tokenizer, config = tiny_components()
    dataset = SimpleNamespace(identity="unchanged-complete-dataset")
    original = trainer.run_identity(config, dataset, llm, tokenizer)
    config["runtime"].update(length_balanced_batches=True, micro_batch_token_budget=4096,
                            max_micro_batch_size=8, adaptive_micro_batch=True)
    trainer.validate_resume({"checkpoint_type": trainer.CHECKPOINT_TYPE, "identity": original},
                            trainer.run_identity(config, dataset, llm, tokenizer))
    config["training"]["batch_size"] += 1
    with pytest.raises(ValueError, match="differs"):
        trainer.validate_resume({"checkpoint_type": trainer.CHECKPOINT_TYPE, "identity": original},
                                trainer.run_identity(config, dataset, llm, tokenizer))


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
    # Read-only execution benchmark must never advance or overwrite the source run.
    checkpoint_bytes = (interrupted / "last.pt").read_bytes()
    config_bytes = (interrupted / "resolved_config.json").read_bytes()
    benchmark_dir = tmp_path / "execution_benchmark"
    run(benchmark_dir, "--resume", str(interrupted / "last.pt"),
        "--benchmark-updates", "1", "--benchmark-warmup-updates", "0")
    assert (interrupted / "last.pt").read_bytes() == checkpoint_bytes
    assert (interrupted / "resolved_config.json").read_bytes() == config_bytes
    assert not (benchmark_dir / "last.pt").exists()
    report = json.loads((benchmark_dir / "execution_benchmark.json").read_text())
    assert report["optimizer_updates"] == 0 and report["checkpoint_step"] == 1
    with pytest.raises(SystemExit):
        run(interrupted, "--resume", str(interrupted / "last.pt"), "--benchmark-updates", "1")
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
