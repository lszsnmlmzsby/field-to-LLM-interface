"""Prepared-input integration using tiny local models; no remote speed claims."""
from __future__ import annotations

import copy
import json
import sys

import pytest
import torch

from test_multisource_readout import config
from test_readout_training_cache import cache_inputs, prepare
from test_point_readout import tiny_components
from tensor_compression.workflows import train_point_readout as trainer
from tensor_compression.workflows.readout_distributed import ParallelContext
from tensor_compression.workflows.readout_input_runtime import PreparedTokenLookup, local_update_indices, queue_following_updates
from tensor_compression.workflows.readout_prefetch import PreparedBatchLoader
from tensor_compression.downstream.multisource_readout_data import MultiSourcePointReadoutDataset
from tensor_compression.downstream.readout_training_cache import PreparedTrainingDataset


def test_prepared_batches_tokens_loss_gradients_and_oom_retry(cache_inputs, monkeypatch):
    qa, cache_dir, tokenizer, training, _ = cache_inputs
    prepare(cache_inputs)
    legacy = MultiSourcePointReadoutDataset(qa, "train")
    prepared = PreparedTrainingDataset(cache_dir)
    _, llm, reference, _, recipe = tiny_components()
    recipe["training"].update(training)
    actual = copy.deepcopy(reference)
    # Hooks are installed on the LLM: use a second matched model for the new path.
    _, second_llm, actual, _, _ = tiny_components(recipe)
    device = torch.device("cpu")
    parallel = ParallelContext(device)
    batches = trainer.shape_batches(legacy.records, 4, 42, 0)
    assert batches == trainer.shape_batches(prepared.records, 4, 42, 0)
    batch = next(b for b in batches if len(b) == 4 and min(legacy.records[b[0]]["grid_shape"]) > 1)
    old_cache = trainer.TrainingTokenCache(tokenizer, training)
    lookup = PreparedTokenLookup(prepared, trainer.TrainingTokenCache(tokenizer, training))
    expected, _, _ = trainer.backward_update(llm, reference, tokenizer, legacy, [batch],
        device, torch.float32, recipe["training"], parallel, 2, token_cache=old_cache)
    expected_gradients = {k: p.grad.clone() for k, p in reference.named_parameters() if p.grad is not None}
    calls = 0
    original_loss = trainer.training_loss

    def failure_after_partial_backward(*args, **kwargs):
        nonlocal calls
        loss, components = original_loss(*args, **kwargs)
        calls += 1
        if calls == 2:
            loss.backward()
            raise torch.cuda.OutOfMemoryError("injected local retry, no GPU involved")
        return loss, components

    try:
        with PreparedBatchLoader(cache_dir, workers=0, max_bytes=2**20) as loader:
            update = loader.load(batch)
            with monkeypatch.context() as patch:
                patch.setattr(trainer, "training_loss", failure_after_partial_backward)
                observed, _, reports = trainer.backward_update(second_llm, actual, tokenizer,
                    prepared, [batch], device, torch.float32, recipe["training"], parallel, 2,
                    token_cache=lookup, prepared_update=update)
            assert reports[0]["oom_retries"] == 1
            assert sum(reports[0]["micro_batch_sizes"]) == len(batch)
            assert observed == pytest.approx(expected, rel=2e-6, abs=2e-6)
            for name, parameter in actual.named_parameters():
                if name in expected_gradients:
                    torch.testing.assert_close(parameter.grad, expected_gradients[name], rtol=3e-4, atol=3e-6)
            loader.release(update.key)
    finally:
        legacy.close()
        prepared.close()


@pytest.mark.parametrize("workers", [0, 2])
def test_legacy_checkpoint_continues_on_prepared_pipeline(cache_inputs, tmp_path, monkeypatch, workers):
    from transformers import Qwen2ForCausalLM

    qa, cache_dir, tokenizer, training, _ = cache_inputs
    manifest = prepare(cache_inputs)
    recipe = trainer.load_config(trainer.ROOT / "configs/field_to_llm_multisource_v2.yaml", "smoke")
    recipe["data"].update(copy.deepcopy(manifest["metadata"]["config"]["data"]))
    recipe["generation"].update(copy.deepcopy(manifest["metadata"]["config"]["generation"]))
    _, example, _, _, recipe = tiny_components(recipe)
    recipe["field_encoder"]["model"]["base_channels"] = 16
    recipe["data"]["qa_dir"] = str(qa)
    recipe["experiment_profile"] = "smoke"
    recipe["training"].update(training, max_updates=3, batch_size=3, gradient_accumulation_steps=2,
                              save_every_updates=2, eval_every_updates=3, log_interval=1)
    recipe["evaluation"]["max_new_tokens"] = training["max_target_tokens"]
    recipe["runtime"].update(length_balanced_batches=True, micro_batch_token_budget=4096)
    model_config = copy.deepcopy(example.config)
    configuration = tmp_path / "recipe.json"
    configuration.write_text(json.dumps(recipe), encoding="utf-8")

    def load_model(args, device):
        torch.manual_seed(123)
        return Qwen2ForCausalLM(copy.deepcopy(model_config)).to(device), torch.float32

    monkeypatch.setattr(trainer.core, "load_tokenizer", lambda args: tokenizer)
    monkeypatch.setattr(trainer.core, "load_llm_with_bounded_host_memory", load_model)
    monkeypatch.setattr(trainer, "generate_from_prompt", lambda *a, **k:
        {"prediction": "[0.0]", "terminated": True, "stop_token_id": 2, "generated_tokens": 3})

    def run(out, *extra):
        monkeypatch.setattr(sys, "argv", ["train_point_readout.py", "--config", str(configuration),
            "--profile", "smoke", "--device", "cpu", "--output-dir", str(out), *extra])
        trainer.main()

    direct, switched = tmp_path / "direct", tmp_path / "switched"
    run(direct)
    run(switched, "--stop-after-updates", "1")
    run(switched, "--resume", str(switched / "last.pt"), "--training-cache", str(cache_dir),
        "--input-workers", str(workers), "--prefetch-updates", "2", "--prefetch-mib", "8", "--profile-updates", "2")
    expected = torch.load(direct / "last.pt", weights_only=True)
    observed = torch.load(switched / "last.pt", weights_only=True)
    assert expected["identity"] == observed["identity"]
    assert expected["step"] == observed["step"] == 3
    assert (expected["epoch"], expected["cursor"]) == (observed["epoch"], observed["cursor"])
    for key in expected["sidecar"]:
        torch.testing.assert_close(expected["sidecar"][key], observed["sidecar"][key], rtol=0, atol=0)
    assert expected["scheduler"] == observed["scheduler"]
    pipeline = json.loads((switched / "input_pipeline.json").read_text())
    assert pipeline["mode"] == "prepared_mmap" and pipeline["workers_per_rank"] == workers
    profiles = [json.loads(line) for line in (switched / "input_profile.jsonl").read_text().splitlines()]
    assert [row["step"] for row in profiles] == [2, 3]
    assert "text_pack" in profiles[0]["per_rank_execution"][0]["input_stages"]["cpu_seconds"]
    summary = json.loads((switched / "run_summary.json").read_text())
    assert summary["prepared_input"][0]["peak_reserved_forecast_bytes"] <= 8 * 2**20


def test_read_ahead_preserves_accumulation_boundaries_and_rank_assignment():
    class Rank:
        rank, world_size = 1, 3
    class Loader:
        prefetch_updates = 3
        def __init__(self):
            self.entries = []
        def prefetch(self, key, indices):
            self.entries.append((key, indices))
            return True
    batches = [list(range(start, start + 5)) for start in range(0, 35, 5)]
    lengths = [100 + i for i in range(35)]
    loader = Loader()
    queue_following_updates(loader, step=10, maximum=20, batches=batches, cursor=2,
        accumulation=2, parallel=Rank(), token_lengths=lengths, balance_by_length=True)
    assert loader.entries == [(11, local_update_indices(batches[2:4], Rank(), lengths, True)),
                              (12, local_update_indices(batches[4:6], Rank(), lengths, True))]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable locally")
def test_prepared_pinned_cuda_update_and_stage_events(cache_inputs):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("Local CUDA device does not support BF16")
    qa, cache_dir, tokenizer, training, _ = cache_inputs
    prepare(cache_inputs)
    legacy = MultiSourcePointReadoutDataset(qa, "train")
    prepared = PreparedTrainingDataset(cache_dir)
    _, reference_llm, reference, _, recipe = tiny_components()
    _, actual_llm, actual, _, _ = tiny_components(recipe)
    recipe["training"].update(training)
    device, dtype = torch.device("cuda", 0), torch.bfloat16
    parallel = ParallelContext(device)
    for llm, sidecar in ((reference_llm, reference), (actual_llm, actual)):
        llm.to(device=device, dtype=dtype)
        sidecar.to(device=device)
    batches = trainer.shape_batches(legacy.records, 2, 42, 0)
    batch = next(b for b in batches if len(b) == 2 and min(legacy.records[b[0]]["grid_shape"]) > 1)
    try:
        expected, _, _ = trainer.backward_update(reference_llm, reference, tokenizer, legacy,
            [batch], device, dtype, recipe["training"], parallel, 2,
            token_cache=trainer.TrainingTokenCache(tokenizer, training))
        with PreparedBatchLoader(cache_dir, workers=0) as loader:
            update = loader.load(batch)
            observed, _, reports = trainer.backward_update(actual_llm, actual, tokenizer, prepared,
                [batch], device, dtype, recipe["training"], parallel, 2,
                prepared_update=update, pin_memory=True, profile_cuda=True)
            assert observed == pytest.approx(expected, rel=1e-5, abs=1e-6)
            expected_params = dict(reference.named_parameters())
            for name, parameter in actual.named_parameters():
                if expected_params[name].grad is not None:
                    torch.testing.assert_close(parameter.grad, expected_params[name].grad, rtol=1e-5, atol=1e-6)
            stages = reports[0]["input_stages"]
            assert stages["cuda_events_enabled"]
            for name in ("host_to_device", "field_memory", "qwen_forward", "answer_ce", "backward"):
                assert stages["cuda_milliseconds"][name] >= 0
            assert stages["cpu_seconds"]["pin_and_transfer_submit"] > 0
            loader.release(update.key)
    finally:
        legacy.close()
        prepared.close()
