"""Real CPU/Gloo multi-process checks; these do not measure A800/Qwen14B readiness."""
from __future__ import annotations

import copy
from datetime import timedelta
import json
import os
import pickle
from pathlib import Path
import random
import sys
import tempfile
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts import train_point_readout as trainer
from scripts.readout_distributed import ParallelContext, partition_batch, evaluation_indices, capture_rng, restore_rng
from test_point_readout import tiny_components, data_fixture
from test_point_readout_performance import TinyDataset, sample_rows


@pytest.mark.parametrize("count", [0, 1, 2, 7, 16, 48, 49])
def test_partitions_preserve_each_question_once(count):
    indices = list(range(count))
    parts = [partition_batch(indices, rank, 3) for rank in range(3)]
    assert sum(parts, []) == indices
    assert max(map(len, parts)) - min(map(len, parts)) <= 1


def test_eval_keeps_field_groups_and_rejects_missing_duplicate_rows(tmp_path):
    dataset = TinyDataset()
    parts = evaluation_indices(dataset.records, 3)
    assert sorted(sum(parts, [])) == list(range(len(dataset)))
    owners = {}
    for rank, indices in enumerate(parts):
        for index in indices:
            state = dataset.records[index]["state_ref"]
            assert owners.setdefault(state, rank) == rank
    files = [tmp_path / f"rank{rank}.jsonl" for rank in range(3)]
    for path, indices in zip(files, parts):
        path.write_text("".join(json.dumps({"qa_id": dataset.records[i]["qa_id"]}) + "\n" for i in indices))
    output = tmp_path / "merged.jsonl"
    result = trainer.merge_evaluation_shards(dataset, files, output)
    assert [r["qa_id"] for r in result] == [r["qa_id"] for r in dataset.records]
    with files[0].open("a") as handle:
        handle.write(json.dumps(result[0]) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        trainer.merge_evaluation_shards(dataset, files, output)
    files[0].write_text("")
    with pytest.raises(ValueError, match="missing"):
        trainer.merge_evaluation_shards(dataset, files, output)


class ToySidecar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.7))
        self.register_buffer("calls", torch.tensor(0))

    def clear(self):
        pass


def test_oom_retries_entire_local_update_without_loss_or_partial_gradients(monkeypatch):
    device, context = torch.device("cpu"), ParallelContext("cpu")
    dataset = [(dict(target=i), torch.tensor([float(i)])) for i in range(7)]
    inject = [True]
    def loss_fn(llm, module, tokenizer, records, fields, *args, **kwargs):
        module.calls.add_(1)
        random.random()
        torch.rand(1)
        if inject[0] and len(records) == 3:
            inject[0] = False
            # Simulate an OOM after earlier backwards and a partial failed backward.
            module.weight.grad.add_(123)
            raise torch.cuda.OutOfMemoryError("injected, not a real GPU allocation")
        loss = (module.weight * fields).square().mean()
        return loss, {"answer_ce": float(loss.detach()), "reconstruction": 0.0}
    monkeypatch.setattr(trainer, "training_loss", loss_fn)
    rng = capture_rng(device)
    actual = ToySidecar()
    values, cap, reports = trainer.backward_update(None, actual, None, dataset,
        [list(range(7))], device, torch.float32, {}, context, 4)
    actual_rng = capture_rng(device)
    restore_rng(rng, device)
    expected = ToySidecar()
    reference, _, _ = trainer.backward_update(None, expected, None, dataset,
        [list(range(7))], device, torch.float32, {}, context, 2)
    assert cap == 2 and reports[0]["oom_retries"] == 1
    assert reports[0]["questions"] == 7 and reports[0]["micro_batch_sizes"] == [2, 2, 2, 1]
    torch.testing.assert_close(actual.weight.grad, expected.weight.grad, rtol=0, atol=0)
    assert values == reference and actual.calls == expected.calls == 4
    assert actual_rng["python_rng"] == random.getstate()
    assert torch.equal(actual_rng["torch_rng"], torch.get_rng_state())


class FileCollectives:
    """Test-only collective transport for Windows builds without Gloo transport.

    Three real processes exchange CPU tensors/objects at matching barriers. This
    exercises the trainer's protocol, NOT the production NCCL/Gloo transport.
    """
    def __init__(self, root, rank, world, barrier):
        self.root, self.rank, self.world, self.barrier = Path(root), rank, world, barrier
        self.sequence = 0

    def exchange(self, value):
        sequence = self.sequence
        self.sequence += 1
        path = self.root / f"collective_{sequence}_{self.rank}.pkl"
        with path.open("wb") as handle:
            pickle.dump(value, handle)
        self.barrier.wait(timeout=120)
        values = []
        for rank in range(self.world):
            with (self.root / f"collective_{sequence}_{rank}.pkl").open("rb") as handle:
                values.append(pickle.load(handle))
        self.barrier.wait(timeout=120)
        path.unlink()
        return values

    def all_reduce(self, tensor, op=None):
        values = torch.stack(self.exchange(tensor.detach().clone()))
        tensor.copy_(values.max(dim=0).values if op == dist.ReduceOp.MAX else values.sum(dim=0))

    def all_gather_object(self, output, value):
        assert not torch.is_inference_mode_enabled(), "Collective receive buffers must be ordinary tensors"
        output[:] = self.exchange(value)

    def broadcast_object_list(self, envelope, src):
        assert not torch.is_inference_mode_enabled(), "Collective receive buffers must be ordinary tensors"
        envelope[:] = self.exchange(envelope)[src]

    def broadcast(self, tensor, src):
        tensor.copy_(self.exchange(tensor.detach().clone())[src])


def _distributed_worker(rank, world, rendezvous, root, config_path, barrier=None):
    os.environ.update(WORLD_SIZE=str(world), RANK=str(rank), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    root = Path(root)
    device = torch.device("cpu")
    # A real tiny Qwen reference, including backward through its frozen layers.
    _, llm, sidecar, tokenizer, config = tiny_components()
    config["evaluation"].update(max_new_tokens=3)
    rows, fields = sample_rows()
    examples = [(rows[i % 2], fields["a"]) for i in range(7)]
    single = ParallelContext(device)
    trainer.backward_update(llm, sidecar, tokenizer, examples, [list(range(7))], device,
        torch.float32, config["training"], single, 7)
    reference = {name: p.grad.clone() for name, p in sidecar.named_parameters() if p.grad is not None}
    sidecar.clear()
    reference_metrics = trainer.evaluate(llm, sidecar, tokenizer, TinyDataset(), device, torch.float32,
                                        config, root / f"reference{rank}.jsonl")
    transport_patch = None
    if barrier is None:
        dist.init_process_group("gloo", store=dist.FileStore(rendezvous, world), rank=rank, world_size=world,
                                timeout=timedelta(seconds=180))
    else:
        transport = FileCollectives(root, rank, world, barrier)
        transport_patch = patch.multiple(dist, is_initialized=lambda: True, get_rank=lambda: rank,
            get_world_size=lambda: world, all_reduce=transport.all_reduce,
            all_gather_object=transport.all_gather_object, broadcast=transport.broadcast,
            broadcast_object_list=transport.broadcast_object_list)
        transport_patch.start()
    try:
        context = ParallelContext(device)
        # Unequal local counts 3/2/2 and smaller chunks 2/1 must preserve the mean.
        _, llm, sidecar, tokenizer, config = tiny_components()
        config["evaluation"].update(max_new_tokens=3)
        config["runtime"].update(eval_batch_size=8, eval_parity_check_questions=32)
        context.synchronize_parameters(sidecar)
        _, _, execution = trainer.backward_update(llm, sidecar, tokenizer, examples, [list(range(7))],
            device, torch.float32, config["training"], context, 2)
        assert [r["questions"] for r in execution] == [3, 2, 2]
        for name, p in sidecar.named_parameters():
            if name in reference:
                torch.testing.assert_close(p.grad, reference[name], rtol=2e-4, atol=2e-6)
        metrics = trainer.evaluate(llm, sidecar, tokenizer, TinyDataset(), device, torch.float32,
            config, root / "distributed_predictions.jsonl", parallel=context)
        assert (root / "distributed_predictions.jsonl").read_bytes() == (root / f"reference{rank}.jsonl").read_bytes()
        for key in ("questions", "macro_task_answer_accuracy", "by_family", "seen_shapes", "heldout_shapes"):
            assert metrics[key] == reference_metrics[key]
        assert sum(m["questions"] for m in metrics["per_rank"]) == 4
        assert any(m["questions"] == 0 for m in metrics["per_rank"])

        # Empty training rank and globally-unused parameter retain correct behavior.
        toy = ToySidecar()
        toy.register_parameter("unused", torch.nn.Parameter(torch.tensor(5.0)))
        if rank < 2:
            (toy.weight * (rank + 1) / 2).backward()
        context.sum_gradients(toy)
        assert toy.weight.grad.item() == 1.5 and toy.unused.grad is None
        assert context.any(rank == 2)  # A stop request on any rank reaches everyone.
        with pytest.raises(RuntimeError, match="rank 1: injected failure"):
            context.check_errors("injected failure" if rank == 1 else None)
        def fail_writer():
            raise RuntimeError("injected writer failure")
        with pytest.raises(RuntimeError, match="injected writer failure"):
            context.main(fail_writer)
        # Only one rank OOMs/retries; peers must wait and still apply exactly
        # the complete seven-question gradient, with no extra optimizer step.
        toy = ToySidecar()
        oom_once = [rank == 1]
        def toy_loss(llm, module, tokenizer, records, fields, *args, **kwargs):
            if oom_once[0]:
                oom_once[0] = False
                raise torch.cuda.OutOfMemoryError("injected on one rank")
            loss = (module.weight * fields).square().mean()
            return loss, {"answer_ce": float(loss.detach()), "reconstruction": 0.0}
        toy_data = [({"target": i}, torch.tensor([float(i)])) for i in range(7)]
        with patch.object(trainer, "training_loss", toy_loss):
            _, cap, retry_reports = trainer.backward_update(None, toy, None, toy_data,
                [list(range(7))], device, torch.float32, {}, context, 2)
        torch.testing.assert_close(toy.weight.grad, torch.tensor(2 * .7 * 13), rtol=1e-6, atol=1e-6)
        assert [r["oom_retries"] for r in retry_reports] == [0, 1, 0]
        assert cap == (1 if rank == 1 else 2)

        # Full CLI: actual tiny Qwen training, saving all ranks, strict resume.
        from transformers import Qwen2ForCausalLM
        llm_config = copy.deepcopy(llm.config)
        def load_model(args, target_device):
            torch.manual_seed(123)
            model = Qwen2ForCausalLM(copy.deepcopy(llm_config)).to(target_device)
            model.config.use_cache = False
            if args.llm_gradient_checkpointing:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            return model, torch.float32
        def run(name, *options):
            argv = ["train_point_readout.py", "--config", config_path, "--profile", "smoke",
                    "--device", "cpu", "--output-dir", str(root / name), *options]
            with patch.object(sys, "argv", argv):
                trainer.main()
        with patch.object(trainer.core, "load_tokenizer", lambda args: tokenizer), \
             patch.object(trainer.core, "load_llm_with_bounded_host_memory", load_model):
            run("complete")
            run("interrupted", "--stop-after-updates", "1")
            run("interrupted", "--resume", str(root / "interrupted/last.pt"),
                "--adaptive-micro-batch", "--eval-batch-size", "8", "--eval-parity-check-questions", "32")
        expected = torch.load(root / "complete/last.pt", weights_only=True)
        actual = torch.load(root / "interrupted/last.pt", weights_only=True)
        assert expected["step"] == actual["step"] == 2
        assert (expected["epoch"], expected["cursor"]) == (actual["epoch"], actual["cursor"])
        assert len(actual["rank_states"]) == world
        assert actual["parallel_execution"]["world_size"] == world
        assert actual["identity"] == expected["identity"]
        for wanted, restored in zip(expected["rank_states"], actual["rank_states"]):
            assert wanted["python_rng"] == restored["python_rng"]
            assert torch.equal(wanted["torch_rng"], restored["torch_rng"])
            assert wanted["micro_batch_limit"] == restored["micro_batch_limit"] == 2
            assert restored["adaptive_micro_batch_limits"] == []
        for name, value in expected["sidecar"].items():
            torch.testing.assert_close(value, actual["sidecar"][name], rtol=0, atol=0)
        (root / f"rank{rank}.passed").write_text("ok")
    finally:
        if transport_patch is None:
            dist.destroy_process_group()
        else:
            transport_patch.stop()


@pytest.mark.parametrize("transport", ["gloo", "file_test_double"])
def test_three_process_training_evaluation_and_exact_resume(data_fixture, tmp_path, transport):
    if transport == "gloo":
        if not dist.is_available() or not dist.is_gloo_available():
            pytest.skip("CPU Gloo unavailable")
        try:
            dist.ProcessGroupGloo.create_device(hostname="127.0.0.1")
        except RuntimeError as exc:
            if "unsupported gloo device" in str(exc):
                pytest.skip("Installed PyTorch exposes Gloo but has no usable Gloo transport")
            raise
    hdf5, qa, config, metadata = data_fixture
    _, _, _, _, config = tiny_components(config)
    config["data"].update(qa_dir=str(qa), hdf5_path=str(hdf5))
    config["model"]["gradient_checkpointing"] = True
    config["training"].update(batch_size=7, gradient_accumulation_steps=2, max_updates=2,
                              save_every_updates=1, eval_every_updates=2, log_interval=1)
    config["runtime"]["max_micro_batch_size"] = 2
    config["experiment_profile"] = "smoke"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    # FileStore's native Windows file API does not handle percent-encoded URIs.
    with tempfile.TemporaryDirectory(prefix="readout_gloo_") as store_dir:
        rendezvous = str(Path(store_dir) / "store")
        barrier = mp.get_context("spawn").Barrier(3) if transport == "file_test_double" else None
        mp.spawn(_distributed_worker, args=(3, rendezvous, str(tmp_path), str(path), barrier), nprocs=3, join=True)
    assert all((tmp_path / f"rank{rank}.passed").is_file() for rank in range(3))


def test_token_report_reduction_keeps_global_maxima_and_overflows():
    from scripts.readout_distributed import merge_token_reports
    tokenizer = tiny_components()[3]
    records, _ = sample_rows()
    training = {"max_prompt_tokens": 10, "max_target_tokens": 96}
    reports = [trainer.preflight_token_budgets(records[rank::3], tokenizer, training,
               raise_on_overflow=False) for rank in range(3)]
    merged = merge_token_reports(reports)
    reference = trainer.preflight_token_budgets(records, tokenizer, training, raise_on_overflow=False)
    for key in ("questions", "max_prompt_tokens_observed", "max_target_tokens_observed",
                "overflow_questions", "prompt_overflow_count", "target_overflow_count", "status"):
        assert merged[key] == reference[key]
    assert {r["qa_id"] for r in merged["overflow_examples"]} == {r["qa_id"] for r in records}
