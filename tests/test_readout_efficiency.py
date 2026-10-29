"""Local fixture/tiny-model checks; no remote Qwen14B performance claims."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import h5py
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(folder))

from scripts import benchmark_readout_efficiency as bench
from test_point_readout import data_fixture, tiny_components
from test_multisource_readout import config as multi_config
from tensor_compression.downstream.point_readout import normalize_field, make_record
from tensor_compression.downstream.point_readout_data import PointReadoutDataset, build_dataset


def test_selection_is_stable_paired_balanced_and_covers_boundaries():
    rows, states = [], {}
    for area in (1, 64, 65, 256, 257, 1024, 1025, 4096, 4097, 16384, 16385):
        for task in ("a", "b"):
            ref = f"{area}-{task}"
            rows.append({"qa_id": ref, "state_ref": ref, "task_type": task, "grid_shape": [1, area]})
            states[ref] = {"source_id": "source"}
    dataset = SimpleNamespace(records=rows, states=states, identity="fixture")
    plan = bench.select_requests(dataset, [64, 256, 1024, 4096, 16384], 2, 17)
    dataset.records = list(reversed(rows))
    assert plan == bench.select_requests(dataset, [64, 256, 1024, 4096, 16384], 2, 17)
    assert len(plan["requests"]) == 10
    assert plan["excluded_above_largest_bin"] == 2
    for stratum in plan["strata"]:
        assert max(r["shape"][0] * r["shape"][1] for r in plan["requests"]
                   if r["bin"] == stratum["bin"]) == stratum["largest_available_cells"]
    assert all(s["tasks"] == {"a": 1, "b": 1} or s["tasks"] == {"b": 1, "a": 1} for s in plan["strata"])
    with pytest.raises(ValueError):
        bench.select_requests(dataset, [64, 32], 2, 17)


def test_raw_single_source_matches_QA_and_detects_mutation(data_fixture):
    source, qa, _, _ = data_fixture
    dataset = PointReadoutDataset(qa, source, "val", cache_size=0)
    raw = bench.RawFields(dataset)
    try:
        row = dataset.records[0]
        request = {"record": row, "raw": raw.read(row["state_ref"])}
        raw.audit(request)
        request["raw"][0, 0] += 1
        with pytest.raises(ValueError, match="changed"):
            raw.audit(request)
    finally:
        raw.close()
        dataset.close()


def test_multisource_reads_original_instead_of_standardized_cache(multi_config, tmp_path):
    qa = tmp_path / "qa"
    build_dataset(multi_config, None, qa)
    dataset = PointReadoutDataset(qa, None, "val", cache_size=0)
    reader = bench.RawFields(dataset)
    try:
        # Include scalar, vector, time-independent and constant fields.
        fields = {}
        for row in dataset.records:
            state = dataset.states[row["state_ref"]]
            fields.setdefault((state["source_id"], state["field"]), row)
        for row in fields.values():
            raw = reader.read(row["state_ref"])
            reader.audit({"record": row, "raw": raw})
        row = next(r for r in fields.values() if dataset.states[r["state_ref"]]["field"] == "t0_fields/density")
        raw = reader.read(row["state_ref"])
        z, _ = reader.normalize(raw)
        assert torch.all(z == 0) and not torch.all(raw == 0)
    finally:
        reader.close()
        dataset.close()


def test_checkpoint_snapshot_pins_hash_and_drops_optimizer(tmp_path):
    path = tmp_path / "last.pt"
    payload = {"checkpoint_type": bench.trainer.CHECKPOINT_TYPE, "step": 123,
               "identity": {"dataset": "x"}, "sidecar": {"weight": torch.ones(3)},
               "optimizer": {"large": torch.zeros(100)}}
    torch.save(payload, path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    state, metadata = bench.load_checkpoint_snapshot(path)
    assert "optimizer" not in state and metadata["sha256"] == before
    assert metadata["step"] == 123
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def setup_request(block=(1, 1)):
    trainer, llm, sidecar, tokenizer, config = tiny_components()
    if block != (1, 1):
        config["memory"]["block_shape"] = list(block)
        trainer, llm, sidecar, tokenizer, config = tiny_components(config)
    raw = torch.arange(15).reshape(3, 5).float().square()
    z, _ = normalize_field(raw)
    row = make_record("test", z, "single_point", {"points": [[2, 3]]})
    request = {"record": row, "raw": raw, "bin": "cells_1_64"}
    normalizer = SimpleNamespace(normalize=lambda x: normalize_field(x))
    config["evaluation"]["max_new_tokens"] = 4
    return trainer, llm, sidecar, tokenizer, config, request, normalizer


@pytest.mark.parametrize("block", [(1, 1), (2, 2)])
def test_timed_generation_matches_existing_decoder_and_reencodes_each_request(block):
    trainer, llm, sidecar, tokenizer, config, request, normalizer = setup_request(block)
    llm.eval()
    sidecar.eval()
    z = normalizer.normalize(request["raw"])[0]
    before = {k: v.clone() for k, v in sidecar.state_dict().items()}
    reference = trainer.generate_answer(llm, sidecar, tokenizer, request["record"], z[None],
        torch.device("cpu"), torch.float32, max_new_tokens=4, max_prompt_tokens=2048)
    encodes = []
    hook = sidecar.memory.register_forward_hook(lambda *args: encodes.append(True))
    try:
        for _ in range(2):
            result = bench.measure_request(request, normalizer, llm, sidecar, tokenizer, torch.device("cpu"),
                                           torch.float32, config, "interface", "natural", 4)
            for key in ("prediction", "terminated", "generated_tokens", "stop_token_id"):
                assert result[key] == reference[key]
            assert result["request_ms"] >= result["ttft_ms"] > 0
            assert result["memory_tokens"] == (15 if block == (1, 1) else 6)
            assert sidecar._bound_state is None
        assert len(encodes) == 2
        assert all(torch.equal(v, before[k]) for k, v in sidecar.state_dict().items())
    finally:
        hook.remove()


def test_fixed_decode_runs_full_budget_even_when_every_token_is_EOS():
    trainer, llm, sidecar, tokenizer, _, request, _ = setup_request()
    llm.eval()
    sidecar.bind(None, mode="no_tensor")
    class EOSHead(torch.nn.Module):
        def forward(self, hidden):
            logits = hidden.new_zeros((hidden.shape[0], 256))
            logits[:, tokenizer.eos_token_id] = 100
            return logits
    llm.set_output_embeddings(EOSHead())
    prompt = trainer.encode_prompt(request["record"], tokenizer, 2048)
    natural = bench.timed_decode(llm, tokenizer, prompt, torch.device("cpu"), torch.float32, 5)
    fixed = bench.timed_decode(llm, tokenizer, prompt, torch.device("cpu"), torch.float32, 5, fixed=True)
    assert natural["terminated"] and natural["generated_tokens"] == 1
    assert not fixed["terminated"] and fixed["generated_tokens"] == 5


def test_context_overflow_is_recorded_without_generation_or_truncation(monkeypatch):
    trainer, llm, sidecar, tokenizer, config, request, normalizer = setup_request()
    llm.config.max_position_embeddings = 10
    def unexpected(*args, **kwargs):
        pytest.fail("Must not decode over-context requests")
    monkeypatch.setattr(bench, "timed_decode", unexpected)
    result = bench.measure_request(request, normalizer, llm, None, tokenizer, torch.device("cpu"),
                                   torch.float32, config, "serialized", "natural", 4)
    assert result["status"] == "context_exceeded" and result["prompt_tokens"] > 10
    assert "prediction" not in result


def test_oom_keeps_failure_and_clears_memory(monkeypatch):
    _, llm, sidecar, tokenizer, config, request, normalizer = setup_request()
    def fail(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("fixture")
    monkeypatch.setattr(bench, "timed_decode", fail)
    result = bench.measure_request(request, normalizer, llm, sidecar, tokenizer, torch.device("cpu"),
                                   torch.float32, config, "interface", "natural", 4)
    assert result["status"] == "oom" and sidecar._bound_state is None


def test_summary_pairs_identical_completed_requests_and_keeps_failures():
    rows = []
    for method, latency in (("serialized", 20.), ("interface", 10.)):
        for number, status in enumerate(("complete", "length_limit", "oom")):
            rows.append({"method": method, "decode_mode": "natural", "bin": "x", "qa_id": str(number),
                         "repeat": 0, "request_ms": latency, "status": status, "generated_tokens": 4,
                         "ttft_ms": 1.})
    summary = bench.summarize(rows, [{"bin": "x"}])
    pair = summary["paired_comparisons"][0]
    assert pair["paired_completed_requests"] == 1
    assert pair["serialized_over_interface_time"] == 2
    assert summary["groups"][0]["statuses"] == {"complete": 1, "length_limit": 1, "oom": 1}
    assert summary["groups"][0]["completed_requests_per_active_second"] == pytest.approx(1 / .03)


def test_audit_only_cli_leaves_training_artifacts_unchanged(data_fixture, tmp_path, monkeypatch):
    source, qa, config, _ = data_fixture
    run, output = tmp_path / "training", tmp_path / "benchmark"
    run.mkdir()
    config = copy.deepcopy(config)
    config["data"].update(qa_dir=str(qa), hdf5_path=str(source))
    (run / "resolved_config.json").write_text(json.dumps(config), encoding="utf-8")
    dataset = PointReadoutDataset(qa, source, "val")
    identity = dataset.identity
    dataset.close()
    torch.save({"checkpoint_type": bench.trainer.CHECKPOINT_TYPE, "identity": {"dataset": identity},
                "sidecar": {}, "step": 17}, run / "last.pt")
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir()}
    monkeypatch.setattr(sys, "argv", ["benchmark", "--run-dir", str(run), "--checkpoint-name", "last.pt",
        "--training-status", "in_progress", "--output-dir", str(output), "--audit-only", "--per-bin", "2"])
    bench.main()
    assert digests == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir()}
    assert (output / "selection.json").is_file()
    assert not (output / "requests.jsonl").exists()
    contract = json.loads((output / "contract.json").read_text(encoding="utf-8"))
    assert contract["checkpoint"]["source_training_status"] == "in_progress"


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() != 1,
                    reason="Single local CUDA device required for tiny-model orchestration check")
def test_full_cli_with_local_tiny_model_only(data_fixture, tmp_path, monkeypatch):
    """Exercise real CUDA timers/peaks and file outputs; no Qwen14B assets used."""
    source, qa, original, _ = data_fixture
    trainer, llm, sidecar, tokenizer, config = tiny_components(original)
    config["data"].update(qa_dir=str(qa), hdf5_path=str(source))
    config["evaluation"]["max_new_tokens"] = 2
    config["training"]["max_target_tokens"] = 2
    trainer.core.set_frozen_llm_execution_mode(llm, checkpoint_training=False)
    dataset = PointReadoutDataset(qa, source, "val")
    identity = trainer.run_identity(config, dataset, llm, tokenizer)
    identity["model_asset"] = {"fixture": "random tiny Qwen, not pretrained Qwen14B"}
    dataset.close()
    run, output = tmp_path / "training", tmp_path / "benchmark"
    run.mkdir()
    (run / "resolved_config.json").write_text(json.dumps(config), encoding="utf-8")
    torch.save({"checkpoint_type": trainer.CHECKPOINT_TYPE, "identity": identity,
                "sidecar": sidecar.state_dict(), "step": 1}, run / "last.pt")
    # Restore the undecorated backbone before the baseline is measured.
    for index, layer in enumerate(llm.model.layers):
        llm.model.layers[index] = layer.base_layer
    monkeypatch.setattr(trainer.core, "load_tokenizer", lambda args: tokenizer)
    monkeypatch.setattr(trainer.core, "load_llm_with_bounded_host_memory",
                        lambda args, device: (llm.to(device), torch.float32))
    monkeypatch.setattr(trainer, "model_asset_identity", lambda *a, **kw: identity["model_asset"])
    monkeypatch.setattr(sys, "argv", ["benchmark", "--run-dir", str(run), "--checkpoint-name", "last.pt",
        "--training-status", "in_progress", "--output-dir", str(output), "--per-bin", "1",
        "--repeats", "1", "--warmup", "1", "--fixed-tokens", "2"])
    bench.main()
    contract = json.loads((output / "contract.json").read_text(encoding="utf-8"))
    assert contract["status"] == "complete" and contract["batch_size"] == 1
    results = [json.loads(line) for line in (output / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["method"] for r in results} == {"interface", "serialized"}
    assert all(r["peak_allocated_gib"] > 0 for r in results)
    assert all(r["generated_tokens"] == 2 for r in results if r["decode_mode"] == "fixed")
    assert all(r["request_ms"] >= r["ttft_ms"] for r in results)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert any(p["paired_completed_requests"] for p in summary["paired_comparisons"])
    assert (output / "summary.csv").is_file()
