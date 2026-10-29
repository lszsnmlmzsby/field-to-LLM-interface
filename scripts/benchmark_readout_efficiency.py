"""Single-GPU, raw-field-to-answer benchmark; existing training runs stay read-only.

Uses the saved QA and recipe. Size strata select timing requests only, never
rebuild crops or change training. Baseline runs before installing any interface.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import h5py
import torch

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(folder))

from scripts import train_point_readout as trainer
from scripts.benchmark_point_readout import serialized_messages
from tensor_compression.downstream.point_readout import normalize_field, validate_record, json_hash
from tensor_compression.downstream.point_readout_data import (
    PointReadoutDataset, load_config, resolve_path, source_path, write_json,
)


def select_requests(dataset, edges, per_bin, seed):
    """Stable, task/source round-robin selection, independent of predictions."""
    if not edges or edges != sorted(set(edges)) or min(edges) <= 0 or per_bin <= 0:
        raise ValueError("Area edges must be unique increasing positive integers; per-bin must be positive")
    groups = defaultdict(lambda: defaultdict(list))
    excluded = 0
    for row in dataset.records:
        h, w = row["grid_shape"]
        size = next((i for i, edge in enumerate(edges) if h * w <= edge), None)
        if size is None:
            excluded += 1
            continue
        state = dataset.states[row["state_ref"]]
        source = state.get("source_id", "single_source")
        groups[size][(row["task_type"], source)].append(row)
    selected, strata = [], []
    for i, upper in enumerate(edges):
        lower = 1 if i == 0 else edges[i - 1] + 1
        label = f"cells_{lower}_{upper}"
        available = sum(len(rows) for rows in groups[i].values())
        queues = []
        for key in sorted(groups[i]):
            rows = sorted(groups[i][key], key=lambda r: hashlib.sha256(
                f"{seed}:{r['qa_id']}".encode()).hexdigest())
            queues.append(deque(rows))
        random.Random(seed + i).shuffle(queues)
        chosen = []
        while queues and len(chosen) < per_bin:
            queue = queues.pop(0)
            chosen.append(queue.popleft())
            if queue:
                queues.append(queue)
        # Retain the actual largest grid in each band, including 128x128 when present.
        all_rows = [row for values in groups[i].values() for row in values]
        largest = max((math.prod(r["grid_shape"]) for r in all_rows), default=None)
        if chosen and max(math.prod(r["grid_shape"]) for r in chosen) < largest:
            candidates = [r for r in all_rows if math.prod(r["grid_shape"]) == largest]
            last = chosen[-1]
            matched = [r for r in candidates if r["task_type"] == last["task_type"] and
                       dataset.states[r["state_ref"]].get("source_id", "single_source") ==
                       dataset.states[last["state_ref"]].get("source_id", "single_source")]
            chosen[-1] = min(matched or candidates, key=lambda r: hashlib.sha256(
                f"{seed}:{r['qa_id']}".encode()).hexdigest())
        for row in chosen:
            selected.append({"bin": label, "qa_id": row["qa_id"], "state_ref": row["state_ref"],
                             "shape": row["grid_shape"], "task": row["task_type"]})
        strata.append({"bin": label, "min_cells": lower, "max_cells": upper,
                       "largest_available_cells": largest,
                       "available_questions": available, "selected_questions": len(chosen),
                       "tasks": dict(Counter(r["task_type"] for r in chosen)),
                       "sources": dict(Counter(dataset.states[r["state_ref"]].get("source_id", "single_source")
                                               for r in chosen))})
    if not selected:
        raise ValueError("No existing QA fits the requested size bins")
    return {"dataset": dataset.identity, "selection_seed": seed,
            "selection": "task/source round-robin, SHA256 order; reserve largest existing grid per bin; timing subset only",
            "excluded_above_largest_bin": excluded, "strata": strata, "requests": selected}


class RawFields:
    """Read original arrays, including for multi-source datasets with QA caches."""
    def __init__(self, dataset):
        self.dataset, self.handle = dataset, None
        self.multisource = hasattr(dataset, "readers")

    def read(self, ref):
        state = self.dataset.states[ref]
        if self.multisource:
            return self.dataset.readers.read(state)
        if self.handle is None:
            self.handle = h5py.File(self.dataset.hdf5_path, "r")
            for key, shape in self.dataset.metadata["source"]["field_layouts"].items():
                if list(self.handle[key].shape) != shape:
                    raise ValueError("Raw source layout changed")
        r, c = state["top_left"]
        h, w = state["grid_shape"]
        raw = torch.from_numpy(self.handle[state["field"]][
            state["sample_index"], state["time_index"], r:r+h, c:c+w].copy())
        if tuple(raw.shape) != (h, w):
            raise ValueError("Raw source crop is out of bounds")
        return raw

    def normalize(self, raw):
        return normalize_field(raw, allow_singleton=self.multisource or
                               self.dataset.metadata.get("allow_singleton", False),
                               allow_constant=self.multisource)

    def audit(self, request):
        row = request["record"]
        z, audit = self.normalize(request["raw"])
        if audit != self.dataset.states[row["state_ref"]]["normalization"]:
            raise ValueError(f"Raw source changed: {row['state_ref']}")
        if not torch.equal(z, self.dataset.field(row["state_ref"])):
            raise ValueError("QA field differs from replay of original source")
        validate_record(row, z)

    def close(self):
        if self.handle is not None:
            self.handle.close()


def load_checkpoint_snapshot(path):
    # Hash and load through the same descriptor: atomic last.pt replacements by
    # the trainer cannot switch the snapshot halfway through this operation.
    with Path(path).open("rb") as handle:
        before = os.fstat(handle.fileno())
        hasher = hashlib.sha256()
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
        digest = hasher.hexdigest()
        handle.seek(0)
        loaded = torch.load(handle, map_location="cpu", weights_only=True)
        after = os.fstat(handle.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Checkpoint was modified in place during loading; retry after saving finishes")
    keep = ("checkpoint_type", "identity", "sidecar", "step", "epoch", "cursor", "best")
    snapshot = {key: loaded[key] for key in keep if key in loaded}
    return snapshot, {"path": str(Path(path).resolve()), "sha256": digest,
                      "bytes": before.st_size, "step": snapshot.get("step"),
                      "epoch": snapshot.get("epoch"), "cursor": snapshot.get("cursor")}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_stats(device, initial=0):
    if device.type != "cuda":
        return {"peak_allocated_gib": None, "incremental_peak_gib": None, "peak_reserved_gib": None}
    peak = torch.cuda.max_memory_allocated(device)
    return {"peak_allocated_gib": peak / 2**30,
            "incremental_peak_gib": max(0, peak - initial) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}


@torch.inference_mode()
def timed_decode(llm, tokenizer, prompt, device, dtype, budget, fixed=False):
    """Same greedy/cache algorithm as trainer; fixed mode explicitly ignores EOS."""
    began = time.perf_counter()
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    stop_ids = set(trainer.generation_stop_ids(llm, tokenizer))
    output_ids, past, terminated, stop_id = [], None, False, None
    first_seconds = None
    with trainer.core.autocast_context(device, dtype):
        for index in range(budget):
            inputs = ids[:, -1:] if past is not None else ids
            kwargs = {"input_ids": inputs, "attention_mask": torch.ones_like(ids),
                      "use_cache": True, "return_dict": True}
            if past is not None:
                kwargs["past_key_values"] = past
            result = trainer.core.decoder_backbone(llm)(**kwargs)
            logits = llm.get_output_embeddings()(result.last_hidden_state[:, -1])
            token = int(logits.argmax(dim=-1).item())
            if index == 0:
                sync(device)
                first_seconds = time.perf_counter() - began
            if token in stop_ids and not fixed:
                terminated, stop_id = True, token
                break
            output_ids.append(token)
            ids = torch.cat((ids, ids.new_tensor([[token]])), dim=1)
            past = result.past_key_values
    sync(device)
    decode_done = time.perf_counter()
    prediction = tokenizer.decode(output_ids, skip_special_tokens=False)
    return {"prediction": prediction, "terminated": terminated, "stop_token_id": stop_id,
            "generated_tokens": len(output_ids) + int(terminated),
            "prefill_first_token_ms": first_seconds * 1000,
            "decode_after_first_ms": (decode_done - began - first_seconds) * 1000}


@torch.inference_mode()
def measure_request(request, raw_fields, llm, sidecar, tokenizer, device, dtype,
                    config, method, decode_mode, fixed_tokens):
    if sidecar is not None:
        sidecar.clear()
    sync(device)
    initial = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    record, raw = request["record"], request["raw"]
    budget = fixed_tokens if decode_mode == "fixed" else int(config["evaluation"]["max_new_tokens"])
    result = {"method": method, "decode_mode": decode_mode, "qa_id": record["qa_id"],
              "bin": request["bin"], "shape": record["grid_shape"], "task": record["task_type"],
              "cells": math.prod(record["grid_shape"]), "max_new_tokens": budget,
              "status": "running", "memory_tokens": 0, "prompt_tokens": None,
              "resident_allocated_gib": initial / 2**30 if device.type == "cuda" else None}
    try:
        z, _ = raw_fields.normalize(raw)
        normalized = time.perf_counter()
        if method == "serialized":
            prompt = trainer.native_chat_ids(tokenizer, serialized_messages(record, z), generation_prompt=True)
        else:
            prompt = trainer.encode_prompt(record, tokenizer, int(config["training"]["max_prompt_tokens"]))
        prepared = time.perf_counter()
        result.update(prompt_tokens=len(prompt), preprocessing_ms=(normalized - started) * 1000,
                      prompt_preparation_ms=(prepared - normalized) * 1000)
        context = getattr(llm.config, "max_position_embeddings", None)
        if context is None:
            raise ValueError("Model must declare a context limit")
        if len(prompt) + budget > context:
            result.update(status="context_exceeded", context_limit=context)
        else:
            if method == "interface":
                with trainer.core.autocast_context(device, dtype):
                    sidecar.bind(z[None, None].to(device), mode="correct")
                bh, bw = config["memory"].get("block_shape", [1, 1])
                h, w = record["grid_shape"]
                result["memory_tokens"] = math.ceil(h / bh) * math.ceil(w / bw)
            sync(device)
            encoded = time.perf_counter()
            result["field_to_memory_ms"] = (encoded - prepared) * 1000
            decoded = timed_decode(llm, tokenizer, prompt, device, dtype, budget, fixed=decode_mode == "fixed")
            result.update(decoded)
            result["ttft_ms"] = (encoded - started) * 1000 + decoded["prefill_first_token_ms"]
            result["status"] = ("fixed_length" if decode_mode == "fixed" else
                                "complete" if decoded["terminated"] else "length_limit")
    except torch.cuda.OutOfMemoryError:
        result["status"] = "oom"
    finally:
        sync(device)
        result["request_ms"] = (time.perf_counter() - started) * 1000
        result.update(memory_stats(device, initial))
        if sidecar is not None:
            sidecar.clear()
    if result["status"] == "oom":
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def summarize(rows, strata):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["decode_mode"], row["bin"])].append(row)
    summaries = []
    for (method, mode, size), items in sorted(groups.items()):
        executed = [r for r in items if r["status"] in ("complete", "fixed_length", "length_limit")]
        completed = [r for r in items if r["status"] in ("complete", "fixed_length")]
        seconds = sum(r["request_ms"] for r in items) / 1000
        row = {"method": method, "decode_mode": mode, "bin": size,
               "attempts": len(items), "unique_questions": len({r["qa_id"] for r in items}),
               "statuses": dict(Counter(r["status"] for r in items)),
               "completed_requests_per_active_second": len(completed) / seconds if seconds else None,
               "serial_generated_tokens_per_active_second": sum(r.get("generated_tokens", 0) for r in items) / seconds if seconds else None,
               "completed_answer_p50_ms": percentile([r["request_ms"] for r in completed], .5),
               "completed_answer_p95_ms": percentile([r["request_ms"] for r in completed], .95)}
        for key in ("request_ms", "ttft_ms", "preprocessing_ms", "prompt_preparation_ms", "field_to_memory_ms",
                    "prefill_first_token_ms", "decode_after_first_ms", "generated_tokens", "prompt_tokens", "memory_tokens", "cells"):
            values = [r[key] for r in executed if r.get(key) is not None]
            row[key + "_mean"] = sum(values) / len(values) if values else None
            if key in ("request_ms", "ttft_ms"):
                row[key + "_p50"] = percentile(values, .5)
                row[key + "_p95"] = percentile(values, .95)
        for key in ("peak_allocated_gib", "incremental_peak_gib", "peak_reserved_gib"):
            values = [r[key] for r in executed if r.get(key) is not None]
            row[key] = max(values) if values else None
        failed_peaks = [r["peak_allocated_gib"] for r in items if r["status"] == "oom"
                        and r.get("peak_allocated_gib") is not None]
        row["oom_peak_allocated_gib"] = max(failed_peaks) if failed_peaks else None
        summaries.append(row)
    paired = []
    index = {(r["method"], r["decode_mode"], r["bin"], r["qa_id"], r["repeat"]): r for r in rows}
    for mode in sorted({r["decode_mode"] for r in rows}):
        for stratum in strata:
            size = stratum["bin"]
            pairs = []
            for r in rows:
                if r["method"] != "interface" or r["decode_mode"] != mode or r["bin"] != size:
                    continue
                b = index.get(("serialized", mode, size, r["qa_id"], r["repeat"]))
                if b and r["status"] in ("complete", "fixed_length") and b["status"] in ("complete", "fixed_length"):
                    pairs.append((b, r))
            paired.append({"decode_mode": mode, "bin": size, "paired_completed_requests": len(pairs),
                           "serialized_over_interface_time": (sum(b["request_ms"] for b, r in pairs) /
                               sum(r["request_ms"] for b, r in pairs)) if pairs else None,
                           "serialized_mean_generated_tokens": sum(b["generated_tokens"] for b, r in pairs) / len(pairs) if pairs else None,
                           "interface_mean_generated_tokens": sum(r["generated_tokens"] for b, r in pairs) / len(pairs) if pairs else None})
    return {"groups": summaries, "paired_comparisons": paired, "strata": strata,
            "throughput_definition": "batch=1, sequential, active request wall time including failed attempts; excludes setup/warmup/logging",
            "latency_definition": "request/TTFT summaries include length-limit requests; completed_answer uses complete/fixed_length only"}


def save_summary(output, rows, plan):
    summary = summarize(rows, plan["strata"])
    write_json(output / "summary.json", summary)
    if summary["groups"]:
        with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary["groups"][0]))
            writer.writeheader()
            for row in summary["groups"]:
                writer.writerow({k: json.dumps(v) if isinstance(v, dict) else v for k, v in row.items()})
    return summary


def command_output(command):
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def save_plot(output, summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; trend data is saved in summary.csv", flush=True)
        return
    modes = sorted({g["decode_mode"] for g in summary["groups"]})
    if not modes:
        return
    fig, axes = plt.subplots(len(modes), 3, figsize=(14, 4 * len(modes)), squeeze=False)
    columns = [("completed_answer_p50_ms", "Completed request latency (median, ms)"),
               ("ttft_ms_p50", "First token latency (median, ms)"),
               ("peak_allocated_gib", "Peak allocated GPU memory (GiB)")]
    for i, mode in enumerate(modes):
        for j, (metric, label) in enumerate(columns):
            ax = axes[i, j]
            for method in ("serialized", "interface"):
                groups = sorted((g for g in summary["groups"] if g["method"] == method and
                    g["decode_mode"] == mode and g.get(metric) is not None and g.get("cells_mean") is not None),
                    key=lambda g: g["cells_mean"])
                if groups:
                    ax.plot([g["cells_mean"] for g in groups], [g[metric] for g in groups], "o-", label=method)
            ax.set(xscale="log", xlabel="Mean input cells in executed requests", ylabel=label,
                   title=f"{mode} decoding")
            ax.grid(alpha=.25)
            if ax.lines:
                ax.legend()
    fig.suptitle("Single-GPU batch=1; missing settings and failure counts: summary.csv")
    fig.tight_layout()
    fig.savefig(output / "size_trends.png", dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Existing run containing resolved_config.json")
    parser.add_argument("--checkpoint-name", default="best.pt", choices=("best.pt", "last.pt"))
    parser.add_argument("--training-status", required=True, choices=("completed", "in_progress"),
                        help="Status of the source training run, independent of checkpoint step")
    parser.add_argument("--output-dir", required=True, help="New directory outside the source run")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--model-dir")
    parser.add_argument("--hdf5-path")
    parser.add_argument("--qa-dir")
    parser.add_argument("--area-edges", type=int, nargs="+", default=[64, 256, 1024, 4096, 16384])
    parser.add_argument("--per-bin", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2, help="Per method/decode mode/size bin")
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--methods", nargs="+", choices=("serialized", "interface"), default=["serialized", "interface"])
    parser.add_argument("--decode-modes", nargs="+", choices=("natural", "fixed"), default=["natural", "fixed"])
    parser.add_argument("--fixed-tokens", type=int, default=32)
    parser.add_argument("--audit-only", action="store_true", help="Audit checkpoint/data and write selection; no model/GPU needed")
    cli = parser.parse_args()
    if min(cli.repeats, cli.per_bin, cli.fixed_tokens) <= 0 or cli.warmup < 1:
        parser.error("Repeats/per-bin/fixed-tokens must be positive and warmup >= 1")
    if len(set(cli.methods)) != len(cli.methods) or len(set(cli.decode_modes)) != len(cli.decode_modes):
        parser.error("Methods and decode modes must be unique")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Run python with one visible GPU, not torchrun")
    return cli


def main():
    cli = parse_args()
    run, output = resolve_path(cli.run_dir), resolve_path(cli.output_dir)
    if output == run or run in output.parents:
        raise ValueError("Benchmark output must be outside the training run")
    if output.exists():
        raise FileExistsError("Use a new benchmark output directory")
    if not cli.audit_only and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError("Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES; use --audit-only on CPU")
    config_path = run / "resolved_config.json"
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    config = load_config(config_path, saved.get("experiment_profile", "full"))
    trainer.validate_config(config)
    if not config["evaluation"].get("use_cache", True):
        raise ValueError("This benchmark targets the current cached autoregressive inference recipe")
    snapshot, checkpoint_info = load_checkpoint_snapshot(run / cli.checkpoint_name)
    checkpoint_info["source_training_status"] = cli.training_status
    qa_dir = resolve_path(cli.qa_dir or config["data"]["qa_dir"])
    dataset = PointReadoutDataset(qa_dir, source_path(config, cli.hdf5_path), cli.split, cache_size=0)
    raw_fields = RawFields(dataset)
    try:
        if snapshot.get("checkpoint_type") != trainer.CHECKPOINT_TYPE or snapshot.get("identity", {}).get("dataset") != dataset.identity:
            raise ValueError("Checkpoint type or QA identity mismatch")
        plan = select_requests(dataset, cli.area_edges, cli.per_bin, cli.seed)
        plan["split"] = cli.split
        rows_by_id = {r["qa_id"]: r for r in dataset.records}
        loaded_raw, requests, source_reads = {}, [], []
        for chosen in plan["requests"]:
            ref = chosen["state_ref"]
            if ref not in loaded_raw:
                began = time.perf_counter()
                loaded_raw[ref] = raw_fields.read(ref)
                source_reads.append({"state_ref": ref, "read_ms": (time.perf_counter() - began) * 1000})
            request = {**chosen, "record": rows_by_id[chosen["qa_id"]], "raw": loaded_raw[ref]}
            raw_fields.audit(request)
            requests.append(request)
        output.mkdir(parents=True)
        write_json(output / "selection.json", plan)
        write_json(output / "source_read_audit.json", {"raw_states": source_reads, "source_to_QA_parity": "passed for every selected request",
                   "timing": "one-off HDF5 reads outside request latency; filesystem cache uncontrolled"})
        contract = {"schema": "readout_efficiency_v1", "checkpoint": checkpoint_info, "config": config,
                    "dataset": dataset.identity, "qa_dir": str(qa_dir), "split": cli.split,
                    "selection_sha256": json_hash(plan), "batch_size": 1, "cli": vars(cli),
                    "code_commit": command_output(["git", "rev-parse", "HEAD"]),
                    "git_status": command_output(["git", "status", "--short"]),
                    "script_sha256": trainer.core.sha256_file(Path(__file__)),
                    "timer": "synchronized perf_counter wall time; raw CPU array through normalization, prompt preparation, transfer, encoder/adapter/memory, full greedy generation and detokenization",
                    "exclusions": ["source disk IO (separately recorded)", "model/checkpoint loading", "warmup", "source/QA audit", "logging"],
                    "field_reuse": False, "text_KV_cache": "within each answer only",
                    "normalization": "exact dataset normalize_field including its integrity hashes",
                    "field_memory_timing": "encoder + spatial adapter + scalar branch; lazy cross-attention K/V projections are included in prefill",
                    "fixed_decode": "auxiliary workload: ignore EOS and emit exactly fixed-tokens, no answer/accuracy interpretation",
                    "accuracy_evaluation": False, "python": platform.python_version(), "torch": torch.__version__,
                    "cuda_runtime": torch.version.cuda, "platform": platform.platform()}
        write_json(output / "contract.json", contract)
        print(f"selection={len(requests)} requests bins={[(s['bin'], s['selected_questions']) for s in plan['strata']]}", flush=True)
        if cli.audit_only:
            print(f"Audit complete. No GPU timing performed. output={output}", flush=True)
            return
        device = torch.device("cuda:0")
        args = trainer.model_args(config, cli.model_dir, checkpointing=False)
        trainer.core.apply_runtime_environment(args)
        trainer.core.seed_everything(args.seed)
        if args.torch_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("Selected GPU does not support BF16")
        tokenizer = trainer.core.load_tokenizer(args)
        llm, dtype = trainer.core.load_llm_with_bounded_host_memory(args, device)
        llm.requires_grad_(False)
        trainer.core.set_frozen_llm_execution_mode(llm, checkpoint_training=False)
        identity = trainer.run_identity(config, dataset, llm, tokenizer)
        identity["model_asset"] = trainer.model_asset_identity(args.model_name_or_path, llm)
        trainer.validate_resume(snapshot, identity)
        import transformers
        contract.update(model_identity=identity, transformers=transformers.__version__, model_dtype=str(dtype),
                        gpu=torch.cuda.get_device_name(device), gpu_total_gib=torch.cuda.get_device_properties(device).total_memory / 2**30,
                        attention_backend=getattr(llm.config, "_attn_implementation", None),
                        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                        nvidia_smi_before=command_output(["nvidia-smi"]), status="running")
        write_json(output / "contract.json", contract)
        sidecar, results = None, []
        with (output / "requests.jsonl").open("x", encoding="utf-8") as handle, (output / "warmup.jsonl").open("x", encoding="utf-8") as warm_log:
            # An untouched baseline is measured first: no disabled-but-resident sidecar.
            for method in (m for m in ("serialized", "interface") if m in cli.methods):
                if method == "interface":
                    encoder, spatial, _, _ = trainer.core.build_scratch_memory_components(args, (1, args.patch_size, args.patch_size))
                    sidecar, _ = trainer.core.build_sidecar(llm, spatial, args, device, encoder)
                    sidecar.load_state_dict(snapshot.pop("sidecar"), strict=True)
                    sidecar.requires_grad_(False)
                    sidecar.eval()
                gc.collect()
                torch.cuda.empty_cache()
                for mode in cli.decode_modes:
                    for stratum in plan["strata"]:
                        active = [r for r in requests if r["bin"] == stratum["bin"]]
                        if not active:
                            continue
                        warm_successes = 0
                        warm_index = 0
                        for warm in range(len(active) + cli.warmup):
                            result = measure_request(active[warm_index], raw_fields, llm, sidecar,
                                tokenizer, device, dtype, config, method, mode, cli.fixed_tokens)
                            warm_log.write(json.dumps({**result, "warmup": warm}, allow_nan=False) + "\n")
                            warm_log.flush()
                            if result["status"] in ("complete", "fixed_length", "length_limit"):
                                warm_successes += 1
                            else:
                                warm_index += 1
                            if warm_successes >= cli.warmup:
                                break
                            if warm_index >= len(active):
                                break
                        for repeat in range(cli.repeats):
                            ordered = list(active)
                            random.Random(cli.seed + repeat).shuffle(ordered)
                            for request in ordered:
                                result = measure_request(request, raw_fields, llm, sidecar, tokenizer,
                                    device, dtype, config, method, mode, cli.fixed_tokens)
                                result["repeat"] = repeat
                                results.append(result)
                                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                                handle.flush()
                            print(f"method={method} mode={mode} bin={stratum['bin']} repeat={repeat + 1}/{cli.repeats} "
                                  f"statuses={dict(Counter(r['status'] for r in results if r['method']==method and r['decode_mode']==mode and r['bin']==stratum['bin']))}", flush=True)
                        save_summary(output, results, plan)
        summary = save_summary(output, results, plan)
        save_plot(output, summary)
        contract.update(status="complete", nvidia_smi_after=command_output(["nvidia-smi"]))
        write_json(output / "contract.json", contract)
        print(f"Complete: {output / 'summary.csv'}", flush=True)
    finally:
        raw_fields.close()
        dataset.close()


if __name__ == "__main__":
    main()
