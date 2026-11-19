"""Shared components used by the current multi-source field workflow."""
from __future__ import annotations
from tensor_compression.workflows.runtime import (
    apply_runtime_environment,
    distributed_barrier,
    distributed_is_initialized,
    distributed_rank,
    distributed_world_size,
    is_main_process,
)
import argparse
import gc
import json
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import torch
import torch.distributed as dist
from torch import nn

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import logging as transformers_logging
except ImportError as exc:  # pragma: no cover - exercised only in missing-dependency envs
    raise ImportError(
        "Qwen model loading requires transformers. "
        "Install it with: pip install transformers accelerate safetensors"
    ) from exc

def read_host_memory_snapshot(proc_root: str | Path = "/proc") -> dict[str, float]:
    """Read Linux host memory without adding a process-monitoring dependency."""

    root = Path(proc_root)
    meminfo_path = root / "meminfo"
    status_path = root / "self" / "status"
    if not meminfo_path.exists():
        return {}

    meminfo: dict[str, int] = {}
    with meminfo_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            name, separator, raw_value = line.partition(":")
            if not separator:
                continue
            parts = raw_value.strip().split()
            if not parts:
                continue
            try:
                value_kib = int(parts[0])
            except ValueError:
                continue
            meminfo[name] = value_kib

    process_rss_kib = 0
    if status_path.exists():
        with status_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.partition(":")[2].strip().split()
                    if parts:
                        process_rss_kib = int(parts[0])
                    break

    divisor = 1024**2
    return {
        "total_gib": float(meminfo.get("MemTotal", 0)) / divisor,
        "available_gib": float(meminfo.get("MemAvailable", meminfo.get("MemFree", 0))) / divisor,
        "process_rss_gib": float(process_rss_kib) / divisor,
    }

def gather_host_memory(device: torch.device) -> list[dict[str, float]]:
    snapshot = read_host_memory_snapshot()
    if not snapshot:
        return []
    local = torch.tensor(
        [snapshot["total_gib"], snapshot["available_gib"], snapshot["process_rss_gib"]],
        dtype=torch.float64,
        device=device,
    )
    if distributed_is_initialized():
        gathered = [torch.empty_like(local) for _ in range(distributed_world_size())]
        dist.all_gather(gathered, local)
    else:
        gathered = [local]
    return [
        {
            "rank": float(rank),
            "total_gib": float(values[0].item()),
            "available_gib": float(values[1].item()),
            "process_rss_gib": float(values[2].item()),
        }
        for rank, values in enumerate(gathered)
    ]

def enforce_host_memory_floor(
    device: torch.device,
    minimum_available_gib: float,
    stage: str,
) -> list[dict[str, float]]:
    reports = gather_host_memory(device)
    available = [item["available_gib"] for item in reports if item["available_gib"] > 0]
    if available and float(minimum_available_gib) > 0 and min(available) < float(minimum_available_gib):
        raise RuntimeError(
            f"Host memory safety guard stopped {stage}: MemAvailable={min(available):.2f} GiB is below "
            f"the configured floor {float(minimum_available_gib):.2f} GiB. Reduce ranks/workers/cache or use "
            "a host with more RAM."
        )
    return reports

def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dict(payload), ensure_ascii=True, allow_nan=False)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(serialized + "\n")

def atomic_torch_save(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_model_dtype(raw: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if raw == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[raw]

def set_frozen_llm_execution_mode(model: nn.Module, checkpoint_training: bool) -> None:
    checkpointing_active = bool(getattr(model, "is_gradient_checkpointing", False))
    # Disable stochastic behavior recursively first. This also covers modules
    # that call functional dropout using their own ``training`` flag.
    model.eval()
    if checkpoint_training and checkpointing_active:
        # Decoder checkpointing is gated on the decoder backbone's training flag.
        # Flip only that flag so all attention/MLP children remain deterministic.
        decoder = getattr(model, "model", model)
        if not isinstance(decoder, nn.Module):
            raise TypeError("The frozen causal LLM exposes a non-module decoder backbone.")
        decoder.training = True

def load_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("Tokenizer has no pad/eos/unk token; cannot build padded batches.")
    return tokenizer

def load_llm(args: argparse.Namespace, device: torch.device):
    model_dtype = resolve_model_dtype(str(args.torch_dtype), device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        dtype=model_dtype,
        trust_remote_code=bool(args.trust_remote_code),
        low_cpu_mem_usage=bool(getattr(args, "low_cpu_mem_usage", True)),
    )
    model.to(device)
    gc.collect()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if bool(args.llm_gradient_checkpointing):
        enable_checkpointing = getattr(model, "gradient_checkpointing_enable", None)
        if not callable(enable_checkpointing):
            raise ValueError("The selected causal LLM does not support gradient checkpointing.")
        try:
            enable_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError as exc:
            raise RuntimeError(
                "Stage-2 soft-prefix training requires non-reentrant gradient checkpointing. "
                "Upgrade transformers or explicitly disable llm_training.llm_gradient_checkpointing "
                "for a memory-bounded smoke run."
            ) from exc
        if not bool(getattr(model, "is_gradient_checkpointing", False)):
            raise RuntimeError("The causal LLM did not report active gradient checkpointing.")
        # Transformers activates decoder checkpointing only when its decoder backbone is in training mode.
        set_frozen_llm_execution_mode(model, checkpoint_training=True)
    else:
        set_frozen_llm_execution_mode(model, checkpoint_training=False)
    return model, model_dtype

def load_llm_with_bounded_host_memory(args: argparse.Namespace, device: torch.device):
    """Construct DDP replicas serially so CPU checkpoint copies never peak together."""

    serialize = bool(getattr(args, "serialize_llm_loading", True))
    minimum_available_gib = float(getattr(args, "min_host_memory_available_gib", 0.0))
    if not distributed_is_initialized() or not serialize:
        enforce_host_memory_floor(device, minimum_available_gib, "LLM loading")
        return load_llm(args, device)

    llm = None
    model_dtype = None
    for load_rank in range(distributed_world_size()):
        reports = enforce_host_memory_floor(
            device,
            minimum_available_gib,
            f"LLM loading for rank {load_rank}",
        )
        if is_main_process() and reports:
            print(
                f"startup=llm_load_rank rank={load_rank}/{distributed_world_size() - 1} "
                f"host_available={min(item['available_gib'] for item in reports):.2f}GiB "
                f"rank_rss="
                + ",".join(
                    f"r{int(item['rank'])}:{item['process_rss_gib']:.2f}GiB"
                    for item in reports
                )
            )

        local_error: BaseException | None = None
        if distributed_rank() == load_rank:
            try:
                llm, model_dtype = load_llm(args, device)
            except BaseException as exc:  # propagate a coherent failure to ranks waiting below
                local_error = exc
        error_payload = [
            None
            if local_error is None
            else f"{type(local_error).__name__}: {str(local_error)[:2000]}"
        ]
        dist.broadcast_object_list(error_payload, src=load_rank)
        if error_payload[0] is not None:
            raise RuntimeError(
                f"Distributed LLM loading failed on rank {load_rank}: {error_payload[0]}"
            ) from local_error
        distributed_barrier()

    if llm is None or model_dtype is None:
        raise RuntimeError("The local rank did not construct its frozen LLM replica.")
    return llm, model_dtype
