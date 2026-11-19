"""Shared execution utilities for training, evaluation and benchmarks."""
from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import torch.distributed as dist
from transformers.utils import logging as transformers_logging
from tensor_compression.utils.pipeline_config import first_nested

def _config_value(config: Mapping[str, Any], paths: Sequence[str], default: Any = None) -> Any:
    value = first_nested(config, list(paths))
    return default if value is None else value

def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def distributed_rank() -> int:
    return int(dist.get_rank()) if distributed_is_initialized() else 0

def distributed_world_size() -> int:
    return int(dist.get_world_size()) if distributed_is_initialized() else 1

def is_main_process() -> bool:
    return distributed_rank() == 0

def distributed_barrier() -> None:
    if distributed_is_initialized():
        dist.barrier()

def local_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

def apply_runtime_environment(args: argparse.Namespace) -> None:
    if args.hf_home:
        os.environ.setdefault("HF_HOME", str(args.hf_home))
    if args.cache_dir:
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.cache_dir) / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(args.cache_dir))
    if not bool(args.console_progress):
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        transformers_logging.disable_progress_bar()

def qa_path(qa_dir: str | Path, split: str) -> Path:
    path = Path(qa_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"QA split file not found: {path}")
    return path

def parse_csv(raw: str | Sequence[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        return [str(part).strip() for part in raw if str(part).strip()]
    return [part.strip() for part in str(raw).split(",") if part.strip()]

def redacted_args(args: argparse.Namespace) -> dict[str, Any]:
    payload = dict(vars(args))
    if payload.get("wandb_api_key"):
        payload["wandb_api_key"] = "***REDACTED***"
    return payload

from torch.utils.data import Dataset, Sampler


class ExactDistributedEvalSampler(Sampler[int]):
    """Shard evaluation records without padding or repeating any example."""

    def __init__(self, dataset: Dataset, rank: int, num_replicas: int) -> None:
        self.dataset = dataset
        self.rank = int(rank)
        self.num_replicas = max(1, int(num_replicas))
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}.")

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        if self.rank >= len(self.dataset):
            return 0
        return ((len(self.dataset) - 1 - self.rank) // self.num_replicas) + 1
