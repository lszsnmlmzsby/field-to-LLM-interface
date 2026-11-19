"""Execution environment shared by model loading and training."""
import argparse
import os
from pathlib import Path
import torch.distributed as dist
from transformers.utils import logging as transformers_logging


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

def apply_runtime_environment(args: argparse.Namespace) -> None:
    if args.hf_home:
        os.environ.setdefault("HF_HOME", str(args.hf_home))
    if args.cache_dir:
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.cache_dir) / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(args.cache_dir))
    if not bool(args.console_progress):
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        transformers_logging.disable_progress_bar()
