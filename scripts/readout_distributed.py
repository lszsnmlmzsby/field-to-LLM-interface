"""Explicit data parallelism for the hook-based frozen-LLM readout trainer.

Gradients are summed once per optimizer update, after every local backward has
finished. This supports empty ranks, uneven batches and local OOM retries without
DDP hooks entering collectives halfway through a failed backward.
"""
from __future__ import annotations

from datetime import timedelta
import os
import random

import torch
import torch.distributed as dist


class ParallelContext:
    def __init__(self, device, *, owned=False):
        self.device = torch.device(device)
        self.owned = owned
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    @classmethod
    def initialize(cls, requested="auto", timeout_minutes=360):
        if timeout_minutes <= 0:
            raise ValueError("Distributed timeout must be positive")
        world = int(os.environ.get("WORLD_SIZE", "1"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
        if world > 1 and device.type == "cuda":
            local_rank = int(os.environ["LOCAL_RANK"])
            if requested not in ("auto", "cuda"):
                raise ValueError("With torchrun use --device auto/cuda, not a fixed CUDA index")
            if not 0 <= local_rank < torch.cuda.device_count():
                raise ValueError("LOCAL_RANK exceeds the number of visible CUDA devices")
            device = torch.device("cuda", local_rank)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        owned = world > 1 and not dist.is_initialized()
        if owned:
            dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo",
                                    timeout=timedelta(minutes=timeout_minutes))
        return cls(device, owned=owned)

    @property
    def primary(self):
        return self.rank == 0

    def gather(self, value):
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def main(self, operation):
        """One writer; propagate its exception instead of stranding other ranks."""
        if self.world_size == 1:
            return operation()
        envelope, error = [None], None
        if self.primary:
            try:
                envelope[0] = (True, operation())
            except Exception as exc:
                error = exc
                envelope[0] = (False, f"{type(exc).__name__}: {exc}")
        dist.broadcast_object_list(envelope, src=0)
        ok, value = envelope[0]
        if not ok:
            if error is not None:
                raise error
            raise RuntimeError(f"Rank 0 failed: {value}")
        return value

    def check_errors(self, error):
        messages = self.gather(error)
        failures = [f"rank {i}: {value}" for i, value in enumerate(messages) if value]
        if failures:
            raise RuntimeError("Distributed operation failed: " + "; ".join(failures))

    def any(self, flag):
        value = torch.tensor(int(flag), device=self.device)
        if self.world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return bool(value.item())

    def sums(self, values):
        value = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(value)
        return value.cpu().tolist()

    @torch.no_grad()
    def synchronize_parameters(self, module):
        if self.world_size > 1:
            for parameter in module.parameters():
                if parameter.requires_grad:
                    dist.broadcast(parameter, src=0)
            for buffer in module.buffers():
                dist.broadcast(buffer, src=0)

    @torch.no_grad()
    def sum_gradients(self, module):
        """Input losses already use the GLOBAL question-count denominator.

        Preserve globally unused gradients as None (AdamW must not update them).
        Bucket the reductions to avoid one collective per small parameter.
        """
        if self.world_size == 1:
            return
        parameters = [p for p in module.parameters() if p.requires_grad]
        used = torch.tensor([p.grad is not None for p in parameters], dtype=torch.int32, device=self.device)
        dist.all_reduce(used, op=dist.ReduceOp.MAX)
        buckets, bucket, size, dtype = [], [], 0, None
        for parameter, present in zip(parameters, used.cpu().tolist()):
            if not present:
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            if bucket and (size + parameter.numel() * parameter.element_size() > 16 * 2**20
                           or parameter.dtype != dtype):
                buckets.append(bucket)
                bucket, size = [], 0
            bucket.append(parameter)
            size += parameter.numel() * parameter.element_size()
            dtype = parameter.dtype
        if bucket:
            buckets.append(bucket)
        for bucket in buckets:
            flat = torch.cat([p.grad.reshape(-1) for p in bucket])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            offset = 0
            for parameter in bucket:
                count = parameter.numel()
                parameter.grad.copy_(flat[offset:offset + count].view_as(parameter))
                offset += count

    def close(self):
        if self.owned and dist.is_initialized():
            dist.destroy_process_group()


def partition_batch(indices, rank, world_size):
    """Balanced contiguous pieces, with no padding, repeats or dropped samples."""
    if not 0 <= rank < world_size:
        raise ValueError("Invalid rank/world size")
    quotient, remainder = divmod(len(indices), world_size)
    start = rank * quotient + min(rank, remainder)
    return indices[start:start + quotient + int(rank < remainder)]


def capture_rng(device):
    return {"python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if device.type == "cuda" and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"], device)


class RecordView:
    """A lightweight view; don't copy millions of QA dictionaries for preflight."""
    def __init__(self, records, indices):
        self.records, self.indices = records, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.records[self.indices[index]]


def evaluation_indices(records, world_size):
    """Keep all questions of a field on one rank, balanced by question count."""
    groups = {}
    for index, row in enumerate(records):
        groups.setdefault(row["state_ref"], []).append(index)
    shards = [[] for _ in range(world_size)]
    for indices in groups.values():
        rank = min(range(world_size), key=lambda r: len(shards[r]))
        shards[rank].extend(indices)
    return [sorted(indices) for indices in shards]


def merge_token_reports(reports):
    merged = dict(reports[0])
    for key in ("questions", "prompt_overflow_count", "target_overflow_count", "overflow_questions"):
        merged[key] = sum(report[key] for report in reports)
    for length, record in (("max_prompt_tokens_observed", "max_prompt_record"),
                           ("max_target_tokens_observed", "max_target_record")):
        largest = max(reports, key=lambda report: report[length])
        merged[length], merged[record] = largest[length], largest[record]
    merged["overflow_examples"] = [example for report in reports for example in report["overflow_examples"]][:8]
    merged["status"] = "overflow" if merged["overflow_questions"] else "ok"
    return merged
