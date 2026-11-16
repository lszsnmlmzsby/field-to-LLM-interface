"""Small training-only helpers for immutable input caches and sampled timing."""
from __future__ import annotations

from contextlib import contextmanager
import time

import torch


class PreparedTokenLookup:
    """Dispatch compact train rows to mmap tokens; retain the legacy eval cache."""

    def __init__(self, dataset, fallback):
        self.dataset, self.fallback = dataset, fallback
        self.reads = 0

    def get(self, row, tokenizer, training):
        index = row.get("_prepared_index")
        if index is None:
            return self.fallback.get(row, tokenizer, training)
        self.reads += 1
        return self.dataset.token_pair(index)

    def statistics(self):
        return {"prepared_token_reads": self.reads, "legacy_cache": self.fallback.statistics()}


class StageTimer:
    """CPU durations always; CUDA events only in explicitly sampled updates.

    CUDA durations may overlap CPU work and must not be added to CPU durations.
    The normal path adds no CUDA synchronizations for measurement.
    """

    def __init__(self, device, *, cuda_events=False):
        self.device = torch.device(device)
        self.cuda_events = bool(cuda_events and self.device.type == "cuda")
        self.seconds, self.events = {}, []

    @contextmanager
    def cpu(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - started

    @contextmanager
    def gpu(self, name):
        if not self.cuda_events:
            yield
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.append((name, start, end))

    def summary(self):
        cuda_ms = {}
        if self.events:
            self.events[-1][2].synchronize()
            for name, start, end in self.events:
                cuda_ms[name] = cuda_ms.get(name, 0.0) + start.elapsed_time(end)
        return {"cpu_seconds": dict(self.seconds), "cuda_milliseconds": cuda_ms,
                "cuda_events_enabled": self.cuda_events,
                "timing_note": "CPU and CUDA durations can overlap; CUDA events measure stream elapsed time including scheduling gaps, not kernel utilization"}


def local_update_indices(batches, parallel, token_lengths, balance_by_length):
    from scripts.readout_distributed import partition_batch, length_balanced_partitions

    parts = [length_balanced_partitions(batch, token_lengths, parallel.world_size)[parallel.rank]
             if balance_by_length else partition_batch(batch, parallel.rank, parallel.world_size)
             for batch in batches]
    return [index for part in parts for index in part]


def queue_following_updates(loader, *, step, maximum, batches, cursor, accumulation,
                            parallel, token_lengths, balance_by_length):
    """Read ahead only within the current epoch; never advance the real cursor."""
    for ahead in range(1, loader.prefetch_updates + 1):
        if step + ahead >= maximum:
            break
        begin = cursor + (ahead - 1) * accumulation
        selected = batches[begin:begin + accumulation]
        # Do not speculate across an epoch boundary. The training loop decides
        # whether a partial accumulated update carries into the next epoch.
        if len(selected) != accumulation:
            break
        indices = local_update_indices(selected, parallel, token_lengths, balance_by_length)
        if not loader.prefetch(step + ahead, indices):
            break
