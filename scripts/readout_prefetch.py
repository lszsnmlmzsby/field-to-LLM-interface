"""Bounded CPU prefetch for immutable, memory-mapped training arrays.

Workers reopen only the prepared cache. They do not receive QA dictionaries,
tokenizers, models, or CUDA tensors. The byte budget is a conservative payload
forecast, not a bound on Python/NumPy interpreter RSS or OS page-cache memory.
The caller owns update order and explicitly releases the current update after
backward and its retries succeed, so retries reuse the same CPU values and tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import queue
import sys
import time
import traceback
from typing import Any, Hashable, Sequence

import numpy as np


class PrefetchWorkerError(RuntimeError):
    """A cache reader failed; the loader has been closed and cannot be reused."""


@dataclass(frozen=True)
class PreparedSample:
    index: int
    field: np.ndarray
    prompt: np.ndarray
    answer: np.ndarray


@dataclass(frozen=True)
class PreparedUpdate:
    key: Hashable
    indices: tuple[int, ...]
    items: tuple[PreparedSample, ...]
    forecast_bytes: int
    by_index: dict[int, PreparedSample]

    def item(self, index: int) -> PreparedSample:
        return self.by_index[int(index)]

    def token_pair(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        sample = self.item(index)
        return sample.prompt, sample.answer


def _open_cache(cache_dir: str):
    # Keep the worker's import and initialization independent of the trainer.
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from tensor_compression.downstream.readout_training_cache import PreparedTrainingDataset

    return PreparedTrainingDataset(cache_dir, validate=False)


def _read_samples(dataset, indices: tuple[int, ...]):
    # Repeated indices retain their multiplicity/order in PreparedUpdate.items,
    # but only one set of arrays is transferred and retained per unique index.
    result = []
    for index in dict.fromkeys(indices):
        prompt, answer = dataset.token_pair(index)
        field = dataset.field_array(index)
        result.append((index, np.asarray(field, dtype=np.float32),
                       np.array(prompt, dtype=np.uint32, copy=True),
                       np.array(answer, dtype=np.uint32, copy=True)))
    return result


def _reader_worker(cache_dir: str, tasks, results):
    dataset = None
    try:
        try:
            dataset = _open_cache(cache_dir)
        except Exception:
            results.put(("initialization_error", None, traceback.format_exc()))
            return
        while True:
            task = tasks.get()
            if task is None:
                return
            key, indices = task
            try:
                samples = _read_samples(dataset, indices)
            except Exception:
                results.put(("error", key, traceback.format_exc()))
                return
            results.put(("result", key, samples))
            del samples
    finally:
        if dataset is not None:
            dataset.close()


class PreparedBatchLoader:
    """Prefetch at most N future updates plus one explicitly retained update.

    ``prefetch(key, indices)`` returns False when the count or byte budget is
    occupied. It raises for an update that cannot fit even in an empty loader.
    ``get(key)`` consumes that submitted key without reordering it, and repeated
    gets of the current key return the same object for OOM retry. Call
    ``release(key)`` before getting a different update. ``load(indices)`` is a
    convenience submission/get for callers that do not schedule ahead.
    """

    def __init__(self, cache_dir, *, workers=2, prefetch_updates=2,
                 max_bytes=256 * 2**20, pin_memory=False, timeout_seconds=120):
        if type(workers) is not int or not 0 <= workers <= 4:
            raise ValueError("workers must be an integer from 0 through 4")
        if type(prefetch_updates) is not int or prefetch_updates < 1:
            raise ValueError("prefetch_updates must be a positive integer")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if not isinstance(timeout_seconds, (int, float)) or not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if pin_memory:
            raise ValueError("Pin assembled micro batches in the main trainer; whole-update pinning is disabled")
        self.cache_dir = str(Path(cache_dir).resolve())
        self.workers = workers
        self.prefetch_updates = prefetch_updates
        self.max_bytes = max_bytes
        self.timeout_seconds = float(timeout_seconds)
        self._dataset = _open_cache(self.cache_dir)
        self._pending: dict[Hashable, dict[str, Any]] = {}
        self._current: PreparedUpdate | None = None
        self._processes = []
        self._tasks = self._results = None
        self._closed = False
        self._reserved_bytes = self._peak_reserved_bytes = 0
        self._submitted = self._consumed = self._released = 0
        self._load_id = 0
        if workers:
            context = mp.get_context("spawn")
            self._tasks = context.Queue(maxsize=prefetch_updates)
            self._results = context.Queue(maxsize=prefetch_updates)
            try:
                for number in range(workers):
                    process = context.Process(target=_reader_worker,
                        args=(self.cache_dir, self._tasks, self._results),
                        name=f"readout-prefetch-{number}", daemon=True)
                    process.start()
                    self._processes.append(process)
            except BaseException:
                self.close()
                raise

    def _check_open(self):
        if self._closed:
            raise RuntimeError("PreparedBatchLoader is closed")

    def _indices(self, indices: Sequence[int]) -> tuple[int, ...]:
        result = []
        for index in indices:
            if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
                raise TypeError("Prepared indices must be integers")
            index = int(index)
            if not 0 <= index < len(self._dataset.index):
                raise IndexError(f"Prepared index {index} is outside the training cache")
            result.append(index)
        return tuple(result)

    def estimate_bytes(self, indices: Sequence[int]) -> int:
        """Reserve arrays, serialization/pinning copies and container overhead.

        Four payload copies cover workers, queue serialization and parent result
        transfer; an additional 4 KiB per unique row and 16 KiB per update cover
        small-array/container overhead. Empty-rank updates still have a budget.
        """
        self._check_open()
        indices = self._indices(indices)
        payload = 0
        unique = set(indices)
        for index in unique:
            row = self._dataset.index[index]
            payload += 4 * (int(row["grid_h"]) * int(row["grid_w"]) +
                            int(row["prompt_len"]) + int(row["answer_len"]))
        return 4 * payload + 4096 * len(unique) + 16 * len(indices) + 16384

    def prefetch(self, key: Hashable, indices: Sequence[int]) -> bool:
        self._check_open()
        hash(key)
        indices = self._indices(indices)
        if self._current is not None and self._current.key == key:
            if self._current.indices != indices:
                raise ValueError("A prepared update key was reused with different indices")
            return True
        if key in self._pending:
            if self._pending[key]["indices"] != indices:
                raise ValueError("A prepared update key was reused with different indices")
            return True
        forecast = self.estimate_bytes(indices)
        if forecast > self.max_bytes:
            raise MemoryError(f"Prepared update requires {forecast} forecast bytes, exceeding "
                              f"max_bytes={self.max_bytes}; no questions were dropped")
        if len(self._pending) >= self.prefetch_updates or self._reserved_bytes + forecast > self.max_bytes:
            return False
        entry = {"indices": indices, "forecast_bytes": forecast, "samples": None}
        if self.workers:
            try:
                self._tasks.put_nowait((key, indices))
            except queue.Full:
                return False
            except BaseException:
                self.close()
                raise
        self._pending[key] = entry
        self._reserved_bytes += forecast
        self._peak_reserved_bytes = max(self._peak_reserved_bytes, self._reserved_bytes)
        self._submitted += 1
        return True

    def _accept_result(self, response):
        kind, key, payload = response
        if kind != "result":
            raise PrefetchWorkerError(f"Prepared cache worker failed ({kind}, key={key!r}):\n{payload}")
        if key not in self._pending or self._pending[key]["samples"] is not None:
            raise PrefetchWorkerError(f"Unexpected or duplicate prepared result for key {key!r}")
        self._pending[key]["samples"] = payload

    def get(self, key: Hashable) -> PreparedUpdate:
        self._check_open()
        if self._current is not None:
            if self._current.key == key:
                return self._current
            raise RuntimeError("Release the current prepared update before getting the next one")
        if key not in self._pending:
            raise KeyError(f"Prepared update {key!r} was not submitted")
        entry = self._pending[key]
        try:
            if not self.workers:
                entry["samples"] = _read_samples(self._dataset, entry["indices"])
            else:
                deadline = time.monotonic() + self.timeout_seconds
                while entry["samples"] is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"Timed out waiting for prepared update {key!r}")
                    try:
                        self._accept_result(self._results.get(timeout=min(.2, remaining)))
                    except queue.Empty:
                        dead = [(p.name, p.exitcode) for p in self._processes if p.exitcode is not None]
                        if dead:
                            raise PrefetchWorkerError(f"Prepared cache workers exited before completing an update: {dead}")
            by_index = {int(index): PreparedSample(int(index), field, prompt, answer)
                        for index, field, prompt, answer in entry["samples"]}
            if set(by_index) != set(entry["indices"]):
                raise PrefetchWorkerError("Prepared result does not contain the exact submitted indices")
            update = PreparedUpdate(key, entry["indices"],
                tuple(by_index[index] for index in entry["indices"]), entry["forecast_bytes"], by_index)
        except BaseException:
            self.close()
            raise
        del self._pending[key]
        self._current = update
        self._consumed += 1
        return update

    def release(self, key: Hashable | None = None):
        self._check_open()
        if self._current is None:
            if key is not None:
                raise KeyError(f"No current prepared update to release for key {key!r}")
            return
        if key is not None and self._current.key != key:
            raise ValueError("Release key does not match the current prepared update")
        self._reserved_bytes -= self._current.forecast_bytes
        self._current = None
        self._released += 1

    def load(self, indices: Sequence[int]) -> PreparedUpdate:
        self._check_open()
        if self._current is not None:
            raise RuntimeError("Release the current prepared update before loading another")
        key = ("__load__", self._load_id)
        self._load_id += 1
        if not self.prefetch(key, indices):
            raise RuntimeError("Prepared prefetch budget is occupied; consume/release submitted updates first")
        return self.get(key)

    def statistics(self) -> dict:
        return {"workers": self.workers, "prefetch_updates": self.prefetch_updates,
                "max_bytes": self.max_bytes, "reserved_forecast_bytes": self._reserved_bytes,
                "peak_reserved_forecast_bytes": self._peak_reserved_bytes,
                "pending_updates": len(self._pending), "active_update": self._current is not None,
                "submitted_updates": self._submitted, "consumed_updates": self._consumed,
                "released_updates": self._released, "closed": self._closed,
                "memory_scope": "payload forecast; excludes interpreter RSS and OS page cache"}

    def close(self):
        if self._closed:
            return
        self._closed = True
        # All tasks are read-only; terminating workers cannot alter cache/cursor.
        # Do not wait for feeder threads holding results that will not be read.
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=2)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=1)
        for channel in (self._tasks, self._results):
            if channel is not None:
                channel.cancel_join_thread()
                channel.close()
        self._pending.clear()
        self._current = None
        self._reserved_bytes = 0
        self._dataset.close()

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
