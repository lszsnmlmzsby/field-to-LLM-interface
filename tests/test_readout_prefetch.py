"""CPU/spawn prefetch checks; no remote GPU performance claims."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts import readout_prefetch as prefetch
from tensor_compression.downstream.point_readout import json_hash
from tensor_compression.downstream.readout_training_cache import ARRAY_FILES, CACHE_FORMAT, INDEX_DTYPE


def cache_fixture(path, *, bad_field_offset=False):
    path.mkdir()
    index = np.zeros(4, dtype=INDEX_DTYPE)
    fields = np.arange(24, dtype=np.float16) / 8
    tokens = np.arange(28, dtype=np.uint32) + 100
    for i in range(4):
        index[i] = (2, 3, i * 6, 4, 3, i * 7,
                    f"qa{i}".encode(), f"state{i}".encode(), 0)
    if bad_field_offset:
        index[2]["field_offset"] = 10000
    np.save(path / "index.npy", index)
    np.save(path / "token_lengths.npy", np.full(4, 7, dtype="<u4"))
    np.save(path / "fields.npy", fields)
    tokens.tofile(path / "tokens.bin")
    manifest = {"format": CACHE_FORMAT, "status": "complete", "split": "train",
                "files": {name: {} for name in ARRAY_FILES}, "metadata": {},
                "identity": json_hash({}), "qa_dir": str(path), "questions": 4,
                "field_values": len(fields), "token_count": len(tokens), "tasks": ["single_point"]}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.mark.parametrize("workers", [0, 2])
def test_order_duplicates_tokens_and_retry_identity(tmp_path, workers):
    cache = cache_fixture(tmp_path / "cache")
    with prefetch.PreparedBatchLoader(cache, workers=workers, timeout_seconds=40) as loader:
        assert loader.prefetch((0, 0), [2, 0, 2, 3])
        assert loader.prefetch((0, 1), [1])
        update = loader.get((0, 0))
        assert update.indices == (2, 0, 2, 3)
        assert [sample.index for sample in update.items] == [2, 0, 2, 3]
        assert update.items[0] is update.items[2]
        assert loader.get((0, 0)) is update
        assert loader.prefetch((0, 0), [2, 0, 2, 3])
        np.testing.assert_array_equal(update.item(2).field, np.arange(12, 18, dtype=np.float32).reshape(1, 2, 3) / 8)
        assert update.item(2).field.dtype == np.float32
        assert update.item(2).field.flags.writeable
        prompt, answer = update.token_pair(2)
        np.testing.assert_array_equal(prompt, np.arange(114, 118, dtype=np.uint32))
        np.testing.assert_array_equal(answer, np.arange(118, 121, dtype=np.uint32))
        with pytest.raises(RuntimeError, match="Release"):
            loader.get((0, 1))
        loader.release((0, 0))
        next_update = loader.get((0, 1))
        assert next_update.indices == (1,)
        loader.release((0, 1))
        assert loader.statistics()["reserved_forecast_bytes"] == 0
        assert loader.statistics()["consumed_updates"] == 2
        processes = list(loader._processes)
    assert not any(process.is_alive() for process in processes)


def test_future_count_and_active_byte_reservations(tmp_path):
    cache = cache_fixture(tmp_path / "cache")
    with prefetch.PreparedBatchLoader(cache, workers=0, prefetch_updates=1) as loader:
        one = loader.estimate_bytes([0])
        loader.max_bytes = one * 2
        assert loader.prefetch("first", [0])
        assert not loader.prefetch("second", [1])  # bounded future count
        loader.get("first")
        assert loader.prefetch("second", [1])  # active + one future fit
        assert loader.statistics()["reserved_forecast_bytes"] == one * 2
        assert not loader.prefetch("third", [2])
        loader.release("first")
        loader.get("second")
        loader.release("second")
        assert loader.statistics()["peak_reserved_forecast_bytes"] == one * 2
    with prefetch.PreparedBatchLoader(cache, workers=0, prefetch_updates=3, max_bytes=one) as loader:
        assert loader.prefetch("first", [0])
        loader.get("first")
        assert not loader.prefetch("second", [1])  # active bytes count against cap
        loader.release("first")
        assert loader.prefetch("second", [1])


def test_spawn_results_can_arrive_in_another_order_without_changing_requested_order(tmp_path):
    cache = cache_fixture(tmp_path / "cache")
    with prefetch.PreparedBatchLoader(cache, workers=1, prefetch_updates=2, timeout_seconds=40) as loader:
        assert loader.prefetch("first-submitted", [0, 2])
        assert loader.prefetch("requested-first", [3, 1, 3])
        # One worker guarantees first-submitted arrives first, while the caller
        # requests the second key. The other result must remain budgeted/usable.
        second = loader.get("requested-first")
        assert second.indices == (3, 1, 3)
        assert loader.statistics()["pending_updates"] == 1
        loader.release()
        assert loader.get("first-submitted").indices == (0, 2)
        loader.release()


def test_budget_rejection_and_key_validation_never_drop_indices(tmp_path):
    cache = cache_fixture(tmp_path / "cache")
    with prefetch.PreparedBatchLoader(cache, workers=0) as loader:
        assert loader.prefetch("same", [3, 1])
        with pytest.raises(ValueError, match="different indices"):
            loader.prefetch("same", [1, 3])
        with pytest.raises(IndexError):
            loader.prefetch("outside", [4])
        with pytest.raises(TypeError):
            loader.prefetch("float", [1.2])
        assert loader.get("same").indices == (3, 1)
        with pytest.raises(ValueError, match="Release key"):
            loader.release("wrong")
        loader.release("same")
        loader.max_bytes = loader.estimate_bytes([0, 1]) - 1
        with pytest.raises(MemoryError, match="no questions were dropped"):
            loader.prefetch("too-large", [0, 1])
        assert loader.statistics()["pending_updates"] == 0
        assert loader.statistics()["reserved_forecast_bytes"] == 0


@pytest.mark.parametrize("workers", [0, 1])
def test_reader_failure_clears_and_closes_loader(tmp_path, workers):
    # Runtime production opening follows a full identity/index audit. Deliberately
    # malformed fixture here exercises failure cleanup in the worker path.
    cache = cache_fixture(tmp_path / "cache", bad_field_offset=True)
    loader = prefetch.PreparedBatchLoader(cache, workers=workers, timeout_seconds=40)
    assert loader.prefetch("bad", [2])
    with pytest.raises((ValueError, prefetch.PrefetchWorkerError)):
        loader.get("bad")
    assert loader.statistics()["closed"]
    assert loader.statistics()["reserved_forecast_bytes"] == 0
    assert not any(process.is_alive() for process in loader._processes)
    with pytest.raises(RuntimeError, match="closed"):
        loader.prefetch("later", [0])
    loader.close()


def _blocked_worker(cache_dir, tasks, results):
    time.sleep(60)


def test_timeout_terminates_spawned_workers_and_discards_pending(tmp_path, monkeypatch):
    cache = cache_fixture(tmp_path / "cache")
    monkeypatch.setattr(prefetch, "_reader_worker", _blocked_worker)
    loader = prefetch.PreparedBatchLoader(cache, workers=1, timeout_seconds=.05)
    assert loader.prefetch("blocked", [0])
    with pytest.raises(TimeoutError, match="Timed out"):
        loader.get("blocked")
    assert loader.statistics()["closed"]
    assert loader.statistics()["pending_updates"] == 0
    assert not any(process.is_alive() for process in loader._processes)


@pytest.mark.parametrize("workers", [0, 1])
def test_empty_rank_and_load_convenience(tmp_path, workers):
    cache = cache_fixture(tmp_path / "cache")
    with prefetch.PreparedBatchLoader(cache, workers=workers, timeout_seconds=40) as loader:
        empty = loader.load([])
        assert empty.indices == empty.items == ()
        assert empty.by_index == {}
        loader.release()
        update = loader.load([3, 0])
        assert update.indices == (3, 0)
        loader.release()


@pytest.mark.parametrize("kwargs", [{"workers": -1}, {"workers": 5}, {"prefetch_updates": 0},
                                    {"max_bytes": 0}, {"timeout_seconds": 0},
                                    {"timeout_seconds": float("inf")}, {"pin_memory": True}])
def test_invalid_configuration_fails_before_opening_cache(tmp_path, kwargs):
    with pytest.raises(ValueError):
        prefetch.PreparedBatchLoader(tmp_path / "absent", **kwargs)
