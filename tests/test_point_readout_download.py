"""Exact QA equivalence and bounded/resumable Range downloading; no real network."""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from scripts.download_point_readout_data import (
    HTTPRangeFile, SelectedFrameCache, inspect_remote_layout, prepare_profile,
)
from tensor_compression.downstream.point_readout_data import (
    PointReadoutDataset, build_dataset, load_config,
)


class RangeResponse(io.BytesIO):
    def __init__(self, payload, headers, status=206):
        super().__init__(payload)
        self.headers, self.status = headers, status


class RangeServer:
    def __init__(self, payload):
        self.payload, self.calls = payload, []
        self.etag = '"immutable-file"'
        self.ignore_range = False
        self.truncate = False
        self.fail_at = None

    def __call__(self, request, timeout):
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range"))
        start, end = map(int, match.groups())
        self.calls.append((start, end + 1))
        if self.fail_at == len(self.calls):
            raise RuntimeError("injected interruption")
        headers = {"Content-Range": f"bytes {start}-{end}/{len(self.payload)}", "ETag": self.etag}
        payload = self.payload[start:end + 1]
        if self.truncate:
            payload = payload[:-1]
        return RangeResponse(payload, headers, 200 if self.ignore_range else 206)


@pytest.fixture
def real_layout_fixture(tmp_path):
    # Generated test fixture only; production always uses the official URL.
    source = tmp_path / "source.hdf5"
    values = np.random.default_rng(9).normal(size=(20, 3, 24, 28)).astype("float32")
    with h5py.File(source, "w") as handle:
        handle.create_dataset("Vx", data=values)
        handle.create_dataset("Vy", data=values * 2 + 3)
    config = load_config(ROOT / "configs/field_to_llm_block_readout.yaml", "smoke")
    config["data"].update(fields=["Vx", "Vy"], train_shapes=["8x8", "8x12"], heldout_shapes=["9x11"])
    config["generation"].update(train_states=8, eval_states_per_shape=2)
    return source, config


def open_cache(source, cache_path):
    server = RangeServer(source.read_bytes())
    remote = HTTPRangeFile("https://fixture/source.hdf5", opener=server, retries=0)
    with h5py.File(remote, "r") as handle:
        layouts = inspect_remote_layout(handle, ["Vx", "Vy"])
    return SelectedFrameCache(remote, layouts, cache_path), remote, server


def test_range_file_matches_seek_reads_and_rejects_unsafe_responses():
    payload = b"\x89HDF\r\n\x1a\n" + bytes(range(256)) * 40
    server = RangeServer(payload)
    remote = HTTPRangeFile("https://fixture/file", opener=server, block_size=64, retries=0)
    remote.seek(61)
    assert remote.read(71) == payload[61:132]
    buffer = bytearray(16)
    assert remote.readinto(buffer) == 16
    assert buffer == payload[132:148]
    remote.seek(-9, 2)
    assert remote.read(100) == payload[-9:]
    assert remote.read(1) == b""
    with pytest.raises(ValueError, match="Unbounded"):
        remote.read()
    server.etag = '"changed"'
    with pytest.raises(ValueError, match="identity changed"):
        remote.fetch(500, 8)
    server.etag = '"immutable-file"'
    server.ignore_range = True
    with pytest.raises(ValueError, match="ignored Range"):
        remote.fetch(500, 8)
    server.ignore_range = False
    server.truncate = True
    with pytest.raises(OSError, match="Truncated"):
        remote.fetch(500, 8)


def test_selected_download_is_exactly_equivalent_and_offline(real_layout_fixture, tmp_path):
    source, config = real_layout_fixture
    complete = build_dataset(config, source, tmp_path / "complete_qa")
    cache, remote, server = open_cache(source, tmp_path / "selected.hdf5")
    try:
        selected = prepare_profile(config, cache, tmp_path / "selected_qa")
        assert selected["files"] == complete["files"]  # byte-identical state and QA JSONLs
        assert selected["trajectory_splits"] == complete["trajectory_splits"]
        assert selected["rejected_attempts"] == complete["rejected_attempts"]
        assert remote.bytes_received < source.stat().st_size / 2
        assert cache.new_frames > 0
        state = next(iter(PointReadoutDataset(tmp_path / "selected_qa", cache.path, "train").states.values()))
        with h5py.File(source, "r") as original:
            assert np.array_equal(cache.handle[state["field"]][state["sample_index"], state["time_index"]],
                                  original[state["field"]][state["sample_index"], state["time_index"]])
        assert np.isnan(cache.handle["Vx"][0, 0]).all() or cache.handle["_frame_sha256"]["Vx"][0, 0]
        with pytest.raises(ValueError, match="selected-frame cache"):
            build_dataset(config, cache.path, tmp_path / "bad_resampling")
        calls = len(server.calls)
        assert prepare_profile(config, cache, tmp_path / "selected_qa")["files"] == selected["files"]
        assert len(server.calls) == calls
    finally:
        cache.close()
        remote.close()
    # The downloader and network are now absent: actual training dataset replay
    # uses exactly the normal on-disk HDF5 path, with no remote dependency.
    for split in ("train", "val", "test"):
        dataset = PointReadoutDataset(tmp_path / "selected_qa", tmp_path / "selected.hdf5", split)
        original = PointReadoutDataset(tmp_path / "complete_qa", source, split)
        try:
            dataset.audit()
            for i in range(len(dataset)):
                assert dataset[i][0] == original[i][0]
                assert torch.equal(dataset[i][1], original[i][1])
        finally:
            dataset.close()
            original.close()


def test_interrupt_restarts_original_sampler_without_redownloading_frames(real_layout_fixture, tmp_path):
    source, config = real_layout_fixture
    reference = build_dataset(config, source, tmp_path / "reference")
    cache_path, qa_path = tmp_path / "selected.hdf5", tmp_path / "qa"
    cache, remote, server = open_cache(source, cache_path)
    server.fail_at = len(server.calls) + 4
    try:
        with pytest.raises(RuntimeError, match="interruption"):
            prepare_profile(config, cache, qa_path)
        saved = cache.new_frames
        assert saved == 3
        assert not qa_path.exists()
    finally:
        cache.close()
        remote.close()
    cache, remote, _ = open_cache(source, cache_path)
    try:
        result = prepare_profile(config, cache, qa_path)
        assert result["files"] == reference["files"]
        assert cache.reused_frames >= saved
        sample, timestep = np.argwhere(cache.handle["_frame_sha256"]["Vx"][:].astype(bool))[0]
        cache.handle["Vx"][sample, timestep, 0, 0] += 1
        with pytest.raises(ValueError, match="Cached raw frame changed"):
            cache.frame("Vx", int(sample), int(timestep))
        altered_config = json.loads(json.dumps(config))
        altered_config["generation"]["seed"] += 1
        with pytest.raises(ValueError, match="another recipe/source"):
            prepare_profile(altered_config, cache, qa_path)
    finally:
        cache.close()
        remote.close()


def test_chunked_source_is_rejected_without_payload_download(tmp_path):
    source = tmp_path / "chunked.hdf5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("Vx", data=np.ones((20, 3, 24, 28), dtype="float32"),
                              chunks=(1, 1, 24, 28), compression="gzip")
    remote = HTTPRangeFile("https://fixture/file", opener=RangeServer(source.read_bytes()), retries=0)
    try:
        with h5py.File(remote, "r") as handle:
            with pytest.raises(ValueError, match="contiguous uncompressed"):
                inspect_remote_layout(handle, ["Vx"])
    finally:
        remote.close()
