"""Download only the exact real PDEBench frames selected by the existing QA builder.

The local HDF5 is an explicitly labelled partial cache with the ORIGINAL axes.
Unfetched frames are NaN, never fabricated training examples. This entry point
builds QA through the original sampler, then verifies offline replay before
publishing each QA directory. Requires an HTTP Range source with contiguous,
uncompressed float32 HDF5 fields; never falls back to a whole-file download.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(location))

from scripts.download_pdebench_field import FILENAME, MD5, URL
from tensor_compression.downstream.point_readout_data import (
    PointReadoutDataset, build_dataset, load_config, resolve_path, write_json,
)

CACHE_VERSION = "point_readout_selected_frames_v1"


class HTTPRangeFile(io.RawIOBase):
    """Bounded HTTP reads for HDF5 metadata; frame payloads use exact ranges."""

    def __init__(self, url, *, timeout=30, retries=3, block_size=4096,
                 opener=None, metadata_limit=16 * 1024**2):
        super().__init__()
        self.url, self.timeout, self.retries = url, timeout, retries
        self.block_size, self.metadata_limit = block_size, metadata_limit
        self.opener = opener or urllib.request.urlopen
        self.position, self.size = 0, None
        self.etag, self.last_modified = None, None
        self.bytes_received, self.requests, self.metadata_bytes = 0, 0, 0
        self.blocks = OrderedDict()
        signature = self.fetch(0, 8)
        if signature != b"\x89HDF\r\n\x1a\n":
            raise ValueError("Remote response is not an HDF5 file")
        if not (self.etag or self.last_modified):
            raise ValueError("Remote file has no ETag/Last-Modified identity; cannot safely cache frames")

    def fetch(self, start, length):
        if start < 0 or length <= 0 or (self.size is not None and start + length > self.size):
            raise ValueError("Invalid HTTP byte range")
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                self.url, headers={"Range": f"bytes={start}-{start + length - 1}",
                                   "Accept-Encoding": "identity"})
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    status = response.status
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                         response.headers.get("Content-Range", ""))
                    if status != 206 or not match:
                        raise ValueError("Source ignored Range; refusing to download the full file")
                    first, last, total = map(int, match.groups())
                    if first != start or last != start + length - 1 or total <= last:
                        raise ValueError("Server returned the wrong byte range")
                    identity = (total, response.headers.get("ETag"), response.headers.get("Last-Modified"))
                    if self.size is not None and identity != (self.size, self.etag, self.last_modified):
                        raise ValueError("Remote source identity changed during download")
                    payload = response.read(length + 1)
                    if len(payload) != length:
                        raise OSError("Truncated HTTP Range response")
                if self.size is None:
                    self.size, self.etag, self.last_modified = identity
                self.requests += 1
                self.bytes_received += len(payload)
                return payload
            except (OSError, urllib.error.URLError) as exc:
                if isinstance(exc, urllib.error.HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                    raise
                if attempt == self.retries:
                    raise
                print(f"range_retry={attempt + 1} offset={start} reason={type(exc).__name__}", flush=True)
                time.sleep(min(2**attempt, 8))

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_CUR:
            offset += self.position
        elif whence == os.SEEK_END:
            offset += self.size
        elif whence != os.SEEK_SET:
            raise ValueError("Invalid seek mode")
        if offset < 0:
            raise ValueError("Negative file position")
        self.position = offset
        return offset

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def read(self, length=-1):
        if self.closed:
            raise ValueError("File is closed")
        if length < 0:
            raise ValueError("Unbounded metadata reads are forbidden")
        length = min(length, max(0, self.size - self.position))
        result = bytearray()
        while len(result) < length:
            block = self.position // self.block_size
            start = block * self.block_size
            if block not in self.blocks:
                count = min(self.block_size, self.size - start)
                if self.metadata_bytes + count > self.metadata_limit:
                    raise ValueError("HDF5 metadata exceeds the download budget; stop and inspect the layout")
                self.blocks[block] = self.fetch(start, count)
                self.metadata_bytes += count
                if len(self.blocks) > 64:
                    self.blocks.popitem(last=False)
            self.blocks.move_to_end(block)
            payload = self.blocks[block]
            offset = self.position - start
            count = min(length - len(result), len(payload) - offset)
            result.extend(payload[offset:offset + count])
            self.position += count
        return bytes(result)

    @property
    def identity(self):
        return {"url": self.url, "size_bytes": self.size, "etag": self.etag,
                "last_modified": self.last_modified}


def inspect_remote_layout(handle, fields):
    layouts = {}
    for field in fields:
        dataset = handle[field]
        if len(dataset.shape) != 4 or dataset.dtype.kind != "f" or dataset.dtype.itemsize != 4:
            raise ValueError(f"{field}: expected [trajectory,time,height,width] float32")
        if dataset.chunks is not None or dataset.compression is not None or dataset.id.get_offset() is None:
            raise ValueError(f"{field}: contiguous uncompressed storage is required; no full-file fallback")
        if dataset.dtype.byteorder not in ("=", "|") and not dataset.dtype.isnative:
            raise ValueError(f"{field}: non-native byte order is not supported")
        layouts[field] = {"shape": list(dataset.shape), "dtype": dataset.dtype.str,
                          "offset": int(dataset.id.get_offset())}
    if len({tuple(v["shape"]) for v in layouts.values()}) != 1:
        raise ValueError("Selected fields must have identical original axes")
    if any(min(v["shape"]) < 1 for v in layouts.values()):
        raise ValueError("Empty source axis")
    return layouts


class SelectedFrameCache:
    def __init__(self, remote, layouts, path):
        self.remote, self.layouts, self.path = remote, layouts, Path(path)
        self.new_frames, self.reused_frames = 0, 0
        self.started = time.perf_counter()
        self.contract = {"version": CACHE_VERSION, "remote": remote.identity, "fields": layouts,
                         "official_full_file_md5": MD5, "full_file_md5_verified": False}
        existing = self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.path, "r+" if existing else "x")
        try:
            if existing:
                if not self.handle.attrs.get("field_qa_range_cache"):
                    raise ValueError("Existing file is not a selected-frame cache; do not overwrite it")
                if json.loads(self.handle.attrs["range_cache_contract"]) != self.contract:
                    raise ValueError("Cache belongs to a different remote source or layout")
                if not self.handle.attrs.get("range_cache_initialized"):
                    raise ValueError("Cache initialization was interrupted; choose a new cache path")
            else:
                self.handle.attrs["field_qa_range_cache"] = CACHE_VERSION
                self.handle.attrs["range_cache_contract"] = json.dumps(self.contract, sort_keys=True)
                hashes = self.handle.create_group("_frame_sha256")
                for field, layout in layouts.items():
                    n, times, h, w = layout["shape"]
                    self.handle.create_dataset(field, shape=(n, times, h, w), dtype=layout["dtype"],
                                               chunks=(1, 1, h, w), fillvalue=np.nan)
                    hashes.create_dataset(field, shape=(n, times), dtype="S64", chunks=(1, times))
                self.handle.attrs["range_cache_initialized"] = True
                self.handle.flush()
        except BaseException:
            self.handle.close()
            raise

    def frame(self, field, sample, timestep):
        layout = self.layouts[field]
        n, times, h, w = layout["shape"]
        if not (0 <= sample < n and 0 <= timestep < times):
            raise ValueError("Frame is outside the original source axes")
        hashes = self.handle["_frame_sha256"][field]
        digest = hashes[sample, timestep]
        if digest:
            frame = self.handle[field][sample, timestep]
            if hashlib.sha256(frame.tobytes()).hexdigest().encode("ascii") != digest:
                raise ValueError(f"Cached raw frame changed: {field}[{sample},{timestep}]")
            self.reused_frames += 1
            return frame
        dtype = np.dtype(layout["dtype"])
        length = h * w * dtype.itemsize
        offset = layout["offset"] + (sample * times + timestep) * length
        payload = self.remote.fetch(offset, length)
        frame = np.frombuffer(payload, dtype=dtype).reshape(h, w).copy()
        # Commit the payload before its completion marker. An interrupted frame
        # without a persisted hash is safely re-fetched, never accepted as zeros.
        self.handle[field][sample, timestep] = frame
        self.handle.flush()
        hashes[sample, timestep] = hashlib.sha256(frame.tobytes()).hexdigest().encode("ascii")
        self.handle.flush()
        self.new_frames += 1
        if self.new_frames == 1 or self.new_frames % 20 == 0:
            elapsed = time.perf_counter() - self.started
            print(f"downloaded_frames={self.new_frames} reused_frames={self.reused_frames} "
                  f"received_mib={self.remote.bytes_received / 2**20:.2f} elapsed_seconds={elapsed:.1f}", flush=True)
        return frame

    def __getitem__(self, field):
        cache = self
        class Field:
            shape = tuple(cache.layouts[field]["shape"])
            def __getitem__(self, key):
                sample, timestep, rows, columns = key
                return cache.frame(field, sample, timestep)[rows, columns]
        if field not in self.layouts:
            raise KeyError(field)
        return Field()

    def close(self):
        self.handle.close()


def prepare_profile(config, cache, qa_dir):
    qa_dir = Path(qa_dir)
    if qa_dir.exists() and any(qa_dir.iterdir()):
        if not (qa_dir / "metadata.json").is_file() or (qa_dir / ".build_in_progress").exists():
            raise FileExistsError(f"Unfinished/unrelated QA directory: {qa_dir}; choose a new directory")
        metadata = json.loads((qa_dir / "metadata.json").read_text(encoding="utf-8"))
        if (metadata.get("config") != json.loads(json.dumps(config))
                or metadata.get("source", {}).get("selected_frame_cache") != cache.contract):
            raise ValueError(f"Existing QA belongs to another recipe/source: {qa_dir}")
        for split in ("train", "val", "test"):
            dataset = PointReadoutDataset(qa_dir, cache.path, split)
            try:
                dataset.audit()
            finally:
                dataset.close()
        print(f"reusing_completed_qa={qa_dir}", flush=True)
        return metadata
    qa_dir.parent.mkdir(parents=True, exist_ok=True)
    # Fresh staging on each invocation allows deterministic restart without
    # deleting any previous output. Frame payloads persist in the shared cache.
    stage = Path(tempfile.mkdtemp(prefix=".range-build-", dir=qa_dir.parent))
    metadata = build_dataset(config, cache.path, stage, source_handle=cache)
    metadata["source"]["selected_frame_cache"] = cache.contract
    write_json(stage / "metadata.json", metadata)
    cache.handle.flush()
    for split in ("train", "val", "test"):
        dataset = PointReadoutDataset(stage, cache.path, split)
        try:
            print(f"offline_audit split={split} {dataset.audit()}", flush=True)
        finally:
            dataset.close()
    if qa_dir.exists():
        qa_dir.rmdir()  # Only an empty destination is permitted, checked above.
    if stage.resolve().parent != qa_dir.resolve().parent:
        raise ValueError("Staging and final QA must stay within the same explicitly selected parent")
    stage.rename(qa_dir)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/field_to_llm_block_readout.yaml"))
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), nargs="+")
    parser.add_argument("--cache-path")
    parser.add_argument("--qa-dir", help="Override QA directory for a single profile")
    parser.add_argument("--probe-only", action="store_true", help="Read metadata and report layout without creating outputs")
    parser.add_argument("--timeout", type=int, default=30)
    cli = parser.parse_args()
    if cli.timeout <= 0 or (not cli.probe_only and not cli.profile):
        parser.error("Use a positive timeout and --profile, or --probe-only")
    if cli.qa_dir and len(cli.profile or []) != 1:
        parser.error("--qa-dir requires exactly one profile")
    profiles = cli.profile or ["pilot"]
    configs = [load_config(cli.config, profile) for profile in profiles]
    fields = configs[0]["data"]["fields"]
    if any(config["data"]["fields"] != fields for config in configs):
        parser.error("All profiles must select the same physical fields")
    with ExitStack() as stack:
        remote = HTTPRangeFile(URL, timeout=cli.timeout)
        stack.callback(remote.close)
        source = stack.enter_context(h5py.File(remote, "r"))
        layouts = inspect_remote_layout(source, fields)
        print(json.dumps({"remote": remote.identity, "field_layouts": layouts,
                          "metadata_received_bytes": remote.bytes_received}), flush=True)
        if cli.probe_only:
            return
        default_cache = "${FIELD_TO_LLM_ROOT}/raw/" + FILENAME.replace(".hdf5", ".selected.hdf5")
        cache = SelectedFrameCache(remote, layouts, resolve_path(cli.cache_path or default_cache))
        stack.callback(cache.close)
        for profile, config in zip(profiles, configs):
            qa = resolve_path(cli.qa_dir or config["data"]["qa_dir"])
            result = prepare_profile(config, cache, qa)
            print(f"completed_profile={profile} qa_dir={qa} counts={result['counts']}", flush=True)
        report = {"cache_path": str(cache.path), "provenance": cache.contract,
                  "profiles": profiles, "downloaded_frames_this_invocation": cache.new_frames,
                  "reused_frames_this_invocation": cache.reused_frames,
                  "received_bytes_this_invocation": remote.bytes_received,
                  "elapsed_seconds": time.perf_counter() - cache.started}
        write_json(cache.path.with_suffix(".report.json"), report)
        print(f"ready_for_offline_training hdf5_path={cache.path}", flush=True)
        print("Set PDEBENCH_HDF5 to this selected cache; QA is already built. "
              "Do not rename it to the complete official filename.", flush=True)


if __name__ == "__main__":
    main()
