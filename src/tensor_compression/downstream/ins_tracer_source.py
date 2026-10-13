"""Complete INS-Tracer validation; local SHA256 is not a publisher checksum."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

FILENAME = "dedalus_v5.1_Baseline2D_INSTracer_icA_noP_db1_nu0.001D0.01_seed0.hdf5"
URL = "http://data-download.obs.cn-northeast-227.dlaicc.com/dedalus_v5.1/Baseline2D/" + FILENAME
SOURCE = {"url": URL, "bytes": 20250643760,
          "fields": {name: [1000, 101, 128, 128] for name in ("sol/u", "sol/v", "sol/b")}}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt_path(path):
    return Path(str(path) + ".verified.json")


def inspect_complete(path, source=SOURCE):
    """Check byte count, all solution values, coordinate layout, and whole-file hash."""
    path = Path(path)
    if path.stat().st_size != source["bytes"]:
        raise ValueError(f"Incomplete/wrong source size: {path.stat().st_size}; expected {source['bytes']}")
    counts = {}
    with h5py.File(path, "r") as handle:
        for key, shape in source["fields"].items():
            dataset = handle[key]
            if list(dataset.shape) != shape or dataset.dtype != np.dtype("float32"):
                raise ValueError(f"Unexpected layout/dtype for {key}")
            for trajectory in range(shape[0]):
                # One time slice at a time: bounded RAM, every solution value visited.
                for time in range(shape[1]):
                    values = dataset[trajectory, time]
                    if not np.isfinite(values).all():
                        raise ValueError(f"Nonfinite values: {key}, trajectory={trajectory}, time={time}")
            counts[key] = int(np.prod(shape))
            print(f"complete_field_scan={key} values={counts[key]}", flush=True)
        n, t, h, w = next(iter(source["fields"].values()))
        if int(handle["args/num_pde"][()]) != n:
            raise ValueError("Trajectory count metadata differs")
        for key, size in (("coord/t", t), ("coord/x", h), ("coord/y", w)):
            coordinates = np.asarray(handle[key][()]).reshape(-1)
            if coordinates.size != size or not np.isfinite(coordinates).all() or not np.all(np.diff(coordinates) > 0):
                raise ValueError(f"Invalid coordinates: {key}")
    print("Computing SHA256 over the complete file...", flush=True)
    return {"source": source, "sha256": sha256(path), "finite_values_scanned": counts,
            "publisher_checksum": None,
            "verification": "complete_byte_count_hdf5_layout_full_finite_scan_and_local_sha256"}


def verify_receipt(path):
    path = Path(path)
    receipt = json.loads(receipt_path(path).read_text(encoding="utf-8"))
    expected_counts = {k: int(np.prod(v)) for k, v in SOURCE["fields"].items()}
    if (receipt.get("source") != SOURCE or path.stat().st_size != SOURCE["bytes"]
            or receipt.get("finite_values_scanned") != expected_counts):
        raise ValueError("Complete INS-Tracer validation receipt does not match the source")
    print("Checking complete INS-Tracer SHA256 before building QA...", flush=True)
    if sha256(path) != receipt["sha256"]:
        raise ValueError("INS-Tracer changed after validation")
    return receipt
