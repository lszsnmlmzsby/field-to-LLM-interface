"""Check installed dependencies, CUDA/BF16 and optional PDEBench axes before training."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import h5py
import torch


def inspect_fields(path, fields):
    with h5py.File(path, "r") as handle:
        if not fields or any(field not in handle for field in fields):
            raise ValueError("Select existing HDF5 field keys")
        layouts = {field: list(handle[field].shape) for field in fields}
    if any(len(s) != 4 for s in layouts.values()) or len({tuple(s) for s in layouts.values()}) != 1:
        raise ValueError("Selected fields must share [trajectory,time,height,width] axes")
    if next(iter(layouts.values()))[0] < 20:
        raise ValueError("Need at least 20 independent source trajectories")
    if min(next(iter(layouts.values()))) < 1:
        raise ValueError("Source HDF5 contains an empty axis")
    return layouts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-path")
    parser.add_argument("--fields", nargs="+", default=["Vx", "Vy", "density", "pressure"])
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output")
    cli = parser.parse_args()
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": {name: importlib.metadata.version(name) for name in
                           ("torch", "numpy", "h5py", "transformers", "accelerate", "huggingface_hub")},
              "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
    if torch.cuda.is_available():
        report.update(gpu=torch.cuda.get_device_name(0), bf16_supported=torch.cuda.is_bf16_supported(),
                      total_vram_gib=torch.cuda.get_device_properties(0).total_memory / 2**30)
        if report["bf16_supported"]:
            # Exercise an actual CUDA operation; detecting a device alone is insufficient.
            x = torch.ones(16, 16, device="cuda", dtype=torch.bfloat16)
            assert bool(torch.isfinite(x @ x).all())
    if cli.require_cuda and not (report["cuda_available"] and report.get("bf16_supported")):
        raise RuntimeError("CUDA and BF16 support are required for the default A6000 recipe")
    if cli.hdf5_path:
        report["field_layouts"] = inspect_fields(cli.hdf5_path, cli.fields)
    text = json.dumps(report, indent=2)
    print(text)
    if cli.output:
        path = Path(cli.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
