"""Check dependency imports, CUDA/BF16, local Qwen assets and optional PDEBench axes."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
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


def inspect_weight_files(path):
    """Validate safetensors headers and index coverage without loading tensors."""
    from safetensors import safe_open

    path = Path(path).resolve(strict=True)
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Model index has no weight_map")
        expected = {}
        for name, filename in weight_map.items():
            expected.setdefault(filename, set()).add(name)
    else:
        expected = {"model.safetensors": None}
    total_bytes = 0
    for filename, names in expected.items():
        shard = (path / filename).resolve()
        if not shard.is_relative_to(path):
            raise ValueError(f"Model shard must be inside the model directory: {filename}")
        if not shard.is_file() or shard.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty model shard: {shard}")
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
        if not keys or (names is not None and not names.issubset(keys)):
            raise ValueError(f"Model shard does not contain its indexed tensors: {filename}")
        total_bytes += shard.stat().st_size
    return {"weight_files": sorted(expected), "weight_size_gib": total_bytes / 2**30,
            "weight_check": "safetensors headers and index coverage; no full-file checksum"}


def inspect_model(path):
    from transformers import AutoConfig, AutoTokenizer

    path = Path(path).resolve(strict=True)
    config = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if config.model_type != "qwen2":
        raise ValueError(f"Expected Qwen2/Qwen2.5, got model_type={config.model_type!r}")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Return [0.0]."}], tokenize=True, add_generation_prompt=True)
    if not prompt:
        raise ValueError("Native chat template produced an empty prompt")
    return {"path": str(path), "model_type": config.model_type,
            "num_hidden_layers": config.num_hidden_layers,
            "tokenizer_class": type(tokenizer).__name__, "chat_template_probe_tokens": len(prompt),
            **inspect_weight_files(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-path")
    parser.add_argument("--model-dir", help="Check a local Qwen safetensors download without loading its weights")
    parser.add_argument("--fields", nargs="+", default=["Vx", "Vy", "density", "pressure"])
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output")
    cli = parser.parse_args()
    # Import the lazy model implementation as well as checking package metadata.
    # An installed transformers package can still fail to import under the base image.
    from transformers import Qwen2ForCausalLM

    report = {"python": platform.python_version(), "python_executable": sys.executable,
              "platform": platform.platform(), "qwen_model_class": Qwen2ForCausalLM.__name__,
              "packages": {name: importlib.metadata.version(name) for name in
                           ("torch", "numpy", "h5py", "transformers", "accelerate", "huggingface_hub")},
              "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
              "gpu_count": torch.cuda.device_count()}
    if torch.cuda.is_available():
        report.update(gpu=torch.cuda.get_device_name(0), bf16_supported=torch.cuda.is_bf16_supported(),
                      total_vram_gib=torch.cuda.get_device_properties(0).total_memory / 2**30)
        if report["bf16_supported"]:
            # Exercise an actual CUDA operation; detecting a device alone is insufficient.
            x = torch.ones(16, 16, device="cuda", dtype=torch.bfloat16)
            assert bool(torch.isfinite(x @ x).all())
    if cli.require_cuda and not (report["cuda_available"] and report.get("bf16_supported")):
        raise RuntimeError("CUDA and BF16 support are required for the default GPU recipe")
    if cli.model_dir:
        report["model_assets"] = inspect_model(cli.model_dir)
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
