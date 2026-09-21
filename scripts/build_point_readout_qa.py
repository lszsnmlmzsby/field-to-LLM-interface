"""Build real-field open numerical readout data without any model checkpoint."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensor_compression.downstream.point_readout import TASKS
from tensor_compression.downstream.point_readout_data import build_dataset, load_config, resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/field_to_llm_point_readout.yaml"))
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--hdf5-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--fields", nargs="+", help="HDF5 dataset keys sharing the same trajectory/time axes")
    parser.add_argument("--tasks", nargs="+", choices=TASKS)
    cli = parser.parse_args()
    config = load_config(cli.config, cli.profile)
    if cli.fields:
        config["data"]["fields"] = cli.fields
    if cli.tasks:
        config["generation"]["tasks"] = cli.tasks
    path = resolve_path(cli.hdf5_path or os.environ.get("PDEBENCH_HDF5") or config["data"]["hdf5_path"])
    output = resolve_path(cli.output_dir or config["data"]["qa_dir"])
    result = build_dataset(config, path, output)
    print(f"dataset={output} tasks={result['tasks']} counts={result['counts']}", flush=True)


if __name__ == "__main__":
    main()
