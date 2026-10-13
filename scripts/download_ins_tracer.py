"""Download the complete author-published INS-Tracer HDF5 and inspect every field."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tensor_compression.downstream.ins_tracer_source import (
    FILENAME, SOURCE, URL, inspect_complete, receipt_path, verify_receipt,
)


def download_partial(partial, retries):
    partial = Path(partial)
    if partial.with_name(partial.name + ".aria2").exists():
        raise ValueError("An aria2 control file exists; resume with aria2 and preserve its control file")
    if retries < 0:
        raise ValueError("retries must be nonnegative")
    for attempt in range(retries + 1):
        try:
            subprocess.run(["curl", "-4", "--http1.1", "--fail", "--location",
                            "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "120",
                            "--continue-at", "-", "--output", str(partial), URL], check=True)
            return
        except subprocess.CalledProcessError as exc:
            if exc.returncode not in {6, 7, 18, 28, 35, 52, 55, 56, 92} or attempt == retries:
                raise
            print(f"Transfer interrupted ({exc.returncode}); resume {attempt + 1}/{retries} in 10s", flush=True)
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--resume-retries", type=int, default=10)
    parser.add_argument("--describe-only", action="store_true")
    cli = parser.parse_args()
    if cli.resume_retries < 0:
        parser.error("--resume-retries must be nonnegative")
    target = Path(cli.output_dir).expanduser().resolve() / FILENAME
    print(json.dumps({**SOURCE, "destination": str(target), "publisher_checksum": None}), flush=True)
    if cli.describe_only:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and receipt_path(target).exists():
        verify_receipt(target)
        print(f"complete_source={target}", flush=True)
        return
    partial = target.with_name(target.name + ".part")
    if target.exists():
        candidate = target
    elif cli.verify_only:
        raise FileNotFoundError(target)
    else:
        if partial.with_name(partial.name + ".aria2").exists():
            raise ValueError("An aria2 control file exists; allocated file size is not completed download progress")
        # A completely transferred file may remain .part after an interrupted scan.
        if not partial.exists() or partial.stat().st_size != SOURCE["bytes"]:
            download_partial(partial, cli.resume_retries)
        candidate = partial
    receipt = inspect_complete(candidate)
    if candidate != target:
        if target.exists():
            raise FileExistsError("Target appeared during download; preserve both files")
        candidate.rename(target)
    receipt_path(target).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"complete_source={target}\nlocal_sha256={receipt['sha256']}", flush=True)
    print("No published checksum was available; this is a local completeness/consistency check.", flush=True)


if __name__ == "__main__":
    main()
