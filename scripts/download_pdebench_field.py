"""Download one official PDEBench CFD shard with resume and MD5 verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

# Source: https://github.com/pdebench/PDEBench/blob/main/pdebench/data_download/pdebench_data_urls.csv
FILENAME = "2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
URL = "https://darus.uni-stuttgart.de/api/access/datafile/164687"
MD5 = "5b21dcccaef4d2145ca579a71153c580"


def verify(path, expected=MD5):
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"MD5 mismatch: {path}. Preserve/investigate the partial file before retrying.")


def download_partial(partial, resume_retries=5):
    """Restart curl after a truncated transfer, using the updated on-disk offset."""
    if resume_retries < 0:
        raise ValueError("resume_retries must be nonnegative")
    partial = Path(partial)
    for attempt in range(resume_retries + 1):
        try:
            subprocess.run(["curl", "--fail", "--location", "--retry", "5", "--connect-timeout", "30",
                            "--continue-at", "-", "--output", str(partial), URL], check=True)
            return
        except subprocess.CalledProcessError as exc:
            # curl --retry does not cover error 18 (an incomplete response).
            # A fresh invocation recalculates the resume offset from the .part file.
            if exc.returncode != 18 or attempt == resume_retries:
                raise
            size = partial.stat().st_size if partial.exists() else 0
            print(f"Partial transfer interrupted; preserving {size} bytes. "
                  f"Resume attempt {attempt + 1}/{resume_retries} in 10 seconds.", flush=True)
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--describe-only", action="store_true")
    parser.add_argument("--resume-retries", type=int, default=5,
                        help="Restart partial transfers (curl error 18) this many times; default: 5")
    cli = parser.parse_args()
    if cli.resume_retries < 0:
        parser.error("--resume-retries must be nonnegative")
    output = Path(cli.output_dir).expanduser().resolve()
    target = output / FILENAME
    print(json.dumps({"filename": FILENAME, "url": URL, "md5": MD5, "destination": str(target)}), flush=True)
    if cli.describe_only:
        return
    output.mkdir(parents=True, exist_ok=True)
    if target.exists():
        verify(target)
    else:
        partial = target.with_suffix(target.suffix + ".part")
        download_partial(partial, cli.resume_retries)
        verify(partial)
        partial.replace(target)
    print(f"verified={target}", flush=True)


if __name__ == "__main__":
    main()
