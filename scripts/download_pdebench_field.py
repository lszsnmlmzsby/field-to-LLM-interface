"""Download one official PDEBench CFD shard with resume and MD5 verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--describe-only", action="store_true")
    cli = parser.parse_args()
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
        subprocess.run(["curl", "--fail", "--location", "--retry", "5", "--connect-timeout", "30",
                        "--continue-at", "-", "--output", str(partial), URL], check=True)
        verify(partial)
        partial.replace(target)
    print(f"verified={target}", flush=True)


if __name__ == "__main__":
    main()
