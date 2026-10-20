"""Download complete The Well sources from pinned official manifests.

No training subsets, selected frames or old/problematic archives are substituted.
The manifest is pinned on the first run and reused on every resume. Metadata can
be fetched separately, then the exact manifest copied to an offline/blocked host.
HF files have publisher checksums. The official Flatiron registry (periodicBC)
does not: its receipts explicitly distinguish local completeness checks.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tensor_compression.downstream.the_well_source import (
    DATASETS, DEFAULT_DATASETS, MANIFEST_NAME, atomic_json, canonical_hash, fetch_manifest,
    finalize_receipt, materialize_file, validate_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True,
                        help="Parent directory; each named dataset gets its own subdirectory")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DEFAULT_DATASETS),
                        help="Complete sources to obtain; default: acoustic inclusions, Euler openBC, Gray-Scott (excludes periodicBC)")
    parser.add_argument("--endpoint", default="https://huggingface.co",
                        help="Transfer endpoint, e.g. https://hf-mirror.com")
    parser.add_argument("--metadata-endpoint", default="https://huggingface.co",
                        help="HF repository API endpoint; provenance is recorded in the manifest")
    parser.add_argument("--provider", choices=("auto", "huggingface", "flatiron"), default="auto",
                        help="auto uses bundled pinned catalogs offline (Flatiron periodicBC, HF others); explicit provider fetches a new catalog only in a new directory")
    parser.add_argument("--revision", default="main",
                        help="Resolved once to a full commit; existing manifests always stay pinned")
    parser.add_argument("--manifest-dir", help="Read existing <dataset>/source_manifest.json from this directory")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Fetch/pin repository metadata and print exact bytes; no data transfer")
    parser.add_argument("--verify-only", action="store_true", help="Require all files to exist; never transfer")
    parser.add_argument("--rehash", action="store_true",
                        help="Repeat full publisher checksum and finite scans instead of stat-cache reuse")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent files (download + verification)")
    parser.add_argument("--resume-retries", type=int, default=10)
    cli = parser.parse_args()
    if cli.workers < 1 or cli.resume_retries < 0:
        parser.error("workers must be positive; resume-retries must be nonnegative")
    if cli.manifest_only and cli.verify_only:
        parser.error("--manifest-only and --verify-only are mutually exclusive")
    if len(set(cli.datasets)) != len(cli.datasets):
        parser.error("Do not repeat a dataset name")
    for dataset in cli.datasets:
        root = Path(cli.output_dir).expanduser().resolve() / dataset
        root.mkdir(parents=True, exist_ok=True)
        path = root / MANIFEST_NAME
        supplied = (Path(cli.manifest_dir).expanduser().resolve() / dataset / MANIFEST_NAME
                    if cli.manifest_dir else None)
        if path.exists():
            manifest = validate_manifest(json.loads(path.read_text(encoding="utf-8")))
            if supplied and canonical_hash(json.loads(supplied.read_text(encoding="utf-8"))) != canonical_hash(manifest):
                raise ValueError("Supplied manifest differs from pinned local source; use a separate output directory")
            if cli.revision != "main" and cli.revision != manifest["revision"]:
                raise ValueError("Revision differs from pinned source; use a separate output directory")
        elif supplied:
            manifest = validate_manifest(json.loads(supplied.read_text(encoding="utf-8")))
            atomic_json(path, manifest)
        elif cli.verify_only:
            raise FileNotFoundError(f"Cannot verify a source without its pinned manifest: {path}")
        else:
            manifest = fetch_manifest(dataset, cli.revision, cli.metadata_endpoint, cli.provider)
            atomic_json(path, manifest)
        if manifest["dataset"] != dataset:
            raise ValueError("Manifest is for a different dataset")
        data_files = [entry for entry in manifest["files"] if entry["split"]]
        print(json.dumps({"dataset": dataset, "revision": manifest["revision"],
                          "provider": manifest.get("provider", "huggingface"),
                          "manifest": str(path), "source_files": len(data_files),
                          "source_bytes": sum(f["size_bytes"] for f in data_files),
                          "all_selected_bytes": sum(f["size_bytes"] for f in manifest["files"]),
                          "excluded_files": len(manifest["excluded_files"]),
                          "metadata_endpoint": manifest["metadata_endpoint"],
                          "transfer_endpoint": "official Flatiron registry URLs" if manifest.get("provider") == "flatiron" else cli.endpoint,
                          "checksum_limitation": manifest.get("checksum_limitation")}), flush=True)
        if cli.manifest_only:
            continue
        records = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=cli.workers) as pool:
            futures = {pool.submit(materialize_file, root, manifest, entry, cli.endpoint,
                                   cli.resume_retries, cli.verify_only, cli.rehash): entry
                       for entry in manifest["files"]}
            try:
                for future in concurrent.futures.as_completed(futures):
                    record = future.result()
                    records.append(record)
                    print(json.dumps({"verified_file": record["path"],
                                      "completed": len(records), "total": len(futures)}), flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        receipt = finalize_receipt(root, manifest, records)
        print(json.dumps({"complete_source": str(root), "revision": receipt["revision"],
                          "split_files": receipt["split_file_counts"],
                          "split_trajectories": receipt["split_trajectory_counts"],
                          "verification": receipt["verification"]}), flush=True)


if __name__ == "__main__":
    main()
