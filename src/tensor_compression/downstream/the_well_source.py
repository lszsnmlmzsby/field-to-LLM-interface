"""Pinned, complete The Well sources and bounded-memory integrity inspection.

For HF sources, the publisher's LFS SHA256 verifies every HDF5 byte. Flatiron's
official registry has no file checksums: its receipts explicitly report only
complete byte counts, layout/finite checks and a locally computed SHA256.
Subsequent receipt checks
use size/mtime caches unless ``rehash=True``; they do not prove protection from
deliberate timestamp-preserving edits. This module never treats an HTTP ETag as
a publisher checksum and never treats a subset as a complete source.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
import concurrent.futures
from collections import Counter
from pathlib import Path, PurePosixPath

import h5py
import numpy as np
import yaml

DATASETS = (
    "acoustic_scattering_inclusions",
    "euler_multi_quadrants_openBC",
    "euler_multi_quadrants_periodicBC",
    "gray_scott_reaction_diffusion",
)
# The active experiment keeps a single Euler source. Other supported sources
# require an explicit --datasets selection and are never downloaded implicitly.
DEFAULT_DATASETS = (
    "acoustic_scattering_inclusions",
    "euler_multi_quadrants_openBC",
    "gray_scott_reaction_diffusion",
)
MANIFEST_NAME = "source_manifest.json"
RECEIPT_NAME = "source_verified.json"
MANIFEST_SCHEMA = "the_well_manifest_v1"
RECEIPT_SCHEMA = "the_well_verified_v1"
SPLITS = ("train", "valid", "test")
FLATIRON_BASE = "https://sdsc-users.flatironinstitute.org/~polymathic/data/the_well/datasets/"
BUNDLED_MANIFEST_DIR = Path(__file__).resolve().parents[3] / "configs" / "source_manifests"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def safe_path(root, relative):
    pure = PurePosixPath(relative)
    if (not relative or pure.is_absolute() or ".." in pure.parts or "\\" in relative
            or ":" in relative or str(pure) != relative):
        raise ValueError(f"Unsafe/noncanonical source path: {relative!r}")
    root = Path(root).resolve()
    result = root.joinpath(*pure.parts).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Source path escapes root: {relative}")
    return result


def _endpoint(value):
    value = value.rstrip("/")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.query or parsed.fragment:
        raise ValueError("HF endpoints must be HTTPS base URLs without credentials/query/fragment")
    return value


def manifest_from_info(dataset, info, metadata_endpoint="https://huggingface.co"):
    """Select ALL current split files from one commit, keeping explicit exclusions."""
    if dataset not in DATASETS:
        raise ValueError(f"Supported datasets: {DATASETS}")
    revision = info.get("sha", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Repository metadata must resolve to a full pinned commit SHA")
    repo_id = f"polymathic-ai/{dataset}"
    if info.get("id", repo_id) != repo_id:
        raise ValueError("Repository identity differs from the requested official dataset")
    selected, excluded, paths = [], [], set()
    for item in info.get("siblings", []):
        name = item["rfilename"]
        safe_path(Path.cwd(), name)
        if name in paths:
            raise ValueError(f"Duplicate manifest path: {name}")
        paths.add(name)
        parts = PurePosixPath(name).parts
        is_hdf5 = name.lower().endswith((".h5", ".hdf5"))
        if parts[0] == "old_and_problematic_data":
            excluded.append({"path": name, "reason": "publisher_archived_old_and_problematic_data"})
            continue
        split = None
        if parts[0] == "data":
            if is_hdf5 and (len(parts) < 3 or parts[1] not in SPLITS):
                raise ValueError(f"Unexpected current data path; inspect before changing scope: {name}")
            split = parts[1] if is_hdf5 else None
        elif is_hdf5:
            raise ValueError(f"Unexpected HDF5 outside current data/ splits: {name}")
        elif not (len(parts) == 1 or parts[0] == "generation"):
            excluded.append({"path": name, "reason": "auxiliary_visualization_or_non_source_asset"})
            continue
        if not is_hdf5 and PurePosixPath(name).suffix.lower() not in (
                ".yaml", ".yml", ".json", ".md", ".txt", ".py", ".toml", ".cfg", ".ini"):
            if PurePosixPath(name).name not in ("LICENSE", "LICENSE.txt", ".gitattributes"):
                excluded.append({"path": name, "reason": "auxiliary_visualization_or_non_source_asset"})
                continue
        lfs = item.get("lfs") or {}
        sha256 = lfs.get("sha256")
        blob = item.get("blobId") or item.get("blob_id")
        size = lfs.get("size", item.get("size"))
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"Missing byte size; request HF API blobs=true: {name}")
        if sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError(f"Invalid publisher SHA256: {name}")
        if is_hdf5 and not sha256:
            raise ValueError(f"Missing publisher LFS SHA256 for complete HDF5: {name}")
        if not sha256 and not (isinstance(blob, str) and re.fullmatch(r"[0-9a-f]{40}", blob)):
            raise ValueError(f"Missing Git blob checksum for metadata: {name}")
        selected.append({"path": name, "split": split, "size_bytes": size,
                         "sha256": sha256, "git_blob_sha1": None if sha256 else blob})
    if not selected or any(not any(f["split"] == split for f in selected) for split in SPLITS):
        raise ValueError("A complete source must contain all official train, valid and test splits")
    manifest = {"schema": MANIFEST_SCHEMA, "provider": "huggingface", "dataset": dataset, "repo_id": repo_id,
                "revision": revision, "metadata_endpoint": _endpoint(metadata_endpoint),
                "scope": "all_current_data_split_hdf5_and_text_metadata",
                "files": sorted(selected, key=lambda f: f["path"]),
                "excluded_files": sorted(excluded, key=lambda f: f["path"])}
    validate_manifest(manifest)
    return manifest


def fetch_manifest(dataset, revision="main", metadata_endpoint="https://huggingface.co", provider="auto"):
    if provider not in ("auto", "huggingface", "flatiron"):
        raise ValueError("Unknown source provider")
    if provider == "auto" and dataset in DATASETS:
        bundled = BUNDLED_MANIFEST_DIR / dataset / MANIFEST_NAME
        if bundled.exists():
            manifest = validate_manifest(json.loads(bundled.read_text(encoding="utf-8")))
            expected_provider = "flatiron" if dataset == "euler_multi_quadrants_periodicBC" else "huggingface"
            if manifest["dataset"] != dataset or manifest.get("provider") != expected_provider:
                raise ValueError("Bundled source manifest has wrong identity/provider")
            if revision not in ("main", manifest["revision"]):
                raise ValueError(f"Bundled source registry is pinned; use --provider {expected_provider} with a new directory to choose another revision")
            return manifest
    if provider == "flatiron" or (provider == "auto" and dataset == "euler_multi_quadrants_periodicBC"):
        return fetch_flatiron_manifest(dataset, None if revision == "main" else revision)
    endpoint = _endpoint(metadata_endpoint)
    url = (f"{endpoint}/api/datasets/polymathic-ai/{dataset}/revision/"
           f"{urllib.parse.quote(revision, safe='')}?blobs=true")
    request = urllib.request.Request(url, headers={"User-Agent": "tensor-compression-source-verifier/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        info = json.load(response)
    return manifest_from_info(dataset, info, endpoint)


def _read_url(url):
    request = urllib.request.Request(url, headers={"User-Agent": "tensor-compression-source-verifier/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def head_fingerprint(url):
    request = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "tensor-compression-source-verifier/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
        if not length or not length.isdigit():
            raise ValueError(f"Publisher did not provide a byte count: {url}")
        return {"size_bytes": int(length), "http_etag": response.headers.get("ETag"),
                "http_last_modified": response.headers.get("Last-Modified")}


def flatiron_manifest_from_registry(dataset, registry, revision, registry_sha256, head=head_fingerprint):
    """Official registry is complete scope; HTTP ETags are only change detectors."""
    if dataset not in DATASETS or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Unsupported dataset or unpinned registry revision")
    source = registry[dataset]
    if set(source) != {"stats", *SPLITS}:
        raise ValueError("Official registry schema changed; inspect source scope before downloading")
    prefix = FLATIRON_BASE + dataset + "/"
    targets = [(None, source["stats"])]
    for split in SPLITS:
        if not source[split]:
            raise ValueError("Official registry must include all splits")
        targets.extend((split, url) for url in source[split])
    def describe(target):
        split, url = target
        if not url.startswith(prefix) or urllib.parse.urlsplit(url).query:
            raise ValueError(f"Registry URL is outside official dataset root: {url}")
        relative = url[len(prefix):]
        safe_path(Path.cwd(), relative)
        return {"path": relative, "split": split, "url": url, "sha256": None,
                "git_blob_sha1": None, "publisher_checksum": None, **head(url)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        entries = list(pool.map(describe, targets))
    manifest = {"schema": MANIFEST_SCHEMA, "provider": "flatiron", "dataset": dataset,
                "repo_id": "PolymathicAI/the_well", "revision": revision,
                "metadata_endpoint": "https://raw.githubusercontent.com",
                "registry_url": f"https://raw.githubusercontent.com/PolymathicAI/the_well/{revision}/the_well/utils/registry.yaml",
                "registry_sha256": registry_sha256,
                "scope": "all_files_in_pinned_official_registry_all_three_splits",
                "checksum_limitation": "No published file checksum in official registry; full size/layout/finite scan and local SHA256 only. HTTP ETag is not a checksum.",
                "files": sorted(entries, key=lambda item: item["path"]), "excluded_files": []}
    return validate_manifest(manifest)


def fetch_flatiron_manifest(dataset, registry_revision=None):
    if registry_revision is None:
        commits = json.loads(_read_url("https://api.github.com/repos/PolymathicAI/the_well/commits"
                                      "?path=the_well/utils/registry.yaml&per_page=1"))
        registry_revision = commits[0]["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", registry_revision):
        raise ValueError("Flatiron registry revision must be a full GitHub commit SHA")
    url = f"https://raw.githubusercontent.com/PolymathicAI/the_well/{registry_revision}/the_well/utils/registry.yaml"
    content = _read_url(url)
    registry = yaml.safe_load(content)
    return flatiron_manifest_from_registry(dataset, registry, registry_revision,
                                           hashlib.sha256(content).hexdigest())


def validate_manifest(manifest):
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("dataset") not in DATASETS:
        raise ValueError("Unsupported The Well manifest")
    provider = manifest.get("provider", "huggingface")
    expected_repo = "PolymathicAI/the_well" if provider == "flatiron" else "polymathic-ai/" + manifest["dataset"]
    if provider not in ("huggingface", "flatiron") or manifest.get("repo_id") != expected_repo:
        raise ValueError("Source must be the official publisher repository")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("revision", "")):
        raise ValueError("Source revision must be pinned")
    if provider == "flatiron":
        expected_registry = ("https://raw.githubusercontent.com/PolymathicAI/the_well/"
                             + manifest["revision"] + "/the_well/utils/registry.yaml")
        if (manifest.get("registry_url") != expected_registry
                or not re.fullmatch(r"[0-9a-f]{64}", manifest.get("registry_sha256", ""))):
            raise ValueError("Flatiron source must retain pinned official registry provenance")
    paths = set()
    for entry in manifest["files"]:
        path = entry["path"]
        safe_path(Path.cwd(), path)
        if path in paths or path.startswith("old_and_problematic_data/"):
            raise ValueError("Duplicate or archived/problematic source in manifest")
        paths.add(path)
        if not isinstance(entry["size_bytes"], int) or entry["size_bytes"] < 0:
            raise ValueError("Invalid manifest byte size")
        if entry["split"] is not None:
            if (entry["split"] not in SPLITS or not path.startswith(f"data/{entry['split']}/")
                    or not path.lower().endswith((".h5", ".hdf5"))
                    or (provider == "huggingface" and not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256") or ""))):
                raise ValueError("Invalid source split/checksum")
        elif path.lower().endswith((".h5", ".hdf5")):
            raise ValueError("HDF5 files must declare their official split")
        if provider == "flatiron":
            if entry.get("url") != FLATIRON_BASE + manifest["dataset"] + "/" + path:
                raise ValueError("Nonofficial Flatiron source URL")
            if entry.get("publisher_checksum") is not None or entry.get("sha256") is not None:
                raise ValueError("Do not invent a publisher checksum for the Flatiron registry")
        elif not entry.get("sha256") and not re.fullmatch(r"[0-9a-f]{40}", entry.get("git_blob_sha1") or ""):
            raise ValueError("Missing source checksum")
    if any(not any(f["split"] == split for f in manifest["files"]) for split in SPLITS):
        raise ValueError("Manifest is missing an official split")
    return manifest


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _boolean(attrs, key):
    if key not in attrs or np.asarray(attrs[key]).shape != () or attrs[key] not in (False, True, 0, 1):
        raise ValueError(f"Missing/invalid boolean HDF5 attribute {key}")
    return bool(attrs[key])


def _blocks(dataset, prefix=(), max_bytes=64 * 1024**2):
    """Contiguous C-order slabs, bounded even for unusually large spatial axes."""
    shape = dataset.shape[len(prefix):]
    if not shape:
        yield np.asarray(dataset[prefix])
    elif int(np.prod(shape)) * dataset.dtype.itemsize <= max_bytes:
        yield np.asarray(dataset[prefix + (slice(None),) * len(shape)])
    else:
        for index in range(shape[0]):
            yield from _blocks(dataset, prefix + (index,), max_bytes)


def _euler_trajectory_count(handle):
    """Resolve the published openBC negative-count defect from all stored fields.

    Caller must first verify the publisher's full-file SHA256. This only resolves
    the root count; normal coordinate, shape and full-value checks still follow.
    """
    counts = {}
    for rank in range(3):
        group = handle.get(f"t{rank}_fields")
        if group is None:
            continue
        for ds in group.values():
            if not isinstance(ds, h5py.Dataset):
                raise ValueError(f"Non-dataset field: {ds.name}")
            if _boolean(ds.attrs, "sample_varying"):
                if not ds.shape or ds.shape[0] < 1:
                    raise ValueError(f"Empty sample axis: {ds.name}")
                counts[ds.name.lstrip("/")] = int(ds.shape[0])
    required = {"t0_fields/density", "t0_fields/energy", "t0_fields/pressure", "t1_fields/momentum"}
    if not required.issubset(counts) or len(set(counts.values())) != 1:
        raise ValueError(f"Cannot resolve Euler trajectory count: inconsistent/missing field sample axes {counts}")
    return next(iter(counts.values())), counts


def inspect_hdf5(path, dataset_name=None, *, publisher_sha256_verified=False):
    """Scan every field value; retain actual axis/field layout, no fixed key guesses.

    Full-trajectory and first-state digests are for duplicate detection. They
    compare all stored scalar/vector fields together, including static fields.
    Equality of an initial state alone is a diagnostic, not proof of leakage
    when PDE parameters or boundary conditions differ.
    """
    with h5py.File(path, "r") as handle:
        name = _text(handle.attrs["dataset_name"])
        if dataset_name is not None and name != dataset_name:
            raise ValueError(f"HDF5 dataset_name {name!r} differs from {dataset_name!r}")
        if int(handle.attrs["n_spatial_dims"]) != 2 or _text(handle.attrs["grid_type"]) != "cartesian":
            raise ValueError("This experiment requires 2D Cartesian sources")
        n = int(handle.attrs["n_trajectories"])
        corrections = []
        if n < 0 and name == "euler_multi_quadrants_openBC" and publisher_sha256_verified:
            actual_n, field_counts = _euler_trajectory_count(handle)
            corrections.append({"attribute": "n_trajectories", "declared": n, "effective": actual_n,
                                "policy": "publisher_sha256_verified_euler_negative_count_v1",
                                "field_sample_counts": field_counts})
            n = actual_n
        dims = handle["dimensions"]
        axes = [_text(value) for value in dims.attrs["spatial_dims"]]
        if n < 1 or len(axes) != 2 or len(set(axes)) != 2:
            raise ValueError(
                f"Invalid trajectory/spatial axis metadata in {path}: "
                f"n_trajectories={handle.attrs['n_trajectories']!r}, "
                f"spatial_dims={dims.attrs['spatial_dims']!r}")
        spatial_shape = [int(dims[axis].shape[-1]) for axis in axes]
        steps = int(dims["time"].shape[-1])
        if min(spatial_shape + [steps]) < 1:
            raise ValueError("Empty coordinates")
        for key in axes + ["time"]:
            coordinates = dims[key]
            if coordinates.ndim not in (1, 2) or (coordinates.ndim == 2 and coordinates.shape[0] != n):
                raise ValueError(f"Unsupported coordinate layout: {key}")
            for values in _blocks(coordinates):
                if not np.isfinite(values).all() or not np.all(np.diff(values, axis=-1) > 0):
                    raise ValueError(f"Nonfinite/nonmonotonic coordinates: {key}")
        fields, counts = [], {}
        trajectory_digests = [hashlib.sha256() for _ in range(n)]
        initial_digests = [hashlib.sha256() for _ in range(n)]
        for rank in range(3):
            group = handle.get(f"t{rank}_fields")
            if group is None:
                continue
            declared = {_text(value) for value in group.attrs.get("field_names", [])}
            if declared != set(group.keys()):
                raise ValueError(f"Field names differ from stored fields: {group.name}")
            for field_name in sorted(group.keys()):
                ds = group[field_name]
                if not isinstance(ds, h5py.Dataset) or ds.dtype.kind != "f":
                    raise ValueError(f"Non-floating field: {ds.name}")
                sample_varying = _boolean(ds.attrs, "sample_varying")
                time_varying = _boolean(ds.attrs, "time_varying")
                varying = list(np.asarray(ds.attrs["dim_varying"]).tolist())
                if len(varying) != 2 or any(v not in (False, True, 0, 1) for v in varying):
                    raise ValueError(f"Invalid dim_varying: {ds.name}")
                varying = [bool(v) for v in varying]
                expected = ([n] if sample_varying else []) + ([steps] if time_varying else [])
                expected += [s for s, varies in zip(spatial_shape, varying) if varies] + [2] * rank
                if list(ds.shape) != expected:
                    raise ValueError(f"Field shape/attribute mismatch {ds.name}: {list(ds.shape)} != {expected}")
                key = ds.name.lstrip("/")
                field = {"key": key, "rank": rank, "sample_varying": sample_varying,
                         "time_varying": time_varying, "dim_varying": varying,
                         "shape": list(ds.shape), "dtype": str(ds.dtype), "components": 2**rank}
                fields.append(field)
                header = json.dumps({"key": key, "shape": expected[1:] if sample_varying else expected,
                                     "dtype": str(ds.dtype)}, sort_keys=True).encode()
                initial_header = json.dumps({"key": key, "spatial_shape": spatial_shape,
                                             "rank": rank, "dtype": str(ds.dtype)}, sort_keys=True).encode()
                total = 0
                for trajectory in range(n if sample_varying else 1):
                    prefix = (trajectory,) if sample_varying else ()
                    field_digest = hashlib.sha256(header)
                    for values in _blocks(ds, prefix):
                        if not np.isfinite(values).all():
                            raise ValueError(f"Nonfinite field: {key}, trajectory={trajectory}")
                        total += values.size
                        field_digest.update(np.ascontiguousarray(values).tobytes())
                    first = hashlib.sha256(initial_header)
                    first_prefix = prefix + ((0,) if time_varying else ())
                    for values in _blocks(ds, first_prefix):
                        first.update(np.ascontiguousarray(values).tobytes())
                    targets = [trajectory] if sample_varying else range(n)
                    for target in targets:
                        trajectory_digests[target].update(field_digest.digest())
                        initial_digests[target].update(first.digest())
                counts[key] = int(total)
                if total != int(np.prod(ds.shape)):
                    raise ValueError(f"Incomplete finite scan: {key}")
                print(f"complete_field_scan={key} values={total}", flush=True)
        if not fields or not any(f["time_varying"] for f in fields):
            raise ValueError("No dynamic solution fields found")
        if corrections:
            print(json.dumps({"metadata_corrections": corrections, "file": str(path)}), flush=True)
        return {"dataset_name": name, "n_trajectories": n, "time_steps": steps,
                "spatial_dims": axes, "spatial_shape": spatial_shape, "fields": fields,
                "finite_values_scanned": counts,
                "trajectory_sha256": [d.hexdigest() for d in trajectory_digests],
                "initial_state_sha256": [d.hexdigest() for d in initial_digests],
                **({"metadata_corrections": corrections} if corrections else {})}


def hash_file(path, *, git_blob=False):
    path = Path(path)
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode()) if git_blob else None
    with path.open("rb") as handle:
        for values in iter(lambda: handle.read(8 * 1024**2), b""):
            sha.update(values)
            if blob is not None:
                blob.update(values)
    return sha.hexdigest(), blob.hexdigest() if blob is not None else None


def verify_file(path, entry, dataset_name):
    path = Path(path)
    before = path.stat()
    if before.st_size != entry["size_bytes"]:
        raise ValueError(f"Incomplete/wrong byte size: {path}")
    sha, blob = hash_file(path, git_blob=bool(entry.get("git_blob_sha1")))
    if entry.get("sha256") and sha != entry["sha256"]:
        raise ValueError(f"Publisher SHA256 mismatch: {path}; preserve/quarantine this file before retry")
    if entry.get("git_blob_sha1") and blob != entry["git_blob_sha1"]:
        raise ValueError(f"Publisher Git blob checksum mismatch: {path}")
    try:
        layout = inspect_hdf5(path, dataset_name, publisher_sha256_verified=bool(entry.get("sha256"))) if entry["split"] else None
    except (ValueError, KeyError, OSError, TypeError) as exc:
        checksum_status = ("publisher checksum passed" if entry.get("sha256") or entry.get("git_blob_sha1")
                           else "byte count checked; no publisher checksum available")
        raise ValueError(f"HDF5 verification failed for {path} ({checksum_status}): {exc}. "
                         "Existing file preserved; do not delete or bypass validation.") from exc
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source changed during verification: {path}")
    has_publisher_checksum = bool(entry.get("sha256") or entry.get("git_blob_sha1"))
    verification = ("publisher_checksum_and_full_field_scan" if layout else "publisher_checksum") if has_publisher_checksum else (
        "complete_byte_count_full_field_scan_local_sha256" if layout else "complete_byte_count_local_sha256")
    return {**entry, "sha256": sha, "mtime_ns": after.st_mtime_ns, "layout": layout,
            "verification": verification}


def file_receipt_path(root, entry):
    name = hashlib.sha256(entry["path"].encode()).hexdigest() + ".json"
    return Path(root) / ".verification" / name


def _cached_entry_ok(path, entry, record):
    if not path.exists():
        return False
    stat = path.stat()
    checks = (bool(re.fullmatch(r"[0-9a-f]{64}", record.get("sha256") or "")),
              record.get("path") == entry["path"], record.get("split") == entry["split"],
              record.get("size_bytes") == entry["size_bytes"] == stat.st_size,
              record.get("mtime_ns") == stat.st_mtime_ns,
              record.get("git_blob_sha1") == entry.get("git_blob_sha1"),
              not entry.get("sha256") or record.get("sha256") == entry["sha256"])
    if not all(checks):
        return False
    if entry["split"]:
        layout = record.get("layout") or {}
        fields = layout.get("fields", [])
        if (not fields or record.get("verification") not in ("publisher_checksum_and_full_field_scan",
                                                            "complete_byte_count_full_field_scan_local_sha256")
                or layout.get("finite_values_scanned") != {
                    f["key"]: int(np.prod(f["shape"])) for f in fields}):
            return False
    return True


def download_file(url, partial, expected_size, retries=10, etag=None):
    """Bounded retries; curl verifies resumed HTTP ranges and never marks complete."""
    partial = Path(partial)
    if retries < 0:
        raise ValueError("retries must be nonnegative")
    if partial.with_name(partial.name + ".aria2").exists():
        raise ValueError("Resume this file with aria2; its preallocated size is not completed progress")
    partial.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries + 1):
        size = partial.stat().st_size if partial.exists() else 0
        if size > expected_size:
            raise ValueError(f"Partial file exceeds publisher byte count: {partial}")
        if size == expected_size and partial.exists():
            return
        command = ["curl", "-4", "--http1.1", "--fail", "--location", "--connect-timeout", "30",
                   "--speed-limit", "1024", "--speed-time", "120", "--continue-at", "-",
                   "--output", str(partial), url]
        if etag:
            command[1:1] = ["--header", "If-Match: " + etag]
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            if partial.stat().st_size != expected_size:
                raise ValueError(f"Transfer ended with unexpected byte count: {partial}")
            return
        if result.returncode not in {5, 6, 7, 18, 22, 28, 35, 52, 55, 56, 92} or attempt == retries:
            raise subprocess.CalledProcessError(result.returncode, command)
        print(f"resume_retry={attempt + 1}/{retries} curl_exit={result.returncode} path={partial}", flush=True)
        time.sleep(min(10, 2 * (attempt + 1)))


def materialize_file(root, manifest, entry, endpoint, retries=10, verify_only=False, rehash=False):
    root = Path(root)
    target = safe_path(root, entry["path"])
    cached_path = file_receipt_path(root, entry)
    if target.exists() and cached_path.exists() and not rehash:
        cached = json.loads(cached_path.read_text(encoding="utf-8"))
        if cached.get("manifest_sha256") == canonical_hash(manifest) and _cached_entry_ok(target, entry, cached["file"]):
            return cached["file"]
    if target.exists():
        candidate = target
    elif verify_only:
        raise FileNotFoundError(target)
    else:
        candidate = target.with_name(target.name + ".part")
        url = entry.get("url") or (f"{_endpoint(endpoint)}/datasets/{manifest['repo_id']}/resolve/{manifest['revision']}/"
                                   f"{urllib.parse.quote(entry['path'], safe='/')}")
        if manifest.get("provider") == "flatiron":
            current = head_fingerprint(url)
            if any(current[key] != entry.get(key) for key in current):
                raise ValueError("Publisher file changed since registry planning; use a new manifest/directory")
        download_file(url, candidate, entry["size_bytes"], retries, entry.get("http_etag"))
    record = verify_file(candidate, entry, manifest["dataset"])
    if candidate != target:
        if target.exists():
            raise FileExistsError(f"Target appeared during transfer; preserve both: {target}")
        candidate.rename(target)
    atomic_json(cached_path, {"manifest_sha256": canonical_hash(manifest), "file": record})
    return record


def finalize_receipt(root, manifest, records):
    root = Path(root)
    expected = {f["path"]: f for f in manifest["files"]}
    if len(records) != len(expected) or {f["path"] for f in records} != set(expected):
        raise ValueError("Cannot mark a partial source complete")
    for record in records:
        if not _cached_entry_ok(safe_path(root, record["path"]), expected[record["path"]], record):
            raise ValueError(f"Stale/incomplete verification: {record['path']}")
    split_counts, trajectory_counts = Counter(), Counter()
    fingerprints = {}
    for record in records:
        if record["split"]:
            split_counts[record["split"]] += 1
            trajectory_counts[record["split"]] += record["layout"]["n_trajectories"]
            for index, digest in enumerate(record["layout"]["trajectory_sha256"]):
                previous = fingerprints.get(digest)
                if previous is not None and previous[0] != record["split"]:
                    raise ValueError(f"Exact complete trajectory overlaps official splits: {previous} and "
                                     f"{record['split']}/{record['path']}:{index}; source remains unapproved")
                fingerprints[digest] = (record["split"], record["path"], index)
    verification = ("all_manifest_files_publisher_checksums_full_field_finite_scans" if manifest.get("provider", "huggingface") == "huggingface"
                    else "all_official_registry_files_byte_counts_full_field_finite_scans_local_sha256_no_publisher_checksum")
    receipt = {"schema": RECEIPT_SCHEMA, "provider": manifest.get("provider", "huggingface"),
               "dataset": manifest["dataset"], "repo_id": manifest["repo_id"],
               "revision": manifest["revision"], "manifest_sha256": canonical_hash(manifest),
               "verification": verification,
               "split_file_counts": dict(split_counts), "split_trajectory_counts": dict(trajectory_counts),
               "files": sorted(records, key=lambda f: f["path"])}
    atomic_json(root / RECEIPT_NAME, receipt)
    return receipt


def verify_dataset_receipt(root, dataset_name=None, *, rehash=False):
    """Validate ALL listed sources; rehash=False reuses full verification stat cache."""
    root = Path(root)
    manifest = validate_manifest(json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8")))
    receipt = json.loads((root / RECEIPT_NAME).read_text(encoding="utf-8"))
    if (receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("manifest_sha256") != canonical_hash(manifest)
            or receipt.get("dataset") != manifest["dataset"] or receipt.get("revision") != manifest["revision"]
            or receipt.get("repo_id") != manifest["repo_id"]
            or (dataset_name is not None and receipt["dataset"] != dataset_name)):
        raise ValueError("Complete source receipt differs from the pinned manifest")
    records = {r["path"]: r for r in receipt["files"]}
    if len(records) != len(receipt["files"]) or set(records) != {r["path"] for r in manifest["files"]}:
        raise ValueError("Complete source receipt is missing manifest files")
    for entry in manifest["files"]:
        path, record = safe_path(root, entry["path"]), records[entry["path"]]
        if not _cached_entry_ok(path, entry, record):
            raise ValueError(f"Source changed/missing; rerun complete verification: {path}")
        if rehash and hash_file(path)[0] != record["sha256"]:
            raise ValueError(f"Source SHA256 differs from verified receipt: {path}")
    return receipt


verify_well_receipt = verify_dataset_receipt
