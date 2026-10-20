"""Small local fixtures verify source completeness; no remote throughput claims."""
import hashlib
import json
import os

import h5py
import numpy as np
import pytest

from tensor_compression.downstream import the_well_source as source


DATASET = "acoustic_scattering_inclusions"


def make_hdf5(path, offset=0, static_shared=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs.update(dataset_name=DATASET, n_spatial_dims=2, grid_type="cartesian", n_trajectories=2)
        dims = f.create_group("dimensions")
        dims.attrs["spatial_dims"] = ["x", "y"]
        for name, size in (("x", 3), ("y", 4), ("time", 5)):
            dims.create_dataset(name, data=np.arange(size, dtype="float32"))
        scalar = f.create_group("t0_fields")
        scalar.attrs["field_names"] = ["pressure", "density", "c"]
        p = scalar.create_dataset("pressure", data=np.arange(120, dtype="float32").reshape(2, 5, 3, 4) + offset)
        p.attrs.update(sample_varying=True, time_varying=True, dim_varying=[True, True])
        dshape = (3, 4) if static_shared else (2, 3, 4)
        d = scalar.create_dataset("density", data=np.ones(dshape, dtype="float32"))
        d.attrs.update(sample_varying=not static_shared, time_varying=False, dim_varying=[True, True])
        c = scalar.create_dataset("c", data=np.ones((2, 3), dtype="float32"))
        c.attrs.update(sample_varying=True, time_varying=False, dim_varying=[True, False])
        vector = f.create_group("t1_fields")
        vector.attrs["field_names"] = ["velocity"]
        v = vector.create_dataset("velocity", data=np.ones((2, 5, 3, 4, 2), dtype="float32") + offset)
        v.attrs.update(sample_varying=True, time_varying=True, dim_varying=[True, True])


def metadata(name, content=None, path=None):
    content = path.read_bytes() if path is not None else content
    item = {"rfilename": name, "size": len(content)}
    if name.endswith(".hdf5"):
        item["lfs"] = {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
    else:
        item["blobId"] = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    return item


def complete_source(tmp_path, duplicate=False):
    siblings = []
    for index, split in enumerate(source.SPLITS):
        name = f"data/{split}/fixture.hdf5"
        path = tmp_path / name
        make_hdf5(path, offset=0 if duplicate else index * 100)
        siblings.append(metadata(name, path=path))
    readme = tmp_path / "README.md"
    readme.write_text("Fixture\n", encoding="utf-8")
    siblings.append(metadata("README.md", path=readme))
    manifest = source.manifest_from_info(DATASET, {"sha": "a" * 40, "siblings": siblings})
    source.atomic_json(tmp_path / source.MANIFEST_NAME, manifest)
    records = [source.verify_file(tmp_path / f["path"], f, DATASET) for f in manifest["files"]]
    return manifest, records


def test_manifest_pins_full_splits_and_transparently_excludes_archives():
    entries = [metadata(f"data/{split}/test.hdf5", content=split.encode()) for split in source.SPLITS]
    entries += [metadata("old_and_problematic_data/train/old.hdf5", content=b"bad"),
                metadata("README.md", content=b"source"), metadata("gif/movie.gif", content=b"gif"),
                metadata("data/stats.yaml", content=b"mean: 0")]
    manifest = source.manifest_from_info(DATASET, {"sha": "c" * 40, "siblings": entries})
    assert len(manifest["files"]) == 5
    assert manifest["revision"] == "c" * 40
    assert len(manifest["excluded_files"]) == 2
    assert all(f["sha256"] for f in manifest["files"] if f["split"])


def test_manifest_requires_all_splits_and_publisher_digest():
    entries = [metadata(f"data/{split}/test.hdf5", content=b"x") for split in source.SPLITS]
    with pytest.raises(ValueError, match="all official"):
        source.manifest_from_info(DATASET, {"sha": "a" * 40, "siblings": entries[:2]})
    entries[0].pop("lfs")
    with pytest.raises(ValueError, match="Missing publisher"):
        source.manifest_from_info(DATASET, {"sha": "a" * 40, "siblings": entries})


def test_manifest_rejects_unsafe_or_unclassified_source():
    with pytest.raises(ValueError, match="Unsafe"):
        source.manifest_from_info(DATASET, {"sha": "a" * 40,
                                          "siblings": [metadata("../escape.hdf5", content=b"x")]})
    with pytest.raises(ValueError, match="Unexpected HDF5"):
        source.manifest_from_info(DATASET, {"sha": "a" * 40,
                                          "siblings": [metadata("other/train.hdf5", content=b"x")]})


@pytest.mark.parametrize("static_shared", [False, True])
def test_hdf5_actual_dynamic_static_vector_and_reduced_axes(tmp_path, static_shared):
    path = tmp_path / "sample.hdf5"
    make_hdf5(path, static_shared=static_shared)
    layout = source.inspect_hdf5(path, DATASET)
    assert layout["spatial_shape"] == [3, 4]
    assert layout["time_steps"] == 5
    assert layout["finite_values_scanned"]["t1_fields/velocity"] == 240
    fields = {f["key"]: f for f in layout["fields"]}
    assert fields["t0_fields/density"]["sample_varying"] is not static_shared
    assert not fields["t0_fields/density"]["time_varying"]
    assert fields["t0_fields/c"]["dim_varying"] == [True, False]
    assert fields["t1_fields/velocity"]["components"] == 2
    assert len(set(layout["trajectory_sha256"])) == 2


def test_full_scan_rejects_nonfinite_last_value(tmp_path):
    path = tmp_path / "sample.hdf5"
    make_hdf5(path)
    with h5py.File(path, "r+") as f:
        f["t1_fields/velocity"][-1, -1, -1, -1, -1] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        source.inspect_hdf5(path)


def test_layout_rejects_false_time_metadata(tmp_path):
    path = tmp_path / "sample.hdf5"
    make_hdf5(path)
    with h5py.File(path, "r+") as f:
        f["t0_fields/pressure"].attrs["time_varying"] = False
    with pytest.raises(ValueError, match="shape/attribute"):
        source.inspect_hdf5(path)


def test_complete_receipt_rejects_missing_modified_files_and_rehashes(tmp_path):
    manifest, records = complete_source(tmp_path)
    with pytest.raises(ValueError, match="partial"):
        source.finalize_receipt(tmp_path, manifest, records[:-1])
    receipt = source.finalize_receipt(tmp_path, manifest, records)
    assert receipt["split_trajectory_counts"] == {"train": 2, "valid": 2, "test": 2}
    assert source.verify_dataset_receipt(tmp_path, rehash=True) == receipt
    path = tmp_path / "README.md"
    stat = path.stat()
    original = path.read_bytes()
    path.write_bytes(b"x" + original[1:])
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    # Same size and timestamp intentionally evade the documented fast stat cache.
    assert source.verify_dataset_receipt(tmp_path)
    with pytest.raises(ValueError, match="SHA256 differs"):
        source.verify_dataset_receipt(tmp_path, rehash=True)
    path.unlink()
    with pytest.raises(ValueError, match="changed/missing"):
        source.verify_dataset_receipt(tmp_path)


def test_cross_split_trajectory_duplicates_block_completion(tmp_path):
    manifest, records = complete_source(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="trajectory overlaps"):
        source.finalize_receipt(tmp_path, manifest, records)
    assert not (tmp_path / source.RECEIPT_NAME).exists()


def test_wrong_publisher_checksum_never_marks_complete(tmp_path):
    path = tmp_path / "data/train/file.hdf5"
    make_hdf5(path)
    entry = {"path": "data/train/file.hdf5", "split": "train", "size_bytes": path.stat().st_size,
             "sha256": "0" * 64, "git_blob_sha1": None}
    with pytest.raises(ValueError, match="Publisher SHA256 mismatch"):
        source.verify_file(path, entry, DATASET)


def test_resume_preserves_complete_partial_and_verifies_before_rename(tmp_path, monkeypatch):
    manifest, records = complete_source(tmp_path)
    entry = next(f for f in manifest["files"] if f["split"] == "train")
    target = tmp_path / entry["path"]
    partial = target.with_name(target.name + ".part")
    target.rename(partial)
    monkeypatch.setattr(source.subprocess, "run", lambda *a, **k: pytest.fail("complete partial should not transfer"))
    record = source.materialize_file(tmp_path, manifest, entry, "https://hf-mirror.com")
    assert target.exists() and not partial.exists()
    assert record["sha256"] == entry["sha256"]
    monkeypatch.setattr(source, "verify_file", lambda *a, **k: pytest.fail("unchanged receipt should be reused"))
    assert source.materialize_file(tmp_path, manifest, entry, "https://hf-mirror.com") == record


def test_aria2_allocated_partial_not_mistaken_for_downloaded_bytes(tmp_path):
    path = tmp_path / "file.part"
    path.write_bytes(b"x")
    path.with_name(path.name + ".aria2").write_bytes(b"control")
    with pytest.raises(ValueError, match="aria2"):
        source.download_file("https://example.test/file", path, 1)


def test_flatiron_registry_complete_and_no_invented_checksum(tmp_path):
    _, _ = complete_source(tmp_path)
    prefix = source.FLATIRON_BASE + DATASET + "/"
    registry = {DATASET: {"stats": prefix + "README.md", **{
        split: [prefix + f"data/{split}/fixture.hdf5"] for split in source.SPLITS}}}
    def head(url):
        return {"size_bytes": (tmp_path / url[len(prefix):]).stat().st_size,
                "http_etag": '"opaque-size-time"', "http_last_modified": "example"}
    manifest = source.flatiron_manifest_from_registry(DATASET, registry, "b" * 40, "c" * 64, head)
    assert manifest["provider"] == "flatiron"
    assert all(f["sha256"] is None for f in manifest["files"])
    assert manifest["checksum_limitation"]
    source.atomic_json(tmp_path / source.MANIFEST_NAME, manifest)
    records = [source.verify_file(tmp_path / f["path"], f, DATASET) for f in manifest["files"]]
    receipt = source.finalize_receipt(tmp_path, manifest, records)
    assert "no_publisher_checksum" in receipt["verification"]
    assert source.verify_dataset_receipt(tmp_path, rehash=True) == receipt


def test_flatiron_registry_rejects_unofficial_host():
    registry = {DATASET: {"stats": "https://example.test/stats.yaml", **{
        split: ["https://example.test/file.hdf5"] for split in source.SPLITS}}}
    with pytest.raises(ValueError, match="outside official"):
        source.flatiron_manifest_from_registry(DATASET, registry, "a" * 40, "b" * 64,
                                               lambda url: pytest.fail("must not request unofficial host"))


def test_resumable_transfer_uses_same_partial_after_bounded_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    target = tmp_path / "resume.part"
    calls = []
    def curl(command, check):
        calls.append(command)
        target.write_bytes(b"abc" if len(calls) == 1 else b"abcdef")
        return SimpleNamespace(returncode=18 if len(calls) == 1 else 0)
    monkeypatch.setattr(source.subprocess, "run", curl)
    monkeypatch.setattr(source.time, "sleep", lambda seconds: None)
    source.download_file("https://example.test/file", target, 6, retries=1, etag='"opaque"')
    assert len(calls) == 2 and target.read_bytes() == b"abcdef"
    assert all("--continue-at" in call and "If-Match: \"opaque\"" in call for call in calls)


@pytest.mark.parametrize("name,expected_bytes,expected_counts", [
    ("acoustic_scattering_inclusions", 291873226752, (28, 4, 4)),
    ("euler_multi_quadrants_openBC", 2647658116720, (100, 20, 20)),
    ("euler_multi_quadrants_periodicBC", 2647655177000, (10, 10, 10)),
    ("gray_scott_reaction_diffusion", 157940711424, (6, 6, 6)),
])
def test_bundled_catalogs_work_without_github_or_hf(monkeypatch, name, expected_bytes, expected_counts):
    monkeypatch.setattr(source.urllib.request, "urlopen",
                        lambda *args, **kwargs: pytest.fail("bundled planning must be offline"))
    manifest = source.fetch_manifest(name, metadata_endpoint="https://hf-mirror.com")
    assert manifest["provider"] == ("flatiron" if name.endswith("periodicBC") else "huggingface")
    assert {s: sum(f["split"] == s for f in manifest["files"]) for s in source.SPLITS} == {
        s: n for s, n in zip(source.SPLITS, expected_counts)}
    assert sum(f["size_bytes"] for f in manifest["files"] if f["split"]) == expected_bytes
    with pytest.raises(ValueError, match="registry is pinned"):
        source.fetch_manifest(name, revision="0" * 40)


def test_default_download_cli_plans_only_active_three_complete_sources(tmp_path, monkeypatch, capsys):
    import sys
    from scripts import download_the_well
    monkeypatch.setattr(source.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("pinned planning must be offline"))
    monkeypatch.setattr(sys, "argv", ["download_the_well.py", "--output-dir", str(tmp_path),
                                    "--endpoint", "https://hf-mirror.com", "--manifest-only"])
    download_the_well.main()
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert {r["dataset"] for r in rows} == {
        "acoustic_scattering_inclusions", "euler_multi_quadrants_openBC", "gray_scott_reaction_diffusion"}
    assert sum(r["source_bytes"] for r in rows) == 3097472054896
    assert not (tmp_path / "euler_multi_quadrants_periodicBC").exists()
    assert all((tmp_path / r["dataset"] / source.MANIFEST_NAME).is_file() for r in rows)
