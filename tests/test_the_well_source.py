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


@pytest.mark.parametrize("bad_count,bad_axes", [(0, False), (2, True)])
def test_verified_bytes_with_invalid_metadata_report_file_and_preserve_source(tmp_path, bad_count, bad_axes):
    path = tmp_path / "tail.hdf5.part"
    make_hdf5(path)
    with h5py.File(path, "r+") as f:
        f.attrs["n_trajectories"] = bad_count
        if bad_axes:
            f["dimensions"].attrs["spatial_dims"] = ["x", "x"]
    original = path.read_bytes()
    entry = {"path": "data/test/tail.hdf5", "split": "test", "size_bytes": len(original),
             "sha256": hashlib.sha256(original).hexdigest(), "git_blob_sha1": None}
    with pytest.raises(ValueError) as error:
        source.verify_file(path, entry, DATASET)
    message = str(error.value)
    assert str(path) in message and "publisher checksum passed" in message
    assert "n_trajectories=" in message and "spatial_dims=" in message
    assert path.read_bytes() == original
    assert not path.with_suffix("").exists()


EULER = "euler_multi_quadrants_openBC"


def make_euler_tail(path, *, offset=0, declared=-1550, momentum_count=10):
    """Tiny spatial/time axes, but the observed ten-trajectory tail and bad attr."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs.update(dataset_name=EULER, n_spatial_dims=2, grid_type="cartesian",
                       n_trajectories=declared)
        dims = f.create_group("dimensions")
        dims.attrs["spatial_dims"] = ["x", "y"]
        for name, size in (("x", 3), ("y", 4), ("time", 5)):
            dims.create_dataset(name, data=np.arange(size, dtype="float32"))
        for rank, names in ((0, ["density", "energy", "pressure"]), (1, ["momentum"])):
            group = f.create_group(f"t{rank}_fields")
            group.attrs["field_names"] = names
            for name in names:
                shape = (momentum_count if rank else 10, 5, 3, 4) + ((2,) if rank else ())
                ds = group.create_dataset(name, data=np.arange(np.prod(shape), dtype="float32").reshape(shape) + offset)
                ds.attrs.update(sample_varying=True, time_varying=True, dim_varying=[True, True])


def euler_entry(path, split="test"):
    return {"path": f"data/{split}/tail.hdf5", "split": split, "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "git_blob_sha1": None}


def test_euler_tail_resume_records_actual_count_without_changing_source(tmp_path, monkeypatch):
    siblings = []
    for index, split in enumerate(source.SPLITS):
        partial = tmp_path / f"data/{split}/tail.hdf5.part"
        make_euler_tail(partial, offset=index * 10000)
        siblings.append(metadata(f"data/{split}/tail.hdf5", path=partial))
    manifest = source.manifest_from_info(EULER, {"sha": "b" * 40, "siblings": siblings})
    source.atomic_json(tmp_path / source.MANIFEST_NAME, manifest)
    monkeypatch.setattr(source.subprocess, "run", lambda *a, **k: pytest.fail("complete partial must not transfer"))
    records = []
    for entry in manifest["files"]:
        target = tmp_path / entry["path"]
        partial = target.with_name(target.name + ".part")
        original = partial.read_bytes()
        record = source.materialize_file(tmp_path, manifest, entry, "https://hf-mirror.com")
        assert target.read_bytes() == original and not partial.exists()
        assert record["sha256"] == entry["sha256"]
        layout = record["layout"]
        assert layout["n_trajectories"] == 10
        assert len(layout["trajectory_sha256"]) == len(layout["initial_state_sha256"]) == 10
        assert layout["finite_values_scanned"] == {
            "t0_fields/density": 600, "t0_fields/energy": 600,
            "t0_fields/pressure": 600, "t1_fields/momentum": 1200}
        correction, = layout["metadata_corrections"]
        assert correction["declared"] == -1550 and correction["effective"] == 10
        assert set(correction["field_sample_counts"].values()) == {10}
        with h5py.File(target, "r") as f:
            assert f.attrs["n_trajectories"] == -1550
        records.append(record)
    receipt = source.finalize_receipt(tmp_path, manifest, records)
    assert receipt["split_trajectory_counts"] == {split: 10 for split in source.SPLITS}
    assert source.verify_dataset_receipt(tmp_path, rehash=True) == receipt
    monkeypatch.setattr(source, "verify_file", lambda *a, **k: pytest.fail("unchanged verified file must reuse receipt"))
    for entry, record in zip(manifest["files"], records):
        assert source.materialize_file(tmp_path, manifest, entry, "https://hf-mirror.com") == record


@pytest.mark.parametrize("defect,match", [
    ("inconsistent_count", "inconsistent/missing"),
    ("zero_count", "trajectory/spatial"),
    ("positive_wrong_count", "shape/attribute"),
    ("missing_field", "inconsistent/missing"),
    ("nonfinite_last_value", "Nonfinite"),
    ("bad_axes", "trajectory/spatial"),
    ("wrong_publisher_checksum", "Publisher SHA256 mismatch"),
    ("no_publisher_checksum", "trajectory/spatial"),
    ("other_dataset", "trajectory/spatial"),
])
def test_euler_negative_count_exception_keeps_integrity_checks(tmp_path, defect, match):
    path = tmp_path / "tail.hdf5.part"
    make_euler_tail(path, momentum_count=9 if defect == "inconsistent_count" else 10)
    name = EULER
    with h5py.File(path, "r+") as f:
        if defect == "zero_count":
            f.attrs["n_trajectories"] = 0
        elif defect == "positive_wrong_count":
            f.attrs["n_trajectories"] = 11
        elif defect == "missing_field":
            del f["t0_fields/energy"]
        elif defect == "nonfinite_last_value":
            f["t1_fields/momentum"][-1, -1, -1, -1, -1] = np.nan
        elif defect == "bad_axes":
            f["dimensions"].attrs["spatial_dims"] = ["x", "x"]
        elif defect == "other_dataset":
            name = DATASET
            f.attrs["dataset_name"] = name
    entry = euler_entry(path)
    if defect == "wrong_publisher_checksum":
        entry["sha256"] = "0" * 64
    elif defect == "no_publisher_checksum":
        entry["sha256"] = None
    original = path.read_bytes()
    with pytest.raises(ValueError, match=match):
        source.verify_file(path, entry, name)
    assert path.read_bytes() == original
    assert not path.with_suffix("").exists()


def test_unverified_euler_inspection_does_not_override_negative_attribute(tmp_path):
    path = tmp_path / "tail.hdf5"
    make_euler_tail(path)
    with pytest.raises(ValueError, match="trajectory/spatial"):
        source.inspect_hdf5(path, EULER)


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
