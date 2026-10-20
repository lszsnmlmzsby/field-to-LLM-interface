"""Tiny synthetic HDF5 integration fixtures, not evidence of remote Qwen readiness."""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(item))

from tensor_compression.downstream import multisource_readout_data as multi
from tensor_compression.downstream import ins_tracer_source
from tensor_compression.downstream.point_readout import TASKS, normalize_field
from tensor_compression.downstream.point_readout_data import PointReadoutDataset, build_dataset
from tensor_compression.downstream.the_well_source import (
    DATASETS, MANIFEST_NAME, atomic_json, finalize_receipt, hash_file, manifest_from_info, verify_file,
)


def write_well(path, dataset, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        f.attrs.update(dataset_name=dataset, n_spatial_dims=2, grid_type="cartesian", n_trajectories=4)
        dims = f.create_group("dimensions")
        dims.attrs["spatial_dims"] = ["x", "y"]
        for name, length in (("x", 4), ("y", 5), ("time", 3)):
            ds = dims.create_dataset(name, data=np.arange(length, dtype="float32"))
            ds.attrs.update(sample_varying=False, time_varying=False)
        for group, fields in (("t0_fields", ["pressure", "density", "A", "B"]), ("t1_fields", ["velocity"])):
            g = f.create_group(group)
            g.attrs["field_names"] = fields
            for name in fields:
                dynamic = name != "density"
                shape = (4, 3, 4, 5) if dynamic else (4, 4, 5)
                if name == "velocity":
                    shape += (2,)
                values = rng.normal(size=shape).astype("float32")
                if name == "density":
                    values[:] = np.arange(4)[:, None, None] + seed
                ds = g.create_dataset(name, data=values)
                ds.attrs.update(sample_varying=True, time_varying=dynamic, dim_varying=[True, True])


@pytest.fixture
def config(tmp_path, monkeypatch):
    sources = []
    for number, dataset in enumerate(DATASETS):
        root = tmp_path / dataset
        siblings = []
        for split_i, split in enumerate(("train", "valid", "test")):
            rel = f"data/{split}/fixture.hdf5"
            write_well(root / rel, dataset, seed=1000 + number * 10 + split_i)
            sha, _ = hash_file(root / rel)
            siblings.append({"rfilename": rel, "lfs": {"sha256": sha, "size": (root / rel).stat().st_size}})
        manifest = manifest_from_info(dataset, {"sha": "a" * 40, "siblings": siblings})
        atomic_json(root / MANIFEST_NAME, manifest)
        records = [verify_file(root / entry["path"], entry, dataset) for entry in manifest["files"]]
        finalize_receipt(root, manifest, records)
        fields = ["t0_fields/pressure", "t1_fields/velocity"]
        if number == 0:
            fields += ["t0_fields/density"]
        if number == 3:
            fields = ["t0_fields/A", "t0_fields/B"]
        sources.append({"id": dataset, "family": "euler" if number in (1, 2) else "acoustic" if number == 0 else "gray_scott",
                        "kind": "the_well", "dataset": dataset, "path": str(root), "fields": fields})
    ins_path = tmp_path / "ins.hdf5"
    with h5py.File(ins_path, "w") as f:
        for index, key in enumerate(("sol/u", "sol/v", "sol/b")):
            f.create_dataset(key, data=np.random.default_rng(index).normal(size=(20, 3, 4, 5)).astype("float32"))
    receipt = {"source": {"fields": {key: [20, 3, 4, 5] for key in ("sol/u", "sol/v", "sol/b")}},
               "sha256": hash_file(ins_path)[0], "verification": "tiny_test_fixture_only"}
    monkeypatch.setattr(ins_tracer_source, "verify_receipt", lambda path: receipt)
    sources.insert(0, {"id": "ins", "family": "ins", "kind": "ins_tracer", "path": str(ins_path),
                       "fields": ["sol/u", "sol/v", "sol/b"]})
    return {"data": {"sources": sources, "max_open_hdf5": 2,
                     "shape_sampling": {"height": [1, 4], "width": [1, 5], "seed": 20261020, "heldout_fraction": .2}},
            "generation": {"seed": 22, "split_seed": 20260905, "tasks": list(TASKS),
                           "train_states": 160, "eval_states": 320, "answer_decimals": 1,
                           "absolute_tolerance": .2, "max_attempts_per_state": 200}}


@pytest.mark.parametrize("include_periodic", [False, True])
def test_complete_multisource_build_replay_and_family_balance(config, tmp_path, include_periodic):
    if not include_periodic:
        config["data"]["sources"] = [s for s in config["data"]["sources"]
                                     if s["id"] != "euler_multi_quadrants_periodicBC"]
    directory = tmp_path / "qa"
    metadata = build_dataset(config, None, directory)
    assert metadata["family_state_counts"]["train"] == {"acoustic": 40, "euler": 40, "gray_scott": 40, "ins": 40}
    assert metadata["source_state_counts"]["train"]["euler_multi_quadrants_openBC"] == (20 if include_periodic else 40)
    if not include_periodic:
        assert "euler_multi_quadrants_periodicBC" not in metadata["source"]["sources"]
        assert metadata["source_state_counts"]["train"]["euler_multi_quadrants_openBC"] == metadata["source_state_counts"]["train"]["ins"]
    assert metadata["static_state_counts"]["train"] > 0
    assert metadata["constant_state_counts"]["train"] > 0
    assert metadata["leakage_audit"]["well_full_trajectory_hashes_checked"] == (48 if include_periodic else 36)
    observed_shapes = set()
    for split in ("train", "val", "test"):
        dataset = PointReadoutDataset(directory, None, split, cache_size=2)
        assert isinstance(dataset, multi.MultiSourcePointReadoutDataset)
        audit = dataset.audit()
        assert audit["questions"] == metadata["counts"][split]
        assert len(dataset.readers.handles) <= 2
        observed_shapes.update(tuple(r["grid_shape"]) for r in dataset.records)
        static = [s for s in dataset.states.values() if s["field"] == "t0_fields/density"]
        assert static and all(s["time_index"] is None for s in static)
        # Ordered vector component replay matches direct HDF indexing + normalizer.
        state = next(s for s in dataset.states.values() if s["field"] == "t1_fields/velocity[1]" and s["split"] == split)
        file_entry = metadata["source"]["files"][state["source_file_id"]]
        source = metadata["source"]["sources"][state["source_id"]]
        r, c = state["top_left"]
        h, w = state["grid_shape"]
        with h5py.File(Path(source["path_hint"]) / file_entry["path"], "r") as f:
            raw = torch.from_numpy(f["t1_fields/velocity"][state["sample_index"], state["time_index"], r:r+h, c:c+w, 1])
        expected, _ = normalize_field(raw, allow_singleton=True, allow_constant=True)
        assert torch.equal(dataset.field(state["state_ref"]), expected)
        dataset.close()
    assert (1, 5) in observed_shapes and (4, 1) in observed_shapes
    assert not set(map(tuple, metadata["train_shapes"])) & set(map(tuple, metadata["heldout_shapes"]))


def test_build_is_deterministic_and_qa_content_immutable(config, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    first = multi.build_multisource_dataset(config, a)
    second = multi.build_multisource_dataset(config, b)
    assert first == second
    with (a / "train.jsonl").open("a", encoding="utf-8") as f:
        f.write("{}\n")
    with pytest.raises(ValueError, match="Dataset file changed"):
        multi.MultiSourcePointReadoutDataset(a, "train")


def test_missing_fields_and_out_of_source_bounds_fail_before_output(config, tmp_path):
    bad = copy.deepcopy(config)
    bad["data"]["sources"][1]["fields"].append("t0_fields/nonexistent")
    with pytest.raises(ValueError, match="Missing requested fields"):
        multi.build_multisource_dataset(bad, tmp_path / "missing")
    assert not (tmp_path / "missing").exists()
    bad = copy.deepcopy(config)
    bad["data"]["shape_sampling"]["width"] = [1, 6]
    with pytest.raises(ValueError, match="source extent"):
        multi.build_multisource_dataset(bad, tmp_path / "outside")


def test_cross_source_exact_trajectory_leak_rejected(config, tmp_path, monkeypatch):
    original = multi._well_receipt
    seen = {}
    def receipt(path, dataset):
        value = copy.deepcopy(original(path, dataset))
        for entry in value["files"]:
            if not entry.get("layout"):
                continue
            if dataset == DATASETS[0] and entry["split"] == "train":
                seen["hash"] = entry["layout"]["trajectory_sha256"][0]
            if dataset == DATASETS[1] and entry["split"] == "test":
                entry["layout"]["trajectory_sha256"][0] = seen["hash"]
        return value
    monkeypatch.setattr(multi, "_well_receipt", receipt)
    with pytest.raises(ValueError, match="Identical full trajectory leaks"):
        multi.build_multisource_dataset(config, tmp_path / "leaky")


def test_labels_can_load_without_sources_but_replay_rejects_changed_source(config, tmp_path):
    directory = tmp_path / "qa"
    multi.build_multisource_dataset(config, directory)
    dataset = multi.MultiSourcePointReadoutDataset(directory, "train")
    state = dataset.states[dataset.records[0]["state_ref"]]
    entry = dataset.metadata["source"]["files"][state["source_file_id"]]
    source = dataset.metadata["source"]["sources"][state["source_id"]]
    path = multi._file_path(source, entry)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    # Prediction scoring only needs the immutable QA labels; it must not read TB of source data.
    labels = multi.MultiSourcePointReadoutDataset(directory, "train")
    assert len(labels) > 0
    with pytest.raises(ValueError, match="Source file changed"):
        dataset.field(state["state_ref"])
    labels.close()
    dataset.close()


def test_complete_receipt_required(config, tmp_path):
    source = config["data"]["sources"][1]
    (Path(source["path"]) / "source_verified.json").unlink()
    with pytest.raises(FileNotFoundError):
        multi.build_multisource_dataset(config, tmp_path / "missing_receipt")


def test_cache_exactness_is_independent_of_manifest_checksum(config, tmp_path):
    directory = tmp_path / "qa"
    metadata = multi.build_multisource_dataset(config, directory)
    state = json.loads((directory / "states.jsonl").read_text().splitlines()[0])
    with h5py.File(directory / "qa_fields.hdf5", "r+") as f:
        f["values"][state["cache_offset"]] += 1
    # A regenerated outer checksum must not hide disagreement with the oracle's field.
    metadata["files"]["qa_fields.hdf5"] = hash_file(directory / "qa_fields.hdf5")[0]
    atomic_json(directory / "metadata.json", metadata)
    dataset = multi.MultiSourcePointReadoutDataset(directory, "train")
    with pytest.raises(ValueError, match="QA field cache changed"):
        dataset.field(state["state_ref"])
    dataset.close()


def test_original_source_audit_is_not_replaced_by_qa_cache(config, tmp_path):
    directory = tmp_path / "qa"
    metadata = multi.build_multisource_dataset(config, directory)
    state = json.loads((directory / "states.jsonl").read_text().splitlines()[0])
    entry = metadata["source"]["files"][state["source_file_id"]]
    source = metadata["source"]["sources"][state["source_id"]]
    path = multi._file_path(source, entry)
    field = next(f for f in entry["selected_fields"] if f["label"] == state["field"])
    index = ([state["sample_index"]] if field["sample_varying"] else [])
    index += [state["time_index"]] if field["time_varying"] else []
    index += state["top_left"]
    index += [field["component"]] if field["component"] is not None else []
    stat = path.stat()
    with h5py.File(path, "r+") as f:
        f[field["key"]][tuple(index)] += 100
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    dataset = multi.MultiSourcePointReadoutDataset(directory, "train")
    # Cached input remains bitwise the original value, but original-source auditing fails.
    dataset.field(state["state_ref"])
    with pytest.raises(ValueError, match="Source crop changed"):
        dataset.audit()
    dataset.close()
