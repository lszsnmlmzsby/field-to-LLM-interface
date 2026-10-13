from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(directory))

from tensor_compression.downstream.point_readout import normalize_field, make_record
from tensor_compression.downstream.point_readout_data import build_dataset, load_config, PointReadoutDataset, source_path
from tensor_compression.downstream.random_readout_shapes import shape_pools, fitting_tasks
from tensor_compression.downstream.ins_tracer_source import inspect_complete


def test_integer_domain_includes_thin_odd_and_large_shapes():
    policy = {"height": [1, 128], "width": [1, 128], "seed": 42, "heldout_fraction": .2}
    train, heldout, anchors = shape_pools(policy, (128, 128))
    assert len(train) + len(heldout) == 128 ** 2
    assert not set(train) & set(heldout)
    assert set(anchors) == {(1, 1), (1, 128), (128, 1), (128, 128)}
    assert all(s in train or s in heldout for s in [(3, 127), (127, 3), (17, 90), (111, 113)])
    assert shape_pools(policy, (128, 128)) == (train, heldout, anchors)
    with pytest.raises(ValueError, match="source extent"):
        shape_pools(policy, (64, 128))


def test_random_shape_dataset_replays_every_question_and_preserves_splits(tmp_path):
    config = load_config(ROOT / "configs/field_to_llm_ins_tracer.yaml", "full")
    config["data"]["require_complete_ins_tracer"] = False  # Synthetic unit-test fixture only.
    config["data"]["shape_sampling"].update(height=[1, 13], width=[1, 19])
    config["generation"].update(train_states=96, eval_states=36)
    source = tmp_path / "fixture.h5"
    with h5py.File(source, "w") as handle:
        for key in config["data"]["fields"]:
            handle[key] = np.random.default_rng(len(key)).normal(size=(20, 3, 13, 19)).astype("float32")
    qa = tmp_path / "qa"
    metadata = build_dataset(config, source, qa)
    repeated = build_dataset(config, source, tmp_path / "repeat")
    assert repeated == metadata
    assert metadata["state_counts"] == {"train": 96, "val": 36, "test": 36}
    assert len(metadata["train_shapes"]) > 20
    assert not set(map(tuple, metadata["train_shapes"])) & set(map(tuple, metadata["heldout_shapes"]))
    assert metadata["inapplicable_tasks"]
    parts = list(map(set, metadata["trajectory_splits"].values()))
    assert all(not a & b for i, a in enumerate(parts) for b in parts[i + 1:])
    for split in ("train", "val", "test"):
        dataset = PointReadoutDataset(qa, source, split)
        try:
            assert dataset.audit()["questions"] == metadata["counts"][split]
            shapes = {tuple(row["grid_shape"]) for row in dataset.records}
            assert {(1, 1), (1, 19), (13, 1), (13, 19)} <= shapes
            if split != "train":
                assert shapes & set(map(tuple, metadata["heldout_shapes"]))
            for row in dataset.records:
                if row["grid_shape"] == [1, 1]:
                    assert row["task_type"] not in ("line_profile", "multi_point")
        finally:
            dataset.close()


def test_singleton_is_explicit_and_other_constant_fields_still_rejected():
    with pytest.raises(ValueError):
        normalize_field(torch.ones(1, 1))
    z, _ = normalize_field(torch.ones(1, 1), allow_singleton=True)
    assert torch.equal(z, torch.zeros(1, 1))
    with pytest.raises(ValueError):
        normalize_field(torch.ones(1, 2), allow_singleton=True)
    with pytest.raises(ValueError):
        normalize_field(torch.tensor([[float("nan")]]), allow_singleton=True)


def test_tiny_regions_are_only_fallbacks():
    args = (["single_point", "multi_point", "line_profile", "region_mean"], (5, 7),
            [2, 3, 4], [2, 3, 4], [(1, 1), (1, 2), (2, 2)], [(1, 1), (2, 2)])
    _, _, _, regions, stats = fitting_tasks(*args)
    assert (1, 1) not in regions and (1, 1) not in stats
    active, _, _, _, stats = fitting_tasks(args[0], (1, 1), *args[2:])
    assert active == ["single_point", "region_mean"] and stats == [(1, 1)]


def test_ins_path_cannot_be_overridden_by_old_pdebench_environment(monkeypatch, tmp_path):
    config = load_config(ROOT / "configs/field_to_llm_ins_tracer.yaml", "full")
    monkeypatch.setenv("PDEBENCH_HDF5", str(tmp_path / "old.h5"))
    monkeypatch.setenv("FIELD_TO_LLM_ROOT", str(tmp_path))
    monkeypatch.delenv("INS_TRACER_HDF5", raising=False)
    assert source_path(config) != tmp_path / "old.h5"
    monkeypatch.setenv("INS_TRACER_HDF5", str(tmp_path / "new.h5"))
    assert source_path(config) == tmp_path / "new.h5"


def test_complete_validation_catches_nan_and_truncation(tmp_path):
    source = tmp_path / "complete.h5"
    shape = [2, 3, 5, 7]
    with h5py.File(source, "w") as handle:
        for field in ("u", "v", "b"):
            handle[f"sol/{field}"] = np.ones(shape, dtype="float32")
        handle["args/num_pde"] = 2
        for key, size in (("t", 3), ("x", 5), ("y", 7)):
            handle[f"coord/{key}"] = np.arange(size, dtype=float)
    manifest = {"url": "fixture", "bytes": source.stat().st_size,
                "fields": {f"sol/{k}": shape for k in ("u", "v", "b")}}
    receipt = inspect_complete(source, manifest)
    assert receipt["publisher_checksum"] is None
    assert receipt["finite_values_scanned"]["sol/b"] == 210
    with h5py.File(source, "r+") as handle:
        handle["sol/b"][1, 2, 4, 6] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        inspect_complete(source, manifest)
    with source.open("r+b") as handle:
        handle.truncate(source.stat().st_size - 1)
    with pytest.raises(ValueError, match="size"):
        inspect_complete(source, manifest)


def test_production_encoder_handles_one_cell_with_group_norm():
    from scripts import train_point_readout as trainer
    config = load_config(ROOT / "configs/field_to_llm_ins_tracer.yaml", "smoke")
    encoder, _, _, _ = trainer.core.build_scratch_memory_components(trainer.model_args(config), (1, 16, 16))
    output = encoder(torch.zeros(1, 1, 1, 1))
    assert output.shape == (1, 8, 1, 1)
    output.sum().backward()
    assert torch.isfinite(output).all()


def test_geometry_check_is_not_training_and_covers_extremes():
    from test_point_readout import tiny_components
    config = load_config(ROOT / "configs/field_to_llm_ins_tracer.yaml", "smoke")
    config["field_encoder"]["model"]["norm"] = "identity"
    trainer, llm, sidecar, tokenizer, config = tiny_components(config)
    config["training"]["batch_size"] = 3
    class Data:
        def __init__(self):
            self.fields = [normalize_field(torch.arange(h*w).reshape(h, w).float(), allow_singleton=True)[0]
                           for h, w in [(1, 1), (1, 13), (9, 1), (9, 13)]]
            self.records = [make_record(str(i), z, "single_point", {"points": [[1, 1]]})
                            for i, z in enumerate(self.fields)]
        def __getitem__(self, i):
            return self.records[i], self.fields[i][None]
    before = {k: v.clone() for k, v in sidecar.state_dict().items()}
    report = trainer.check_geometry(llm, sidecar, tokenizer, Data(), torch.device("cpu"), torch.float32, config)
    assert report["optimizer_updates"] == 0
    assert all(row["batch_size"] == 3 for row in report["checks"])
    assert {tuple(row["shape"]) for row in report["checks"]} == {(1, 1), (1, 13), (9, 1), (9, 13)}
    assert all(torch.equal(v, sidecar.state_dict()[k]) for k, v in before.items())
    assert all(p.grad is None for p in sidecar.parameters())


def test_download_resume_after_network_error_and_aria2_guard(tmp_path, monkeypatch):
    from scripts import download_ins_tracer as downloader
    path = tmp_path / "source.part"
    path.write_bytes(b"first")
    observed = []
    def transfer(command, check):
        observed.append(path.read_bytes())
        assert "--continue-at" in command and "--http1.1" in command
        with path.open("ab") as handle:
            handle.write(b"next")
        if len(observed) == 1:
            raise subprocess.CalledProcessError(92, command)
    monkeypatch.setattr(downloader.subprocess, "run", transfer)
    monkeypatch.setattr(downloader.time, "sleep", lambda seconds: None)
    downloader.download_partial(path, 1)
    assert observed == [b"first", b"firstnext"]
    path.with_name(path.name + ".aria2").write_bytes(b"control")
    with pytest.raises(ValueError, match="aria2"):
        downloader.download_partial(path, 1)
    assert len(observed) == 2


def test_full_size_aria2_partial_is_not_mistaken_for_complete(tmp_path, monkeypatch):
    from scripts import download_ins_tracer as downloader
    partial = tmp_path / (downloader.FILENAME + ".part")
    partial.write_bytes(b"allocated")
    partial.with_name(partial.name + ".aria2").write_bytes(b"unfinished ranges")
    monkeypatch.setattr(downloader, "SOURCE", {**downloader.SOURCE, "bytes": partial.stat().st_size})
    monkeypatch.setattr(sys, "argv", ["download_ins_tracer.py", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(downloader, "inspect_complete", lambda *a: pytest.fail("Must preserve unfinished aria2 file"))
    with pytest.raises(ValueError, match="aria2"):
        downloader.main()
    assert partial.read_bytes() == b"allocated"
    assert not (tmp_path / downloader.FILENAME).exists()


@pytest.mark.parametrize("shape", [(1, 1), (1, 19), (13, 1), (3, 17)])
@pytest.mark.parametrize("block", [(1, 1), (2, 2)])
def test_positive_shapes_forward_backward_on_tiny_qwen(shape, block):
    from test_point_readout import tiny_components
    config = load_config(ROOT / "configs/field_to_llm_ins_tracer.yaml", "smoke")
    config["memory"]["block_shape"] = list(block)
    # tiny_components uses 8 base channels: choose identity to avoid 1 value/group.
    # The production config has 32 base channels, i.e. 4 values/group even at 1x1.
    config["field_encoder"]["model"]["norm"] = "identity"
    trainer, llm, sidecar, tokenizer, config = tiny_components(config)
    z, _ = normalize_field(torch.arange(shape[0] * shape[1]).reshape(shape).float(), allow_singleton=True)
    row = make_record("boundary", z, "single_point", {"points": [[shape[0], shape[1]]]})
    loss, _ = trainer.training_loss(llm, sidecar, tokenizer, [row], z[None, None],
                                    torch.device("cpu"), torch.float32, config["training"])
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in sidecar.parameters())
    sidecar.clear()
