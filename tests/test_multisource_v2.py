"""Executable-query integration on tiny fixtures, not remote model readiness."""
from __future__ import annotations

import copy
import json
import random
from collections import Counter, defaultdict

import pytest

from test_multisource_readout import config  # Reuse complete-source fixture/receipts.
from tensor_compression.downstream import multisource_readout_data as multi
from tensor_compression.downstream.field_query_program import PROGRAM_TASKS, TEMPLATE_SPLITS
from tensor_compression.downstream.point_readout_data import PointReadoutDataset
from tensor_compression.downstream.the_well_source import atomic_json, hash_file


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def v2_config(config):
    result = copy.deepcopy(config)
    result["data"]["sources"] = [source for source in result["data"]["sources"]
                                  if source["id"] != "euler_multi_quadrants_periodicBC"]
    result["generation"].update(
        tasks=list(PROGRAM_TASKS), train_states=144, eval_states=160,
        train_states_by_source={"ins": 24, "acoustic_scattering_inclusions": 32,
                                "euler_multi_quadrants_openBC": 64, "gray_scott_reaction_diffusion": 24},
        query_program={"enabled": True, "tasks_per_state": 11, "diagnostic_states_per_source": 4,
                       "read_count_max": 8, "line_count_max": 5, "region_value_max_cells": 8,
                       "statistic_max_cells": 16,
                       "capability_weights": {"readout": 4, "localization": 3, "aggregation": 2, "composition": 1}},
    )
    return result


def test_explicit_budgets_complete_inventory_and_paired_language(v2_config, tmp_path):
    original = copy.deepcopy(v2_config)
    directory = tmp_path / "queries"
    metadata = multi.build_multisource_dataset(v2_config, directory)
    assert v2_config == original
    assert metadata["source_state_counts"]["train"] == v2_config["generation"]["train_states_by_source"]
    assert set(metadata["source_state_counts"]["val"].values()) == {40}
    assert set(metadata["source_state_counts"]["test"].values()) == {40}
    assert metadata["state_counts"] == {"train": 144, "val": 160, "test": 160}
    assert len(metadata["source"]["files"]) == 10  # Complete three-split inventory plus INS.
    assert metadata["leakage_audit"]["well_full_trajectory_hashes_checked"] == 36
    assert len(set(sum(metadata["trajectory_splits"].values(), []))) == 56
    assert metadata["diagnostics"]["adds_training_states"] is False
    states = rows(directory / "states.jsonl")
    assert len(states) == 464 and len({state["state_ref"] for state in states}) == 464
    assert all(state["roughness"] >= 0 for state in states)
    for key, values in metadata["coverage"].items():
        split, source = key.split("/", 1)
        assert values["sampled_crops"] == metadata["source_state_counts"][split][source]
        assert 1 <= values["field_frames"] <= values["sampled_crops"]
    for split in ("train", "val", "test"):
        main = PointReadoutDataset(directory, None, split, cache_size=2)
        assert main.audit()["questions"] == metadata["counts"][split]
        assert all(row["language_group"] == "seen" for row in main.records)
        assert {row["template_family"] for row in main.records} <= set(TEMPLATE_SPLITS["train"])
        assert len({row["semantic_id"] for row in main.records}) == len(main.records)
        per_state = Counter(row["state_ref"] for row in main.records)
        assert max(per_state.values()) <= 11
        assert metadata["query_budget_shortfall"][split] == metadata["state_counts"][split] * 11 - len(main)
        assert sum(metadata["capability_question_counts"][split].values()) == len(main)
        for row in main.records:
            expected = .02 if row["task_type"] == "threshold_fraction" else .2
            assert row["oracle"]["absolute_tolerance"] == expected
        if split != "train":
            language = PointReadoutDataset(directory, None, split, cache_size=2, suite="language")
            assert language.identity == main.identity
            assert language.audit()["states"] == 16
            assert len(language) == metadata["diagnostics"]["counts"][split]
            pairs = defaultdict(list)
            for row in language.records:
                pairs[row["semantic_id"]].append(row)
            for pair in pairs.values():
                assert len(pair) == 2 and pair[0]["oracle"] == pair[1]["oracle"]
                assert pair[0]["question"] != pair[1]["question"]
                assert pair[0]["answer"] == pair[1]["answer"]
                assert {row["language_group"] for row in pair} == {"seen", "heldout"}
            assert {row["template_family"] for row in language.records if row["language_group"] == "heldout"} <= set(TEMPLATE_SPLITS[split])
            language.close()
        main.close()
    with pytest.raises(ValueError, match="val/test"):
        multi.MultiSourcePointReadoutDataset(directory, "train", suite="language")


def test_deterministic_build_and_crop_stream_independent_of_query_budget(v2_config, tmp_path):
    first = multi.build_multisource_dataset(v2_config, tmp_path / "a")
    second = multi.build_multisource_dataset(v2_config, tmp_path / "b")
    assert first == second
    alternative = copy.deepcopy(v2_config)
    alternative["generation"]["query_program"]["tasks_per_state"] = 5
    third = multi.build_multisource_dataset(alternative, tmp_path / "c")
    assert first["files"]["states.jsonl"] == third["files"]["states.jsonl"]
    assert first["files"]["qa_fields.hdf5"] == third["files"]["qa_fields.hdf5"]
    assert first["trajectory_splits"] == third["trajectory_splits"]
    assert third["counts"]["train"] < first["counts"]["train"]


@pytest.mark.parametrize("case", ["missing", "unknown", "sum", "zero", "fraction"])
def test_explicit_source_budget_rejected_without_partial_build(v2_config, tmp_path, case):
    budget = v2_config["generation"]["train_states_by_source"]
    if case == "missing":
        budget.pop("ins")
    elif case == "unknown":
        budget["other"] = budget.pop("ins")
    elif case == "sum":
        budget["ins"] += 1
    elif case == "zero":
        budget["euler_multi_quadrants_openBC"] += budget["ins"]
        budget["ins"] = 0
    else:
        budget["ins"] = float(budget["ins"])
    with pytest.raises(ValueError, match="train_states_by_source"):
        multi.build_multisource_dataset(v2_config, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_language_files_and_partitions_checked_even_with_rehashed_manifest(v2_config, tmp_path):
    directory = tmp_path / "queries"
    metadata = multi.build_multisource_dataset(v2_config, directory)
    data = rows(directory / "train.jsonl")
    data[0]["language_group"] = "heldout"
    (directory / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in data), encoding="utf-8")
    metadata["files"]["train.jsonl"] = hash_file(directory / "train.jsonl")[0]
    atomic_json(directory / "metadata.json", metadata)
    with pytest.raises(ValueError, match="language template partition"):
        multi.MultiSourcePointReadoutDataset(directory, "train")
    (directory / "diagnostics_test.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset file changed"):
        multi.MultiSourcePointReadoutDataset(directory, "val", suite="language")


def test_trajectory_and_time_cycles_cover_before_repeating():
    field = {"label": "p", "sample_varying": True, "time_varying": True}
    entry = {"selected_fields": [field], "trajectory_partition": {"train": [0, 1, 2]},
             "layout": {"time_steps": 5}}
    sampler = multi._TrajectoryCycle({"file": entry}, "train", random.Random(7))
    samples = [sampler.next("p") for _ in range(15)]
    assert len({value[3] for value in samples[:3]}) == 3
    by_sample = defaultdict(list)
    for _, _, _, sample, time in samples:
        by_sample[sample].append(time)
    assert all(set(times) == set(range(5)) for times in by_sample.values())


def test_no_diagnostic_files_when_explicitly_disabled(v2_config, tmp_path):
    v2_config["generation"]["query_program"]["diagnostic_states_per_source"] = 0
    metadata = multi.build_multisource_dataset(v2_config, tmp_path / "queries")
    assert not any(name.startswith("diagnostics_") for name in metadata["files"])
    with pytest.raises(ValueError, match="no paired language"):
        multi.MultiSourcePointReadoutDataset(tmp_path / "queries", "test", suite="language")
