from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensor_compression.downstream.point_readout import (
    make_record, normalize_field, score_answer, summarize_evaluation,
)
from tensor_compression.downstream.point_readout_data import load_config, source_path


def test_new_constant_policy_is_explicit_exact_and_keeps_tie_answers():
    raw = torch.full((127, 113), .1)
    z, audit = normalize_field(raw, allow_constant=True)
    assert torch.count_nonzero(z) == 0
    assert audit["std"] == 0
    with pytest.raises(ValueError, match="constant"):
        normalize_field(raw)
    record = make_record("constant", z, "region_argmax", {"start": [2, 3], "size": [2, 2]})
    assert len(record["oracle"]["valid_coordinates"]) == 4
    assert score_answer(record, "[3, 4]")["all_correct"]
    assert not score_answer(record, "[1, 1]")["all_correct"]
    with pytest.raises(ValueError, match="nonfinite"):
        normalize_field(torch.full((2, 2), float("nan")), allow_constant=True)


def test_family_and_source_metrics_do_not_change_original_aggregate():
    z = torch.arange(4).reshape(2, 2).float()
    row = make_record("a", z, "single_point", {"points": [[1, 1]]})
    entries = [dict(task_type="single_point", shape="2x2", field="p", source_id="a",
                    family="fluid", score=score_answer(row, "[0]")),
               dict(task_type="single_point", shape="2x2", field="p", source_id="b",
                    family="wave", score=score_answer(row, "[10]"))]
    metrics = summarize_evaluation(entries, [[2, 2]])
    assert metrics["answer_accuracy"] == .5
    assert metrics["by_family"]["fluid"]["answer_accuracy"] == 1
    assert metrics["by_source_id"]["b"]["answer_accuracy"] == 0
    assert metrics["questions"] == 2
    del entries[1]["family"]
    with pytest.raises(ValueError, match="provenance"):
        summarize_evaluation(entries, [[2, 2]])


def test_expanded_recipe_preserves_user_budget_and_legacy_source_override():
    config = load_config(ROOT / "configs/field_to_llm_multisource.yaml", "full")
    assert config["generation"]["train_states"] == 16384 * 8
    assert config["training"]["epochs"] == 2
    assert config["training"]["batch_size"] == 16
    assert config["data"]["shape_sampling"]["height"] == [1, 128]
    assert config["data"]["shape_sampling"]["width"] == [1, 128]
    assert len(config["generation"]["tasks"]) == 11
    assert len(config["data"]["sources"]) == 4
    assert {s["id"] for s in config["data"]["sources"]} == {
        "ins_tracer", "acoustic_scattering_inclusions", "euler_multi_quadrants_openBC",
        "gray_scott_reaction_diffusion"}
    assert "multisource_openbc_v1" in config["data"]["qa_dir"]
    assert len({s["family"] for s in config["data"]["sources"]}) == 4
    assert {s["kind"] for s in config["data"]["sources"]} == {"the_well", "ins_tracer"}
    assert source_path(config) is None
    with pytest.raises(ValueError, match="Multi-source"):
        source_path(config, "wrong-single-file.h5")
