"""Truth, language isolation and scoring tests for executable field queries."""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensor_compression.downstream.field_query_program import (
    DEFAULT_POLICY, PROGRAM_TASKS, QUERY_PROTOCOL, TEMPLATE_SPLITS,
    execute_program, make_program_record, query_diagnostics, render_program_question,
    sample_program, selector_points, validate_program, validate_program_record,
)
from tensor_compression.downstream.point_readout import (
    TASKS, build_prompt, make_record, normalize_field, score_answer, validate_record,
)


def field(shape=(17, 19)):
    return normalize_field(torch.arange(shape[0] * shape[1]).reshape(shape).float(),
                           allow_singleton=True)[0]


def program(task, expression):
    return {"version": 1, "task": task, "expression": expression}


def region(start, size):
    return {"kind": "region", "start": start, "size": size}


@pytest.mark.parametrize("task", PROGRAM_TASKS)
def test_all_tasks_replay_across_seen_and_heldout_language(task):
    z = field()
    query = sample_program(task, list(z.shape), random.Random(42), DEFAULT_POLICY, z)
    rows = []
    for split, templates in TEMPLATE_SPLITS.items():
        for template in templates:
            row = make_program_record("source/state", z, task, query, template,
                                      language_group="seen" if split == "train" else "heldout")
            validate_record(row, z)
            assert score_answer(row, row["answer"])["all_correct"]
            prompt = build_prompt(row)
            assert row["question"] in prompt
            assert "query_program" not in prompt and "intermediate_values" not in prompt
            rows.append(row)
    assert len({row["semantic_id"] for row in rows}) == 1
    assert len({row["qa_id"] for row in rows}) == len(rows)
    assert len({row["question"] for row in rows}) == len(rows)
    assert len({row["answer"] for row in rows}) == 1
    assert not set(TEMPLATE_SPLITS["train"]) & (set(TEMPLATE_SPLITS["val"]) | set(TEMPLATE_SPLITS["test"]))
    assert not set(TEMPLATE_SPLITS["val"]) & set(TEMPLATE_SPLITS["test"])


def test_reverse_strided_line_order_and_language():
    z = torch.arange(35).reshape(5, 7).float()
    selector = {"kind": "line", "start": [3, 7], "step": [0, -2], "count": 4}
    query = program("line_profile", {"op": "read", "selector": selector})
    assert selector_points(selector, [5, 7]) == [[3, 7], [3, 5], [3, 3], [3, 1]]
    assert execute_program(query, z)[0] == [20., 18., 16., 14.]
    direct = render_program_question(query, [5, 7], "direct")
    interval = render_program_question(query, [5, 7], "interval")
    assert "2 cell(s) left" in direct and "4 cells" in direct
    assert "column indices from 7 through 1 inclusive, increment -2" in interval


def test_interval_and_relative_rectangles_refer_to_same_cells():
    query = program("region_values", {"op": "read", "selector": region([3, 7], [4, 5])})
    interval = render_program_question(query, [17, 19], "interval")
    relative = render_program_question(query, [17, 19], "relative")
    assert "rows 3 through 6 and columns 7 through 11" in interval
    assert "2 rows down and 6 columns right" in relative
    assert "4 rows downward and 5 columns rightward including that cell" in relative


def test_difference_is_ordered_and_preserves_intermediate_truth():
    z = torch.tensor([[1., 4., 9.], [2., 5., 8.]])
    query = program("point_difference", {"op": "subtract",
        "left": {"op": "read", "selector": {"kind": "points", "points": [[1, 1]]}},
        "right": {"op": "read", "selector": {"kind": "points", "points": [[2, 3]]}}})
    assert execute_program(query, z) == ([-7.], [], [1., 8.])
    row = make_program_record("state", z, "point_difference", query, "direct")
    assert "first quantity minus the second" in row["question"]
    validate_record(row, z)
    swapped = copy.deepcopy(query)
    swapped["expression"]["left"], swapped["expression"]["right"] = query["expression"]["right"], query["expression"]["left"]
    assert execute_program(swapped, z)[0] == [7.]


def test_mean_difference_uses_unrounded_operands():
    z = torch.tensor([[.14, .17, .23], [.26, .38, .49]])
    query = program("region_mean_difference", {"op": "subtract",
        "left": {"op": "mean", "selector": region([1, 1], [1, 2])},
        "right": {"op": "mean", "selector": region([2, 2], [1, 2])}})
    values, _, intermediates = execute_program(query, z)
    expected = z[0, :2].double().mean().item() - z[1, 1:].double().mean().item()
    assert values == [expected]
    assert values == [intermediates[0] - intermediates[1]]


def test_threshold_fraction_strict_greater_precision_and_tolerance():
    z = torch.tensor([[0., 1., 1.], [2., 3., 4.]])
    query = program("threshold_fraction", {"op": "fraction_gt", "selector": region([1, 1], [2, 3]), "target": 1.})
    row = make_program_record("state", z, "threshold_fraction", query, "direct")
    assert row["oracle"]["values"] == [.5]
    assert row["oracle"]["absolute_tolerance"] == .02
    assert row["oracle"]["answer_decimals"] == 3
    assert "strictly greater than 1.0" in row["question"]
    assert score_answer(row, "[0.51]")["all_correct"]
    assert not score_answer(row, "[0.55]")["all_correct"]
    assert not score_answer(row, "[50]")["all_correct"]
    bad = copy.deepcopy(row)
    bad["oracle"]["absolute_tolerance"] = .2
    with pytest.raises(ValueError, match="Fraction scoring"):
        validate_record(bad)


@pytest.mark.parametrize("task,op", [("region_argmin", "argmin"), ("region_argmax", "argmax"), ("nearest_value_location", "nearest")])
def test_compact_all_ties_preserve_every_accepted_cell(task, op):
    z = torch.zeros(128, 128)
    expr = {"op": op, "selector": region([1, 1], [64, 64])}
    if op == "nearest":
        expr["target"] = .1
    row = make_program_record("constant", z, task, program(task, expr), "direct")
    ties = row["oracle"]["valid_coordinates"]
    assert ties == {"encoding": "rectangle", "start": [1, 1], "size": [64, 64]}
    assert len(json.dumps(ties)) < 100
    validate_record(row, z)
    assert score_answer(row, "[64,64]")["all_correct"]
    assert not score_answer(row, "[64,65]")["all_correct"]
    assert not score_answer(row, "[1.1,2]")["all_correct"]


def test_disjoint_tie_runs_do_not_accept_mixed_coordinates():
    z = torch.tensor([[0., 0., 2., 0.], [2., 0., 0., 2.], [2., 2., 2., 2.]])
    query = program("region_argmin", {"op": "argmin", "selector": region([1, 1], [3, 4])})
    row = make_program_record("state", z, "region_argmin", query, "direct")
    assert row["oracle"]["valid_coordinates"] == {"encoding": "row_runs", "runs": [[1, 1, 2], [1, 4, 4], [2, 2, 3]]}
    validate_record(row, z)
    for point in [[1, 1], [1, 2], [1, 4], [2, 2], [2, 3]]:
        assert score_answer(row, json.dumps(point))["all_correct"]
    for point in [[1, 3], [2, 1], [2, 4], [3, 2]]:
        assert not score_answer(row, json.dumps(point))["all_correct"]


def test_nearest_target_displayed_value_defines_ties():
    z = torch.tensor([[0.125, 0.375, 0.875]])
    query = program("nearest_value_location", {"op": "nearest", "selector": region([1, 1], [1, 3]), "target": .25})
    row = make_program_record("state", z, "nearest_value_location", query, "direct")
    assert "0.25" in row["question"]
    assert score_answer(row, "[1,1]")["all_correct"]
    assert score_answer(row, "[1,2]")["all_correct"]
    assert not score_answer(row, "[1,3]")["all_correct"]


@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (17, 1), (7, 11)])
def test_singleton_and_odd_shapes_skip_only_impossible_tasks(shape):
    z = field(shape)
    impossible = {"multi_point", "line_profile", "point_difference", "region_mean", "region_std", "region_mean_difference"}
    for task in PROGRAM_TASKS:
        for seed in range(10):
            query = sample_program(task, list(shape), random.Random(seed), DEFAULT_POLICY, z)
            if shape == (1, 1) and task in impossible:
                assert query is None
                continue
            assert query is not None
            validate_record(make_program_record("state", z, task, query, "direct"), z)


def test_random_integer_lengths_rectangles_and_both_line_directions_have_support():
    z = field((17, 19))
    counts, shapes, strides = set(), set(), set()
    for seed in range(150):
        line = sample_program("line_profile", list(z.shape), random.Random(seed), DEFAULT_POLICY, z)["expression"]["selector"]
        counts.add(line["count"])
        strides.add(tuple(line["step"]))
        query = sample_program("region_values", list(z.shape), random.Random(seed), DEFAULT_POLICY, z)
        size = query["expression"]["selector"]["size"]
        assert size[0] * size[1] <= 32
        shapes.add(tuple(size))
    assert {3, 5, 7, 11, 17} <= counts
    assert any(dr < 0 or dc < 0 for dr, dc in strides)
    assert any(abs(dr or dc) > 1 for dr, dc in strides)
    assert any(r != c and r not in {1, 2, 4, 8, 16} for r, c in shapes)


@pytest.mark.parametrize("task", ["region_mean", "region_std", "region_mean_difference"])
def test_standardized_whole_field_statistics_excluded(task):
    z = field((7, 11))
    for seed in range(100):
        query = sample_program(task, list(z.shape), random.Random(seed), DEFAULT_POLICY, z)
        expr = query["expression"]
        leaves = [expr["left"], expr["right"]] if "left" in expr else [expr]
        assert all(leaf["selector"]["size"] != [7, 11] for leaf in leaves)
    bad = program("region_mean", {"op": "mean", "selector": region([1, 1], [7, 11])})
    with pytest.raises(ValueError, match="proper subregion"):
        validate_program(bad, [7, 11])


@pytest.mark.parametrize("change", ["question", "task", "shape", "oracle", "template", "identity", "metadata", "operands"])
def test_tampered_records_are_rejected(change):
    z = field()
    query = sample_program("point_difference", list(z.shape), random.Random(8), DEFAULT_POLICY, z)
    row = make_program_record("state", z, "point_difference", query, "direct")
    if change == "question": row["question"] += " Then negate the answer."
    elif change == "task": row["task_type"] = "multi_point"
    elif change == "shape": row["grid_shape"] = [1, 1]
    elif change == "oracle": row["oracle"]["values"][0] += 1
    elif change == "template": row["language_group"] = "heldout"
    elif change == "identity": row["semantic_id"] = "bad"
    elif change == "metadata": row["query_metadata"]["query_point_count"] = 999
    else: row["oracle"]["intermediate_values"][0] += 1
    with pytest.raises(ValueError):
        validate_record(row, z)


@pytest.mark.parametrize("selector", [
    {"kind": "points", "points": [[True, 1]]},
    {"kind": "points", "points": [[1, 1], [1, 1]]},
    {"kind": "line", "start": [1, 1], "step": [1, 1], "count": 2},
    {"kind": "line", "start": [1, 1], "step": [0, -1], "count": 2},
    {"kind": "region", "start": [1, 1], "size": [0, 1]},
    {"kind": "region", "start": [2, 2], "size": [4, 4]},
])
def test_malformed_and_out_of_bounds_selectors_rejected(selector):
    with pytest.raises(ValueError):
        selector_points(selector, [4, 4])


def test_geometry_diagnostics_exact_unique_coverage_and_blocks():
    z = field((5, 7))
    query = program("region_mean_difference", {"op": "subtract",
        "left": {"op": "mean", "selector": region([1, 1], [2, 3])},
        "right": {"op": "mean", "selector": region([2, 2], [2, 3])}})
    row = make_program_record("state", z, "region_mean_difference", query, "direct")
    metrics = query_diagnostics(row, (2, 2))
    assert metrics["query_point_count"] == 10
    assert metrics["operand_point_count"] == 12
    assert metrics["output_count"] == 1 and metrics["operation_count"] == 5
    assert metrics["query_extent_cells"] == 12 and metrics["distinct_blocks"] == 4
    assert metrics["block_relation"] == "multiple_blocks"
    legacy = make_record("state", z, "multi_point", {"points": [[1, 1], [2, 2]]})
    assert query_diagnostics(legacy, (2, 2))["block_relation"] == "single_block"


def test_legacy_contract_and_templates_remain_unchanged():
    assert len(TASKS) == 11
    assert len(PROGRAM_TASKS) == 14
    z = field((3, 4))
    legacy = make_record("state", z, "single_point", {"points": [[1, 2]]})
    validate_record(legacy, z)
    assert "query_protocol" not in legacy
    invalid = copy.deepcopy(legacy)
    invalid["query_protocol"] = "unknown"
    with pytest.raises(ValueError, match="Unknown query protocol"):
        validate_record(invalid, z)


def test_holdout_templates_cannot_be_labelled_seen():
    z = field()
    query = sample_program("single_point", list(z.shape), random.Random(0), DEFAULT_POLICY, z)
    with pytest.raises(ValueError, match="Language group"):
        make_program_record("state", z, "single_point", query, "relative", language_group="seen")
