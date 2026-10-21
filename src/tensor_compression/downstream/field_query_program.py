"""Executable field queries with deterministic, independently held-out language.

Programs, intermediate operands and diagnostics are audit data. The inference
prompt continues to expose only the grid shape and the rendered question.
"""
from __future__ import annotations

import copy
import json
import math
import random
from collections.abc import Mapping, Sequence

import torch

from .point_readout import LOCATION_TASKS, READ_TASKS, STAT_TASKS, TASKS, json_hash, score_answer


QUERY_PROTOCOL = "field_query_program_v1"
PROGRAM_TASKS = TASKS + ("point_difference", "region_mean_difference", "threshold_fraction")
TASK_CAPABILITIES = {**dict.fromkeys(READ_TASKS, "readout"),
                     **dict.fromkeys(STAT_TASKS, "aggregation"),
                     **dict.fromkeys(LOCATION_TASKS, "localization"),
                     **dict.fromkeys(PROGRAM_TASKS[len(TASKS):], "composition")}
TEMPLATE_SPLITS = {"train": ("direct", "question", "region_first", "coordinate_tuple"),
                   "val": ("interval",), "test": ("relative",)}
_TEMPLATES = frozenset(sum(TEMPLATE_SPLITS.values(), ()))
DEFAULT_POLICY = {"read_count_max": 32, "line_count_max": 32,
                  "region_value_max_cells": 32, "statistic_max_cells": 4096}
_OPS = {"region_mean": "mean", "region_std": "std", "region_min": "min", "region_max": "max",
        "region_argmin": "argmin", "region_argmax": "argmax", "nearest_value_location": "nearest",
        "threshold_fraction": "fraction_gt"}


def _shape(shape: Sequence[int]) -> tuple[int, int]:
    if len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("Grid shape must contain two positive integers")
    return shape[0], shape[1]


def _keys(value, expected):
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"Invalid query keys; expected {sorted(expected)}")


def _selector_geometry(selector: Mapping, shape: Sequence[int]):
    """Validate without materializing a possibly large rectangle; count/bounds."""
    h, w = _shape(shape)
    kind = selector.get("kind")
    if kind == "points":
        _keys(selector, ("kind", "points"))
        points = selector["points"]
        if (not isinstance(points, list) or not points
                or any(not isinstance(p, list) or len(p) != 2 or any(type(n) is not int for n in p)
                       or not (1 <= p[0] <= h and 1 <= p[1] <= w) for p in points)):
            raise ValueError("Coordinates must be one-based integers inside the grid")
        if len({tuple(p) for p in points}) != len(points):
            raise ValueError("A selector cannot contain repeated positions")
        return len(points), (min(p[0] for p in points), min(p[1] for p in points),
                             max(p[0] for p in points), max(p[1] for p in points))
    elif kind == "line":
        _keys(selector, ("kind", "start", "step", "count"))
        start, step, count = selector["start"], selector["step"], selector["count"]
        if (not isinstance(start, list) or len(start) != 2
                or any(type(n) is not int for n in start)
                or not isinstance(step, list) or len(step) != 2
                or any(type(n) is not int for n in step) or sum(n != 0 for n in step) != 1
                or type(count) is not int or not 2 <= count <= max(h, w)):
            raise ValueError("Line requires one nonzero signed integer step and a legal count >= 2")
        end = [start[0] + (count - 1) * step[0], start[1] + (count - 1) * step[1]]
        bounds = (min(start[0], end[0]), min(start[1], end[1]), max(start[0], end[0]), max(start[1], end[1]))
    elif kind == "region":
        _keys(selector, ("kind", "start", "size"))
        start, size = selector["start"], selector["size"]
        if (not isinstance(start, list) or len(start) != 2 or any(type(n) is not int for n in start)
                or not isinstance(size, list) or len(size) != 2
                or any(type(n) is not int or n < 1 for n in size) or size[0] > h or size[1] > w):
            raise ValueError("Region requires positive integer height and width within the grid")
        count = size[0] * size[1]
        bounds = (start[0], start[1], start[0] + size[0] - 1, start[1] + size[1] - 1)
    else:
        raise ValueError("Unknown selector kind")
    if not (1 <= bounds[0] <= bounds[2] <= h and 1 <= bounds[1] <= bounds[3] <= w):
        raise ValueError("Coordinates must be one-based integers inside the grid")
    return count, bounds


def selector_points(selector: Mapping, shape: Sequence[int]) -> list[list[int]]:
    """Expand a typed selector, preserving the exact answer order."""
    _selector_geometry(selector, shape)
    if selector["kind"] == "points":
        return [list(p) for p in selector["points"]]
    r, c = selector["start"]
    if selector["kind"] == "line":
        dr, dc = selector["step"]
        return [[r + i * dr, c + i * dc] for i in range(selector["count"])]
    h, w = selector["size"]
    return [[r + i, c + j] for i in range(h) for j in range(w)]


def _leaves(expression):
    if expression["op"] == "subtract":
        return [expression["left"], expression["right"]]
    return [expression]


def validate_program(program: Mapping, shape: Sequence[int]) -> None:
    _shape(shape)
    _keys(program, ("version", "task", "expression"))
    task, expression = program["task"], program["expression"]
    if type(program["version"]) is not int or program["version"] != 1 or task not in PROGRAM_TASKS:
        raise ValueError("Unsupported query program version or task")
    if not isinstance(expression, Mapping):
        raise ValueError("Query expression must be a mapping")
    if task in {"point_difference", "region_mean_difference"}:
        _keys(expression, ("op", "left", "right"))
        if expression["op"] != "subtract":
            raise ValueError("Difference query requires subtraction")
        leaves = [expression["left"], expression["right"]]
        expected_op = "read" if task == "point_difference" else "mean"
    else:
        leaves = [expression]
        expected_op = "read" if task in READ_TASKS else _OPS[task]
    for leaf in leaves:
        _keys(leaf, ("op", "selector", "target") if expected_op in {"nearest", "fraction_gt"}
              else ("op", "selector"))
        if leaf["op"] != expected_op:
            raise ValueError("Operator does not match the declared task")
        selector = leaf["selector"]
        if not isinstance(selector, Mapping):
            raise ValueError("Selector must be a mapping")
        count, _ = _selector_geometry(selector, shape)
        expected_kind = ("points" if task in {"single_point", "multi_point", "point_difference"}
                         else "line" if task == "line_profile" else "region")
        if selector["kind"] != expected_kind:
            raise ValueError("Selector kind does not match the declared task")
        if task in {"single_point", "point_difference"} and count != 1:
            raise ValueError("Single-point operand requires exactly one point")
        if task == "multi_point" and count < 2:
            raise ValueError("Multi-point read requires at least two points")
        if expected_op in {"mean", "std"} and count == shape[0] * shape[1]:
            raise ValueError("Mean/std queries must use a proper subregion, not the standardized whole field")
        if "target" in leaf and (type(leaf["target"]) not in (int, float) or not math.isfinite(leaf["target"])):
            raise ValueError("Displayed query target must be finite")
    if len(leaves) == 2 and leaves[0]["selector"] == leaves[1]["selector"]:
        raise ValueError("Difference operands must select different locations")


def _compact_ties(mask, selector):
    r, c = selector["start"]
    h, w = selector["size"]
    mask = mask.reshape(h, w).cpu()
    if bool(mask.all()):
        return [r, c], {"encoding": "rectangle", "start": [r, c], "size": [h, w]}
    runs = []
    for i in range(h):
        columns = mask[i].nonzero().flatten().tolist()
        for column in columns:
            absolute = column + c
            if runs and runs[-1][0] == r + i and runs[-1][2] + 1 == absolute:
                runs[-1][2] = absolute
            else:
                runs.append([r + i, absolute, absolute])
    return runs[0][:2], {"encoding": "row_runs", "runs": runs}


def _validate_ties(ties, selector, shape):
    """Return the canonical first tie, checking all encoded coordinates."""
    _, bounds = _selector_geometry(selector, shape)
    if not isinstance(ties, Mapping):
        raise ValueError("Program coordinate ties require compact encoding")
    if ties.get("encoding") == "rectangle":
        _keys(ties, ("encoding", "start", "size"))
        _, encoded = _selector_geometry({"kind": "region", "start": ties["start"], "size": ties["size"]}, shape)
        if encoded != bounds:
            raise ValueError("All-tie rectangle must equal the query rectangle")
        return list(ties["start"])
    _keys(ties, ("encoding", "runs"))
    if ties["encoding"] != "row_runs" or not isinstance(ties["runs"], list) or not ties["runs"]:
        raise ValueError("Invalid coordinate tie runs")
    previous = None
    for run in ties["runs"]:
        if (not isinstance(run, list) or len(run) != 3 or any(type(n) is not int for n in run)
                or not (bounds[0] <= run[0] <= bounds[2] and bounds[1] <= run[1] <= run[2] <= bounds[3])
                or (previous and (run[0] < previous[0] or (run[0] == previous[0] and run[1] <= previous[2] + 1)))):
            raise ValueError("Invalid, unsorted or overlapping coordinate tie run")
        previous = run
    return ties["runs"][0][:2]


def coordinate_is_accepted(record, values):
    if (not isinstance(values, (list, tuple)) or len(values) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) or int(v) != v for v in values)):
        return False
    ties = record["oracle"]["valid_coordinates"]
    r, c = values
    if ties["encoding"] == "rectangle":
        sr, sc = ties["start"]
        h, w = ties["size"]
        return sr <= r < sr + h and sc <= c < sc + w
    return any(row == r and first <= c <= last for row, first, last in ties["runs"])


def execute_program(program: Mapping, z: torch.Tensor) -> tuple[list, list | dict, list]:
    """Return exact final values, accepted coordinate ties, and scalar operands."""
    if z.ndim != 2 or not bool(torch.isfinite(z).all()):
        raise ValueError("Query source must be a finite 2-D field")
    validate_program(program, list(z.shape))

    def execute(leaf):
        selector = leaf["selector"]
        if selector["kind"] == "region":
            r, c = selector["start"]
            h, w = selector["size"]
            values = z[r - 1:r - 1 + h, c - 1:c - 1 + w].reshape(-1).double()
        else:
            points = selector_points(selector, list(z.shape))
            indices = torch.tensor(points, device=z.device, dtype=torch.long) - 1
            values = z[indices[:, 0], indices[:, 1]].double()
        op = leaf["op"]
        if op == "read":
            return values.tolist(), []
        if op in {"argmin", "argmax", "nearest"}:
            if op == "nearest":
                distance = (values - leaf["target"]).abs()
                mask = distance == distance.min()
            else:
                mask = values == (values.min() if op == "argmin" else values.max())
            return _compact_ties(mask, selector)
        value = {"mean": lambda: values.mean(), "std": lambda: values.std(unbiased=False),
                 "min": lambda: values.min(), "max": lambda: values.max(),
                 "fraction_gt": lambda: (values > leaf["target"]).double().mean()}[op]()
        return [float(value)], []

    expression = program["expression"]
    if expression["op"] == "subtract":
        left, _ = execute(expression["left"])
        right, _ = execute(expression["right"])
        return [left[0] - right[0]], [], [left[0], right[0]]
    values, coordinates = execute(expression)
    return values, coordinates, []


def _sample_region(shape, rng, max_cells, *, proper=False):
    h, w = shape
    # All fitting positive integer rectangles have support, including long/thin
    # and non-power-of-two shapes. Choose height then a legal width; not uniform
    # over rectangle areas, and not described as such in experiment reports.
    heights = [rh for rh in range(1, min(h, max_cells) + 1)
               if not (proper and rh == h and w == 1)]
    if not heights:
        return None
    rh = rng.choice(heights)
    width_max = min(w, max_cells // rh)
    if proper and rh == h:
        width_max = min(width_max, w - 1)
    if width_max < 1:
        return None
    rw = rng.randint(1, width_max)
    return {"kind": "region", "start": [rng.randint(1, h - rh + 1), rng.randint(1, w - rw + 1)],
            "size": [rh, rw]}


def sample_program(task: str, shape: Sequence[int], rng: random.Random, policy: Mapping,
                   z: torch.Tensor) -> dict | None:
    h, w = _shape(shape)
    if task not in PROGRAM_TASKS:
        raise ValueError(f"Unsupported query task: {task}")
    limits = {key: policy.get(key, value) for key, value in DEFAULT_POLICY.items()}
    if any(type(n) is not int or n < 1 for n in limits.values()):
        raise ValueError("Query size caps must be positive integers")
    if list(z.shape) != list(shape):
        raise ValueError("Sampling field shape mismatch")
    if task in {"single_point", "multi_point", "point_difference"}:
        maximum = min(h * w, limits["read_count_max"])
        if task != "single_point" and (h * w < 2 or (task == "multi_point" and maximum < 2)):
            return None
        count = 1 if task == "single_point" else 2 if task == "point_difference" else rng.randint(2, maximum)
        indices = rng.sample(range(h * w), count)
        points = [[n // w + 1, n % w + 1] for n in indices]
        expression = {"op": "read", "selector": {"kind": "points", "points": points}}
        if task == "point_difference":
            expression = {"op": "subtract", "left": {"op": "read", "selector": {"kind": "points", "points": points[:1]}},
                          "right": {"op": "read", "selector": {"kind": "points", "points": points[1:]}}}
    elif task == "line_profile":
        options = [(axis, count) for axis, extent in enumerate(shape)
                   for count in range(2, min(extent, limits["line_count_max"]) + 1)]
        if not options:
            return None
        axis, count = rng.choice(options)
        stride = rng.randint(1, (shape[axis] - 1) // (count - 1)) * rng.choice((-1, 1))
        span = (count - 1) * abs(stride)
        start = [rng.randint(1, h), rng.randint(1, w)]
        start[axis] = rng.randint(1, shape[axis] - span) + (span if stride < 0 else 0)
        step = [0, 0]
        step[axis] = stride
        expression = {"op": "read", "selector": {"kind": "line", "start": start, "step": step, "count": count}}
    else:
        proper = task in {"region_mean", "region_std", "region_mean_difference"}
        cap = limits["region_value_max_cells"] if task == "region_values" else limits["statistic_max_cells"]
        selector = _sample_region(shape, rng, cap, proper=proper)
        if selector is None:
            return None
        if task == "region_mean_difference":
            other = None
            for _ in range(32):
                candidate = _sample_region(shape, rng, cap, proper=True)
                if candidate != selector:
                    other = candidate
                    break
            if other is None:
                # A deterministic legal alternative avoids silently dropping a
                # feasible task because random proposals repeatedly collide.
                row, col = selector["start"]
                index = ((row - 1) * w + col) % (h * w)
                other = {"kind": "region", "start": [index // w + 1, index % w + 1], "size": [1, 1]}
            expression = {"op": "subtract", "left": {"op": "mean", "selector": selector},
                          "right": {"op": "mean", "selector": other}}
        else:
            expression = {"op": "read" if task == "region_values" else _OPS[task], "selector": selector}
            if task in {"nearest_value_location", "threshold_fraction"}:
                sr, sc = selector["start"]
                rh, rw = selector["size"]
                r, c = rng.randint(sr, sr + rh - 1), rng.randint(sc, sc + rw - 1)
                expression["target"] = round(float(z[r - 1, c - 1]), 1) or 0.0
    program = {"version": 1, "task": task, "expression": expression}
    validate_program(program, shape)
    return program


def _selector_text(selector, style):
    kind = selector["kind"]
    if kind == "points":
        points = selector["points"]
        if style == "relative":
            return ("the ordered cells at offsets (down, right) "
                    + "; ".join(f"({r - 1}, {c - 1})" for r, c in points)
                    + " from the top-left cell (row 1, column 1); offsets count moves, with zero meaning no move")
        if style == "coordinate_tuple":
            return "the ordered (row, column) coordinates " + "; ".join(f"({r}, {c})" for r, c in points)
        if style == "interval":
            return "the cells in this exact sequence: " + "; ".join(f"[row={r}; column={c}]" for r, c in points)
        return "the cells, in listed order, " + "; ".join(f"row {r}, column {c}" for r, c in points)
    r, c = selector["start"]
    if kind == "region":
        h, w = selector["size"]
        if style == "interval":
            return f"rows {r} through {r + h - 1} and columns {c} through {c + w - 1}, with both endpoints included"
        if style == "relative":
            return (f"the rectangle reached by moving {r - 1} rows down and {c - 1} columns right from "
                    f"the top-left cell (row 1, column 1), then taking {h} rows downward and {w} columns rightward including that cell")
        if style == "coordinate_tuple":
            return f"the rectangle with origin (row, column) = ({r}, {c}) and extent (height, width) = ({h}, {w})"
        return f"the rectangle of height {h} and width {w} whose top-left cell is row {r}, column {c}"
    dr, dc = selector["step"]
    count = selector["count"]
    direction = "down" if dr > 0 else "up" if dr < 0 else "right" if dc > 0 else "left"
    stride = abs(dr or dc)
    if style == "interval":
        axis, start, end, increment, fixed = ("row", r, r + (count - 1) * dr, dr, f"column {c}") if dr else (
            "column", c, c + (count - 1) * dc, dc, f"row {r}")
        return f"the ordered {axis} indices from {start} through {end} inclusive, increment {increment}, at fixed {fixed} ({count} cells)"
    origin = (f"an offset of {r - 1} rows down and {c - 1} columns right from cell (row 1, column 1)"
              if style == "relative" else f"(row, column) = ({r}, {c})" if style == "coordinate_tuple"
              else f"row {r}, column {c}")
    return f"the {count} cells starting at {origin}, including the start, moving {stride} cell(s) {direction} per step in that order"


def _description(leaf, style):
    selector = _selector_text(leaf["selector"], style)
    op = leaf["op"]
    names = {"mean": "arithmetic mean", "std": "population standard deviation (variance divided by the number of selected cells)",
             "min": "minimum", "max": "maximum"}
    if op == "read":
        order = " Traverse rows from top to bottom and each row from left to right." if leaf["selector"]["kind"] == "region" else " Preserve the specified order."
        return f"the standardized values at {selector}", order
    if op in names:
        return f"the {names[op]} of standardized values over {selector}", ""
    if op == "fraction_gt":
        return (f"the fraction of cells in {selector} whose standardized value is strictly greater than {leaf['target']}",
                " Divide the qualifying count by the selected cell count; return a fraction from 0 to 1, not a percentage.")
    property_text = (f"closest in absolute difference to {leaf['target']}" if op == "nearest"
                     else "minimal" if op == "argmin" else "maximal")
    return f"one cell in {selector} whose standardized value is {property_text}", ""


def render_program_question(program: Mapping, shape: Sequence[int], template_id: str) -> str:
    validate_program(program, shape)
    if template_id not in _TEMPLATES:
        raise ValueError("Unknown language template family")
    expression = program["expression"]
    if expression["op"] == "subtract":
        left, _ = _description(expression["left"], template_id)
        right, _ = _description(expression["right"], template_id)
        description, extra = f"the first quantity minus the second, where the first is {left} and the second is {right}", ""
    else:
        description, extra = _description(expression, template_id)
    if template_id == "question":
        question = f"Using the supplied grid, what is {description}?"
    elif template_id == "region_first":
        # Reference the selection first, independently of the requested operation.
        selections = " Selection B: ".join(_selector_text(leaf["selector"], template_id) for leaf in _leaves(expression))
        question = f"Selection A: {selections}. For these locations, calculate or identify {description}."
    elif template_id == "coordinate_tuple":
        question = f"Grid query. Requested result: {description}."
    elif template_id == "interval":
        question = f"Use the following explicit index specification to obtain {description}."
    elif template_id == "relative":
        question = f"Navigate from the grid's top-left cell as described and give {description}."
    else:
        question = f"Return {description}."
    if program["task"] in LOCATION_TASKS:
        contract = " Return [row, column] in the full grid's one-based coordinates. Any tied optimal cell is accepted."
    else:
        contract = " Return only a JSON array of numbers, using [value] for a single result."
    return question + extra + contract


def _metadata(program, shape):
    leaves = _leaves(program["expression"])
    geometry = [_selector_geometry(leaf["selector"], shape) for leaf in leaves]
    count = sum(item[0] for item in geometry)
    if len(geometry) == 2:
        a, b = geometry[0][1], geometry[1][1]
        count -= max(0, min(a[2], b[2]) - max(a[0], b[0]) + 1) * max(0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
    output_count = geometry[0][0] if program["task"] in READ_TASKS else 2 if program["task"] in LOCATION_TASKS else 1
    operations = sum(1 if leaf["op"] == "read" else 2 for leaf in leaves) + (len(leaves) == 2)
    extent = ((max(item[1][2] for item in geometry) - min(item[1][0] for item in geometry) + 1)
              * (max(item[1][3] for item in geometry) - min(item[1][1] for item in geometry) + 1))
    return {"query_point_count": count, "operand_point_count": sum(item[0] for item in geometry), "output_count": output_count,
            "operation_count": int(operations), "query_extent_cells": extent}


def _difficulty(point_count):
    return "1" if point_count == 1 else "2-4" if point_count <= 4 else "5-16" if point_count <= 16 else "17-64" if point_count <= 64 else "65-256" if point_count <= 256 else "257-1024" if point_count <= 1024 else "1025+"


def make_program_record(state_ref: str, z: torch.Tensor, task: str, program: Mapping, template_id: str, *,
                        decimals: int = 1, tolerance: float = .2, language_group: str = "seen") -> dict:
    if task != program.get("task"):
        raise ValueError("Record task and program task disagree")
    if task == "threshold_fraction":
        decimals, tolerance = 3, .02
    if (type(decimals) is not int or not 0 <= decimals <= 12 or type(tolerance) not in (int, float)
            or not math.isfinite(tolerance) or tolerance <= .5 * 10 ** -decimals):
        raise ValueError("Answer rounding must be strictly finer than a finite positive tolerance")
    if language_group not in {"seen", "heldout"}:
        raise ValueError("Invalid language group")
    if (template_id in TEMPLATE_SPLITS["train"]) != (language_group == "seen"):
        raise ValueError("Language group does not match template partition")
    values, coordinates, operands = execute_program(program, z)
    shape = list(z.shape)
    semantic_id = json_hash({"state_ref": state_ref, "query_protocol": QUERY_PROTOCOL, "query_program": program})[:24]
    metadata = _metadata(program, shape)
    answer = values if coordinates else [round(value, decimals) or 0.0 for value in values]
    return {"qa_id": json_hash({"semantic_id": semantic_id, "template_family": template_id})[:24],
            "state_ref": state_ref, "task_type": task, "grid_shape": shape, "query_protocol": QUERY_PROTOCOL,
            "semantic_id": semantic_id, "template_family": template_id, "language_group": language_group,
            "capability": TASK_CAPABILITIES[task], "difficulty": _difficulty(metadata["query_point_count"]),
            "query_metadata": metadata, "question": render_program_question(program, shape, template_id),
            "answer": json.dumps(answer, allow_nan=False),
            "oracle": {"query_program": copy.deepcopy(program), "values": values,
                       "intermediate_values": operands, "value_space": "stored_fp16_z", "absolute_tolerance": tolerance,
                       "answer_decimals": decimals, "answer_kind": "coordinate" if coordinates else "values",
                       "valid_coordinates": coordinates}}


def validate_program_record(record: Mapping, z: torch.Tensor | None = None) -> None:
    if record.get("query_protocol") != QUERY_PROTOCOL:
        raise ValueError("Unknown query protocol")
    oracle, task, shape = record["oracle"], record["task_type"], record["grid_shape"]
    program = oracle["query_program"]
    validate_program(program, shape)
    if task != program["task"] or oracle["value_space"] != "stored_fp16_z":
        raise ValueError("Program task or value space mismatch")
    if record["question"] != render_program_question(program, shape, record["template_family"]):
        raise ValueError("Question and query program disagree")
    if record["language_group"] not in {"seen", "heldout"} or (record["template_family"] in TEMPLATE_SPLITS["train"]) != (record["language_group"] == "seen"):
        raise ValueError("Language group and template partition disagree")
    metadata = _metadata(program, shape)
    if (record["query_metadata"] != metadata or record["capability"] != TASK_CAPABILITIES[task]
            or record["difficulty"] != _difficulty(metadata["query_point_count"])):
        raise ValueError("Query diagnostic metadata disagree")
    semantic_id = json_hash({"state_ref": record["state_ref"], "query_protocol": QUERY_PROTOCOL, "query_program": program})[:24]
    if record["semantic_id"] != semantic_id or record["qa_id"] != json_hash({"semantic_id": semantic_id, "template_family": record["template_family"]})[:24]:
        raise ValueError("Query identity mismatch")
    values, coordinates = oracle["values"], oracle["valid_coordinates"]
    if not isinstance(values, list) or len(values) != metadata["output_count"] or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("Invalid oracle values")
    coordinate_task = task in LOCATION_TASKS
    if oracle["answer_kind"] != ("coordinate" if coordinate_task else "values"):
        raise ValueError("Oracle answer kind mismatch")
    if coordinate_task:
        canonical = _validate_ties(coordinates, program["expression"]["selector"], shape)
        if values != canonical or any(type(n) is not int for n in values):
            raise ValueError("Invalid canonical coordinate or tie set")
    elif coordinates:
        raise ValueError("Numerical answers cannot have coordinate alternatives")
    tol, decimals = oracle["absolute_tolerance"], oracle["answer_decimals"]
    if (type(decimals) is not int or not 0 <= decimals <= 12 or type(tol) not in (int, float)
            or not math.isfinite(tol) or tol <= .5 * 10 ** -decimals):
        raise ValueError("Invalid answer precision/tolerance")
    if task == "threshold_fraction" and (tol != .02 or decimals != 3 or not 0 <= values[0] <= 1):
        raise ValueError("Fraction scoring requires precision 3 and absolute tolerance 0.02")
    operands = oracle["intermediate_values"]
    if (not isinstance(operands, list) or len(operands) != (2 if program["expression"]["op"] == "subtract" else 0)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in operands)
            or (operands and values != [operands[0] - operands[1]])):
        raise ValueError("Invalid intermediate oracle values")
    expected_answer = values if coordinate_task else [round(v, decimals) or 0.0 for v in values]
    if record["answer"] != json.dumps(expected_answer, allow_nan=False):
        raise ValueError("Reference answer disagrees with its precision contract")
    if z is not None:
        replay = execute_program(program, z)
        if list(z.shape) != shape or (values, coordinates, operands) != replay:
            raise ValueError("Oracle does not replay from the source field")
    if not score_answer(record, record["answer"])["all_correct"]:
        raise ValueError("Reference answer fails its scoring contract")


def query_diagnostics(record: Mapping, block_shape=(1, 1)) -> dict:
    bh, bw = _shape(block_shape)
    if record.get("query_protocol") == QUERY_PROTOCOL:
        program = record["oracle"]["query_program"]
        metadata = _metadata(program, record["grid_shape"])
        points = {tuple(p) for leaf in _leaves(program["expression"])
                  for p in selector_points(leaf["selector"], record["grid_shape"])}
        extra = {key: record[key] for key in ("capability", "difficulty", "language_group", "template_family")}
    else:
        from .point_readout import query_points
        points = {tuple(p) for p in query_points(record["task_type"], record["oracle"]["query_spec"], record["grid_shape"])}
        metadata = {"query_point_count": len(points), "operand_point_count": len(points),
                    "output_count": len(record["oracle"]["values"]), "operation_count": 1 if record["task_type"] in READ_TASKS else 2,
                    "query_extent_cells": (max(r for r, _ in points) - min(r for r, _ in points) + 1) * (max(c for _, c in points) - min(c for _, c in points) + 1)}
        extra = {"capability": TASK_CAPABILITIES[record["task_type"]], "difficulty": _difficulty(len(points)),
                 "language_group": "legacy", "template_family": f"legacy_{record['template_variant']}"}
    blocks = {((r - 1) // bh, (c - 1) // bw) for r, c in points}
    return {**metadata, **extra, "distinct_blocks": len(blocks),
            "block_relation": "single_block" if len(blocks) == 1 else "multiple_blocks",
            "boundary_point_fraction": sum(r % bh == 0 or c % bw == 0 for r, c in points) / len(points)}
