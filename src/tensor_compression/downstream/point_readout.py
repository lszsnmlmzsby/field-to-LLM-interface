"""Open numerical readout protocol: data truth, public prompts and strict scoring.

Coordinates in questions are one-based. Model answers are ordered JSON arrays
of approximate standardized values, generated over the full vocabulary.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

import torch


FORMAT = "standardized_point_readout_v1"
PROMPT_VERSION = "ordered_values_one_based_v1"
TASKS = ("single_point", "multi_point", "line_profile", "region_values")
DEFAULT_TOLERANCE = 0.2


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     allow_nan=False).encode()).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def normalize_field(raw: torch.Tensor) -> tuple[torch.Tensor, dict]:
    raw = raw.detach().cpu().float().contiguous()
    if raw.ndim != 2 or min(raw.shape) < 1 or not bool(torch.isfinite(raw).all()):
        raise ValueError("nonfinite_or_invalid_field")
    mean, std = raw.mean(), raw.std(unbiased=False)
    scale = std + 1e-6
    z = ((raw - mean) / scale).half().float()
    if not bool(torch.isfinite(z).all()) or float(z.min()) == float(z.max()):
        raise ValueError("constant_or_nonfinite_standardized_field")
    return z, {"mean": float(mean), "std": float(std), "scale": float(scale),
               "raw_sha256": tensor_hash(raw), "z_sha256": tensor_hash(z)}


def query_points(task: str, spec: Mapping, shape: Sequence[int]) -> list[list[int]]:
    """Expand the audit-only query specification; never pass it to the model."""
    h, w = shape
    if task in {"single_point", "multi_point"}:
        points = spec["points"]
        if task == "single_point" and len(points) != 1:
            raise ValueError("single_point requires exactly one point")
        if task == "multi_point" and len(points) < 2:
            raise ValueError("multi_point requires at least two points")
    elif task == "line_profile":
        start = spec["start"]
        step = spec["step"]
        count = spec["count"]
        if step not in ([0, 1], [1, 0]) or type(count) is not int or count < 2:
            raise ValueError("Line requires a positive horizontal/vertical unit step and count >= 2")
        points = [[start[0] + i * step[0], start[1] + i * step[1]] for i in range(count)]
    elif task == "region_values":
        start, size = spec["start"], spec["size"]
        if len(size) != 2 or any(type(n) is not int or n < 2 for n in size):
            raise ValueError("Region dimensions must be integers >= 2")
        points = [[start[0] + i, start[1] + j] for i in range(size[0]) for j in range(size[1])]
    else:
        raise ValueError(f"Unsupported point readout task: {task}")
    if not points or any(len(p) != 2 or any(type(v) is not int for v in p)
                         or not (1 <= p[0] <= h and 1 <= p[1] <= w) for p in points):
        raise ValueError("Query coordinates must be one-based integers inside the field")
    if len(set(map(tuple, points))) != len(points):
        raise ValueError("Query contains duplicate positions")
    return [list(p) for p in points]


def render_question(task: str, spec: Mapping, shape: Sequence[int], variant: int = 0) -> str:
    points = query_points(task, spec, shape)
    if variant not in (0, 1, 2):
        raise ValueError("Unsupported question template")
    if task in {"single_point", "multi_point"}:
        locations = "; ".join(f"row {r}, column {c}" for r, c in points)
        stems = ("Read the approximate standardized values at", "Report the approximate z values at",
                 "What are the approximate standardized field values at")
        return f"{stems[variant]} {locations}, in the order listed?"
    if task == "line_profile":
        r, c = spec["start"]
        direction = "rightward" if spec["step"] == [0, 1] else "downward"
        stems = ("Read a short standardized-value profile", "Report a sequence of approximate z values",
                 "Extract a short profile of standardized field values")
        return (f"{stems[variant]}, starting at row {r}, column {c}, moving {direction} "
                f"one cell at a time, for {spec['count']} cells including the starting cell.")
    r, c = spec["start"]
    h, w = spec["size"]
    stems = ("Read the approximate standardized values", "Report the approximate z values",
             "Extract the standardized field values")
    return (f"{stems[variant]} in the {h} by {w} region whose top-left cell is row {r}, "
            f"column {c}. Traverse each row left to right, then proceed to the next row.")


def build_prompt(record: Mapping) -> str:
    """Explicit inference allowlist: only public grid shape and question text."""
    h, w = record["grid_shape"]
    return (f"The supplied field memory represents a {h} by {w} standardized numerical grid.\n"
            "Coordinates use one-based row and column indices. Read approximate values directly "
            "from the field. Return only one JSON array of numbers in the requested order, "
            "including for a single value. Do not output coordinates or explanations.\n\n"
            f"Question: {record['question']}\nAnswer:")


def sample_spec(task: str, shape: Sequence[int], rng: random.Random, *,
                multi_counts=(2, 3, 4), line_counts=(3, 4, 5), region_shapes=((2, 2),)) -> dict:
    h, w = shape
    if task == "single_point":
        return {"points": [[rng.randrange(h) + 1, rng.randrange(w) + 1]]}
    if task == "multi_point":
        count = rng.choice(multi_counts)
        indices = rng.sample(range(h * w), count)
        return {"points": [[n // w + 1, n % w + 1] for n in indices]}
    if task == "line_profile":
        options = [(n, step) for n in line_counts for step, extent in (([0, 1], w), ([1, 0], h))
                   if n <= extent]
        if not options:
            raise ValueError("No configured line length fits this field")
        count, step = rng.choice(options)
        return {"start": [rng.randrange(h - (count - 1) * step[0]) + 1,
                          rng.randrange(w - (count - 1) * step[1]) + 1],
                "step": step, "count": count}
    if task == "region_values":
        sizes = [(rh, rw) for rh, rw in region_shapes if rh <= h and rw <= w]
        if not sizes:
            raise ValueError("No configured region fits this field")
        rh, rw = rng.choice(sizes)
        return {"start": [rng.randrange(h - rh + 1) + 1, rng.randrange(w - rw + 1) + 1],
                "size": [rh, rw]}
    raise ValueError(f"Unsupported task: {task}")


def make_record(state_ref: str, z: torch.Tensor, task: str, spec: Mapping, *,
                variant: int = 0, decimals: int = 1, tolerance: float = DEFAULT_TOLERANCE) -> dict:
    if type(decimals) is not int or decimals < 0 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Invalid answer precision/tolerance")
    if 0.5 * 10 ** -decimals >= tolerance:
        raise ValueError("Answer rounding must be strictly finer than the scoring tolerance")
    shape = list(z.shape)
    points = query_points(task, spec, shape)
    values = [float(z[r - 1, c - 1]) for r, c in points]
    answer = [round(value, decimals) or 0.0 for value in values]
    identity = {"state_ref": state_ref, "task": task, "spec": spec, "variant": variant}
    return {"qa_id": json_hash(identity)[:24], "state_ref": state_ref, "task_type": task,
            "grid_shape": shape, "template_variant": variant,
            "question": render_question(task, spec, shape, variant),
            "answer": json.dumps(answer, allow_nan=False),
            "oracle": {"query_spec": dict(spec), "values": values,
                       "value_space": "stored_fp16_z", "absolute_tolerance": tolerance}}


def validate_record(record: Mapping, z: torch.Tensor | None = None) -> None:
    oracle = record["oracle"]
    if oracle["value_space"] != "stored_fp16_z":
        raise ValueError("Unexpected oracle value space")
    points = query_points(record["task_type"], oracle["query_spec"], record["grid_shape"])
    if record["question"] != render_question(record["task_type"], oracle["query_spec"],
                                               record["grid_shape"], record["template_variant"]):
        raise ValueError("Question and audit query specification disagree")
    values = oracle["values"]
    if len(values) != len(points) or any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError("Invalid oracle values")
    tol = oracle["absolute_tolerance"]
    if type(tol) not in (int, float) or not math.isfinite(tol) or tol <= 0:
        raise ValueError("Invalid oracle tolerance")
    if z is not None:
        if list(z.shape) != record["grid_shape"] or values != [float(z[r - 1, c - 1]) for r, c in points]:
            raise ValueError("Oracle does not replay from the source field")
    if not score_answer(record, record["answer"])["all_correct"]:
        raise ValueError("Reference answer fails its own scoring contract")


def parse_answer(text: str) -> tuple[list[float] | None, str | None]:
    """Strict JSON prevents coordinate extraction, substring hits and alternatives."""
    try:
        value = json.loads(text.strip())
    except (ValueError, TypeError, AttributeError):
        return None, "invalid_json"
    if not isinstance(value, list) or not value:
        return None, "not_nonempty_array"
    if any(type(x) not in (int, float) for x in value):
        return None, "nonnumeric_item"
    try:
        values = [float(x) for x in value]
    except OverflowError:
        return None, "nonfinite_number"
    if not all(map(math.isfinite, values)):
        return None, "nonfinite_number"
    return values, None


def score_answer(record: Mapping, prediction: str, *, terminated: bool = True) -> dict:
    expected = record["oracle"]["values"]
    tolerance = float(record["oracle"]["absolute_tolerance"])
    values, error = parse_answer(prediction)
    observed_count = len(values) if values is not None else 0
    if not terminated:
        error = "generation_truncated"
    elif error is None and observed_count != len(expected):
        error = "wrong_value_count"
    valid = error is None
    errors = [abs(a - b) for a, b in zip(values, expected)] if valid else []
    hits = sum(err <= tolerance for err in errors)
    return {"valid": valid, "error": error, "expected_count": len(expected),
            "observed_count": observed_count, "missing_count": max(0, len(expected) - observed_count),
            "extra_count": max(0, observed_count - len(expected)), "correct_values": hits,
            "point_accuracy": hits / len(expected), "all_correct": valid and hits == len(expected),
            "absolute_errors": errors, "parsed_values": values}


def summarize_scores(rows: Sequence[Mapping]) -> dict:
    if not rows:
        raise ValueError("Cannot score an empty evaluation set")

    def aggregate(group):
        n = len(group)
        expected = sum(r["score"]["expected_count"] for r in group)
        return {"questions": n, "values": expected,
                "valid_rate": sum(r["score"]["valid"] for r in group) / n,
                "point_accuracy": sum(r["score"]["correct_values"] for r in group) / expected,
                "all_correct_rate": sum(r["score"]["all_correct"] for r in group) / n,
                "errors": dict(Counter(r["score"]["error"] for r in group if r["score"]["error"]))}

    result = aggregate(rows)
    for key in ("task_type", "shape", "field"):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[key])].append(row)
        result[f"by_{key}"] = {name: aggregate(group) for name, group in sorted(groups.items())}
    result["macro_task_point_accuracy"] = sum(v["point_accuracy"] for v in result["by_task_type"].values()) / len(result["by_task_type"])
    return result
