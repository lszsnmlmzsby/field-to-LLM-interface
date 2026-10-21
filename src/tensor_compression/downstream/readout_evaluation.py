"""Additional readout diagnostics; none of these change the principal QA score.

Annotations come from immutable queries and source manifests, never neural inputs.
Language pairs require both answers to be correct: agreement alone is insufficient.
"""
from __future__ import annotations

import json
import copy
import random
from collections import defaultdict

from .point_readout import json_hash, score_answer, summarize_evaluation, summarize_scores


def count_bucket(value):
    """Inclusive, disjoint bins suitable for counts and areas of arbitrary shapes."""
    value = int(value)
    if value < 1:
        raise ValueError("Diagnostic counts must be positive")
    for lower, upper in ((1, 1), (2, 4), (5, 16), (17, 64), (65, 256),
                         (257, 1024), (1025, 4096), (4097, 16384)):
        if value <= upper:
            return str(lower) if lower == upper else f"{lower}-{upper}"
    return "16385+"


def evaluation_annotations(record, block_shape=(1, 1)):
    from .field_query_program import query_diagnostics

    tags = dict(query_diagnostics(record, block_shape=block_shape))
    for key in ("semantic_id", "query_protocol", "language_group", "template_family",
                "capability", "difficulty", "input_pair_id", "input_pair_member"):
        if key in record:
            tags[key] = record[key]
    tags["state_ref"] = record["state_ref"]
    for count, bucket in (("query_point_count", "query_points_bucket"),
                          ("query_extent_cells", "query_area_bucket"),
                          ("output_count", "output_length_bucket")):
        tags[bucket] = count_bucket(tags[count])
    # Pair membership must never hide a mismatched oracle or another field.
    tags["label_signature"] = json_hash({"values": record["oracle"]["values"],
        "answer_kind": record["oracle"]["answer_kind"],
        "valid_coordinates": record["oracle"].get("valid_coordinates", []),
        "absolute_tolerance": record["oracle"]["absolute_tolerance"]})
    return tags


def paired_language_metrics(rows):
    if not any(row.get("language_group") == "heldout" for row in rows):
        return None
    groups = defaultdict(dict)
    for row in rows:
        if "semantic_id" not in row or row.get("language_group") not in ("seen", "heldout"):
            continue
        key = (row["state_ref"], row["semantic_id"])
        family = row["language_group"]
        if family in groups[key]:
            raise ValueError(f"Duplicate {family} member of a language pair: {key}")
        groups[key][family] = row
    complete = [pair for pair in groups.values() if set(pair) == {"seen", "heldout"}]
    if not any("heldout" in pair for pair in groups.values()):
        return None
    counters = defaultdict(int)
    for pair in complete:
        seen, heldout = pair["seen"], pair["heldout"]
        if seen["label_signature"] != heldout["label_signature"]:
            raise ValueError("Paired language questions disagree on the reference answer")
        a, b = bool(seen["score"]["all_correct"]), bool(heldout["score"]["all_correct"])
        counters["seen_correct"] += a
        counters["heldout_correct"] += b
        counters["both_correct"] += a and b
        counters["seen_correct_heldout_wrong"] += a and not b
        counters["seen_wrong_heldout_correct"] += not a and b
        # Valid JSON/length is necessary for agreement, but agreement is not accuracy.
        counters["valid_prediction_agreement"] += bool(
            seen["score"]["valid"] and heldout["score"]["valid"] and
            seen["score"]["parsed_values"] == heldout["score"]["parsed_values"])
    n = len(complete)
    result = {"complete_pairs": n, "incomplete_pairs": len(groups) - n,
              "primary_metric": "both_correct_rate"}
    for key in ("seen_correct", "heldout_correct", "both_correct", "seen_correct_heldout_wrong",
                "seen_wrong_heldout_correct", "valid_prediction_agreement"):
        result[f"{key}_count"] = counters[key]
        result[f"{key}_rate"] = counters[key] / n if n else None
    return result


def summarize_readout_evaluation(rows, train_shapes):
    result = summarize_evaluation(rows, train_shapes)
    for key in ("capability", "difficulty", "language_group", "template_family", "query_area_bucket",
                "query_points_bucket", "output_length_bucket", "block_relation", "operation_count", "roughness_bin"):
        groups = defaultdict(list)
        for row in rows:
            if key in row:
                groups[str(row[key])].append(row)
        if groups:
            if sum(map(len, groups.values())) != len(rows):
                raise ValueError(f"Incomplete diagnostic annotation: {key}")
            result[f"by_{key}"] = {name: summarize_scores(group) for name, group in sorted(groups.items())}
    pairs = paired_language_metrics(rows)
    if pairs is not None:
        result["paired_language"] = pairs
    field_pairs = paired_field_metrics(rows)
    if field_pairs is not None:
        result["paired_fields"] = field_pairs
    return result


def zero_baseline_metrics(records, states):
    """Score fixed zero answers only on numerical tasks using the actual tolerances.

    This baseline is deliberately given each question's required output length.
    It measures value-distribution shortcuts, not the ability to parse a prompt.
    Coordinate tasks are excluded; their invalid zero coordinates are uninformative.
    """
    rows = []
    for row in records:
        if row["oracle"]["answer_kind"] == "coordinate":
            continue
        state = states[row["state_ref"]]
        prediction = json.dumps([0.0] * len(row["oracle"]["values"]))
        entry = {"task_type": row["task_type"], "shape": "x".join(map(str, row["grid_shape"])),
                 "field": state["field"], "score": score_answer(row, prediction)}
        entry.update({key: state[key] for key in ("source_id", "family", "is_static", "is_constant") if key in state})
        rows.append(entry)
    return {"name": "constant_zero_with_oracle_output_length", "diagnostic_only": True,
            "numerical_questions": len(rows), "excluded_coordinate_questions": len(records) - len(rows),
            "metrics": summarize_evaluation(rows, []) if rows else None}


def paired_field_metrics(rows):
    groups = defaultdict(dict)
    for row in rows:
        if "input_pair_id" not in row:
            continue
        member = row["input_pair_member"]
        if member not in ("original", "alternative") or member in groups[row["input_pair_id"]]:
            raise ValueError("Duplicate or invalid field-pair member")
        groups[row["input_pair_id"]][member] = row
    if not groups:
        return None
    pairs = [pair for pair in groups.values() if set(pair) == {"original", "alternative"}]
    both = sum(all(row["score"]["all_correct"] for row in pair.values()) for pair in pairs)
    changed = sum(bool(pair["original"]["score"]["valid"] and pair["alternative"]["score"]["valid"] and
                       pair["original"]["score"]["parsed_values"] != pair["alternative"]["score"]["parsed_values"])
                  for pair in pairs)
    return {"complete_pairs": len(pairs), "incomplete_pairs": len(groups) - len(pairs),
            "both_correct_count": both, "both_correct_rate": both / len(pairs) if pairs else None,
            "valid_changed_prediction_count": changed,
            "valid_changed_prediction_rate": changed / len(pairs) if pairs else None,
            "primary_metric": "both_correct_rate", "diagnostic_only": True}


def _disjoint_reference_answers(first, second):
    a, b = first["oracle"], second["oracle"]
    if a["answer_kind"] != b["answer_kind"] or len(a["values"]) != len(b["values"]):
        raise ValueError("Matched-query pair changed the answer structure")
    if a["answer_kind"] != "coordinate":
        tolerance_sum = a["absolute_tolerance"] + b["absolute_tolerance"]
        return any(abs(x - y) > tolerance_sum for x, y in zip(a["values"], b["values"]))
    from .field_query_program import coordinate_is_accepted, selector_points
    if "query_program" in a:
        points = selector_points(a["query_program"]["expression"]["selector"], first["grid_shape"])
        return not any(coordinate_is_accepted(first, point) and coordinate_is_accepted(second, point)
                       for point in points)
    return not (set(map(tuple, a["valid_coordinates"])) & set(map(tuple, b["valid_coordinates"])))


class FieldPairDataset:
    """Bounded grounding diagnostic on two genuine fields from one eval split.

    No raw data or QA file is edited. The underlying dataset identity stays the
    checkpoint identity; the selected queries are saved separately by the caller.
    Selection uses only base.records, never the all-split states dictionary.
    """
    suite = "field-pairs"

    def __init__(self, base, *, max_pairs=128, max_candidates=4096, seed=0):
        from .field_query_program import QUERY_PROTOCOL, make_program_record
        from .point_readout import make_record, validate_record

        if any(type(value) is not int or value < 1 for value in (max_pairs, max_candidates)):
            raise ValueError("Field-pair limits must be positive integers")
        # Some dataset implementations expose only per-state split provenance.
        # Check every participating state rather than trusting an optional wrapper
        # attribute or the all-split state dictionary's contents.
        actual_splits = {base.states[row["state_ref"]].get("split") for row in base.records}
        if len(actual_splits) != 1 or not actual_splits <= {"val", "test"}:
            raise ValueError("Grounding diagnostic requires states from one validation or test split")
        self.split = next(iter(actual_splits))
        if getattr(base, "split", self.split) != self.split:
            raise ValueError("Grounding diagnostic split disagrees with participating states")
        self.base, self.records = base, []
        self.metadata, self.identity = base.metadata, base.identity
        self.states = base.states
        self.state_indices, shapes = {}, defaultdict(list)
        for index, row in enumerate(base.records):
            if row["state_ref"] not in self.state_indices:
                self.state_indices[row["state_ref"]] = index
                shapes[tuple(row["grid_shape"])].append(row["state_ref"])
        eligible = [index for index, row in enumerate(base.records)
                    if len(shapes[tuple(row["grid_shape"])]) > 1]
        rng = random.Random(seed)
        rng.shuffle(eligible)
        attempted, selected_ids = 0, set()
        for index in eligible:
            if len(self.records) // 2 >= max_pairs or attempted >= max_candidates:
                break
            row = base.records[index]
            if row["qa_id"] in selected_ids:
                continue
            candidates = list(shapes[tuple(row["grid_shape"])])
            rng.shuffle(candidates)
            for other_ref in candidates[:32]:
                if attempted >= max_candidates:
                    break
                if other_ref == row["state_ref"]:
                    continue
                attempted += 1
                _, other_field = base[self.state_indices[other_ref]]
                z = other_field[0]
                if row.get("query_protocol") == QUERY_PROTOCOL:
                    other = make_program_record(other_ref, z, row["task_type"], row["oracle"]["query_program"],
                        row["template_family"], decimals=row["oracle"]["answer_decimals"],
                        tolerance=row["oracle"]["absolute_tolerance"], language_group=row["language_group"])
                else:
                    other = make_record(other_ref, z, row["task_type"], row["oracle"]["query_spec"],
                        variant=row["template_variant"], decimals=int(base.metadata.get("scoring", {}).get("answer_decimals", 1)),
                        tolerance=row["oracle"]["absolute_tolerance"])
                if other["question"] != row["question"] or other["grid_shape"] != row["grid_shape"]:
                    raise ValueError("Grounding pair must preserve the exact question and shape")
                if other["qa_id"] in selected_ids or not _disjoint_reference_answers(row, other):
                    continue
                _, original_field = base[index]
                validate_record(row, original_field[0])
                validate_record(other, z)
                pair_id = json_hash({"original": row["qa_id"], "alternative": other["qa_id"]})[:24]
                for member, record in (("original", row), ("alternative", other)):
                    entry = copy.deepcopy(record)
                    entry.update(input_pair_id=pair_id, input_pair_member=member)
                    self.records.append(entry)
                    selected_ids.add(entry["qa_id"])
                break
        self.selection = {"status": "available" if self.records else "unavailable",
            "diagnostic_only": True, "eligible_queries": len(eligible), "attempted_candidates": attempted,
            "accepted_pairs": len(self.records) // 2, "requested_pairs": max_pairs,
            "max_candidates": max_candidates, "max_alternatives_per_query": 32,
            "queries_sha256": json_hash(self.records), "dataset_identity": self.identity,
            "seed": seed, "label_rule": "nonoverlapping_tolerance_intervals_or_coordinate_tie_sets",
            "selection_limitation": "Selected changed-answer pairs; not a representative overall accuracy estimate"}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from .point_readout import validate_record
        row = self.records[index]
        _, field = self.base[self.state_indices[row["state_ref"]]]
        validate_record(row, field[0])
        return row, field

    def close(self):
        self.base.close()
