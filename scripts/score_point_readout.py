"""Score saved generated answers; no language-model weights or GPU required."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensor_compression.downstream.point_readout import score_answer
from tensor_compression.downstream.point_readout_data import PointReadoutDataset, read_jsonl, write_json
from tensor_compression.downstream.readout_evaluation import (
    evaluation_annotations, summarize_readout_evaluation, zero_baseline_metrics,
)


def score_predictions(dataset, predictions, *, block_shape=None):
    by_id = {}
    input_modes, shapes = set(), set()
    allowed = {row["qa_id"] for row in dataset.records}
    for row in predictions:
        key = row["qa_id"]
        if key not in allowed or key in by_id:
            raise ValueError(f"Unknown or duplicate prediction ID: {key}")
        if not isinstance(row.get("prediction"), str) or type(row.get("terminated", True)) is not bool:
            raise ValueError("Predictions require a string and an optional boolean terminated flag")
        by_id[key] = row
        input_modes.add(row.get("input_mode", "correct"))
        if "block_shape" in row:
            shape = row["block_shape"]
            if len(shape) != 2 or any(type(value) is not int or value < 1 for value in shape):
                raise ValueError("Prediction block_shape must have two positive integers")
            shapes.add(tuple(shape))
    if len(input_modes) > 1 or not input_modes <= {"correct", "disabled"}:
        raise ValueError("Cannot mix input modes in one evaluation")
    if len(shapes) > 1:
        raise ValueError("Cannot mix block shapes in one evaluation")
    if block_shape is not None and shapes and shapes != {tuple(block_shape)}:
        raise ValueError("Requested block shape disagrees with prediction provenance")
    block_shape = tuple(block_shape or next(iter(shapes), (1, 1)))
    if len(block_shape) != 2 or any(type(value) is not int or value < 1 for value in block_shape):
        raise ValueError("block_shape must have two positive integers")
    scored = []
    for row in dataset.records:
        prediction = by_id.get(row["qa_id"], {"prediction": "", "terminated": False})
        score = score_answer(row, prediction["prediction"], terminated=prediction.get("terminated", True))
        if row["qa_id"] not in by_id:
            score["error"] = "missing_prediction"
        state = dataset.states[row["state_ref"]]
        scored.append({"qa_id": row["qa_id"], "task_type": row["task_type"],
                       "shape": "x".join(map(str, row["grid_shape"])),
                       "field": state["field"], "score": score,
                       **evaluation_annotations(row, block_shape),
                       **{key: state[key] for key in ("source_id", "family", "is_static", "is_constant", "roughness_bin") if key in state}})
    metrics = summarize_readout_evaluation(scored, dataset.metadata["train_shapes"])
    metrics["zero_value_baseline"] = zero_baseline_metrics(dataset.records, dataset.states)
    suite, input_mode = getattr(dataset, "suite", "main"), next(iter(input_modes), "correct")
    metrics.update(input_mode=input_mode, evaluation_suite=suite,
                   diagnostic_only=input_mode != "correct" or suite != "main")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--eval-suite", choices=("main", "language"), default="main")
    parser.add_argument("--block-shape", nargs=2, type=int, metavar=("H", "W"),
                        help="Needed only for old prediction files without block_shape provenance")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    cli = parser.parse_args()
    # This path is never opened: scoring uses the immutable manifest and labels.
    options = {"suite": cli.eval_suite} if cli.eval_suite != "main" else {}
    dataset = PointReadoutDataset(cli.qa_dir, ".", cli.split, cache_size=0, **options)
    try:
        metrics = score_predictions(dataset, read_jsonl(cli.predictions), block_shape=cli.block_shape)
        destination = Path(cli.output)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite metrics: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination, metrics)
        print(json.dumps(metrics), flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
