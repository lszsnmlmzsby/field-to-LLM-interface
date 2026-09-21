"""Score saved generated answers; no language-model weights or GPU required."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensor_compression.downstream.point_readout import score_answer, summarize_scores
from tensor_compression.downstream.point_readout_data import PointReadoutDataset, read_jsonl, write_json


def score_predictions(dataset, predictions):
    by_id = {}
    allowed = {row["qa_id"] for row in dataset.records}
    for row in predictions:
        key = row["qa_id"]
        if key not in allowed or key in by_id:
            raise ValueError(f"Unknown or duplicate prediction ID: {key}")
        if not isinstance(row.get("prediction"), str) or type(row.get("terminated", True)) is not bool:
            raise ValueError("Predictions require a string and an optional boolean terminated flag")
        by_id[key] = row
    scored = []
    for row in dataset.records:
        prediction = by_id.get(row["qa_id"], {"prediction": "", "terminated": False})
        score = score_answer(row, prediction["prediction"], terminated=prediction.get("terminated", True))
        if row["qa_id"] not in by_id:
            score["error"] = "missing_prediction"
        scored.append({"qa_id": row["qa_id"], "task_type": row["task_type"],
                       "shape": "x".join(map(str, row["grid_shape"])),
                       "field": dataset.states[row["state_ref"]]["field"], "score": score})
    return summarize_scores(scored)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    cli = parser.parse_args()
    # This path is never opened: scoring uses the immutable manifest and labels.
    dataset = PointReadoutDataset(cli.qa_dir, ".", cli.split, cache_size=0)
    try:
        metrics = score_predictions(dataset, read_jsonl(cli.predictions))
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
