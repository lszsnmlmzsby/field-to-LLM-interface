"""Compare completed standalone evaluations on exactly the same QA split."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(reference, compressed, split="val"):
    directories = [Path(reference), Path(compressed)]
    contracts = [read(d / "contract.json") for d in directories]
    blocks = [c["recipe"]["memory"].get("block_shape", [1, 1]) for c in contracts]
    normalized = copy.deepcopy(contracts)
    for contract in normalized:
        contract["recipe"]["memory"].pop("block_shape", None)
    if normalized[0] != normalized[1]:
        raise ValueError("Dataset, model, tokenizer and training recipes must match apart from block_shape")
    metrics = [read(d / f"{split}_metrics.json") for d in directories]
    ids = []
    tokens = []
    for directory in directories:
        rows = [json.loads(line) for line in (directory / f"{split}_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        keys = [r["qa_id"] for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate question IDs")
        ids.append(set(keys))
        tokens.append([r["field_tokens"] for r in rows])
    if not ids[0] or ids[0] != ids[1] or any(m["questions"] != len(ids[0]) for m in metrics):
        raise ValueError("Evaluations must cover identical complete question sets")
    print(f"split={split} questions={len(ids[0])} reference={blocks[0]} compressed={blocks[1]}")
    print("task                         ref_answer  cmp_answer  delta_pp  ref_unit  cmp_unit")
    for task, left in metrics[0]["by_task_type"].items():
        right = metrics[1]["by_task_type"][task]
        print(f"{task:28s} {left['answer_accuracy']:9.2%} {right['answer_accuracy']:11.2%} "
              f"{100*(right['answer_accuracy']-left['answer_accuracy']):+9.2f} "
              f"{left['unit_accuracy']:9.2%} {right['unit_accuracy']:9.2%}")
    for group in (None, "seen_shapes", "heldout_shapes"):
        pair = [m[group] if group else m for m in metrics]
        if all(p is not None for p in pair):
            print(f"{group or 'all'} answer=" + "/".join(f"{p['macro_task_answer_accuracy']:.2%}" for p in pair))
    for block, metric, lengths in zip(blocks, metrics, tokens):
        print(f"block={block} field_tokens={min(lengths)}..{max(lengths)} "
              f"elapsed_seconds={metric['elapsed_seconds']:.1f} peak_allocated_gib={metric['peak_allocated_gib']} "
              f"generated_tokens={metric['generated_tokens']}")
    print("Runtime is exploratory: answer lengths can differ; use repeated controlled timing for formal speed claims.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Standalone --evaluate-only output directory")
    parser.add_argument("--compressed", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    cli = parser.parse_args()
    compare(cli.reference, cli.compressed, cli.split)


if __name__ == "__main__":
    main()
