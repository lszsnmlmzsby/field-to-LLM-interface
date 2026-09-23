"""Evaluate untouched Qwen on the same generative QA, with the full field as text."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from scripts import train_point_readout as trainer
from tensor_compression.downstream.point_readout import build_prompt, score_answer, summarize_evaluation
from tensor_compression.downstream.point_readout_data import (
    PointReadoutDataset, load_config, read_jsonl, resolve_path, write_json,
)

BASELINE = "frozen_qwen_full_serialized_field_v1"


def serialized_messages(record, field):
    """Only the public question, shape and complete input field enter the prompt."""
    h, w = record["grid_shape"]
    grid = field.detach().cpu().float()
    if grid.ndim == 3 and grid.shape[0] == 1:
        grid = grid[0]
    if list(grid.shape) != [h, w] or not bool(torch.isfinite(grid).all()):
        raise ValueError("Serialized field must match the complete public grid shape")
    # JSON float representations round-trip the stored FP16->FP32 values exactly.
    # Never select queried points, re-normalize, or round to the answer precision.
    matrix = json.dumps(grid.tolist(), allow_nan=False, separators=(",", ":"))
    instructions = build_prompt(record).split("\n", 1)[1].removesuffix("\nAnswer:")
    content = (f"The supplied field is a {h} by {w} standardized numerical grid.\n"
               f"Grid values as a JSON array of rows (top to bottom, each row left to right):\n{matrix}\n\n"
               f"{instructions}")
    return [{"role": "system", "content": trainer.SYSTEM_MESSAGE}, {"role": "user", "content": content}]


def encode_serialized_prompt(record, field, tokenizer, max_prompt_tokens):
    prompt = trainer.native_chat_ids(tokenizer, serialized_messages(record, field), generation_prompt=True)
    if not prompt or len(prompt) > max_prompt_tokens:
        raise ValueError(f"Serialized prompt {record['qa_id']} has {len(prompt)} tokens; "
                         f"limit={max_prompt_tokens}. Truncation is forbidden.")
    return prompt


def score_complete_predictions(dataset, predictions):
    allowed = {r["qa_id"] for r in dataset.records}
    by_id = {}
    for p in predictions:
        key = p["qa_id"]
        if key not in allowed or key in by_id:
            raise ValueError("Unknown or duplicate prediction ID in paired benchmark")
        if not isinstance(p.get("prediction"), str) or type(p.get("terminated")) is not bool:
            raise ValueError("Paired predictions need text and an explicit boolean terminated flag")
        by_id[key] = p
    if set(by_id) != allowed:
        raise ValueError("Paired benchmark requires predictions for every question of the selected split")
    scored = []
    for row in dataset.records:
        p = by_id[row["qa_id"]]
        scored.append({"qa_id": row["qa_id"], "task_type": row["task_type"],
                       "shape": "x".join(map(str, row["grid_shape"])),
                       "field": dataset.states[row["state_ref"]]["field"],
                       "score": score_answer(row, p["prediction"], terminated=p["terminated"])})
    return summarize_evaluation(scored, dataset.metadata["train_shapes"])


def compare_metrics(baseline, interface):
    def pair(left, right):
        if left is None or right is None:
            return None
        key = "macro_task_answer_accuracy" if "macro_task_answer_accuracy" in left else "answer_accuracy"
        return {"questions": left["questions"], "baseline": left[key], "interface": right[key],
                "interface_minus_baseline_pp": 100 * (right[key] - left[key]),
                "baseline_unit_accuracy": left["unit_accuracy"], "interface_unit_accuracy": right["unit_accuracy"]}
    def shape_group(key):
        result = pair(baseline[key], interface[key])
        if result is not None:
            result["by_task_type"] = {task: pair(value, interface[key]["by_task_type"][task])
                                      for task, value in baseline[key]["by_task_type"].items()}
        return result
    return {"metric": "macro_task_answer_accuracy (per task: answer_accuracy)",
            "overall": pair(baseline, interface),
            **{key: shape_group(key) for key in ("seen_shapes", "heldout_shapes")},
            "by_task_type": {task: pair(value, interface["by_task_type"][task])
                             for task, value in baseline["by_task_type"].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/field_to_llm_point_readout.yaml"))
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--qa-dir")
    parser.add_argument("--hdf5-path")
    parser.add_argument("--model-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-prompt-tokens", type=int, default=16384)
    parser.add_argument("--compare-predictions", help="Interface predictions with a matching adjacent contract.json")
    parser.add_argument("--audit-only", action="store_true", help="Check data and tokenize all prompts without loading model weights")
    parser.add_argument("--verbose", action="store_true")
    cli = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Use single-device python, not multi-rank torchrun")
    if cli.max_prompt_tokens <= 0:
        parser.error("--max-prompt-tokens must be positive")
    config = load_config(cli.config, cli.profile)
    trainer.validate_config(config)
    qa_dir = resolve_path(cli.qa_dir or config["data"]["qa_dir"])
    hdf5_path = resolve_path(cli.hdf5_path or os.environ.get("PDEBENCH_HDF5") or config["data"]["hdf5_path"])
    output = resolve_path(cli.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty benchmark output directory")
    output.mkdir(parents=True, exist_ok=True)
    dataset = PointReadoutDataset(qa_dir, hdf5_path, cli.split, cache_size=config["data"].get("input_cache_size", 8))
    try:
        print(f"data_audit split={cli.split} {dataset.audit()}", flush=True)
        interface_metrics = None
        if cli.compare_predictions:
            comparison_path = resolve_path(cli.compare_predictions)
            comparison_contract = json.loads((comparison_path.parent / "contract.json").read_text(encoding="utf-8"))
            interface_metrics = score_complete_predictions(dataset, read_jsonl(comparison_path))
        args = trainer.model_args(config, cli.model_dir, checkpointing=False)
        trainer.core.apply_runtime_environment(args)
        trainer.core.seed_everything(args.seed)
        tokenizer = trainer.core.load_tokenizer(args)
        lengths = []
        for index in range(len(dataset)):
            row, field = dataset[index]
            lengths.append(len(encode_serialized_prompt(row, field, tokenizer, cli.max_prompt_tokens)))
        audit = {"dataset": dataset.identity, "split": cli.split, "questions": len(dataset),
                 "prompt_tokens_min": min(lengths), "prompt_tokens_max": max(lengths),
                 "prompt_tokens_mean": sum(lengths) / len(lengths), "truncation": False,
                 "serialization": "complete row-major JSON, exact stored FP16->FP32 values"}
        write_json(output / "data_and_prompt_audit.json", audit)
        print(f"prompt_audit questions={len(dataset)} tokens_min={min(lengths)} tokens_max={max(lengths)}", flush=True)
        if cli.audit_only:
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cli.device == "auto" else torch.device(cli.device)
        if device.type == "cuda" and config["model"]["torch_dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("Selected GPU does not support BF16")
        llm, dtype = trainer.core.load_llm_with_bounded_host_memory(args, device)
        llm.requires_grad_(False)
        llm.eval()
        context_limit = getattr(llm.config, "max_position_embeddings", None)
        if context_limit and max(lengths) + config["evaluation"]["max_new_tokens"] > context_limit:
            raise ValueError("Full matrix plus answer exceeds model context; truncation is forbidden")
        # No field encoder, adapter or bridge is ever constructed/attached here.
        identity = trainer.run_identity(config, dataset, llm, tokenizer)
        identity["model_asset"] = trainer.model_asset_identity(args.model_name_or_path, llm, verbose=cli.verbose)
        if interface_metrics is not None and comparison_contract != identity:
            raise ValueError("Interface contract differs in dataset/model/tokenizer/recipe; comparison refused")
        contract = {"baseline": BASELINE, "reference_interface_identity": identity, **audit,
                    "decoding": "full-vocabulary greedy", "stop_token_ids": trainer.generation_stop_ids(llm, tokenizer),
                    "max_prompt_tokens": cli.max_prompt_tokens, "max_new_tokens": config["evaluation"]["max_new_tokens"],
                    "use_cache": config["evaluation"].get("use_cache", True),
                    "scoring": dataset.metadata["scoring"], "trainable_parameters": 0}
        if interface_metrics is not None:
            contract["comparison_predictions_sha256"] = trainer.core.sha256_file(comparison_path)
        write_json(output / "baseline_contract.json", contract)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        predictions = []
        with (output / f"{cli.split}_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for index in range(len(dataset)):
                row, field = dataset[index]
                prompt = encode_serialized_prompt(row, field, tokenizer, cli.max_prompt_tokens)
                generated = trainer.generate_from_prompt(llm, tokenizer, prompt, device, dtype,
                    max_new_tokens=config["evaluation"]["max_new_tokens"], use_cache=config["evaluation"].get("use_cache", True))
                prediction = {"qa_id": row["qa_id"], "task_type": row["task_type"], "prompt_tokens": len(prompt),
                              "shape": "x".join(map(str, row["grid_shape"])),
                              "field": dataset.states[row["state_ref"]]["field"], **generated,
                              "score": score_answer(row, generated["prediction"], terminated=generated["terminated"])}
                predictions.append(prediction)
                handle.write(json.dumps(prediction, allow_nan=False) + "\n")
                interval = 100 if cli.verbose else 500
                if (index + 1) % interval == 0:
                    print(f"baseline questions={index + 1}/{len(dataset)}", flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        metrics = score_complete_predictions(dataset, predictions)
        metrics.update(elapsed_seconds=time.perf_counter() - start,
                       peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
                       generated_tokens=sum(p["generated_tokens"] for p in predictions), **audit)
        write_json(output / f"{cli.split}_metrics.json", metrics)
        trainer.print_metric_summary("baseline", metrics)
        if interface_metrics is not None:
            comparison = compare_metrics(metrics, interface_metrics)
            write_json(output / "comparison.json", comparison)
            trainer.print_metric_summary("interface", interface_metrics)
            for name in ("overall", "seen_shapes", "heldout_shapes"):
                if comparison[name] is not None:
                    print(f"comparison {name} delta_pp={comparison[name]['interface_minus_baseline_pp']:+.2f}", flush=True)
            print("task (all shapes)             baseline  interface  delta_pp", flush=True)
            for task, row in comparison["by_task_type"].items():
                print(f"{task:28s} {row['baseline']:8.2%} {row['interface']:10.2%} "
                      f"{row['interface_minus_baseline_pp']:+8.2f}", flush=True)
        print(f"results={output}", flush=True)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
