"""Interactive real-field QA: serialized Qwen, trained interface, or both."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from scripts import train_point_readout as trainer
from scripts.benchmark_point_readout import serialized_messages
from tensor_compression.downstream.point_readout_data import PointReadoutDataset, load_config, resolve_path

FREE_SYSTEM = ("You answer questions about a supplied numerical field. Answer naturally in the user's "
               "language. Values are standardized, not in original physical units. "
               "Coordinates are one-based [row, column] within the supplied grid.")


def initial_messages(question, shape, field, mode, response_format):
    # No reference answer, task spec or oracle enters the conversation.
    record = {"grid_shape": list(shape), "question": question}
    messages = (serialized_messages(record, field) if mode == "baseline"
                else trainer.chat_messages(record))
    if response_format == "free":
        h, w = shape
        if mode == "baseline":
            matrix = json.dumps(field.squeeze(0).tolist(), allow_nan=False, separators=(",", ":"))
            intro = f"The supplied field is a {h} by {w} standardized numerical grid.\nGrid rows:\n{matrix}"
        else:
            intro = f"The supplied field memory represents a {h} by {w} standardized numerical grid."
        messages = [{"role": "system", "content": FREE_SYSTEM},
                    {"role": "user", "content": f"{intro}\n\nQuestion: {question}"}]
    return messages


def conversation_messages(history, question, shape, field, mode, response_format):
    if history:
        return history + [{"role": "user", "content": question}]
    return initial_messages(question, shape, field, mode, response_format)


@torch.inference_mode()
def reply(llm, sidecar, tokenizer, messages, field, mode, device, dtype, max_prompt_tokens, max_new_tokens):
    prompt = trainer.native_chat_ids(tokenizer, messages, generation_prompt=True)
    context = getattr(llm.config, "max_position_embeddings", None)
    if len(prompt) > max_prompt_tokens or (context and len(prompt) + max_new_tokens > context):
        raise ValueError("Conversation exceeds the context budget; use /reset. No text was truncated.")
    try:
        if sidecar is not None:
            sidecar.clear()
            with trainer.core.autocast_context(device, dtype):
                sidecar.bind(field.unsqueeze(0).to(device) if mode == "interface" else None,
                             mode="correct" if mode == "interface" else "no_tensor")
        result = trainer.generate_from_prompt(llm, tokenizer, prompt, device, dtype,
                                              max_new_tokens=max_new_tokens)
        return {**result, "prompt_tokens": len(prompt)}
    finally:
        if sidecar is not None:
            sidecar.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/field_to_llm_point_readout.yaml"))
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="full")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--qa-dir")
    parser.add_argument("--hdf5-path")
    parser.add_argument("--model-dir")
    parser.add_argument("--block-shape", type=int, nargs=2, metavar=("H", "W"))
    parser.add_argument("--checkpoint", help="Point-readout best.pt or last.pt; required for interface/both")
    parser.add_argument("--mode", choices=("baseline", "interface", "both"), default="both")
    parser.add_argument("--response-format", choices=("json", "free"), default="json")
    parser.add_argument("--field-index", type=int, default=0, help="Zero-based unique field index in selected split")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-prompt-tokens", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--single-turn", action="store_true", help="Do not retain conversation history")
    parser.add_argument("--question", help="Answer one question and exit")
    parser.add_argument("--transcript", help="Write a new JSONL transcript (refuses overwrite)")
    cli = parser.parse_args()
    if cli.mode != "baseline" and not cli.checkpoint:
        parser.error("--checkpoint is required for interface/both")
    if min(cli.max_prompt_tokens, cli.max_new_tokens) <= 0:
        parser.error("Token budgets must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Use python on one device, not multi-rank torchrun")
    if cli.question is not None and (not cli.question.strip() or cli.question.startswith("/")):
        parser.error("--question must be a nonempty question, not an interactive command")
    config = load_config(cli.config, cli.profile)
    if cli.block_shape is not None:
        config["memory"]["block_shape"] = cli.block_shape
    trainer.validate_config(config)
    dataset = PointReadoutDataset(resolve_path(cli.qa_dir or config["data"]["qa_dir"]),
        resolve_path(cli.hdf5_path or os.environ.get("PDEBENCH_HDF5") or config["data"]["hdf5_path"]), cli.split)
    transcript = None
    try:
        refs = list(dict.fromkeys(row["state_ref"] for row in dataset.records))
        if not 0 <= cli.field_index < len(refs):
            raise ValueError(f"--field-index must be in 0..{len(refs)-1}")
        if cli.transcript:
            path = resolve_path(cli.transcript)
            path.parent.mkdir(parents=True, exist_ok=True)
            transcript = path.open("x", encoding="utf-8")
        args = trainer.model_args(config, cli.model_dir, checkpointing=False)
        trainer.core.apply_runtime_environment(args)
        trainer.core.seed_everything(args.seed)
        tokenizer = trainer.core.load_tokenizer(args)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cli.device == "auto" else torch.device(cli.device)
        llm, dtype = trainer.core.load_llm_with_bounded_host_memory(args, device)
        llm.requires_grad_(False)
        llm.eval()
        sidecar = None
        if cli.mode != "baseline":
            identity = trainer.run_identity(config, dataset, llm, tokenizer)
            identity["model_asset"] = trainer.model_asset_identity(args.model_name_or_path, llm)
            checkpoint = torch.load(resolve_path(cli.checkpoint), map_location="cpu", weights_only=True)
            trainer.validate_resume(checkpoint, identity)
            encoder, spatial, _, _ = trainer.core.build_scratch_memory_components(args, (1, args.patch_size, args.patch_size))
            sidecar, _ = trainer.core.build_sidecar(llm, spatial, args, device, encoder)
            sidecar.load_state_dict(checkpoint["sidecar"], strict=True)
            sidecar.eval()
            del checkpoint
        modes = ("baseline", "interface") if cli.mode == "both" else (cli.mode,)
        histories = {mode: [] for mode in modes}
        index = cli.field_index

        def show_field():
            state = dataset.states[refs[index]]
            print(f"field={index}/{len(refs)-1} variable={state['field']} shape={state['grid_shape']} "
                  f"split={cli.split} source={refs[index]}", flush=True)

        show_field()
        print("/field N: select field; /grid: show standardized matrix; /examples: list questions; "
              "/reset: clear history; /quit: exit. Other input is your question.\n"
              f"format={cli.response_format}; history={'off' if cli.single_turn else 'on'}; "
              "custom chat is exploratory, not an automatically scored benchmark.", flush=True)
        while True:
            try:
                question = cli.question if cli.question is not None else input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                if cli.question is not None:
                    break
                continue
            if question == "/quit":
                break
            if question == "/reset":
                histories = {mode: [] for mode in modes}
                print("History cleared.")
                continue
            if question.startswith("/field "):
                try:
                    selected = int(question.split(maxsplit=1)[1])
                    if not 0 <= selected < len(refs):
                        raise ValueError()
                    index = selected
                    histories = {mode: [] for mode in modes}
                    show_field()
                except ValueError:
                    print(f"Use /field N with N in 0..{len(refs)-1}")
                continue
            if question == "/examples":
                for row in dataset.records:
                    if row["state_ref"] == refs[index]:
                        print(f"{row['task_type']}: {row['question']}")
                continue
            field = dataset.field(refs[index]).unsqueeze(0)
            if question == "/grid":
                print(json.dumps(field.squeeze(0).tolist(), allow_nan=False))
                continue
            state = dataset.states[refs[index]]
            entry = {"state_ref": refs[index], "field_index": index, "split": cli.split,
                     "dataset": dataset.identity, "checkpoint": cli.checkpoint,
                     "response_format": cli.response_format, "single_turn": cli.single_turn,
                     "question": question, "responses": {}}
            # Commit histories only after both answers succeed.
            pending = {}
            try:
                for mode in modes:
                    messages = conversation_messages(histories[mode], question, state["grid_shape"],
                                                       field, mode, cli.response_format)
                    result = reply(llm, sidecar, tokenizer, messages, field, mode, device, dtype,
                                   cli.max_prompt_tokens, cli.max_new_tokens)
                    entry["responses"][mode] = {**result, "messages": messages}
                    pending[mode] = messages + [{"role": "assistant", "content": result["prediction"]}]
                    print(f"{mode}> {result['prediction']}", flush=True)
                    if not result["terminated"]:
                        print("[Reached generation limit; answer may be incomplete.]")
                if not cli.single_turn:
                    histories = pending
                if transcript:
                    transcript.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                    transcript.flush()
            except ValueError as exc:
                if cli.question is not None:
                    raise
                print(f"Not added to history: {exc}")
            if cli.question is not None:
                break
    finally:
        dataset.close()
        if transcript:
            transcript.close()


if __name__ == "__main__":
    main()
