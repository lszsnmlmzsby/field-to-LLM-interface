"""Single-device open numerical generation using the existing frozen-Qwen sidecar."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from scripts import train_tensor_qwen_cross_attention as core
from tensor_compression.downstream.point_readout import (
    FORMAT, PROMPT_VERSION, build_prompt, json_hash, score_answer, summarize_scores,
)
from tensor_compression.downstream.point_readout_data import (
    PointReadoutDataset, load_config, resolve_path, write_json,
)

CHECKPOINT_TYPE = "qwen_standardized_field_qa_v2"


def model_args(config, model_dir=None, *, checkpointing=True):
    """Small explicit adapter to existing architecture/loader helpers, not its MCQ CLI."""
    model, data = config["model"], config["data"]
    memory, bridge = config["memory"], config["cross_attention"]
    training = config["training"]
    model_path = model_dir or os.environ.get("FIELD_TO_LLM_MODEL_DIR") or model["name_or_path"]
    hf_home = os.environ.get("FIELD_TO_LLM_HF_HOME")
    return argparse.Namespace(
        model_name_or_path=model_path, cache_dir=hf_home, hf_home=hf_home,
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        torch_dtype=model.get("torch_dtype", "bfloat16"),
        llm_gradient_checkpointing=checkpointing and bool(model.get("gradient_checkpointing", True)),
        low_cpu_mem_usage=True, min_host_memory_available_gib=0, console_progress=True,
        shape_mode="variable", patch_size=int(data["patch_size"]),
        seed=int(config["runtime"]["seed"]),
        field_encoder_config=copy.deepcopy(config["field_encoder"]),
        spatial_adapter_config=copy.deepcopy(config["spatial_adapter"]),
        freeze_spatial_backbone=False, latent_channel_policy="all",
        value_fourier_bands=int(memory["value_fourier_bands"]),
        value_hidden_dim=int(memory["value_hidden_dim"]),
        bridge_dim=int(bridge["bridge_dim"]), bridge_heads=int(bridge["heads"]),
        cross_attention_layers=list(bridge["layers_1based"]),
        bridge_dropout=float(bridge.get("dropout", 0)), gate_init=float(bridge.get("gate_init", 0)),
        lr=float(training["lr"]), gate_lr=float(training["gate_lr"]),
        weight_decay=float(training.get("weight_decay", 0)),
    )


def encode_prompt(record, tokenizer, max_prompt_tokens):
    ids = tokenizer(build_prompt(record), add_special_tokens=True, truncation=False)["input_ids"]
    if not ids or len(ids) > max_prompt_tokens:
        raise ValueError(f"Prompt {record['qa_id']} has {len(ids)} tokens; truncation is forbidden")
    return ids


def training_tensors(records, tokenizer, training):
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("Numerical generation requires an EOS and a padding token")
    encoded = []
    for row in records:
        prompt = encode_prompt(row, tokenizer, int(training["max_prompt_tokens"]))
        answer = tokenizer(row["answer"], add_special_tokens=False, truncation=False)["input_ids"]
        answer = answer + [int(tokenizer.eos_token_id)]
        if len(answer) > int(training["max_target_tokens"]):
            raise ValueError(f"Answer {row['qa_id']} exceeds max_target_tokens; truncation is forbidden")
        encoded.append((prompt, answer))
    width = max(len(p) + len(a) for p, a in encoded)
    ids = torch.full((len(records), width), tokenizer.pad_token_id, dtype=torch.long)
    mask, labels = torch.zeros_like(ids), torch.full_like(ids, -100)
    for i, (prompt, answer) in enumerate(encoded):
        ids[i, :len(prompt) + len(answer)] = torch.tensor(prompt + answer)
        mask[i, :len(prompt) + len(answer)] = 1
        labels[i, len(prompt):len(prompt) + len(answer)] = torch.tensor(answer)
    return ids, mask, labels


def training_loss(llm, sidecar, tokenizer, records, fields, device, dtype, training):
    ids, mask, labels = [x.to(device) for x in training_tensors(records, tokenizer, training)]
    try:
        with core.autocast_context(device, dtype):
            memory = sidecar.bind(fields.to(device), mode="correct")
            hidden = core.decoder_backbone(llm)(input_ids=ids, attention_mask=mask,
                                                use_cache=False, return_dict=True).last_hidden_state
            # Equal question weights even when answer lengths differ. Project only
            # answer positions to the full vocabulary; never materialize prompt logits.
            answer_ce = torch.stack([core.full_answer_ce(hidden[i:i+1], labels[i:i+1],
                                                        llm.get_output_embeddings())
                                     for i in range(len(records))]).mean()
            reconstruction = memory.reconstruction_loss
            loss = answer_ce + float(training.get("value_reconstruction_weight", 0.01)) * reconstruction
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite generation training loss")
        return loss, {"answer_ce": float(answer_ce.detach()), "reconstruction": float(reconstruction.detach())}
    except BaseException:
        sidecar.clear()
        raise


@torch.inference_mode()
def generate_answer(llm, sidecar, tokenizer, record, field, device, dtype, *,
                    max_prompt_tokens=384, max_new_tokens=96, use_cache=True):
    """One example at a time; field binding and text cache live for one answer only."""
    if max_new_tokens <= 0 or tokenizer.eos_token_id is None:
        raise ValueError("Generation requires a positive token budget and an EOS token")
    prompt = encode_prompt(record, tokenizer, max_prompt_tokens)
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    output_ids, past = [], None
    terminated = False
    try:
        with core.autocast_context(device, dtype):
            sidecar.bind(field.unsqueeze(0).to(device), mode="correct")
            for _ in range(max_new_tokens):
                inputs = ids[:, -1:] if use_cache and past is not None else ids
                kwargs = {"input_ids": inputs, "attention_mask": torch.ones_like(ids),
                          "use_cache": use_cache, "return_dict": True}
                if use_cache and past is not None:
                    kwargs["past_key_values"] = past
                outputs = core.decoder_backbone(llm)(**kwargs)
                logits = llm.get_output_embeddings()(outputs.last_hidden_state[:, -1])
                next_id = int(logits.argmax(dim=-1).item())
                if next_id == tokenizer.eos_token_id:
                    terminated = True
                    break
                output_ids.append(next_id)
                ids = torch.cat((ids, ids.new_tensor([[next_id]])), dim=1)
                if use_cache:
                    past = outputs.past_key_values
    finally:
        sidecar.clear()
    return {"prediction": tokenizer.decode(output_ids, skip_special_tokens=False),
            "terminated": terminated, "generated_tokens": len(output_ids) + int(terminated)}


def shape_batches(records, batch_size, seed, epoch):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rng, buckets = random.Random(seed + epoch), defaultdict(list)
    for i, row in enumerate(records):
        buckets[tuple(row["grid_shape"])].append(i)
    batches = []
    for shape in sorted(buckets):
        indices = buckets[shape]
        rng.shuffle(indices)
        batches.extend(indices[start:start + batch_size] for start in range(0, len(indices), batch_size))
    rng.shuffle(batches)
    return batches


def evaluate(llm, sidecar, tokenizer, dataset, device, dtype, config, output_path):
    core.set_frozen_llm_execution_mode(llm, checkpoint_training=False)
    sidecar.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start, scored = time.perf_counter(), []
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(len(dataset)):
            row, field = dataset[index]
            prediction = generate_answer(
                llm, sidecar, tokenizer, row, field, device, dtype,
                max_prompt_tokens=int(config["training"]["max_prompt_tokens"]),
                max_new_tokens=int(config["evaluation"]["max_new_tokens"]),
                use_cache=bool(config["evaluation"].get("use_cache", True)))
            score = score_answer(row, prediction["prediction"], terminated=prediction["terminated"])
            entry = {"qa_id": row["qa_id"], "task_type": row["task_type"],
                     "shape": "x".join(map(str, row["grid_shape"])),
                     "field": dataset.states[row["state_ref"]]["field"], **prediction, "score": score}
            scored.append(entry)
            handle.write(json.dumps(entry, allow_nan=False) + "\n")
            if (index + 1) % 100 == 0:
                print(f"evaluation questions={index + 1}/{len(dataset)}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metrics = summarize_scores(scored)
    seen_shapes = {"x".join(map(str, shape)) for shape in dataset.metadata["train_shapes"]}
    seen_rows = [row for row in scored if row["shape"] in seen_shapes]
    metrics["seen_shapes"] = summarize_scores(seen_rows) if seen_rows else None
    heldout_rows = [row for row in scored if row["shape"] not in seen_shapes]
    metrics["heldout_shapes"] = summarize_scores(heldout_rows) if heldout_rows else None
    metrics.update(elapsed_seconds=time.perf_counter() - start,
                   peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
                   generated_tokens=sum(row["generated_tokens"] for row in scored))
    return metrics


def validate_config(config):
    t, e = config["training"], config["evaluation"]
    for key in ("batch_size", "gradient_accumulation_steps", "epochs", "max_prompt_tokens",
                "max_target_tokens", "log_interval", "save_every_updates", "eval_every_updates"):
        if type(t[key]) is not int or t[key] <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
    if t.get("max_updates") is not None and (type(t["max_updates"]) is not int or t["max_updates"] <= 0):
        raise ValueError("max_updates must be null (derive from epochs) or a positive integer")
    if e["max_new_tokens"] < t["max_target_tokens"]:
        raise ValueError("Generation budget must cover training answers including EOS")
    for key in ("lr", "gate_lr", "grad_clip_norm"):
        if not math.isfinite(t[key]) or t[key] <= 0:
            raise ValueError(f"Invalid training.{key}")
    if t.get("value_reconstruction_weight", 0) < 0:
        raise ValueError("Reconstruction loss weight must be nonnegative")
    layers = config["cross_attention"]["layers_1based"]
    if not layers or min(layers) < 1 or len(set(layers)) != len(layers):
        raise ValueError("Cross-attention layers must be positive and unique")
    bridge = config["cross_attention"]
    if bridge["heads"] <= 0 or bridge["bridge_dim"] % bridge["heads"]:
        raise ValueError("Bridge dimension must be divisible by positive head count")
    if config["model"]["torch_dtype"] not in ("bfloat16", "float32"):
        raise ValueError("Use bfloat16 on A6000 or float32 for CPU tests; FP16 needs a separate scaler")


def run_identity(config, dataset, llm, tokenizer):
    # Machine-local paths may change on relocation; architecture and recipe may not.
    recipe = {key: copy.deepcopy(config[key]) for key in (
        "model", "field_encoder", "spatial_adapter", "memory", "cross_attention", "training", "evaluation", "runtime")}
    recipe["model"].pop("name_or_path", None)
    identity = {"protocol": FORMAT, "prompt_version": PROMPT_VERSION, "dataset": dataset.identity,
            "patch_size": config["data"]["patch_size"], "recipe": recipe,
            "llm_config": {k: v for k, v in llm.config.to_dict().items()
                           if k not in ("_name_or_path", "transformers_version", "torch_dtype", "dtype")},
            "tokenizer_sha256": json_hash(tokenizer.get_vocab()),
            "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id}
    # Transformers configs may contain integer-keyed dictionaries (e.g. id2label).
    # Use the same canonical JSON representation in disk contracts and checkpoints.
    return json.loads(json.dumps(identity, allow_nan=False))


def model_asset_identity(model_name_or_path, llm):
    """Bind local weight contents, or the resolved immutable Hub commit."""
    path = Path(model_name_or_path).expanduser()
    if path.is_dir():
        indices = list(path.glob("*.safetensors.index.json")) or list(path.glob("pytorch_model.bin.index.json"))
        if indices:
            if len(indices) != 1:
                raise ValueError("Ambiguous model weight index")
            names = sorted(set(json.loads(indices[0].read_text(encoding="utf-8"))["weight_map"].values()))
        else:
            names = [p.name for p in sorted(path.glob("*.safetensors"))] or ["pytorch_model.bin"]
        manifest = {}
        for name in names:
            relative = Path(name)
            weight_path = path / relative
            # Hub snapshots commonly symlink their shards to a shared blob store.
            # Allow those read-only links, but reject index paths escaping the model directory.
            if relative.is_absolute() or ".." in relative.parts or not weight_path.is_file():
                raise ValueError("Invalid or missing model weight shard")
            print(f"model_asset_audit shard={name}", flush=True)
            manifest[name] = core.sha256_file(weight_path)
        return {"local_weights_sha256": json_hash(manifest)}
    return {"hub_model": str(model_name_or_path), "revision": getattr(llm.config, "_commit_hash", None)}


def validate_resume(checkpoint, identity):
    if checkpoint.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise ValueError("Use a point-readout checkpoint, not a previous multiple-choice checkpoint")
    if checkpoint.get("identity") != identity:
        raise ValueError("Checkpoint dataset/model/tokenizer/recipe differs; resume cannot change an experiment")


def planned_updates(records, training, seed):
    if training.get("max_updates") is not None:
        return training["max_updates"]
    count = len(shape_batches(records, training["batch_size"], seed, 0))
    return math.ceil(count * training["epochs"] / training["gradient_accumulation_steps"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/field_to_llm_point_readout.yaml"))
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--qa-dir")
    parser.add_argument("--hdf5-path")
    parser.add_argument("--model-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", help="Strict resume from this numerical-generation experiment")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--audit-only", action="store_true", help="Replay data without loading Qwen")
    parser.add_argument("--stop-after-updates", type=int, help="Save and stop after N additional updates; preserves the full resume schedule")
    cli = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This workstation entry point is single-device; use python, not multi-rank torchrun")
    if cli.evaluate_only and not cli.resume:
        parser.error("--evaluate-only requires --resume")
    if cli.stop_after_updates is not None and cli.stop_after_updates <= 0:
        parser.error("--stop-after-updates must be positive")
    if cli.split == "test" and not (cli.evaluate_only or cli.audit_only):
        parser.error("The test split is accessible only through explicit evaluation/audit")
    config = load_config(cli.config, cli.profile)
    validate_config(config)
    qa_dir = resolve_path(cli.qa_dir or config["data"]["qa_dir"])
    hdf5_path = resolve_path(cli.hdf5_path or os.environ.get("PDEBENCH_HDF5") or config["data"]["hdf5_path"])
    output = resolve_path(cli.output_dir)
    if output.exists() and any(output.iterdir()) and not (cli.resume and not cli.evaluate_only):
        raise FileExistsError("Use a new empty output directory (except when resuming training)")
    output.mkdir(parents=True, exist_ok=True)
    datasets = {}
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested; saving at the next completed optimizer update.", flush=True)
    previous_handlers = {}
    try:
        splits = [cli.split] if cli.evaluate_only else ["train", "val"]
        if cli.audit_only:
            splits = ["train", "val", "test"]
        for split in splits:
            dataset = PointReadoutDataset(qa_dir, hdf5_path, split, cache_size=int(config["data"].get("input_cache_size", 8)))
            datasets[split] = dataset
            print(f"data_audit split={split} {dataset.audit()}", flush=True)
        if cli.audit_only:
            write_json(output / "data_audit.json", {s: {"questions": len(d), "identity": d.identity}
                                                    for s, d in datasets.items()})
            return
        dataset = next(iter(datasets.values()))
        expected_generation = config["generation"]
        for key in ("absolute_tolerance", "answer_decimals"):
            if expected_generation[key] != dataset.metadata["scoring"][key]:
                raise ValueError(f"Configured {key} differs from the frozen dataset scoring protocol")
        # The immutable dataset defines selected tasks, including builder --tasks
        # overrides. No task IDs or point counts are supplied to the neural model.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if cli.device == "auto" else torch.device(cli.device)
        if device.type == "cuda" and config["model"]["torch_dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("Selected CUDA device does not support BF16")
        args = model_args(config, cli.model_dir, checkpointing=not cli.evaluate_only)
        core.apply_runtime_environment(args)
        core.seed_everything(args.seed)
        tokenizer = core.load_tokenizer(args)
        for d in datasets.values():
            for row in d.records:
                training_tensors([row], tokenizer, config["training"])
        llm, dtype = core.load_llm_with_bounded_host_memory(args, device)
        core.seed_everything(args.seed)
        encoder, spatial, _, _ = core.build_scratch_memory_components(args, (1, args.patch_size, args.patch_size))
        sidecar, report = core.build_sidecar(llm, spatial, args, device, encoder)
        identity = run_identity(config, dataset, llm, tokenizer)
        identity["model_asset"] = model_asset_identity(args.model_name_or_path, llm)
        existing_contract = output / "contract.json"
        if existing_contract.exists() and json.loads(existing_contract.read_text(encoding="utf-8")) != identity:
            raise ValueError("Output directory belongs to another experiment")
        checkpoint = torch.load(cli.resume, map_location="cpu", weights_only=True) if cli.resume else None
        if checkpoint:
            validate_resume(checkpoint, identity)
            sidecar.load_state_dict(checkpoint["sidecar"], strict=True)
        write_json(output / "contract.json", identity)
        write_json(output / "resolved_config.json", config)
        write_json(output / "architecture.json", report)
        if cli.evaluate_only:
            metrics = evaluate(llm, sidecar, tokenizer, datasets[cli.split], device, dtype, config,
                               output / f"{cli.split}_predictions.jsonl")
            write_json(output / f"{cli.split}_metrics.json", metrics)
            print(json.dumps(metrics), flush=True)
            return
        optimizer, _ = core.build_optimizer(sidecar, args)
        training = config["training"]
        maximum = planned_updates(datasets["train"].records, training, args.seed)
        scheduler, _ = core.build_sidecar_scheduler(optimizer, "cosine", maximum,
                                                   float(training.get("warmup_ratio", .03)),
                                                   float(training.get("min_lr_ratio", .2)))
        step, epoch, cursor, best = 0, 0, 0, -1.0
        validation_pending = False
        if checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            step, epoch, cursor, best = (checkpoint[k] for k in ("step", "epoch", "cursor", "best"))
            validation_pending = checkpoint.get("validation_pending", False)
            random.setstate(checkpoint["python_rng"])
            torch.set_rng_state(checkpoint["torch_rng"])
            if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
                torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
        if step > maximum:
            raise ValueError("Checkpoint step exceeds configured training budget")
        train_data = datasets["train"]
        batches = shape_batches(train_data.records, training["batch_size"], args.seed, epoch)
        invocation_start_step = step
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, request_stop)

        def save(path):
            core.atomic_torch_save(path, {"checkpoint_type": CHECKPOINT_TYPE, "identity": identity,
                "sidecar": {k: v.detach().cpu().clone() for k, v in sidecar.state_dict().items()},
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "step": step, "epoch": epoch, "cursor": cursor, "best": best,
                "validation_pending": validation_pending,
                "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None})

        def validate_and_save():
            nonlocal best, validation_pending
            metrics = evaluate(llm, sidecar, tokenizer, datasets["val"], device, dtype, config,
                               output / f"val_predictions_{step}.jsonl")
            core.append_jsonl(output / "validation.jsonl", {"step": step, **metrics})
            score = metrics["seen_shapes"]["macro_task_answer_accuracy"]
            validation_pending = False
            if score > best:
                best = score
                save(output / "best.pt")
            print(f"validation step={step} macro_task_answer_accuracy={score:.4f}", flush=True)
            save(output / "last.pt")

        print(f"startup=point_readout tasks={dataset.metadata['tasks']} updates={maximum} "
              f"batch={training['batch_size']} accumulation={training['gradient_accumulation_steps']}", flush=True)
        start = time.perf_counter()
        write_json(output / "training_plan.json", {"planned_updates": maximum, "train_questions": len(train_data),
            "tasks": dataset.metadata["tasks"], "batch_size": training["batch_size"],
            "gradient_accumulation_steps": training["gradient_accumulation_steps"],
            "selection_metric": "seen_shapes.macro_task_answer_accuracy"})
        # A stop or decoder failure immediately before validation must not skip it,
        # including a checkpoint already at the final optimizer update.
        if validation_pending:
            validate_and_save()
        while step < maximum and not stop_requested:
            micro_batches = []
            for _ in range(training["gradient_accumulation_steps"]):
                if cursor >= len(batches):
                    epoch += 1
                    cursor = 0
                    batches = shape_batches(train_data.records, training["batch_size"], args.seed, epoch)
                micro_batches.append(batches[cursor])
                cursor += 1
            total_records = sum(map(len, micro_batches))
            optimizer.zero_grad(set_to_none=True)
            core.set_frozen_llm_execution_mode(llm, checkpoint_training=True)
            sidecar.train()
            loss_value = 0.0
            for indices in micro_batches:
                examples = [train_data[i] for i in indices]
                records, fields = zip(*examples)
                try:
                    loss, _ = training_loss(llm, sidecar, tokenizer, records, torch.stack(fields), device, dtype, training)
                    weight = len(indices) / total_records
                    (loss * weight).backward()
                    loss_value += float(loss.detach()) * weight
                finally:
                    # Non-reentrant checkpointing can revisit bridges during backward.
                    sidecar.clear()
            grad_norm = torch.nn.utils.clip_grad_norm_(sidecar.parameters(), training["grad_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % training["log_interval"] == 0 or step == 1:
                entry = {"step": step, "loss": loss_value, "grad_norm": float(grad_norm),
                         "elapsed_seconds": time.perf_counter() - start}
                core.append_jsonl(output / "train.jsonl", entry)
                print(json.dumps(entry), flush=True)
            should_stop = stop_requested or (cli.stop_after_updates is not None and step - invocation_start_step >= cli.stop_after_updates)
            validation_pending = step % training["eval_every_updates"] == 0 or step == maximum
            # Save before potentially long validation, so termination loses no completed update.
            if step % training["save_every_updates"] == 0 or step % training["eval_every_updates"] == 0 or step == maximum or should_stop:
                save(output / "last.pt")
            if should_stop and step < maximum:
                break
            if validation_pending:
                validate_and_save()
            if stop_requested:
                break
        # Also covers a signal arriving between the last periodic-save check and
        # the loop condition, and resuming a finished checkpoint into a new directory.
        save(output / "last.pt")
        write_json(output / "run_summary.json", {"status": "completed" if step == maximum else "stopped",
            "step": step, "planned_updates": maximum, "best_score": best if best >= 0 else None,
            "elapsed_this_invocation_seconds": time.perf_counter() - start, "last_checkpoint": str(output / "last.pt")})
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        for d in datasets.values():
            d.close()


if __name__ == "__main__":
    main()
