"""Single/multi-device numerical generation using the existing frozen-Qwen sidecar."""
from __future__ import annotations

import argparse
from array import array
import copy
import gc
import json
import math
import os
import random
import signal
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from scripts import train_tensor_qwen_cross_attention as core
from scripts.readout_distributed import (
    ParallelContext, RecordView, capture_rng, restore_rng, partition_batch,
    evaluation_indices, merge_token_reports,
)
from tensor_compression.downstream.point_readout import (
    FORMAT, PROMPT_VERSION, build_prompt, json_hash, score_answer,
)
from tensor_compression.downstream.point_readout_data import (
    PointReadoutDataset, load_config, resolve_path, source_path, write_json,
)
from tensor_compression.downstream.readout_evaluation import (
    FieldPairDataset, evaluation_annotations, summarize_readout_evaluation, zero_baseline_metrics,
)

CHECKPOINT_TYPE = "qwen_standardized_field_qa_v2"
TEXT_ENCODING_VERSION = "native_chat_v1"
SYSTEM_MESSAGE = ("You answer questions about a supplied numerical field. "
                  "Return only the requested JSON array, with no explanation.")


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
        memory_block_shape=tuple(memory.get("block_shape", (1, 1))),
        bridge_dim=int(bridge["bridge_dim"]), bridge_heads=int(bridge["heads"]),
        cross_attention_layers=list(bridge["layers_1based"]),
        bridge_dropout=float(bridge.get("dropout", 0)), gate_init=float(bridge.get("gate_init", 0)),
        lr=float(training["lr"]), gate_lr=float(training["gate_lr"]),
        weight_decay=float(training.get("weight_decay", 0)),
    )


def chat_messages(record):
    return [{"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": build_prompt(record).removesuffix("\nAnswer:")}]


def native_chat_ids(tokenizer, messages, *, generation_prompt):
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("The numerical QA interface requires the model's native chat template")
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation_prompt,
                                         return_dict=False)


class TokenBudgetError(ValueError):
    """A validated native chat exceeds a length budget, never a tokenizer error."""

    def __init__(self, *, qa_id, prompt_tokens, target_tokens, prompt_limit, target_limit,
                 report=None, message=None):
        self.qa_id = qa_id
        self.prompt_tokens, self.target_tokens = prompt_tokens, target_tokens
        self.prompt_limit, self.target_limit = prompt_limit, target_limit
        self.report = report
        if message is None:
            message = (f"QA {qa_id}: Prompt has {prompt_tokens} tokens (max_prompt_tokens={prompt_limit}); "
                       f"Answer has {target_tokens} tokens including EOS (max_target_tokens={target_limit}); "
                       "truncation is forbidden")
        super().__init__(message)


def encode_prompt(record, tokenizer, max_prompt_tokens):
    ids = native_chat_ids(tokenizer, chat_messages(record), generation_prompt=True)
    if not ids:
        raise ValueError(f"Native chat prompt {record['qa_id']} is empty")
    if len(ids) > max_prompt_tokens:
        raise TokenBudgetError(qa_id=record["qa_id"], prompt_tokens=len(ids), target_tokens=None,
            prompt_limit=max_prompt_tokens, target_limit=None,
            message=(f"Prompt {record['qa_id']} has {len(ids)} tokens "
                     f"(max_prompt_tokens={max_prompt_tokens}); truncation is forbidden"))
    return ids


def encode_training_record(row, tokenizer, training):
    # Check both complete lengths only after validating the native conversation.
    # A prompt overflow must not hide a second target overflow or a broken template.
    prompt = native_chat_ids(tokenizer, chat_messages(row), generation_prompt=True)
    if not prompt:
        raise ValueError(f"Native chat prompt {row['qa_id']} is empty")
    # Tokenize the complete conversation to preserve native assistant boundaries
    # and context-sensitive BPE, rather than joining separately tokenized strings.
    complete = native_chat_ids(tokenizer, chat_messages(row) +
                               [{"role": "assistant", "content": row["answer"]}], generation_prompt=False)
    if complete[:len(prompt)] != prompt:
        raise ValueError("Native chat training sequence does not share the generation prompt prefix")
    answer = complete[len(prompt):]
    if tokenizer.eos_token_id not in answer:
        raise ValueError("Native assistant template must terminate with tokenizer.eos_token_id")
    answer = answer[:answer.index(tokenizer.eos_token_id) + 1]
    prompt_limit, target_limit = int(training["max_prompt_tokens"]), int(training["max_target_tokens"])
    if len(prompt) > prompt_limit or len(answer) > target_limit:
        raise TokenBudgetError(qa_id=row["qa_id"], prompt_tokens=len(prompt), target_tokens=len(answer),
                               prompt_limit=prompt_limit, target_limit=target_limit)
    return prompt, answer


class TrainingTokenCache:
    """Run-local bounded CPU token cache; no tensor memory or autograd state.

    Cache keys include the actual prompt and answer, not just a possibly reused
    QA ID. Compact integer arrays avoid one Python integer object per token.
    max_bytes bounds accounted token/key storage; max_entries also bounds the
    Python dictionary overhead. Zero disables caching without changing results.
    """

    def __init__(self, tokenizer, training, *, max_bytes=256 * 2**20, max_entries=262144):
        if max_bytes < 0 or max_entries < 0:
            raise ValueError("Token cache limits must be nonnegative")
        if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
            raise ValueError("Numerical generation requires an EOS and a padding token")
        self.tokenizer = tokenizer
        self.limits = (int(training["max_prompt_tokens"]), int(training["max_target_tokens"]))
        self.signature = (tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.chat_template)
        self.max_bytes, self.max_entries = int(max_bytes), int(max_entries)
        self.entries, self.bytes = OrderedDict(), 0
        self.hits = self.misses = 0

    def statistics(self):
        return {"hits": self.hits, "misses": self.misses, "entries": len(self.entries),
                "accounted_bytes": self.bytes, "max_bytes": self.max_bytes, "max_entries": self.max_entries}

    def get(self, row, tokenizer, training):
        if (tokenizer is not self.tokenizer or
                self.signature != (tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.chat_template) or
                self.limits != (int(training["max_prompt_tokens"]), int(training["max_target_tokens"]))):
            raise ValueError("Token cache belongs to a different tokenizer or token budget")
        key = json_hash({"messages": chat_messages(row), "answer": row["answer"]})
        cached = self.entries.get(key)
        if cached is not None:
            self.hits += 1
            self.entries.move_to_end(key)
            return cached[0]
        self.misses += 1
        pair = encode_training_record(row, tokenizer, training)
        if not self.max_bytes or not self.max_entries:
            return pair
        packed = tuple(array("I", tokens) for tokens in pair)
        cost = sys.getsizeof(key) + sys.getsizeof(packed) + sum(sys.getsizeof(tokens) for tokens in packed)
        if self.max_entries and cost <= self.max_bytes:
            while self.entries and (self.bytes + cost > self.max_bytes or len(self.entries) >= self.max_entries):
                _, (_, evicted_cost) = self.entries.popitem(last=False)
                self.bytes -= evicted_cost
            self.entries[key] = packed, cost
            self.bytes += cost
        return packed


def preflight_token_budgets(records, tokenizer, training, *, token_cache=None, split="unknown",
                            raise_on_overflow=True):
    """Scan every record and collect budget failures; semantic errors still stop immediately."""
    report = {"questions": 0, "max_prompt_tokens_observed": 0, "max_target_tokens_observed": 0,
              "prompt_limit": int(training["max_prompt_tokens"]),
              "target_limit_including_eos": int(training["max_target_tokens"]),
              "prompt_overflow_count": 0, "target_overflow_count": 0,
              "overflow_questions": 0, "overflow_examples": [],
              "max_prompt_record": None, "max_target_record": None}
    for index, row in enumerate(records):
        try:
            prompt, answer = (token_cache.get(row, tokenizer, training) if token_cache is not None else
                              encode_training_record(row, tokenizer, training))
            prompt_length, target_length = len(prompt), len(answer)
        except TokenBudgetError as exc:
            prompt_length, target_length = exc.prompt_tokens, exc.target_tokens
        # This is bounded diagnostic metadata, not a second cache of all records.
        detail = {"qa_id": row["qa_id"], "task_type": row.get("task_type"),
                  "template_family": row.get("template_family"),
                  "prompt_tokens": prompt_length, "target_tokens_including_eos": target_length}
        report["questions"] += 1
        if prompt_length > report["max_prompt_tokens_observed"]:
            report["max_prompt_tokens_observed"], report["max_prompt_record"] = prompt_length, detail
        if target_length > report["max_target_tokens_observed"]:
            report["max_target_tokens_observed"], report["max_target_record"] = target_length, detail
        prompt_overflow = prompt_length > report["prompt_limit"]
        target_overflow = target_length > report["target_limit_including_eos"]
        report["prompt_overflow_count"] += int(prompt_overflow)
        report["target_overflow_count"] += int(target_overflow)
        if prompt_overflow or target_overflow:
            report["overflow_questions"] += 1
            if len(report["overflow_examples"]) < 8:
                report["overflow_examples"].append(detail)
        if (index + 1) % 100000 == 0:
            print(f"token_preflight split={split} checked={index + 1}/{len(records)} "
                  f"max_prompt={report['max_prompt_tokens_observed']} "
                  f"max_target={report['max_target_tokens_observed']} "
                  f"prompt_overflow={report['prompt_overflow_count']} "
                  f"target_overflow={report['target_overflow_count']}", flush=True)
    report["status"] = "overflow" if report["overflow_questions"] else "ok"
    if raise_on_overflow and report["overflow_questions"]:
        raise TokenBudgetError(qa_id=None, prompt_tokens=report["max_prompt_tokens_observed"],
            target_tokens=report["max_target_tokens_observed"], prompt_limit=report["prompt_limit"],
            target_limit=report["target_limit_including_eos"], report=report,
            message=(f"Token preflight split={split}: {report['overflow_questions']}/{report['questions']} "
                     "questions exceed max_prompt_tokens or max_target_tokens; "
                     f"observed maxima are {report['max_prompt_tokens_observed']} / "
                     f"{report['max_target_tokens_observed']}; truncation is forbidden"))
    return report


def training_tensors(records, tokenizer, training, *, token_cache=None):
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("Numerical generation requires an EOS and a padding token")
    encoded = []
    for row in records:
        prompt, answer = (token_cache.get(row, tokenizer, training) if token_cache is not None else
                          encode_training_record(row, tokenizer, training))
        encoded.append((prompt, answer))
    width = max(len(p) + len(a) for p, a in encoded)
    ids = torch.full((len(records), width), tokenizer.pad_token_id, dtype=torch.long)
    mask, labels = torch.zeros_like(ids), torch.full_like(ids, -100)
    for i, (prompt, answer) in enumerate(encoded):
        ids[i, :len(prompt) + len(answer)] = torch.tensor(list(prompt) + list(answer))
        mask[i, :len(prompt) + len(answer)] = 1
        labels[i, len(prompt):len(prompt) + len(answer)] = torch.tensor(list(answer))
    return ids, mask, labels


def training_loss(llm, sidecar, tokenizer, records, fields, device, dtype, training, *, token_cache=None):
    ids, mask, labels = [x.to(device) for x in training_tensors(records, tokenizer, training, token_cache=token_cache)]
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


def _backward_chunks(llm, sidecar, tokenizer, dataset, local_batches, device, dtype,
                     training, token_cache, micro_batch_size, global_count):
    values, sizes = [0.0, 0.0, 0.0], []
    for indices in local_batches:
        for start in range(0, len(indices), micro_batch_size):
            chunk = indices[start:start + micro_batch_size]
            records, fields = zip(*(dataset[index] for index in chunk))
            try:
                loss, components = training_loss(llm, sidecar, tokenizer, records, torch.stack(fields),
                    device, dtype, training, token_cache=token_cache)
                weight = len(chunk) / global_count
                (loss * weight).backward()
                values[0] += float(loss.detach()) * weight
                values[1] += components["answer_ce"] * weight
                values[2] += components["reconstruction"] * weight
                sizes.append(len(chunk))
                del loss
            finally:
                # Hooks must remain bound throughout checkpoint recomputation.
                sidecar.clear()
    return values, sizes


def backward_update(llm, sidecar, tokenizer, dataset, batches, device, dtype, training,
                    parallel, micro_batch_size, *, token_cache=None, auto_micro_batch=True):
    """One complete global update; retry local OOMs BEFORE any gradient collective.

    Retry all local contributions, not only the failed chunk: backward may have
    partially accumulated gradients. Preserve sample membership and global loss
    normalization. The optimizer/scheduler advance only after success on all ranks.
    """
    global_count = sum(map(len, batches))
    if global_count <= 0:
        raise ValueError("An optimizer update must contain questions")
    local_batches = [partition_batch(batch, parallel.rank, parallel.world_size) for batch in batches]
    initial_rng = capture_rng(device)
    initial_buffers = {name: buffer.detach().clone() for name, buffer in sidecar.named_buffers()}
    attempts, error, values, sizes = 0, None, [0.0] * 3, []
    while True:
        sidecar.zero_grad(set_to_none=True)
        failed_size = None
        try:
            values, sizes = _backward_chunks(llm, sidecar, tokenizer, dataset, local_batches,
                device, dtype, training, token_cache, micro_batch_size, global_count)
            break
        except torch.cuda.OutOfMemoryError as exc:
            attempted = min(micro_batch_size, max(map(len, local_batches), default=0))
            if not auto_micro_batch or attempted <= 1:
                error = f"CUDA OOM even at micro batch {attempted}; update not applied: {exc}"
            else:
                failed_size = attempted
        except Exception as exc:
            if parallel.world_size == 1:
                raise
            error = f"{type(exc).__name__}: {exc}"
        # Outside except: release traceback-held activations before empty_cache.
        sidecar.clear()
        sidecar.zero_grad(set_to_none=True)
        if error:
            break
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        restore_rng(initial_rng, device)
        with torch.no_grad():
            for name, buffer in sidecar.named_buffers():
                buffer.copy_(initial_buffers[name])
        micro_batch_size = max(1, failed_size // 2)
        attempts += 1
        print(f"oom_retry rank={parallel.rank} micro_batch={micro_batch_size} "
              f"global_questions={global_count} retry_entire_local_update=true", flush=True)
    parallel.check_errors(error)
    parallel.sum_gradients(sidecar)
    values = parallel.sums(values)
    execution = parallel.gather({"rank": parallel.rank, "questions": sum(map(len, local_batches)),
        "micro_batch_sizes": sizes, "micro_batch_limit": micro_batch_size, "oom_retries": attempts})
    if sum(item["questions"] for item in execution) != global_count:
        raise RuntimeError("Global question accounting differs across ranks")
    return values, micro_batch_size, execution


def generation_stop_ids(llm, tokenizer):
    ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()
    for source in (getattr(llm, "generation_config", None), getattr(llm, "config", None)):
        value = getattr(source, "eos_token_id", None)
        if value is not None:
            ids.update(int(n) for n in (value if isinstance(value, (list, tuple)) else [value]))
    if not ids:
        raise ValueError("The model/tokenizer must define an EOS token")
    return sorted(ids)


@torch.inference_mode()
def generate_from_prompt(llm, tokenizer, prompt, device, dtype, *, max_new_tokens=96, use_cache=True):
    """Shared full-vocabulary greedy decoder for field-memory and serialized inputs."""
    if not prompt or max_new_tokens <= 0 or tokenizer.eos_token_id is None:
        raise ValueError("Generation requires a positive token budget and an EOS token")
    stop_ids = set(generation_stop_ids(llm, tokenizer))
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    output_ids, past = [], None
    terminated = False
    stop_token_id = None
    with core.autocast_context(device, dtype):
        for _ in range(max_new_tokens):
            inputs = ids[:, -1:] if use_cache and past is not None else ids
            kwargs = {"input_ids": inputs, "attention_mask": torch.ones_like(ids),
                      "use_cache": use_cache, "return_dict": True}
            if use_cache and past is not None:
                kwargs["past_key_values"] = past
            outputs = core.decoder_backbone(llm)(**kwargs)
            logits = llm.get_output_embeddings()(outputs.last_hidden_state[:, -1])
            next_id = int(logits.argmax(dim=-1).item())
            if next_id in stop_ids:
                terminated = True
                stop_token_id = next_id
                break
            output_ids.append(next_id)
            ids = torch.cat((ids, ids.new_tensor([[next_id]])), dim=1)
            if use_cache:
                past = outputs.past_key_values
    return {"prediction": tokenizer.decode(output_ids, skip_special_tokens=False),
            "terminated": terminated, "stop_token_id": stop_token_id,
            "generated_tokens": len(output_ids) + int(terminated)}


@torch.inference_mode()
def generate_answer(llm, sidecar, tokenizer, record, field, device, dtype, *,
                    max_prompt_tokens=384, max_new_tokens=96, use_cache=True):
    """Field binding and text cache live for one answer only."""
    prompt = encode_prompt(record, tokenizer, max_prompt_tokens)
    try:
        with core.autocast_context(device, dtype):
            sidecar.bind(field.unsqueeze(0).to(device), mode="correct")
        return generate_from_prompt(llm, tokenizer, prompt, device, dtype,
                                    max_new_tokens=max_new_tokens, use_cache=use_cache)
    finally:
        sidecar.clear()


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


def batch_statistics(batches, batch_size):
    sizes = [len(batch) for batch in batches]
    return {"micro_batches_per_epoch": len(sizes), "questions_per_epoch": sum(sizes),
            "micro_batch_min": min(sizes), "micro_batch_max": max(sizes),
            "micro_batch_mean": sum(sizes) / len(sizes),
            "partial_micro_batches_per_epoch": sum(n < batch_size for n in sizes)}


def prepare_output_directory(path, *, resume=False, overwrite=False):
    """Reuse a run name only by preserving its previous contents in a sibling backup."""
    if not path or "${" in str(path):
        raise ValueError("Set output path environment variables or pass an explicit path")
    path = Path(path).expanduser()
    if path.is_symlink():
        raise ValueError("Output directory must not be a symlink")
    path = path.resolve()
    if overwrite and resume:
        raise ValueError("Overwrite cannot be combined with resume")
    if path.exists() and not path.is_dir():
        raise ValueError("Output path must be a directory")
    if path.exists() and any(path.iterdir()):
        if overwrite:
            if (path == ROOT or path in ROOT.parents or path == Path.cwd()
                    or path in Path.cwd().parents):
                raise ValueError("Cannot overwrite the project or a working-directory ancestor")
            if not all((path / name).is_file() for name in ("contract.json", "resolved_config.json")):
                raise ValueError("Overwrite requires a recognized training output directory")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = path.with_name(f"{path.name}.backup-{stamp}-{uuid4().hex[:8]}")
            path.rename(backup)
            print(f"previous_output_backup={backup}", flush=True)
        elif not resume:
            raise FileExistsError("Use a new empty output directory, --resume, or --overwrite-output-dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


@torch.inference_mode()
def evaluate(llm, sidecar, tokenizer, dataset, device, dtype, config, output_path, *,
             verbose=False, input_mode="correct", parallel=None):
    if parallel is not None and parallel.world_size > 1:
        return evaluate_parallel(llm, sidecar, tokenizer, dataset, device, dtype, config,
                                 output_path, parallel, verbose=verbose, input_mode=input_mode)
    if input_mode not in ("correct", "disabled"):
        raise ValueError("Evaluation input_mode must be correct or disabled")
    core.set_frozen_llm_execution_mode(llm, checkpoint_training=False)
    sidecar.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start, scored = time.perf_counter(), []
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    active_state = None
    field_encodes = 0
    reuse_memory = bool(config["evaluation"].get("reuse_field_memory", True))
    # Only one field is retained, and only during this evaluation invocation.
    # Text KV is always initialized afresh for every question. Dataset order is
    # unchanged, including when one state appears in disjoint runs of records.
    sidecar.clear()
    try:
        if input_mode == "disabled":
            # no_tensor disables each bridge entirely; zero-valued field inputs
            # would still pass through learned encoders and are a different test.
            sidecar.bind(None, mode="no_tensor")
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for index in range(len(dataset)):
                if input_mode == "disabled":
                    row, field = dataset.records[index], None
                else:
                    row, field = dataset[index]
                prompt = encode_prompt(row, tokenizer, int(config["training"]["max_prompt_tokens"]))
                if input_mode == "correct" and (not reuse_memory or active_state != row["state_ref"]):
                    sidecar.clear()
                    with core.autocast_context(device, dtype):
                        sidecar.bind(field.unsqueeze(0).to(device), mode="correct")
                    active_state = row["state_ref"]
                    field_encodes += 1
                prediction = generate_from_prompt(
                    llm, tokenizer, prompt, device, dtype,
                    max_new_tokens=int(config["evaluation"]["max_new_tokens"]),
                    use_cache=bool(config["evaluation"].get("use_cache", True)))
                score = score_answer(row, prediction["prediction"], terminated=prediction["terminated"])
                state = dataset.states[row["state_ref"]]
                entry = {"qa_id": row["qa_id"], "task_type": row["task_type"],
                         "shape": "x".join(map(str, row["grid_shape"])),
                         "field_tokens": math.prod((n + p - 1) // p for n, p in
                                                   zip(row["grid_shape"], sidecar.memory.block_shape)),
                         "block_shape": list(sidecar.memory.block_shape),
                         "input_mode": input_mode,
                         "field": state["field"], **prediction, "score": score,
                         **evaluation_annotations(row, sidecar.memory.block_shape)}
                entry.update({key: state[key] for key in ("source_id", "family", "is_static", "is_constant", "roughness_bin") if key in state})
                scored.append(entry)
                handle.write(json.dumps(entry, allow_nan=False) + "\n")
                if verbose and (index + 1) % 100 == 0:
                    print(f"evaluation questions={index + 1}/{len(dataset)}", flush=True)
    finally:
        sidecar.clear()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metrics = summarize_readout_evaluation(scored, dataset.metadata["train_shapes"])
    metrics["zero_value_baseline"] = zero_baseline_metrics(dataset.records, dataset.states)
    suite = getattr(dataset, "suite", "main")
    metrics.update(input_mode=input_mode, diagnostic_only=input_mode != "correct" or suite != "main",
                   evaluation_suite=suite)
    if hasattr(dataset, "selection"):
        metrics["field_pair_selection"] = dataset.selection
    metrics.update(elapsed_seconds=time.perf_counter() - start,
                   peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
                   generated_tokens=sum(row["generated_tokens"] for row in scored),
                   field_memory_encodes=field_encodes,
                   field_memory_reuses=len(scored) - field_encodes if input_mode == "correct" else 0)
    return metrics


class EvaluationView:
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices
        self.records = RecordView(dataset.records, indices)
        self.states, self.metadata = dataset.states, dataset.metadata
        self.suite = getattr(dataset, "suite", "main")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def merge_evaluation_shards(dataset, paths, output_path):
    predictions = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["qa_id"] in predictions:
                    raise ValueError("Duplicate QA in distributed evaluation")
                predictions[row["qa_id"]] = row
    expected = [row["qa_id"] for row in dataset.records]
    if len(expected) != len(set(expected)) or set(expected) != predictions.keys():
        raise ValueError("Distributed evaluation has missing or unexpected QA IDs")
    ordered = [predictions[key] for key in expected]
    temporary = Path(str(output_path) + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in ordered:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(output_path)
    return ordered


@torch.inference_mode(False)
def evaluate_parallel(llm, sidecar, tokenizer, dataset, device, dtype, config,
                      output_path, parallel, *, verbose=False, input_mode="correct"):
    # Local generation re-enters inference_mode, but distributed object
    # collectives must allocate ordinary tensors: backend threads mutate their
    # receive buffers outside Python's thread-local inference_mode context.
    start = time.perf_counter()
    output_path = Path(output_path)
    paths = [output_path.with_name(f"{output_path.stem}.rank{rank}.jsonl")
             for rank in range(parallel.world_size)]
    indices = evaluation_indices(dataset.records, parallel.world_size)[parallel.rank]
    error, local_metrics = None, None
    try:
        if indices:
            local_metrics = evaluate(llm, sidecar, tokenizer, EvaluationView(dataset, indices),
                device, dtype, config, paths[parallel.rank], verbose=verbose, input_mode=input_mode)
        else:
            paths[parallel.rank].write_text("", encoding="utf-8")
            local_metrics = {"field_memory_encodes": 0, "field_memory_reuses": 0,
                             "peak_allocated_gib": None, "questions": 0}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    parallel.check_errors(error)
    summaries = parallel.gather({key: local_metrics[key] for key in (
        "field_memory_encodes", "field_memory_reuses", "peak_allocated_gib", "questions")})
    def merge():
        scored = merge_evaluation_shards(dataset, paths, output_path)
        metrics = summarize_readout_evaluation(scored, dataset.metadata["train_shapes"])
        metrics["zero_value_baseline"] = zero_baseline_metrics(dataset.records, dataset.states)
        suite = getattr(dataset, "suite", "main")
        peaks = [m["peak_allocated_gib"] for m in summaries if m["peak_allocated_gib"] is not None]
        metrics.update(input_mode=input_mode, diagnostic_only=input_mode != "correct" or suite != "main",
            evaluation_suite=suite, elapsed_seconds=time.perf_counter() - start,
            generated_tokens=sum(row["generated_tokens"] for row in scored),
            field_memory_encodes=sum(m["field_memory_encodes"] for m in summaries),
            field_memory_reuses=sum(m["field_memory_reuses"] for m in summaries),
            peak_allocated_gib=max(peaks) if peaks else None,
            distributed_world_size=parallel.world_size, per_rank=summaries)
        if hasattr(dataset, "selection"):
            metrics["field_pair_selection"] = dataset.selection
        return metrics
    return parallel.main(merge)


def print_metric_summary(name, metrics):
    seen, heldout = metrics["seen_shapes"], metrics["heldout_shapes"]
    seen_text = f"{seen['macro_task_answer_accuracy']:.4f}" if seen else "n/a"
    heldout_text = f"{heldout['macro_task_answer_accuracy']:.4f}" if heldout else "n/a"
    print(f"{name} questions={metrics['questions']} all={metrics['macro_task_answer_accuracy']:.4f} "
          f"seen={seen_text} heldout={heldout_text} valid={metrics['valid_rate']:.4f} "
          f"truncated={metrics['errors'].get('generation_truncated', 0)}", flush=True)


def validate_config(config):
    from tensor_compression.downstream.block_tokens import block_shape
    block_shape(config["memory"].get("block_shape", (1, 1)))
    t, e = config["training"], config["evaluation"]
    micro = config["runtime"].get("max_micro_batch_size", t["batch_size"])
    if type(micro) is not int or micro <= 0:
        raise ValueError("runtime.max_micro_batch_size must be a positive integer")
    if type(config["runtime"].get("auto_micro_batch", True)) is not bool:
        raise ValueError("runtime.auto_micro_batch must be boolean")
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
        raise ValueError("Use bfloat16 on a supported GPU or float32 for CPU tests; FP16 needs a separate scaler")


def run_identity(config, dataset, llm, tokenizer):
    # Machine-local paths may change on relocation; architecture and recipe may not.
    recipe = {key: copy.deepcopy(config[key]) for key in (
        "model", "field_encoder", "spatial_adapter", "memory", "cross_attention", "training", "evaluation", "runtime")}
    recipe["model"].pop("name_or_path", None)
    # Execution chunking can be reduced to fit memory without changing the
    # global batch/optimizer recipe. Floating-point rounding need not be bitwise identical.
    for key in ("max_micro_batch_size", "auto_micro_batch"):
        recipe["runtime"].pop(key, None)
    identity = {"protocol": FORMAT, "prompt_version": PROMPT_VERSION, "dataset": dataset.identity,
            "text_encoding": {"version": TEXT_ENCODING_VERSION, "system_message": SYSTEM_MESSAGE,
                              "chat_template_sha256": json_hash(tokenizer.get_chat_template()),
                              "stop_token_ids": generation_stop_ids(llm, tokenizer)},
            "patch_size": config["data"]["patch_size"], "recipe": recipe,
            "llm_config": {k: v for k, v in llm.config.to_dict().items()
                           if k not in ("_name_or_path", "transformers_version", "torch_dtype", "dtype")},
            "tokenizer_sha256": json_hash(tokenizer.get_vocab()),
            "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id}
    # Transformers configs may contain integer-keyed dictionaries (e.g. id2label).
    # Use the same canonical JSON representation in disk contracts and checkpoints.
    return json.loads(json.dumps(identity, allow_nan=False))


def model_asset_identity(model_name_or_path, llm, *, verbose=False):
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
            if verbose:
                print(f"model_asset_audit shard={name}", flush=True)
            manifest[name] = core.sha256_file(weight_path)
        return {"local_weights_sha256": json_hash(manifest)}
    return {"hub_model": str(model_name_or_path), "revision": getattr(llm.config, "_commit_hash", None)}


def validate_resume(checkpoint, identity):
    if checkpoint.get("checkpoint_type") != CHECKPOINT_TYPE:
        raise ValueError("Use a point-readout checkpoint, not a previous multiple-choice checkpoint")
    if checkpoint.get("identity", {}).get("text_encoding") != identity.get("text_encoding"):
        raise ValueError("Checkpoint chat template/stop protocol differs; reuse v2 QA data but start a new run")
    if checkpoint.get("identity") != identity:
        raise ValueError("Checkpoint dataset/model/tokenizer/recipe differs; resume cannot change an experiment")


def planned_updates(records, training, seed):
    if training.get("max_updates") is not None:
        return training["max_updates"]
    count = len(shape_batches(records, training["batch_size"], seed, 0))
    return math.ceil(count * training["epochs"] / training["gradient_accumulation_steps"])


def geometry_probe_indices(rows):
    """One deterministic example for each of the existing four geometry extrema."""
    if not rows:
        raise ValueError("Geometry checks require nonempty training records")
    criteria = [lambda r: math.prod(r["grid_shape"]),
                lambda r: (-r["grid_shape"][0], r["grid_shape"][1]),
                lambda r: (-r["grid_shape"][1], r["grid_shape"][0]),
                lambda r: -math.prod(r["grid_shape"])]
    return list(dict.fromkeys(max(range(len(rows)), key=lambda i: key(rows[i])) for key in criteria))


def check_geometry(llm, sidecar, tokenizer, dataset, device, dtype, config, *, parallel=None):
    """Stress each boundary at the requested batch size, without optimizer updates."""
    indices = geometry_probe_indices(dataset.records)
    report = []
    parallel = parallel or ParallelContext(device)
    batch_size = config["training"]["batch_size"]
    micro_limit = config["runtime"].get("max_micro_batch_size", batch_size)
    sidecar.train()
    for index in indices:
        row, field = dataset[index]
        sidecar.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        try:
            values, micro_limit, execution = backward_update(llm, sidecar, tokenizer, dataset,
                [[index] * batch_size], device, dtype, config["training"], parallel, micro_limit,
                auto_micro_batch=config["runtime"].get("auto_micro_batch", True))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            entry = {"shape": row["grid_shape"], "batch_size": batch_size,
                     "repeated_example_for_memory_check": True, "answer_ce": values[1],
                     "reconstruction": values[2], "per_rank_execution": execution,
                     "seconds": time.perf_counter() - started,
                     "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30
                     if device.type == "cuda" else None}
            report.append(entry)
            print("geometry_check=" + json.dumps(entry), flush=True)
        finally:
            sidecar.clear()
            sidecar.zero_grad(set_to_none=True)
    return {"status": "geometry_forward_backward_only", "optimizer_updates": 0, "checks": report,
            "limitation": "Does not measure optimizer-state memory or full validation/generation cost"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/field_to_llm_point_readout.yaml"))
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--qa-dir")
    parser.add_argument("--hdf5-path")
    parser.add_argument("--model-dir")
    parser.add_argument("--block-shape", type=int, nargs=2, metavar=("H", "W"),
                        help="Cells per field token; e.g. 1 1 or 2 2. Requires a matching checkpoint when resuming.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite-output-dir", action="store_true",
                        help="Archive an existing training output to a sibling backup, then start fresh; stop its process first")
    for option, help_text in (
        ("batch-size", "GLOBAL question batch upper bound across all ranks, grouped by grid shape"),
        ("gradient-accumulation-steps", "Global shape batches accumulated per optimizer update"),
        ("epochs", "Training epochs when config max_updates is null"),
        ("save-every-updates", "Checkpoint cadence in optimizer updates"),
        ("eval-every-updates", "Validation cadence in optimizer updates"),
    ):
        parser.add_argument("--" + option, type=int, help=help_text + "; overrides the selected profile")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--micro-batch-size", type=int,
                        help="Maximum per-device forward/backward batch; splitting preserves the global batch")
    parser.add_argument("--no-auto-micro-batch", action="store_true",
                        help="Fail on CUDA OOM instead of retrying the complete local update with smaller chunks")
    parser.add_argument("--distributed-timeout-minutes", type=int, default=360,
                        help="Collective timeout, including complete data audit/preflight (default: 360)")
    parser.add_argument("--resume", help="Strict resume from this numerical-generation experiment")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--eval-suite", choices=("main", "language", "field-pairs"), default="main",
                        help="Diagnostic pairs are explicit evaluations, never periodic training validation")
    parser.add_argument("--diagnostic-pairs", type=int, default=128,
                        help="Maximum accepted matched-query field pairs (default: 128)")
    parser.add_argument("--diagnostic-max-candidates", type=int, default=4096,
                        help="Maximum alternative-field candidates read for field pairs (default: 4096)")
    parser.add_argument("--input-mode", choices=("correct", "disabled"), default="correct",
                        help="disabled is an evaluation-only diagnostic that disables field cross-attention")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--audit-only", action="store_true", help="Replay data without loading Qwen")
    parser.add_argument("--token-check-only", action="store_true",
                        help="Audit and tokenize every main train/val/test row, then exit before loading Qwen weights")
    parser.add_argument("--geometry-check-only", action="store_true",
                        help="Load real Qwen and check boundary-shape forward/backward; no training updates")
    parser.add_argument("--stop-after-updates", type=int, help="Save and stop after N additional updates; preserves the full resume schedule")
    parser.add_argument("--verbose", action="store_true", help="Show detailed training, model-shard and evaluation progress")
    parser.add_argument("--console-every-updates", type=int, default=200,
                        help="Console cadence only; detailed train.jsonl stays unchanged (default: 200)")
    cli = parser.parse_args()
    if cli.overwrite_output_dir and (cli.resume or cli.evaluate_only or cli.audit_only or cli.geometry_check_only or cli.token_check_only):
        parser.error("--overwrite-output-dir is only for fresh training")
    if cli.token_check_only and (cli.resume or cli.evaluate_only or cli.audit_only or cli.geometry_check_only
                                or cli.eval_suite != "main" or cli.input_mode != "correct" or cli.stop_after_updates):
        parser.error("--token-check-only checks all main splits and cannot combine with another action or diagnostic suite")
    if cli.geometry_check_only and (cli.audit_only or cli.evaluate_only or cli.resume):
        parser.error("--geometry-check-only requires a fresh run and cannot be combined with audit/evaluation/resume")
    if cli.evaluate_only and not cli.resume:
        parser.error("--evaluate-only requires --resume")
    if (cli.eval_suite != "main" or cli.input_mode != "correct") and not cli.evaluate_only:
        parser.error("Diagnostic --eval-suite and --input-mode disabled require --evaluate-only")
    if cli.diagnostic_pairs <= 0 or cli.diagnostic_max_candidates <= 0:
        parser.error("Diagnostic pair and candidate limits must be positive")
    if cli.eval_suite != "field-pairs" and (cli.diagnostic_pairs != 128 or cli.diagnostic_max_candidates != 4096):
        parser.error("Diagnostic pair limits apply only to --eval-suite field-pairs")
    if cli.stop_after_updates is not None and cli.stop_after_updates <= 0:
        parser.error("--stop-after-updates must be positive")
    if cli.console_every_updates <= 0:
        parser.error("--console-every-updates must be positive")
    if cli.micro_batch_size is not None and cli.micro_batch_size <= 0:
        parser.error("--micro-batch-size must be positive")
    if cli.split == "test" and not (cli.evaluate_only or cli.audit_only or cli.token_check_only):
        parser.error("The test split is accessible only through explicit evaluation/audit")
    config = load_config(cli.config, cli.profile)
    for key in ("batch_size", "gradient_accumulation_steps", "epochs", "save_every_updates", "eval_every_updates"):
        value = getattr(cli, key)
        if value is not None:
            if value <= 0:
                parser.error(f"--{key.replace('_', '-')} must be positive")
            config["training"][key] = value
    if cli.epochs is not None and config["training"].get("max_updates") is not None:
        parser.error("--epochs requires a profile with max_updates: null")
    if cli.block_shape is not None:
        config["memory"]["block_shape"] = cli.block_shape
    if cli.micro_batch_size is not None:
        config["runtime"]["max_micro_batch_size"] = cli.micro_batch_size
    if cli.no_auto_micro_batch:
        config["runtime"]["auto_micro_batch"] = False
    validate_config(config)
    qa_dir = resolve_path(cli.qa_dir or config["data"]["qa_dir"])
    hdf5_path = source_path(config, cli.hdf5_path)
    parallel = ParallelContext.initialize(cli.device, cli.distributed_timeout_minutes)
    device = parallel.device
    # Directory mutation and all shared artifacts have exactly one writer.
    try:
        output = Path(parallel.main(lambda: str(prepare_output_directory(os.path.expandvars(cli.output_dir),
            resume=bool(cli.resume and not cli.evaluate_only), overwrite=cli.overwrite_output_dir))))
    except BaseException:
        parallel.close()
        raise
    def write_shared(path, value):
        return parallel.main(lambda: write_json(path, value))
    def say(message):
        if parallel.primary:
            print(message, flush=True)
    datasets = {}
    geometry_original_records = {}
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested; saving at the next completed optimizer update.", flush=True)
    previous_handlers = {}
    try:
        splits = [cli.split] if cli.evaluate_only else ["train", "val"]
        if cli.geometry_check_only:
            splits = ["train"]
        if cli.audit_only or cli.token_check_only:
            splits = ["train", "val", "test"]
        say(f"parallel_setup world_size={parallel.world_size} device_type={device.type} "
            f"global_batch_limit={config['training']['batch_size']} "
            f"micro_batch_limit={config['runtime'].get('max_micro_batch_size', config['training']['batch_size'])}")
        for split in splits:
            say(f"dataset_load split={split} validating_manifest_and_loading_records=true")
            suite_options = {"suite": cli.eval_suite} if cli.eval_suite == "language" else {}
            dataset = PointReadoutDataset(qa_dir, hdf5_path, split,
                cache_size=int(config["data"].get("input_cache_size", 8)), **suite_options)
            datasets[split] = dataset
            if cli.geometry_check_only:
                original_records = dataset.records
                probe_indices = geometry_probe_indices(original_records)
                geometry_original_records[split] = original_records
                dataset.records = [original_records[index] for index in probe_indices]
                write_shared(output / "geometry_preflight_scope.json", {
                    "scope": "geometry_boundary_probes_only", "full_dataset_audited": False,
                    "optimizer_updates": 0, "dataset_identity": dataset.identity,
                    "loaded_splits": ["train"], "total_train_questions": len(original_records),
                    "checked_questions": len(dataset.records), "selected_indices": probe_indices,
                    "qa_ids": [row["qa_id"] for row in dataset.records],
                    "grid_shapes": [row["grid_shape"] for row in dataset.records],
                    "selection_criteria": ["maximum_area", "minimum_height_then_maximum_width",
                                           "minimum_width_then_maximum_height", "minimum_area"],
                    "limitation": "Only selected geometry probes are audited/tokenized; use token-check-only for all main splits"})
            audit_name = "geometry_data_audit" if cli.geometry_check_only else "data_audit"
            say(f"{audit_name}_start split={split} questions={len(dataset)} audit_rank=0")
            audit = parallel.main(dataset.audit)
            say(f"{audit_name} split={split} {audit}")
            if cli.eval_suite == "field-pairs":
                paired = FieldPairDataset(dataset, max_pairs=cli.diagnostic_pairs,
                    max_candidates=cli.diagnostic_max_candidates, seed=int(config["runtime"]["seed"]))
                datasets[split] = paired
                write_shared(output / "field_pair_selection.json", paired.selection)
                def write_pairs():
                    with (output / "field_pair_queries.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
                        for row in paired.records:
                            handle.write(json.dumps(row, allow_nan=False) + "\n")
                parallel.main(write_pairs)
                say("field_pair_selection=" + json.dumps(paired.selection))
                if not paired.records:
                    write_shared(output / f"{split}_metrics.json", {"status": "unavailable",
                        "diagnostic_only": True, "evaluation_suite": "field-pairs",
                        "field_pair_selection": paired.selection})
                    return
        if cli.audit_only:
            write_shared(output / "data_audit.json", {s: {"questions": len(d), "identity": d.identity}
                                                    for s, d in datasets.items()})
            return
        dataset = next(iter(datasets.values()))
        expected_generation = config["generation"]
        for key in ("absolute_tolerance", "answer_decimals"):
            if expected_generation[key] != dataset.metadata["scoring"][key]:
                raise ValueError(f"Configured {key} differs from the frozen dataset scoring protocol")
        # The immutable dataset defines selected tasks, including builder --tasks
        # overrides. No task IDs or point counts are supplied to the neural model.
        if device.type == "cuda" and config["model"]["torch_dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("Selected CUDA device does not support BF16")
        args = model_args(config, cli.model_dir, checkpointing=not cli.evaluate_only)
        core.apply_runtime_environment(args)
        core.seed_everything(args.seed)
        tokenizer = core.load_tokenizer(args)
        token_cache = TrainingTokenCache(tokenizer, config["training"],
            max_bytes=int(config["runtime"].get("token_cache_bytes", 256 * 2**20)),
            max_entries=int(config["runtime"].get("token_cache_entries", 262144)))
        token_reports = {}
        for split, d in datasets.items():
            say(f"token_preflight split={split} questions={len(d.records)} world_size={parallel.world_size}")
            rows = RecordView(d.records, range(parallel.rank, len(d.records), parallel.world_size))
            local_report, error = None, None
            try:
                local_report = preflight_token_budgets(rows, tokenizer, config["training"],
                    token_cache=token_cache, split=f"{split}/rank{parallel.rank}", raise_on_overflow=False)
            except Exception as exc:
                if parallel.world_size == 1:
                    raise
                error = f"{type(exc).__name__}: {exc}"
            parallel.check_errors(error)
            token_reports[split] = merge_token_reports(parallel.gather(local_report))
        token_report_path = output / "token_preflight.json"
        write_shared(token_report_path, token_reports)
        overflow_splits = [split for split, report in token_reports.items() if report["overflow_questions"]]
        if overflow_splits:
            raise ValueError(f"Token budgets exceeded in {', '.join(overflow_splits)}; all requested splits "
                f"were scanned. See {token_report_path.resolve()}. Review training.max_prompt_tokens, "
                "training.max_target_tokens and evaluation.max_new_tokens against the observed maxima; "
                "no records were truncated and no language-model weights were loaded.")
        if cli.token_check_only:
            say("token_check_complete=" + json.dumps(token_reports))
            return
        llm, dtype = core.load_llm_with_bounded_host_memory(args, device)
        core.seed_everything(args.seed)
        encoder, spatial, _, _ = core.build_scratch_memory_components(args, (1, args.patch_size, args.patch_size))
        sidecar, report = core.build_sidecar(llm, spatial, args, device, encoder)
        identity = run_identity(config, dataset, llm, tokenizer)
        identity["model_asset"] = parallel.main(lambda: model_asset_identity(args.model_name_or_path, llm, verbose=cli.verbose))
        if len(set(parallel.gather(json_hash(identity)))) != 1:
            raise ValueError("Ranks loaded different model/tokenizer/dataset/config identities")
        existing_contract = output / "contract.json"
        if existing_contract.exists() and json.loads(existing_contract.read_text(encoding="utf-8")) != identity:
            raise ValueError("Output directory belongs to another experiment")
        checkpoint = torch.load(cli.resume, map_location="cpu", weights_only=True) if cli.resume else None
        if checkpoint:
            validate_resume(checkpoint, identity)
            sidecar.load_state_dict(checkpoint["sidecar"], strict=True)
        parallel.synchronize_parameters(sidecar)
        write_shared(output / "contract.json", identity)
        write_shared(output / "resolved_config.json", config)
        write_shared(output / "architecture.json", report)
        if cli.geometry_check_only:
            geometry = check_geometry(llm, sidecar, tokenizer, datasets["train"], device, dtype, config, parallel=parallel)
            reports = parallel.gather(geometry)
            write_shared(output / "geometry_check.json", geometry if parallel.world_size == 1 else
                         {"world_size": parallel.world_size, "per_rank": reports})
            return
        if cli.evaluate_only:
            metrics = evaluate(llm, sidecar, tokenizer, datasets[cli.split], device, dtype, config,
                               output / f"{cli.split}_predictions.jsonl", verbose=cli.verbose,
                               input_mode=cli.input_mode, parallel=parallel)
            write_shared(output / f"{cli.split}_metrics.json", metrics)
            write_shared(output / "evaluation_contract.json", {"checkpoint": str(Path(cli.resume).resolve()),
                "split": cli.split, "suite": cli.eval_suite, "input_mode": cli.input_mode,
                "diagnostic_only": cli.eval_suite != "main" or cli.input_mode != "correct"})
            if parallel.primary:
                print_metric_summary(cli.split, metrics)
            return
        optimizer, _ = core.build_optimizer(sidecar, args)
        training = config["training"]
        maximum = planned_updates(datasets["train"].records, training, args.seed)
        scheduler, _ = core.build_sidecar_scheduler(optimizer, "cosine", maximum,
                                                   float(training.get("warmup_ratio", .03)),
                                                   float(training.get("min_lr_ratio", .2)))
        step, epoch, cursor, best = 0, 0, 0, -1.0
        micro_batch_size = int(config["runtime"].get("max_micro_batch_size", training["batch_size"]))
        validation_pending = False
        if checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            step, epoch, cursor, best = (checkpoint[k] for k in ("step", "epoch", "cursor", "best"))
            validation_pending = checkpoint.get("validation_pending", False)
            execution = checkpoint.get("parallel_execution", {"world_size": 1})
            if execution["world_size"] != parallel.world_size:
                raise ValueError("Strict training resume requires the same GPU/process count; evaluation may use another count")
            rank_states = checkpoint.get("rank_states", [checkpoint])
            restore_rng(rank_states[parallel.rank], device)
            micro_batch_size = min(micro_batch_size, rank_states[parallel.rank].get("micro_batch_limit", micro_batch_size))
        if step > maximum:
            raise ValueError("Checkpoint step exceeds configured training budget")
        train_data = datasets["train"]
        batches = shape_batches(train_data.records, training["batch_size"], args.seed, epoch)
        batch_stats = batch_statistics(batches, training["batch_size"])
        invocation_start_step = step
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, request_stop)

        def save(path):
            states = parallel.gather({**capture_rng(device), "micro_batch_limit": micro_batch_size})
            parallel.main(lambda: core.atomic_torch_save(path, {
                "checkpoint_type": CHECKPOINT_TYPE, "identity": identity,
                "sidecar": {k: v.detach().cpu().clone() for k, v in sidecar.state_dict().items()},
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "step": step, "epoch": epoch, "cursor": cursor, "best": best,
                "validation_pending": validation_pending, **states[0],
                "rank_states": states, "parallel_execution": {"world_size": parallel.world_size,
                    "global_batch_size": training["batch_size"], "gradient_reduction": "global_question_mean"}}))

        def validate_and_save():
            nonlocal best, validation_pending
            metrics = evaluate(llm, sidecar, tokenizer, datasets["val"], device, dtype, config,
                               output / f"val_predictions_{step}.jsonl", verbose=cli.verbose, parallel=parallel)
            parallel.main(lambda: core.append_jsonl(output / "validation.jsonl", {"step": step, **metrics}))
            score = metrics["seen_shapes"]["macro_task_answer_accuracy"]
            validation_pending = False
            if score > best:
                best = score
                save(output / "best.pt")
            heldout = metrics["heldout_shapes"]
            heldout_text = f"{heldout['macro_task_answer_accuracy']:.4f}" if heldout else "n/a"
            say(f"validation step={step} macro_task_answer_accuracy={score:.4f} "
                  f"heldout={heldout_text} "
                  f"valid_rate={metrics['valid_rate']:.4f} "
                  f"truncated={metrics['errors'].get('generation_truncated', 0)}/{metrics['questions']}")
            save(output / "last.pt")

        say(f"startup=point_readout tasks={dataset.metadata['tasks']} updates={maximum} "
              f"block_shape={args.memory_block_shape} "
              f"global_batch={training['batch_size']} world_size={parallel.world_size} "
              f"per_device_micro_batch_limit={micro_batch_size} accumulation={training['gradient_accumulation_steps']} "
              f"text_encoding={TEXT_ENCODING_VERSION} stop_token_ids={generation_stop_ids(llm, tokenizer)}")
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        say("batch_statistics=" + json.dumps(batch_stats))
        write_shared(output / "training_plan.json", {"planned_updates": maximum, "train_questions": len(train_data),
            "tasks": dataset.metadata["tasks"], "batch_size": training["batch_size"],
            "gradient_accumulation_steps": training["gradient_accumulation_steps"],
            "effective_batch_upper_bound": training["batch_size"] * training["gradient_accumulation_steps"],
            "world_size": parallel.world_size, "per_device_micro_batch_limit": micro_batch_size,
            "auto_micro_batch": config["runtime"].get("auto_micro_batch", True),
            **batch_stats,
            "selection_metric": "seen_shapes.macro_task_answer_accuracy"})
        # A stop or decoder failure immediately before validation must not skip it,
        # including a checkpoint already at the final optimizer update.
        if validation_pending:
            validate_and_save()
        while step < maximum:
            stop_requested = parallel.any(stop_requested)
            if stop_requested:
                break
            micro_batches = []
            for _ in range(training["gradient_accumulation_steps"]):
                if cursor >= len(batches):
                    if training.get("max_updates") is None and epoch + 1 >= training["epochs"]:
                        break  # A final partial update must not consume another epoch.
                    epoch += 1
                    cursor = 0
                    batches = shape_batches(train_data.records, training["batch_size"], args.seed, epoch)
                micro_batches.append(batches[cursor])
                cursor += 1
            total_records = sum(map(len, micro_batches))
            optimizer.zero_grad(set_to_none=True)
            core.set_frozen_llm_execution_mode(llm, checkpoint_training=True)
            sidecar.train()
            values, micro_batch_size, execution = backward_update(
                llm, sidecar, tokenizer, train_data, micro_batches, device, dtype, training,
                parallel, micro_batch_size, token_cache=token_cache,
                auto_micro_batch=config["runtime"].get("auto_micro_batch", True))
            loss_value, answer_ce, reconstruction = values
            grad_norm = torch.nn.utils.clip_grad_norm_(sidecar.parameters(), training["grad_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            log_to_file = step % training["log_interval"] == 0 or step == 1
            log_to_console = log_to_file if cli.verbose else step == 1 or step % cli.console_every_updates == 0 or step == maximum
            if log_to_file or log_to_console:
                memory_by_rank = parallel.gather({"rank": parallel.rank,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None})
                entry = {"step": step, "loss": loss_value, "grad_norm": float(grad_norm),
                         "answer_ce": answer_ce,
                         "value_reconstruction": reconstruction,
                         "value_reconstruction_weight": float(training.get("value_reconstruction_weight", .01)),
                         "gate_values": sidecar.gate_values(),
                         "questions_in_update": total_records,
                         "micro_batch_sizes": [len(indices) for indices in micro_batches],
                         "per_rank_execution": execution,
                         "per_rank_memory": memory_by_rank,
                         "peak_allocated_gib": max(m["peak_allocated_gib"] for m in memory_by_rank)
                         if device.type == "cuda" else None,
                         "peak_reserved_gib": max(m["peak_reserved_gib"] for m in memory_by_rank)
                         if device.type == "cuda" else None,
                         "elapsed_seconds": time.perf_counter() - start}
                if log_to_file and parallel.primary:
                    core.append_jsonl(output / "train.jsonl", entry)
                if log_to_console and parallel.primary:
                    print(json.dumps(entry), flush=True)
            stop_requested = parallel.any(stop_requested)
            should_stop = stop_requested or (cli.stop_after_updates is not None and step - invocation_start_step >= cli.stop_after_updates)
            validation_pending = step % training["eval_every_updates"] == 0 or step == maximum
            # Save before potentially long validation, so termination loses no completed update.
            if step % training["save_every_updates"] == 0 or step % training["eval_every_updates"] == 0 or step == maximum or should_stop:
                save(output / "last.pt")
            if should_stop and step < maximum:
                break
            if validation_pending:
                validate_and_save()
            stop_requested = parallel.any(stop_requested)
            if stop_requested:
                break
        # Also covers a signal arriving between the last periodic-save check and
        # the loop condition, and resuming a finished checkpoint into a new directory.
        save(output / "last.pt")
        stop_requested = parallel.any(stop_requested)
        write_shared(output / "run_summary.json", {"status": "completed" if step == maximum else "stopped",
            "step": step, "planned_updates": maximum, "best_score": best if best >= 0 else None,
            "elapsed_this_invocation_seconds": time.perf_counter() - start, "last_checkpoint": str(output / "last.pt"),
            "world_size": parallel.world_size,
            "token_cache_this_invocation": parallel.gather(token_cache.statistics())})
        if stop_requested:
            raise SystemExit(130)  # Let a shell loop using set -e stop after Ctrl+C.
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        for split, d in datasets.items():
            if split in geometry_original_records:
                d.records = geometry_original_records[split]
            d.close()
        parallel.close()


if __name__ == "__main__":
    main()
