"""Disposable, fully audited training arrays; never a replacement source dataset.

Preparation streams QA JSONL and preserves its order. Runtime readers mmap only
numeric arrays and fixed-width identifiers, without loading millions of dicts.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import torch

from .multisource_readout_data import _Readers
from .patch_qa_contract import sha256_file
from .point_readout import FORMAT, PROMPT_VERSION, json_hash, normalize_field, tensor_hash, validate_record


CACHE_FORMAT = "readout_training_arrays_v1"
INDEX_DTYPE = np.dtype([
    ("grid_h", "<u4"), ("grid_w", "<u4"), ("field_offset", "<u8"),
    ("prompt_len", "<u4"), ("answer_len", "<u4"), ("token_offset", "<u8"),
    ("qa_id", "S24"), ("state_ref", "S64"), ("task_index", "<u2"),
])
ARRAY_FILES = ("index.npy", "token_lengths.npy", "fields.npy", "tokens.bin")
_CHUNK = 1 << 20


def training_token_identity(tokenizer, *, text_encoding_version, system_message):
    """Bind tokens to actual tokenizer behavior, not only a model path string."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    template = (tokenizer.get_chat_template() if hasattr(tokenizer, "get_chat_template")
                else getattr(tokenizer, "chat_template", None))
    result = {
        "text_encoding_version": text_encoding_version, "system_message": system_message,
        "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "vocab_sha256": json_hash(tokenizer.get_vocab()),
        "chat_template_sha256": json_hash(template),
        "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "special_tokens_map": {str(k): str(v) for k, v in
                               getattr(tokenizer, "special_tokens_map", {}).items()},
        "backend_sha256": json_hash(backend.to_str()) if backend is not None else None,
    }
    for package in ("transformers", "tokenizers"):
        try:
            result[f"{package}_version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[f"{package}_version"] = None
    # Backend serialization includes merges/normalizers. Slow tokenizer assets
    # need an additional content identity when a local tokenizer is available.
    assets = {}
    location = Path(str(getattr(tokenizer, "name_or_path", "")))
    if location.is_dir():
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model",
                     "chat_template.jinja"):
            if (location / name).is_file():
                assets[name] = sha256_file(location / name)
    result["local_tokenizer_assets"] = assets
    return result


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _limits(training):
    result = {k: int(training[k]) for k in ("max_prompt_tokens", "max_target_tokens")}
    if min(result.values()) <= 0:
        raise ValueError("Training token budgets must be positive")
    return result


def _qa_metadata(qa_dir, *, check_files=True):
    qa_dir = Path(qa_dir)
    if (qa_dir / ".build_in_progress").exists():
        raise ValueError("QA dataset build did not finish")
    metadata = _json(qa_dir / "metadata.json")
    if (metadata.get("format") != FORMAT or metadata.get("prompt_version") != PROMPT_VERSION
            or metadata.get("source", {}).get("kind") != "multisource"):
        raise ValueError("Prepared training requires the multi-source QA protocol")
    expected = {"states.jsonl", "train.jsonl", "val.jsonl", "test.jsonl", "qa_fields.hdf5"}
    if (metadata.get("query_program", {}).get("enabled") and
            metadata["generation"].get("query_program", {}).get("diagnostic_states_per_source", 16) > 0):
        expected.update(("diagnostics_val.jsonl", "diagnostics_test.jsonl"))
    if set(metadata["files"]) != expected:
        raise ValueError("Incomplete QA dataset manifest")
    if check_files:
        for name, digest in metadata["files"].items():
            if sha256_file(qa_dir / name) != digest:
                raise ValueError(f"Dataset file changed: {name}")
    return metadata


def _source_stats(metadata):
    source = metadata["source"]
    readers = _Readers(source["sources"], source["files"],
                       int(metadata["config"]["data"].get("max_open_hdf5", 8)))
    try:
        for file_id in source["files"]:
            readers.check_file(file_id)
    finally:
        readers.close()


def _fixed_ascii(value, width, label):
    encoded = value.encode("ascii")
    if len(encoded) != width:
        raise ValueError(f"Unexpected {label} width")
    return encoded


def _token_report(limits):
    return {"questions": 0, "max_prompt_tokens_observed": 0, "max_target_tokens_observed": 0,
            "prompt_limit": limits["max_prompt_tokens"],
            "target_limit_including_eos": limits["max_target_tokens"],
            "prompt_overflow_count": 0, "target_overflow_count": 0,
            "overflow_questions": 0, "overflow_examples": [],
            "max_prompt_record": None, "max_target_record": None, "status": "ok"}


def prepare_training_cache(qa_dir, output_dir, tokenizer, training, *, encode_record,
                           token_identity, progress=None):
    """Audit every training QA/source crop, then atomically publish derived arrays.

    Interrupted output remains marked incomplete. Existing nonempty destinations
    are never overwritten or cleaned up. No source or QA file is modified.
    """
    qa_dir, output_dir = Path(qa_dir).resolve(), Path(output_dir).resolve()
    if qa_dir == output_dir or qa_dir in output_dir.parents or output_dir in qa_dir.parents:
        raise ValueError("Prepared cache must be separate from the source QA directory")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Cache destination is nonempty: {output_dir}")
    if progress:
        progress({"phase": "verify_source_qa", "qa_dir": str(qa_dir)})
    metadata = _qa_metadata(qa_dir)
    _source_stats(metadata)
    limits, report = _limits(training), _token_report(_limits(training))
    count = int(metadata["counts"]["train"])
    if count < 1:
        raise ValueError("Training split is empty")
    partitions = metadata["trajectory_splits"]
    flattened = [x for ids in partitions.values() for x in ids]
    if set(partitions) != {"train", "val", "test"} or len(flattened) != len(set(flattened)):
        raise ValueError("Trajectory leakage or incomplete partitions")
    partition_sets = {split: set(ids) for split, ids in partitions.items()}
    if progress:
        progress({"phase": "load_state_index", "training_questions": count})
    states = {}
    for state in _lines(qa_dir / "states.jsonl"):
        ref = state["state_ref"]
        if ref in states:
            raise ValueError("Duplicate state references")
        if f"{state['source_file_id']}:{state['sample_index']}" not in partition_sets[state["split"]]:
            raise ValueError("State violates trajectory partition")
        states[ref] = state
    tasks = list(metadata["tasks"])
    task_indices = {name: i for i, name in enumerate(tasks)}
    if len(tasks) > np.iinfo(np.uint16).max or len(task_indices) != len(tasks):
        raise ValueError("Invalid task inventory")
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / ".build_in_progress"
    with marker.open("x", encoding="utf-8") as handle:
        handle.write(CACHE_FORMAT + "\n")
    index = lengths = fields = None
    source = metadata["source"]
    readers = _Readers(source["sources"], source["files"],
                       int(metadata["config"]["data"].get("max_open_hdf5", 8)))
    token_offset, rows_written = 0, 0
    audited_states, family_questions = set(), Counter()
    try:
        index = np.lib.format.open_memmap(output_dir / "index.npy", mode="w+", dtype=INDEX_DTYPE, shape=(count,))
        lengths = np.lib.format.open_memmap(output_dir / "token_lengths.npy", mode="w+", dtype="<u4", shape=(count,))
        with h5py.File(qa_dir / "qa_fields.hdf5", "r") as cached:
            values = cached["values"]
            if values.ndim != 1 or values.dtype != np.dtype("float16"):
                raise ValueError("Unexpected exact QA field cache representation")
            if progress:
                progress({"phase": "copy_fields", "copied_values": 0, "total_values": len(values)})
            fields = np.lib.format.open_memmap(output_dir / "fields.npy", mode="w+", dtype="<f2", shape=values.shape)
            for start in range(0, len(values), _CHUNK):
                fields[start:start + _CHUNK] = values[start:start + _CHUNK]
                copied = min(start + _CHUNK, len(values))
                if progress and (copied == len(values) or copied % (64 * _CHUNK) == 0):
                    progress({"phase": "copy_fields", "copied_values": copied, "total_values": len(values)})
            previous, z = None, None
            if progress:
                progress({"phase": "audit_and_tokenize", "questions": 0, "total_questions": count})
            with (output_dir / "tokens.bin").open("xb") as token_out:
                for row in _lines(qa_dir / "train.jsonl"):
                    if rows_written >= count:
                        raise ValueError("Training count differs from the QA manifest")
                    state = states[row["state_ref"]]
                    if state["split"] != "train" or row["grid_shape"] != state["grid_shape"]:
                        raise ValueError("Question references a wrong split/shape")
                    if row["task_type"] not in task_indices:
                        raise ValueError("Question differs from manifest tasks")
                    scoring = metadata.get("task_scoring", {}).get(row["task_type"], metadata["scoring"])
                    if row["oracle"]["absolute_tolerance"] != scoring["absolute_tolerance"]:
                        raise ValueError("Question differs from manifest scoring policy")
                    if metadata.get("query_program", {}).get("enabled"):
                        from .field_query_program import TEMPLATE_SPLITS
                        if (row.get("language_group") != "seen" or not row.get("query_protocol")
                                or row.get("template_family") not in TEMPLATE_SPLITS["train"]
                                or row["oracle"].get("answer_decimals") != scoring["answer_decimals"]):
                            raise ValueError("Question violates the training query protocol")
                    h, w = state["grid_shape"]
                    offset = int(state["cache_offset"])
                    if min(h, w) < 1 or max(h, w) > np.iinfo(np.uint32).max or offset < 0 or offset + h*w > len(fields):
                        raise ValueError("QA field offset/shape is out of bounds")
                    if previous != row["state_ref"]:
                        z, audit = normalize_field(readers.read(state), allow_singleton=True, allow_constant=True)
                        if audit != state["normalization"]:
                            raise ValueError(f"Source crop changed: {row['state_ref']}")
                        copied = torch.from_numpy(np.array(fields[offset:offset + h*w], dtype=np.float32).reshape(h, w))
                        if tensor_hash(copied) != audit["z_sha256"] or not torch.equal(copied, z):
                            raise ValueError("Cached QA differs from original source")
                        previous = row["state_ref"]
                        audited_states.add(previous)
                    validate_record(row, z)
                    prompt, answer = encode_record(row, tokenizer, training)
                    pl, al = len(prompt), len(answer)
                    if not pl or not al or pl > limits["max_prompt_tokens"] or al > limits["max_target_tokens"]:
                        raise ValueError("Encoded question exceeds its immutable token budgets")
                    if answer[-1] != tokenizer.eos_token_id or tokenizer.eos_token_id not in answer:
                        raise ValueError("Encoded answer must terminate with native EOS")
                    for tokens in (prompt, answer):
                        if any(int(token) != token or token < 0 or token > np.iinfo(np.uint32).max for token in tokens):
                            raise ValueError("Token IDs must fit uint32 exactly")
                        token_out.write(np.asarray(tokens, dtype="<u4").tobytes())
                    index[rows_written] = (h, w, offset, pl, al, token_offset,
                        _fixed_ascii(row["qa_id"], 24, "qa_id"), _fixed_ascii(row["state_ref"], 64, "state_ref"),
                        task_indices[row["task_type"]])
                    lengths[rows_written] = pl + al
                    token_offset += pl + al
                    rows_written += 1
                    family_questions[state["source_family"]] += 1
                    report["questions"] += 1
                    detail = {"qa_id": row["qa_id"], "task_type": row["task_type"],
                              "template_family": row.get("template_family"), "prompt_tokens": pl,
                              "target_tokens_including_eos": al}
                    if pl > report["max_prompt_tokens_observed"]:
                        report["max_prompt_tokens_observed"], report["max_prompt_record"] = pl, detail
                    if al > report["max_target_tokens_observed"]:
                        report["max_target_tokens_observed"], report["max_target_record"] = al, detail
                    if progress and rows_written % 10000 == 0:
                        progress({"phase": "audit_and_tokenize", "questions": rows_written,
                                  "total_questions": count, "tokens": token_offset})
                token_out.flush()
                os.fsync(token_out.fileno())
        if rows_written != count:
            raise ValueError("Training count differs from the QA manifest")
        ordered_ids = np.sort(index["qa_id"])
        if bool(np.any(ordered_ids[1:] == ordered_ids[:-1])):
            raise ValueError("Duplicate training question IDs")
        del ordered_ids
        for array in (index, lengths, fields):
            array.flush()
        field_count = len(fields)
    finally:
        readers.close()
        # NumPy views (including ones retained by an exception traceback) own
        # the mapping too. Release our references instead of forcibly unmapping
        # memory that a still-live array could access during error reporting.
        index = lengths = fields = None
        array = None
    # Recheck the immutable inputs after a potentially long streaming build.
    if progress:
        progress({"phase": "verify_and_publish", "questions": rows_written, "tokens": token_offset})
    if _qa_metadata(qa_dir) != metadata:
        raise ValueError("QA metadata changed during cache preparation")
    _source_stats(metadata)
    manifest = {
        "format": CACHE_FORMAT, "status": "complete", "split": "train",
        "qa_dir": str(qa_dir), "metadata": metadata, "identity": json_hash(metadata),
        "qa_metadata_sha256": sha256_file(qa_dir / "metadata.json"),
        "token_identity": token_identity, "training_limits": limits,
        "tasks": tasks, "questions": count, "token_count": token_offset, "field_values": field_count,
        "token_report": report,
        "source_audit": {"questions": count, "states": len(audited_states), "identity": json_hash(metadata),
                         "suite": "main", "family_questions": dict(family_questions),
                         "all_training_questions_replayed": True, "original_source_crops_replayed": True},
        "files": {name: {"sha256": sha256_file(output_dir / name), "size_bytes": (output_dir / name).stat().st_size}
                  for name in ARRAY_FILES},
        "purpose": "Derived exact training crops/tokens; not complete original source data",
    }
    temp = output_dir / "manifest.json.pending"
    with temp.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, output_dir / "manifest.json")
    marker.unlink()
    if progress:
        progress({"status": "complete", "questions": count, "tokens": token_offset})
    return manifest


def _manifest(cache_dir):
    cache_dir = Path(cache_dir)
    if (cache_dir / ".build_in_progress").exists() or not (cache_dir / "manifest.json").is_file():
        raise ValueError("Prepared training cache is incomplete")
    manifest = _json(cache_dir / "manifest.json")
    if (manifest.get("format") != CACHE_FORMAT or manifest.get("status") != "complete"
            or manifest.get("split") != "train" or set(manifest.get("files", {})) != set(ARRAY_FILES)
            or manifest.get("identity") != json_hash(manifest.get("metadata"))):
        raise ValueError("Invalid prepared training cache manifest")
    return manifest


def validate_training_cache(cache_dir, *, qa_dir=None, token_identity=None, training=None):
    """Full once-per-launch leader validation; workers may then use validate=False."""
    cache_dir = Path(cache_dir)
    manifest = _manifest(cache_dir)
    if token_identity is not None and token_identity != manifest["token_identity"]:
        raise ValueError("Prepared tokens belong to a different tokenizer/encoding identity")
    if training is not None and _limits(training) != manifest["training_limits"]:
        raise ValueError("Prepared tokens belong to different training token budgets")
    qa_dir = Path(qa_dir or manifest["qa_dir"])
    if (sha256_file(qa_dir / "metadata.json") != manifest["qa_metadata_sha256"]
            or _qa_metadata(qa_dir) != manifest["metadata"]):
        raise ValueError("Prepared cache belongs to a different QA dataset")
    _source_stats(manifest["metadata"])
    for name, expected in manifest["files"].items():
        path = cache_dir / name
        if path.stat().st_size != expected["size_bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"Prepared cache file changed: {name}")
    dataset = PreparedTrainingDataset(cache_dir, validate=False)
    try:
        previous_end = 0
        for start in range(0, len(dataset), _CHUNK):
            rows = dataset.index[start:start + _CHUNK]
            total = rows["prompt_len"].astype(np.uint64) + rows["answer_len"]
            offsets = rows["token_offset"]
            if (not len(rows) or offsets[0] != previous_end or
                    np.any(offsets[1:] != offsets[:-1] + total[:-1]) or
                    np.any(rows["prompt_len"] == 0) or np.any(rows["answer_len"] == 0) or
                    np.any(rows["prompt_len"] > manifest["training_limits"]["max_prompt_tokens"]) or
                    np.any(rows["answer_len"] > manifest["training_limits"]["max_target_tokens"]) or
                    np.any(total != dataset.token_lengths[start:start + len(rows)]) or
                    np.any(rows["grid_h"] == 0) or np.any(rows["grid_w"] == 0) or
                    np.any(rows["field_offset"] + rows["grid_h"].astype(np.uint64) * rows["grid_w"] > len(dataset.fields)) or
                    np.any(rows["task_index"] >= len(manifest["tasks"]))):
                raise ValueError("Prepared cache index violates bounds/order")
            previous_end = int(offsets[-1] + total[-1])
        if previous_end != len(dataset.tokens):
            raise ValueError("Prepared token count differs from its index")
    finally:
        dataset.close()
    return manifest


class _PreparedRecords(Sequence):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset.index)

    @property
    def shape_columns(self):
        """Read-only height/width columns for batch planning without dict decode."""
        return self.dataset.index["grid_h"], self.dataset.index["grid_w"]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = self.dataset.index[index]
        return {"grid_shape": [int(row["grid_h"]), int(row["grid_w"])],
                "qa_id": bytes(row["qa_id"]).decode("ascii"),
                "state_ref": bytes(row["state_ref"]).decode("ascii"),
                "task_type": self.dataset.manifest["tasks"][int(row["task_index"])],
                "_prepared_index": index}


class PreparedTrainingDataset:
    """Minimal training-only mmap reader; validation/evaluation use original QA."""
    def __init__(self, cache_dir, *, validate=True, qa_dir=None, token_identity=None, training=None):
        self.cache_dir = Path(cache_dir).resolve()
        self.manifest = (validate_training_cache(self.cache_dir, qa_dir=qa_dir,
                            token_identity=token_identity, training=training) if validate else _manifest(self.cache_dir))
        self.metadata, self.identity = self.manifest["metadata"], self.manifest["identity"]
        self.qa_dir = Path(qa_dir or self.manifest["qa_dir"])
        self.suite = "main"
        self.index = self.token_lengths = self.fields = self.tokens = None
        try:
            self.index = np.load(self.cache_dir / "index.npy", mmap_mode="r", allow_pickle=False)
            self.token_lengths = np.load(self.cache_dir / "token_lengths.npy", mmap_mode="r", allow_pickle=False)
            self.fields = np.load(self.cache_dir / "fields.npy", mmap_mode="r", allow_pickle=False)
            self.tokens = np.memmap(self.cache_dir / "tokens.bin", dtype="<u4", mode="r")
            if (self.index.dtype != INDEX_DTYPE or self.index.shape != (self.manifest["questions"],)
                    or self.token_lengths.dtype != np.dtype("<u4") or self.token_lengths.shape != self.index.shape
                    or self.fields.dtype != np.dtype("<f2") or self.fields.shape != (self.manifest["field_values"],)
                    or self.tokens.shape != (self.manifest["token_count"],)):
                raise ValueError("Prepared cache array schema differs from manifest")
            self.records = _PreparedRecords(self)
        except BaseException:
            self.close()
            raise

    def __len__(self):
        return len(self.index)

    def token_pair(self, index):
        row = self.index[index]
        start, pl, al = int(row["token_offset"]), int(row["prompt_len"]), int(row["answer_len"])
        return self.tokens[start:start + pl], self.tokens[start + pl:start + pl + al]

    def field_array(self, index):
        row = self.index[index]
        start, h, w = int(row["field_offset"]), int(row["grid_h"]), int(row["grid_w"])
        return np.array(self.fields[start:start + h*w], dtype=np.float32).reshape(1, h, w)

    def __getitem__(self, index):
        return self.records[index], torch.from_numpy(self.field_array(index))

    def audit(self):
        return self.manifest["source_audit"]

    def close(self):
        # Do not call the private _mmap.close(): callers may still hold lengths,
        # shape columns or token views. Their last reference safely releases the
        # OS mapping. A closed dataset drops its own references immediately.
        for name in ("index", "token_lengths", "fields", "tokens"):
            setattr(self, name, None)

