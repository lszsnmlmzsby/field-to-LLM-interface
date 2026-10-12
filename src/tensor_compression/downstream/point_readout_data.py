"""Real HDF5 trajectory-disjoint datasets for standardized numerical generation."""
from __future__ import annotations

import copy
import json
import os
import random
from collections import Counter, OrderedDict
from pathlib import Path

import h5py
import torch

from tensor_compression.downstream.patch_qa_contract import sha256_file
from tensor_compression.downstream.point_readout import (
    FORMAT, PROMPT_VERSION, TASKS, STAT_TASKS, LOCATION_TASKS, json_hash, make_record, normalize_field,
    sample_spec, validate_record,
)
from tensor_compression.downstream.variable_shape import experiment_config, parse_shapes
from tensor_compression.utils.pipeline_config import load_yaml_mapping


def load_config(path, profile):
    return experiment_config(load_yaml_mapping(path), profile)


def resolve_path(value) -> Path:
    expanded = os.path.expandvars(str(value))
    if not value or "${" in expanded:
        raise ValueError(f"Set the environment variable or pass an explicit path: {value}")
    return Path(expanded).expanduser().resolve()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def build_dataset(config, hdf5_path, output_dir):
    config = copy.deepcopy(config)
    data, generation = config["data"], config["generation"]
    tasks = list(generation["tasks"])
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(TASKS):
        raise ValueError(f"Select unique tasks from {TASKS}")
    fields = list(data["fields"])
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("Select at least one unique HDF5 field")
    train_shapes = parse_shapes(data["train_shapes"], allow_odd=True)
    heldout = parse_shapes(data.get("heldout_shapes", []), allow_odd=True) if data.get("heldout_shapes") else []
    if set(train_shapes) & set(heldout):
        raise ValueError("Held-out shapes must be disjoint from training shapes")
    multi_counts = list(generation.get("multi_counts", [2, 3, 4]))
    line_counts = list(generation.get("line_counts", [3, 4, 5]))
    region_shapes = generation.get("region_shapes", [[2, 2]])
    statistic_shapes = generation.get("statistic_region_shapes", [[2, 2], [4, 4]])
    for name, counts in (("multi_counts", multi_counts), ("line_counts", line_counts)):
        if not counts or any(type(n) is not int or n < 2 for n in counts):
            raise ValueError(f"{name} must contain integers >= 2")
    if not region_shapes or any(len(size) != 2 or any(type(n) is not int or n < 2 for n in size)
                               for size in region_shapes):
        raise ValueError("region_shapes must contain integer dimensions >= 2")
    if not statistic_shapes or any(len(size) != 2 or any(type(n) is not int or n < 2 for n in size)
                                  for size in statistic_shapes):
        raise ValueError("statistic_region_shapes must contain integer dimensions >= 2")
    for h, w in train_shapes + heldout:
        if "multi_point" in tasks and max(multi_counts) > h * w:
            raise ValueError("A multi-point query exceeds the number of grid cells")
        if "line_profile" in tasks and max(line_counts) > min(h, w):
            raise ValueError("Every configured line length must fit both axes")
        if "region_values" in tasks and any(rh > h or rw > w for rh, rw in region_shapes):
            raise ValueError("Every configured region must fit every shape")
        if set(tasks) & set(STAT_TASKS + LOCATION_TASKS) and any(rh > h or rw > w for rh, rw in statistic_shapes):
            raise ValueError("Every statistic/search region must fit every shape")
    train_count = int(generation["train_states"])
    eval_count = int(generation["eval_states_per_shape"])
    if train_count < len(train_shapes) * len(fields) or eval_count < len(fields):
        raise ValueError("State counts must cover every shape and field")
    variants = list(generation.get("template_variants", [0, 1]))
    if not variants or any(v not in (0, 1, 2) for v in variants):
        raise ValueError("Invalid question template variants")
    decimals = generation.get("answer_decimals", 1)
    tolerance = float(generation.get("absolute_tolerance", 0.2))
    max_attempts = int(generation.get("max_attempts_per_state", 100))
    if max_attempts <= 0:
        raise ValueError("max_attempts_per_state must be positive")
    hdf5_path, output_dir = Path(hdf5_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a new empty dataset directory: {output_dir}")
    rng = random.Random(int(generation["seed"]))
    rejected = Counter()
    with h5py.File(hdf5_path, "r") as handle:
        layouts = {field: list(handle[field].shape) for field in fields}
        if any(len(shape) != 4 for shape in layouts.values()) or len({tuple(s) for s in layouts.values()}) != 1:
            raise ValueError("Fields must share [trajectory,time,height,width] axes in this HDF5 file")
        n, times, source_h, source_w = layouts[fields[0]]
        if min(times, source_h, source_w) < 1:
            raise ValueError("Source HDF5 contains an empty axis")
        if any(h > source_h or w > source_w for h, w in train_shapes + heldout):
            raise ValueError("A requested crop exceeds the source field")
        samples = list(range(n))
        random.Random(int(generation["split_seed"])).shuffle(samples)
        partitions = {"train": samples[:int(n * .8)], "val": samples[int(n * .8):int(n * .9)],
                      "test": samples[int(n * .9):]}
        if n < 20 or min(map(len, partitions.values())) < 2:
            raise ValueError("Need at least 20 source trajectories for 80/10/10 splits with >=2 each")
        output_dir.mkdir(parents=True, exist_ok=True)
        marker = output_dir / ".build_in_progress"
        marker.write_text(FORMAT, encoding="utf-8")
        states, records = [], {}
        seen_sources = set()
        for split, ids in partitions.items():
            if split == "train":
                pairs = [(shape, field) for shape in train_shapes for field in fields]
                plan = [pairs[i % len(pairs)] for i in range(train_count)]
            else:
                plan = [(shape, fields[i % len(fields)]) for shape in train_shapes + heldout
                        for i in range(eval_count)]
            rows = []
            for shape, field in plan:
                h, w = shape
                for _ in range(max_attempts):
                    source = {"field": field, "sample_index": rng.choice(ids),
                              "time_index": rng.randrange(times),
                              "top_left": [rng.randrange(source_h - h + 1), rng.randrange(source_w - w + 1)],
                              "grid_shape": [h, w]}
                    key = json_hash(source)
                    if key in seen_sources:
                        rejected["duplicate_source_crop"] += 1
                        continue
                    r, c = source["top_left"]
                    raw = torch.from_numpy(handle[field][source["sample_index"], source["time_index"],
                                                         r:r + h, c:c + w].copy())
                    try:
                        z, audit = normalize_field(raw)
                    except ValueError as exc:
                        rejected[str(exc)] += 1
                        continue
                    break
                else:
                    raise ValueError(f"Cannot sample a unique valid {split} crop for {field} {shape}; "
                                     "reduce state counts or add source data")
                seen_sources.add(key)
                state = {"state_ref": key, "split": split, **source, "normalization": audit}
                states.append(state)
                for task in tasks:
                    spec = sample_spec(task, shape, rng, multi_counts=multi_counts,
                                       line_counts=line_counts, region_shapes=region_shapes,
                                       statistic_region_shapes=statistic_shapes, z=z)
                    record = make_record(key, z, task, spec, variant=rng.choice(variants),
                                         decimals=decimals, tolerance=tolerance)
                    validate_record(record, z)
                    rows.append(record)
            records[split] = rows
            print(f"built split={split} states={len(plan)} questions={len(rows)}", flush=True)
    write_jsonl(output_dir / "states.jsonl", states)
    for split, rows in records.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    files = ["states.jsonl", "train.jsonl", "val.jsonl", "test.jsonl"]
    metadata = {"format": FORMAT, "prompt_version": PROMPT_VERSION,
                "source": {"path_hint": str(hdf5_path), "field_layouts": layouts},
                "trajectory_splits": partitions, "train_shapes": train_shapes, "heldout_shapes": heldout,
                "tasks": tasks, "generation": generation, "config": config,
                "normalization": "float32_population_zscore_eps1e-6_then_fp16",
                "scoring": {"absolute_tolerance": tolerance, "answer_decimals": decimals,
                            "invalid_answer_counts_as_wrong": True},
                "rejected_attempts": dict(rejected),
                "counts": {split: len(rows) for split, rows in records.items()},
                "task_counts": {split: dict(Counter(r["task_type"] for r in rows)) for split, rows in records.items()},
                "files": {name: sha256_file(output_dir / name) for name in files}}
    write_json(output_dir / "metadata.json", metadata)
    marker.unlink()
    return metadata


class PointReadoutDataset:
    """Lazy raw-patch replay; metadata/oracles never become neural routing inputs."""

    def __init__(self, qa_dir, hdf5_path, split, cache_size=8):
        if split not in ("train", "val", "test") or cache_size < 0:
            raise ValueError("Invalid split/cache size")
        self.qa_dir, self.hdf5_path = Path(qa_dir), Path(hdf5_path)
        self.cache_size, self.cache = cache_size, OrderedDict()
        self.handle = None
        if (self.qa_dir / ".build_in_progress").exists():
            raise ValueError("Dataset build did not finish")
        self.metadata = json.loads((self.qa_dir / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata["format"] != FORMAT or self.metadata["prompt_version"] != PROMPT_VERSION:
            raise ValueError("Unsupported field-QA protocol: rebuild v2 data; v1 prompts/checkpoints are incompatible")
        expected_files = {"states.jsonl", "train.jsonl", "val.jsonl", "test.jsonl"}
        if set(self.metadata["files"]) != expected_files:
            raise ValueError("Incomplete dataset manifest")
        for name, digest in self.metadata["files"].items():
            if sha256_file(self.qa_dir / name) != digest:
                raise ValueError(f"Dataset file changed: {name}")
        self.identity = json_hash(self.metadata)
        partitions = self.metadata["trajectory_splits"]
        if set(partitions) != {"train", "val", "test"}:
            raise ValueError("Incomplete trajectory partition")
        flattened = [n for ids in partitions.values() for n in ids]
        if len(flattened) != len(set(flattened)):
            raise ValueError("Trajectory leakage between splits")
        states = read_jsonl(self.qa_dir / "states.jsonl")
        self.states = {s["state_ref"]: s for s in states}
        if len(states) != len(self.states):
            raise ValueError("Duplicate state references")
        for state in states:
            if state["sample_index"] not in partitions[state["split"]]:
                raise ValueError("State violates its trajectory partition")
        self.records = read_jsonl(self.qa_dir / f"{split}.jsonl")
        if not self.records or len({r["qa_id"] for r in self.records}) != len(self.records):
            raise ValueError("Empty dataset or duplicate question IDs")
        for row in self.records:
            if row["task_type"] not in self.metadata["tasks"]:
                raise ValueError("Question task is not declared in the manifest")
            state = self.states[row["state_ref"]]
            if state["split"] != split or row["grid_shape"] != state["grid_shape"]:
                raise ValueError("Question references a wrong split/shape")
            if row["oracle"]["absolute_tolerance"] != self.metadata["scoring"]["absolute_tolerance"]:
                raise ValueError("Per-record scoring tolerance differs from manifest")
            validate_record(row)

    def __len__(self):
        return len(self.records)

    def field(self, state_ref):
        if state_ref in self.cache:
            self.cache.move_to_end(state_ref)
            return self.cache[state_ref]
        if self.handle is None:
            self.handle = h5py.File(self.hdf5_path, "r")
            for field, shape in self.metadata["source"]["field_layouts"].items():
                if list(self.handle[field].shape) != shape:
                    self.close()
                    raise ValueError("HDF5 field layout differs from the dataset source")
        state = self.states[state_ref]
        r, c = state["top_left"]
        h, w = state["grid_shape"]
        dataset = self.handle[state["field"]]
        sample, time = state["sample_index"], state["time_index"]
        if not (0 <= sample < dataset.shape[0] and 0 <= time < dataset.shape[1]
                and 0 <= r <= dataset.shape[2] - h and 0 <= c <= dataset.shape[3] - w):
            raise ValueError("Source crop is outside HDF5 bounds")
        z, audit = normalize_field(torch.from_numpy(dataset[sample, time, r:r+h, c:c+w].copy()))
        if audit != state["normalization"]:
            raise ValueError(f"Source crop changed: {state_ref}")
        if self.cache_size:
            self.cache[state_ref] = z
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return z

    def __getitem__(self, index):
        row = self.records[index]
        z = self.field(row["state_ref"])
        validate_record(row, z)
        return row, z.unsqueeze(0)

    def audit(self):
        for index in range(len(self)):
            self[index]
        return {"questions": len(self), "states": len({r['state_ref'] for r in self.records}),
                "identity": self.identity}

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        self.cache.clear()
