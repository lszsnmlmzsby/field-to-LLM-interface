"""Versioned, trajectory-disjoint scalar QA from complete INS and The Well sources.

The source inventory is complete; the QA manifest explicitly records the sampled
crops. Vector components are separate scalar examples and static fields have no
time index. Neither source identifiers nor physical metadata enter neural prompts.
"""
from __future__ import annotations

import copy
import json
import math
import random
from collections import Counter, OrderedDict, defaultdict
from contextlib import nullcontext
from pathlib import Path

import h5py
import torch

from tensor_compression.downstream.patch_qa_contract import sha256_file
from tensor_compression.downstream.point_readout import (
    FORMAT, PROMPT_VERSION, TASKS, json_hash, make_record, normalize_field,
    sample_spec, tensor_hash, validate_record,
)
from tensor_compression.downstream.random_readout_shapes import fitting_tasks, shape_pools


SPLITS = ("train", "val", "test")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _line(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _resolve(value):
    from tensor_compression.downstream.point_readout_data import resolve_path
    return resolve_path(value)


def _well_receipt(path, dataset):
    from tensor_compression.downstream.the_well_source import verify_dataset_receipt
    return verify_dataset_receipt(path, dataset_name=dataset)


def _inventory(config):
    """Bind every official HDF5 and every trajectory, not just sampled files."""
    sources, files, partitions = {}, {}, {s: [] for s in SPLITS}
    trajectory_hashes, initial_hashes = {}, defaultdict(set)
    source_configs = config["data"]["sources"]
    if not source_configs:
        raise ValueError("At least one source is required")
    for spec in source_configs:
        source_id, family, kind = spec["id"], spec["family"], spec["kind"]
        if not source_id or source_id in sources or not family:
            raise ValueError("Source IDs must be unique and families nonempty")
        requested = spec.get("fields")
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("Every source must explicitly select unique HDF5 field keys")
        path = _resolve(spec["path"])
        info = {"id": source_id, "family": family, "kind": kind,
                "path_hint": str(path), "fields": requested}
        if kind == "ins_tracer":
            from tensor_compression.downstream.ins_tracer_source import verify_receipt
            receipt = verify_receipt(path)
            shapes = receipt["source"]["fields"]
            if set(requested) - set(shapes):
                raise ValueError("Requested INS field is not present in complete-source receipt")
            n, t, h, w = shapes[requested[0]]
            if any(shapes[key] != [n, t, h, w] for key in requested):
                raise ValueError("INS fields must share trajectory/time/spatial axes")
            samples = list(range(n))
            random.Random(int(config["generation"]["split_seed"])).shuffle(samples)
            groups = {"train": samples[:int(n * .8)], "val": samples[int(n * .8):int(n * .9)],
                      "test": samples[int(n * .9):]}
            if n < 20 or min(map(len, groups.values())) < 2:
                raise ValueError("INS needs >=20 trajectories and >=2 per split")
            layout = {"n_trajectories": n, "time_steps": t, "spatial_dims": ["x", "y"],
                      "spatial_shape": [h, w], "fields": [
                          {"key": key, "rank": 0, "sample_varying": True, "time_varying": True,
                           "dim_varying": [True, True], "shape": shapes[key], "components": 1}
                          for key in requested]}
            entries = [{"path": path.name, "size_bytes": path.stat().st_size,
                        "mtime_ns": path.stat().st_mtime_ns, "sha256": receipt["sha256"],
                        "layout": layout, "trajectory_partition": groups}]
            info["verification"] = receipt
        elif kind == "the_well":
            receipt = _well_receipt(path, spec["dataset"])
            entries = [entry for entry in receipt["files"] if entry.get("layout") is not None]
            if not entries:
                raise ValueError("Complete Well receipt contains no HDF5 files")
            info.update(dataset=spec["dataset"], revision=receipt["revision"],
                        manifest_sha256=receipt["manifest_sha256"],
                        verification_schema=receipt["schema"])
        else:
            raise ValueError(f"Unsupported source kind: {kind}")
        source_splits = set()
        for entry in entries:
            layout = entry["layout"]
            available = {f["key"]: f for f in layout["fields"]}
            if set(requested) - set(available):
                raise ValueError(f"Missing requested fields in {source_id}/{entry['path']}: "
                                 f"{set(requested) - set(available)}")
            selected = []
            for key in requested:
                field = available[key]
                if field["rank"] not in (0, 1) or list(field["dim_varying"]) != [True, True]:
                    raise ValueError(f"Only full 2D scalar/vector fields are supported: {key}")
                if field["rank"] == 1 and field["components"] != 2:
                    raise ValueError(f"A 2D vector needs exactly two ordered components: {key}")
                for component in (range(field["components"]) if field["rank"] else (None,)):
                    selected.append({**field, "component": component,
                                     "label": key if component is None else f"{key}[{component}]"})
            file_id = json_hash({"source_id": source_id, "path": entry["path"], "sha256": entry["sha256"]})
            groups = entry.get("trajectory_partition")
            if groups is None:
                split = {"valid": "val", "train": "train", "val": "val", "test": "test"}.get(entry.get("split"))
                if split is None:
                    raise ValueError(f"HDF5 file lacks an official split: {entry['path']}")
                groups = {split: list(range(int(layout["n_trajectories"])))}
            sources_hash = entry.get("trajectory_sha256", layout.get("trajectory_sha256"))
            initial = entry.get("initial_state_sha256", layout.get("initial_state_sha256"))
            if kind == "the_well" and (not sources_hash or len(sources_hash) != layout["n_trajectories"]):
                raise ValueError("Well receipt must include full-trajectory content hashes")
            for split, indexes in groups.items():
                source_splits.add(split)
                for index in indexes:
                    identity = f"{file_id}:{index}"
                    partitions[split].append(identity)
                    if sources_hash:
                        digest = sources_hash[index]
                        previous = trajectory_hashes.get(digest)
                        if previous is not None and previous != split:
                            raise ValueError(f"Identical full trajectory leaks across {previous}/{split}: {source_id}")
                        trajectory_hashes[digest] = split
                    if initial:
                        initial_hashes[initial[index]].add(split)
            files[file_id] = {"source_id": source_id, "family": family, "path": entry["path"],
                              "sha256": entry["sha256"], "size_bytes": entry["size_bytes"],
                              "mtime_ns": entry["mtime_ns"], "layout": layout,
                              "selected_fields": selected, "trajectory_partition": groups}
        if source_splits != set(SPLITS):
            raise ValueError(f"Source {source_id} must contain the complete train/val/test inventory")
        sources[source_id] = info
    return sources, files, partitions, {
        "well_full_trajectory_hashes_checked": len(trajectory_hashes),
        "well_identical_trajectories_across_splits": 0,
        "shared_initial_state_hashes_across_splits": sum(len(s) > 1 for s in initial_hashes.values()),
        "initial_state_policy": "report_only; equal initial state can have different physical parameters",
        "ins_policy": "original seeded 80/10/10 trajectory partition; no full-trajectory content hash receipt",
    }


def _file_path(source, entry, overrides=None):
    root = Path((overrides or {}).get(source["id"], source["path_hint"]))
    return root if source["kind"] == "ins_tracer" else root / entry["path"]


class _Readers:
    def __init__(self, sources, files, max_open=8, overrides=None):
        self.sources, self.files, self.handles = sources, files, OrderedDict()
        self.max_open, self.overrides = max_open, overrides
        if max_open < 1:
            raise ValueError("max_open_hdf5 must be positive")

    def check_file(self, file_id):
        entry = self.files[file_id]
        path = _file_path(self.sources[entry["source_id"]], entry, self.overrides)
        stat = path.stat()
        if stat.st_size != entry["size_bytes"] or stat.st_mtime_ns != entry["mtime_ns"]:
            raise ValueError(f"Source file changed after QA build: {path}")
        return path

    def read(self, state):
        file_id = state["source_file_id"]
        entry = self.files[file_id]
        if file_id not in self.handles:
            path = self.check_file(file_id)
            handle = h5py.File(path, "r")
            try:
                for field in entry["selected_fields"]:
                    if list(handle[field["key"]].shape) != field["shape"]:
                        raise ValueError(f"Source field layout changed: {path}:{field['key']}")
            except Exception:
                handle.close()
                raise
            self.handles[file_id] = handle
            while len(self.handles) > self.max_open:
                self.handles.popitem(last=False)[1].close()
        self.handles.move_to_end(file_id)
        field = next(f for f in entry["selected_fields"] if f["label"] == state["field"])
        index = []
        if field["sample_varying"]:
            index.append(state["sample_index"])
        if field["time_varying"]:
            index.append(state["time_index"])
        r, c = state["top_left"]
        h, w = state["grid_shape"]
        index.extend((slice(r, r+h), slice(c, c+w)))
        if field["component"] is not None:
            index.append(field["component"])
        raw = torch.from_numpy(self.handles[file_id][field["key"]][tuple(index)].copy())
        if tuple(raw.shape) != (h, w):
            raise ValueError("Source crop is outside HDF5 bounds")
        return raw

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def _balanced_counts(items, count):
    """Deterministic exact totals with imbalance of at most one sampled state."""
    return {item: count // len(items) + (i < count % len(items)) for i, item in enumerate(items)}


def _source_budgets(generation, sources, families, family_sources, split, total):
    explicit = generation.get("train_states_by_source") if split == "train" else None
    if explicit is not None:
        if (not isinstance(explicit, dict) or set(explicit) != set(sources)
                or any(type(n) is not int or n < 1 for n in explicit.values())
                or sum(explicit.values()) != total):
            raise ValueError("train_states_by_source must specify every source with positive integer "
                             "counts summing exactly to train_states")
        return {family: {sid: explicit[sid] for sid in family_sources[family]} for family in families}
    return {family: _balanced_counts(family_sources[family], count)
            for family, count in _balanced_counts(families, total).items()}


def _rng(seed, *parts):
    """Independent reproducible streams prevent language changes from moving crops."""
    return random.Random(int(json_hash([int(seed), *parts]), 16))


class _TrajectoryCycle:
    """Cover available trajectories before repeating within each scalar field.

    Each source's official split is fixed upstream. Cycling does not imply that
    adjacent times or spatial crops are statistically independent.
    """
    def __init__(self, eligible, split, rng):
        self.rng, self.split = rng, split
        self.choices, self.positions, self.times = defaultdict(list), Counter(), {}
        for file_id, entry in eligible.items():
            for field in entry["selected_fields"]:
                indexes = entry["trajectory_partition"][split]
                if not field["sample_varying"]:
                    indexes = indexes[:1]
                for sample in indexes:
                    self.choices[field["label"]].append((file_id, entry, field, sample))
        for candidates in self.choices.values():
            rng.shuffle(candidates)

    def next(self, label):
        candidates, position = self.choices[label], self.positions[label]
        if position and position % len(candidates) == 0:
            self.rng.shuffle(candidates)
        self.positions[label] += 1
        file_id, entry, field, sample = candidates[position % len(candidates)]
        time = None
        if field["time_varying"]:
            key = (file_id, label, sample if field["sample_varying"] else None)
            steps = int(entry["layout"]["time_steps"])
            if key not in self.times:
                strides = [stride for stride in range(1, steps + 1) if math.gcd(stride, steps) == 1]
                self.times[key] = [self.rng.randrange(steps), self.rng.choice(strides), 0]
            offset, stride, visits = self.times[key]
            time = (offset + visits * stride) % steps
            self.times[key][2] += 1
        return file_id, entry, field, sample, time


def _roughness(z, constant):
    if constant:
        return 0.0, "constant"
    square_sum, edges = 0.0, 0
    for axis in (0, 1):
        if z.shape[axis] > 1:
            difference = torch.diff(z.float(), dim=axis)
            square_sum += float(difference.square().sum())
            edges += difference.numel()
    value = math.sqrt(square_sum / edges) if edges else 0.0
    return value, "smooth" if value < .25 else "moderate" if value < 1 else "sharp"


def _sample_programs(tasks, shape, z, policy, seed, state_ref):
    from tensor_compression.downstream.field_query_program import TASK_CAPABILITIES, sample_program
    programs, unavailable = {}, []
    for task in tasks:
        program = sample_program(task, shape, _rng(seed, state_ref, "program", task), policy, z)
        if program is None:
            unavailable.append(task)
        else:
            programs[task] = program
    wanted = int(policy.get("tasks_per_state", len(tasks)))
    weights = policy.get("capability_weights", {"readout": 4, "localization": 3,
                                                "aggregation": 2, "composition": 1})
    selection = _rng(seed, state_ref, "task_selection")
    selected = []
    while programs and len(selected) < wanted:
        groups = Counter(TASK_CAPABILITIES[task] for task in programs)
        candidates = list(programs)
        probabilities = [weights[TASK_CAPABILITIES[task]] / groups[TASK_CAPABILITIES[task]] for task in candidates]
        task = selection.choices(candidates, weights=probabilities, k=1)[0]
        selected.append((task, programs.pop(task)))
    return selected, unavailable


def _query_policy(generation):
    policy = generation.get("query_program", {})
    if not isinstance(policy, dict):
        raise ValueError("query_program must be a mapping")
    if not policy.get("enabled", False):
        return None
    from tensor_compression.downstream.field_query_program import TASK_CAPABILITIES
    count = policy.get("tasks_per_state", len(generation["tasks"]))
    diagnostic_count = policy.get("diagnostic_states_per_source", 16)
    if type(count) is not int or count < 1 or count > len(generation["tasks"]):
        raise ValueError("tasks_per_state must be a positive integer no greater than the selected tasks")
    if type(diagnostic_count) is not int or diagnostic_count < 0:
        raise ValueError("diagnostic_states_per_source must be a nonnegative integer")
    weights = policy.get("capability_weights", {"readout": 4, "localization": 3,
                                                "aggregation": 2, "composition": 1})
    if (not isinstance(weights, dict) or set(weights) != set(TASK_CAPABILITIES.values())
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in weights.values())):
        raise ValueError("capability_weights must give every capability a finite positive weight")
    for name in ("read_count_max", "line_count_max", "region_value_max_cells", "statistic_max_cells"):
        if name in policy and (type(policy[name]) is not int or policy[name] < 1):
            raise ValueError(f"{name} must be a positive integer")
    return policy


def _shape_domains(policy, files):
    # Numeric bounds are strict, never silently clipped to a smaller source.
    shapes = {tuple(entry["layout"]["spatial_shape"]) for entry in files.values()}
    if any("source" in policy[axis] for axis in ("height", "width")):
        raise ValueError("Multisource protocol requires explicit shared integer shape bounds")
    for extent in shapes:
        shape_pools(policy, extent)  # Checks every file, including unselected trajectories.
    return shape_pools(policy, next(iter(shapes)))


def _plan_source(files, source_id, split, count, pool, heldout, anchors, seen_shapes, rng):
    eligible = {file_id: entry for file_id, entry in files.items()
                if entry["source_id"] == source_id and split in entry["trajectory_partition"]}
    labels = sorted({f["label"] for entry in eligible.values() for f in entry["selected_fields"]})
    if count < len(anchors) * len(labels) * (1 if split == "train" else 2):
        raise ValueError("State budget must cover every source/field boundary and evaluation holdout")
    pairs = [(shape, label) for shape in anchors for label in labels]
    for i in range(count - len(pairs)):
        shapes = pool if split == "train" else heldout if i % 2 == 0 else seen_shapes
        pairs.append((rng.choice(shapes), labels[i % len(labels)]))
    return pairs, eligible


def build_multisource_dataset(config, output_dir):
    """Build streaming immutable JSONL with exact family/source state budgets."""
    config = copy.deepcopy(config)
    data, generation = config["data"], config["generation"]
    query_policy = _query_policy(generation)
    if query_policy is not None:
        from tensor_compression.downstream.field_query_program import (
            PROGRAM_TASKS, TEMPLATE_SPLITS, make_program_record,
        )
        supported = PROGRAM_TASKS
    else:
        supported = TASKS
    tasks = generation["tasks"]
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(supported):
        raise ValueError("Select unique supported tasks")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a new empty dataset directory: {output_dir}")
    sources, files, partitions, leakage = _inventory(config)
    pool, heldout, anchors = _shape_domains(data["shape_sampling"], files)
    families = sorted({source["family"] for source in sources.values()})
    family_sources = {family: sorted(s for s, info in sources.items() if info["family"] == family)
                      for family in families}
    train_count, eval_count = int(generation["train_states"]), int(generation["eval_states"])
    if min(train_count, eval_count) <= 0:
        raise ValueError("Training and evaluation state budgets must be positive")
    allocations = {split: _source_budgets(generation, sources, families, family_sources, split,
                                        train_count if split == "train" else eval_count) for split in SPLITS}
    multi, lines = generation.get("multi_counts", [2, 3, 4]), generation.get("line_counts", [2, 3, 4, 5])
    regions = generation.get("region_shapes", [[1, 1], [1, 2], [2, 1], [2, 2]])
    stats = generation.get("statistic_region_shapes", [[1, 1], [1, 2], [2, 1], [2, 2], [4, 4]])
    variants = generation.get("template_variants", [0, 1])
    attempts = int(generation.get("max_attempts_per_state", 200))
    decimals, tolerance = generation.get("answer_decimals", 1), float(generation.get("absolute_tolerance", .2))
    if attempts < 1 or not variants or any(v not in (0, 1, 2) for v in variants):
        raise ValueError("Invalid attempts/template variants")
    rng = random.Random(int(generation["seed"]))
    readers = _Readers(sources, files, int(data.get("max_open_hdf5", 8)))
    seen_crops, train_shapes = set(), set()
    counts, task_counts, state_counts = {}, {}, {}
    family_counts, source_counts, field_counts, shape_counts = {}, {}, {}, {}
    static_counts, constant_counts, rejected, inapplicable = Counter(), Counter(), Counter(), Counter()
    capability_counts, difficulty_counts, question_source_counts = {}, {}, {}
    diagnostic_counts, diagnostic_states = Counter(), Counter()
    shortfalls, roughness_counts = Counter(), Counter()
    coverage_sets = defaultdict(lambda: defaultdict(set))
    coverage = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / ".build_in_progress"
    marker.write_text(FORMAT, encoding="utf-8")
    try:
        with (output_dir / "states.jsonl").open("w", encoding="utf-8", newline="\n") as state_out, \
                h5py.File(output_dir / "qa_fields.hdf5", "w") as cache_file:
            cached_values = cache_file.create_dataset("values", shape=(0,), maxshape=(None,),
                                                      chunks=(131072,), dtype="float16")
            cache_file.attrs["purpose"] = "Exact normalized sampled QA crops; not a complete source dataset"
            for split in SPLITS:
                budget = train_count if split == "train" else eval_count
                fcounts, scounts, fieldcount, shapes, tasks_count = Counter(), Counter(), Counter(), Counter(), Counter()
                capabilities, difficulties, qsource = Counter(), Counter(), Counter()
                total = 0
                diagnostics_enabled = query_policy is not None and split != "train" and query_policy.get("diagnostic_states_per_source", 16) > 0
                diagnostic_context = ((output_dir / f"diagnostics_{split}.jsonl").open("w", encoding="utf-8", newline="\n")
                                      if diagnostics_enabled else nullcontext(None))
                with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as qa_out, diagnostic_context as diag_out:
                    for family, source_allocations in allocations[split].items():
                        for source_id, count in source_allocations.items():
                            shape_rng = _rng(generation["seed"], split, source_id, "shapes") if query_policy is not None else rng
                            crop_rng = _rng(generation["seed"], split, source_id, "crops") if query_policy is not None else rng
                            plan, eligible = _plan_source(files, source_id, split, count, pool, heldout, anchors,
                                                          sorted(train_shapes), shape_rng)
                            cycle = (_TrajectoryCycle(eligible, split, _rng(generation["seed"], split, source_id, "trajectories"))
                                     if query_policy is not None else None)
                            # Spread diagnostic states through the source plan, not only boundary anchors.
                            diag_limit = min(count, query_policy.get("diagnostic_states_per_source", 16)) if diagnostics_enabled else 0
                            diag_indices = set(_rng(generation["seed"], split, source_id, "diagnostic_states").sample(range(count), diag_limit))
                            field_candidates = {}
                            for file_id, entry in eligible.items():
                                for field in entry["selected_fields"]:
                                    field_candidates.setdefault(field["label"], []).append((file_id, entry, field))
                            field_weights = {label: [len(e["trajectory_partition"][split]) if f["sample_varying"] else 1
                                                      for _, e, f in candidates]
                                             for label, candidates in field_candidates.items()}
                            for plan_index, (shape, label) in enumerate(plan):
                                candidates = field_candidates[label]
                                h, w = shape
                                for _ in range(attempts):
                                    # Weight files by trajectory count, so shards do not change the sampling law.
                                    if cycle is None:
                                        file_id, entry, field = rng.choices(candidates, weights=field_weights[label], k=1)[0]
                                        sample = rng.choice(entry["trajectory_partition"][split])
                                        time = rng.randrange(entry["layout"]["time_steps"]) if field["time_varying"] else None
                                    else:
                                        file_id, entry, field, sample, time = cycle.next(label)
                                    sh, sw = entry["layout"]["spatial_shape"]
                                    state = {"source_id": source_id, "family": family, "source_family": family,
                                             "source_file_id": file_id,
                                             "field": label, "field_key": field["key"], "component": field["component"],
                                             "sample_index": sample,
                                             "time_index": time,
                                             "top_left": [crop_rng.randrange(sh-h+1), crop_rng.randrange(sw-w+1)],
                                             "grid_shape": list(shape)}
                                    # Shared static fields do not become distinct examples by changing trajectory IDs.
                                    crop_identity = dict(state)
                                    if not field["sample_varying"]:
                                        crop_identity["sample_index"] = None
                                    key = json_hash(crop_identity)
                                    if key in seen_crops:
                                        rejected["duplicate_source_crop"] += 1
                                        continue
                                    raw = readers.read(state)
                                    z, audit = normalize_field(raw, allow_singleton=True, allow_constant=True)
                                    break
                                else:
                                    raise ValueError(f"Cannot sample unique {split}/{source_id}/{label}/{shape}; "
                                                     "budget exceeds available distinct crops")
                                seen_crops.add(key)
                                offset = len(cached_values)
                                cached_values.resize((offset + z.numel(),))
                                cached_values[offset:] = z.half().numpy().reshape(-1)
                                state.update(state_ref=key, split=split, normalization=audit, cache_offset=offset,
                                             is_static=not field["time_varying"],
                                             is_constant=bool(torch.all(raw == raw.reshape(-1)[0])))
                                if query_policy is not None:
                                    state["roughness"], state["roughness_bin"] = _roughness(z, state["is_constant"])
                                    roughness_counts[f"{split}/{source_id}/{state['roughness_bin']}"] += 1
                                    observed = coverage_sets[f"{split}/{source_id}"]
                                    observed["source_files"].add(file_id)
                                    if field["sample_varying"]:
                                        observed["trajectories"].add((file_id, sample))
                                    observed["field_trajectories"].add((file_id, label, sample if field["sample_varying"] else None))
                                    observed["field_frames"].add((file_id, label, sample if field["sample_varying"] else None, time))
                                    observed["top_left_coordinates"].add(tuple(state["top_left"]))
                                    observed["shapes"].add(shape)
                                _line(state_out, state)
                                if split == "train":
                                    train_shapes.add(shape)
                                fcounts[family] += 1
                                scounts[source_id] += 1
                                fieldcount[f"{source_id}/{label}"] += 1
                                shapes[f"{h}x{w}"] += 1
                                static_counts[split] += state["is_static"]
                                constant_counts[split] += state["is_constant"]
                                if query_policy is None:
                                    active, mc, lc, rs, ss = fitting_tasks(tasks, shape, multi, lines, regions, stats)
                                    programs = [(task, None) for task in active]
                                    unavailable = set(tasks) - set(active)
                                else:
                                    programs, unavailable = _sample_programs(tasks, shape, z, query_policy, generation["seed"], key)
                                    shortfalls[split] += int(query_policy.get("tasks_per_state", len(tasks))) - len(programs)
                                    if not programs:
                                        raise ValueError(f"No query program fits {split}/{source_id}/{shape}")
                                for task in unavailable:
                                    inapplicable[f"{split}/{task}"] += 1
                                if plan_index in diag_indices:
                                    diagnostic_states[f"{split}/{source_id}"] += 1
                                for task, program in programs:
                                    if query_policy is None:
                                        spec = sample_spec(task, shape, rng, multi_counts=mc, line_counts=lc,
                                                           region_shapes=rs, statistic_region_shapes=ss, z=z)
                                        row = make_record(key, z, task, spec, variant=rng.choice(variants),
                                                          decimals=decimals, tolerance=tolerance)
                                    else:
                                        template = _rng(generation["seed"], key, task, "main_language").choice(TEMPLATE_SPLITS["train"])
                                        row = make_program_record(key, z, task, program, template, decimals=decimals,
                                                                  tolerance=tolerance, language_group="seen")
                                        capabilities[row["capability"]] += 1
                                        difficulty_key = (row["difficulty"] if isinstance(row["difficulty"], str)
                                                          else json.dumps(row["difficulty"], sort_keys=True, separators=(",", ":")))
                                        difficulties[difficulty_key] += 1
                                        qsource[f"{source_id}/{row['capability']}/{task}"] += 1
                                        if plan_index in diag_indices:
                                            _line(diag_out, row)
                                            heldout_template = _rng(generation["seed"], key, task, "heldout_language").choice(TEMPLATE_SPLITS[split])
                                            heldout_row = make_program_record(key, z, task, program, heldout_template,
                                                                             decimals=decimals, tolerance=tolerance,
                                                                             language_group="heldout")
                                            validate_record(heldout_row, z)
                                            _line(diag_out, heldout_row)
                                            diagnostic_counts[split] += 2
                                    validate_record(row, z)
                                    _line(qa_out, row)
                                    tasks_count[task] += 1
                                    total += 1
                                built_states = sum(fcounts.values())
                                if built_states % 1024 == 0:
                                    print(f"build_progress split={split} states={built_states}/{budget} "
                                          f"questions={total}", flush=True)
                counts[split], state_counts[split], task_counts[split] = total, sum(fcounts.values()), dict(tasks_count)
                family_counts[split], source_counts[split] = dict(fcounts), dict(scounts)
                field_counts[split], shape_counts[split] = dict(fieldcount), dict(shapes)
                capability_counts[split], difficulty_counts[split] = dict(capabilities), dict(difficulties)
                question_source_counts[split] = dict(qsource)
                for key in list(coverage_sets):
                    if key.startswith(split + "/"):
                        coverage[key] = {name: len(values) for name, values in coverage_sets.pop(key).items()}
                        coverage[key]["sampled_crops"] = scounts[key.split("/", 1)[1]]
                print(f"built split={split} states={state_counts[split]} questions={total} families={dict(fcounts)}", flush=True)
    finally:
        readers.close()
    names = ["states.jsonl", "train.jsonl", "val.jsonl", "test.jsonl", "qa_fields.hdf5"]
    if query_policy is not None and query_policy.get("diagnostic_states_per_source", 16) > 0:
        names += ["diagnostics_val.jsonl", "diagnostics_test.jsonl"]
    metadata = {"format": FORMAT, "prompt_version": PROMPT_VERSION,
                "source": {"kind": "multisource", "sources": sources, "files": files},
                "trajectory_splits": partitions, "leakage_audit": leakage,
                "train_shapes": sorted(train_shapes), "heldout_shapes": heldout,
                "shape_sampling": data["shape_sampling"], "shape_boundary_anchors": anchors,
                "tasks": tasks, "generation": generation, "config": config,
                "normalization": "float32_population_zscore_eps1e-6_then_fp16_constant_to_zero",
                "allow_singleton": True, "allow_constant": True,
                "scoring": {"absolute_tolerance": tolerance, "answer_decimals": decimals,
                            "invalid_answer_counts_as_wrong": True},
                "counts": counts, "task_counts": task_counts, "state_counts": state_counts,
                "family_state_counts": family_counts, "source_state_counts": source_counts,
                "field_state_counts": field_counts, "shape_counts": shape_counts,
                "static_state_counts": dict(static_counts), "constant_state_counts": dict(constant_counts),
                "sampling": ("explicit training source budgets; evaluation equal family then source" if generation.get("train_states_by_source")
                             else "equal family, then equal source") + (
                                 "; balanced scalar components; shuffled trajectory cycles and time cycles within each field" if query_policy is not None
                                 else "; balanced scalar components; trajectories weighted equally"),
                "qa_field_cache": {"file": "qa_fields.hdf5", "dataset": "values", "dtype": "float16",
                                   "semantics": "exact existing fp16-rounded normalized sampled crops; no extra quantization",
                                   "source_audit": "audit replays original source crops and compares raw/z hashes and cache"},
                "inapplicable_tasks": dict(inapplicable), "rejected_attempts": dict(rejected),
                "files": {name: sha256_file(output_dir / name) for name in names}}
    if query_policy is not None:
        metadata.update(query_program={"enabled": True, "template_splits": TEMPLATE_SPLITS,
                                       "main_language_group": "seen", "one_main_question_per_selected_program": True},
                        capability_question_counts=capability_counts, difficulty_question_counts=difficulty_counts,
                        capability_weight_policy="weighted capability draws, then equal task weights within capability, without replacement per crop; realized counts reported",
                        source_capability_task_question_counts=question_source_counts,
                        query_budget_shortfall=dict(shortfalls), coverage=coverage,
                        coverage_policy="counts of distinct sampled keys; adjacent frames/crops are not asserted independent",
                        roughness_state_counts=dict(roughness_counts),
                        roughness_definition="RMS first-neighbor difference of normalized crop; constant separate, smooth<0.25, moderate<1, sharp>=1",
                        diagnostics={"counts": dict(diagnostic_counts), "states_by_source": dict(diagnostic_states),
                                     "purpose": "paired seen/heldout wording for identical semantic queries on existing evaluation states",
                                     "adds_training_states": False},
                        task_scoring={task: {"answer_decimals": 3 if task == "threshold_fraction" else decimals,
                                             "absolute_tolerance": .02 if task == "threshold_fraction" else tolerance} for task in tasks})
    _write(output_dir / "metadata.json", metadata)
    marker.unlink()
    return metadata


class MultiSourcePointReadoutDataset:
    """Lazy, bounded-handle replay of immutable scalar crops across source files."""
    def __init__(self, qa_dir, split, cache_size=8, source_root_overrides=None, suite="main"):
        if split not in SPLITS or cache_size < 0:
            raise ValueError("Invalid split/cache size")
        if suite not in ("main", "language") or (suite == "language" and split == "train"):
            raise ValueError("Language diagnostic suite is available only for val/test")
        self.suite = suite
        self.qa_dir, self.cache_size, self.cache = Path(qa_dir), cache_size, OrderedDict()
        if (self.qa_dir / ".build_in_progress").exists():
            raise ValueError("Dataset build did not finish")
        self.metadata = _json(self.qa_dir / "metadata.json")
        if (self.metadata.get("format") != FORMAT or self.metadata.get("prompt_version") != PROMPT_VERSION
                or self.metadata.get("source", {}).get("kind") != "multisource"):
            raise ValueError("Unsupported multi-source QA protocol")
        expected_files = {"states.jsonl", "train.jsonl", "val.jsonl", "test.jsonl", "qa_fields.hdf5"}
        query_enabled = self.metadata.get("query_program", {}).get("enabled", False)
        diagnostic_count = self.metadata["generation"].get("query_program", {}).get("diagnostic_states_per_source", 16)
        if query_enabled and diagnostic_count > 0:
            expected_files.update(("diagnostics_val.jsonl", "diagnostics_test.jsonl"))
        if set(self.metadata["files"]) != expected_files:
            raise ValueError("Incomplete dataset manifest")
        if suite == "language" and f"diagnostics_{split}.jsonl" not in expected_files:
            raise ValueError("This dataset has no paired language diagnostics; rebuild with query_program enabled")
        for name, digest in self.metadata["files"].items():
            if sha256_file(self.qa_dir / name) != digest:
                raise ValueError(f"Dataset file changed: {name}")
        self.identity = json_hash(self.metadata)
        parts = self.metadata["trajectory_splits"]
        flattened = [x for ids in parts.values() for x in ids]
        if set(parts) != set(SPLITS) or len(flattened) != len(set(flattened)):
            raise ValueError("Trajectory leakage or incomplete partitions")
        part_sets = {s: set(ids) for s, ids in parts.items()}
        from tensor_compression.downstream.point_readout_data import read_jsonl
        states = read_jsonl(self.qa_dir / "states.jsonl")
        self.states = {state["state_ref"]: state for state in states}
        if len(states) != len(self.states):
            raise ValueError("Duplicate state references")
        for state in states:
            if f"{state['source_file_id']}:{state['sample_index']}" not in part_sets[state["split"]]:
                raise ValueError("State violates trajectory partition")
        self.records = read_jsonl(self.qa_dir / (f"diagnostics_{split}.jsonl" if suite == "language" else f"{split}.jsonl"))
        if not self.records or len({r["qa_id"] for r in self.records}) != len(self.records):
            raise ValueError("Empty dataset or duplicate question IDs")
        for row in self.records:
            state = self.states[row["state_ref"]]
            if state["split"] != split or row["grid_shape"] != state["grid_shape"]:
                raise ValueError("Question references a wrong split/shape")
            scoring = self.metadata.get("task_scoring", {}).get(row["task_type"], self.metadata["scoring"])
            if (row["task_type"] not in self.metadata["tasks"] or
                    row["oracle"]["absolute_tolerance"] != scoring["absolute_tolerance"]):
                raise ValueError("Question differs from manifest task/scoring policy")
            if query_enabled:
                from tensor_compression.downstream.field_query_program import TEMPLATE_SPLITS
                if row["oracle"].get("answer_decimals") != scoring["answer_decimals"]:
                    raise ValueError("Question differs from manifest answer precision policy")
                allowed_templates = TEMPLATE_SPLITS["train"] if row.get("language_group") == "seen" else TEMPLATE_SPLITS[split]
                if (not row.get("query_protocol") or row.get("template_family") not in allowed_templates
                        or row.get("language_group") not in ("seen", "heldout")
                        or (suite == "main" and row.get("language_group") != "seen")):
                    raise ValueError("Question violates the language template partition")
            validate_record(row)
        if suite == "language":
            pairs = defaultdict(list)
            for row in self.records:
                pairs[row["semantic_id"]].append(row)
            for rows in pairs.values():
                if (len(rows) != 2 or {row["language_group"] for row in rows} != {"seen", "heldout"}
                        or rows[0]["oracle"] != rows[1]["oracle"]
                        or rows[0]["state_ref"] != rows[1]["state_ref"]):
                    raise ValueError("Language diagnostics must pair identical semantic queries and oracles")
        source = self.metadata["source"]
        self.readers = _Readers(source["sources"], source["files"],
                                int(self.metadata["config"]["data"].get("max_open_hdf5", 8)), source_root_overrides)
        self.cache_handle, self.checked_sources = None, set()

    def __len__(self):
        return len(self.records)

    def field(self, state_ref):
        if state_ref in self.cache:
            self.cache.move_to_end(state_ref)
            return self.cache[state_ref]
        state = self.states[state_ref]
        if state["source_file_id"] not in self.checked_sources:
            self.readers.check_file(state["source_file_id"])
            self.checked_sources.add(state["source_file_id"])
        if self.cache_handle is None:
            self.cache_handle = h5py.File(self.qa_dir / "qa_fields.hdf5", "r")
        h, w = state["grid_shape"]
        start = state["cache_offset"]
        values = self.cache_handle["values"][start:start + h*w]
        if values.size != h*w:
            raise ValueError(f"Incomplete QA field cache: {state_ref}")
        z = torch.from_numpy(values.reshape(h, w)).float()
        if tensor_hash(z) != state["normalization"]["z_sha256"]:
            raise ValueError(f"QA field cache changed: {state_ref}")
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
        # Audit original data, then train from compact exact QA caches. Never use a
        # sampled QA cache as evidence that complete source verification succeeded.
        previous, original = None, None
        for row in self.records:
            state = self.states[row["state_ref"]]
            if state["state_ref"] != previous:
                original, audit = normalize_field(self.readers.read(state), allow_singleton=True, allow_constant=True)
                if audit != state["normalization"]:
                    raise ValueError(f"Source crop changed: {state['state_ref']}")
                if not torch.equal(original, self.field(state["state_ref"])):
                    raise ValueError(f"Cached QA differs from original source: {state['state_ref']}")
                previous = state["state_ref"]
            validate_record(row, original)
        return {"questions": len(self), "states": len({r["state_ref"] for r in self.records}),
                "identity": self.identity,
                "suite": self.suite,
                "family_questions": dict(Counter(self.states[r["state_ref"]]["source_family"] for r in self.records))}

    def close(self):
        self.readers.close()
        if self.cache_handle is not None:
            self.cache_handle.close()
            self.cache_handle = None
        self.checked_sources.clear()
        self.cache.clear()
