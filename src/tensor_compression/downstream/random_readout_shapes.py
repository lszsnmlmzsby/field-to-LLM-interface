"""Reproducible integer-grid sampling with trajectory-independent shape holdouts."""
from __future__ import annotations

import random


def shape_pools(policy, source_shape):
    bounds = []
    for axis, extent in zip(("height", "width"), source_shape):
        interval = policy[axis]
        if (len(interval) != 2 or any(type(n) is not int for n in interval)
                or not 1 <= interval[0] <= interval[1] <= extent):
            raise ValueError(f"Invalid {axis} interval {interval}; source extent is {extent}")
        bounds.append(interval)
    (h0, h1), (w0, w1) = bounds
    candidates = [(h, w) for h in range(h0, h1 + 1) for w in range(w0, w1 + 1)]
    fraction = policy.get("heldout_fraction", .2)
    if type(fraction) not in (int, float) or not 0 < fraction < 1:
        raise ValueError("heldout_fraction must be between zero and one")
    # Boundary cases ensure the smoke run can exercise the largest grid and thin axes.
    anchors = list(dict.fromkeys([(h0, w0), (h0, w1), (h1, w0), (h1, w1)]))
    eligible = [s for s in candidates if s not in anchors]
    random.Random(int(policy["seed"])).shuffle(eligible)
    count = int(len(candidates) * fraction)
    if not 1 <= count < len(eligible):
        raise ValueError("Shape domain is too small for training and held-out shapes")
    heldout = sorted(eligible[:count])
    heldout_set = set(heldout)
    return [s for s in candidates if s not in heldout_set], heldout, anchors


def training_plan(pool, anchors, fields, count, rng):
    if count < len(anchors) * len(fields):
        raise ValueError("Random-shape training must cover every boundary shape and field")
    plan = [(s, f) for s in anchors for f in fields]
    plan.extend((rng.choice(pool), fields[i % len(fields)]) for i in range(count - len(plan)))
    return plan


def evaluation_plan(seen, heldout, anchors, fields, count, rng):
    boundary = [(s, f) for s in anchors for f in fields]
    if count < 2 * len(boundary):
        raise ValueError("Evaluation count must cover boundaries and shape holdouts")
    plan = list(boundary)
    for i in range(count - len(plan)):
        pool = heldout if i % 2 == 0 else seen
        plan.append((rng.choice(pool), fields[i % len(fields)]))
    return plan


def fitting_tasks(tasks, shape, multi_counts, line_counts, regions, statistic_regions):
    h, w = shape
    multi = [n for n in multi_counts if n <= h * w]
    lines = [n for n in line_counts if n <= max(h, w)]
    region = [(rh, rw) for rh, rw in regions if rh <= h and rw <= w]
    stats = [(rh, rw) for rh, rw in statistic_regions if rh <= h and rw <= w]
    # Keep singleton regions for the degenerate 1x1 grid, not easy questions on large grids.
    if any(rh * rw > 1 for rh, rw in region):
        region = [(rh, rw) for rh, rw in region if rh * rw > 1]
    if any(rh * rw > 1 for rh, rw in stats):
        stats = [(rh, rw) for rh, rw in stats if rh * rw > 1]
    valid = []
    for task in tasks:
        fits = (True if task == "single_point" else bool(multi) if task == "multi_point"
                else bool(lines) if task == "line_profile"
                else bool(region) if task == "region_values" else bool(stats))
        if fits:
            valid.append(task)
    if not valid:
        raise ValueError(f"No configured tasks fit shape {shape}")
    return valid, multi, lines, region, stats
