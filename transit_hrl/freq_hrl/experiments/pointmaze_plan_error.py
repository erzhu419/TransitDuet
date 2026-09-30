"""Forecast/control accounting on recorded native rollouts, without simulation."""

import json
from pathlib import Path

import numpy as np
from freq_hrl.domains.mujoco.pointmaze_regime import PointMazeRegimeDriver
from .pointmaze_plan_validity_branching import _task_options
from scripts import pointmaze_plan_error_stage53_spec as spec


def event_partitions(target, regime_changes, period):
    horizon = len(target)
    regime = np.zeros(horizon, dtype=bool)
    changed = np.asarray(regime_changes, dtype=int) + 1
    regime[changed[changed < horizon]] = True
    movement = np.diff(np.asarray(target, dtype=np.float64), axis=0)
    axis = np.argmax(np.abs(movement), axis=1)
    geometry = np.zeros(horizon, dtype=bool)
    geometry[2:] = (axis[1:] != axis[:-1]) | (
        (np.einsum("ij,ij->i", movement[1:], movement[:-1]) < 0) & ~regime[2:])
    labels = {k: np.empty(horizon, dtype=np.int64) for k in spec.PARTITIONS}
    for start in range(0, horizon, period):
        stop = min(start + period, horizon)
        first_increment = max(0, start + 1 - spec.source.LOOKBACK_STEPS) + 1
        history = slice(first_increment, start + 1)
        future = slice(start + 1, stop)
        has_regime = bool(regime[history].any() or regime[future].any())
        has_geometry = bool(geometry[history].any() or geometry[future].any())
        has_history = bool(regime[history].any() or geometry[history].any())
        has_future = bool(regime[future].any() or geometry[future].any())
        labels["event"][start:stop] = int(has_regime) + 2 * int(has_geometry)
        labels["timing"][start:stop] = int(has_history) + 2 * int(has_future)
        labels["phase"][start:stop] = np.minimum(2, 3 * np.arange(stop - start) // period)
    return labels


def decompose(raw, row, expected_measurements):
    measured = raw["measurement"]
    target = raw["target_before"]
    np.testing.assert_array_equal(measured, expected_measurements, err_msg="wrong exogenous path or event alignment")
    np.testing.assert_array_equal(target, measured[:, :2], err_msg="reward target must be the pre-action target")
    target, reference, achieved = [np.asarray(x, dtype=np.float64) for x in
                                  (target, raw["lower_reference"], raw["achieved_after"])]
    controller, forecast = achieved - reference, reference - target
    values = np.column_stack((np.einsum("ij,ij->i", forecast, forecast),
        np.einsum("ij,ij->i", controller, controller),
        2 * np.einsum("ij,ij->i", controller, forecast),
        np.einsum("ij,ij->i", achieved - target, achieved - target))) * spec.source.DT_SECONDS
    np.testing.assert_allclose(values[:, 3], values[:, :3].sum(axis=1), atol=1e-12, rtol=1e-12,
                               err_msg="signed forecast/controller decomposition does not close")
    distance = np.asarray(raw["distance"])
    native_ise = float(np.dot(distance, distance) * spec.source.DT_SECONDS)
    np.testing.assert_allclose(native_ise, row["tracking_squared_error_integral"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(values[:, 0].sum(), row["reference_target_squared_error_integral"], atol=1e-12, rtol=1e-12)
    reward = np.asarray(raw["reward"], dtype=np.float64)
    np.testing.assert_allclose(reward.sum(), row["episode_return"], atol=1e-12, rtol=1e-12)
    # Native distance is a float32 norm; vector accounting keeps float64 algebra.
    np.testing.assert_allclose(values[:, 3].sum(), native_ise, atol=1e-7, rtol=5e-7)
    return np.column_stack((values, reward))


def reduce_paths(values, partitions):
    totals = np.zeros(len(spec.METRICS))
    panels = {name: {label: {"steps": 0, "sums": np.zeros(len(spec.METRICS))}
              for label in labels} for name, labels in spec.PARTITIONS.items()}
    for path, labels in zip(values, partitions):
        totals += path.sum(axis=0)
        for name, buckets in panels.items():
            for index, bucket in enumerate(buckets.values()):
                selected = labels[name] == index
                bucket["steps"] += int(selected.sum())
                bucket["sums"] += path[selected].sum(axis=0)
    for buckets in panels.values():
        np.testing.assert_allclose(sum(b["sums"] for b in buckets.values()), totals, atol=1e-10, rtol=1e-12)
    return dict(zip(spec.METRICS, (totals / len(values)).tolist())), panels


def contrasts(means):
    return {f"period{p}:curve_minus_hold:{m}": means[str(p)]["deterministic"]["target_curve"][m]
            - means[str(p)]["deterministic"]["target_hold"][m]
            for p in spec.source.PERIODS for m in spec.ERROR_METRICS}


def bootstrap(root_rows):
    values = np.asarray([[row["endpoints"][k] for k in spec.ENDPOINTS] for row in root_rows])
    indices = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED)).integers(
        0, len(values), (spec.BOOTSTRAP_DRAWS, len(values)))
    tail = .05 / (2 * len(spec.ENDPOINTS))
    bounds = np.quantile(values[indices].mean(axis=1), [tail, 1 - tail], axis=0)
    return {key: {"mean": float(values[:, i].mean()), "ci": bounds[:, i].tolist(),
        "effect": "positive" if bounds[0, i] > 0 else "negative" if bounds[1, i] < 0 else "inconclusive"}
        for i, key in enumerate(spec.ENDPOINTS)}


def diagnose(source_directory):
    source_directory = Path(source_directory)
    qualified = json.loads((source_directory / "qualification_summary.json").read_text())
    roots = spec.source.roots(preflight=False)
    if (qualified["status"] != "complete" or qualified["protocol"] != spec.source.EXPERIMENT_PROTOCOL
            or qualified["contract"] != spec.source.contract()
            or [row["root"] for row in qualified["root_rows"]] != list(roots)):
        raise ValueError("Stage53 requires the complete qualified Stage52 roster")
    root_rows, pooled, reads, regenerated, processed = [], {}, 0, 0, 0
    for root in roots:
        args = spec.source.arguments(root, preflight=False)
        seeds = spec.source.seed_roles(root, preflight=False)["evaluation"]
        cell = json.loads((source_directory / "cells" / f"replicate_{root}" / "result.json").read_text())
        if cell["status"] != "complete" or cell["root"] != root:
            raise ValueError("Stage53 source cell incomplete or mismatched")
        expected, partitions = {}, {}
        for seed in seeds:
            driver = PointMazeRegimeDriver(seed=seed, horizon=args.horizon,
                dt_seconds=spec.source.DT_SECONDS, **_task_options(args))
            expected[seed] = np.array([np.concatenate(driver.sample(t)) for t in range(args.horizon)])
            partitions[seed] = {p: event_partitions(expected[seed][:, :2], driver.regime_change_steps, p)
                                for p in spec.source.PERIODS}
            regenerated += 1
        means, root_panels = {}, {}
        for period in spec.source.PERIODS:
            means[str(period)] = {}
            for mode in spec.source.MODES:
                means[str(period)][mode] = {}
                for policy in spec.source.POLICIES:
                    rows = cell["evaluation_rows"][str(period)][policy][mode]
                    if [r["seed"] for r in rows] != seeds:
                        raise ValueError("Stage53 source path roster changed")
                    paths = []
                    for row in rows:
                        path = source_directory / "cells" / f"replicate_{root}_raw" / str(period) / policy / mode / f"episode_{row['seed']}.npz"
                        with np.load(path) as raw:
                            paths.append(decompose(raw, row, expected[row["seed"]]))
                        reads += 1
                        processed += len(paths[-1])
                    group_means, group_panels = reduce_paths(paths, [partitions[s][period] for s in seeds])
                    means[str(period)][mode][policy] = group_means
                    root_panels[(period, mode, policy)] = group_panels
                for name, labels in spec.PARTITIONS.items():
                    key = (period, mode, name)
                    buckets = pooled.setdefault(key, {label: {"steps": 0, "delta": np.zeros(len(spec.METRICS)),
                        "episode_denominator": 0} for label in labels})
                    curve, held = [root_panels[(period, mode, p)][name] for p in ("target_curve", "target_hold")]
                    for label, bucket in buckets.items():
                        if curve[label]["steps"] != held[label]["steps"]:
                            raise ValueError("paired event exposures differ")
                        bucket["steps"] += curve[label]["steps"]
                        bucket["delta"] += curve[label]["sums"] - held[label]["sums"]
                        bucket["episode_denominator"] += len(seeds)
        root_rows.append({"root": root, "means": means, "endpoints": contrasts(means)})
        print(f"diagnosed root{root}; raw_reads={reads}", flush=True)
    cost = {"raw_trace_reads": reads, "recorded_steps_processed": processed,
            "exogenous_driver_regenerations": regenerated, "new_native_steps": 0, "optimizer_steps": 0}
    if cost != spec.budget():
        raise ValueError("Stage53 offline accounting changed")
    strata = []
    for (period, mode, name), buckets in pooled.items():
        panel = {"period": period, "mode": mode, "partition": name, "buckets": {}}
        for label, bucket in buckets.items():
            n, seconds = bucket["episode_denominator"], bucket["steps"] * spec.source.DT_SECONDS
            panel["buckets"][label] = {"seconds_per_episode": seconds / n,
                "per_episode_delta": dict(zip(spec.METRICS, (bucket["delta"] / n).tolist())),
                "conditional_squared_error_delta": dict(zip(spec.ERROR_METRICS,
                    (bucket["delta"][:4] / seconds).tolist())) if seconds else None,
                "conditional_reward_delta": float(bucket["delta"][4] / bucket["steps"]) if seconds else None}
        strata.append(panel)
    return {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "root_rows": root_rows, "primary_endpoints": bootstrap(root_rows), "strata": strata, "cost": cost}
