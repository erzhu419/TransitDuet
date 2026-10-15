#!/usr/bin/env python3
"""Within-hour/stop permutations distinguish dependence from centering artifacts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from freqduet_residual_dependence import START, STOP, describe_moments, moments
from run_freqduet_v49_residual_dependence import items, cluster_summary


def permutation_audit(travel, holding):
    frames = dict(travel=travel, holding=holding)
    specs, strata, window_indices = {}, {}, {}
    for component, frame in frames.items():
        specs[component] = {}
        for keys, name in ((["bus_id"], "same_bus"), (["direction", "station_id"], "same_directed_stop")):
            left, right = [], []
            for indices in frame.groupby(keys).indices.values():
                left.extend(indices[:-1])
                right.extend(indices[1:])
            a, b = np.array(left, int), np.array(right, int)
            if name == "same_bus":
                same = frame.trip_id.to_numpy()[a] == frame.trip_id.to_numpy()[b]
                specs[component]["same_bus_same_trip"] = (a[same], b[same])
                specs[component]["same_bus_other_trip"] = (a[~same], b[~same])
            else:
                different = frame.bus_id.to_numpy()[a] != frame.bus_id.to_numpy()[b]
                specs[component]["same_directed_stop_different_bus"] = (a[different], b[different])
        strata[component] = list(frame.groupby(["direction", "station_id", "hour"]).indices.values())
        window_indices[component] = {}
        for horizon in (900, 1800):
            edges = np.arange(START, STOP + 1, horizon)
            indices = np.searchsorted(edges, frame.time_s, side="left") - 1
            valid = (indices >= 0) & (indices < len(edges) - 1)
            window_indices[component][horizon] = (indices, valid, len(edges) - 1)
    keys = ["bus_id", "trip_id", "direction", "station_id"]
    incoming = travel.rename(columns={"station_id": "departure_station_id", "arrival_station_id": "station_id"})
    joined = holding.reset_index(names="holding_index").merge(incoming.reset_index(names="travel_index"),
        on=keys, validate="one_to_one", suffixes=("_holding", "_travel"))
    ti, hi = joined.travel_index.to_numpy(), joined.holding_index.to_numpy()

    def scores(values):
        result = {}
        for component, transforms in values.items():
            for transform, v in transforms.items():
                for relation, (a, b) in specs[component].items():
                    if len(a):
                        result[(component, relation, transform, "correlation")] = describe_moments(moments(v[a], v[b]))["correlation"]
                for horizon, (indices, valid, n) in window_indices[component].items():
                    sums = np.bincount(indices[valid], weights=v[valid], minlength=n)
                    result[(component, f"window_{horizon}s_lag1", transform, "correlation")] = describe_moments(moments(sums[:-1], sums[1:]))["correlation"]
                    result[(component, f"window_{horizon}s", transform, "squared_sum")] = float(np.mean(sums ** 2))
        if len(ti):
            for transform in ("residual", "centered"):
                result[("incoming_cross", "completed_travel_to_holding", transform, "correlation")] = describe_moments(
                    moments(values["travel"][transform][ti], values["holding"][transform][hi]))["correlation"]
        return result

    actual = {component: {key: frame[key].to_numpy() for key in ("residual", "centered")} for component, frame in frames.items()}
    observed = scores(actual)
    null = {key: [] for key in observed}
    rng = np.random.default_rng(20261021)
    for _ in range(200):
        shuffled = {}
        for component in frames:
            raw = actual[component]["residual"]
            means = raw - actual[component]["centered"]
            permuted = raw.copy()
            for indices in strata[component]:
                permuted[indices] = raw[rng.permutation(indices)]
            shuffled[component] = dict(residual=permuted, centered=permuted - means)
        for key, value in scores(shuffled).items():
            null[key].append(value)
    rows = []
    for key, value in observed.items():
        reference = np.asarray(null[key])
        reference = reference[np.isfinite(reference)]
        if not len(reference) or not np.isfinite(value):
            continue
        rows.append(dict(zip(["component", "relation", "transform", "measure"], key), observed=value,
            permutation_mean=float(reference.mean()), permutation_low=float(np.quantile(reference, .025)),
            permutation_high=float(np.quantile(reference, .975)), excess=value - reference.mean(), permutations=len(reference)))
    return pd.DataFrame(rows)


def run_day(arguments):
    root, seed, scenario, cohort = arguments
    cell = root / "days" / f"seed{seed}_eval{scenario}"
    permutation_audit(pd.read_csv(cell / "travel_records.csv"), pd.read_csv(cell / "holding_records.csv")).to_csv(
        cell / "centering_null.csv", index=False)
    print(f"CENTERING_NULL_DONE seed={seed} scenario={scenario}", flush=True)
    return f"seed{seed}_eval{scenario}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "batch", "aggregate"])
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "aggregate":
        parts = [pd.read_csv(args.out_root / "days" / f"seed{s}_eval{d}" / "centering_null.csv").assign(
            train_seed=s, eval_seed=d, cohort=c) for s, d, c in items()]
        frame = pd.concat(parts, ignore_index=True)
        out = args.out_root / "summary"
        frame.to_csv(out / "centering_null_days.csv", index=False)
        cluster_summary(frame.rename(columns={"measure": "statistic"}),
            ["component", "relation", "transform", "statistic"],
            ["observed", "permutation_mean", "excess"]).to_csv(
            out / "centering_null.csv", index=False)
        (out / "centering_null_status.json").write_text(json.dumps(dict(days=12, permutations_per_day=200,
            demand_clusters=6, policy_changed=False, source_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))) + "\n")
        print("CENTERING_NULL_AGGREGATED days=12 demand_clusters=6", flush=True)
    elif args.command == "batch":
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            completed = list(pool.map(run_day, [(args.out_root, *item) for item in items()[args.start:args.end]]))
        out = args.out_root / "null_exports" / f"batch_{args.start}_{args.end}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "done.json").write_text(json.dumps(dict(completed=completed)) + "\n")
    else:
        tests = Path(__file__).resolve().parents[1] / "tests"
        suite = unittest.defaultTestLoader.discover(str(tests), pattern="test_v49*.py")
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        run_day((args.out_root, *items()[0]))
        out = args.out_root / "null_smoke_export"
        out.mkdir(parents=True, exist_ok=True)
        (out / "done.json").write_text('{"centering_null_smoke": true}\n')
