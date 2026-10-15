#!/usr/bin/env python3
"""V49 read-only, prefix-issued residual dependence audit on frozen service days."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from freqduet_boundary_apc import boundary_events
from freqduet_residual_dependence import (identify_holding, identified_legs, holding_records,
    travel_records, pair_audit, window_audit, incoming_travel_holding)
from run_freqduet_v45_completed_calibration import items


def run_day(arguments):
    root, base, seed, scenario, cohort = arguments
    label = f"seed{seed}_eval{scenario}"
    bank = base / "protocol_v6_v45_completed_calibration" / "observed" / label
    telemetry = base / "protocol_v6_v47_boundary_apc" / "observed" / label
    for cell in (bank, telemetry):
        control = json.loads((cell / "done.json").read_text())
        if not control["matches_frozen_control"]:
            raise ValueError("audit requires certified unchanged actual policies")
    if not control["matches_v45_prefix_flows"]:
        raise ValueError("telemetry and snapshot service days differ")
    captures = json.loads((bank / "forecasts.json").read_text())
    routes = captures["clock_3600"]["pool"]["routes"]
    events = pd.read_csv(telemetry / "actual_events.csv")
    labels = identify_holding(json.loads((bank / "teacher.json").read_text()),
        pd.read_csv(bank / "holds.csv"), events, routes)
    complete = boundary_events(events, pd.read_csv(telemetry / "launches.csv"),
                               pd.read_csv(telemetry / "terminals.csv"), routes)
    legs, incomplete = identified_legs(complete, routes)
    models = {int(capture["pool"]["time_s"]): json.loads((base / "protocol_v6_v46_travel_blocks" /
        "origins" / label / name / "forecasts.json").read_text())["travel_model"]
        for name, capture in captures.items()}
    holding = holding_records(labels, captures)
    travel, deviations, lengths = travel_records(legs, models)
    output = root / "days" / label
    output.mkdir(parents=True, exist_ok=True)
    support = []
    for component, frame, dev, length in (("travel", travel, deviations, lengths), ("holding", holding, None, None)):
        frame.to_csv(output / f"{component}_records.csv", index=False)
        pair_audit(frame, dev, length).to_csv(output / f"{component}_pairs.csv", index=False)
        windows, temporal = window_audit(frame)
        windows.to_csv(output / f"{component}_windows.csv", index=False)
        temporal.to_csv(output / f"{component}_temporal.csv", index=False)
        hourly = frame.groupby(["direction", "station_id", "hour"]).agg(events=("residual", "size"),
            bias=("residual", "mean"), mse=("residual", lambda v: float(np.mean(v ** 2))),
            independent_variance=("variance", "mean"))
        hourly.to_csv(output / f"{component}_hourly.csv")
        support.append(dict(component=component, labels=len(frame), unsupported_labels=int((frame.support == 0).sum()),
            bias=frame.residual.mean(), mae=frame.residual.abs().mean(), mse=np.mean(frame.residual ** 2),
            mean_model_variance=frame.variance.mean(), distinct_origins=frame.origin.nunique(),
            hourly_strata=len(hourly), singleton_hourly_strata=int((hourly.events == 1).sum())))
    incoming_travel_holding(travel, holding).to_csv(output / "incoming_cross.csv", index=False)
    pd.DataFrame(support).to_csv(output / "support.csv", index=False)
    (output / "done.json").write_text(json.dumps(dict(train_seed=seed, eval_seed=scenario, cohort=cohort,
        teacher_labels=len(labels), identified_completed_legs=len(legs), incomplete_departed_legs=incomplete,
        policy_changed=False, source_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))) + "\n")
    print(f"RESIDUAL_DAY_DONE {label} travel={len(travel)} holding={len(holding)} incomplete={incomplete}", flush=True)
    return label


def batch(root, base, start, end, workers):
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        labels = list(pool.map(run_day, [(root, base, *item) for item in items()[start:end]]))
    output = root / "exports" / f"batch_{start}_{end}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "done.json").write_text(json.dumps(dict(completed=labels)) + "\n")


def cluster_summary(frame, keys, values):
    # Actor seeds share exogenous demand. Average actors first, then resample six demand clusters.
    clusters = frame.groupby(keys + ["eval_seed"])[values].mean().reset_index()
    rows = []
    rng = np.random.default_rng(20261020)
    for key, group in clusters.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        for value in values:
            data = group[value].dropna().to_numpy()
            if not len(data):
                continue
            boot = rng.choice(data, (2000, len(data)), replace=True).mean(axis=1)
            rows.append(dict(zip(keys, key), measure=value, demand_clusters=len(data),
                mean=float(data.mean()), low=float(np.quantile(boot, .025)), high=float(np.quantile(boot, .975))))
    return pd.DataFrame(rows)


def aggregate(root):
    tables = {}
    for seed, scenario, cohort in items():
        cell = root / "days" / f"seed{seed}_eval{scenario}"
        done = json.loads((cell / "done.json").read_text())
        if done["policy_changed"]:
            raise ValueError("residual audit changed a controller")
        for name in ("support", "travel_pairs", "holding_pairs", "travel_temporal", "holding_temporal", "incoming_cross"):
            tables.setdefault(name, []).append(pd.read_csv(cell / f"{name}.csv").assign(
                train_seed=seed, eval_seed=scenario, cohort=cohort))
        for component in ("travel", "holding"):
            windows = pd.read_csv(cell / f"{component}_windows.csv")
            values = ["squared_sum", "diagonal", "same_bus_cross", "same_stop_other_bus_cross",
                "same_direction_other_cross", "opposite_direction_other_cross", "independent_variance"]
            day = windows.groupby(["horizon_s", "transform"])[values].mean().reset_index()
            day["mse_to_independent_variance"] = day.squared_sum / day.independent_variance
            tables.setdefault(component + "_window_parts", []).append(day.assign(
                train_seed=seed, eval_seed=scenario, cohort=cohort))
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    for name, parts in tables.items():
        frame = pd.concat(parts, ignore_index=True)
        frame.to_csv(output / (name + "_days.csv"), index=False)
        if name.endswith("pairs"):
            keys, values = ["relation", "lag", "transform", "pair_scope"], ["correlation", "covariance", "same_origin_model_covariance"]
        elif name.endswith("temporal"):
            keys, values = ["horizon_s", "lag", "transform"], ["correlation", "covariance"]
        elif name.endswith("window_parts"):
            keys, values = ["horizon_s", "transform"], ["squared_sum", "diagonal", "same_bus_cross",
                "same_stop_other_bus_cross", "same_direction_other_cross", "opposite_direction_other_cross",
                "independent_variance", "mse_to_independent_variance"]
        elif name == "incoming_cross":
            keys, values = ["transform"], ["correlation", "covariance"]
        else:
            keys, values = ["component"], ["bias", "mae", "mse", "mean_model_variance"]
        cluster_summary(frame, keys, values).to_csv(output / (name + ".csv"), index=False)
    status = dict(days=12, actor_seeds=2, demand_clusters=6, policy_changed=False,
        strict_prefix_issued=True, teacher_states_for_residual_labels_only=True,
        centering="retrospective day/direction/stop/hour means for diagnosis only",
        comparisons="observed pair moments; V46 shared-block covariance only for same-origin pairs",
        source_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"),
        scope="Descriptive reused-demand audit, not causal or counterfactual control validation")
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "batch", "aggregate"])
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--data-base", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "aggregate":
        aggregate(args.out_root)
    elif args.command == "batch":
        batch(args.out_root, args.data_base, args.start, args.end, args.workers)
    else:
        tests = Path(__file__).resolve().parents[1] / "tests"
        suite = unittest.defaultTestLoader.discover(str(tests), pattern="test_v49*.py")
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        run_day((args.out_root, args.data_base, *items()[0]))
        output = args.out_root / "exports"
        output.mkdir(parents=True, exist_ok=True)
        (output / "done.json").write_text('{"prefix_identity_smoke": true}\n')
