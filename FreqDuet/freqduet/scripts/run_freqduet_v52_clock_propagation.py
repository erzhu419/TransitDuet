#!/usr/bin/env python3
"""Frozen forecast replay with telescoping, identity-matched arrival error attribution."""

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
from freqduet_holding_distribution import DRAWS, METRICS
from freqduet_passenger_distribution import passenger_ensemble
from freqduet_travel_distribution import score_draws
from run_freqduet_v49_residual_dependence import cluster_summary
from run_freqduet_v50_conditioned_travel import forecast_ensemble
import run_freqduet_v46_travel_blocks as previous

MODES = ("passenger_joint", "conditioned_independent", "conditioned_shared")
COMPONENTS = ("initial_boundary", "launch", "travel", "service", "decision", "holding")
STAGES = {"departure": 4, "arrival": 1, "ready": 2, "action": 3}


def items():
    clocks = sorted([clock + offset for clock in previous.COMMON for offset in (-900, 0)])
    return [(s, d, c, clock) for s, d, c in previous.previous.items() for clock in clocks]


def component(previous_stage, stage):
    if stage == "arrival" and previous_stage == "departure":
        return "travel"
    if stage == "ready" and previous_stage == "arrival":
        return "service"
    if stage == "action" and previous_stage == "ready":
        return "decision"
    if stage == "departure" and previous_stage == "action":
        return "holding"
    if previous_stage == "launch" and stage == "departure":
        return "launch"
    raise ValueError(f"unsupported within-trip clock transition: {previous_stage}->{stage}")


def decompose(predicted, actual, observed, launches):
    """Conditional contributions telescope; capped scores below retain all draws."""
    now, end = observed["time_s"], observed["time_s"] + 1800
    identity = ["trip_id", "station_id", "stage"]
    if predicted.duplicated(identity).any() or actual.duplicated(identity).any():
        raise ValueError("duplicate trip/stop/stage clock")
    predictions = predicted.set_index(identity).time_s.to_dict()
    launch = launches.set_index("trip_id").time_s.to_dict()
    initial = {v["trip_id"] for v in observed["vehicles"] if v["on_route"]}
    rows = []
    future_events = actual.loc[(actual.time_s > now) & (actual.time_s <= end)]
    for trip, group in future_events.groupby("trip_id", sort=False):
        future = group.copy()
        route = observed["routes"][str(bool(group.direction.iloc[0]))]["station_ids"]
        future["order"] = [route.index(int(s)) * 4 + STAGES[k] for s, k in zip(future.station_id, future.stage)]
        future = future.sort_values("order")
        contribution = dict.fromkeys(COMPONENTS, 0.0)
        is_initial = trip in initial
        available = is_initial or (trip, route[0], "departure") in predictions
        previous_true = now if is_initial else launch[trip]
        previous_pred = now if is_initial else predictions.get((trip, route[0], "departure"), end)
        if not is_initial:
            contribution["launch"] = previous_pred - previous_true
        previous_stage, hops = ("origin" if is_initial else "departure"), 0
        # The first-stop departure already anchors newly launched trips.
        if not is_initial:
            future = future.loc[~((future.station_id == route[0]) & (future.stage == "departure"))]
        for row in future.itertuples():
            key = (trip, row.station_id, row.stage)
            available = available and key in predictions
            if available:
                value = predictions[key]
                name = "initial_boundary" if previous_stage == "origin" else component(previous_stage, row.stage)
                contribution[name] += (value - previous_pred) - (row.time_s - previous_true)
                previous_true, previous_pred, previous_stage = row.time_s, value, row.stage
            if row.stage != "arrival":
                continue
            hops += 1
            error = predictions.get(key, end) - row.time_s
            if available and not np.isclose(sum(contribution.values()), error, atol=1e-8):
                raise ValueError("clock contributions do not telescope")
            rows.append(dict(trip_id=trip, station_id=row.station_id, actual_s=row.time_s,
                trip_source="initial" if is_initial else "future_launch", hops=hops,
                lead="0-900" if row.time_s <= now + 900 else "900-1800",
                hour_crossed=(row.time_s // 3600 > now // 3600), available=key in predictions,
                complete_chain=available, error=error,
                **{name: contribution[name] if available else np.nan for name in COMPONENTS}))
    return rows


def arrival_scores(traces, actual, observed, launches):
    paths = pd.DataFrame([dict(draw=draw, **row) for draw, trace in enumerate(traces)
        for row in decompose(pd.DataFrame(trace), actual, observed, launches)])
    output = []
    for (trip, station), group in paths.groupby(["trip_id", "station_id"]):
        if len(group) != DRAWS or group.draw.nunique() != DRAWS:
            raise ValueError("capped arrival score lost a forecast draw")
        target = group.iloc[0]
        score = score_draws(group.error.to_numpy() + target.actual_s, target.actual_s)
        matched = group.loc[group.complete_chain]
        details = {}
        for name in COMPONENTS:
            details[name + "_bias"] = float(matched[name].mean())
            details[name + "_cross"] = float((matched[name] * matched.error).mean())
        output.append(dict(trip_id=trip, station_id=station, lead=target.lead,
            trip_source=target.trip_source, hops=int(target.hops), hour_crossed=bool(target.hour_crossed),
            available_paths=int(group.available.sum()), complete_paths=len(matched),
            matched_bias=float(matched.error.mean()), matched_mse=float((matched.error ** 2).mean()),
            **score, **details))
    return pd.DataFrame(output)


def run_origin(arguments):
    root, base, seed, scenario, cohort, clock = arguments
    label = f"seed{seed}_eval{scenario}"
    observed = json.loads((base / "protocol_v6_v45_completed_calibration" / "observed" / label /
        "forecasts.json").read_text())[f"clock_{clock}"]
    cell = base / "protocol_v6_v47_boundary_apc" / "observed" / label
    certified = json.loads((cell / "done.json").read_text())
    if not certified["matches_frozen_control"] or not certified["matches_v45_prefix_flows"]:
        raise ValueError("unmatched actual controller or prefix flows")
    launches = pd.read_csv(cell / "launches.csv")
    actual = boundary_events(pd.read_csv(cell / "actual_events.csv"), launches,
        pd.read_csv(cell / "terminals.csv"), observed["pool"]["routes"])
    old = json.loads((base / "protocol_v6_v48_passenger_uncertainty" / "origins" / label /
        f"clock_{clock}" / "forecasts.json").read_text())
    new = json.loads((base / "protocol_v6_v50_conditioned_travel" / "origins" / label /
        f"clock_{clock}" / "forecasts.json").read_text())
    frames = []
    for mode in MODES:
        traces = [[] for _ in range(DRAWS)]
        sink = lambda draw, row: traces[draw].append(row)
        if mode == "passenger_joint":
            result = passenger_ensemble(observed["pool"], 1800, observed["distribution_model"],
                old["travel_model"], old["passenger_model"], mode, event_sink=sink)
            reference = old["forecasts"][mode]["1800"]
        else:
            result = forecast_ensemble(observed["pool"], 1800, observed["distribution_model"],
                new["travel_model"], old["passenger_model"], mode, event_sink=sink)
            reference = new["forecasts"][mode]["1800"]
        for metric in METRICS:
            np.testing.assert_allclose(result["draws"][metric], reference["draws"][metric], rtol=1e-12, atol=1e-8)
        frames.append(arrival_scores(traces, actual, observed["pool"], launches).assign(mode=mode))
    out = root / "origins" / label / f"clock_{clock}"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out / "arrivals.csv", index=False)
    (out / "done.json").write_text(json.dumps(dict(exact_replay=True, policy_changed=False,
        source_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))) + "\n")
    print(f"CLOCK_AUDIT_DONE {label} clock={clock} exact_64_draw_replay=True", flush=True)
    return str(out.relative_to(root))


def aggregate(root):
    frames = []
    for seed, scenario, cohort, clock in items():
        frames.append(pd.read_csv(root / "origins" / f"seed{seed}_eval{scenario}" / f"clock_{clock}" /
            "arrivals.csv").assign(train_seed=seed, eval_seed=scenario, cohort=cohort, clock=clock))
    frame = pd.concat(frames, ignore_index=True)
    frame["mae"] = frame.error.abs()
    frame["mse"] = frame.error ** 2
    columns = ["mae", "mse", "error", "crps", "covered", "width", "spread", "available_paths", "complete_paths",
        "matched_bias", "matched_mse", *[name + suffix for name in COMPONENTS for suffix in ("_bias", "_cross")]]
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    summary = frame.groupby(["cohort", "mode", "lead", "trip_source"])[columns].mean()
    summary.to_csv(out / "clock_components.csv")
    frame.groupby(["mode", "lead", "hour_crossed"])[columns].mean().to_csv(out / "hour_boundary.csv")
    frame.groupby(["mode", "lead", "hops"])[columns].mean().to_csv(out / "hops.csv")
    matched = frame.loc[frame.complete_paths > 0]
    means = matched.groupby(["mode", "lead"])[["matched_bias", *[c + "_bias" for c in COMPONENTS]]].mean()
    if not np.allclose(means[[c + "_bias" for c in COMPONENTS]].sum(axis=1), means.matched_bias, atol=1e-8):
        raise ValueError("summary changed the matched-chain attribution population")
    means.to_csv(out / "matched_bias_components.csv")
    day = frame.groupby(["train_seed", "eval_seed", "cohort", "mode", "lead"])[["mae", "covered"]].mean().reset_index()
    old = day.loc[day["mode"] == "passenger_joint", ["train_seed", "eval_seed", "lead", "mae", "covered"]]
    paired = day.merge(old, on=["train_seed", "eval_seed", "lead"], suffixes=("", "_v48"), validate="many_to_one")
    paired["mae_delta"], paired["coverage_delta"] = paired.mae - paired.mae_v48, paired.covered - paired.covered_v48
    result = cluster_summary(paired, ["mode", "lead"], ["mae_delta", "coverage_delta"])
    result.to_csv(out / "paired_clusters.csv", index=False)
    (out / "status.json").write_text(json.dumps(dict(origins=len(items()), draws=DRAWS, modes=MODES,
        policy_changed=False, exact_replay=True, all_demand_seeds_previously_examined=True)) + "\n")
    print(summary[["mae", "error", "covered", "available_paths", "complete_paths"]].to_string(), flush=True)
    print(means.to_string(), flush=True)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "batch", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--data-base", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=72)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "aggregate":
        aggregate(args.out_root)
    else:
        if args.command == "smoke":
            suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1] / "tests"), pattern="test_v52*.py")
            if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
                raise SystemExit(1)
            run_origin((args.out_root, args.data_base, *items()[0]))
            output = args.out_root / "smoke_export"
        else:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
                list(pool.map(run_origin, [(args.out_root, args.data_base, *item) for item in items()[args.start:args.end]]))
            output = args.out_root / "exports" / f"batch_{args.start}_{args.end}"
        output.mkdir(parents=True, exist_ok=True)
        (output / "done.json").write_text('{"clock_propagation_complete": true}\n')
