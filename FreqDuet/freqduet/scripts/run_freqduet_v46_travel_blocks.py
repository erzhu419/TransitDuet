#!/usr/bin/env python3
"""V46: completed-leg travel distributions on certified V45 causal snapshots."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from freqduet_completed_window_calibration import CompletedWindows
from freqduet_holding_distribution import DRAWS, METRICS, ensemble
from freqduet_travel_distribution import MODES, completed_legs, fit_travel, score_draws, travel_ensemble
import run_freqduet_v45_completed_calibration as previous

CLOCKS = tuple(range(3600, 48601, 900))
COMMON = (7200, 21600, 36000)


def items():
    return [(seed, scenario, cohort, clock) for seed, scenario, cohort in previous.items() for clock in CLOCKS]


def state_scores(predicted, actual, now, mode):
    rows = []
    grouped = dict(tuple(predicted.groupby(["trip_id", "station_id", "stage"])))
    empty = predicted.iloc[:0]
    emitted = set(grouped)
    for truth in actual.itertuples():
        key = (truth.trip_id, truth.station_id, truth.stage)
        if not now < truth.time_s <= now + 1800 and key not in emitted:
            continue
        selected = grouped.get(key, empty)
        if selected.draw.duplicated().any():
            raise ValueError("duplicate predicted trip/stop/stage")
        # An absent event has not happened by the horizon. Score its capped
        # clock instead of silently discarding slow/censored trajectories.
        clocks = np.full(DRAWS, now + 1800.0)
        clocks[selected.draw.to_numpy(dtype=int)] = np.clip(selected.time_s, now, now + 1800)
        target = float(np.clip(truth.time_s, now, now + 1800))
        lead = ("previously_completed" if truth.time_s <= now else "later_than_horizon" if truth.time_s > now + 1800
                else "0-900" if truth.time_s <= now + 900 else "900-1800")
        rows.append(dict(mode=mode, metric=truth.stage + "_capped_clock_s",
                         lead=lead, paths=len(selected), **score_draws(clocks, target)))
        if truth.stage == "action" and now < truth.time_s <= now + 1800:
            for metric in ("forward_gap_s", "follower_gap_s", "action_s", "load"):
                value = getattr(truth, metric)
                values = selected[metric].dropna().to_numpy()
                if pd.notna(value) and len(values):
                    rows.append(dict(mode=mode, metric=metric, lead="all", paths=len(values),
                                     **score_draws(values, value)))
    return rows


def run_origin(arguments):
    root, bank, seed, scenario, cohort, clock = arguments
    cell = bank / "observed" / f"seed{seed}_eval{scenario}"
    control = json.loads((cell / "done.json").read_text())
    if not control["matches_frozen_control"]:
        raise ValueError("input snapshot lacks a matched frozen deployment")
    capture = json.loads((cell / "forecasts.json").read_text())[f"clock_{clock}"]
    observed, holding = capture["pool"], capture["distribution_model"]
    if holding["latest_time_s"] >= clock:
        raise ValueError("noncausal holding model in input bank")
    actual = pd.read_csv(cell / "actual_events.csv")
    model = fit_travel(completed_legs(actual, observed["routes"]), observed)
    results, states = {}, []
    for mode in MODES:
        traces = []
        results[mode] = {str(horizon): travel_ensemble(observed, horizon, holding, model, mode,
            event_sink=(lambda draw, row: traces.append(dict(draw=draw, **row)))
                if horizon == 1800 and clock in COMMON else None) for horizon in (900, 1800)}
        if traces:
            states.extend(state_scores(pd.DataFrame(traces), actual, clock, mode))
    output = root / "origins" / f"seed{seed}_eval{scenario}" / f"clock_{clock}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "forecasts.json").write_text(json.dumps(dict(cohort=cohort, travel_model=model, forecasts=results)) + "\n")
    if states:
        pd.DataFrame(states).to_csv(output / "state_scores.csv", index=False)
    (output / "done.json").write_text(json.dumps(dict(samples=model["samples"],
        latest_completion_s=model["latest_completion_s"], control_source=control["control_source"],
        source_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))) + "\n")
    print(f"TRAVEL_ORIGIN_DONE seed={seed} scenario={scenario} clock={clock} completed_legs={model['samples']}", flush=True)
    return f"seed{seed}_eval{scenario}/clock_{clock}"


def batch(root, bank, start, end, workers):
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        completed = list(pool.map(run_origin, [(root, bank, *item) for item in items()[start:end]]))
    output = root / "exports" / f"batch_{start}_{end}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "done.json").write_text(json.dumps(dict(completed=completed)) + "\n")


def aggregate(root, bank):
    rows, states, supports = [], [], []
    for seed, scenario, cohort in previous.items():
        cell = bank / "observed" / f"seed{seed}_eval{scenario}"
        captures = json.loads((cell / "forecasts.json").read_text())
        holds = pd.read_csv(cell / "holds.csv")
        timeline = np.load(cell / "exposure.npz")["timeline"]
        calibrations = {mode: CompletedWindows() for mode in ("legacy", *MODES)}
        for clock in CLOCKS:
            capture = captures[f"clock_{clock}"]
            origin = root / "origins" / f"seed{seed}_eval{scenario}" / f"clock_{clock}"
            values = json.loads((origin / "forecasts.json").read_text())
            model = values["travel_model"]
            if model["latest_completion_s"] is not None and model["latest_completion_s"] >= clock:
                raise ValueError("unfinished leg used as a travel label")
            supports.append(dict(train_seed=seed, eval_seed=scenario, clock=clock, samples=model["samples"],
                                 legs_without_history=sum(n == 0 for n in model["counts"].values())))
            forecasts = dict(legacy=capture["ensemble"], **values["forecasts"])
            truth_fn = lambda now, horizon: previous.outcomes(timeline, holds, now, horizon)
            for mode, horizons in forecasts.items():
                calibrator = calibrations[mode]
                calibrator.release(clock, truth_fn)
                for horizon in (900, 1800):
                    truth = truth_fn(clock, horizon)
                    draws = horizons[str(horizon)]["draws"]
                    corrected = calibrator.issue(clock, horizon, previous.observable_draws(draws))
                    for metric in METRICS:
                        identity = dict(train_seed=seed, eval_seed=scenario, cohort=cohort, clock=clock,
                            common_clock=clock in COMMON, horizon_s=horizon, metric=metric, actual=truth[metric])
                        score = score_draws(draws[metric], truth[metric])
                        rows.append(dict(**identity, mode=mode, mature=True, **score))
                        if metric in corrected:
                            value = corrected[metric]
                            lo, hi = value["low"], value["high"]
                            rows.append(dict(**identity, mode=mode + "_calibrated", mature=value["mature"],
                                error=value["point"] - truth[metric], crps=np.nan, spread=np.nan,
                                covered=float(lo <= truth[metric] <= hi) if lo is not None else np.nan,
                                width=hi - lo if lo is not None else np.nan))
                        historical = capture["forecasts"][f"historical_{horizon}"][metric]
                        if mode == "legacy":
                            rows.append(dict(**identity, mode="historical", mature=False,
                                error=historical - truth[metric], crps=np.nan, spread=np.nan, covered=np.nan, width=np.nan))
            if clock in COMMON:
                states.append(pd.read_csv(origin / "state_scores.csv").assign(
                    train_seed=seed, eval_seed=scenario, cohort=cohort, clock=clock))
    frame = pd.DataFrame(rows)
    frame["abs_error"] = frame.error.abs()
    summaries = []
    for scope, data in (("all_origins", frame), ("common_clocks", frame.loc[frame.common_clock])):
        summaries.append(data.groupby(["cohort", "mode", "horizon_s", "metric"]).agg(
            contexts=("error", "size"), mae=("abs_error", "mean"), bias=("error", "mean"),
            mean_actual=("actual", "mean"), crps=("crps", "mean"), coverage=("covered", "mean"),
            width=("width", "mean"), mature=("mature", "sum")).reset_index().assign(scope=scope))
    errors = pd.concat(summaries, ignore_index=True)
    state = pd.concat(states, ignore_index=True)
    state["abs_error"] = state.error.abs()
    state_summary = state.groupby(["cohort", "mode", "metric", "lead"]).agg(
        events=("error", "size"), mae=("abs_error", "mean"), bias=("error", "mean"),
        crps=("crps", "mean"), coverage=("covered", "mean"), width=("width", "mean"),
        mean_available_paths=("paths", "mean"))
    # Days, not the 51 overlapping within-day windows, are paired units here.
    paired = frame.loc[frame.horizon_s == 1800].groupby(
        ["train_seed", "eval_seed", "cohort", "mode", "metric"]).agg(mae=("abs_error", "mean"), bias=("error", "mean"))
    paired = paired.reset_index()
    legacy = paired.loc[paired["mode"] == "legacy", ["train_seed", "eval_seed", "metric", "mae"]].rename(columns={"mae": "legacy_mae"})
    paired = paired.merge(legacy, on=["train_seed", "eval_seed", "metric"], validate="many_to_one")
    paired["mae_delta_vs_legacy"] = paired.mae - paired.legacy_mae
    gates = {}
    for mode in MODES:
        gates[mode] = {}
        for cohort in errors.cohort.unique():
            data = errors.loc[(errors.scope == "all_origins") & (errors.horizon_s == 1800) & (errors.cohort == cohort)]
            candidate = data.loc[data["mode"] == mode].set_index("metric")
            old = data.loc[data["mode"] == "legacy"].set_index("metric")
            history = data.loc[data["mode"] == "historical"].set_index("metric")
            hold = candidate.loc[METRICS[-1]]
            gates[mode][cohort] = dict(
                holding_mae_no_worse_than_legacy=bool(hold.mae <= old.loc[METRICS[-1]].mae),
                holding_mae_no_worse_than_history=bool(hold.mae <= history.loc[METRICS[-1]].mae),
                holding_bias_le_5pct=bool(abs(hold.bias) <= .05 * hold.mean_actual),
                passenger_mae_within_5pct=bool(all(candidate.loc[m].mae <= 1.05 * history.loc[m].mae for m in METRICS[:2])),
                holding_coverage_ge_80pct=bool(hold.coverage >= .8))
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    errors.to_csv(output / "errors.csv", index=False)
    state_summary.to_csv(output / "state_errors.csv")
    paired.to_csv(output / "day_paired.csv", index=False)
    pd.DataFrame(supports).groupby("clock").agg(min_samples=("samples", "min"),
        max_legs_without_history=("legs_without_history", "max")).to_csv(output / "support.csv")
    frame.to_csv(root / "prediction_records.csv", index=False)
    status = dict(days=12, origins_per_day=51, origins=len(items()), modes=list(MODES), gates=gates,
        policy_changed=False, input_bank=bank.name, all_demand_seeds_previously_examined=True,
        waiting_calibrated=False, source_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"),
        scope="Engineering regression on certified snapshots; fresh demand and counterfactual validation remain required")
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    print(errors.loc[(errors.scope == "all_origins") & (errors.horizon_s == 1800) &
                     (errors.metric == METRICS[-1])].to_string(index=False), flush=True)


def smoke(root, bank):
    tests = Path(__file__).resolve().parents[1] / "tests"
    suite = unittest.defaultTestLoader.discover(str(tests), pattern="test_v4[12456]*.py")
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)
    seed, scenario, cohort = previous.items()[0]
    capture = json.loads((bank / "observed" / f"seed{seed}_eval{scenario}" / "forecasts.json").read_text())["clock_36000"]
    legacy = ensemble(capture["pool"], 1800, capture["distribution_model"])
    for metric in METRICS:
        np.testing.assert_allclose(legacy["draws"][metric], capture["ensemble"]["1800"]["draws"][metric], rtol=1e-12, atol=1e-8)
    run_origin((root, bank, seed, scenario, cohort, 36000))
    output = root / "exports"
    output.mkdir(parents=True, exist_ok=True)
    (output / "done.json").write_text('{"legacy_exact": true, "new_modes_complete": true}\n')
    print("TRAVEL_SMOKE_PASS legacy_exact=True", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "batch", "aggregate"])
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=612)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out_root, args.input_bank)
    elif args.command == "batch":
        batch(args.out_root, args.input_bank, args.start, args.end, args.workers)
    else:
        aggregate(args.out_root, args.input_bank)
