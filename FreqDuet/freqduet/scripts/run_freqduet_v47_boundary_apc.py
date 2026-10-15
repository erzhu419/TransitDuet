#!/usr/bin/env python3
"""V47 read-only boundary/APC acquisition, causal travel replay and flow audit."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from freqduet_boundary_apc import APCTrace, arrival_intervals, audit_flows, boundary_events
from freqduet_completed_window_calibration import CompletedWindows
from freqduet_fleet_pool_forecast import freeze_pool
from freqduet_holding_distribution import METRICS
from freqduet_travel_distribution import completed_legs, fit_travel, score_draws, travel_ensemble
import run_freqduet_v42_distribution as distribution
from run_freqduet_v44_segment_audit import SegmentObserver
import run_freqduet_v45_completed_calibration as calibration_protocol
import run_freqduet_v46_travel_blocks as previous


class TelemetryObserver(SegmentObserver):
    def __init__(self, *args):
        super().__init__(*args)
        self.flows = APCTrace()

    def capture_flow(self, runner, label, pool):
        if label.startswith("clock_") and label not in self.closed_snapshots:
            self.closed_snapshots[label] = dict(time_s=pool["time_s"], flows=self.flows.freeze(pool))

    def attach(self, runner):
        super().attach(runner)
        original, env = runner.env.step, runner.env

        def step(*args, **kwargs):
            result = original(*args, **kwargs)
            now = int(env.current_time)
            if now in previous.CLOCKS and f"clock_{now}" not in self.closed_snapshots:
                self.capture_flow(runner, f"clock_{now}", freeze_pool(runner, self.telemetry, None, None))
            return result

        env.step = step


def acquire(arguments):
    root, bank, seed, scenario, cohort = arguments
    original, captured = distribution.DistributionObserver, []

    class Capture(TelemetryObserver):
        def __init__(self, *args):
            super().__init__(*args)
            captured.append(self)

    distribution.DistributionObserver = Capture
    try:
        label = distribution.run_day((root, seed, scenario, cohort))
    finally:
        distribution.DistributionObserver = original
        for observer in captured:
            observer.close()
    observer, cell = captured[0], root / "observed" / label
    old = json.loads((bank / "observed" / label / "forecasts.json").read_text())
    if set(observer.closed_snapshots) != set(old):
        raise ValueError("missing acquisition origin")
    for clock, snapshot in observer.closed_snapshots.items():
        if snapshot["flows"] != old[clock]["pool"]["flows"]:
            raise ValueError(f"APC logger changed the prefix flow state at {clock}")
    for name, rows in (("launches", observer.launches), ("apc", observer.flows.rows), ("actual_events", observer.actual_events)):
        pd.DataFrame(rows).to_csv(cell / f"{name}.csv", index=False)
    routes = old["clock_7200"]["pool"]["routes"]
    events = boundary_events(pd.DataFrame(observer.actual_events), pd.DataFrame(observer.launches),
                             pd.DataFrame(observer.terminals), routes)
    legs = completed_legs(events, routes)
    legs.to_csv(cell / "completed_legs.csv", index=False)
    intervals = arrival_intervals(pd.DataFrame(observer.flows.rows))
    if (intervals.arrivals < 0).any():
        raise ValueError("APC mass balance contains negative inferred arrivals")
    support = fit_travel(legs, old["clock_36000"]["pool"])
    if any(count == 0 for count in support["counts"].values()):
        raise ValueError("terminal legs still lack completed samples")
    done = json.loads((cell / "done.json").read_text())
    done.update(matches_v45_prefix_flows=True, apc_events=len(observer.flows.rows),
                complete_leg_samples=len(legs), negative_arrival_intervals=0)
    (cell / "done.json").write_text(json.dumps(done) + "\n")
    print(f"BOUNDARY_APC_DAY_DONE {label} apc={len(observer.flows.rows)} all_42_legs_observed=True", flush=True)
    return label


def forecast(arguments):
    root, bank, seed, scenario, cohort, clock = arguments
    cell = root / "observed" / f"seed{seed}_eval{scenario}"
    done = json.loads((cell / "done.json").read_text())
    if not done.get("matches_v45_prefix_flows"):
        raise ValueError("acquisition/prefix verification is incomplete")
    capture = json.loads((bank / "observed" / cell.name / "forecasts.json").read_text())[f"clock_{clock}"]
    observed = capture["pool"]
    model = fit_travel(pd.read_csv(cell / "completed_legs.csv"), observed)
    traces = []
    results = {str(horizon): travel_ensemble(observed, horizon, capture["distribution_model"], model, "shared_block",
        event_sink=(lambda draw, row: traces.append(dict(draw=draw, **row)))
            if horizon == 1800 and clock in previous.COMMON else None) for horizon in (900, 1800)}
    output = root / "origins" / cell.name / f"clock_{clock}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "forecasts.json").write_text(json.dumps(dict(travel_model=model, forecasts=results)) + "\n")
    if traces:
        actual = boundary_events(pd.read_csv(cell / "actual_events.csv"), pd.read_csv(cell / "launches.csv"),
                                 pd.read_csv(cell / "terminals.csv"), observed["routes"])
        pd.DataFrame(previous.state_scores(pd.DataFrame(traces), actual, clock, "boundary_shared")).to_csv(
            output / "state_scores.csv", index=False)
    print(f"BOUNDARY_ORIGIN_DONE {cell.name} clock={clock} no_history={sum(n == 0 for n in model['counts'].values())}", flush=True)
    return str(output.relative_to(root))


def batch(root, bank, command, start, end, workers):
    function, entries = (acquire, calibration_protocol.items()) if command == "acquire" else (forecast, previous.items())
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        completed = list(pool.map(function, [(root, bank, *item) for item in entries[start:end]]))
    output = root / "exports" / f"{command}_{start}_{end}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "done.json").write_text(json.dumps(dict(completed=completed)) + "\n")


def aggregate(root, bank, v46):
    records, flow_rows, supports, state_rows = [], [], [], []
    for seed, scenario, cohort in calibration_protocol.items():
        label = f"seed{seed}_eval{scenario}"
        cell = root / "observed" / label
        captures = json.loads((bank / "observed" / label / "forecasts.json").read_text())
        holds = pd.read_csv(cell / "holds.csv")
        timeline = np.load(cell / "exposure.npz")["timeline"]
        apc = pd.read_csv(cell / "apc.csv")
        intervals = arrival_intervals(apc)
        calibration = CompletedWindows()
        for clock in previous.CLOCKS:
            capture = captures[f"clock_{clock}"]
            origin = root / "origins" / label / f"clock_{clock}"
            result = json.loads((origin / "forecasts.json").read_text())
            model = result["travel_model"]
            if model["latest_completion_s"] >= clock:
                raise ValueError("future completed leg in travel fit")
            supports.append(dict(train_seed=seed, eval_seed=scenario, clock=clock,
                                 no_history=sum(n == 0 for n in model["counts"].values())))
            flow_rows.extend(dict(cohort=cohort, train_seed=seed, eval_seed=scenario, clock=clock, **row)
                             for row in audit_flows(apc, intervals, capture["pool"]))
            truth_fn = lambda now, horizon: calibration_protocol.outcomes(timeline, holds, now, horizon)
            calibration.release(clock, truth_fn)
            for horizon in (900, 1800):
                truth = truth_fn(clock, horizon)
                draws = result["forecasts"][str(horizon)]["draws"]
                corrected = calibration.issue(clock, horizon, calibration_protocol.observable_draws(draws))
                for metric in METRICS:
                    identity = dict(cohort=cohort, train_seed=seed, eval_seed=scenario, clock=clock,
                                    common_clock=clock in previous.COMMON, horizon_s=horizon, metric=metric, actual=truth[metric])
                    records.append(dict(**identity, mode="boundary_shared", mature=True, **score_draws(draws[metric], truth[metric])))
                    if metric in corrected:
                        value = corrected[metric]
                        lo, hi = value["low"], value["high"]
                        records.append(dict(**identity, mode="boundary_shared_calibrated", mature=value["mature"],
                            error=value["point"] - truth[metric], crps=np.nan,
                            covered=float(lo <= truth[metric] <= hi) if lo is not None else np.nan,
                            width=hi - lo if lo is not None else np.nan))
            if clock in previous.COMMON:
                state_rows.append(pd.read_csv(origin / "state_scores.csv").assign(cohort=cohort))
    old = pd.read_csv(v46 / "prediction_records.csv")
    old = old.loc[old["mode"].isin(["historical", "legacy", "shared_block", "shared_block_calibrated"])]
    frame = pd.concat([old, pd.DataFrame(records)], ignore_index=True)
    frame["abs_error"] = frame.error.abs()
    tables = []
    for scope, data in (("all_origins", frame), ("common_clocks", frame.loc[frame.common_clock])):
        tables.append(data.groupby(["cohort", "mode", "horizon_s", "metric"]).agg(
            contexts=("error", "size"), mae=("abs_error", "mean"), bias=("error", "mean"), mean_actual=("actual", "mean"),
            crps=("crps", "mean"), coverage=("covered", "mean"), width=("width", "mean"), mature=("mature", "sum")).reset_index().assign(scope=scope))
    flow = pd.DataFrame(flow_rows)
    flow["abs_error"] = flow.error.abs()
    state = pd.concat(state_rows, ignore_index=True)
    state["abs_error"] = state.error.abs()
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    errors = pd.concat(tables, ignore_index=True)
    errors.to_csv(output / "errors.csv", index=False)
    flow.groupby(["cohort", "mode", "metric"]).agg(n=("error", "size"), mae=("abs_error", "mean"),
        bias=("error", "mean"), mean_actual=("actual", "mean")).to_csv(output / "apc_audit.csv")
    state.groupby(["cohort", "mode", "metric", "lead"]).agg(n=("error", "size"), mae=("abs_error", "mean"),
        crps=("crps", "mean"), coverage=("covered", "mean"), available_paths=("paths", "mean")).to_csv(output / "state_errors.csv")
    pd.DataFrame(supports).groupby("clock").agg(max_no_history=("no_history", "max")).to_csv(output / "support.csv")
    paired = frame.loc[frame.horizon_s == 1800].groupby(["train_seed", "eval_seed", "cohort", "mode", "metric"]).abs_error.mean().unstack("mode")
    paired["boundary_delta_vs_v46"] = paired.boundary_shared - paired.shared_block
    paired.to_csv(output / "day_paired.csv")
    frame.to_csv(root / "prediction_records.csv", index=False)
    flow.to_csv(root / "apc_audit_records.csv", index=False)
    status = dict(days=12, origins=612, policy_changed=False, prefix_flows_match_v45=True,
        future_apc_used_only_as_audit_labels=True, waiting_calibrated=False,
        source_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"), all_seeds_previously_examined=True)
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(errors.loc[(errors.scope == "all_origins") & (errors.horizon_s == 1800) &
                     (errors["mode"].isin(["boundary_shared", "shared_block", "historical"]))].to_string(index=False), flush=True)
    print(pd.read_csv(output / "apc_audit.csv").to_string(index=False), flush=True)
    print("BOUNDARY_APC_AGGREGATE_DONE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "acquire", "forecast", "aggregate"])
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--v46-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        tests = Path(__file__).resolve().parents[1] / "tests"
        suite = unittest.defaultTestLoader.discover(str(tests), pattern="test_v4[124567]*.py")
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        entry = calibration_protocol.items()[0]
        acquire((args.out_root, args.input_bank, *entry))
        forecast((args.out_root, args.input_bank, *entry, 36000))
        export = args.out_root / "exports"
        export.mkdir(parents=True, exist_ok=True)
        (export / "done.json").write_text('{"matched_control": true, "prefix_flows_exact": true}\n')
        print("BOUNDARY_APC_SMOKE_PASS", flush=True)
    elif args.command == "aggregate":
        aggregate(args.out_root, args.input_bank, args.v46_root)
    else:
        batch(args.out_root, args.input_bank, args.command, args.start, args.end, args.workers)
