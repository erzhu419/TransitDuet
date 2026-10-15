#!/usr/bin/env python3
"""Online completed-window endpoint calibration, without controller overrides."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from freqduet_completed_window_calibration import CompletedWindows
from freqduet_fleet_pool_forecast import freeze_pool
from freqduet_holding_distribution import METRICS
import run_freqduet_v42_distribution as distribution
from run_freqduet_v44_segment_audit import SegmentObserver


def items():
    return distribution.items() + [(seed, scenario, "fresh_demand")
        for seed, scenario in itertools.product(distribution.joint.SEEDS, (19511021, 19511027))]


def outcomes(timeline, holds, origin, horizon):
    actual = distribution.previous.at(timeline, origin + horizon) - distribution.previous.at(timeline, origin)
    holding = np.maximum(0, np.minimum(origin + horizon - 1, holds.time_s + holds.action_s)
                         - np.maximum(origin - 1, holds.time_s)).sum()
    return dict(zip(METRICS, [*actual, float(holding)]))


class CalibrationObserver(SegmentObserver):
    def __init__(self, *args):
        super().__init__(*args)
        self.calibration = CompletedWindows()

    def capture_flow(self, runner, label, pool):
        if not label.startswith("clock_") or label in self.closed_snapshots:
            return
        now = pool["time_s"]
        timeline, holds = np.asarray(self.timeline), pd.DataFrame(self.holds)
        self.calibration.release(now, lambda origin, horizon: outcomes(timeline, holds, origin, horizon))
        super().capture_flow(runner, label, pool)
        capture = self.closed_snapshots[label]
        capture["calibrated"] = {str(horizon): self.calibration.issue(now, horizon,
            capture["ensemble"][str(horizon)]["draws"]) for horizon in (900, 1800)}

    def forecast_event_sink(self, label, horizon):
        return super().forecast_event_sink(label, horizon) if label in ("clock_7200", "clock_21600", "clock_36000") else None

    def attach(self, runner):
        super().attach(runner)
        env, original = runner.env, runner.env.step

        def step(*args, **kwargs):
            result = original(*args, **kwargs)
            now = int(env.current_time)
            if 3600 <= now <= 48600 and now % 900 == 0 and f"clock_{now}" not in self.closed_snapshots:
                self.capture_flow(runner, f"clock_{now}", freeze_pool(runner, self.telemetry, None, None))
            return result

        env.step = step


def run_day(arguments):
    original, captured = distribution.DistributionObserver, []

    class Capture(CalibrationObserver):
        def __init__(self, *args):
            super().__init__(*args)
            captured.append(self)

    distribution.DistributionObserver = Capture
    try:
        label = distribution.run_day(arguments)
    finally:
        distribution.DistributionObserver = original
        for observer in captured:
            observer.close()
    root = arguments[0]
    cell = root / "observed" / label
    for name in ("actual_events", "forecast_events", "requested_actions"):
        pd.DataFrame(getattr(captured[0], name)).to_csv(cell / f"{name}.csv", index=False)
    print(f"CALIBRATION_DAY_DONE {label} admitted={captured[0].calibration.admitted}", flush=True)
    return label


def batch(root, start, end, workers):
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        completed = list(pool.map(run_day, [(root, *item) for item in items()[start:end]]))
    export = root / "exports" / f"batch_{start}_{end}"
    export.mkdir(parents=True, exist_ok=True)
    (export / "done.json").write_text(json.dumps(dict(completed=completed)) + "\n")


def aggregate(root):
    rows, controls = [], []
    for seed, scenario, cohort in items():
        cell = root / "observed" / f"seed{seed}_eval{scenario}"
        done = json.loads((cell / "done.json").read_text())
        if not done["matches_frozen_control"]:
            raise ValueError("calibration observer changed the actual policy")
        controls.append(done["control_source"])
        captures = json.loads((cell / "forecasts.json").read_text())
        if len(captures) != 51:
            raise ValueError("missing periodic forecast windows")
        timeline = np.load(cell / "exposure.npz")["timeline"]
        holds = pd.read_csv(cell / "holds.csv")
        for origin, capture in captures.items():
            now = capture["pool"]["time_s"]
            for horizon in (900, 1800):
                truth = outcomes(timeline, holds, now, horizon)
                for metric in METRICS:
                    corrected = capture["calibrated"][str(horizon)][metric]
                    if corrected["latest_completion"] is not None and corrected["latest_completion"] >= now:
                        raise ValueError("partial-window calibration leakage")
                    raw_draws = capture["ensemble"][str(horizon)]["draws"][metric]
                    for mode in ("raw", "calibrated", "historical"):
                        if mode == "raw":
                            point, (lo, hi) = float(np.mean(raw_draws)), np.quantile(raw_draws, [.05, .95])
                        elif mode == "calibrated":
                            point, lo, hi = corrected["point"], corrected["low"], corrected["high"]
                        else:
                            point = capture["forecasts"][f"historical_{horizon}"][metric]
                            lo, hi = None, None
                        rows.append(dict(cohort=cohort, train_seed=seed, eval_seed=scenario, origin=origin,
                            common_clock=now in (7200, 21600, 36000), horizon_s=horizon, metric=metric, mode=mode,
                            actual=truth[metric], point=point, error=point - truth[metric],
                            mature=corrected["mature"], samples=corrected["samples"],
                            covered=float(lo <= truth[metric] <= hi) if lo is not None else np.nan,
                            width=hi - lo if lo is not None else np.nan))
    frame = pd.DataFrame(rows)
    archive = pd.read_csv(root.parent / "protocol_v6_v43_state_contract" / "prediction_records.csv")
    archive = archive.loc[archive["mode"] == "categorical"]
    raw = frame.loc[frame.common_clock & (frame["mode"] == "raw") & (frame.cohort != "fresh_demand")]
    keys = ["cohort", "train_seed", "eval_seed", "origin", "horizon_s", "metric"]
    matched = raw.merge(archive, on=keys, suffixes=("_new", "_v43"), validate="one_to_one")
    if len(matched) != len(raw):
        raise ValueError("missing V43 raw common-clock comparator")
    np.testing.assert_allclose(matched.point, matched.predicted, rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(matched.actual_new, matched.actual_v43, rtol=1e-12, atol=1e-8)
    frame["abs_error"] = frame.error.abs()
    result = []
    for scope, subset in (("all_origins", frame), ("common_clocks", frame.loc[frame.common_clock])):
        group = subset.groupby(["cohort", "mode", "horizon_s", "metric"]).agg(
            contexts=("error", "size"), mae=("abs_error", "mean"), bias=("error", "mean"),
            mean_actual=("actual", "mean"), mature=("mature", "sum"),
            coverage=("covered", "mean"), width=("width", "mean"))
        result.append(group.reset_index().assign(scope=scope))
    errors = pd.concat(result, ignore_index=True)
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    errors.to_csv(out / "errors.csv", index=False)
    frame.to_csv(root / "calibration_records.csv", index=False)
    gates = {}
    for cohort in ("development", "new_demand", "fresh_demand"):
        common = errors.loc[(errors.cohort == cohort) & (errors.scope == "common_clocks") & (errors.horizon_s == 1800)]
        cat = common.loc[common["mode"] == "calibrated"].set_index("metric")
        base = common.loc[common["mode"] == "historical"].set_index("metric")
        hold = cat.loc["holding_vehicle_seconds"]
        gates[cohort] = dict(holding_mae_no_worse=bool(hold.mae <= base.loc["holding_vehicle_seconds"].mae),
            holding_bias_le_5pct=bool(abs(hold.bias) <= .05 * hold.mean_actual),
            passenger_mae_within_5pct=bool(all(cat.loc[m].mae <= 1.05 * base.loc[m].mae for m in METRICS[:2])),
            mature_holding_coverage_ge_80pct=bool(hold.coverage >= .8))
    status = dict(experiment=root.name, days=12, forecast_origins_per_day=51,
        policy_changed=False, matches_frozen_controls=True, raw_common_clocks_match_v43=True,
        archived_controls=controls.count("v35_archive"),
        new_unobserved_controls=controls.count("same_seed_unobserved_replay"), gates=gates,
        forecast_screen_pass=all(all(g.values()) for g in gates.values()),
        source_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"),
        scope="Endpoint calibration only; not a validation of counterfactual action response")
    (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    print(errors.loc[(errors.scope == "common_clocks") & (errors.horizon_s == 1800)].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "batch", "aggregate"])
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1] / "tests"), pattern="test_v4[1245]*.py")
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        run_day((args.out_root, *items()[0]))
        (args.out_root / "done.json").write_text('{"matched_control": true}\n')
    elif args.command == "batch":
        batch(args.out_root, args.start, args.end, args.workers)
    else:
        aggregate(args.out_root)
