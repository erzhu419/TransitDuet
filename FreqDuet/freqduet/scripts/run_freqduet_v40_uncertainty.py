#!/usr/bin/env python3
"""Causal online interval calibration without changing the frozen policy."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path

import pandas as pd

from freqduet_fleet_pool_forecast import freeze_pool, remaining_trip_s
from freqduet_return_uncertainty import ReturnIntervals, interval_summary
import run_freqduet_v39_fleet_shadow as shadow


class IntervalObserver(shadow.PoolObserver):
    def __init__(self, target, detailed):
        super().__init__(target, detailed)
        self.intervals = ReturnIntervals()
        self.periodic_snapshots = {}

    def attach(self, runner):
        super().attach(runner)
        original, cursor = runner.env.step, 0

        def step(action, *args, **kwargs):
            nonlocal cursor
            result = original(action, *args, **kwargs)
            now = float(runner.env.current_time)
            for event in self.terminals[cursor:]:
                self.intervals.resolve(event["trip_id"], event["time_s"], now)
            cursor = len(self.terminals)
            if now >= 1800 and now % 300 == 0 and not runner.env.done:
                snapshot = freeze_pool(runner, self.telemetry, None, None)
                for vehicle in snapshot["vehicles"]:
                    if vehicle["on_route"]:
                        predicted = now + remaining_trip_s(snapshot, vehicle)
                        self.intervals.issue(now, vehicle["trip_id"], predicted)
                if now in (7200, 21600, 36000):
                    self.periodic_snapshots[str(int(now))] = snapshot
            return result

        runner.env.step = step


def items():
    return list(itertools.product(shadow.joint.SEEDS, shadow.joint.SCENARIOS))


def run_item(arguments):
    root, v37, seed, scenario = arguments
    original, instance = shadow.observed.Observer, []

    class CaptureObserver(IntervalObserver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instance.append(self)

    shadow.observed.Observer = CaptureObserver
    try:
        bank = pd.read_csv(v37 / "records" / f"{shadow.joint.shard_name(seed, scenario, 7200, 0)}.csv")
        label = f"seed{seed}_eval{scenario}_reference"
        shadow.observed.replay(root, v37, seed, scenario, 7200, label, int(bank.first_actor_s.iloc[0]), None, True)
    finally:
        shadow.observed.Observer = original
    observer = instance[0]
    cell = root / "observed" / label
    pd.DataFrame(observer.intervals.rows).to_csv(cell / "interval_predictions.csv", index=False)
    (cell / "periodic_snapshots.json").write_text(json.dumps(observer.periodic_snapshots) + "\n")
    (cell / "interval_summary.json").write_text(json.dumps(interval_summary(observer.intervals.rows)) + "\n")
    return label


def batch(root, v37, start, end, workers):
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        complete = list(pool.map(run_item, [(root, v37, *item) for item in items()[start:end]]))
    export = root / "exports" / f"batch_{start}_{end}"
    export.mkdir(parents=True, exist_ok=True)
    (export / "interval_done.json").write_text(json.dumps(dict(completed=complete,
        source_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))) + "\n")


def aggregate(root):
    rows, summaries = [], []
    for seed, scenario in items():
        cell = root / "observed" / f"seed{seed}_eval{scenario}_reference"
        frame = pd.read_csv(cell / "interval_predictions.csv")
        if not (frame.latest_completion_s.dropna() < frame.loc[frame.latest_completion_s.notna(), "origin_s"]).all():
            raise ValueError("interval calibration used a future or simultaneous completion")
        rows.extend(frame.where(pd.notna(frame), None).to_dict("records"))
        for entry in json.loads((cell / "interval_summary.json").read_text()):
            summaries.append(dict(train_seed=seed, eval_seed=scenario, **entry))
    # Read records directly so warmup None fields are not interpreted as labels.
    frame = pd.DataFrame(rows)
    output = []
    for band, group in frame.groupby("band"):
        resolved = group.loc[group.actual_return_s.notna()]
        calibrated = resolved.loc[resolved.available]
        output.append(dict(band=band, forecasts=len(resolved), calibrated=len(calibrated),
                           coverage=float(calibrated.covered.astype(float).mean()),
                           mean_width_s=float(calibrated.width_s.mean()),
                           interval_score_s=float(calibrated.interval_score_s.mean()),
                           point_mae_s=float(resolved.residual_s.abs().mean())))
    summary = root / "uncertainty_summary"
    summary.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output).to_csv(summary / "intervals.csv", index=False)
    pd.DataFrame(summaries).to_csv(summary / "per_day.csv", index=False)
    status = dict(status="complete_prequential_intervals", matched_days=4, policy_changed=False,
                  calibration="one prediction per trip and predicted-lead band, last 128 completed residuals, min 20",
                  intervals="empirical nominal 90 percent; temporal dependence precludes a coverage guarantee",
                  future_label_checks_pass=True, independent_confirmation=False)
    (summary / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    print(pd.DataFrame(output).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["batch", "aggregate"])
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v37-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "batch":
        batch(args.out_root, args.v37_root, args.start, args.end, args.workers)
    else:
        aggregate(args.out_root)
