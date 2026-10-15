#!/usr/bin/env python3
"""Frozen-policy categorical holding forecast: development and new-demand audit."""

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

from freqduet_closed_loop_forecast import ACTIONS, holding_features
from freqduet_holding_distribution import DRAWS, METRICS, ensemble, fit_distribution, probabilities_from_features
import run_freqduet_v41_closed_loop as previous

joint = previous.uncertainty.shadow.joint


def items():
    return [(seed, scenario, cohort) for cohort, scenarios in
            (("development", (45007, 934011)), ("new_demand", (45013, 934029)))
            for seed, scenario in itertools.product(joint.SEEDS, scenarios)]


class DistributionObserver(previous.ClosedLoopObserver):
    def __init__(self, *args):
        super().__init__(*args)
        self.teacher = []

    def capture_flow(self, runner, label, pool):
        if not label.startswith("clock_"):
            return
        super().capture_flow(runner, label, pool)
        model = fit_distribution(self.holding_history.rows, pool["time_s"])
        capture = self.closed_snapshots[label]
        capture["distribution_model"] = model
        capture["ensemble"] = {str(horizon): ensemble(pool, horizon, model) for horizon in (900, 1800)}
        print(f"DISTRIBUTION_ORIGIN {label} samples={model['samples']} draws={DRAWS}", flush=True)

    def observe(self, runner, obs, key, previous_action, action):
        super().observe(runner, obs, key, previous_action, action)
        values = previous.holding_observation(runner, key, previous_action)
        if values is not None:
            self.teacher.append(dict(time_s=values["time_s"], features=holding_features(values).tolist(),
                                     action_s=float(np.asarray(action).item())))


def run_day(arguments):
    root, seed, scenario, cohort = arguments
    observer = DistributionObserver(7200, True)
    original = joint.optimized_runtime

    def runtime(out_root):
        matrix, Runner, config, torch = original(out_root)

        class ObservedRunner(Runner):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                observer.attach(self)

            def _lower_action_for_agent(self, obs, key, last_action=0.0, deterministic=False):
                action = super()._lower_action_for_agent(obs, key, last_action=last_action, deterministic=deterministic)
                observer.observe(self, obs, key, last_action, action)
                return action

        return matrix, ObservedRunner, config, torch

    joint.optimized_runtime = runtime
    label = f"seed{seed}_eval{scenario}"
    try:
        branch = joint.run_branch(root, seed, scenario, 7200, label)
    finally:
        joint.optimized_runtime = original
    archive = joint.TRAIN_ROOT.parent / "protocol_v6_v35_actor_cross_b200/cells" / (
        f"{joint.CURRENT}_seed{seed}_U200_L200/evaluation/evaluation.csv")
    old = pd.read_csv(archive).set_index("eval_seed").loc[scenario].to_dict()
    old["eval_seed"] = scenario
    joint.check_unchanged(branch["episode"], old)
    cell = root / "observed" / label
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "forecasts.json").write_text(json.dumps(observer.closed_snapshots) + "\n")
    (cell / "teacher.json").write_text(json.dumps(observer.teacher) + "\n")
    for name in ("holds", "terminals"):
        pd.DataFrame(getattr(observer, name)).to_csv(cell / f"{name}.csv", index=False)
    np.savez_compressed(cell / "exposure.npz", timeline=np.asarray(observer.timeline))
    (cell / "done.json").write_text(json.dumps(dict(cohort=cohort, matches_v35=True,
        source_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))) + "\n")
    print(f"DISTRIBUTION_DAY_DONE {label} cohort={cohort} matches_v35=True", flush=True)
    return label


def batch(root, start, end, workers):
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        completed = list(pool.map(run_day, [(root, *item) for item in items()[start:end]]))
    export = root / "exports" / f"batch_{start}_{end}"
    export.mkdir(parents=True, exist_ok=True)
    (export / "done.json").write_text(json.dumps(dict(completed=completed)) + "\n")


def aggregate(root):
    rows, teacher_rows, teacher_totals, return_rows = [], [], [], []
    for seed, scenario, cohort in items():
        cell = root / "observed" / f"seed{seed}_eval{scenario}"
        if not json.loads((cell / "done.json").read_text())["matches_v35"]:
            raise ValueError("forecast perturbed the frozen policy trajectory")
        snapshots = json.loads((cell / "forecasts.json").read_text())
        if len(snapshots) != 3:
            raise ValueError("missing common-clock forecast origin")
        timeline = np.load(cell / "exposure.npz")["timeline"]
        holds = pd.read_csv(cell / "holds.csv")
        terminals = pd.read_csv(cell / "terminals.csv").set_index("trip_id")
        teacher = json.loads((cell / "teacher.json").read_text())
        for origin, capture in snapshots.items():
            now, model = capture["pool"]["time_s"], capture["distribution_model"]
            if model["latest_time_s"] >= now:
                raise ValueError("categorical model used simultaneous/future actions")
            for horizon in (900, 1800):
                actual = previous.at(timeline, now + horizon) - previous.at(timeline, now)
                actual_hold = np.maximum(0, np.minimum(now + horizon - 1, holds.time_s + holds.action_s)
                                          - np.maximum(now - 1, holds.time_s)).sum()
                truths = dict(zip(METRICS, [*actual, float(actual_hold)]))
                sampled = capture["ensemble"][str(horizon)]
                for mode in ("historical", "response", "categorical"):
                    point = sampled["mean"] if mode == "categorical" else capture["forecasts"][f"{mode}_{horizon}"]
                    for metric, truth in truths.items():
                        draws = np.asarray(sampled["draws"][metric]) if mode == "categorical" else None
                        lo, hi = np.quantile(draws, [.05, .95]) if draws is not None else (np.nan, np.nan)
                        rows.append(dict(train_seed=seed, eval_seed=scenario, cohort=cohort, origin=origin,
                            mode=mode, horizon_s=horizon, metric=metric, actual=truth, predicted=point[metric],
                            error=point[metric] - truth, low=lo, high=hi,
                            covered=float(lo <= truth <= hi) if draws is not None else np.nan,
                            mc_se=float(np.std(draws, ddof=1) / np.sqrt(len(draws))) if draws is not None else np.nan))
                # Future observed states/times are used only for teacher-forced
                # error attribution, never for fitting or closed-loop forecasts.
                future = [row for row in teacher if now <= row["time_s"] < now + horizon]
                predicted_sum, expected_future_exposure = 0.0, 0.0
                for row in future:
                    probs = probabilities_from_features(model, row["features"])
                    label = int(np.argmin(abs(ACTIONS - row["action_s"])))
                    expected = float(probs @ ACTIONS)
                    predicted_sum += expected
                    expected_future_exposure += float(probs @ np.minimum(ACTIONS,
                        max(now + horizon - 1 - row["time_s"], 0)))
                    teacher_rows.append(dict(cohort=cohort, horizon_s=horizon, brier=float(
                        np.sum((probs - np.eye(7)[label]) ** 2)), nll=float(-np.log(max(probs[label], 1e-300))),
                        mean_error_s=expected - row["action_s"]))
                carry = holds.loc[holds.time_s < now]
                carry_exposure = np.maximum(0, np.minimum(now + horizon - 1, carry.time_s + carry.action_s)
                                            - np.maximum(now - 1, carry.time_s)).sum()
                teacher_totals.append(dict(train_seed=seed, eval_seed=scenario, cohort=cohort,
                    origin=origin, horizon_s=horizon, commands=len(future),
                    actual=sum(row["action_s"] for row in future), predicted=predicted_sum,
                    error=predicted_sum - sum(row["action_s"] for row in future),
                    exposure_error=expected_future_exposure + carry_exposure - actual_hold))
                for trip, times in sampled["returns"].items():
                    truth = float(terminals.loc[int(trip)].time_s)
                    returned = np.array([time is not None for time in times])
                    # Censored returns stay censored, rather than dropping late
                    # trajectories to produce deceptively narrow timing bands.
                    times = np.array([time if time is not None else np.inf for time in times])
                    sorted_times = np.sort(times)
                    lo, hi = sorted_times[int(.05 * (DRAWS - 1))], sorted_times[int(np.ceil(.95 * (DRAWS - 1)))]
                    return_rows.append(dict(cohort=cohort, horizon_s=horizon, return_probability=float(returned.mean()),
                        returned_actual=float(truth <= now + horizon), finite_band=float(np.isfinite(hi)),
                        covered=float(lo <= truth <= hi) if np.isfinite(hi) else np.nan,
                        width_s=hi - lo if np.isfinite(hi) else np.nan))
    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame["absolute_error"] = frame.error.abs()
    frame["width"] = frame.high - frame.low
    groups = ["cohort", "mode", "horizon_s", "metric"]
    errors = frame.groupby(groups).agg(contexts=("error", "size"), mae=("absolute_error", "mean"),
        bias=("error", "mean"), mean_actual=("actual", "mean"), coverage=("covered", "mean"),
        width=("width", "mean"), mc_se=("mc_se", "mean"))
    errors.to_csv(summary / "prediction_errors.csv")
    frame.groupby(["cohort", "train_seed", "eval_seed", "mode", "horizon_s", "metric"]).agg(
        mae=("absolute_error", "mean"), bias=("error", "mean")).to_csv(summary / "day_errors.csv")
    frame.to_csv(root / "prediction_records.csv", index=False)
    pd.DataFrame(teacher_rows).groupby(["cohort", "horizon_s"]).mean().to_csv(summary / "teacher_accuracy.csv")
    pd.DataFrame(teacher_totals).groupby(["cohort", "horizon_s"]).agg(
        contexts=("error", "size"), commands=("commands", "mean"), actual=("actual", "mean"),
        predicted=("predicted", "mean"), bias=("error", "mean"),
        exposure_bias=("exposure_error", "mean")).to_csv(summary / "teacher_accumulation.csv")
    returned = pd.DataFrame(return_rows)
    returned["brier"] = (returned.return_probability - returned.returned_actual) ** 2
    returned.groupby(["cohort", "horizon_s"]).mean().to_csv(summary / "return_uncertainty.csv")
    gates = {}
    for cohort in ("development", "new_demand"):
        cat = errors.loc[(cohort, "categorical", 1800)]
        base = errors.loc[(cohort, "historical", 1800)]
        holding = cat.loc["holding_vehicle_seconds"]
        gates[cohort] = dict(
            holding_mae_no_worse=bool(holding.mae <= base.loc["holding_vehicle_seconds"].mae),
            holding_relative_bias_le_5pct=bool(abs(holding.bias) <= .05 * holding.mean_actual),
            passenger_mae_within_5pct=bool(all(cat.loc[m].mae <= 1.05 * base.loc[m].mae
                for m in ("waiting_person_seconds", "onboard_person_seconds"))),
            holding_band_coverage_ge_80pct=bool(holding.coverage >= .8))
    status = dict(status="complete_distribution_shadow", days=8, draws=DRAWS, policy_changed=False,
        matches_v35=True, gates=gates, forecast_screen_pass=all(all(g.values()) for g in gates.values()),
        new_demand_scope="Two unused V41 demand seeds crossed with the same two frozen actors; not new policy seeds",
        uncertainty_scope="Holding sampling only; flow/travel and future upper replanning uncertainty remain unmodelled")
    (summary / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    print(errors.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "batch", "aggregate"])
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1] / "tests"),
                                                   pattern="test_v4[012]*.py")
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        run_day((args.out_root, *items()[0]))
        (args.out_root / "done.json").write_text(json.dumps(dict(tests_passed=True, matches_v35=True)) + "\n")
    elif args.command == "batch":
        batch(args.out_root, args.start, args.end, args.workers)
    else:
        aggregate(args.out_root)
