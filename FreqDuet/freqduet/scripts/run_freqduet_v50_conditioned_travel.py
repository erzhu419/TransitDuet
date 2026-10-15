#!/usr/bin/env python3
"""V50 prefix-conditioned travel: preparation, prequential scores and frozen forecasts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from freqduet_closed_loop_forecast import fluid_rollout
from freqduet_conditioned_travel import MODES, fit_mean, fit_noise, road_key, ConditionedTravelPath
from freqduet_holding_distribution import DRAWS, METRICS, probabilities, sample_action
from freqduet_passenger_distribution import PassengerProcess
from freqduet_travel_distribution import score_draws
from run_freqduet_v49_residual_dependence import cluster_summary
import run_freqduet_v46_travel_blocks as previous


def prepare_day(arguments):
    root, base, seed, scenario, cohort = arguments
    label = f"seed{seed}_eval{scenario}"
    captures = json.loads((base / "protocol_v6_v45_completed_calibration" / "observed" / label / "forecasts.json").read_text())
    control = json.loads((base / "protocol_v6_v47_boundary_apc" / "observed" / label / "done.json").read_text())
    if not control["matches_frozen_control"] or not control["matches_v45_prefix_flows"]:
        raise ValueError("travel preparation lacks matched frozen telemetry")
    legs = pd.read_csv(base / "protocol_v6_v47_boundary_apc" / "observed" / label / "completed_legs.csv", dtype={"direction": str})
    models = {clock: fit_mean(legs, captures[f"clock_{clock}"]["pool"]) for clock in previous.CLOCKS}
    innovations = pd.read_csv(base / "protocol_v6_v49_residual_dependence" / "days" / label / "travel_records.csv")
    innovations["direction"] = innovations.direction.map(str)
    innovations["expected"] = [models[r.origin]["means"][road_key(r.direction, r.station_id)] for r in innovations.itertuples()]
    scores, support = [], []
    for clock, model in models.items():
        model = fit_noise(model, innovations, legs)
        models[clock] = model
        old = json.loads((base / "protocol_v6_v46_travel_blocks" / "origins" / label / f"clock_{clock}" / "forecasts.json").read_text())["travel_model"]
        support.append(dict(clock=clock, samples=model["samples"], prequential_noise_legs=model["prequential_noise_legs"],
            shared_fraction=model["shared_fraction"], within_bin_dof=model["within_bin_dof"],
            latest_completion_s=model["latest_completion_s"], noise_latest_completion_s=model["noise_latest_completion_s"]))
        for row in innovations.loc[innovations.origin == clock].itertuples():
            key = road_key(row.direction, row.station_id)
            for mode, draws in (("legacy", old["profiles"][key]),
                    ("conditioned", model["means"][key] * np.asarray(model["ratios"][key]))):
                scores.append(dict(origin=clock, mode=mode, **score_draws(draws, row.actual)))
    output = root / "prepared" / label
    output.mkdir(parents=True, exist_ok=True)
    (output / "models.json").write_text(json.dumps(models) + "\n")
    frame = pd.DataFrame(scores)
    frame["abs_error"] = frame.error.abs()
    frame.groupby("mode").agg(events=("error", "size"), mae=("abs_error", "mean"), bias=("error", "mean"),
        crps=("crps", "mean"), coverage=("covered", "mean"), width=("width", "mean")).to_csv(output / "scores.csv")
    pd.DataFrame(support).to_csv(output / "support.csv", index=False)
    innovations.to_csv(output / "innovations.csv", index=False)
    (output / "done.json").write_text(json.dumps(dict(seed=seed, scenario=scenario, policy_changed=False,
        source_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))) + "\n")
    print(f"CONDITIONED_PREPARED {label}", flush=True)
    return label


def prepare_summary(root):
    scores, support = [], []
    for seed, scenario, cohort in previous.previous.items():
        cell = root / "prepared" / f"seed{seed}_eval{scenario}"
        scores.append(pd.read_csv(cell / "scores.csv").assign(train_seed=seed, eval_seed=scenario, cohort=cohort))
        support.append(pd.read_csv(cell / "support.csv").assign(train_seed=seed, eval_seed=scenario))
    frame = pd.concat(scores, ignore_index=True)
    baseline = frame.loc[frame["mode"] == "legacy", ["train_seed", "eval_seed", "mae", "crps", "coverage"]]
    paired = frame.loc[frame["mode"] == "conditioned"].merge(baseline, on=["train_seed", "eval_seed"], suffixes=("", "_legacy"), validate="one_to_one")
    for metric in ("mae", "crps", "coverage"):
        paired[metric + "_delta"] = paired[metric] - paired[metric + "_legacy"]
    out = root / "prepare_summary"
    out.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out / "day_paired.csv", index=False)
    result = cluster_summary(paired.assign(scope="all"), ["scope"], ["mae_delta", "crps_delta", "coverage_delta"])
    result.to_csv(out / "cluster.csv", index=False)
    cov = paired.groupby("cohort")[["mae", "mae_legacy", "coverage", "coverage_legacy"]].mean()
    demand = paired.groupby("eval_seed")[["mae", "mae_legacy"]].mean()
    passed = bool((cov.mae <= cov.mae_legacy).all() and (cov.coverage >= cov.coverage_legacy).all() and
                  (demand.mae <= 1.05 * demand.mae_legacy).all())
    prefixes = pd.concat(support, ignore_index=True)
    if (prefixes.latest_completion_s >= prefixes.clock).any() or (prefixes.noise_latest_completion_s >= prefixes.clock).any():
        raise ValueError("incomplete/simultaneous label in prepared model")
    status = dict(days=12, demand_clusters=6, prequential_screen_pass=passed, policy_changed=False)
    (out / "status.json").write_text(json.dumps(status) + "\n")
    print(cov.to_string(), flush=True)
    print(result.to_string(index=False), flush=True)
    print(json.dumps(status), flush=True)


def forecast_ensemble(observed, horizon, holding, travel, passengers, mode, event_sink=None):
    metrics, errors = {metric: [] for metric in METRICS}, []
    for draw in range(DRAWS):
        holding_rng = np.random.default_rng(20261015 + draw)
        path = ConditionedTravelPath(travel, mode, draw)

        def predict(vehicle, time, fraction):
            route = observed["routes"][str(vehicle["direction"])]
            return path(dict(vehicle, travel_station_id=route["station_ids"][vehicle["station_index"]]), time, fraction)

        result = fluid_rollout(observed, horizon, "response",
            action_predictor=lambda v: sample_action(probabilities(holding, v), holding_rng.random()),
            event_sink=None if event_sink is None else lambda row: event_sink(draw, row),
            travel_predictor=predict, passenger_process=PassengerProcess(passengers, "passenger_joint", draw))
        errors.append(abs(result["passenger_mass_error"]))
        for metric in metrics:
            metrics[metric].append(result[metric])
    return dict(mean={metric: float(np.mean(v)) for metric, v in metrics.items()}, draws=metrics,
                max_mass_error=max(errors))


def run_origin(arguments):
    root, base, seed, scenario, cohort, clock = arguments
    label = f"seed{seed}_eval{scenario}"
    capture = json.loads((base / "protocol_v6_v45_completed_calibration" / "observed" / label / "forecasts.json").read_text())[f"clock_{clock}"]
    reference = json.loads((base / "protocol_v6_v48_passenger_uncertainty" / "origins" / label / f"clock_{clock}" / "forecasts.json").read_text())
    model = json.loads((root / "prepared" / label / "models.json").read_text())[str(clock)]
    results, states = {"passenger_joint": reference["forecasts"]["passenger_joint"]}, []
    for mode in MODES:
        traces = []
        results[mode] = {str(h): forecast_ensemble(capture["pool"], h, capture["distribution_model"], model,
            reference["passenger_model"], mode, event_sink=(lambda draw, row: traces.append(dict(draw=draw, **row)))
                if h == 1800 and clock in previous.COMMON else None) for h in (900, 1800)}
        if traces:
            actual = pd.read_csv(base / "protocol_v6_v47_boundary_apc" / "observed" / label / "actual_events.csv")
            states.extend(previous.state_scores(pd.DataFrame(traces), actual, clock, mode))
    output = root / "origins" / label / f"clock_{clock}"
    output.mkdir(parents=True, exist_ok=True)
    # V46 aggregation checks the generic completed-prefix fields in this model.
    (output / "forecasts.json").write_text(json.dumps(dict(cohort=cohort, travel_model=model, forecasts=results)) + "\n")
    if states:
        old = pd.read_csv(base / "protocol_v6_v48_passenger_uncertainty" / "origins" / label / f"clock_{clock}" / "state_scores.csv")
        pd.concat([old.loc[old["mode"] == "passenger_joint"], pd.DataFrame(states)], ignore_index=True).to_csv(output / "state_scores.csv", index=False)
    error = max(results[m][str(h)]["max_mass_error"] for m in MODES for h in (900, 1800))
    (output / "done.json").write_text(json.dumps(dict(max_mass_error=error, policy_changed=False,
        source_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))) + "\n")
    print(f"CONDITIONED_ORIGIN_DONE {label} clock={clock} mass_error={error:.3g}", flush=True)
    return str(output.relative_to(root))


def batch(root, base, start, end, workers, prepare=False):
    work = previous.previous.items() if prepare else previous.items()
    function = prepare_day if prepare else run_origin
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        completed = list(pool.map(function, [(root, base, *item) for item in work[start:end]]))
    out = root / ("prepare_exports" if prepare else "exports") / f"batch_{start}_{end}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "done.json").write_text(json.dumps(dict(completed=completed)) + "\n")


def aggregate(root, base):
    previous.aggregate(root, base / "protocol_v6_v45_completed_calibration", modes=("passenger_joint", *MODES))
    out = root / "summary"
    paired = pd.read_csv(out / "day_paired.csv")
    reference = paired.loc[paired["mode"] == "passenger_joint", ["train_seed", "eval_seed", "metric", "mae"]].rename(columns={"mae": "v48_mae"})
    paired = paired.merge(reference, on=["train_seed", "eval_seed", "metric"], validate="many_to_one")
    paired["mae_delta_vs_v48"] = paired.mae - paired.v48_mae
    paired.to_csv(out / "day_paired.csv", index=False)
    cluster_summary(paired, ["mode", "metric"], ["mae_delta_vs_v48"]).to_csv(out / "demand_cluster_deltas.csv", index=False)
    errors = pd.read_csv(out / "errors.csv")
    print(errors.loc[(errors.scope == "all_origins") & (errors.horizon_s == 1800) &
        errors["mode"].isin(("passenger_joint", *MODES)) & errors.metric.isin(METRICS[:2] + METRICS[-1:])].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "prepare", "prepare-summary", "batch", "aggregate"])
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--data-base", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=612)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prepare-summary":
        prepare_summary(args.out_root)
    elif args.command == "aggregate":
        aggregate(args.out_root, args.data_base)
    elif args.command in ("prepare", "batch"):
        batch(args.out_root, args.data_base, args.start, args.end, args.workers, args.command == "prepare")
    else:
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1] / "tests"), pattern="test_v50*.py")
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        prepare_day((args.out_root, args.data_base, *previous.previous.items()[0]))
        seed, scenario, cohort = previous.previous.items()[0]
        run_origin((args.out_root, args.data_base, seed, scenario, cohort, 3600))
        out = args.out_root / "smoke_export"
        out.mkdir(parents=True, exist_ok=True)
        (out / "done.json").write_text('{"conditioned_travel_smoke": true}\n')
