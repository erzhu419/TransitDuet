#!/usr/bin/env python3
"""V51 prefix-trained turnover response; four-cell travel/holding forecast diagnosis."""

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
from freqduet_conditioned_travel import ConditionedTravelPath
from freqduet_holding_distribution import ACTIONS, DRAWS, METRICS, probabilities_from_features
from freqduet_passenger_distribution import PassengerProcess, passenger_ensemble
from freqduet_residual_dependence import identify_holding, issued_origin
from freqduet_travel_distribution import TravelPath
from freqduet_turnover_holding import augment_labels, freeze_turnover, TurnoverResponse, fit_turnover_tilt, tilted_probabilities
from run_freqduet_v49_residual_dependence import cluster_summary
import run_freqduet_v46_travel_blocks as previous

MODES = ("turnover_legacy", "turnover_shared")


def read_day(base, seed, scenario):
    label = f"seed{seed}_eval{scenario}"
    bank = base / "protocol_v6_v45_completed_calibration" / "observed" / label
    telemetry = base / "protocol_v6_v47_boundary_apc" / "observed" / label
    captures = json.loads((bank / "forecasts.json").read_text())
    labels = identify_holding(json.loads((bank / "teacher.json").read_text()), pd.read_csv(bank / "holds.csv"),
        pd.read_csv(telemetry / "actual_events.csv"), captures["clock_3600"]["pool"]["routes"])
    launches, terminals = pd.read_csv(telemetry / "launches.csv"), pd.read_csv(telemetry / "terminals.csv")
    return label, captures, augment_labels(labels, launches, terminals), launches, terminals


def score_action(probabilities, actual):
    mean = float(probabilities @ ACTIONS)
    label = int(np.argmin(abs(ACTIONS - actual)))
    crps = probabilities @ abs(ACTIONS - actual) - .5 * np.sum(
        probabilities[:, None] * probabilities[None, :] * abs(ACTIONS[:, None] - ACTIONS[None, :]))
    return dict(error=mean - actual, crps=float(crps), nll=float(-np.log(probabilities[label])))


def prepare_day(arguments):
    root, base, seed, scenario, cohort = arguments
    label, captures, labels, launches, terminals = read_day(base, seed, scenario)
    models, scores = {}, []
    prequential = []
    for row in labels.itertuples():
        clock = issued_origin(previous.CLOCKS, row.time_s)
        if clock is not None:
            prequential.append(dict(time_s=row.time_s, action_s=row.action_s,
                context=row.augmented_features[12:], base_probabilities=probabilities_from_features(
                    captures[f"clock_{clock}"]["distribution_model"], row.features)))
    prequential = pd.DataFrame(prequential)
    for clock in previous.CLOCKS:
        model = dict(base=captures[f"clock_{clock}"]["distribution_model"], correction=fit_turnover_tilt(prequential, clock))
        models[clock] = dict(model=model, history=freeze_turnover(labels, launches, terminals, captures[f"clock_{clock}"]["pool"]))
    for row in labels.itertuples():
        clock = issued_origin(previous.CLOCKS, row.time_s)
        if clock is None:
            continue
        model = models[clock]["model"]
        base_probs = probabilities_from_features(model["base"], row.features)
        new_probs = tilted_probabilities(model["base"], row.features, row.augmented_features[12:], model["correction"])
        for mode, probabilities in (("legacy", base_probs), ("turnover", new_probs)):
            scores.append(dict(mode=mode, **score_action(probabilities, row.action_s)))
    output = root / "prepared" / label
    output.mkdir(parents=True, exist_ok=True)
    (output / "models.json").write_text(json.dumps(models) + "\n")
    frame = pd.DataFrame(scores)
    frame["abs_error"] = frame.error.abs()
    frame.groupby("mode").agg(events=("error", "size"), mae=("abs_error", "mean"), bias=("error", "mean"),
        crps=("crps", "mean"), nll=("nll", "mean")).to_csv(output / "scores.csv")
    print(f"TURNOVER_PREPARED {label}", flush=True)
    return label


def prepare_summary(root):
    frame = pd.concat([pd.read_csv(root / "prepared" / f"seed{s}_eval{d}" / "scores.csv").assign(
        train_seed=s, eval_seed=d, cohort=c) for s, d, c in previous.previous.items()], ignore_index=True)
    old = frame.loc[frame["mode"] == "legacy", ["train_seed", "eval_seed", "mae", "crps", "nll"]]
    paired = frame.loc[frame["mode"] == "turnover"].merge(old, on=["train_seed", "eval_seed"], suffixes=("", "_legacy"), validate="one_to_one")
    for metric in ("mae", "crps", "nll"):
        paired[metric + "_delta"] = paired[metric] - paired[metric + "_legacy"]
    out = root / "prepare_summary"
    out.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out / "day_paired.csv", index=False)
    result = cluster_summary(paired.assign(scope="all"), ["scope"], ["mae_delta", "crps_delta", "nll_delta"])
    result.to_csv(out / "cluster.csv", index=False)
    mean = paired.groupby("cohort")[["mae", "mae_legacy", "nll", "nll_legacy"]].mean()
    passed = bool((mean.mae <= mean.mae_legacy).all() and (mean.nll <= mean.nll_legacy).all())
    (out / "status.json").write_text(json.dumps(dict(days=12, demand_clusters=6, prequential_screen_pass=passed, policy_changed=False)) + "\n")
    print(mean.to_string(), flush=True)
    print(result.to_string(index=False), flush=True)
    print(f"TURNOVER_PREQUENTIAL_PASS={passed}", flush=True)


def ensemble(observed, horizon, response, travel, passengers, mode, event_sink=None):
    metrics, errors = {metric: [] for metric in METRICS}, []
    for draw in range(DRAWS):
        holding = TurnoverResponse(response["model"], response["history"], observed["routes"], draw)
        path = (TravelPath(travel, "shared_block", np.random.default_rng(20261016 + draw))
                if mode == "turnover_legacy" else ConditionedTravelPath(travel, "conditioned_shared", draw))

        def predict(vehicle, time, fraction):
            route = observed["routes"][str(vehicle["direction"])]
            return path(dict(vehicle, travel_station_id=route["station_ids"][vehicle["station_index"]]), time, fraction)

        def sink(row):
            holding.observe(row)
            if event_sink is not None:
                event_sink(draw, row)

        result = fluid_rollout(observed, horizon, "response", action_predictor=holding,
            event_sink=sink, travel_predictor=predict, passenger_process=PassengerProcess(passengers, "passenger_joint", draw))
        errors.append(abs(result["passenger_mass_error"]))
        for metric in metrics:
            metrics[metric].append(result[metric])
    return dict(mean={metric: float(np.mean(v)) for metric, v in metrics.items()}, draws=metrics, max_mass_error=max(errors))


def run_origin(arguments):
    root, base, seed, scenario, cohort, clock = arguments
    label = f"seed{seed}_eval{scenario}"
    capture = json.loads((base / "protocol_v6_v45_completed_calibration" / "observed" / label / "forecasts.json").read_text())[f"clock_{clock}"]
    old = json.loads((base / "protocol_v6_v48_passenger_uncertainty" / "origins" / label / f"clock_{clock}" / "forecasts.json").read_text())
    new = json.loads((base / "protocol_v6_v50_conditioned_travel" / "origins" / label / f"clock_{clock}" / "forecasts.json").read_text())
    response = json.loads((root / "prepared" / label / "models.json").read_text())[str(clock)]
    results = dict(passenger_joint=old["forecasts"]["passenger_joint"], conditioned_shared=new["forecasts"]["conditioned_shared"])
    states = []
    for mode, travel in (("turnover_legacy", old["travel_model"]), ("turnover_shared", new["travel_model"])):
        traces = []
        results[mode] = {str(h): ensemble(capture["pool"], h, response, travel, old["passenger_model"], mode,
            event_sink=(lambda draw, row: traces.append(dict(draw=draw, **row))) if h == 1800 and clock in previous.COMMON else None) for h in (900, 1800)}
        if traces:
            actual = pd.read_csv(base / "protocol_v6_v47_boundary_apc" / "observed" / label / "actual_events.csv")
            states.extend(previous.state_scores(pd.DataFrame(traces), actual, clock, mode))
    output = root / "origins" / label / f"clock_{clock}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "forecasts.json").write_text(json.dumps(dict(cohort=cohort, travel_model=new["travel_model"], forecasts=results)) + "\n")
    if states:
        old_states = pd.read_csv(base / "protocol_v6_v50_conditioned_travel" / "origins" / label / f"clock_{clock}" / "state_scores.csv")
        pd.concat([old_states.loc[old_states["mode"].isin(("passenger_joint", "conditioned_shared"))], pd.DataFrame(states)], ignore_index=True).to_csv(output / "state_scores.csv", index=False)
    (output / "done.json").write_text(json.dumps(dict(policy_changed=False,
        source_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))) + "\n")
    print(f"TURNOVER_ORIGIN_DONE {label} clock={clock}", flush=True)
    return str(output.relative_to(root))


def aggregate(root, base):
    previous.aggregate(root, base / "protocol_v6_v45_completed_calibration", modes=("passenger_joint", "conditioned_shared", *MODES))
    out = root / "summary"
    paired = pd.read_csv(out / "day_paired.csv").pivot(index=["train_seed", "eval_seed", "metric"], columns="mode", values="mae")
    paired["travel_only_delta"] = paired.conditioned_shared - paired.passenger_joint
    paired["holding_only_delta"] = paired.turnover_legacy - paired.passenger_joint
    paired["joint_delta"] = paired.turnover_shared - paired.passenger_joint
    paired["holding_delta_on_new_travel"] = paired.turnover_shared - paired.conditioned_shared
    paired["error_interaction"] = paired.joint_delta - paired.travel_only_delta - paired.holding_only_delta
    paired = paired.reset_index()
    paired.to_csv(out / "factorial_days.csv", index=False)
    result = cluster_summary(paired, ["metric"], ["travel_only_delta", "holding_only_delta", "joint_delta",
        "holding_delta_on_new_travel", "error_interaction"])
    result.to_csv(out / "factorial_clusters.csv", index=False)
    print(result.to_string(index=False), flush=True)


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
        prepare = args.command == "prepare"
        work = previous.previous.items() if prepare else previous.items()
        function = prepare_day if prepare else run_origin
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            completed = list(pool.map(function, [(args.out_root, args.data_base, *item) for item in work[args.start:args.end]]))
        out = args.out_root / ("prepare_exports" if prepare else "exports") / f"batch_{args.start}_{args.end}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "done.json").write_text(json.dumps(dict(completed=completed)) + "\n")
    else:
        tests = Path(__file__).resolve().parents[1] / "tests"
        suite = unittest.TestSuite(unittest.defaultTestLoader.discover(str(tests), pattern=pattern)
            for pattern in ("test_v4*.py", "test_v50*.py", "test_v51*.py"))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        seed, scenario, cohort = previous.previous.items()[0]
        label = f"seed{seed}_eval{scenario}"
        capture = json.loads((args.data_base / "protocol_v6_v45_completed_calibration" / "observed" / label / "forecasts.json").read_text())["clock_3600"]
        old = json.loads((args.data_base / "protocol_v6_v48_passenger_uncertainty" / "origins" / label / "clock_3600" / "forecasts.json").read_text())
        replay = passenger_ensemble(capture["pool"], 1800, capture["distribution_model"], old["travel_model"], old["passenger_model"], "passenger_joint")
        for metric in METRICS:
            np.testing.assert_allclose(replay["draws"][metric], old["forecasts"]["passenger_joint"]["1800"]["draws"][metric], rtol=1e-12, atol=1e-8)
        prepare_day((args.out_root, args.data_base, seed, scenario, cohort))
        run_origin((args.out_root, args.data_base, seed, scenario, cohort, 3600))
        out = args.out_root / "smoke_export"
        out.mkdir(parents=True, exist_ok=True)
        (out / "done.json").write_text('{"v48_exact": true, "turnover_smoke": true}\n')
