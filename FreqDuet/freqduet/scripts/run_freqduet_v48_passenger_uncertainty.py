#!/usr/bin/env python3
"""V48: passenger process uncertainty with V46 travel and historical APC means."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from freqduet_holding_distribution import METRICS
from freqduet_passenger_distribution import MODES, fit_passengers, passenger_ensemble
from freqduet_travel_distribution import travel_ensemble
import run_freqduet_v46_travel_blocks as previous


def run_origin(arguments):
    root, bank, v46, telemetry, seed, scenario, cohort, clock = arguments
    label = f"seed{seed}_eval{scenario}"
    control = json.loads((telemetry / "observed" / label / "done.json").read_text())
    if not control["matches_frozen_control"] or not control["matches_v45_prefix_flows"]:
        raise ValueError("passenger telemetry lacks the frozen-control/prefix contract")
    capture = json.loads((bank / "observed" / label / "forecasts.json").read_text())[f"clock_{clock}"]
    base = json.loads((v46 / "origins" / label / f"clock_{clock}" / "forecasts.json").read_text())
    observed, holding = capture["pool"], capture["distribution_model"]
    model = fit_passengers(pd.read_csv(telemetry / "observed" / label / "apc.csv"), observed)
    if model["latest_time_s"] >= clock or holding["latest_time_s"] >= clock:
        raise ValueError("simultaneous/future APC or holding observation in forecast")
    results = {"shared_block": base["forecasts"]["shared_block"]}
    states = []
    for mode in MODES:
        traces = []
        results[mode] = {str(horizon): passenger_ensemble(observed, horizon, holding, base["travel_model"], model, mode,
            event_sink=(lambda draw, row: traces.append(dict(draw=draw, **row)))
                if horizon == 1800 and clock in previous.COMMON else None) for horizon in (900, 1800)}
        if traces:
            actual = pd.read_csv(telemetry / "observed" / label / "actual_events.csv")
            states.extend(previous.state_scores(pd.DataFrame(traces), actual, clock, mode))
    output = root / "origins" / label / f"clock_{clock}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "forecasts.json").write_text(json.dumps(dict(cohort=cohort, travel_model=base["travel_model"],
        passenger_model=model, forecasts=results)) + "\n")
    if states:
        old = pd.read_csv(v46 / "origins" / label / f"clock_{clock}" / "state_scores.csv")
        pd.concat([old.loc[old["mode"] == "shared_block"], pd.DataFrame(states)], ignore_index=True).to_csv(
            output / "state_scores.csv", index=False)
    error = max(results[mode][str(horizon)]["max_mass_error"] for mode in MODES for horizon in (900, 1800))
    (output / "done.json").write_text(json.dumps(dict(max_mass_error=error,
        source_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))) + "\n")
    print(f"PASSENGER_ORIGIN_DONE {label} clock={clock} mass_error={error:.3g}", flush=True)
    return str(output.relative_to(root))


def batch(root, bank, v46, telemetry, start, end, workers):
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        completed = list(pool.map(run_origin, [(root, bank, v46, telemetry, *item) for item in previous.items()[start:end]]))
    output = root / "exports" / f"batch_{start}_{end}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "done.json").write_text(json.dumps(dict(completed=completed)) + "\n")


def aggregate(root, bank):
    previous.aggregate(root, bank, modes=("shared_block", *MODES))
    output = root / "summary"
    errors = pd.read_csv(output / "errors.csv")
    gates = {}
    for mode in MODES:
        gates[mode] = {}
        for cohort in errors.cohort.unique():
            data = errors.loc[(errors.scope == "all_origins") & (errors.horizon_s == 1800) & (errors.cohort == cohort)]
            candidate = data.loc[data["mode"] == mode].set_index("metric")
            reference = data.loc[data["mode"] == "shared_block"].set_index("metric")
            history = data.loc[data["mode"] == "historical"].set_index("metric")
            hold = candidate.loc[METRICS[-1]]
            gates[mode][cohort] = dict(
                holding_mae_no_worse_than_v46=bool(hold.mae <= reference.loc[METRICS[-1]].mae),
                holding_mae_no_worse_than_history=bool(hold.mae <= history.loc[METRICS[-1]].mae),
                holding_bias_le_5pct=bool(abs(hold.bias) <= .05 * hold.mean_actual),
                passenger_mae_within_5pct=bool(all(candidate.loc[m].mae <= 1.05 * history.loc[m].mae for m in METRICS[:2])),
                holding_coverage_ge_80pct=bool(hold.coverage >= .8))
    paired = pd.read_csv(output / "day_paired.csv")
    base = paired.loc[paired["mode"] == "shared_block", ["train_seed", "eval_seed", "metric", "mae"]].rename(columns={"mae": "v46_mae"})
    paired = paired.merge(base, on=["train_seed", "eval_seed", "metric"], validate="many_to_one")
    paired["mae_delta_vs_v46"] = paired.mae - paired.v46_mae
    paired.to_csv(output / "day_paired.csv", index=False)
    mass = max(json.loads((root / "origins" / f"seed{seed}_eval{scenario}" / f"clock_{clock}" / "done.json").read_text())["max_mass_error"]
               for seed, scenario, cohort, clock in previous.items())
    status = json.loads((output / "status.json").read_text())
    status.update(v46_reference_gates=gates, max_passenger_mass_error=mass,
        forecast_screen_pass={mode: all(all(values.values()) for values in cohorts.values()) for mode, cohorts in gates.items()},
        historical_apc_means_unchanged=True, rate_parameter_uncertainty_modelled=False,
        station_arrivals_conditionally_independent=True)
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    print("PASSENGER_AGGREGATE_DONE", flush=True)


def smoke(root, bank, v46, telemetry):
    tests = Path(__file__).resolve().parents[1] / "tests"
    suite = unittest.defaultTestLoader.discover(str(tests), pattern="test_v4[1245678]*.py")
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)
    seed, scenario, cohort = previous.previous.items()[0]
    capture = json.loads((bank / "observed" / f"seed{seed}_eval{scenario}" / "forecasts.json").read_text())["clock_36000"]
    base = json.loads((v46 / "origins" / f"seed{seed}_eval{scenario}" / "clock_36000" / "forecasts.json").read_text())
    legacy = travel_ensemble(capture["pool"], 1800, capture["distribution_model"], base["travel_model"], "shared_block")
    for metric in METRICS:
        np.testing.assert_allclose(legacy["draws"][metric], base["forecasts"]["shared_block"]["1800"]["draws"][metric], rtol=1e-12, atol=1e-8)
    run_origin((root, bank, v46, telemetry, seed, scenario, cohort, 36000))
    output = root / "exports"
    output.mkdir(parents=True, exist_ok=True)
    (output / "done.json").write_text('{"v46_exact": true, "passenger_modes_complete": true}\n')
    print("PASSENGER_SMOKE_PASS v46_exact=True", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "batch", "aggregate"])
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--v46-root", type=Path, required=True)
    parser.add_argument("--telemetry-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=612)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out_root, args.input_bank, args.v46_root, args.telemetry_root)
    elif args.command == "batch":
        batch(args.out_root, args.input_bank, args.v46_root, args.telemetry_root, args.start, args.end, args.workers)
    else:
        aggregate(args.out_root, args.input_bank)
