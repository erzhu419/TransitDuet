#!/usr/bin/env python3
"""Read-only service tradeoff and repeat-noise audit of completed V81 pairs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v81_regularity_projection as pilot


EXTRA = ("waiting_person_s", "onboard_person_s", "dispatch_backlog_trip_s")
METRICS = pilot.pilot.METRICS + EXTRA


def repeat_variance(values):
    values = np.asarray(values, dtype=float)
    days, repeats = values.shape[:2]

    def components(x):
        within = x.var(axis=1, ddof=1).mean(axis=0)
        between = x.mean(axis=1).var(axis=0, ddof=1) - within / repeats
        return within, between

    within, between = components(values)
    rng = np.random.default_rng(81082)
    boot = np.asarray([components(values[i])[1] for i in rng.integers(days, size=(10000, days))])
    return dict(repeat_means=values.mean(axis=0).tolist(),
        conditional_sampling_variance=within.tolist(),
        between_day_variance_untruncated=between.tolist(),
        between_day_variance_ci95=np.quantile(boot, [.025, .975], axis=0).T.tolist(),
        mean_standard_error=(values.mean(axis=1).std(axis=0, ddof=1) / np.sqrt(days)).tolist(),
        sampling_variance_of_mean=(within / (repeats * days)).tolist(),
        all_repeats_positive=(values > 0).all(axis=1).sum(axis=0).tolist(),
        all_repeats_negative=(values < 0).all(axis=1).sum(axis=0).tolist(),
        mixed_repeat_sign=((values.min(axis=1) < 0) & (values.max(axis=1) > 0)).sum(axis=0).tolist())


def metrics(row):
    return {m: float(row[m]) for m in pilot.pilot.METRICS} | {
        m: float(row["stream_audit"]["exposure"][m]) for m in EXTRA}


def describe(rows):
    result = {}
    for contrast, (left, right) in pilot.CONTRASTS.items():
        days = []
        for day in pilot.DAYS:
            pairs = [r for r in rows if r["scenario_seed"] == day]
            deltas = [{m: metrics(r["results"][left])[m] - metrics(r["results"][right])[m]
                       for m in METRICS} for r in pairs]
            days.append({m: float(np.mean([d[m] for d in deltas])) for m in METRICS})
        gains = np.maximum(-np.asarray([d["passenger_person_s"] for d in days]), 0.)
        result[contrast] = dict(
            passenger_improved_days=sum(d["passenger_person_s"] < 0 for d in days),
            cv_squared_worsened_days=sum(d["cv_squared"] > 0 for d in days),
            passenger_and_cv_improved_days=sum(d["passenger_person_s"] < 0 and d["cv_squared"] <= 0 for d in days),
            largest_day_fraction_of_positive_passenger_gains=float(gains.max() / gains.sum()) if gains.sum() else 0.,
            exposure_effects={m: pilot.pilot.event.labels.expanded.paired_interval([d[m] for d in days], 81000)
                             for m in EXTRA})
    return dict(contrasts=result,
        absolute_means={label: {m: float(np.mean([metrics(r["results"][label])[m] for r in rows]))
                               for m in METRICS} for label in ("base", "raw", "projected")},
        action_counts={label: {str(int(a)): sum(r["results"][label]["action_counts"][str(int(a))] for r in rows)
                               for a in pilot.pilot.step.ACTIONS} for label in ("base", "raw", "projected")})


def run(root, out):
    rows = [json.loads((root / "pairs" / f"p{i}.json").read_text()) for i in range(len(pilot.ITEMS))]
    original = json.loads((root / "summary" / "status.json").read_text())
    for mode in ("stochastic", "deterministic"):
        reproduced = pilot.intervals([r for r in rows if r["mode"] == mode])
        for contrast in pilot.CONTRASTS:
            for metric in pilot.pilot.METRICS:
                for key in ("mean", "ci95"):
                    np.testing.assert_allclose(reproduced[contrast][metric][key],
                        original["effects"][mode][contrast][metric][key], rtol=1e-12, atol=1e-12)
    noise = {}
    for seed in (None,) + tuple(pilot.pilot.SEEDS):
        chosen = [r for r in rows if r["mode"] == "stochastic" and (seed is None or r["train_seed"] == seed)]
        noise[str(seed) if seed else "combined"] = {}
        for contrast in pilot.CONTRASTS:
            values = np.asarray([[np.mean([[r["effects"][contrast][m] for m in pilot.pilot.METRICS]
                for r in chosen if (r["scenario_seed"], r["rep"]) == (day, rep)], axis=0)
                for rep in range(3)] for day in pilot.DAYS])
            noise[str(seed) if seed else "combined"][contrast] = repeat_variance(values)
    pilot.pilot.write_json(out / "summary" / "status.json", dict(status="projection_tradeoff_audit_complete",
        metrics=pilot.pilot.METRICS, modes={mode: describe([r for r in rows if r["mode"] == mode])
            for mode in ("stochastic", "deterministic")}, sampling_noise=noise,
        original_means_and_intervals_reproduced=True, new_rollouts=0, new_fits=0,
        noise_scope="independent event-uniform repeats conditional on the same two actors and demand day",
        between_day_variance_not_clipped=True, v81_gates_changed=False, main_policy_changed=False))
    print("V81_PROJECTION_TRADEOFF_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v81-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
        str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v81_projection_tradeoff.py", "-v"], check=True)
    run(args.v81_root, args.out_root)
