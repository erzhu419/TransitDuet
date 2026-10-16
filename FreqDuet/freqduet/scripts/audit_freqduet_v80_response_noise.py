#!/usr/bin/env python3
"""Paired-repeat reliability of the already completed V80 responses."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v80_direction_response as response


def repeat_moments(values, seed=80081):
    values = np.asarray(values, dtype=float)
    left, right = values[:, 0], values[:, 1]
    n = len(values)
    between = ((left - left.mean(0)) * (right - right.mean(0))).sum(0) / (n - 1)
    within = np.square(left - right).mean(0) / 2.
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(n, size=(10000, n))]
    a, b = sampled[:, :, 0], sampled[:, :, 1]
    boot = ((a - a.mean(1, keepdims=True)) * (b - b.mean(1, keepdims=True))).sum(1) / (n - 1)
    return dict(repeat_means=values.mean(0).tolist(),
        conditional_sampling_variance=within.tolist(),
        between_day_covariance=between.tolist(),
        between_day_covariance_ci95=np.quantile(boot, [.025, .975], axis=0).T.tolist(),
        current_mean_standard_error=(values.mean(1).std(0, ddof=1) / np.sqrt(n)).tolist(),
        sampling_variance_contribution_to_mean=(within / (2. * n)).tolist(),
        both_positive=(np.logical_and(left > 0, right > 0).sum(0)).tolist(),
        both_negative=(np.logical_and(left < 0, right < 0).sum(0)).tolist(),
        opposite_sign=((left * right < 0).sum(0)).tolist(),
        at_least_one_zero=(np.logical_or(left == 0, right == 0).sum(0)).tolist())


def run(root, out):
    rows = [json.loads((root / "items" / f"i{i}.json").read_text()) for i in range(len(response.ITEMS))]
    labels = ("centered", "positive", "full", "full_minus_centered")
    arrays, by_seed = {}, {}
    for seed in response.pilot.SEEDS:
        arrays[seed] = {}
        for label in labels:
            arrays[seed][label] = np.asarray([[response.responses({name: np.array([
                row["results"][name][m] for m in response.COSTS]) for name in response.SCALES})[label]
                for rep in response.REPEATS
                for row in rows if (row["train_seed"], row["scenario_seed"], row["rep"]) == (seed, day, rep)]
                for day in response.DAYS])
        by_seed[str(seed)] = {label: repeat_moments(values) for label, values in arrays[seed].items()}
    combined = {label: repeat_moments(np.mean([arrays[s][label] for s in response.pilot.SEEDS], axis=0))
                for label in labels}
    original = json.loads((root / "summary" / "status.json").read_text())
    for label in labels:
        np.testing.assert_allclose(combined[label]["current_mean_standard_error"],
                                   original["combined"][label]["standard_error"], rtol=1e-12, atol=1e-12)
    response.pilot.write_json(out / "summary" / "status.json", dict(status="paired_response_noise_audit_complete",
        metrics=response.COSTS, days=len(response.DAYS), repeats=len(response.REPEATS),
        by_seed=by_seed, combined=combined, v80_standard_errors_reproduced=True,
        variance_scope="conditional sampling variance assumes independent repeats within a demand day",
        between_day_covariance_is_untruncated=True, new_rollouts=0,
        actor_updates=0, dual_updates=0, main_policy_changed=False, v80_estimands_changed=False))
    print("V80_RESPONSE_NOISE_AUDIT_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v80-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
        str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v80_response_noise.py", "-v"], check=True)
    run(args.v80_root, args.out_root)


if __name__ == "__main__":
    main()
