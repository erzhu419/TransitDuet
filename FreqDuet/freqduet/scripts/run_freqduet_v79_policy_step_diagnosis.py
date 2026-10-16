#!/usr/bin/env python3
"""Read-only diagnosis of V78's failed complete-day regularity constraint."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t

import freqduet_system_policy_step as step
import run_freqduet_v78_system_policy_step as pilot


COSTS = ("passenger_1000_person_hours", "cv_squared", "readiness_delay_s")


def jackknife_gradients(days):
    full = step.gradient_days(days).mean(0)
    deleted = np.asarray([step.gradient_days(days[:i] + days[i + 1:]).mean(0)
                          for i in range(len(days))])
    # Recompute all other-day baselines on deletion: daily terms are not independent.
    pseudo = len(days) * full - (len(days) - 1) * deleted
    return full, pseudo


def uncertainty(mean, pseudo):
    se = pseudo.std(axis=0, ddof=1) / np.sqrt(len(pseudo))
    half = student_t.ppf(.975, len(pseudo) - 1) * se
    return dict(mean=np.asarray(mean).tolist(), standard_error=se.tolist(),
                ci95=np.stack((mean - half, mean + half), axis=-1).tolist())


def path_overlap(days, theta):
    rows = []
    for day in days:
        p, features = day["probabilities"], day["features"]
        log_p = np.full_like(p, -np.inf)
        np.log(p, out=log_p, where=p > 0)
        bias = features @ theta
        log_z = logsumexp(log_p + bias, axis=1)
        kl = log_z - (p * bias).sum(1)
        sampled_bias = bias[np.arange(len(p)), day["actions"]]
        rows.append(dict(scenario_seed=int(day["scenario_seed"]), decisions=len(p),
                         mean_state_kl=float(kl.mean()), summed_state_kl=float(kl.sum()),
                         path_log_likelihood_ratio=float((sampled_bias - log_z).sum())))
    logs = np.asarray([r["path_log_likelihood_ratio"] for r in rows])
    weights = np.exp(logs - logsumexp(logs))
    return dict(days=rows, mean_summed_state_kl=float(np.mean([r["summed_state_kl"] for r in rows])),
                path_weight_effective_days=float(1. / np.square(weights).sum()),
                max_normalized_path_weight=float(weights.max()),
                importance_weighted_outcomes_used=False)


def split_directions(days):
    middle = len(days) // 2
    halves = (days[:middle], days[middle:])
    result = []
    for fit_days, held_days in (halves, halves[::-1]):
        fitted = step.fit_step(fit_days)
        theta = np.asarray(fitted["theta"])
        mean, pseudo = jackknife_gradients(held_days)
        result.append(dict(fitting_days=[int(d["scenario_seed"]) for d in fit_days],
            scoring_days=[int(d["scenario_seed"]) for d in held_days],
            theta=theta.tolist(), update_identified=fitted["update_identified"],
            fitting_directional_costs=fitted["directional_cost_changes"],
            other_half_direction=uncertainty(mean @ theta, pseudo @ theta)))
    return result


def diagnose_seed(root, out, seed):
    days = []
    for day in pilot.TRAIN_DAYS:
        with np.load(root / "training" / f"seed{seed}_day{day}.npz") as saved:
            days.append(dict(saved))
    adapter = json.loads((root / "adapters" / f"seed{seed}.json").read_text())
    theta = np.asarray(adapter["theta"])
    mean, pseudo = jackknife_gradients(days)
    np.testing.assert_allclose(mean, adapter["gradient_means"], rtol=1e-12, atol=1e-12)
    result = dict(train_seed=seed, costs=COSTS, features=step.FEATURES, theta=theta.tolist(),
        fixed_fitted_direction=uncertainty(mean @ theta, pseudo @ theta),
        feature_gradients=uncertainty(mean, pseudo), split_training_days=split_directions(days),
        path_overlap=path_overlap(days, theta),
        uncertainty_scope="descriptive conditional-on-fitted-direction; not selection-corrected",
        source_gradients_reproduced=True, deployed_candidate_fits=0, main_policy_changed=False)
    pilot.write_json(out / "seeds" / f"seed{seed}.json", result)
    print("V79_SEED_DIAGNOSIS_DONE " + str(seed), flush=True)


def pair_diagnostics(rows):
    output = {}
    for mode in ("stochastic", "deterministic"):
        chosen = [r for r in rows if r["mode"] == mode]
        day_rows = []
        for day in pilot.VALID_DAYS:
            selected = [r for r in chosen if r["scenario_seed"] == day]
            delta = {m: float(np.mean([r["effects"][m] for r in selected])) for m in pilot.METRICS}
            delta["headway_cv"] = float(np.mean([np.sqrt(r["candidate"]["cv_squared"]) -
                                                   np.sqrt(r["reference"]["cv_squared"]) for r in selected]))
            for part in ("waiting_person_s", "onboard_person_s"):
                delta[part] = float(np.mean([r["candidate"]["stream_audit"]["exposure"][part] -
                    r["reference"]["stream_audit"]["exposure"][part] for r in selected]))
            day_rows.append(dict(scenario_seed=day, effects=delta))
        gains = np.asarray([max(-r["effects"]["passenger_person_s"], 0.) for r in day_rows])
        output[mode] = dict(days=day_rows,
            passenger_improved_days=sum(r["effects"]["passenger_person_s"] < 0 for r in day_rows),
            cv_squared_worsened_days=sum(r["effects"]["cv_squared"] > 0 for r in day_rows),
            largest_day_fraction_of_positive_passenger_gains=float(gains.max() / gains.sum()) if gains.sum() else 0.,
            exposure_effects={part: pilot.event.labels.expanded.paired_interval(
                [r["effects"][part] for r in day_rows], 79000) for part in
                ("waiting_person_s", "onboard_person_s", "headway_cv")},
            absolute_means={side: {m: float(np.mean([r[side][m] for r in chosen]))
                for m in pilot.METRICS} for side in ("reference", "candidate")},
            action_counts={side: {str(int(a)): sum(r[side]["action_counts"][str(int(a))] for r in chosen)
                for a in step.ACTIONS} for side in ("reference", "candidate")})
    return output


def aggregate(root, out):
    seeds = {str(s): json.loads((out / "seeds" / f"seed{s}.json").read_text()) for s in pilot.SEEDS}
    rows = [json.loads((root / "pairs" / f"p{i}.json").read_text()) for i in range(len(pilot.EVAL_ITEMS))]
    pilot.write_json(out / "summary" / "status.json", dict(status="policy_step_gradient_diagnosis_complete",
        seeds=seeds, consumed_validation=pair_diagnostics(rows), new_rollouts=0,
        heldout_data_reclassified="V78 validation is now consumed development evidence",
        production_candidate_fits=0, actor_updates=0, dual_updates=0, main_policy_changed=False,
        v78_gates_changed=False, split_fits_are_diagnostics_not_candidates=True))
    print("V79_POLICY_STEP_DIAGNOSIS_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seed", "aggregate", "smoke"))
    parser.add_argument("--v78-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=pilot.SEEDS)
    args = parser.parse_args()
    if args.command == "smoke":
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
            str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v79_*.py", "-v"], check=True)
        pilot.write_json(args.out_root / "smoke" / "done.json", dict(status="diagnostic_tests_passed"))
    elif args.command == "seed":
        diagnose_seed(args.v78_root, args.out_root, args.seed)
    else:
        aggregate(args.v78_root, args.out_root)


if __name__ == "__main__":
    main()
