#!/usr/bin/env python3
"""Describe v24 seed and disturbance heterogeneity after the frozen gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    analyze_mujoco_v24_policy_mean_upper_projection_target_development as analysis,
    mujoco_v24_policy_mean_upper_projection_target_development_spec as spec,
)


def summarize(rows):
    return {
        "path_count": len(rows),
        "reward": fmean(float(row["episode_return"]) for row in rows),
        "step_reward": fmean(float(row["reward_mean"]) for row in rows),
        "episode_length": fmean(float(row["episode_length"]) for row in rows),
        "natural_termination_count": sum(int(float(row["mdp_terminal_count"])) for row in rows),
        "forward_reward": fmean(float(row["forward_reward_sum"]) for row in rows),
        "control_reward": fmean(float(row["control_reward_sum"]) for row in rows),
        "horizon_reached_count": sum(
            int(row["episode_length"]) == spec.EPISODE_HORIZON for row in rows
        ),
        "component_correction": fmean(
            float(row["terminal_reserve_component_correction_rms_mean"])
            for row in rows
        ),
        "total_correction": fmean(
            float(row["terminal_reserve_correction_rms_mean"])
            for row in rows
        ),
    }


def diagnose(run_name):
    frozen = analysis.analyze(run_name)
    environments = []
    for environment in spec.ENVIRONMENTS:
        loaded = {
            (arm, seed): analysis._load_cell(run_name, environment, arm, seed)
            for arm in spec.ARMS for seed in spec.OPTIMIZER_SEEDS
        }
        arms = {}
        for arm in spec.ARMS:
            root_rows = []
            for seed in spec.OPTIMIZER_SEEDS:
                summary, rows = loaded[arm, seed]
                root_rows.append({
                    "optimizer_seed": seed,
                    **summarize(rows),
                    "selected_checkpoint_iteration": summary["selected_checkpoint_iteration"],
                    "selection_score": summary["checkpoint_selection_score"],
                    "upper_mse": summary["projection_consistency_weight_training"]["upper"]["unweighted_mse_mean"],
                    "lower_mse": summary["projection_consistency_weight_training"]["lower"]["unweighted_mse_mean"],
                    "sampled_target_delta_rms": summary["upper_policy_mean_projection_target_training"]["sampled_target_delta_rms_mean"],
                })
            arms[arm] = {
                **summarize([
                    row for seed in spec.OPTIMIZER_SEEDS
                    for row in loaded[arm, seed][1]
                ]),
                "roots": root_rows,
            }
        candidate = arms[spec.POLICY_MEAN_UNIFORM_010]["roots"]
        control = arms[spec.DECISION_TIME_UNIFORM_010]["roots"]
        paired = [{
            "optimizer_seed": seed,
            "reward_difference": candidate[i]["reward"] - control[i]["reward"],
            "relative_reward_difference": (
                (candidate[i]["reward"] - control[i]["reward"])
                / max(abs(control[i]["reward"]), 1.0)
            ),
        } for i, seed in enumerate(spec.OPTIMIZER_SEEDS)]
        leave_one_out = []
        for omitted in range(len(spec.OPTIMIZER_SEEDS)):
            included = [i for i in range(len(candidate)) if i != omitted]
            difference = fmean(paired[i]["reward_difference"] for i in included)
            reference = fmean(control[i]["reward"] for i in included)
            leave_one_out.append({
                "omitted_root": spec.OPTIMIZER_SEEDS[omitted],
                "relative_reward_difference": difference / max(abs(reference), 1.0),
            })
        conditions = {}
        for mode in spec.EVALUATION_DISTURBANCE_MODES:
            conditions[mode] = {
                arm: summarize([
                    row for seed in spec.OPTIMIZER_SEEDS
                    for row in loaded[arm, seed][1]
                    if row["disturbance_mode"] == mode
                ])
                for arm in spec.ARMS
            }
        environments.append({
            "environment": environment,
            "arms": arms,
            "candidate_vs_causal_paired_roots": paired,
            "candidate_vs_causal_leave_one_root_out": leave_one_out,
            "disturbance_conditions": conditions,
        })
    return {
        "run_name": run_name,
        "frozen_status": frozen["status"],
        "frozen_advance_gate": frozen["advance_gate"],
        "frozen_cell_audit_failure_count": len(frozen["cell_audit_failures"]),
        "interpretation": (
            "Post-outcome diagnostics, not selection or confirmation. "
            "Four optimizer roots per environment; shared evaluation paths "
            "and disturbance conditions are repeated measurements. "
            "Leave-one-root-out ranges describe sensitivity, not confidence intervals."
        ),
        "environments": environments,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    result = diagnose(args.run_name)
    target = ROOT / "results" / args.run_name / "analysis" / "diagnostics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
