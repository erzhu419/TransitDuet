#!/usr/bin/env python3
"""Analyze the frozen MuJoCo v14.17 native primal-dual/CVaR screen."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    analyze_mujoco_v14_16_crossed_restoration_mechanism_screen as base,
)
from scripts import mujoco_v14_17_native_pd_cvar_screen_spec as spec  # noqa: E402


ANALYSIS_VERSION = "mujoco_v14_17_native_pd_cvar_screen_analysis_v1"
EXPECTED_MERGED_MANIFEST_STATUS = "development_screen_complete_unanalyzed"
RETURN_THRESHOLD = -0.02
FREQUENCY_THRESHOLD = -math.log(0.95)


def _engineering_gate(
    summary: dict[str, Any], *, arm: str
) -> dict[str, Any]:
    arm_spec = spec.ARMS[str(arm)]
    selected_iteration = int(summary.get(
        "selected_checkpoint_iteration", -2
    ))
    checks = {
        "protocol_identity": bool(
            summary.get("protocol_version")
            == spec.FROZEN_CORE_PROTOCOL_VERSION
            and summary.get("protocol_version_selection")
            == spec.FROZEN_CORE_PROTOCOL_VERSION
        ),
        "trained_checkpoint_selected": bool(
            selected_iteration
            >= spec.ANALYSIS_TRAINED_CHECKPOINT_MINIMUM_ITERATION
        ),
        "reward_actor_not_frozen": not bool(summary.get(
            "deployment_frequency_restoration_freeze_reward_actor", False
        )),
    }
    if arm in spec.NATIVE_PD_ARMS:
        upper_scale = float(summary.get(
            "upper_constraint_violation_scale_final", 0.0
        ))
        lower_scale = float(summary.get(
            "lower_constraint_violation_scale_final", 0.0
        ))
        upper_lambda = float(summary.get(
            "upper_constraint_lambda_final", math.nan
        ))
        lower_lambda = float(summary.get(
            "lower_constraint_lambda_final", math.nan
        ))
        checks.update({
            "native_dual_normalized": bool(
                summary.get("constraint_dual_normalization") == "ema_abs"
            ),
            "native_upper_cost_observed": bool(
                math.isfinite(upper_scale) and upper_scale > 0.0
            ),
            "native_lower_cost_observed": bool(
                math.isfinite(lower_scale) and lower_scale > 0.0
            ),
            "native_duals_finite": bool(
                math.isfinite(upper_lambda)
                and math.isfinite(lower_lambda)
                and upper_lambda >= 0.0
                and lower_lambda >= 0.0
            ),
        })
    if arm in spec.CVAR_ARMS:
        checks.update({
            "cvar_projection": bool(
                summary.get("deployment_frequency_projection_objective")
                == "violation_cvar"
            ),
            "cvar_selection": bool(
                summary.get("deployment_frequency_closed_loop_risk_mode")
                == "mode_cvar"
            ),
            "cvar_alpha": bool(
                math.isclose(
                    float(summary.get(
                        "deployment_frequency_closed_loop_cvar_alpha",
                        math.nan,
                    )),
                    spec.CLOSED_LOOP_CVAR_ALPHA,
                )
            ),
        })
    if arm in spec.CLOSED_LOOP_ARMS:
        prefix = "deployment_frequency_closed_loop_guard_"
        reward_violations = int(summary.get(
            f"{prefix}selected_reward_violation_count", -1
        ))
        frequency_violations = int(summary.get(
            f"{prefix}selected_frequency_violation_count", -1
        ))
        effective_updates = int(summary.get(
            f"{prefix}effective_update_count", 0
        ))
        checks.update({
            "closed_loop_reward_feasible": bool(
                0 <= reward_violations
                <= spec.MAXIMUM_CLOSED_LOOP_REWARD_VIOLATIONS
            ),
            "closed_loop_frequency_feasible": bool(
                0 <= frequency_violations
                <= spec.MAXIMUM_CLOSED_LOOP_FREQUENCY_VIOLATIONS
            ),
            "closed_loop_restoration_exercised": bool(
                effective_updates
                >= spec.MINIMUM_CLOSED_LOOP_EFFECTIVE_UPDATES
            ),
            "closed_loop_contract": bool(
                summary.get(f"{prefix}contract")
                == spec.expected_closed_loop_guard_contract(arm)
            ),
        })
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "selected_checkpoint_iteration": selected_iteration,
    }


def _optimizer_seed_rows(
    replicate_rows: list[dict[str, Any]], *, arm: str
) -> list[dict[str, Any]]:
    metrics = ("normalized_episode_return", *base.FREQUENCY_METRICS)
    rows = []
    for seed in spec.OPTIMIZER_SEEDS:
        selected = [
            row for row in replicate_rows
            if row["arm"] == arm and int(row["optimizer_seed"]) == int(seed)
        ]
        if len(selected) != len(spec.ENVIRONMENTS):
            raise ValueError("v14.17 optimizer seed lacks an environment")
        rows.append({
            "arm": arm,
            "optimizer_seed": int(seed),
            **{
                metric: float(np.mean([
                    float(row[metric]) for row in selected
                ]))
                for metric in metrics
            },
        })
    return rows


def analyze(run_dir: Path) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    preregistration = base._read_json(run / "preregistration.json")
    manifest = base._read_json(run / "merged" / "cell_manifest.json")
    if not (
        preregistration.get("development_protocol_version")
        == spec.DEVELOPMENT_PROTOCOL_VERSION
        and preregistration.get("frozen_algorithm_revision")
        == spec.FROZEN_ALGORITHM_REVISION
        and preregistration.get("frozen_source_manifest_sha256")
        == spec.FROZEN_SOURCE_MANIFEST_SHA256
        and preregistration.get("dispatched_environment_subset")
        == list(spec.ENVIRONMENTS)
        and preregistration.get("dispatched_optimizer_seed_subset")
        == list(spec.OPTIMIZER_SEEDS)
        and preregistration.get("arms")
        == json.loads(json.dumps(spec.ARMS))
    ):
        raise ValueError("v14.17 preregistration identity mismatch")
    if manifest.get("status") != EXPECTED_MERGED_MANIFEST_STATUS:
        raise ValueError("v14.17 merged cell manifest is not valid")
    expected_cells = (
        len(spec.ENVIRONMENTS)
        * len(spec.OPTIMIZER_SEEDS)
        * (len(spec.ARMS) + 1)
    )
    if int(manifest.get("cell_count", -1)) != expected_cells:
        raise ValueError("v14.17 merged cell count is incomplete")

    replicate_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    for environment in spec.ENVIRONMENTS:
        for optimizer_seed in spec.OPTIMIZER_SEEDS:
            baseline_path = base._cell_dir(
                run,
                environment=environment,
                arm=spec.MATCHED_COMPARATOR_ARM,
                optimizer_seed=optimizer_seed,
            )
            baseline_rows = base._read_rows(
                baseline_path / "evaluation_rows.csv"
            )
            for arm in spec.LEARNED_ARMS:
                path = base._cell_dir(
                    run,
                    environment=environment,
                    arm=arm,
                    optimizer_seed=optimizer_seed,
                )
                summary = base._read_json(path / "cell_summary.json")
                effects = base._paired_path_effects(
                    base._read_rows(path / "evaluation_rows.csv"),
                    baseline_rows,
                )
                pooled = base._pooled_effects(effects)
                gate = base._effect_gate(pooled)
                engineering = _engineering_gate(summary, arm=arm)
                replicate_rows.append({
                    "environment": environment,
                    "optimizer_seed": int(optimizer_seed),
                    "arm": arm,
                    **pooled,
                    "endpoint_pass_count": int(gate["pass_count"]),
                    "complete_effect_gate": bool(gate["complete"]),
                    "engineering_pass": bool(engineering["pass"]),
                    "engineering_checks": engineering["checks"],
                    "selected_checkpoint_iteration": int(engineering[
                        "selected_checkpoint_iteration"
                    ]),
                    "upper_constraint_lambda_final": float(summary.get(
                        "upper_constraint_lambda_final", 0.0
                    )),
                    "lower_constraint_lambda_final": float(summary.get(
                        "lower_constraint_lambda_final", 0.0
                    )),
                    "upper_constraint_violation_scale_final": float(
                        summary.get(
                            "upper_constraint_violation_scale_final", 0.0
                        )
                    ),
                    "lower_constraint_violation_scale_final": float(
                        summary.get(
                            "lower_constraint_violation_scale_final", 0.0
                        )
                    ),
                    "closed_loop_risk_mode": str(summary.get(
                        "deployment_frequency_closed_loop_risk_mode",
                        "disabled",
                    )),
                })
                for row in effects:
                    path_rows.append({
                        "environment": environment,
                        "optimizer_seed": int(optimizer_seed),
                        "arm": arm,
                        **row,
                    })

    metrics = ("normalized_episode_return", *base.FREQUENCY_METRICS)
    arm_rows: list[dict[str, Any]] = []
    optimizer_seed_rows: list[dict[str, Any]] = []
    for arm in spec.LEARNED_ARMS:
        selected = [row for row in replicate_rows if row["arm"] == arm]
        means = {
            metric: float(np.mean([float(row[metric]) for row in selected]))
            for metric in metrics
        }
        environment_gates = {}
        for environment in spec.ENVIRONMENTS:
            environment_rows = [
                row for row in selected if row["environment"] == environment
            ]
            environment_gates[environment] = base._effect_gate({
                metric: float(np.mean([
                    float(row[metric]) for row in environment_rows
                ]))
                for metric in metrics
            })
        seed_rows = _optimizer_seed_rows(replicate_rows, arm=arm)
        optimizer_seed_rows.extend(seed_rows)
        arm_rows.append({
            "arm": arm,
            **means,
            "replicate_count": len(selected),
            "optimizer_seed_count": len(seed_rows),
            "engineering_pass_count": sum(
                bool(row["engineering_pass"]) for row in selected
            ),
            "complete_effect_gate_count": sum(
                bool(row["complete_effect_gate"]) for row in selected
            ),
            "environment_complete_count": sum(
                bool(gate["complete"])
                for gate in environment_gates.values()
            ),
            "environment_gates": environment_gates,
            "mean_endpoint_margin": float(np.mean([
                means["normalized_episode_return"] - RETURN_THRESHOLD,
                *(
                    means[metric] - FREQUENCY_THRESHOLD
                    for metric in base.FREQUENCY_METRICS
                ),
            ])),
            "optimizer_seed_metric_ranges": {
                metric: [
                    float(min(row[metric] for row in seed_rows)),
                    float(max(row[metric] for row in seed_rows)),
                ]
                for metric in metrics
            },
        })

    arm_rows.sort(
        key=lambda row: (
            int(row["environment_complete_count"]),
            int(row["complete_effect_gate_count"]),
            int(row["engineering_pass_count"]),
            float(row["mean_endpoint_margin"]),
        ),
        reverse=True,
    )
    primary = next(
        row for row in arm_rows if row["arm"] == spec.PRIMARY_CANDIDATE_ARM
    )
    expected_replicates = len(spec.ENVIRONMENTS) * len(spec.OPTIMIZER_SEEDS)
    primary_ready = bool(
        int(primary["environment_complete_count"]) == len(spec.ENVIRONMENTS)
        and int(primary["engineering_pass_count"]) == expected_replicates
    )
    status = (
        "primary_mechanism_ready_for_fresh_multiseed_development"
        if primary_ready else "primary_mechanism_not_ready"
    )
    all_cell_paths = [
        run / "anchors" / environment / f"replicate_{optimizer_seed}"
        for environment in spec.ENVIRONMENTS
        for optimizer_seed in spec.OPTIMIZER_SEEDS
    ] + [
        base._cell_dir(
            run,
            environment=environment,
            arm=arm,
            optimizer_seed=optimizer_seed,
        )
        for environment in spec.ENVIRONMENTS
        for optimizer_seed in spec.OPTIMIZER_SEEDS
        for arm in spec.ARMS
    ]
    return {
        "analysis_version": ANALYSIS_VERSION,
        "status": status,
        "evidence_role": "mechanism_screen_development_not_confirmation",
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_algorithm_revision": spec.FROZEN_ALGORITHM_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "statistical_unit": "optimizer_seed",
        "optimizer_seed_count": len(spec.OPTIMIZER_SEEDS),
        "heldout_paths_are_not_replicates": True,
        "return_noninferiority_threshold": RETURN_THRESHOLD,
        "frequency_log_reduction_threshold": FREQUENCY_THRESHOLD,
        "primary_candidate_arm": spec.PRIMARY_CANDIDATE_ARM,
        "primary_ready": primary_ready,
        "arm_ranking": arm_rows,
        "replicate_rows": replicate_rows,
        "optimizer_seed_rows": optimizer_seed_rows,
        "path_rows": path_rows,
        "input_sha256": base._input_sha256(run, all_cell_paths),
        "claim_boundary": (
            "Three optimizer seeds are sufficient only to reject or nominate "
            "a mechanism. Environment and held-out rollout paths are paired "
            "observations, not independent replicates. Ranges are reported "
            "instead of an underpowered confidence interval. A nominated "
            "mechanism requires a frozen larger multiseed development screen "
            "and fresh confirmation seeds."
        ),
    }


def write_analysis(output_dir: Path, decision: dict[str, Any]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    decision_path = output / "decision.json"
    if decision_path.exists() and decision_path.read_text(
        encoding="utf-8"
    ) != rendered:
        raise RuntimeError("existing v14.17 decision differs")
    decision_path.write_text(rendered, encoding="utf-8")
    base._write_csv(output / "arm_ranking.csv", decision["arm_ranking"])
    base._write_csv(
        output / "replicate_rows.csv", decision["replicate_rows"]
    )
    base._write_csv(
        output / "optimizer_seed_rows.csv", decision["optimizer_seed_rows"]
    )
    base._write_csv(output / "path_rows.csv", decision["path_rows"])
    lines = [
        "# MuJoCo v14.17 Native Primal-Dual/CVaR Mechanism Screen",
        "",
        f"- Status: `{decision['status']}`",
        f"- Primary arm: `{decision['primary_candidate_arm']}`",
        f"- Optimizer seeds: `{decision['optimizer_seed_count']}`",
        "- Statistical unit: optimizer seed; held-out paths are paired only.",
        "",
        "| rank | arm | env complete | cell complete | engineering | return | mean margin |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(decision["arm_ranking"], start=1):
        lines.append(
            f"| {rank} | {row['arm']} | "
            f"{row['environment_complete_count']}/{len(spec.ENVIRONMENTS)} | "
            f"{row['complete_effect_gate_count']}/{row['replicate_count']} | "
            f"{row['engineering_pass_count']}/{row['replicate_count']} | "
            f"{row['normalized_episode_return']:.6f} | "
            f"{row['mean_endpoint_margin']:.6f} |"
        )
    lines.extend(("", decision["claim_boundary"], ""))
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    decision = analyze(args.run_dir)
    write_analysis(args.output_dir, decision)
    print(
        f"mujoco_v14_17_mechanism status={decision['status']} "
        f"primary_ready={decision['primary_ready']}"
    )


if __name__ == "__main__":
    main()
