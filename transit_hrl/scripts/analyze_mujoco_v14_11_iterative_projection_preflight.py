#!/usr/bin/env python3
"""Audit the scoped single-seed MuJoCo v14.11 preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import mujoco_v14_11_iterative_projection_screen_spec as spec  # noqa: E402
from scripts.analyze_mujoco_v14_11_iterative_projection_screen import (  # noqa: E402
    MAX_METRICS,
    METRICS,
    TRACE_KEYS,
    _actor_rms_difference,
    _read_rows,
)


ANALYSIS_VERSION = "mujoco_v14_11_iterative_projection_preflight_analysis_v1"
FREQUENCY_METRICS = (
    "LowerLFDriftAbs",
    "RawLowerLFDriftAbs",
    "LatentLowerLFDriftAbs",
    "UpperHFPowerAbs",
    "LatentUpperHFPowerAbs",
)


def _cell_dir(
    run_dir: Path, *, environment: str, arm: str, optimizer_seed: int
) -> Path:
    return (
        run_dir / "cells" / environment / arm
        / f"replicate_{int(optimizer_seed)}"
    )


def _load_cell(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    summary = json.loads((path / "cell_summary.json").read_text(encoding="utf-8"))
    history = json.loads((path / "training_history.json").read_text(encoding="utf-8"))
    rows = _read_rows(path / "evaluation_rows.csv")
    if not isinstance(summary, dict) or not isinstance(history, list) or not history:
        raise ValueError(f"invalid v14.11 preflight cell: {path}")
    return summary, history, rows


def _mode_payload(rows: list[dict[str, str]], mode: str) -> dict[str, Any]:
    selected = sorted(
        (row for row in rows if row["disturbance_mode"] == mode),
        key=lambda row: int(row["seed"]),
    )
    if len(selected) != len(spec.DEVELOPMENT_EVALUATION_SEEDS):
        raise ValueError(f"v14.11 preflight mode is incomplete: {mode}")
    payload = {
        metric: (
            float(max(float(row[metric]) for row in selected))
            if metric in MAX_METRICS
            else float(np.mean([float(row[metric]) for row in selected]))
        )
        for metric in METRICS
    }
    payload["traces"] = {
        trace: tuple(str(row[trace]) for row in selected)
        for trace in TRACE_KEYS
    }
    return payload


def _projection_diagnostics(
    history: list[dict[str, Any]], arm_spec: dict[str, object]
) -> dict[str, Any]:
    by_level: dict[str, dict[str, Any]] = {}
    passes = []
    total_accepted_steps = 0
    total_multistep_updates = 0
    total_reward_budget_violations = 0
    all_reduction_fractions: list[float] = []
    for level in ("upper", "lower"):
        prefix = f"{level}_deployment_frequency"
        active = float(arm_spec[f"{prefix}_dual_lr"]) > 0.0
        diagnostic_rows = [
            row for row in history
            if float(row.get(f"{prefix}_enabled", 0.0)) > 0.5
        ]
        violating_rows = [
            row for row in diagnostic_rows
            if float(row.get(
                f"{prefix}_projection_target_reached_before", 0.0
            )) < 0.5
        ]
        attempted_rows = [
            row for row in history
            if float(row.get(
                f"{prefix}_projection_steps_attempted",
                row.get(f"{prefix}_guard_attempted", 0.0),
            )) > 0.5
        ]
        accepted_rows = [
            row for row in attempted_rows
            if float(row.get(
                f"{prefix}_projection_steps_accepted",
                row.get(f"{prefix}_guard_accepted", 0.0),
            )) > 0.5
        ]
        attempted_steps = int(round(sum(float(row.get(
            f"{prefix}_projection_steps_attempted",
            row.get(f"{prefix}_guard_attempted", 0.0),
        )) for row in attempted_rows)))
        accepted_steps = int(round(sum(float(row.get(
            f"{prefix}_projection_steps_accepted",
            row.get(f"{prefix}_guard_accepted", 0.0),
        )) for row in accepted_rows)))
        multistep_updates = sum(
            float(row.get(
                f"{prefix}_projection_steps_accepted",
                row.get(f"{prefix}_guard_accepted", 0.0),
            )) >= spec.MINIMUM_ITERATIVE_ACCEPTED_STEPS
            for row in accepted_rows
        )
        reductions = [
            float(row[f"{prefix}_power_before"])
            - float(row[f"{prefix}_power_after"])
            for row in accepted_rows
        ]
        reduction_fractions = [
            reduction / max(float(row[f"{prefix}_power_before"]), 1e-12)
            for row, reduction in zip(accepted_rows, reductions, strict=True)
        ]
        reward_tolerance = float(
            arm_spec[f"{prefix}_reward_tolerance"]
        )
        reward_budget_violations = sum(
            float(row.get(f"{prefix}_guard_reward_loss_delta", 0.0))
            > reward_tolerance + 1e-8
            for row in diagnostic_rows
        )
        target_reached_after = sum(
            float(row.get(
                f"{prefix}_projection_target_reached_after", 0.0
            )) > 0.5
            for row in diagnostic_rows
        )
        requested_steps_match = bool(diagnostic_rows) and all(
            int(round(float(row.get(
                f"{prefix}_projection_steps_requested", -1.0
            )))) == int(arm_spec[f"{prefix}_max_projection_steps"])
            for row in diagnostic_rows
        )
        cumulative_budget_match = bool(diagnostic_rows) and all(
            abs(float(row.get(
                f"{prefix}_projection_reward_tolerance", float("nan")
            )) - reward_tolerance) <= 1e-15
            for row in diagnostic_rows
        )
        feasibility_or_correction = bool(
            diagnostic_rows
            and (
                not violating_rows
                or (
                    attempted_rows
                    and accepted_rows
                    and float(np.mean(reductions)) > 0.0
                )
            )
        )
        passed = bool(
            not active
            or (
                feasibility_or_correction
                and requested_steps_match
                and cumulative_budget_match
                and reward_budget_violations
                <= spec.MAXIMUM_PROJECTION_REWARD_BUDGET_VIOLATIONS
            )
        )
        passes.append(passed)
        total_accepted_steps += accepted_steps
        total_multistep_updates += multistep_updates
        total_reward_budget_violations += reward_budget_violations
        all_reduction_fractions.extend(reduction_fractions)
        by_level[level] = {
            "active": active,
            "diagnostic_update_count": len(diagnostic_rows),
            "violating_update_count": len(violating_rows),
            "guard_attempted_update_count": len(attempted_rows),
            "guard_accepted_update_count": len(accepted_rows),
            "projection_steps_attempted": attempted_steps,
            "projection_steps_accepted": accepted_steps,
            "multistep_update_count": multistep_updates,
            "accepted_power_reduction_mean": float(
                np.mean(reductions) if reductions else 0.0
            ),
            "accepted_power_reduction_fraction_mean": float(
                np.mean(reduction_fractions) if reduction_fractions else 0.0
            ),
            "target_reached_after_count": target_reached_after,
            "reward_budget_violation_count": reward_budget_violations,
            "requested_steps_match": requested_steps_match,
            "cumulative_reward_budget_match": cumulative_budget_match,
            "pass": passed,
        }
    iterative = int(arm_spec[
        "upper_deployment_frequency_max_projection_steps"
    ]) > 1
    iterative_mechanism_pass = bool(
        not iterative
        or (
            total_accepted_steps >= spec.MINIMUM_ITERATIVE_ACCEPTED_STEPS
            and total_multistep_updates
            >= spec.MINIMUM_ITERATIVE_MULTISTEP_UPDATES
        )
    )
    return {
        "pass": bool(
            all(passes)
            and iterative_mechanism_pass
            and total_reward_budget_violations
            <= spec.MAXIMUM_PROJECTION_REWARD_BUDGET_VIOLATIONS
        ),
        "iterative": iterative,
        "iterative_mechanism_pass": iterative_mechanism_pass,
        "projection_steps_accepted": total_accepted_steps,
        "multistep_update_count": total_multistep_updates,
        "reward_budget_violation_count": total_reward_budget_violations,
        "accepted_power_reduction_fraction_mean": float(
            np.mean(all_reduction_fractions)
            if all_reduction_fractions else 0.0
        ),
        "by_level": by_level,
    }


def _input_sha256(paths: list[Path], *, root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = str(path.relative_to(root)).encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def analyze_preflight(
    run_dir: Path, *, environment: str, optimizer_seed: int
) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    if environment not in spec.ENVIRONMENTS:
        raise ValueError(f"invalid v14.11 preflight environment: {environment}")
    if int(optimizer_seed) not in spec.OPTIMIZER_SEEDS:
        raise ValueError("invalid v14.11 preflight optimizer seed")
    preregistration_path = run / "preregistration.json"
    manifest_path = run / "merged" / "cell_manifest.json"
    sync_path = run / "merged" / "run_scoped_result_sync.json"
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sync = json.loads(sync_path.read_text(encoding="utf-8"))
    expected_count = len(spec.ARMS) + 1
    identity_pass = bool(
        preregistration.get("status")
        == "frozen_before_v14_11_iterative_projection_outcome_access"
        and preregistration.get("development_protocol_version")
        == spec.DEVELOPMENT_PROTOCOL_VERSION
        and preregistration.get("frozen_algorithm_revision")
        == spec.FROZEN_ALGORITHM_REVISION
        and preregistration.get("frozen_source_manifest_sha256")
        == spec.FROZEN_SOURCE_MANIFEST_SHA256
        and manifest.get("status") == "development_scope_complete_unanalyzed"
        and manifest.get("full_screen_scope") is False
        and manifest.get("environments") == [environment]
        and manifest.get("optimizer_seeds") == [int(optimizer_seed)]
        and set(manifest.get("arms", [])) == set(spec.ARMS)
        and set(manifest.get("phases", [])) == {"anchor", "continuation"}
        and int(manifest.get("cell_count", -1)) == expected_count
        and sync.get("status") == "run_scoped_result_sync_complete"
        and int(sync.get("cell_count", -1)) == expected_count
    )
    if not identity_pass:
        raise ValueError("v14.11 preflight scope or frozen identity mismatch")

    required_paths = [preregistration_path, manifest_path, sync_path]
    anchor_path = (
        run / "anchors" / environment / f"replicate_{int(optimizer_seed)}"
    )
    required_paths.extend(
        anchor_path / name for name in (
            "cell_summary.json", "training_history.json",
            "evaluation_rows.csv", "checkpoint.pt",
        )
    )
    anchor_summary, _, _ = _load_cell(anchor_path)
    if not (
        anchor_summary.get("protocol_version")
        == spec.FROZEN_CORE_PROTOCOL_VERSION
        and anchor_summary.get("environment") == environment
        and int(anchor_summary.get("optimizer_seed", -1))
        == int(optimizer_seed)
        and anchor_summary.get("code_revision")
        == spec.FROZEN_ALGORITHM_REVISION
        and anchor_summary.get("source_manifest_sha256")
        == spec.FROZEN_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("v14.11 preflight anchor identity mismatch")
    cells: dict[str, tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]] = {}
    for arm in spec.ARMS:
        path = _cell_dir(
            run, environment=environment, arm=arm, optimizer_seed=optimizer_seed
        )
        required_paths.extend(
            path / name for name in (
                "cell_summary.json", "training_history.json",
                "evaluation_rows.csv", "checkpoint.pt",
            )
        )
        cell = _load_cell(path)
        summary = cell[0]
        if not (
            summary.get("protocol_version") == spec.FROZEN_CORE_PROTOCOL_VERSION
            and summary.get("environment") == environment
            and int(summary.get("optimizer_seed", -1)) == int(optimizer_seed)
            and summary.get("code_revision") == spec.FROZEN_ALGORITHM_REVISION
            and summary.get("source_manifest_sha256")
            == spec.FROZEN_SOURCE_MANIFEST_SHA256
        ):
            raise ValueError(f"v14.11 preflight cell identity mismatch: {arm}")
        continuation = dict(summary.get("paired_checkpoint_continuation") or {})
        if not (
            continuation.get("enabled") is True
            and continuation.get("checkpoint_file_sha256")
            == anchor_summary.get("checkpoint_file_sha256")
            and continuation.get("checkpoint_parameter_sha256")
            == anchor_summary.get("frozen_parameter_sha256")
        ):
            raise ValueError(f"v14.11 preflight anchor provenance mismatch: {arm}")
        if (
            spec.ARMS[arm]["checkpoint_score_mode"]
            == "paired_relative_frequency_feasibility_first"
        ):
            baseline = dict(
                summary.get("paired_relative_checkpoint_baseline") or {}
            )
            expected_paths = (
                len(spec.CONTINUATION_SELECTION_SEEDS)
                * len(spec.TRAINING_DISTURBANCE_MODES)
            )
            if not (
                baseline.get("enabled") is True
                and int(baseline.get("row_count", -1)) == expected_paths
                and int(baseline.get("heldout_rows_used", -1)) == 0
                and baseline.get("parameter_sha256")
                == continuation.get("checkpoint_parameter_sha256")
            ):
                raise ValueError(
                    f"v14.11 preflight paired baseline mismatch: {arm}"
                )
        cells[arm] = cell

    modes = spec.EVALUATION_DISTURBANCE_MODES
    mode_payloads = {
        arm: {mode: _mode_payload(cell[2], mode) for mode in modes}
        for arm, cell in cells.items()
    }
    calibration_actor = _actor_rms_difference(
        _cell_dir(
            run,
            environment=environment,
            arm=spec.CALIBRATION_ARM,
            optimizer_seed=optimizer_seed,
        ) / "checkpoint.pt",
        _cell_dir(
            run,
            environment=environment,
            arm=spec.BASE_CONTROL_ARM,
            optimizer_seed=optimizer_seed,
        ) / "checkpoint.pt",
    )
    calibration_conditions = []
    for mode in modes:
        baseline = mode_payloads[spec.BASE_CONTROL_ARM][mode]
        candidate = mode_payloads[spec.CALIBRATION_ARM][mode]
        exact_reward = bool(
            abs(candidate["episode_return"] - baseline["episode_return"])
            <= spec.MAXIMUM_ABSOLUTE_RETURN_DIFFERENCE
        )
        exact_traces = bool(candidate["traces"] == baseline["traces"])
        lower_reductions = [
            1.0 - candidate[metric] / max(baseline[metric], 1e-12)
            for metric in ("LowerLFDriftAbs", "RawLowerLFDriftAbs")
        ]
        upper_reduction = (
            1.0 - candidate["UpperHFPowerAbs"]
            / max(baseline["UpperHFPowerAbs"], 1e-12)
        )
        passed = bool(
            exact_reward
            and exact_traces
            and min(lower_reductions)
            >= spec.MINIMUM_PROJECTION_LOWER_REDUCTION_FRACTION
            and upper_reduction
            >= spec.MINIMUM_PROJECTION_UPPER_REDUCTION_FRACTION
        )
        calibration_conditions.append({
            "disturbance_mode": mode,
            "exact_reward": exact_reward,
            "exact_traces": exact_traces,
            "minimum_lower_reduction_fraction": float(min(lower_reductions)),
            "upper_reduction_fraction": float(upper_reduction),
            "pass": passed,
        })
    calibration_pass = bool(
        calibration_actor["combined"] <= spec.MAXIMUM_CALIBRATION_ACTOR_RMS
        and all(row["pass"] for row in calibration_conditions)
    )

    arm_status: dict[str, dict[str, Any]] = {}
    for arm in spec.LEARNED_ARMS:
        arm_spec = spec.ARMS[arm]
        summary, history, _ = cells[arm]
        actor = _actor_rms_difference(
            _cell_dir(
                run, environment=environment, arm=arm,
                optimizer_seed=optimizer_seed,
            ) / "checkpoint.pt",
            _cell_dir(
                run, environment=environment, arm=spec.MATCHED_COMPARATOR_ARM,
                optimizer_seed=optimizer_seed,
            ) / "checkpoint.pt",
        )
        projection = _projection_diagnostics(history, arm_spec)
        conditions = []
        for mode in modes:
            baseline = mode_payloads[spec.MATCHED_COMPARATOR_ARM][mode]
            candidate = mode_payloads[arm][mode]
            reward_margin = (
                spec.RETURN_NONINFERIORITY_MARGIN_FRACTION
                * max(abs(baseline["episode_return"]), 1.0)
            )
            reward_difference = (
                candidate["episode_return"] - baseline["episode_return"]
            )
            reductions = {
                metric: float(
                    1.0 - candidate[metric] / max(baseline[metric], 1e-12)
                )
                for metric in FREQUENCY_METRICS
            }
            targets = {
                metric: (
                    float(arm_spec[
                        "lower_deployment_frequency_reference_reduction_fraction"
                    ])
                    if metric.startswith(("Lower", "RawLower", "LatentLower"))
                    else float(arm_spec[
                        "upper_deployment_frequency_reference_reduction_fraction"
                    ])
                )
                for metric in FREQUENCY_METRICS
            }
            reward_floor_pass = bool(reward_difference >= -reward_margin)
            frequency_pass = bool(all(
                reductions[metric] + 1e-12 >= targets[metric]
                for metric in FREQUENCY_METRICS
            ))
            conditions.append({
                "disturbance_mode": mode,
                "reward_difference": float(reward_difference),
                "reward_noninferiority_margin": float(reward_margin),
                "reward_floor_pass": reward_floor_pass,
                "frequency_reduction_fraction": reductions,
                "registered_reduction_fraction": targets,
                "frequency_target_pass": frequency_pass,
                "pass": bool(reward_floor_pass and frequency_pass),
            })
        trained = bool(
            int(summary.get("selected_checkpoint_iteration", -2))
            >= spec.ANALYSIS_TRAINED_CHECKPOINT_MINIMUM_ITERATION
        )
        actor_changed = bool(
            actor["combined"] >= spec.MINIMUM_LEARNED_PARAMETER_RMS
        )
        action_changed = bool(any(
            mode_payloads[arm][mode]["traces"]["ExecutedActionTraceSHA256"]
            != mode_payloads[spec.MATCHED_COMPARATOR_ARM][mode]["traces"][
                "ExecutedActionTraceSHA256"
            ]
            for mode in modes
        ))
        strict_reward_signal = bool(any(
            row["reward_difference"] > 0.0 for row in conditions
        ))
        base_preflight_pass = bool(
            trained
            and actor_changed
            and action_changed
            and projection["pass"]
            and strict_reward_signal
            and all(row["pass"] for row in conditions)
        )
        arm_status[arm] = {
            "selected_checkpoint_iteration": int(
                summary.get("selected_checkpoint_iteration", -2)
            ),
            "trained_checkpoint_pass": trained,
            "paired_actor_rms": actor,
            "actor_changed_pass": actor_changed,
            "action_changed_pass": action_changed,
            "deployment_projection": projection,
            "strict_reward_signal_pass": strict_reward_signal,
            "all_condition_gates_pass": bool(
                all(row["pass"] for row in conditions)
            ),
            "base_preflight_pass": base_preflight_pass,
            "conditions": conditions,
        }

    legacy_reduction = float(
        arm_status[spec.LEGACY_ONE_STEP_ARM]["deployment_projection"][
            "accepted_power_reduction_fraction_mean"
        ]
    )
    eligible = []
    for arm, status in arm_status.items():
        iterative = arm in spec.ITERATIVE_ARMS
        reduction = float(status["deployment_projection"][
            "accepted_power_reduction_fraction_mean"
        ])
        gain = reduction - legacy_reduction
        gain_pass = bool(
            iterative
            and gain + 1e-12
            >= spec.MINIMUM_ITERATIVE_REDUCTION_GAIN_OVER_ONE_STEP
        )
        passed = bool(status["base_preflight_pass"] and gain_pass)
        status.update({
            "iterative_candidate": iterative,
            "one_step_reduction_fraction": legacy_reduction,
            "iterative_reduction_gain_over_one_step": float(gain),
            "iterative_reduction_gain_pass": gain_pass,
            "preflight_pass": passed,
        })
        if passed:
            eligible.append(arm)

    selected = max(
        eligible,
        key=lambda arm: min(
            min(row["frequency_reduction_fraction"].values())
            for row in arm_status[arm]["conditions"]
        ),
        default=None,
    )
    return {
        "analysis_version": ANALYSIS_VERSION,
        "status": (
            "expand_to_multiseed_screen"
            if calibration_pass and selected is not None
            else "do_not_expand"
        ),
        "evidence_role": "single_optimizer_seed_mechanism_preflight_no_ci",
        "environment": environment,
        "optimizer_seed": int(optimizer_seed),
        "calibration_pass": calibration_pass,
        "calibration_actor_rms": calibration_actor,
        "calibration_conditions": calibration_conditions,
        "eligible_arms": eligible,
        "selected_arm": selected,
        "arm_status": arm_status,
        "input_sha256": _input_sha256(required_paths, root=run),
        "claim_boundary": (
            "This single-optimizer-seed preflight can reject a broken mechanism "
            "or authorize a larger development screen. It cannot support a "
            "performance, robustness, or statistical significance claim."
        ),
    }


def write_analysis(output_dir: Path, decision: dict[str, Any]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# MuJoCo v14.11 Iterative-Projection Preflight",
        "",
        f"- Status: `{decision['status']}`",
        f"- Calibration pass: `{decision['calibration_pass']}`",
        f"- Selected arm: `{decision['selected_arm']}`",
        "- Evidence role: single optimizer seed, mechanism preflight, no CI.",
        "",
        "| arm | trained | actor | action | projection | conditions | reward signal | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, status in decision["arm_status"].items():
        lines.append(
            f"| {arm} | {status['trained_checkpoint_pass']} | "
            f"{status['actor_changed_pass']} | {status['action_changed_pass']} | "
            f"{status['deployment_projection']['pass']} | "
            f"{status['all_condition_gates_pass']} | "
            f"{status['strict_reward_signal_pass']} | "
            f"{status['preflight_pass']} |"
        )
    lines.extend(("", decision["claim_boundary"], ""))
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--environment", default="HalfCheetah-v5")
    parser.add_argument("--optimizer-seed", type=int, required=True)
    args = parser.parse_args()
    decision = analyze_preflight(
        args.run_dir,
        environment=args.environment,
        optimizer_seed=args.optimizer_seed,
    )
    write_analysis(args.output_dir, decision)
    print(
        f"mujoco_v14_11_preflight status={decision['status']} "
        f"selected={decision['selected_arm']}"
    )


if __name__ == "__main__":
    main()
