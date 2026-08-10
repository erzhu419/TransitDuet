#!/usr/bin/env python3
"""Audit the scoped single-seed MuJoCo v14.14 preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import mujoco_v14_14_closed_loop_actor_guard_screen_spec as spec  # noqa: E402
from scripts.analyze_mujoco_v14_11_iterative_projection_screen import (  # noqa: E402
    MAX_METRICS,
    METRICS,
    TRACE_KEYS,
    _actor_rms_difference,
)
from scripts.submit_mujoco_v14_14_closed_loop_actor_guard_screen_scheduleurm import (  # noqa: E402
    _closed_loop_guard_contract_valid,
)


ANALYSIS_VERSION = "mujoco_v14_14_closed_loop_actor_guard_preflight_analysis_v1"
FREQUENCY_METRICS = (
    "LowerLFDriftAbs",
    "RawLowerLFDriftAbs",
    "LatentLowerLFDriftAbs",
    "UpperHFPowerAbs",
    "LatentUpperHFPowerAbs",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = (
        len(spec.DEVELOPMENT_EVALUATION_SEEDS)
        * len(spec.EVALUATION_DISTURBANCE_MODES)
    )
    if len(rows) != expected:
        raise ValueError(
            f"v14.14 cell has {len(rows)} rows; expected {expected}: {path}"
        )
    keys = [(str(row["disturbance_mode"]), int(row["seed"])) for row in rows]
    expected_keys = {
        (mode, int(seed))
        for mode in spec.EVALUATION_DISTURBANCE_MODES
        for seed in spec.DEVELOPMENT_EVALUATION_SEEDS
    }
    if len(set(keys)) != len(keys) or set(keys) != expected_keys:
        raise ValueError(f"v14.14 evaluation path registry mismatch: {path}")
    return rows


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
        raise ValueError(f"invalid v14.14 preflight cell: {path}")
    return summary, history, rows


def _mode_payload(rows: list[dict[str, str]], mode: str) -> dict[str, Any]:
    selected = sorted(
        (row for row in rows if row["disturbance_mode"] == mode),
        key=lambda row: int(row["seed"]),
    )
    if len(selected) != len(spec.DEVELOPMENT_EVALUATION_SEEDS):
        raise ValueError(f"v14.14 preflight mode is incomplete: {mode}")
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
    passes: list[bool] = []
    total_accepted_steps = 0
    total_reward_budget_violations = 0
    total_group_reward_budget_violations = 0
    total_groups_target_reached = 0
    all_robust_excess_reductions: list[float] = []
    expected_groupwise = bool(
        arm_spec["deployment_frequency_groupwise_robust"]
    )
    expected_replay = bool(
        arm_spec["deployment_frequency_anchor_state_replay"]
    )
    expected_group_count = (
        spec.MINIMUM_REPLAY_GROUP_COUNT
        if expected_replay else (
            spec.MINIMUM_GROUPWISE_GROUP_COUNT
            if expected_groupwise else 1
        )
    )
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
        robust_excess_reductions = [
            float(row[f"{prefix}_normalized_signed_excess_before"])
            - float(row[f"{prefix}_normalized_signed_excess_after"])
            for row in accepted_rows
        ]
        reward_tolerance = float(
            arm_spec[f"{prefix}_reward_tolerance"]
        )
        reward_budget_violations = sum(
            float(row.get(f"{prefix}_guard_reward_loss_delta", 0.0))
            > reward_tolerance + 1e-8
            for row in diagnostic_rows
        )
        group_reward_budget_violations = int(round(sum(
            float(row.get(
                f"{prefix}_group_reward_budget_violation_count", 0.0
            ))
            for row in diagnostic_rows
        )))
        groups_target_reached = int(round(sum(
            float(row.get(
                f"{prefix}_groups_target_reached_after", 0.0
            ))
            for row in diagnostic_rows
        )))
        group_count_match = bool(diagnostic_rows) and all(
            int(round(float(row.get(
                f"{prefix}_group_count", -1.0
            )))) == expected_group_count
            for row in diagnostic_rows
        )
        reward_group_count_match = bool(diagnostic_rows) and all(
            int(round(float(row.get(
                f"{prefix}_reward_guard_group_count", -1.0
            )))) == (
                spec.EXPECTED_REWARD_GUARD_GROUP_COUNT
                if expected_groupwise else 1
            )
            for row in diagnostic_rows
        )
        replay_flag_match = bool(diagnostic_rows) and all(
            bool(round(float(row.get(
                f"{prefix}_anchor_state_replay_enabled", -1.0
            )))) is expected_replay
            for row in diagnostic_rows
        )
        groupwise_flag_match = bool(diagnostic_rows) and all(
            bool(round(float(row.get(
                f"{prefix}_groupwise_robust", -1.0
            )))) is expected_groupwise
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
                    and float(np.mean(robust_excess_reductions)) > 0.0
                )
            )
        )
        groupwise_mechanism_pass = bool(
            not expected_groupwise
            or (
                group_count_match
                and reward_group_count_match
                and replay_flag_match
                and groupwise_flag_match
                and accepted_steps >= spec.MINIMUM_GROUPWISE_ACCEPTED_STEPS
                and groups_target_reached
                >= spec.MINIMUM_GROUPWISE_TARGET_REACHED_GROUPS
                and group_reward_budget_violations
                <= spec.MAXIMUM_GROUP_REWARD_BUDGET_VIOLATIONS
            )
        )
        passed = bool(
            not active
            or (
                feasibility_or_correction
                and requested_steps_match
                and cumulative_budget_match
                and group_count_match
                and reward_group_count_match
                and replay_flag_match
                and groupwise_flag_match
                and groupwise_mechanism_pass
                and reward_budget_violations
                <= spec.MAXIMUM_PROJECTION_REWARD_BUDGET_VIOLATIONS
            )
        )
        passes.append(passed)
        total_accepted_steps += accepted_steps
        total_reward_budget_violations += reward_budget_violations
        total_group_reward_budget_violations += (
            group_reward_budget_violations
        )
        total_groups_target_reached += groups_target_reached
        all_robust_excess_reductions.extend(robust_excess_reductions)
        by_level[level] = {
            "active": active,
            "diagnostic_update_count": len(diagnostic_rows),
            "violating_update_count": len(violating_rows),
            "guard_attempted_update_count": len(attempted_rows),
            "guard_accepted_update_count": len(accepted_rows),
            "projection_steps_attempted": attempted_steps,
            "projection_steps_accepted": accepted_steps,
            "accepted_robust_excess_reduction_mean": float(
                np.mean(robust_excess_reductions)
                if robust_excess_reductions else 0.0
            ),
            "reward_budget_violation_count": reward_budget_violations,
            "group_reward_budget_violation_count": (
                group_reward_budget_violations
            ),
            "groups_target_reached_total": groups_target_reached,
            "group_count_match": group_count_match,
            "reward_group_count_match": reward_group_count_match,
            "anchor_state_replay_flag_match": replay_flag_match,
            "groupwise_flag_match": groupwise_flag_match,
            "groupwise_mechanism_pass": groupwise_mechanism_pass,
            "requested_steps_match": requested_steps_match,
            "cumulative_reward_budget_match": cumulative_budget_match,
            "pass": passed,
        }
    return {
        "pass": bool(
            all(passes)
            and total_reward_budget_violations
            <= spec.MAXIMUM_PROJECTION_REWARD_BUDGET_VIOLATIONS
            and total_group_reward_budget_violations
            <= spec.MAXIMUM_GROUP_REWARD_BUDGET_VIOLATIONS
        ),
        "groupwise_robust": expected_groupwise,
        "anchor_state_replay": expected_replay,
        "expected_frequency_group_count": expected_group_count,
        "projection_steps_accepted": total_accepted_steps,
        "reward_budget_violation_count": total_reward_budget_violations,
        "group_reward_budget_violation_count": (
            total_group_reward_budget_violations
        ),
        "groups_target_reached_total": total_groups_target_reached,
        "accepted_robust_excess_reduction_mean": float(
            np.mean(all_robust_excess_reductions)
            if all_robust_excess_reductions else 0.0
        ),
        "by_level": by_level,
    }


def _trust_region_diagnostics(
    history: list[dict[str, Any]], arm_spec: dict[str, object]
) -> dict[str, Any]:
    expected = bool(
        arm_spec["deployment_frequency_ppo_trust_region"]
    )
    expected_groups = (
        spec.MINIMUM_REPLAY_GROUP_COUNT
        if bool(arm_spec["deployment_frequency_anchor_state_replay"])
        else spec.MINIMUM_GROUPWISE_GROUP_COUNT
    )
    by_level: dict[str, dict[str, Any]] = {}
    passes: list[bool] = []
    total_accepted = 0
    total_final_reward_violations = 0
    for level in ("upper", "lower"):
        prefix = f"{level}_deployment_frequency_ppo_guard"
        rows = [row for row in history if int(row.get("iteration", -1)) >= 0]
        enabled_match = bool(rows) and all(
            bool(round(float(row.get(f"{prefix}_enabled", -1.0))))
            is expected
            for row in rows
        )
        accepted = sum(
            float(row.get(f"{prefix}_step_fraction", 0.0)) > 0.0
            for row in rows
            if float(row.get(f"{prefix}_enabled", 0.0)) > 0.5
        )
        final_reward_violations = int(round(sum(
            float(row.get(
                f"{prefix}_group_reward_budget_violation_count", 0.0
            ))
            for row in rows
        )))
        group_count_match = bool(rows) and all(
            int(round(float(row.get(
                f"{prefix}_frequency_group_count", -1.0
            )))) == expected_groups
            for row in rows
        ) if expected else True
        reward_group_count_match = bool(rows) and all(
            int(round(float(row.get(
                f"{prefix}_reward_group_count", -1.0
            )))) == spec.EXPECTED_REWARD_GUARD_GROUP_COUNT
            for row in rows
        ) if expected else True
        feasibility_preserved = bool(rows) and all(
            float(row.get(
                f"{prefix}_frequency_excess_after", float("inf")
            ))
            <= max(
                float(row.get(
                    f"{prefix}_frequency_excess_before", float("inf")
                )),
                float(arm_spec[
                    f"{level}_deployment_frequency_target_tolerance"
                ]),
            ) + 1e-6
            for row in rows
        ) if expected else True
        passed = bool(
            enabled_match
            and (
                not expected
                or (
                    accepted >= spec.MINIMUM_TRUST_ACCEPTED_STEPS
                    and final_reward_violations == 0
                    and group_count_match
                    and reward_group_count_match
                    and feasibility_preserved
                )
            )
        )
        passes.append(passed)
        total_accepted += accepted
        total_final_reward_violations += final_reward_violations
        by_level[level] = {
            "expected": expected,
            "enabled_match": enabled_match,
            "accepted_update_count": accepted,
            "final_group_reward_violation_count": (
                final_reward_violations
            ),
            "frequency_group_count_match": group_count_match,
            "reward_group_count_match": reward_group_count_match,
            "feasibility_preserved": feasibility_preserved,
            "pass": passed,
        }
    return {
        "expected": expected,
        "accepted_update_count": total_accepted,
        "final_group_reward_violation_count": (
            total_final_reward_violations
        ),
        "by_level": by_level,
        "pass": all(passes),
    }


def _anchor_replay_diagnostics(
    summary: dict[str, Any], arm_spec: dict[str, object]
) -> dict[str, Any]:
    expected = bool(
        arm_spec["deployment_frequency_anchor_state_replay"]
    )
    enabled = bool(summary.get(
        "deployment_frequency_anchor_state_replay_enabled", False
    ))
    path_count = int(summary.get(
        "deployment_frequency_anchor_state_replay_path_count", -1
    ))
    contract = str(summary.get(
        "deployment_frequency_anchor_state_replay_contract", ""
    ))
    upper_transitions = int(summary.get(
        "deployment_frequency_anchor_state_replay_upper_transitions", -1
    ))
    lower_transitions = int(summary.get(
        "deployment_frequency_anchor_state_replay_lower_transitions", -1
    ))
    passed = bool(
        (
            enabled
            and path_count == spec.EXPECTED_ANCHOR_REPLAY_PATH_COUNT
            and contract
            == "deterministic_frozen_anchor_deployment_trajectory_v1"
            and upper_transitions > 0
            and lower_transitions > 0
        )
        if expected else (
            not enabled
            and path_count == 0
            and contract == "disabled"
            and upper_transitions == 0
            and lower_transitions == 0
        )
    )
    return {
        "expected": expected,
        "enabled": enabled,
        "path_count": path_count,
        "contract": contract,
        "upper_transitions": upper_transitions,
        "lower_transitions": lower_transitions,
        "pass": passed,
    }


def _closed_loop_guard_diagnostics(
    summary: dict[str, Any],
    history: list[dict[str, Any]],
    arm_spec: dict[str, object],
) -> dict[str, Any]:
    expected = bool(
        arm_spec["deployment_frequency_closed_loop_trust_region"]
    )
    prefix = "deployment_frequency_closed_loop_guard_"
    contract_valid = _closed_loop_guard_contract_valid(
        summary, history, expected=expected
    )
    effective_updates = int(summary.get(
        f"{prefix}effective_update_count", 0
    ))
    selected_reward_violations = int(summary.get(
        f"{prefix}selected_reward_violation_count", 0 if not expected else -1
    ))
    selected_frequency_violations = int(summary.get(
        f"{prefix}selected_frequency_violation_count", 0 if not expected else -1
    ))
    initial_frequency_violations = int(summary.get(
        f"{prefix}initial_frequency_violation_count", 0 if not expected else -1
    ))
    passed = bool(
        contract_valid
        and (
            not expected
            or (
                effective_updates
                >= spec.MINIMUM_CLOSED_LOOP_EFFECTIVE_UPDATES
                and selected_reward_violations
                <= spec.MAXIMUM_CLOSED_LOOP_REWARD_VIOLATIONS
                and selected_frequency_violations
                <= spec.MAXIMUM_CLOSED_LOOP_FREQUENCY_VIOLATIONS
                and selected_frequency_violations
                <= initial_frequency_violations
            )
        )
    )
    return {
        "expected": expected,
        "contract_valid": contract_valid,
        "effective_update_count": effective_updates,
        "evaluation_count": int(summary.get(
            f"{prefix}evaluation_count", 0
        )),
        "initial_rank": summary.get(f"{prefix}initial_rank", []),
        "training_final_rank": summary.get(
            f"{prefix}training_final_rank", []
        ),
        "selected_rank": summary.get(f"{prefix}selected_rank", []),
        "initial_frequency_violation_count": initial_frequency_violations,
        "selected_reward_violation_count": selected_reward_violations,
        "selected_frequency_violation_count": selected_frequency_violations,
        "pass": passed,
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
        raise ValueError(f"invalid v14.14 preflight environment: {environment}")
    if int(optimizer_seed) not in spec.OPTIMIZER_SEEDS:
        raise ValueError("invalid v14.14 preflight optimizer seed")
    preregistration_path = run / "preregistration.json"
    manifest_path = run / "merged" / "cell_manifest.json"
    sync_path = run / "merged" / "run_scoped_result_sync.json"
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sync = json.loads(sync_path.read_text(encoding="utf-8"))
    expected_count = len(spec.ARMS) + 1
    identity_pass = bool(
        preregistration.get("status")
        == "frozen_before_v14_14_closed_loop_actor_guard_outcome_access"
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
        raise ValueError("v14.14 preflight scope or frozen identity mismatch")

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
        raise ValueError("v14.14 preflight anchor identity mismatch")
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
            raise ValueError(f"v14.14 preflight cell identity mismatch: {arm}")
        continuation = dict(summary.get("paired_checkpoint_continuation") or {})
        if not (
            continuation.get("enabled") is True
            and continuation.get("checkpoint_file_sha256")
            == anchor_summary.get("checkpoint_file_sha256")
            and continuation.get("checkpoint_parameter_sha256")
            == anchor_summary.get("frozen_parameter_sha256")
        ):
            raise ValueError(f"v14.14 preflight anchor provenance mismatch: {arm}")
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
                    f"v14.14 preflight paired baseline mismatch: {arm}"
                )
            selected_diagnostics = dict(
                summary.get("selected_checkpoint_diagnostics") or {}
            )
            checkpoint_rows = [
                row for row in cell[1]
                if bool(row.get("checkpoint_evaluation_performed", False))
            ]
            if not (
                int(selected_diagnostics.get("constraint_count", -1))
                == len(spec.TRAINING_DISTURBANCE_MODES) * 6
                and isinstance(
                    selected_diagnostics.get("worst_constraint"), dict
                )
                and checkpoint_rows
                and all(
                    int(dict(row.get(
                        "checkpoint_selection_diagnostics"
                    ) or {}).get("constraint_count", -1))
                    == len(spec.TRAINING_DISTURBANCE_MODES) * 6
                    for row in checkpoint_rows
                )
            ):
                raise ValueError(
                    f"v14.14 checkpoint diagnostics are incomplete: {arm}"
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
        trust_region = _trust_region_diagnostics(history, arm_spec)
        anchor_replay = _anchor_replay_diagnostics(summary, arm_spec)
        closed_loop_guard = _closed_loop_guard_diagnostics(
            summary, history, arm_spec
        )
        selected_checkpoint_diagnostics = dict(
            summary.get("selected_checkpoint_diagnostics") or {}
        )
        selection_constraints = list(
            selected_checkpoint_diagnostics.get("constraints") or []
        )
        selection_feasibility_pass = bool(
            len(selection_constraints)
            == 6 * len(spec.TRAINING_DISTURBANCE_MODES)
            and all(
                float(item.get("normalized_violation", float("inf")))
                <= 1e-10
                for item in selection_constraints
            )
        )
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
            and trust_region["pass"]
            and anchor_replay["pass"]
            and closed_loop_guard["pass"]
            and selection_feasibility_pass
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
            "ppo_trust_region": trust_region,
            "anchor_state_replay": anchor_replay,
            "closed_loop_guard": closed_loop_guard,
            "selected_checkpoint_diagnostics": (
                selected_checkpoint_diagnostics
            ),
            "selection_feasibility_pass": selection_feasibility_pass,
            "strict_reward_signal_pass": strict_reward_signal,
            "all_condition_gates_pass": bool(
                all(row["pass"] for row in conditions)
            ),
            "base_preflight_pass": base_preflight_pass,
            "conditions": conditions,
        }

    eligible = []
    for arm, status in arm_status.items():
        authorizing = arm in spec.AUTHORIZING_ARMS
        passed = bool(status["base_preflight_pass"] and authorizing)
        status.update({
            "groupwise_candidate": arm in spec.GROUPWISE_ARMS,
            "replay_candidate": arm in spec.REPLAY_ARMS,
            "trust_candidate": arm in spec.TRUST_ARMS,
            "closed_loop_candidate": arm in spec.CLOSED_LOOP_ARMS,
            "authorizing_candidate": authorizing,
            "preflight_pass": passed,
        })
        if passed:
            eligible.append(arm)

    selected = max(
        eligible,
        key=lambda arm: (
            min(
                min(row["frequency_reduction_fraction"].values())
                for row in arm_status[arm]["conditions"]
            ),
            -float(spec.ARMS[arm][
                "upper_deployment_frequency_reward_tolerance"
            ]),
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
        "# MuJoCo v14.14 Closed-Loop Actor-Guard Preflight",
        "",
        f"- Status: `{decision['status']}`",
        f"- Calibration pass: `{decision['calibration_pass']}`",
        f"- Selected arm: `{decision['selected_arm']}`",
        "- Evidence role: single optimizer seed, mechanism preflight, no CI.",
        "",
        "| arm | trained | actor | action | replay | trust | closed loop | selection | projection | heldout | reward signal | authorize | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, status in decision["arm_status"].items():
        lines.append(
            f"| {arm} | {status['trained_checkpoint_pass']} | "
            f"{status['actor_changed_pass']} | {status['action_changed_pass']} | "
            f"{status['anchor_state_replay']['pass']} | "
            f"{status['ppo_trust_region']['pass']} | "
            f"{status['closed_loop_guard']['pass']} | "
            f"{status['selection_feasibility_pass']} | "
            f"{status['deployment_projection']['pass']} | "
            f"{status['all_condition_gates_pass']} | "
            f"{status['strict_reward_signal_pass']} | "
            f"{status['authorizing_candidate']} | "
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
        f"mujoco_v14_14_preflight status={decision['status']} "
        f"selected={decision['selected_arm']}"
    )


if __name__ == "__main__":
    main()
