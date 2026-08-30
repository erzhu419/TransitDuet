"""Fail-closed evidence registry for the Freq-HRL manuscript.

The project historically accumulated several independent claim matrices.  This
module provides one auditable reporting boundary: an artifact is not eligible
for manuscript claims unless it is registered here, hash verified, and assigned
an explicit paper disposition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "freq_hrl_authoritative_evidence_registry_v1"
DEVELOPMENT_ADJUDICATION_SCHEMA_VERSION = (
    "freq_hrl_development_preflight_adjudication_v1"
)
MUJOCO_V15_DISTILLATION_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v15_distillation_development_v1"
)
MUJOCO_V16_GAUGE_TRAINING_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v16_gauge_training_development_v1"
)
MUJOCO_V16_1_AUDIT_GAUGE_PAIRED_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v16_1_audit_gauge_paired_development_v1"
)
MUJOCO_V16_2_MACRO_HOLD_GAUGE_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v16_2_macro_hold_gauge_development_v1"
)
DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = Path("transit_hrl/evidence/authoritative_registry_v1.json")
DEFAULT_OUTPUT_DIR = Path(
    "transit_hrl/results/authoritative_evidence_registry_latest"
)
DEFAULT_MD_OUTPUT = Path(
    "transit_hrl/md/freq_hrl_authoritative_claim_ledger_2026-08-09.md"
)

PAPER_DISPOSITIONS = {
    "positive_main_or_si",
    "mixed_or_negative_main_or_si",
    "development_only",
    "excluded_legacy",
}
EVIDENCE_STAGES = {"confirmatory", "development", "legacy"}
CONFIRMATORY_DECISIONS = {"supported", "mixed", "not_supported"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return data


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_registry(path: Path) -> dict[str, Any]:
    registry = _read_json(Path(path))
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("authoritative evidence registry schema mismatch")
    return registry


def _verify_artifacts(
    record: dict[str, Any], repository_root: Path
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"{record.get('evidence_id')}: no source artifacts")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("artifact declarations must be JSON objects")
        role = str(artifact.get("role", ""))
        relative = Path(str(artifact.get("path", "")))
        expected = str(artifact.get("sha256", ""))
        if not role or role in resolved:
            raise ValueError(
                f"{record.get('evidence_id')}: duplicate or empty artifact role"
            )
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"{record.get('evidence_id')}: artifact path must be repository-relative"
            )
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if len(expected) != 64 or actual != expected:
            raise ValueError(
                f"{record.get('evidence_id')}: SHA-256 mismatch for {relative}"
            )
        resolved[role] = path
    return resolved


def _mujoco_v12_facts(paths: dict[str, Path]) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    rows = list(decision.get("environment_rows") or [])
    if (
        decision.get("status") != "confirmatory_supported"
        or decision.get("primary_gate_pass") is not True
        or decision.get("integrity_status") != "valid"
        or len(rows) != 3
    ):
        raise ValueError("MuJoCo v12 registered decision no longer matches")
    return {
        "decision_status": "supported",
        "source_decision_status": str(decision["status"]),
        "primary_gate_pass": True,
        "primary_gate_count": int(decision["primary_gate_count"]),
        "independent_optimizer_replicates_per_arm": int(
            decision["optimizer_replicates_per_environment_arm"]
        ),
        "heldout_paths_per_cell": int(decision["heldout_paths_per_cell"]),
        "environments": [
            {
                "environment": str(row["environment"]),
                "return_difference": float(row["episode_return_difference"]),
                "return_ci95": [
                    float(row["episode_return_difference_ci95_lower"]),
                    float(row["episode_return_difference_ci95_upper"]),
                ],
                "responsibility_drift_reduction": float(
                    row["relative_drift_reduction"]
                ),
                "familywise_drift_reduction_lower": float(
                    row["drift_reduction_familywise_lower_bound"]
                ),
            }
            for row in rows
        ],
    }


def _mujoco_v13_facts(paths: dict[str, Path]) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    rows = list(decision.get("environment_rows") or [])
    if (
        decision.get("status") != "confirmatory_primary_gate_failed"
        or decision.get("primary_gate_pass") is not False
        or decision.get("integrity_status") != "valid"
        or len(rows) != 3
    ):
        raise ValueError("MuJoCo v13 registered decision no longer matches")
    failed = []
    for row in rows:
        gates = {
            "return_noninferiority": bool(row["return_noninferiority_pass"]),
            "responsibility_drift": bool(
                row["minimum_responsibility_drift_reduction_pass"]
            ),
            "raw_lower_drift": bool(
                row["minimum_raw_lower_drift_reduction_pass"]
            ),
            "upper_hf_budget": bool(row["upper_hf_budget_pass"]),
        }
        failed.append({
            "environment": str(row["environment"]),
            "environment_gate_pass": bool(row["environment_primary_gate_pass"]),
            "failed_gates": sorted(key for key, value in gates.items() if not value),
            "return_difference": float(row["episode_return_difference"]),
            "raw_lower_drift_reduction": float(
                row["relative_raw_lower_drift_reduction"]
            ),
            "upper_hf_rms_familywise_upper": float(
                row["upper_hf_rms_familywise_upper_bound"]
            ),
        })
    if all(row["environment_gate_pass"] for row in failed):
        raise ValueError("MuJoCo v13 must retain its registered failed gate")
    return {
        "decision_status": "not_supported",
        "source_decision_status": str(decision["status"]),
        "primary_gate_pass": False,
        "primary_gate_count": int(decision["primary_gate_count"]),
        "independent_optimizer_replicates_per_arm": int(
            decision["optimizer_replicates_per_environment_arm"]
        ),
        "heldout_paths_per_cell": int(decision["heldout_paths_per_cell"]),
        "environments": failed,
    }


def _quant_v74_facts(paths: dict[str, Path]) -> dict[str, Any]:
    summary = _read_json(paths["summary"])
    rows = [
        row for row in _read_csv(paths["paired_effects"])
        if row.get("hypothesis_role") == "primary_baseline"
    ]
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["claim_status"])
        counts[status] = counts.get(status, 0) + 1
    expected = {
        "supported_improvement": int(
            summary["primary_supported_improvement_count"]
        ),
        "supported_harm": int(summary["primary_supported_harm_count"]),
    }
    if (
        summary.get("status") != "valid"
        or len(rows) != int(summary["primary_hypothesis_count"])
        or counts.get("supported_improvement", 0)
        != expected["supported_improvement"]
        or counts.get("supported_harm", 0) != expected["supported_harm"]
    ):
        raise ValueError("Quant v7.4 registered primary analysis no longer matches")
    primary_rows = []
    for row in rows:
        primary_rows.append({
            "comparator": str(row["comparator_variant_id"]),
            "metric": str(row["metric"]),
            "directional_improvement_mean": float(
                row["directional_improvement_mean"]
            ),
            "ci95": [
                float(row["directional_ci95_low"]),
                float(row["directional_ci95_high"]),
            ],
            "effect_size_dz": float(row["paired_effect_size_dz"]),
            "holm_p": float(row["p_value_holm"]),
            "status": str(row["claim_status"]),
        })
    return {
        "decision_status": "mixed",
        "primary_hypothesis_count": len(rows),
        "primary_status_counts": {
            "supported_improvement": counts.get("supported_improvement", 0),
            "supported_harm": counts.get("supported_harm", 0),
            "inconclusive": counts.get("inconclusive", 0),
        },
        "independent_training_replicates": int(
            summary["independent_training_replicates"]
        ),
        "heldout_paths_per_replicate": int(
            summary["heldout_paths_per_replicate"]
        ),
        "scenario_count": int(summary["scenario_count"]),
        "primary_rows": primary_rows,
    }


def _mujoco_v14_facts(paths: dict[str, Path]) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    if (
        decision.get("status") != "no_behavior_safe_candidate"
        or decision.get("evidence_role")
        != "development_screen_not_confirmatory"
        or decision.get("selected_arm") is not None
    ):
        raise ValueError("MuJoCo v14 registered screen decision no longer matches")
    return {
        "decision_status": str(decision["status"]),
        "selected_arm": None,
        "eligible_arms": list(decision.get("eligible_arms") or []),
        "arm_status": dict(decision.get("arm_status") or {}),
        "selection_confidence": float(decision["selection_confidence"]),
        "bootstrap_draws": int(decision["bootstrap_draws"]),
    }


def _mujoco_v14_1_facts(paths: dict[str, Path]) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    arm_status = dict(decision.get("arm_status") or {})
    if (
        decision.get("status") != "no_behavior_safe_candidate"
        or decision.get("evidence_role")
        != "development_screen_not_confirmatory"
        or decision.get("selected_arm") is not None
        or decision.get("gate_granularity")
        != "environment_by_disturbance_mode"
        or not arm_status
        or any(
            int(status.get("passed_gate_count", -1)) != 0
            or int(status.get("total_gate_count", -1)) != 15
            for status in arm_status.values()
        )
    ):
        raise ValueError("MuJoCo v14.1 registered screen decision no longer matches")
    return {
        "decision_status": str(decision["status"]),
        "selected_arm": None,
        "eligible_arms": list(decision.get("eligible_arms") or []),
        "arm_status": arm_status,
        "gate_granularity": str(decision["gate_granularity"]),
        "selection_confidence": float(decision["selection_confidence"]),
        "bootstrap_draws": int(decision["bootstrap_draws"]),
    }


def _mujoco_v14_2_facts(paths: dict[str, Path]) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    arm_status = dict(decision.get("arm_status") or {})
    if (
        decision.get("status") != "no_behavior_safe_candidate"
        or decision.get("evidence_role")
        != "development_screen_not_confirmatory"
        or decision.get("development_protocol_version")
        != "mujoco_v14_2_physical_router_screen_v1"
        or decision.get("selected_arm") is not None
        or decision.get("gate_granularity")
        != "environment_by_disturbance_mode"
        or len(arm_status) != 8
        or any(
            int(status.get("total_gate_count", -1)) != 15
            for status in arm_status.values()
        )
        or any(
            int(status.get(
                "strict_responsibility_improvement_condition_count", -1
            )) != 15
            for status in arm_status.values()
        )
        or max(
            int(status.get("passed_gate_count", -1))
            for status in arm_status.values()
        ) != 2
    ):
        raise ValueError("MuJoCo v14.2 registered screen decision no longer matches")
    return {
        "decision_status": str(decision["status"]),
        "selected_arm": None,
        "eligible_arms": list(decision.get("eligible_arms") or []),
        "arm_status": arm_status,
        "gate_granularity": str(decision["gate_granularity"]),
        "selection_confidence": float(decision["selection_confidence"]),
        "bootstrap_draws": int(decision["bootstrap_draws"]),
        "maximum_complete_condition_count": max(
            int(status["passed_gate_count"])
            for status in arm_status.values()
        ),
    }


def _mujoco_v14_3_facts(paths: dict[str, Path]) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    arm_status = dict(decision.get("arm_status") or {})
    best_arm = "router_a004_s010_reward"
    if (
        decision.get("status") != "no_behavior_safe_candidate"
        or decision.get("evidence_role")
        != "development_screen_not_confirmatory"
        or decision.get("development_protocol_version")
        != "mujoco_v14_3_partial_router_screen_v1"
        or decision.get("selected_arm") is not None
        or decision.get("gate_granularity")
        != "environment_by_disturbance_mode"
        or len(arm_status) != 7
        or best_arm not in arm_status
        or any(
            int(status.get("total_gate_count", -1)) != 15
            for status in arm_status.values()
        )
        or max(
            int(status.get("passed_gate_count", -1))
            for status in arm_status.values()
        ) != 4
        or int(
            arm_status[best_arm].get(
                "strict_responsibility_improvement_condition_count", -1
            )
        ) != 15
        or int(
            arm_status[best_arm].get(
                "strict_raw_improvement_condition_count", -1
            )
        ) != 10
    ):
        raise ValueError(
            "MuJoCo v14.3 registered screen decision no longer matches"
        )
    return {
        "decision_status": str(decision["status"]),
        "selected_arm": None,
        "eligible_arms": list(decision.get("eligible_arms") or []),
        "arm_status": arm_status,
        "gate_granularity": str(decision["gate_granularity"]),
        "selection_confidence": float(decision["selection_confidence"]),
        "bootstrap_draws": int(decision["bootstrap_draws"]),
        "maximum_complete_condition_count": max(
            int(status["passed_gate_count"])
            for status in arm_status.values()
        ),
        "best_arm": best_arm,
        "best_arm_responsibility_improvement_condition_count": int(
            arm_status[best_arm][
                "strict_responsibility_improvement_condition_count"
            ]
        ),
        "best_arm_raw_improvement_condition_count": int(
            arm_status[best_arm]["strict_raw_improvement_condition_count"]
        ),
    }


def _mujoco_v14_4_facts(paths: dict[str, Path]) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    arm_status = dict(decision.get("arm_status") or {})
    rows = _read_csv(paths["environment_condition_gates"])
    best_joint_arm = "router_a004_s010_linear_w0125_r0375"
    fastest_ramp_arm = "router_a004_s010_linear_w000_r025"
    expected_arms = set(arm_status)
    if (
        decision.get("status") != "no_behavior_safe_candidate"
        or decision.get("evidence_role")
        != "development_screen_not_confirmatory"
        or decision.get("development_protocol_version")
        != "mujoco_v14_4_router_homotopy_screen_v1"
        or decision.get("selected_arm") is not None
        or decision.get("gate_granularity")
        != "environment_by_disturbance_mode"
        or len(arm_status) != 7
        or {best_joint_arm, fastest_ramp_arm} - expected_arms
        or len(rows) != 105
        or {row.get("arm") for row in rows} != expected_arms
        or any(
            sum(row.get("arm") == arm for row in rows) != 15
            for arm in expected_arms
        )
        or any(
            int(status.get("total_gate_count", -1)) != 15
            for status in arm_status.values()
        )
        or max(
            int(status.get("passed_gate_count", -1))
            for status in arm_status.values()
        ) != 3
    ):
        raise ValueError(
            "MuJoCo v14.4 registered screen decision no longer matches"
        )

    def pass_count(arm: str, field: str) -> int:
        return sum(
            row["arm"] == arm and row.get(field) == "True"
            for row in rows
        )

    if (
        pass_count(best_joint_arm, "condition_gate_pass") != 3
        or pass_count(fastest_ramp_arm, "reward_noninferiority_pass") != 10
        or pass_count(fastest_ramp_arm, "responsibility_drift_pass") != 5
        or pass_count(fastest_ramp_arm, "raw_lower_drift_pass") != 5
        or any(
            pass_count(arm, "effective_lower_activity_pass") != 15
            or pass_count(arm, "router_clip_pass") != 15
            or pass_count(arm, "reconstruction_integrity_pass") != 15
            for arm in expected_arms
        )
    ):
        raise ValueError(
            "MuJoCo v14.4 registered condition gates no longer match"
        )
    return {
        "decision_status": str(decision["status"]),
        "selected_arm": None,
        "eligible_arms": list(decision.get("eligible_arms") or []),
        "arm_status": arm_status,
        "gate_granularity": str(decision["gate_granularity"]),
        "selection_confidence": float(decision["selection_confidence"]),
        "bootstrap_draws": int(decision["bootstrap_draws"]),
        "maximum_complete_condition_count": 3,
        "best_joint_arm": best_joint_arm,
        "fastest_ramp_arm": fastest_ramp_arm,
        "fastest_ramp_reward_noninferiority_condition_count": 10,
        "fastest_ramp_responsibility_condition_count": 5,
        "fastest_ramp_raw_condition_count": 5,
    }


def _mujoco_v14_5_facts(paths: dict[str, Path]) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    arm_status = dict(decision.get("arm_status") or {})
    rows = _read_csv(paths["environment_condition_gates"])
    full_drift_arms = {
        "router_s010_ua020_la100",
        "router_s015_ua005_la010",
    }
    expected_arms = set(arm_status)
    if (
        decision.get("status") != "no_behavior_safe_candidate"
        or decision.get("evidence_role")
        != "development_screen_not_confirmatory"
        or decision.get("development_protocol_version")
        != "mujoco_v14_5_paired_anchor_screen_v1"
        or decision.get("selected_arm") is not None
        or decision.get("gate_granularity")
        != "environment_by_disturbance_mode"
        or len(arm_status) != 7
        or full_drift_arms - expected_arms
        or len(rows) != 105
        or {row.get("arm") for row in rows} != expected_arms
        or any(
            sum(row.get("arm") == arm for row in rows) != 15
            for arm in expected_arms
        )
        or any(
            int(status.get("total_gate_count", -1)) != 15
            for status in arm_status.values()
        )
        or max(
            int(status.get("passed_gate_count", -1))
            for status in arm_status.values()
        ) != 5
    ):
        raise ValueError(
            "MuJoCo v14.5 registered screen decision no longer matches"
        )

    def pass_count(arm: str, field: str) -> int:
        return sum(
            row["arm"] == arm and row.get(field) == "True"
            for row in rows
        )

    if (
        any(
            pass_count(arm, "reward_noninferiority_pass") != 5
            or pass_count(arm, "effective_lower_activity_pass") != 15
            or pass_count(arm, "router_clip_pass") != 15
            or pass_count(arm, "reconstruction_integrity_pass") != 15
            for arm in expected_arms
        )
        or any(
            pass_count(arm, "responsibility_drift_pass") != 15
            or pass_count(arm, "raw_lower_drift_pass") != 15
            or pass_count(arm, "upper_hf_budget_pass") != 10
            or pass_count(arm, "condition_gate_pass") != 5
            for arm in full_drift_arms
        )
        or any(
            status.get("trained_checkpoint_gate_pass") is not False
            for status in arm_status.values()
        )
    ):
        raise ValueError(
            "MuJoCo v14.5 registered condition gates no longer match"
        )
    return {
        "decision_status": str(decision["status"]),
        "selected_arm": None,
        "eligible_arms": list(decision.get("eligible_arms") or []),
        "arm_status": arm_status,
        "gate_granularity": str(decision["gate_granularity"]),
        "selection_confidence": float(decision["selection_confidence"]),
        "bootstrap_draws": int(decision["bootstrap_draws"]),
        "maximum_complete_condition_count": 5,
        "full_drift_arms": sorted(full_drift_arms),
        "full_drift_arm_reward_noninferiority_condition_count": 5,
        "full_drift_arm_upper_hf_condition_count": 10,
        "all_arms_trained_checkpoint_gate_pass": False,
    }


def _mujoco_v14_6_facts(paths: dict[str, Path]) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    arm_status = dict(decision.get("arm_status") or {})
    rows = _read_csv(paths["environment_condition_gates"])
    expected_arms = {
        "conservative_s0025",
        "conservative_s0050",
        "conservative_s0075",
        "conservative_s0100",
        "conservative_s0125",
        "conservative_s0150",
        "conservative_s0200",
    }
    full_drift_arms = {
        "conservative_s0075",
        "conservative_s0100",
        "conservative_s0125",
        "conservative_s0150",
        "conservative_s0200",
    }
    if (
        decision.get("status") != "no_behavior_safe_candidate"
        or decision.get("evidence_role")
        != "development_screen_not_confirmatory"
        or decision.get("development_protocol_version")
        != "mujoco_v14_6_conservative_transfer_screen_v1"
        or decision.get("selected_arm") is not None
        or decision.get("gate_granularity")
        != "environment_by_disturbance_mode"
        or set(arm_status) != expected_arms
        or len(rows) != 105
        or {row.get("arm") for row in rows} != expected_arms
        or any(
            sum(row.get("arm") == arm for row in rows) != 15
            for arm in expected_arms
        )
        or any(
            int(status.get("total_gate_count", -1)) != 15
            for status in arm_status.values()
        )
        or max(
            int(status.get("passed_gate_count", -1))
            for status in arm_status.values()
        ) != 10
    ):
        raise ValueError(
            "MuJoCo v14.6 registered screen decision no longer matches"
        )

    def pass_count(arm: str, field: str) -> int:
        return sum(
            row["arm"] == arm and row.get(field) == "True"
            for row in rows
        )

    if (
        any(
            pass_count(arm, "exact_return_pass") != 15
            or pass_count(arm, "exact_trace_pass") != 15
            or pass_count(arm, "reward_noninferiority_pass") != 15
            or pass_count(arm, "effective_lower_activity_pass") != 15
            or pass_count(arm, "router_clip_pass") != 15
            or pass_count(arm, "reconstruction_integrity_pass") != 15
            or pass_count(arm, "function_preserving_pass") != 15
            or arm_status[arm].get(
                "exact_selected_parameter_hash_gate_pass"
            ) is not True
            or arm_status[arm].get("trained_checkpoint_gate_pass") is not True
            for arm in expected_arms
        )
        or any(
            pass_count(arm, "responsibility_drift_pass") != 15
            or pass_count(arm, "raw_lower_drift_pass") != 15
            or pass_count(arm, "upper_hf_budget_pass") != 10
            or pass_count(arm, "condition_gate_pass") != 10
            for arm in full_drift_arms
        )
        or pass_count("conservative_s0050", "condition_gate_pass") != 10
        or pass_count("conservative_s0025", "condition_gate_pass") != 0
    ):
        raise ValueError(
            "MuJoCo v14.6 registered condition gates no longer match"
        )
    return {
        "decision_status": str(decision["status"]),
        "selected_arm": None,
        "eligible_arms": list(decision.get("eligible_arms") or []),
        "arm_status": arm_status,
        "gate_granularity": str(decision["gate_granularity"]),
        "selection_confidence": float(decision["selection_confidence"]),
        "bootstrap_draws": int(decision["bootstrap_draws"]),
        "maximum_complete_condition_count": 10,
        "full_drift_arms": sorted(full_drift_arms),
        "full_drift_arm_reward_noninferiority_condition_count": 15,
        "full_drift_arm_upper_hf_condition_count": 10,
        "all_arms_exact_return_trace_pass": True,
        "all_arms_exact_parameter_hash_pass": True,
        "all_arms_trained_checkpoint_gate_pass": True,
    }


def _development_preflight_adjudication_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    if (
        decision.get("schema_version")
        != DEVELOPMENT_ADJUDICATION_SCHEMA_VERSION
        or not str(decision.get("status", ""))
        or not str(decision.get("source_run", ""))
        or not str(decision.get("development_protocol_version", ""))
        or decision.get("full_screen_launched") is not False
        or decision.get("selected_arm") is not None
        or int(decision.get("optimizer_replicates", 0)) < 1
        or int(decision.get("heldout_paths_per_continuation", 0)) < 1
        or int(decision.get("completed_anchor_cells", 0)) < 1
        or int(decision.get("completed_continuation_cells", 0)) < 1
        or len(str(decision.get("frozen_algorithm_revision", ""))) != 40
        or len(str(decision.get("frozen_source_manifest_sha256", ""))) != 64
    ):
        raise ValueError("development preflight adjudication is incomplete")
    if report is None:
        raise ValueError("development preflight adjudication lacks its report")
    if (
        _sha256(report) != decision.get("source_report_sha256")
        or not report.as_posix().endswith(str(decision.get("source_report", "")))
    ):
        raise ValueError("development preflight report identity drifted")
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": str(decision["integrity_status"]),
        "source_run": str(decision["source_run"]),
        "development_protocol_version": str(
            decision["development_protocol_version"]
        ),
        "environment": str(decision["environment"]),
        "optimizer_replicates": int(decision["optimizer_replicates"]),
        "heldout_paths_per_continuation": int(
            decision["heldout_paths_per_continuation"]
        ),
        "completed_anchor_cells": int(decision["completed_anchor_cells"]),
        "completed_continuation_cells": int(
            decision["completed_continuation_cells"]
        ),
        "selected_arm": None,
        "full_screen_launched": False,
    }


def _mujoco_mechanism_preflight_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    status = str(decision.get("status", ""))
    selected = decision.get("selected_arm")
    eligible = list(decision.get("eligible_arms") or [])
    arm_status = dict(decision.get("arm_status") or {})
    if (
        status not in {"do_not_expand", "expand_to_multiseed_screen"}
        or decision.get("evidence_role")
        != "single_optimizer_seed_mechanism_preflight_no_ci"
        or not str(decision.get("analysis_version", "")).startswith("mujoco_v14_")
        or str(decision.get("environment", "")) != "HalfCheetah-v5"
        or int(decision.get("optimizer_seed", -1)) < 0
        or len(str(decision.get("input_sha256", ""))) != 64
        or not arm_status
        or (status == "do_not_expand" and selected is not None)
        or (
            status == "expand_to_multiseed_screen"
            and (not isinstance(selected, str) or selected not in eligible)
        )
    ):
        raise ValueError("MuJoCo mechanism preflight decision is incomplete")
    if report is None:
        raise ValueError("MuJoCo mechanism preflight lacks its report")
    report_text = report.read_text(encoding="utf-8")
    if (
        f"Status: `{status}`" not in report_text
        or f"Selected arm: `{selected}`" not in report_text
        or "single optimizer seed" not in report_text.lower()
        or "no ci" not in report_text.lower()
    ):
        raise ValueError("MuJoCo mechanism preflight report drifted")
    return {
        "decision_status": status,
        "integrity_status": "valid",
        "analysis_version": str(decision["analysis_version"]),
        "environment": str(decision["environment"]),
        "optimizer_seed": int(decision["optimizer_seed"]),
        "calibration_pass": bool(decision.get("calibration_pass", False)),
        "eligible_arms": eligible,
        "selected_arm": selected,
        "arm_count": len(arm_status),
        "input_sha256": str(decision["input_sha256"]),
    }


def _mujoco_v14_15_multiseed_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    primary = _read_csv(paths["primary_contrasts"])
    replicates = _read_csv(paths["replicate_rows"])
    report = paths.get("report")
    environment_gates = list(decision.get("environment_gates") or [])
    expected_complete = {
        "HalfCheetah-v5": 0,
        "Hopper-v5": 7,
        "Walker2d-v5": 1,
    }
    observed_complete = {
        str(row.get("environment")): int(row.get("complete_gate_count", -1))
        for row in environment_gates
    }
    if (
        decision.get("status") != "candidate_not_ready_for_confirmation"
        or decision.get("evidence_role")
        != "candidate_fixed_multiseed_development_no_confirmation"
        or decision.get("development_protocol_version")
        != "mujoco_v14_15_restoration_multiseed_development_screen_v2"
        or decision.get("primary_contrast_pass") is not False
        or decision.get("environment_complete_gate_pass") is not False
        or int(decision.get("optimizer_seed_count", -1)) != 15
        or int(decision.get("environment_count", -1)) != 3
        or int(decision.get("aggregate_complete_gate_count", -1)) != 8
        or int(decision.get("aggregate_replicate_count", -1)) != 45
        or observed_complete != expected_complete
        or len(primary) != 18
        or len(replicates) != 45
        or sum(row.get("candidate_preflight_pass") == "True" for row in replicates)
        != 8
        or report is None
        or "Status: `candidate_not_ready_for_confirmation`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError("MuJoCo v14.15 multiseed decision no longer matches")
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "development_protocol_version": str(
            decision["development_protocol_version"]
        ),
        "optimizer_seed_count": int(decision["optimizer_seed_count"]),
        "environment_count": int(decision["environment_count"]),
        "primary_contrast_pass": False,
        "complete_candidate_cells": int(
            decision["aggregate_complete_gate_count"]
        ),
        "candidate_cell_count": int(decision["aggregate_replicate_count"]),
        "complete_cells_by_environment": expected_complete,
        "aggregate_complete_fraction_wilson_lower": float(
            decision["aggregate_complete_gate_fraction_lower"]
        ),
        "heldout_paths_are_not_replicates": bool(
            decision["heldout_paths_are_not_replicates"]
        ),
        "input_sha256": str(decision["input_sha256"]),
    }


def _mujoco_v14_16_mechanism_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    ranking = _read_csv(paths["arm_ranking"])
    replicates = _read_csv(paths["replicate_rows"])
    manifest = _read_json(paths["cell_manifest"])
    sync = _read_json(paths["result_sync"])
    report = paths.get("report")
    primary_arm = "l2_path_freeze_crossreplay"
    best_arm = "l2_path_trainreplay"
    primary_rows = [row for row in replicates if row.get("arm") == primary_arm]
    primary_ranking = [row for row in ranking if row.get("arm") == primary_arm]
    best_ranking = [row for row in ranking if row.get("arm") == best_arm]
    lineages = sync.get("task_attempt_lineage") or {}
    reroute_count = sum(
        row.get("selection") == "unique_successful_reroute"
        for row in lineages.values()
        if isinstance(row, dict)
    )
    snapshots = sync.get("scheduler_snapshots") or []
    if (
        decision.get("status") != "primary_mechanism_not_ready"
        or decision.get("evidence_role")
        != "mechanism_screen_development_not_confirmation"
        or decision.get("development_protocol_version")
        != "mujoco_v14_16_crossed_restoration_mechanism_screen_v2"
        or decision.get("primary_candidate_arm") != primary_arm
        or decision.get("primary_ready") is not False
        or int(decision.get("optimizer_seed_count", -1)) != 3
        or len(ranking) != 5
        or len(replicates) != 45
        or len(primary_rows) != 9
        or len(primary_ranking) != 1
        or int(primary_ranking[0].get("engineering_pass_count", -1)) != 0
        or int(primary_ranking[0].get("complete_effect_gate_count", -1)) != 0
        or int(primary_ranking[0].get("environment_complete_count", -1)) != 0
        or len(best_ranking) != 1
        or int(best_ranking[0].get("engineering_pass_count", -1)) != 2
        or int(best_ranking[0].get("complete_effect_gate_count", -1)) != 2
        or int(best_ranking[0].get("environment_complete_count", -1)) != 1
        or manifest.get("status") != "development_screen_complete_unanalyzed"
        or int(manifest.get("cell_count", -1)) != 81
        or int(manifest.get("anchor_cell_count", -1)) != 9
        or int(manifest.get("continuation_cell_count", -1)) != 72
        or sync.get("status") != "run_scoped_result_sync_complete"
        or int(sync.get("cell_count", -1)) != 81
        or len(lineages) != 81
        or reroute_count != 15
        or len(snapshots) != 1
        or int(snapshots[0].get("task_record_count", -1)) != 9
        or report is None
        or "Status: `primary_mechanism_not_ready`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError("MuJoCo v14.16 mechanism decision no longer matches")
    trained_primary = sum(
        int(row.get("selected_checkpoint_iteration", -1)) >= 7
        for row in primary_rows
    )
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "development_protocol_version": str(
            decision["development_protocol_version"]
        ),
        "optimizer_seed_count": int(decision["optimizer_seed_count"]),
        "environment_count": 3,
        "merged_cell_count": int(manifest["cell_count"]),
        "rerouted_success_count": int(reroute_count),
        "archived_anchor_record_count": int(
            snapshots[0]["task_record_count"]
        ),
        "primary_arm": primary_arm,
        "primary_engineering_pass_count": 0,
        "primary_complete_effect_gate_count": 0,
        "primary_trained_checkpoint_count": int(trained_primary),
        "primary_fallback_checkpoint_count": 9 - int(trained_primary),
        "best_diagnostic_arm": best_arm,
        "best_environment_complete_count": 1,
        "best_engineering_pass_count": 2,
        "heldout_paths_are_not_replicates": bool(
            decision["heldout_paths_are_not_replicates"]
        ),
        "input_sha256": str(decision["input_sha256"]),
    }


def _mujoco_v14_29_portfolio_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    cell_rows = _read_csv(paths["cell_rows"])
    preregistration = _read_json(paths["preregistration"])
    qualification = _read_json(paths["anchor_qualification"])
    report = paths.get("report")
    environment_rows = list(decision.get("environment_results") or [])
    cells = list(decision.get("cells") or [])
    anchors = list(qualification.get("anchors") or [])
    expected_support = {
        "HalfCheetah-v5": 16,
        "Hopper-v5": 16,
        "Walker2d-v5": 15,
    }
    observed_support = {
        str(row.get("environment")): int(row.get("supported_count", -1))
        for row in environment_rows
    }
    frozen_revision = str(preregistration.get("frozen_algorithm_revision", ""))
    frozen_manifest = str(
        preregistration.get("frozen_source_manifest_sha256", "")
    )
    router_rows = [
        row for row in cells
        if row.get("selected_source") == "function_preserving_router_adapter"
    ]
    actor_rows = [
        row for row in cells
        if str(row.get("selected_source", "")).startswith(
            "paired_finite_difference:"
        )
    ]
    abstentions = [row for row in cells if row.get("selected_source") is None]
    csv_index = {
        (str(row.get("environment")), int(row.get("optimizer_seed", -1))): row
        for row in cell_rows
    }
    cell_index = {
        (str(row.get("environment")), int(row.get("optimizer_seed", -1))): row
        for row in cells
    }
    unique_parameters = {
        environment: len({
            str(row.get("parameter_sha256"))
            for row in anchors
            if row.get("environment") == environment
        })
        for environment in expected_support
    }
    if (
        preregistration.get("status")
        != "frozen_before_v14_29_confirmatory_outcome_access"
        or preregistration.get("development_protocol_version")
        != "mujoco_v14_29_portfolio_confirmatory_v1"
        or preregistration.get("evidence_role")
        != "fresh_optimizer_seed_mechanism_portfolio_confirmation"
        or len(preregistration.get("optimizer_seeds") or []) != 16
        or len(preregistration.get("validation_roots") or []) != 32
        or preregistration.get("scheduler_contract", {}).get("scheduler")
        != "scheduleurm"
        or preregistration.get("scheduler_contract", {}).get("require_node")
        is not None
        or preregistration.get("scheduler_contract", {}).get("slurm_used")
        is not False
        or len(frozen_revision) != 40
        or len(frozen_manifest) != 64
        or qualification.get("status") != "fresh_anchor_bank_qualified"
        or int(qualification.get("anchor_count", -1)) != 48
        or int(qualification.get("qualified_anchor_count", -1)) != 48
        or len(anchors) != 48
        or not all(row.get("qualified") is True for row in anchors)
        or qualification.get("frozen_algorithm_revision") != frozen_revision
        or qualification.get("frozen_source_manifest_sha256") != frozen_manifest
        or any(count != 16 for count in unique_parameters.values())
        or decision.get("status") != "mechanism_portfolio_confirmed"
        or decision.get("analysis_version")
        != "mujoco_v14_29_portfolio_confirmatory_v1"
        or decision.get("statistical_unit") != "optimizer_seed"
        or decision.get("inference_scope")
        != "fresh_optimizer_seeds_conditional_on_frozen_validation_path_panel"
        or decision.get("frozen_algorithm_revision") != frozen_revision
        or decision.get("frozen_source_manifest_sha256") != frozen_manifest
        or int(decision.get("cell_count", -1)) != 48
        or int(decision.get("supported_cell_count", -1)) != 47
        or len(environment_rows) != 3
        or observed_support != expected_support
        or any(int(row.get("optimizer_seed_count", -1)) != 16 for row in environment_rows)
        or any(row.get("confirmatory_gate_pass") is not True for row in environment_rows)
        or any(float(row.get("success_rate_wilson_lower", 0.0)) <= 0.5 for row in environment_rows)
        or len(cells) != 48
        or len(cell_rows) != 48
        or csv_index.keys() != cell_index.keys()
        or any(
            (csv_index[key].get("validation_supported") == "True")
            != bool(cell_index[key].get("validation_supported"))
            for key in cell_index
        )
        or len(router_rows) != 38
        or len(actor_rows) != 9
        or len(abstentions) != 1
        or any(row.get("selected_router_trace_invariant") is not True for row in router_rows)
        or any(
            float(row.get("validation_reward_violation_count", 1.0)) != 0.0
            for row in cells if row.get("validation_supported") is True
        )
        or report is None
        or "`mechanism_portfolio_confirmed`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError("MuJoCo v14.29 portfolio decision no longer matches")
    return {
        "decision_status": "supported",
        "integrity_status": "valid",
        "development_protocol_version": str(
            preregistration["development_protocol_version"]
        ),
        "frozen_algorithm_revision": frozen_revision,
        "frozen_source_manifest_sha256": frozen_manifest,
        "optimizer_seed_count_per_environment": 16,
        "environment_count": 3,
        "cell_count": 48,
        "supported_cell_count": 47,
        "supported_count_by_environment": expected_support,
        "wilson_lower_by_environment": {
            str(row["environment"]): float(row["success_rate_wilson_lower"])
            for row in environment_rows
        },
        "router_selection_count": len(router_rows),
        "actor_selection_count": len(actor_rows),
        "abstention_count": len(abstentions),
        "all_selected_router_traces_invariant": True,
        "all_supported_cells_respect_reward_floor": True,
        "inference_scope": str(decision["inference_scope"]),
    }


def _mujoco_v15_distillation_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    runs = list(decision.get("runs") or [])
    expected_labels = ["v15", "v15.1", "v15.2"]
    expected_statuses = [
        "raw_policy_distillation_preflight_not_supported",
        "bounded_raw_policy_preflight_not_supported",
        "multisource_raw_policy_preflight_not_supported",
    ]
    expected_environment_status = {
        "HalfCheetah-v5",
        "Hopper-v5",
        "Walker2d-v5",
    }
    if (
        decision.get("schema_version")
        != MUJOCO_V15_DISTILLATION_SCHEMA_VERSION
        or decision.get("status")
        != "universal_raw_policy_distillation_not_supported"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != "three_stage_single_optimizer_seed_development_not_confirmatory"
        or int(decision.get("optimizer_seed", -1)) != 2978317753
        or int(decision.get("environment_count", -1)) != 3
        or set(decision.get("environments") or []) != expected_environment_status
        or decision.get("universal_supported") is not False
        or decision.get("supported_environment_in_every_run") != "Hopper-v5"
        or [row.get("label") for row in runs] != expected_labels
        or [row.get("analysis_status") for row in runs] != expected_statuses
        or any(int(row.get("validation_supported_count", -1)) != 1 for row in runs)
        or any(len(str(row.get("frozen_algorithm_revision", ""))) != 40 for row in runs)
        or any(
            set((row.get("environment_status") or {}).keys())
            != expected_environment_status
            or (row.get("environment_status") or {}).get("Hopper-v5")
            != "supported"
            for row in runs
        )
        or report is None
        or "`universal_raw_policy_distillation_not_supported`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError("MuJoCo v15 distillation decision no longer matches")
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "optimizer_seed_count": 1,
        "environment_count": 3,
        "run_count": 3,
        "candidate_counts": {
            str(row["label"]): int(row["candidate_count"])
            for row in runs
        },
        "validation_supported_count_by_run": {
            str(row["label"]): int(row["validation_supported_count"])
            for row in runs
        },
        "supported_environment_in_every_run": "Hopper-v5",
        "universal_supported": False,
    }


def _mujoco_v16_gauge_training_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    gates = dict(decision.get("gate_counts") or {})
    environments = dict(decision.get("environment_results") or {})
    expected_gates = {
        "exact_reconstruction": 9,
        "reward_noninferiority": 6,
        "canonical_frequency_reduction": 0,
        "latent_noninferiority_vs_joint": 4,
        "latent_constraint_improvement": 5,
    }
    expected_environments = {
        "HalfCheetah-v5",
        "Hopper-v5",
        "Walker2d-v5",
    }
    task_status = dict(decision.get("task_status_counts") or {})
    tasks = list(decision.get("scheduler_tasks") or [])
    if (
        decision.get("schema_version")
        != MUJOCO_V16_GAUGE_TRAINING_SCHEMA_VERSION
        or decision.get("status")
        != "training_time_gauge_preflight_not_supported"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != "three_environment_three_optimizer_seed_development_not_confirmatory"
        or int(decision.get("optimizer_seed_count", -1)) != 3
        or int(decision.get("environment_count", -1)) != 3
        or int(decision.get("training_cell_count", -1)) != 27
        or int(decision.get("paired_analysis_cell_count", -1)) != 9
        or int(decision.get("scheduler_task_count", -1)) != 27
        or len(tasks) != 27
        or len(set(tasks)) != 27
        or task_status != {"done": 27, "failed": 0, "cancelled": 0}
        or gates != expected_gates
        or set(environments) != expected_environments
        or environments["Hopper-v5"].get("latent_environment_gate") is not False
        or any(
            int(row.get("latent_improvement_count", -1)) < 0
            for row in environments.values()
        )
        or len(str(decision.get("frozen_algorithm_revision", ""))) != 40
        or len(str(decision.get("frozen_source_manifest_sha256", ""))) != 64
        or report is None
        or "`training_time_gauge_preflight_not_supported`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError("MuJoCo v16 gauge-training decision no longer matches")
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "optimizer_seed_count": 3,
        "environment_count": 3,
        "training_cell_count": 27,
        "paired_analysis_cell_count": 9,
        "gate_counts": expected_gates,
        "task_status_counts": task_status,
        "support_gate": False,
    }


def _mujoco_v16_1_audit_gauge_paired_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    gates = dict(decision.get("gate_counts") or {})
    environments = dict(decision.get("environment_results") or {})
    expected_gates = {
        "candidate_selected_trained_checkpoint": 1,
        "selection_constraints_feasible": 4,
        "reward_noninferiority": 9,
        "canonical_frequency_reduction": 1,
        "latent_noninferiority_vs_control": 9,
        "exact_reconstruction": 9,
        "adaptive_cutoff_active": 9,
    }
    expected_environments = {
        "HalfCheetah-v5",
        "Hopper-v5",
        "Walker2d-v5",
    }
    task_status = dict(decision.get("task_status_counts") or {})
    tasks = list(decision.get("scheduler_tasks") or [])
    if (
        decision.get("schema_version")
        != MUJOCO_V16_1_AUDIT_GAUGE_PAIRED_SCHEMA_VERSION
        or decision.get("status")
        != "audit_gauge_paired_preflight_not_supported"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != "paired_training_mechanism_development_not_confirmatory"
        or int(decision.get("optimizer_seed_count", -1)) != 3
        or int(decision.get("environment_count", -1)) != 3
        or int(decision.get("anchor_training_cell_count", -1)) != 9
        or int(decision.get("continuation_training_cell_count", -1)) != 18
        or int(decision.get("paired_analysis_cell_count", -1)) != 9
        or int(decision.get("scheduler_task_count", -1)) != 27
        or len(tasks) != 27
        or len(set(tasks)) != 27
        or task_status != {"done": 27, "failed": 0, "cancelled": 0}
        or gates != expected_gates
        or set(environments) != expected_environments
        or any(
            row.get("environment_gate") is not False
            or int(row.get("cell_count", -1)) != 3
            for row in environments.values()
        )
        or len(str(decision.get("frozen_algorithm_revision", ""))) != 40
        or len(str(decision.get("frozen_source_manifest_sha256", ""))) != 64
        or report is None
        or "`audit_gauge_paired_preflight_not_supported`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError("MuJoCo v16.1 audit-gauge decision no longer matches")
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "optimizer_seed_count": 3,
        "environment_count": 3,
        "training_cell_count": 27,
        "paired_analysis_cell_count": 9,
        "gate_counts": expected_gates,
        "task_status_counts": task_status,
        "support_gate": False,
    }


def _mujoco_v16_2_macro_hold_gauge_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    gates = dict(decision.get("gate_counts") or {})
    environments = dict(decision.get("environment_results") or {})
    expected_gates = {
        "trained_checkpoint": 9,
        "reward_noninferiority": 5,
        "exact_reconstruction": 9,
        "zero_router_clipping": 9,
        "upper_hf_budget": 4,
        "lower_lf_reduction": 6,
        "joint_merit_reduction": 6,
        "all_cell_gates": 2,
    }
    expected_supported = {
        "HalfCheetah-v5": 0,
        "Hopper-v5": 1,
        "Walker2d-v5": 1,
    }
    task_status = dict(decision.get("task_status_counts") or {})
    tasks = list(decision.get("scheduler_tasks") or [])
    if (
        decision.get("schema_version")
        != MUJOCO_V16_2_MACRO_HOLD_GAUGE_SCHEMA_VERSION
        or decision.get("status") != "macro_hold_gauge_screen_not_supported"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != "macro_rate_gauge_mechanism_development_not_confirmatory"
        or int(decision.get("optimizer_seed_count", -1)) != 3
        or int(decision.get("environment_count", -1)) != 3
        or int(decision.get("training_cell_count", -1)) != 27
        or int(decision.get("paired_analysis_cell_count", -1)) != 9
        or int(decision.get("scheduler_task_count", -1)) != 27
        or len(tasks) != 27
        or len(set(tasks)) != 27
        or task_status != {"done": 27, "failed": 0, "cancelled": 0}
        or gates != expected_gates
        or set(environments) != set(expected_supported)
        or any(
            row.get("environment_gate") is not False
            or int(row.get("cell_count", -1)) != 3
            or int(row.get("supported_count", -1)) != expected_supported[name]
            for name, row in environments.items()
        )
        or len(str(decision.get("frozen_algorithm_revision", ""))) != 40
        or len(str(decision.get("frozen_source_manifest_sha256", ""))) != 64
        or report is None
        or "`macro_hold_gauge_screen_not_supported`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError("MuJoCo v16.2 macro-hold decision no longer matches")
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "optimizer_seed_count": 3,
        "environment_count": 3,
        "training_cell_count": 27,
        "paired_analysis_cell_count": 9,
        "gate_counts": expected_gates,
        "task_status_counts": task_status,
        "support_gate": False,
    }


PARSERS = {
    "mujoco_v12": _mujoco_v12_facts,
    "mujoco_v13": _mujoco_v13_facts,
    "quant_v74": _quant_v74_facts,
    "mujoco_v14": _mujoco_v14_facts,
    "mujoco_v14_1": _mujoco_v14_1_facts,
    "mujoco_v14_2": _mujoco_v14_2_facts,
    "mujoco_v14_3": _mujoco_v14_3_facts,
    "mujoco_v14_4": _mujoco_v14_4_facts,
    "mujoco_v14_5": _mujoco_v14_5_facts,
    "mujoco_v14_6": _mujoco_v14_6_facts,
    "development_preflight_adjudication": (
        _development_preflight_adjudication_facts
    ),
    "mujoco_mechanism_preflight": _mujoco_mechanism_preflight_facts,
    "mujoco_v14_15_multiseed": _mujoco_v14_15_multiseed_facts,
    "mujoco_v14_16_mechanism": _mujoco_v14_16_mechanism_facts,
    "mujoco_v14_29_portfolio": _mujoco_v14_29_portfolio_facts,
    "mujoco_v15_distillation": _mujoco_v15_distillation_facts,
    "mujoco_v16_gauge_training": _mujoco_v16_gauge_training_facts,
    "mujoco_v16_1_audit_gauge_paired": (
        _mujoco_v16_1_audit_gauge_paired_facts
    ),
    "mujoco_v16_2_macro_hold_gauge": _mujoco_v16_2_macro_hold_gauge_facts,
    "opaque_legacy": lambda paths: {
        "decision_status": "excluded_legacy",
        "artifact_count": len(paths),
    },
}


def _paper_eligibility(record: dict[str, Any]) -> tuple[bool, bool]:
    disposition = str(record["paper_disposition"])
    stage = str(record["evidence_stage"])
    decision = str(record["decision"])
    integrity = str(record["integrity_status"])
    selection = str(record["selection_access"])
    confirmatory_ready = bool(
        stage == "confirmatory"
        and integrity == "valid"
        and selection == "frozen_before_heldout"
        and decision in CONFIRMATORY_DECISIONS
    )
    reportable = bool(
        confirmatory_ready
        and disposition in {
            "positive_main_or_si",
            "mixed_or_negative_main_or_si",
        }
    )
    positive = bool(
        reportable
        and disposition == "positive_main_or_si"
        and decision == "supported"
    )
    return reportable, positive


def validate_registry(
    registry: dict[str, Any], repository_root: Path
) -> list[dict[str, Any]]:
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("authoritative evidence registry schema mismatch")
    policy = registry.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("registry policy is missing")
    if policy.get("unregistered_artifacts") != "excluded":
        raise ValueError("the authoritative registry must fail closed")
    records = registry.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("the authoritative registry has no records")
    seen: set[str] = set()
    enriched: list[dict[str, Any]] = []
    for source in records:
        if not isinstance(source, dict):
            raise ValueError("registry records must be JSON objects")
        record = dict(source)
        evidence_id = str(record.get("evidence_id", ""))
        if not evidence_id or evidence_id in seen:
            raise ValueError("registry evidence_id values must be unique")
        seen.add(evidence_id)
        if record.get("evidence_stage") not in EVIDENCE_STAGES:
            raise ValueError(f"{evidence_id}: invalid evidence stage")
        if record.get("paper_disposition") not in PAPER_DISPOSITIONS:
            raise ValueError(f"{evidence_id}: invalid paper disposition")
        parser_name = str(record.get("parser", ""))
        if parser_name not in PARSERS:
            raise ValueError(f"{evidence_id}: unknown evidence parser")
        paths = _verify_artifacts(record, Path(repository_root))
        facts = PARSERS[parser_name](paths)
        if str(facts["decision_status"]) != str(record["decision"]):
            raise ValueError(f"{evidence_id}: registered decision drifted")
        if (
            "integrity_status" in facts
            and str(facts["integrity_status"])
            != str(record["integrity_status"])
        ):
            raise ValueError(f"{evidence_id}: registered integrity drifted")
        reportable, positive = _paper_eligibility(record)
        disposition = str(record["paper_disposition"])
        if disposition == "positive_main_or_si" and not positive:
            raise ValueError(
                f"{evidence_id}: positive disposition lacks confirmatory support"
            )
        if disposition == "mixed_or_negative_main_or_si" and not reportable:
            raise ValueError(
                f"{evidence_id}: reportable negative/mixed result is not valid"
            )
        if disposition == "development_only" and record["evidence_stage"] != "development":
            raise ValueError(f"{evidence_id}: development disposition/stage mismatch")
        if disposition == "excluded_legacy" and reportable:
            raise ValueError(f"{evidence_id}: legacy evidence cannot be reportable")
        record.update({
            "artifact_hashes_verified": True,
            "manuscript_reportable": reportable,
            "positive_claim_supported": positive,
            "facts": facts,
        })
        enriched.append(record)
    return enriched


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "evidence_id",
        "claim_id",
        "domain",
        "evidence_stage",
        "decision",
        "paper_disposition",
        "manuscript_reportable",
        "positive_claim_supported",
        "allowed_wording",
        "forbidden_wording",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def build_registry_outputs(
    *,
    registry_path: Path,
    repository_root: Path,
    output_dir: Path,
    md_output: Path,
) -> dict[str, Any]:
    registry = load_registry(Path(registry_path))
    records = validate_registry(registry, Path(repository_root))
    output_dir.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "registry_sha256": _sha256(Path(registry_path)),
        "policy": registry["policy"],
        "record_count": len(records),
        "reportable_record_count": sum(
            bool(row["manuscript_reportable"]) for row in records
        ),
        "positive_supported_record_count": sum(
            bool(row["positive_claim_supported"]) for row in records
        ),
        "mixed_or_negative_record_count": sum(
            row["paper_disposition"] == "mixed_or_negative_main_or_si"
            for row in records
        ),
        "development_record_count": sum(
            row["paper_disposition"] == "development_only" for row in records
        ),
        "excluded_legacy_record_count": sum(
            row["paper_disposition"] == "excluded_legacy" for row in records
        ),
        "records": records,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "records.csv", records)
    lines = [
        "# Freq-HRL Authoritative Evidence Ledger",
        "",
        f"Date: {registry.get('snapshot_date', 'unknown')}",
        "",
        "This is the only manuscript claim ledger. Unregistered artifacts and the old independent claim generators are excluded by default.",
        "",
        "| Evidence | Domain | Stage | Decision | Paper use | Positive claim |",
        "|---|---|---|---|---|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['evidence_id']} | {row['domain']} | "
            f"{row['evidence_stage']} | {row['decision']} | "
            f"{row['paper_disposition']} | "
            f"{str(bool(row['positive_claim_supported'])).lower()} |"
        )
    lines.extend(["", "## Allowed Wording", ""])
    for row in records:
        lines.extend([
            f"### {row['evidence_id']}",
            "",
            str(row["allowed_wording"]),
            "",
            f"Forbidden: {row['forbidden_wording']}",
            "",
        ])
    report = "\n".join(lines)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    md_output.write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=DEFAULT_REPOSITORY_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    summary = build_registry_outputs(
        registry_path=args.registry,
        repository_root=args.repository_root,
        output_dir=args.output_dir,
        md_output=args.md_output,
    )
    print(
        "authoritative_evidence_registry "
        f"records={summary['record_count']} "
        f"reportable={summary['reportable_record_count']} "
        f"positive={summary['positive_supported_record_count']}"
    )


if __name__ == "__main__":
    main()
