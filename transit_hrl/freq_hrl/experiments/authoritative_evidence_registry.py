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
MUJOCO_V17_ZERO_DC_PLAN_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v17_zero_dc_plan_development_v1"
)
MUJOCO_V17_1_HEADROOM_HOMOTOPY_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v17_1_headroom_homotopy_development_v1"
)
MUJOCO_V17_2_SMOOTH_MACRO_GAUGE_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v17_2_smooth_macro_gauge_development_v1"
)
MUJOCO_V17_3_AUDIT_OPTIMAL_MACRO_GAUGE_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v17_3_audit_optimal_macro_gauge_development_v1"
)
MUJOCO_V17_4_STREAMING_AUDIT_PROJECTION_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v17_4_streaming_audit_projection_development_v1"
)
MUJOCO_V17_5_FEASIBILITY_DIAGNOSTIC_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v17_5_feasibility_diagnostic_development_v1"
)
MUJOCO_V17_6_FULL_HORIZON_ORACLE_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v17_6_full_horizon_oracle_development_v1"
)
MUJOCO_V17_8_CAUSAL_FIR_DISTILLATION_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v17_8_causal_fir_distillation_development_v1"
)
MUJOCO_V17_9_PREFIX_HPF_FIR_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v17_9_prefix_hpf_fir_development_v1"
)
MUJOCO_V17_10_HORIZON_RESERVOIR_FIR_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v17_10_horizon_reservoir_fir_development_v1"
)
MUJOCO_V17_11_FRACTIONAL_RESERVOIR_FIR_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v17_11_fractional_reservoir_fir_development_v1"
)
MUJOCO_V17_12_NEAREST_FEASIBLE_ACTION_ORACLE_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v17_12_nearest_feasible_action_oracle_development_v1"
)
MUJOCO_V17_13_CAUSAL_ACTOR_ADAPTER_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v17_13_causal_actor_adapter_development_v1"
)
MUJOCO_V17_14_EXHAUSTIVE_ACTOR_ORACLE_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v17_14_exhaustive_actor_oracle_development_v1"
)
MUJOCO_V18_1_STATE_ACTOR_DATASET_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v18_1_state_actor_dataset_development_v1"
)
MUJOCO_V18_2_STATE_CONDITIONED_ACTOR_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v18_2_state_conditioned_actor_development_v1"
)
MUJOCO_V18_3_CAUSAL_JOINT_PROJECTION_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v18_3_causal_joint_projection_development_v1"
)
MUJOCO_V18_4_RECEDING_JOINT_PROJECTION_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v18_4_receding_joint_projection_development_v1"
)
MUJOCO_V18_5_ACTOR_FLOOR_SIGNAL_SCHEMA_VERSION = (
    "freq_hrl_mujoco_v18_5_actor_floor_signal_development_v1"
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


def _mujoco_v17_zero_dc_plan_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    gates = dict(decision.get("gate_counts") or {})
    environments = dict(decision.get("environment_results") or {})
    expected_gates = {
        "trained_checkpoint": 9,
        "reward_noninferior": 3,
        "smooth_upper_ablation": 7,
        "candidate_upper_hf_reduction": 5,
        "candidate_upper_hf_budget": 6,
        "raw_lower_lf_reduction_vs_smooth": 9,
        "raw_lower_lf_reduction_vs_latent": 9,
        "raw_joint_merit_reduction": 9,
        "complete_macro_zero_sum": 9,
        "projection_active": 9,
        "responsibility_reconstruction_exact": 9,
        "all_cell_gates": 1,
    }
    expected_supported = {
        "HalfCheetah-v5": 1,
        "Hopper-v5": 0,
        "Walker2d-v5": 0,
    }
    task_status = dict(decision.get("task_status_counts") or {})
    tasks = list(decision.get("scheduler_tasks") or [])
    if (
        decision.get("schema_version")
        != MUJOCO_V17_ZERO_DC_PLAN_SCHEMA_VERSION
        or decision.get("status") != "zero_dc_plan_screen_not_supported"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != "raw_action_frequency_architecture_development_not_confirmatory"
        or int(decision.get("optimizer_seed_count", -1)) != 3
        or int(decision.get("environment_count", -1)) != 3
        or int(decision.get("training_cell_count", -1)) != 27
        or int(decision.get("paired_analysis_cell_count", -1)) != 9
        or int(decision.get("scheduler_successful_task_count", -1)) != 27
        or int(decision.get("scheduler_attempt_count", -1)) != 63
        or len(tasks) != 27
        or len(set(tasks)) != 27
        or task_status != {"done": 27, "failed": 36, "cancelled": 0}
        or decision.get("failed_attempt_exit_codes") != {"0": 36}
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
        or "`zero_dc_plan_screen_not_supported`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError("MuJoCo v17 zero-DC plan decision no longer matches")
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "optimizer_seed_count": 3,
        "environment_count": 3,
        "training_cell_count": 27,
        "paired_analysis_cell_count": 9,
        "gate_counts": expected_gates,
        "task_status_counts": task_status,
        "scheduler_attempt_count": 63,
        "all_failed_attempts_exit_zero": True,
        "support_gate": False,
    }


def _mujoco_v17_1_headroom_homotopy_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    candidates = dict(decision.get("candidate_results") or {})
    expected_reward_counts = {
        "headroom_exact": 0,
        "headroom_homotopy": 1,
        "headroom_homotopy_promotion_05": 1,
        "headroom_homotopy_promotion_10": 1,
    }
    expected_upper_counts = {
        "headroom_exact": 1,
        "headroom_homotopy": 0,
        "headroom_homotopy_promotion_05": 2,
        "headroom_homotopy_promotion_10": 0,
    }
    expected_joint_counts = {
        "headroom_exact": 3,
        "headroom_homotopy": 3,
        "headroom_homotopy_promotion_05": 2,
        "headroom_homotopy_promotion_10": 2,
    }
    task_status = dict(decision.get("task_status_counts") or {})
    tasks = list(decision.get("scheduler_tasks") or [])
    valid_candidates = bool(
        set(candidates) == set(expected_reward_counts)
        and all(
            row.get("eligible_for_fresh_multiseed") is False
            and dict(row.get("gate_counts") or {}).get(
                "trained_checkpoint"
            ) == 3
            and dict(row.get("gate_counts") or {}).get(
                "reward_noninferior"
            ) == expected_reward_counts[name]
            and dict(row.get("gate_counts") or {}).get(
                "upper_hf_nonworsening"
            ) == expected_upper_counts[name]
            and dict(row.get("gate_counts") or {}).get(
                "raw_lower_lf_reduction"
            ) == 3
            and dict(row.get("gate_counts") or {}).get(
                "raw_lower_lf_reduction_vs_latent"
            ) == 3
            and dict(row.get("gate_counts") or {}).get(
                "raw_joint_merit_reduction"
            ) == expected_joint_counts[name]
            and dict(row.get("gate_counts") or {}).get(
                "complete_macro_zero_sum"
            ) == 3
            and dict(row.get("gate_counts") or {}).get(
                "projection_active"
            ) == 3
            and dict(row.get("gate_counts") or {}).get(
                "responsibility_reconstruction_exact"
            ) == 3
            for name, row in candidates.items()
        )
    )
    if (
        decision.get("schema_version")
        != MUJOCO_V17_1_HEADROOM_HOMOTOPY_SCHEMA_VERSION
        or decision.get("status")
        != "headroom_homotopy_preflight_not_supported"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != "headroom_homotopy_architecture_development_not_confirmatory"
        or int(decision.get("optimizer_seed_count", -1)) != 1
        or int(decision.get("environment_count", -1)) != 3
        or int(decision.get("training_cell_count", -1)) != 15
        or int(decision.get("paired_candidate_cell_count", -1)) != 12
        or int(decision.get("scheduler_successful_task_count", -1)) != 15
        or int(decision.get("scheduler_attempt_count", -1)) != 15
        or len(tasks) != 15
        or len(set(tasks)) != 15
        or task_status != {"done": 15, "failed": 0, "cancelled": 0}
        or decision.get("selected_arm_for_fresh_multiseed") is not None
        or not valid_candidates
        or len(str(decision.get("frozen_algorithm_revision", ""))) != 40
        or len(str(decision.get("frozen_source_manifest_sha256", ""))) != 64
        or report is None
        or "`headroom_homotopy_preflight_not_supported`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError("MuJoCo v17.1 headroom-homotopy decision no longer matches")
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "optimizer_seed_count": 1,
        "environment_count": 3,
        "training_cell_count": 15,
        "paired_candidate_cell_count": 12,
        "reward_noninferiority_counts": expected_reward_counts,
        "upper_hf_nonworsening_counts": expected_upper_counts,
        "joint_merit_reduction_counts": expected_joint_counts,
        "task_status_counts": task_status,
        "selected_arm_for_fresh_multiseed": None,
        "support_gate": False,
    }


def _mujoco_v17_2_smooth_macro_gauge_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    mechanics = dict(decision.get("paired_mechanics") or {})
    alpha_results = dict(decision.get("alpha_results") or {})
    expected_frequency_counts = {
        "alpha_005": {
            "upper": 2,
            "lower": 0,
            "joint": 0,
            "bounded": 3,
        },
        "alpha_010": {
            "upper": 1,
            "lower": 0,
            "joint": 0,
            "bounded": 2,
        },
        "alpha_020": {
            "upper": 2,
            "lower": 0,
            "joint": 0,
            "bounded": 3,
        },
    }
    valid_alpha_results = bool(
        set(alpha_results) == set(expected_frequency_counts)
        and all(
            row.get("eligible_for_leakage_active_multiseed") is False
            and int(row.get("supported_environment_count", -1)) == 0
            and int(row.get("upper_hf_reduction_environment_count", -1))
            == expected_frequency_counts[name]["upper"]
            and int(row.get("lower_lf_reduction_environment_count", -1))
            == expected_frequency_counts[name]["lower"]
            and int(row.get("joint_merit_reduction_environment_count", -1))
            == expected_frequency_counts[name]["joint"]
            and int(
                row.get("component_projection_bounded_environment_count", -1)
            ) == expected_frequency_counts[name]["bounded"]
            for name, row in alpha_results.items()
        )
    )
    task_status = dict(decision.get("task_status_counts") or {})
    tasks = list(decision.get("scheduler_tasks") or [])
    if (
        decision.get("schema_version")
        != MUJOCO_V17_2_SMOOTH_MACRO_GAUGE_SCHEMA_VERSION
        or decision.get("status")
        != "smooth_macro_gauge_preflight_not_supported"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != "paired_smooth_macro_gauge_development_not_confirmatory"
        or int(decision.get("optimizer_seed_count", -1)) != 1
        or int(decision.get("environment_count", -1)) != 3
        or int(decision.get("alpha_count", -1)) != 3
        or int(decision.get("training_cell_count", -1)) != 9
        or int(decision.get("heldout_path_count_per_intervention", -1)) != 40
        or int(decision.get("paired_path_count", -1)) != 360
        or int(decision.get("evaluation_row_count", -1)) != 720
        or int(decision.get("scheduler_successful_task_count", -1)) != 9
        or int(decision.get("scheduler_attempt_count", -1)) != 9
        or len(tasks) != 9
        or len(set(tasks)) != 9
        or task_status != {"done": 9, "failed": 0, "cancelled": 0}
        or decision.get("selected_alpha_for_leakage_active_multiseed")
        is not None
        or any(
            int(mechanics.get(key, -1)) != 9
            for key in (
                "reward_trace_exact_cells",
                "executed_action_trace_exact_cells",
                "latent_policy_trace_exact_cells",
                "reward_numeric_exact_cells",
                "latent_metrics_exact_cells",
                "router_reconstruction_exact_cells",
                "responsibility_reconstruction_exact_cells",
                "explicit_protocol_structure_valid_cells",
            )
        )
        or int(mechanics.get("legacy_protocol_invalid_candidate_rows", -1))
        != 2
        or not valid_alpha_results
        or len(str(decision.get("frozen_algorithm_revision", ""))) != 40
        or len(str(decision.get("frozen_source_manifest_sha256", ""))) != 64
        or len(str(decision.get("analysis_repair_revision", ""))) != 40
        or report is None
        or "`smooth_macro_gauge_preflight_not_supported`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError("MuJoCo v17.2 smooth-macro-gauge decision no longer matches")
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "optimizer_seed_count": 1,
        "environment_count": 3,
        "alpha_count": 3,
        "training_cell_count": 9,
        "paired_path_count": 360,
        "paired_mechanics": mechanics,
        "frequency_gate_counts": expected_frequency_counts,
        "task_status_counts": task_status,
        "selected_alpha_for_leakage_active_multiseed": None,
        "support_gate": False,
    }


def _mujoco_v17_3_audit_optimal_macro_gauge_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    mechanics = dict(decision.get("paired_mechanics") or {})
    gate_counts = dict(decision.get("frequency_gate_counts") or {})
    environment_results = dict(decision.get("environment_results") or {})
    task_status = dict(decision.get("task_status_counts") or {})
    tasks = list(decision.get("scheduler_tasks") or [])
    expected_gate_counts = {
        "upper_hf_reduction": 0,
        "lower_lf_reduction": 2,
        "joint_merit_reduction": 2,
    }
    expected_reductions = {
        "HalfCheetah-v5": (
            -10.308916864436272,
            -0.4828350401935079,
            -2.563558020775433,
        ),
        "Hopper-v5": (
            -0.9473236652368211,
            0.6652080326214834,
            0.5648472461436531,
        ),
        "Walker2d-v5": (
            -7.334877321297106,
            0.7264887815443557,
            0.72506933330081,
        ),
    }
    valid_environment_results = bool(
        set(environment_results) == set(expected_reductions)
        and all(
            bool(row.get("supported")) is False
            and abs(
                float(row.get("upper_hf_relative_reduction", float("inf")))
                - reductions[0]
            ) <= 1e-12
            and abs(
                float(row.get("lower_lf_relative_reduction", float("inf")))
                - reductions[1]
            ) <= 1e-12
            and abs(
                float(row.get("joint_merit_relative_reduction", float("inf")))
                - reductions[2]
            ) <= 1e-12
            and float(row.get("reward_difference", float("inf"))) == 0.0
            for name, reductions in expected_reductions.items()
            for row in [environment_results[name]]
        )
    )
    if (
        decision.get("schema_version")
        != MUJOCO_V17_3_AUDIT_OPTIMAL_MACRO_GAUGE_SCHEMA_VERSION
        or decision.get("status")
        != "audit_optimal_macro_gauge_preflight_not_supported"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != "paired_audit_optimal_macro_gauge_development_not_confirmatory"
        or int(decision.get("optimizer_seed_count", -1)) != 1
        or int(decision.get("environment_count", -1)) != 3
        or int(decision.get("training_cell_count", -1)) != 3
        or int(decision.get("heldout_path_count_per_intervention", -1)) != 40
        or int(decision.get("paired_path_count", -1)) != 120
        or int(decision.get("evaluation_row_count", -1)) != 240
        or int(decision.get("scheduler_successful_task_count", -1)) != 3
        or int(decision.get("scheduler_attempt_count", -1)) != 3
        or len(tasks) != 3
        or len(set(tasks)) != 3
        or task_status != {"done": 3, "failed": 0, "cancelled": 0}
        or decision.get("eligible_for_leakage_active_multiseed") is not False
        or any(
            int(mechanics.get(key, -1)) != 3
            for key in (
                "reward_trace_exact_cells",
                "executed_action_trace_exact_cells",
                "latent_policy_trace_exact_cells",
                "reward_numeric_exact_cells",
                "latent_metrics_exact_cells",
                "router_reconstruction_exact_cells",
                "responsibility_reconstruction_exact_cells",
                "component_projection_bounded_cells",
                "explicit_protocol_structure_valid_cells",
            )
        )
        or gate_counts != expected_gate_counts
        or not valid_environment_results
        or len(str(decision.get("frozen_algorithm_revision", ""))) != 40
        or len(str(decision.get("frozen_source_manifest_sha256", ""))) != 64
        or report is None
        or "`audit_optimal_macro_gauge_preflight_not_supported`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError(
            "MuJoCo v17.3 audit-optimal-macro-gauge decision no longer matches"
        )
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "optimizer_seed_count": 1,
        "environment_count": 3,
        "training_cell_count": 3,
        "paired_path_count": 120,
        "paired_mechanics": mechanics,
        "frequency_gate_counts": expected_gate_counts,
        "environment_results": environment_results,
        "task_status_counts": task_status,
        "eligible_for_leakage_active_multiseed": False,
        "support_gate": False,
    }


def _mujoco_v17_4_streaming_audit_projection_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    mechanics = dict(decision.get("paired_mechanics") or {})
    gate_counts = dict(decision.get("frequency_gate_counts") or {})
    environment_results = dict(decision.get("environment_results") or {})
    task_status = dict(decision.get("task_status_counts") or {})
    tasks = list(decision.get("scheduler_tasks") or [])
    expected_gate_counts = {
        "upper_hf_absolute_budget": 3,
        "lower_lf_absolute_budget": 1,
        "upper_budget_feasibility": 2,
        "lower_lf_reduction": 3,
        "joint_merit_reduction": 3,
    }
    expected_results = {
        "HalfCheetah-v5": (
            0.0012511922691996196,
            0.0049767574431776075,
            0.00894753176621209,
            0.0035564969601090657,
            -2.977612047076674,
            0.6025164198310862,
            0.41237218007554444,
            0.9755083333333333,
            0.6462874999999999,
            False,
        ),
        "Hopper-v5": (
            0.009774690046568857,
            0.00454284863807549,
            0.0607934625113658,
            0.0221787328835983,
            0.5352437144878947,
            0.6351789819595847,
            0.6291243620525505,
            1.0,
            0.2827287197570771,
            False,
        ),
        "Walker2d-v5": (
            0.00017910787017102884,
            0.0010765051333661018,
            0.0176855280861979,
            0.0016908052682744641,
            -5.010373147411974,
            0.9043961107616572,
            0.8804663410523197,
            1.0,
            0.9026213838434728,
            True,
        ),
    }
    metric_keys = (
        "control_upper_hf_power",
        "candidate_upper_hf_power",
        "control_lower_lf_power",
        "candidate_lower_lf_power",
        "upper_hf_relative_reduction",
        "lower_lf_relative_reduction",
        "joint_merit_relative_reduction",
        "mean_upper_budget_feasible_rate",
        "mean_lower_budget_satisfied_rate",
    )
    valid_environment_results = bool(
        set(environment_results) == set(expected_results)
        and all(
            all(
                abs(float(row.get(key, float("inf"))) - expected[index])
                <= 1e-12
                for index, key in enumerate(metric_keys)
            )
            and bool(row.get("supported")) is expected[-1]
            and float(row.get("reward_difference", float("inf"))) == 0.0
            for name, expected in expected_results.items()
            for row in [environment_results[name]]
        )
    )
    if (
        decision.get("schema_version")
        != MUJOCO_V17_4_STREAMING_AUDIT_PROJECTION_SCHEMA_VERSION
        or decision.get("status")
        != "streaming_audit_projection_preflight_not_supported"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != "paired_streaming_audit_projection_development_not_confirmatory"
        or decision.get("frozen_algorithm_revision")
        != "91451c1ee0b3bbc488152fc1b4994a3ed5e0436c"
        or decision.get("frozen_source_manifest_sha256")
        != "f371ecffe5182d83c778ba42033ae7f44f50881aa635924ff8f406eaa7927ab6"
        or int(decision.get("optimizer_seed_count", -1)) != 1
        or int(decision.get("environment_count", -1)) != 3
        or int(decision.get("training_cell_count", -1)) != 3
        or int(decision.get("heldout_path_count_per_intervention", -1)) != 40
        or int(decision.get("paired_path_count", -1)) != 120
        or int(decision.get("evaluation_row_count", -1)) != 240
        or int(decision.get("scheduler_successful_task_count", -1)) != 3
        or int(decision.get("scheduler_attempt_count", -1)) != 3
        or tasks != ["t85521", "t85522", "t85523"]
        or task_status != {"done": 3, "failed": 0, "cancelled": 0}
        or decision.get("eligible_for_streaming_projection_multiseed")
        is not False
        or any(
            int(mechanics.get(key, -1)) != 3
            for key in (
                "reward_trace_exact_cells",
                "executed_action_trace_exact_cells",
                "latent_policy_trace_exact_cells",
                "reward_numeric_exact_cells",
                "latent_metrics_exact_cells",
                "router_reconstruction_exact_cells",
                "responsibility_reconstruction_exact_cells",
                "explicit_protocol_structure_valid_cells",
            )
        )
        or gate_counts != expected_gate_counts
        or not valid_environment_results
        or report is None
        or "`streaming_audit_projection_preflight_not_supported`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError(
            "MuJoCo v17.4 streaming-audit-projection decision no longer matches"
        )
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "optimizer_seed_count": 1,
        "environment_count": 3,
        "training_cell_count": 3,
        "paired_path_count": 120,
        "paired_mechanics": mechanics,
        "frequency_gate_counts": expected_gate_counts,
        "environment_results": environment_results,
        "task_status_counts": task_status,
        "eligible_for_streaming_projection_multiseed": False,
        "support_gate": False,
    }


def _mujoco_v17_5_feasibility_diagnostic_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    environment_results = dict(decision.get("environment_results") or {})
    improvement_counts = dict(
        decision.get("endpoint_improvement_counts") or {}
    )
    task_status = dict(decision.get("task_status_counts") or {})
    expected_counts = {
        "episode_return": 3,
        "upper_hf_power": 0,
        "lower_lf_drift": 1,
        "lower_budget_violation": 1,
        "joint_budget_feasible_rate": 0,
    }
    required_metrics = (
        "v17_4_episode_return",
        "v17_5_episode_return",
        "v17_4_upper_hf_power",
        "v17_5_upper_hf_power",
        "v17_4_lower_lf_drift",
        "v17_5_lower_lf_drift",
        "v17_4_lower_budget_violation_rms",
        "v17_5_lower_budget_violation_rms",
        "v17_4_joint_budget_feasible_rate",
        "v17_5_joint_budget_feasible_rate",
        "v17_4_local_budget_regret_rms",
        "v17_5_local_budget_regret_rms",
    )
    valid_results = bool(
        set(environment_results)
        == {"HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"}
        and all(
            all(
                key in row and float(row[key]) == float(row[key])
                for key in required_metrics
            )
            for row in environment_results.values()
        )
        and all(
            float(row["v17_5_local_budget_regret_rms"]) <= 1e-12
            for row in environment_results.values()
        )
    )
    if (
        decision.get("schema_version")
        != MUJOCO_V17_5_FEASIBILITY_DIAGNOSTIC_SCHEMA_VERSION
        or decision.get("status")
        != "greedy_feasibility_projection_not_advanced"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != "rejected_v17_4_path_reuse_development_diagnostic_not_confirmatory"
        or decision.get("diagnostic_code_revision")
        != "50fa967174d49ccdf8134df9471d636c3cc7d30b"
        or decision.get("diagnostic_source_manifest_sha256")
        != "e1870eb33ddb034120eb0c42e5b1164845cd8f5a777edbc1961bf9d947f75357"
        or int(decision.get("optimizer_seed_count", -1)) != 1
        or int(decision.get("environment_count", -1)) != 3
        or int(decision.get("reused_path_count_per_router", -1)) != 120
        or int(decision.get("evaluation_row_count", -1)) != 240
        or list(decision.get("scheduler_tasks") or [])
        != ["t85547", "t85548", "t85549"]
        or task_status != {"done": 3, "failed": 0, "cancelled": 0}
        or int(decision.get("legacy_replay_exact_environment_count", -1)) != 3
        or int(
            decision.get("cross_router_trace_divergent_environment_count", -1)
        ) != 3
        or int(
            decision.get("local_budget_regret_eliminated_environment_count", -1)
        ) != 3
        or improvement_counts != expected_counts
        or decision.get("eligible_for_fresh_v17_5_preflight") is not False
        or not valid_results
        or report is None
        or "`greedy_feasibility_projection_not_advanced`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError(
            "MuJoCo v17.5 feasibility diagnostic decision no longer matches"
        )
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "optimizer_seed_count": 1,
        "environment_count": 3,
        "reused_path_count_per_router": 120,
        "legacy_replay_exact_environment_count": 3,
        "cross_router_trace_divergent_environment_count": 3,
        "local_budget_regret_eliminated_environment_count": 3,
        "endpoint_improvement_counts": expected_counts,
        "environment_results": environment_results,
        "task_status_counts": task_status,
        "eligible_for_fresh_v17_5_preflight": False,
        "support_gate": False,
    }


def _mujoco_v17_6_full_horizon_oracle_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    environment_results = dict(decision.get("environment_results") or {})
    overall = dict(decision.get("overall") or {})
    numerical_audit = dict(decision.get("numerical_audit") or {})
    task_status = dict(decision.get("task_status_counts") or {})
    actor_floor_cases = list(decision.get("actor_floor_cases") or [])
    expected_environment_counts = {
        "HalfCheetah-v5": (40, 0, 40, 40, 0),
        "Hopper-v5": (40, 0, 33, 33, 7),
        "Walker2d-v5": (40, 32, 40, 8, 0),
    }
    observed_environment_counts = {
        environment: (
            int(row.get("path_count", -1)),
            int(row.get("baseline_joint_feasible_path_count", -1)),
            int(row.get("oracle_joint_feasible_path_count", -1)),
            int(row.get("recoverable_path_count", -1)),
            int(row.get("oracle_infeasible_path_count", -1)),
        )
        for environment, row in environment_results.items()
    }
    actor_case_keys = {
        (
            str(row.get("environment", "")),
            str(row.get("disturbance_mode", "")),
            int(row.get("evaluation_seed", -1)),
        )
        for row in actor_floor_cases
    }
    expected_actor_case_keys = {
        ("Hopper-v5", "low_frequency", 294864529),
        ("Hopper-v5", "standard", 294864529),
        ("Hopper-v5", "high_frequency", 294864529),
        ("Hopper-v5", "ood_chirp", 2802248628),
        ("Hopper-v5", "ood_chirp", 294864529),
        ("Hopper-v5", "mixed", 2802248628),
        ("Hopper-v5", "mixed", 294864529),
    }
    lower_budget = float(
        dict(decision.get("registered_budgets") or {}).get(
            "lower_lpf32_power", float("nan")
        )
    )
    if (
        decision.get("schema_version")
        != MUJOCO_V17_6_FULL_HORIZON_ORACLE_SCHEMA_VERSION
        or decision.get("status")
        != "mixed_router_recoverability_and_actor_floor"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != "rejected_v17_4_frozen_total_path_oracle_not_confirmatory"
        or decision.get("frozen_oracle_revision")
        != "5a6efa2dccb441334b55cabd25556fc78b55ad3b"
        or decision.get("frozen_source_manifest_sha256")
        != "8d001993b7da2913052ce9ee91ff329410592dede1c8b2aae48da7a1054bc0d1"
        or int(decision.get("source_checkpoint_optimizer_seed_count", -1)) != 1
        or int(decision.get("environment_count", -1)) != 3
        or int(decision.get("disturbance_mode_count", -1)) != 5
        or int(decision.get("evaluation_seed_count_per_environment_mode", -1))
        != 8
        or int(decision.get("path_count", -1)) != 120
        or int(decision.get("legacy_replay_exact_path_count", -1)) != 120
        or int(decision.get("scheduler_successful_task_count", -1)) != 120
        or int(decision.get("scheduler_attempt_count", -1)) != 120
        or decision.get("scheduler_task_id_first") != "t85559"
        or decision.get("scheduler_task_id_last") != "t85678"
        or list(decision.get("scheduler_nodes") or []) != ["node003"]
        or task_status != {"done": 120, "failed": 0, "cancelled": 0}
        or observed_environment_counts != expected_environment_counts
        or int(overall.get("baseline_joint_feasible_path_count", -1)) != 32
        or int(overall.get("oracle_joint_feasible_path_count", -1)) != 113
        or int(overall.get("recoverable_path_count", -1)) != 81
        or int(overall.get("oracle_infeasible_path_count", -1)) != 7
        or int(
            overall.get("upper_budget_physically_infeasible_path_count", -1)
        ) != 0
        or actor_case_keys != expected_actor_case_keys
        or len(actor_floor_cases) != 7
        or not all(
            float(row.get("lower_power_at_upper_constrained_floor", 0.0))
            > lower_budget
            for row in actor_floor_cases
        )
        or float(numerical_audit.get("bound_violation_max", float("inf")))
        > 1e-10
        or float(
            numerical_audit.get("reconstruction_error_max", float("inf"))
        ) > 1e-12
        or float(numerical_audit.get("kkt_residual_max", float("inf")))
        > 1e-5
        or float(
            numerical_audit.get("solver_optimality_max", float("inf"))
        ) > 1e-5
        or decision.get("eligible_for_causal_router_rebuild") is not True
        or decision.get("eligible_for_actor_feasibility_rebuild") is not True
        or decision.get("eligible_for_confirmatory_claim") is not False
        or report is None
        or "`mixed_router_recoverability_and_actor_floor`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError(
            "MuJoCo v17.6 full-horizon oracle decision no longer matches"
        )
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "source_checkpoint_optimizer_seed_count": 1,
        "environment_count": 3,
        "disturbance_mode_count": 5,
        "evaluation_seed_count_per_environment_mode": 8,
        "path_count": 120,
        "legacy_replay_exact_path_count": 120,
        "overall": overall,
        "environment_results": environment_results,
        "actor_floor_case_count": 7,
        "actor_floor_cases": actor_floor_cases,
        "numerical_audit": numerical_audit,
        "task_status_counts": task_status,
        "eligible_for_causal_router_rebuild": True,
        "eligible_for_actor_feasibility_rebuild": True,
        "eligible_for_confirmatory_claim": False,
        "support_gate": False,
    }


def _mujoco_v17_8_causal_fir_distillation_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    selected = dict(decision.get("selected_candidate") or {})
    diagnostic = dict(
        decision.get("unconstrained_gain_one_diagnostic") or {}
    )
    scheduler = dict(decision.get("scheduler") or {})
    gate = dict(decision.get("advancement_gate") or {})
    environment_results = dict(selected.get("environment_results") or {})
    observed_recovery = {
        environment: int(row.get("recovered_failure_count", -1))
        for environment, row in environment_results.items()
    }
    if (
        decision.get("schema_version")
        != MUJOCO_V17_8_CAUSAL_FIR_DISTILLATION_SCHEMA_VERSION
        or decision.get("status")
        != "grouped_causal_fir_stopped_before_fresh_path_access"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != (
            "reused_path_grouped_selection_then_fresh_path_validation_"
            "not_confirmatory"
        )
        or decision.get("frozen_core_revision")
        != "a120651572e7a35614527bd2be18bd3b52f0c14f"
        or decision.get("frozen_source_manifest_sha256")
        != "e6819fe80ae428755ffd355dbb8c22eece71a7dff9a7da8c750b8029d4b072c7"
        or int(decision.get("source_checkpoint_optimizer_seed_count", -1))
        != 1
        or int(decision.get("environment_count", -1)) != 3
        or int(decision.get("disturbance_mode_count", -1)) != 5
        or int(decision.get("grouped_seed_fold_count", -1)) != 8
        or int(decision.get("path_count", -1)) != 120
        or int(decision.get("candidate_count", -1)) != 80
        or scheduler.get("dataset_task_id_first") != "t85715"
        or scheduler.get("dataset_task_id_last") != "t85834"
        or int(scheduler.get("dataset_successful_task_count", -1)) != 120
        or scheduler.get("selection_task_id") != "t85836"
        or scheduler.get("selection_task_status") != "done"
        or list(scheduler.get("nodes") or []) != ["node003"]
        or scheduler.get("slurm_used") is not False
        or selected.get("candidate_id")
        != "fir_w64_ridge1e-05_gain0.80"
        or int(selected.get("valid_path_count", -1)) != 120
        or int(selected.get("upper_budget_path_count", -1)) != 120
        or int(selected.get("oracle_recoverable_failure_count", -1)) != 81
        or int(selected.get("recovered_failure_count", -1)) != 7
        or int(selected.get("baseline_feasible_path_count", -1)) != 32
        or int(selected.get("preserved_baseline_feasible_path_count", -1))
        != 0
        or observed_recovery
        != {"HalfCheetah-v5": 7, "Hopper-v5": 0, "Walker2d-v5": 0}
        or diagnostic.get("candidate_id")
        != "fir_w48_ridge1e-05_gain1.00"
        or int(diagnostic.get("upper_budget_path_count", -1)) != 90
        or int(diagnostic.get("recovered_failure_count", -1)) != 58
        or int(diagnostic.get("preserved_baseline_feasible_path_count", -1))
        != 32
        or gate.get("all_paths_numerically_and_physically_valid") is not True
        or gate.get("all_paths_meet_endpoint_upper_budget") is not True
        or gate.get("total_recovery_gate") is not False
        or gate.get("walker_baseline_feasible_preservation_gate") is not False
        or int(decision.get("fresh_validation_seed_count", -1)) != 8
        or decision.get("fresh_validation_paths_accessed") is not False
        or decision.get("fresh_path_access_allowed") is not False
        or decision.get("support_gate") is not False
        or report is None
        or "`grouped_causal_fir_stopped_before_fresh_path_access`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError(
            "MuJoCo v17.8 causal FIR decision no longer matches"
        )
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "optimizer_seed_count": 1,
        "environment_count": 3,
        "grouped_seed_fold_count": 8,
        "path_count": 120,
        "candidate_count": 80,
        "selected_candidate_id": str(selected["candidate_id"]),
        "selected_upper_budget_path_count": 120,
        "selected_recovered_failure_count": 7,
        "selected_preserved_baseline_feasible_path_count": 0,
        "diagnostic_gain_one_upper_budget_path_count": 90,
        "diagnostic_gain_one_recovered_failure_count": 58,
        "diagnostic_gain_one_preserved_baseline_feasible_path_count": 32,
        "fresh_validation_paths_accessed": False,
        "eligible_for_fresh_path_validation": False,
        "support_gate": False,
    }


def _mujoco_v17_9_prefix_hpf_fir_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    selected = dict(decision.get("selected_candidate") or {})
    scheduler = dict(decision.get("scheduler") or {})
    gate = dict(decision.get("advancement_gate") or {})
    environment_results = dict(selected.get("environment_results") or {})
    observed_recovery = {
        environment: int(row.get("recovered_failure_count", -1))
        for environment, row in environment_results.items()
    }
    if (
        decision.get("schema_version")
        != MUJOCO_V17_9_PREFIX_HPF_FIR_SCHEMA_VERSION
        or decision.get("status")
        != "prefix_hpf_fir_stopped_before_fresh_path_access"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != "post_v17_8_reused_path_prefix_hpf_projection_not_confirmatory"
        or decision.get("frozen_core_revision")
        != "85dc42eaa1518727d6975d8c09faf1345763f28a"
        or decision.get("frozen_source_manifest_sha256")
        != "24d5649b51b7d2ce30d20c7a4b991f70809ee8a92586c20c30d321a3032a44e2"
        or int(decision.get("environment_count", -1)) != 3
        or int(decision.get("disturbance_mode_count", -1)) != 5
        or int(decision.get("grouped_seed_fold_count", -1)) != 8
        or int(decision.get("path_count", -1)) != 120
        or int(decision.get("candidate_count", -1)) != 8
        or scheduler.get("selection_task_id") != "t85838"
        or scheduler.get("selection_task_status") != "done"
        or list(scheduler.get("nodes") or []) != ["node003"]
        or scheduler.get("slurm_used") is not False
        or selected.get("candidate_id")
        != "prefix_hpf_fir_w64_ridge1e-03_gain1.00"
        or int(selected.get("valid_path_count", -1)) != 120
        or int(selected.get("upper_budget_path_count", -1)) != 120
        or int(selected.get("oracle_recoverable_failure_count", -1)) != 81
        or int(selected.get("recovered_failure_count", -1)) != 48
        or int(selected.get("preserved_baseline_feasible_path_count", -1))
        != 32
        or int(selected.get("prefix_infeasible_path_count", -1)) != 0
        or observed_recovery
        != {"HalfCheetah-v5": 40, "Hopper-v5": 0, "Walker2d-v5": 8}
        or gate.get("all_paths_numerically_and_physically_valid") is not True
        or gate.get("all_paths_meet_endpoint_upper_budget") is not True
        or gate.get("mean_lower_power_no_worse_each_environment") is not True
        or gate.get("walker_baseline_feasible_preservation_gate") is not True
        or gate.get("total_recovery_gate") is not False
        or gate.get("environment_recovery_gates") is not False
        or decision.get("fresh_validation_paths_accessed") is not False
        or decision.get("fresh_path_access_allowed") is not False
        or decision.get("support_gate") is not False
        or report is None
        or "`prefix_hpf_fir_stopped_before_fresh_path_access`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError("MuJoCo v17.9 prefix-HPF FIR decision drifted")
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "grouped_seed_fold_count": 8,
        "path_count": 120,
        "candidate_count": 8,
        "selected_upper_budget_path_count": 120,
        "selected_recovered_failure_count": 48,
        "selected_preserved_baseline_feasible_path_count": 32,
        "selected_prefix_infeasible_path_count": 0,
        "recovered_failures_by_environment": observed_recovery,
        "fresh_validation_paths_accessed": False,
        "eligible_for_fresh_path_validation": False,
        "support_gate": False,
    }


def _mujoco_v17_10_horizon_reservoir_fir_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    selected = dict(decision.get("selected_candidate") or {})
    diagnostic = dict(decision.get("largest_reservoir_diagnostic") or {})
    scheduler = dict(decision.get("scheduler") or {})
    if (
        decision.get("schema_version")
        != MUJOCO_V17_10_HORIZON_RESERVOIR_FIR_SCHEMA_VERSION
        or decision.get("status")
        != "horizon_reservoir_fir_stopped_before_fresh_path_access"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != (
            "post_v17_9_reused_path_horizon_reservoir_selection_"
            "not_confirmatory"
        )
        or decision.get("frozen_core_revision")
        != "f849d15c0b8c7f8c0f99e0bdf69f9b892d20da36"
        or decision.get("frozen_source_manifest_sha256")
        != "9e78071e94f6bba8c589fb765dc4378ad9be7268662defd91e9bfc231457c892"
        or int(decision.get("grouped_seed_fold_count", -1)) != 8
        or int(decision.get("path_count", -1)) != 120
        or int(decision.get("candidate_count", -1)) != 32
        or scheduler.get("selection_task_id") != "t85840"
        or scheduler.get("selection_task_status") != "done"
        or list(scheduler.get("nodes") or []) != ["node003"]
        or scheduler.get("slurm_used") is not False
        or selected.get("candidate_id")
        != "reservoir0_fir_w64_ridge1e-03_gain1.00"
        or int(selected.get("valid_path_count", -1)) != 120
        or int(selected.get("upper_budget_path_count", -1)) != 120
        or int(selected.get("recovered_failure_count", -1)) != 48
        or int(selected.get("preserved_baseline_feasible_path_count", -1))
        != 32
        or int(diagnostic.get("energy_reserve_steps", -1)) != 82
        or int(diagnostic.get("valid_path_count", -1)) != 113
        or int(diagnostic.get("upper_budget_path_count", -1)) != 120
        or int(diagnostic.get("recovered_failure_count", -1)) != 63
        or diagnostic.get("recovered_failure_count_by_environment")
        != {"HalfCheetah-v5": 40, "Hopper-v5": 15, "Walker2d-v5": 8}
        or decision.get("fresh_validation_paths_accessed") is not False
        or decision.get("fresh_path_access_allowed") is not False
        or decision.get("support_gate") is not False
        or report is None
        or "`horizon_reservoir_fir_stopped_before_fresh_path_access`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError("MuJoCo v17.10 reservoir decision drifted")
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "grouped_seed_fold_count": 8,
        "path_count": 120,
        "candidate_count": 32,
        "selected_energy_reserve_steps": 0,
        "selected_recovered_failure_count": 48,
        "largest_reservoir_steps": 82,
        "largest_reservoir_valid_path_count": 113,
        "largest_reservoir_recovered_failure_count": 63,
        "fresh_validation_paths_accessed": False,
        "eligible_for_fresh_path_validation": False,
        "support_gate": False,
    }


def _mujoco_v17_11_fractional_reservoir_fir_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    selected = dict(decision.get("selected_candidate") or {})
    diagnostic = dict(decision.get("best_hopper_diagnostic") or {})
    scheduler = dict(decision.get("scheduler") or {})
    gate = dict(decision.get("advancement_gate") or {})
    observed_recovery = dict(
        selected.get("recovered_failure_count_by_environment") or {}
    )
    if (
        decision.get("schema_version")
        != MUJOCO_V17_11_FRACTIONAL_RESERVOIR_FIR_SCHEMA_VERSION
        or decision.get("status")
        != "fractional_reservoir_fir_stops_router_only_development"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != "final_router_only_reused_path_fractional_reservoir_not_confirmatory"
        or decision.get("frozen_core_revision")
        != "1578e24ecc75bc480f1d41803dc13a19e49b5c5f"
        or decision.get("frozen_source_manifest_sha256")
        != "06557c3f016fc7cbd7cd7f9f4f730f9d6700be7f1dfa07ad0f567ac72e83e8c6"
        or int(decision.get("grouped_seed_fold_count", -1)) != 8
        or int(decision.get("path_count", -1)) != 120
        or int(decision.get("candidate_count", -1)) != 40
        or scheduler.get("selection_task_id") != "t85842"
        or scheduler.get("selection_task_status") != "done"
        or list(scheduler.get("nodes") or []) != ["node003"]
        or scheduler.get("slurm_used") is not False
        or selected.get("candidate_id")
        != "fractional_reservoir80_rho0.75_fir_w64_ridge1e-03_gain1.00"
        or int(selected.get("energy_reserve_steps", -1)) != 80
        or float(selected.get("energy_borrow_fraction", -1.0)) != 0.75
        or int(selected.get("valid_path_count", -1)) != 120
        or int(selected.get("upper_budget_path_count", -1)) != 120
        or int(selected.get("oracle_recoverable_failure_count", -1)) != 81
        or int(selected.get("recovered_failure_count", -1)) != 62
        or int(selected.get("preserved_baseline_feasible_path_count", -1))
        != 32
        or observed_recovery
        != {"HalfCheetah-v5": 40, "Hopper-v5": 14, "Walker2d-v5": 8}
        or int(diagnostic.get("valid_path_count", -1)) != 119
        or int(diagnostic.get("recovered_failure_count", -1)) != 64
        or int(diagnostic.get("hopper_recovered_failure_count", -1)) != 16
        or gate.get("all_paths_numerically_and_physically_valid") is not True
        or gate.get("all_paths_meet_endpoint_upper_budget") is not True
        or gate.get("fractional_envelope_feasible_on_all_paths") is not True
        or gate.get("minimum_horizon_certified_on_all_paths") is not True
        or gate.get("total_recovery_gate") is not False
        or gate.get("environment_recovery_gates") is not False
        or decision.get("router_only_development_closed") is not True
        or decision.get("fresh_validation_paths_accessed") is not False
        or decision.get("fresh_path_access_allowed") is not False
        or decision.get("support_gate") is not False
        or report is None
        or "`fractional_reservoir_fir_stops_router_only_development`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError("MuJoCo v17.11 fractional reservoir decision drifted")
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "grouped_seed_fold_count": 8,
        "path_count": 120,
        "candidate_count": 40,
        "selected_energy_reserve_steps": 80,
        "selected_energy_borrow_fraction": 0.75,
        "selected_recovered_failure_count": 62,
        "selected_preserved_baseline_feasible_path_count": 32,
        "recovered_failures_by_environment": observed_recovery,
        "best_diagnostic_valid_path_count": 119,
        "best_diagnostic_hopper_recovered_failure_count": 16,
        "router_only_development_closed": True,
        "fresh_validation_paths_accessed": False,
        "eligible_for_fresh_path_validation": False,
        "support_gate": False,
    }


def _mujoco_v17_12_nearest_feasible_action_oracle_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    scheduler = dict(decision.get("scheduler") or {})
    correction = dict(
        decision.get("frequency_only_actor_floor_correction") or {}
    )
    deployment = dict(
        decision.get("deployment_total_box_diagnostic") or {}
    )
    gate = dict(decision.get("advancement_gate") or {})
    targets = dict(decision.get("server_target_policy") or {})
    expected_gate_keys = {
        "expected_reference_feasible_path_count",
        "expected_actor_floor_path_count",
        "all_frequency_only_targets_feasible",
        "all_reference_feasible_paths_preserved_exactly",
        "all_actor_floor_targets_change_total_action",
        "actor_floor_total_correction_rms_gate",
        "actor_floor_total_correction_abs_gate",
        "all_actor_floor_deployment_targets_feasible",
        "server_target_count_complete",
    }
    if (
        decision.get("schema_version")
        != MUJOCO_V17_12_NEAREST_FEASIBLE_ACTION_ORACLE_SCHEMA_VERSION
        or decision.get("status")
        != "nearest_feasible_targets_authorize_causal_actor_adapter"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != "post_router_boundary_reused_path_actor_target_oracle_not_confirmatory"
        or decision.get("development_protocol_version")
        != "mujoco_v17_12_nearest_feasible_action_v1"
        or decision.get("frozen_core_revision")
        != "3ca8ee6d2709e77af7b7c4d022cbeb83886d0a75"
        or decision.get("frozen_source_manifest_sha256")
        != "718a7dd61ea62fa21b52531299a2e0d2e2b22175891f7a7fbdf03b7f58a5410a"
        or decision.get("source_dataset_run")
        != "mujoco_v17_8_causal_fir_dataset_20260831_r1"
        or int(decision.get("path_count", -1)) != 120
        or int(decision.get("reference_feasible_path_count", -1)) != 113
        or int(decision.get("actor_floor_path_count", -1)) != 7
        or int(decision.get("frequency_target_feasible_path_count", -1))
        != 120
        or int(decision.get(
            "deployment_target_feasible_actor_floor_path_count", -1
        )) != 7
        or scheduler.get("oracle_task_id") != "t85844"
        or scheduler.get("oracle_task_status") != "done"
        or list(scheduler.get("nodes") or []) != ["node003"]
        or scheduler.get("slurm_used") is not False
        or float(correction.get("total_action_rms_mean", -1.0))
        != 0.00234273764738525
        or float(correction.get("total_action_rms_maximum", -1.0))
        != 0.008117855266084743
        or float(correction.get("total_action_abs_maximum", -1.0))
        != 0.03687370520663191
        or deployment.get("selects_actor_target") is not False
        or float(deployment.get("total_action_rms_maximum", -1.0))
        != 0.08110926566037498
        or set(gate) != expected_gate_keys
        or not all(value is True for value in gate.values())
        or int(targets.get("server_target_count", -1)) != 7
        or targets.get("server_targets_pulled_locally") is not False
        or decision.get("fresh_validation_paths_accessed") is not False
        or decision.get("causal_actor_adapter_authorized") is not True
        or decision.get("manuscript_support_gate") is not False
        or report is None
        or "`nearest_feasible_targets_authorize_causal_actor_adapter`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError("MuJoCo v17.12 actor target oracle decision drifted")
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "path_count": 120,
        "reference_feasible_path_count": 113,
        "actor_floor_path_count": 7,
        "frequency_target_feasible_path_count": 120,
        "actor_floor_total_action_rms_mean": float(
            correction["total_action_rms_mean"]
        ),
        "actor_floor_total_action_rms_maximum": float(
            correction["total_action_rms_maximum"]
        ),
        "actor_floor_total_action_abs_maximum": float(
            correction["total_action_abs_maximum"]
        ),
        "server_target_count": 7,
        "fresh_validation_paths_accessed": False,
        "causal_actor_adapter_authorized": True,
        "support_gate": False,
    }


def _mujoco_v17_13_causal_actor_adapter_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    scheduler = dict(decision.get("scheduler") or {})
    target = dict(decision.get("target_audit") or {})
    selected = dict(decision.get("selected_candidate") or {})
    recovery = dict(selected.get("actor_floor_recovery_by_seed") or {})
    frontier = dict(decision.get("full_oracle_frontier") or {})
    distribution = dict(
        frontier.get("candidate_count_by_actor_floor_recovery") or {}
    )
    gate = dict(decision.get("advancement_gate") or {})
    expected_gate_keys = {
        "actor_floor_target_fidelity_gate",
        "all_actor_floor_paths_change_executed_action",
        "all_actor_floor_paths_recovered",
        "all_actor_floor_seed_groups_recovered",
        "all_actor_floor_targets_change_executed_action",
        "all_paths_valid",
        "all_reference_feasible_paths_preserved",
        "expected_actor_floor_path_count",
        "expected_path_count",
        "expected_reference_feasible_path_count",
        "reference_feasible_trust_region_gate",
    }
    false_gates = {
        key for key, value in gate.items() if value is False
    }
    if (
        decision.get("schema_version")
        != MUJOCO_V17_13_CAUSAL_ACTOR_ADAPTER_SCHEMA_VERSION
        or decision.get("status")
        != "causal_actor_adapter_stops_before_fresh_path_access"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != "grouped_reused_path_causal_actor_target_distillation_not_confirmatory"
        or decision.get("development_protocol_version")
        != "mujoco_v17_13_causal_actor_adapter_v1"
        or decision.get("frozen_core_revision")
        != "f7b2254d960824ff02b6b6fc69dd6f1202ee2093"
        or decision.get("frozen_source_manifest_sha256")
        != "a4bdf5eb8c436114ee28556bbc978910b4c1cfcd10bf9a29caf747c48cae27d2"
        or int(decision.get("grouped_seed_fold_count", -1)) != 8
        or int(decision.get("path_count", -1)) != 120
        or int(decision.get("candidate_count", -1)) != 900
        or int(decision.get("full_oracle_candidate_count", -1)) != 48
        or list(decision.get("full_oracle_gain_values") or []) != [0.5, 1.0]
        or scheduler.get("selection_task_id") != "t85846"
        or scheduler.get("selection_task_status") != "done"
        or list(scheduler.get("nodes") or []) != ["node003"]
        or int(scheduler.get("cpu_cores", -1)) != 32
        or int(scheduler.get("oracle_workers", -1)) != 32
        or scheduler.get("slurm_used") is not False
        or int(target.get("actor_floor_path_count", -1)) != 7
        or int(target.get("post_clipping_nonzero_target_path_count", -1))
        != 7
        or selected.get("candidate_id")
        != "actor_fir_w8_ridge1e-04_floorw256_gain1.00_cap0.010"
        or int(selected.get("valid_path_count", -1)) != 120
        or int(selected.get("reference_feasible_path_count", -1)) != 113
        or int(selected.get(
            "reference_feasible_preserved_path_count", -1
        )) != 113
        or int(selected.get("actor_floor_path_count", -1)) != 7
        or int(selected.get("actor_floor_recovered_path_count", -1)) != 3
        or int(selected.get(
            "actor_floor_executed_nonzero_path_count", -1
        )) != 7
        or float(selected.get("actor_floor_target_normalized_mse", -1.0))
        != 0.6099372445889338
        or recovery
        != {
            "2802248628": {"recovered": 1, "total": 2},
            "294864529": {"recovered": 2, "total": 5},
        }
        or int(frontier.get(
            "maximum_actor_floor_recovered_path_count", -1
        )) != 3
        or int(frontier.get(
            "maximum_corrected_joint_feasible_path_count", -1
        )) != 116
        or frontier.get("all_candidates_preserved_reference_feasible_paths")
        is not True
        or distribution != {"2": 32, "3": 16}
        or set(gate) != expected_gate_keys
        or false_gates
        != {
            "all_actor_floor_paths_recovered",
            "all_actor_floor_seed_groups_recovered",
        }
        or decision.get("fresh_validation_paths_accessed") is not False
        or decision.get("fresh_path_access_allowed") is not False
        or decision.get("support_gate") is not False
        or report is None
        or "`causal_actor_adapter_stops_before_fresh_path_access`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError("MuJoCo v17.13 causal actor decision drifted")
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "grouped_seed_fold_count": 8,
        "path_count": 120,
        "candidate_count": 900,
        "full_oracle_candidate_count": 48,
        "full_oracle_gain_values": [0.5, 1.0],
        "selected_actor_floor_recovered_path_count": 3,
        "selected_reference_feasible_preserved_path_count": 113,
        "selected_actor_floor_target_normalized_mse": float(
            selected["actor_floor_target_normalized_mse"]
        ),
        "maximum_actor_floor_recovered_path_count": 3,
        "unexamined_aggressive_gain_values": [1.5, 2.0],
        "fresh_validation_paths_accessed": False,
        "eligible_for_fresh_path_validation": False,
        "support_gate": False,
    }


def _mujoco_v17_14_exhaustive_actor_oracle_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    audit = dict(decision.get("candidate_audit") or {})
    scheduler = dict(decision.get("scheduler") or {})
    selected = dict(decision.get("selected_candidate") or {})
    recovery = dict(selected.get("actor_floor_recovery_by_seed") or {})
    frontier = dict(decision.get("full_grid_frontier") or {})
    distribution = dict(
        frontier.get("candidate_count_by_actor_floor_recovery") or {}
    )
    by_gain = dict(frontier.get("by_output_gain") or {})
    unresolved = dict(decision.get("unresolved_path") or {})
    gate = dict(decision.get("advancement_gate") or {})
    expected_gate_keys = {
        "actor_floor_target_fidelity_gate",
        "all_actor_floor_paths_change_executed_action",
        "all_actor_floor_paths_recovered",
        "all_actor_floor_seed_groups_recovered",
        "all_actor_floor_targets_change_executed_action",
        "all_paths_valid",
        "all_reference_feasible_paths_preserved",
        "expected_actor_floor_path_count",
        "expected_path_count",
        "expected_reference_feasible_path_count",
        "reference_feasible_trust_region_gate",
    }
    false_gates = {key for key, value in gate.items() if value is False}
    if (
        decision.get("schema_version")
        != MUJOCO_V17_14_EXHAUSTIVE_ACTOR_ORACLE_SCHEMA_VERSION
        or decision.get("status")
        != "exhaustive_actor_oracle_closes_frozen_linear_fir_grid"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != "exhaustive_reused_path_full_grid_exact_oracle_not_confirmatory"
        or decision.get("development_protocol_version")
        != "mujoco_v17_14_exhaustive_actor_oracle_v1"
        or decision.get("frozen_core_revision")
        != "5c382979eeffaf7fde19be99835ee0ddc9e9b986"
        or decision.get("frozen_source_manifest_sha256")
        != "32dadb19d67f9b5bea6be95043d01a36ef1f4a6d39df3bdb8925c781d7f4b41d"
        or int(decision.get("path_count", -1)) != 120
        or audit
        != {
            "v17_13_exact_candidate_count": 48,
            "new_exact_candidate_count": 852,
            "combined_exact_candidate_count": 900,
            "passing_candidate_count": 0,
            "frozen_linear_fir_grid_closed": True,
        }
        or scheduler.get("task_id") != "t85847"
        or scheduler.get("task_status") != "done"
        or list(scheduler.get("nodes") or []) != ["node003"]
        or int(scheduler.get("cpu_cores", -1)) != 32
        or int(scheduler.get("oracle_workers", -1)) != 32
        or int(scheduler.get("peak_ram_mb", -1)) != 3063
        or scheduler.get("slurm_used") is not False
        or selected.get("candidate_id")
        != "actor_fir_w8_ridge1e-04_floorw256_gain1.50_cap0.010"
        or int(selected.get("valid_path_count", -1)) != 120
        or int(selected.get("corrected_joint_feasible_path_count", -1))
        != 119
        or int(selected.get("reference_feasible_path_count", -1)) != 113
        or int(selected.get(
            "reference_feasible_preserved_path_count", -1
        ))
        != 113
        or int(selected.get("actor_floor_path_count", -1)) != 7
        or int(selected.get("actor_floor_recovered_path_count", -1)) != 6
        or int(selected.get(
            "actor_floor_executed_nonzero_path_count", -1
        ))
        != 7
        or float(selected.get("actor_floor_target_normalized_mse", -1.0))
        != 0.639795865064592
        or recovery
        != {
            "2802248628": {"recovered": 2, "total": 2},
            "294864529": {"recovered": 4, "total": 5},
        }
        or int(frontier.get(
            "all_reference_feasible_preserved_candidate_count", -1
        ))
        != 900
        or int(frontier.get(
            "maximum_actor_floor_recovered_path_count", -1
        ))
        != 6
        or int(frontier.get(
            "maximum_corrected_joint_feasible_path_count", -1
        ))
        != 119
        or distribution
        != {"2": 497, "3": 207, "4": 71, "5": 48, "6": 77}
        or {
            gain: int(values.get("maximum_actor_floor_recovered_path_count", -1))
            for gain, values in by_gain.items()
        }
        != {"0.5": 2, "1.0": 3, "1.5": 6, "2.0": 6}
        or unresolved.get("environment") != "Hopper-v5"
        or unresolved.get("disturbance_mode") != "ood_chirp"
        or int(unresolved.get("evaluation_seed", -1)) != 294864529
        or float(unresolved.get("corrected_lower_power", -1.0))
        != 0.0025170921934271812
        or float(unresolved.get("lower_power_budget", -1.0)) != 0.00225625
        or set(gate) != expected_gate_keys
        or false_gates
        != {
            "all_actor_floor_paths_recovered",
            "all_actor_floor_seed_groups_recovered",
        }
        or decision.get("fresh_validation_paths_accessed") is not False
        or decision.get("fresh_path_access_allowed") is not False
        or decision.get("support_gate") is not False
        or report is None
        or "`exhaustive_actor_oracle_closes_frozen_linear_fir_grid`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError("MuJoCo v17.14 exhaustive actor decision drifted")
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "path_count": 120,
        "combined_exact_candidate_count": 900,
        "passing_candidate_count": 0,
        "selected_actor_floor_recovered_path_count": 6,
        "selected_reference_feasible_preserved_path_count": 113,
        "maximum_actor_floor_recovered_path_count": 6,
        "unresolved_environment": "Hopper-v5",
        "unresolved_disturbance_mode": "ood_chirp",
        "unresolved_evaluation_seed": 294864529,
        "frozen_linear_fir_grid_closed": True,
        "fresh_validation_paths_accessed": False,
        "eligible_for_fresh_path_validation": False,
        "support_gate": False,
    }


def _mujoco_v18_1_state_actor_dataset_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    scheduler = dict(decision.get("scheduler") or {})
    validation = dict(decision.get("validation") or {})
    if (
        decision.get("schema_version")
        != MUJOCO_V18_1_STATE_ACTOR_DATASET_SCHEMA_VERSION
        or decision.get("status")
        != "causal_state_dataset_validated_on_reused_paths"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != "reused_path_causal_actor_state_export_not_model_selection_or_confirmatory"
        or decision.get("development_protocol_version")
        != "mujoco_v18_1_state_actor_dataset_v1"
        or decision.get("frozen_core_revision")
        != "f94f1f4a6a35d70f6b6d144bd886644e7efb2393"
        or decision.get("frozen_source_manifest_sha256")
        != "b06a97fc8f18129a2e1a9c23a52acb01ea683d1445c672074b9a6901ece23af6"
        or scheduler.get("task_id_first") != "t85848"
        or scheduler.get("task_id_last") != "t85967"
        or int(scheduler.get("task_count", -1)) != 120
        or int(scheduler.get("done_task_count", -1)) != 120
        or list(scheduler.get("nodes") or []) != ["node003"]
        or int(scheduler.get("cpu_cores_per_task", -1)) != 1
        or scheduler.get("slurm_used") is not False
        or int(validation.get("path_count", -1)) != 120
        or int(validation.get("reference_feasible_path_count", -1)) != 113
        or int(validation.get("actor_floor_path_count", -1)) != 7
        or dict(validation.get("actor_floor_path_count_by_seed") or {})
        != {"2802248628": 2, "294864529": 5}
        or dict(validation.get("state_path_count_by_environment") or {})
        != {
            "HalfCheetah-v5": 40,
            "Hopper-v5": 40,
            "Walker2d-v5": 40,
        }
        or dict(validation.get("trajectory_step_count_by_environment") or {})
        != {
            "HalfCheetah-v5": 40000,
            "Hopper-v5": 3322,
            "Walker2d-v5": 6549,
        }
        or validation.get("pretransition_causal_alignment_valid") is not True
        or validation.get("target_labels_read_during_export") is not False
        or validation.get("target_labels_used_only_as_training_outputs")
        is not True
        or int(validation.get("server_only_npz_path_count", -1)) != 120
        or decision.get("fresh_validation_paths_accessed") is not False
        or decision.get("support_gate") is not False
        or report is None
        or "`causal_state_dataset_validated_on_reused_paths`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError("MuJoCo v18.1 causal state dataset decision drifted")
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "path_count": 120,
        "done_task_count": 120,
        "state_path_count_by_environment": dict(
            validation["state_path_count_by_environment"]
        ),
        "trajectory_step_count_by_environment": dict(
            validation["trajectory_step_count_by_environment"]
        ),
        "pretransition_causal_alignment_valid": True,
        "target_labels_read_during_export": False,
        "server_only_npz_path_count": 120,
        "fresh_validation_paths_accessed": False,
        "support_gate": False,
    }


def _mujoco_v18_2_state_conditioned_actor_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    scheduler = dict(decision.get("scheduler") or {})
    selected = dict(decision.get("selected_candidate") or {})
    recovery = dict(selected.get("actor_floor_recovery_by_seed") or {})
    frontier = dict(decision.get("candidate_frontier") or {})
    comparison = dict(decision.get("comparison_to_v17_14") or {})
    gate = dict(decision.get("advancement_gate") or {})
    expected_gate_keys = {
        "actor_floor_target_fidelity_gate",
        "all_actor_floor_paths_change_executed_action",
        "all_actor_floor_paths_recovered",
        "all_actor_floor_seed_groups_recovered",
        "all_actor_floor_targets_change_executed_action",
        "all_paths_valid",
        "all_reference_feasible_paths_preserved",
        "expected_actor_floor_path_count",
        "expected_path_count",
        "expected_reference_feasible_path_count",
        "reference_feasible_trust_region_gate",
    }
    false_gates = {key for key, value in gate.items() if value is False}
    if (
        decision.get("schema_version")
        != MUJOCO_V18_2_STATE_CONDITIONED_ACTOR_SCHEMA_VERSION
        or decision.get("status")
        != "state_conditioned_actor_stops_before_fresh_path_access"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != "grouped_reused_path_state_conditioned_actor_selection_not_confirmatory"
        or decision.get("development_protocol_version")
        != "mujoco_v18_2_state_conditioned_actor_v1"
        or decision.get("frozen_core_revision")
        != "6ebf63c77c5c8ecf2e0784b7361eb90a6d71caf9"
        or decision.get("frozen_source_manifest_sha256")
        != "d5842e0972d59182be55881b52de9b4ac074b5b355aa1b4c929e4b7f0a849099"
        or int(decision.get("path_count", -1)) != 120
        or int(decision.get("grouped_seed_fold_count", -1)) != 8
        or int(decision.get("candidate_count", -1)) != 16
        or int(decision.get("full_oracle_candidate_count", -1)) != 16
        or scheduler.get("task_id") != "t85969"
        or scheduler.get("task_status") != "done"
        or list(scheduler.get("nodes") or []) != ["node003"]
        or int(scheduler.get("cpu_cores", -1)) != 16
        or int(scheduler.get("workers", -1)) != 16
        or scheduler.get("peak_ram_sample_available") is not False
        or scheduler.get("slurm_used") is not False
        or selected.get("candidate_id")
        != "state_mlp_w1_h32_floorw64_cap0.025"
        or int(selected.get("valid_path_count", -1)) != 120
        or int(selected.get("corrected_joint_feasible_path_count", -1))
        != 116
        or int(selected.get("reference_feasible_path_count", -1)) != 113
        or int(selected.get(
            "reference_feasible_preserved_path_count", -1
        ))
        != 113
        or int(selected.get("actor_floor_path_count", -1)) != 7
        or int(selected.get("actor_floor_recovered_path_count", -1)) != 3
        or int(selected.get(
            "actor_floor_executed_nonzero_path_count", -1
        ))
        != 7
        or float(selected.get("actor_floor_target_normalized_mse", -1.0))
        != 0.9523924970694609
        or recovery
        != {
            "2802248628": {"recovered": 0, "total": 2},
            "294864529": {"recovered": 3, "total": 5},
        }
        or int(frontier.get(
            "all_reference_feasible_preserved_candidate_count", -1
        ))
        != 16
        or int(frontier.get(
            "maximum_actor_floor_recovered_path_count", -1
        ))
        != 3
        or int(frontier.get(
            "maximum_corrected_joint_feasible_path_count", -1
        ))
        != 116
        or dict(frontier.get("candidate_count_by_actor_floor_recovery") or {})
        != {"2": 14, "3": 2}
        or int(comparison.get(
            "v17_14_actor_floor_recovered_path_count", -1
        ))
        != 6
        or int(comparison.get("recovery_change", 0)) != -3
        or comparison.get("state_conditioning_improved_frozen_reused_panel")
        is not False
        or set(gate) != expected_gate_keys
        or false_gates
        != {
            "actor_floor_target_fidelity_gate",
            "all_actor_floor_paths_recovered",
            "all_actor_floor_seed_groups_recovered",
        }
        or decision.get("fresh_validation_paths_accessed") is not False
        or decision.get("fresh_path_access_allowed") is not False
        or decision.get("support_gate") is not False
        or report is None
        or "`state_conditioned_actor_stops_before_fresh_path_access`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError("MuJoCo v18.2 state actor decision drifted")
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "path_count": 120,
        "grouped_seed_fold_count": 8,
        "candidate_count": 16,
        "full_oracle_candidate_count": 16,
        "selected_actor_floor_recovered_path_count": 3,
        "selected_reference_feasible_preserved_path_count": 113,
        "selected_actor_floor_target_normalized_mse": float(
            selected["actor_floor_target_normalized_mse"]
        ),
        "selected_actor_floor_recovery_by_seed": recovery,
        "v17_14_actor_floor_recovered_path_count": 6,
        "state_conditioning_improved_frozen_reused_panel": False,
        "fresh_validation_paths_accessed": False,
        "eligible_for_fresh_path_validation": False,
        "support_gate": False,
    }


def _mujoco_v18_3_causal_joint_projection_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    scheduler = dict(decision.get("scheduler") or {})
    selected = dict(decision.get("selected_candidate") or {})
    recovery = dict(selected.get("actor_floor_recovery_by_seed") or {})
    prefix = dict(decision.get("prefix_ledger_candidate") or {})
    gate = dict(decision.get("advancement_gate") or {})
    expected_gate_keys = {
        "actor_floor_trust_region_gate",
        "all_actor_floor_paths_change_executed_action",
        "all_actor_floor_paths_recovered",
        "all_actor_floor_seed_groups_recovered",
        "all_paths_directly_joint_feasible",
        "all_paths_exact_oracle_feasible",
        "all_paths_valid",
        "all_reference_feasible_paths_preserved",
        "global_correction_abs_gate",
        "reference_feasible_trust_region_gate",
    }
    false_gates = {key for key, value in gate.items() if value is False}
    if (
        decision.get("schema_version")
        != MUJOCO_V18_3_CAUSAL_JOINT_PROJECTION_SCHEMA_VERSION
        or decision.get("status")
        != "causal_joint_projection_stops_before_fresh_path_access"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != "reused_path_label_free_causal_joint_projection_not_confirmatory"
        or decision.get("development_protocol_version")
        != "mujoco_v18_3_causal_joint_projection_v1"
        or decision.get("frozen_core_revision")
        != "212c571e630512d37682f9984727b7f4016740e8"
        or decision.get("frozen_source_manifest_sha256")
        != "11a7a0087c9369e744b6516cd7f6571a9f2b0562eae649f7d5d89cf62052a3a2"
        or int(decision.get("path_count", -1)) != 120
        or int(decision.get("candidate_count", -1)) != 2
        or decision.get("target_labels_accessed") is not False
        or scheduler.get("task_id") != "t85971"
        or scheduler.get("task_status") != "done"
        or list(scheduler.get("nodes") or []) != ["node003"]
        or int(scheduler.get("cpu_cores", -1)) != 16
        or int(scheduler.get("workers", -1)) != 16
        or int(scheduler.get("peak_ram_mb", -1)) != 2183
        or scheduler.get("slurm_used") is not False
        or selected.get("candidate_id")
        != "joint_projection_instantaneous"
        or int(selected.get("valid_path_count", -1)) != 120
        or int(selected.get("direct_joint_feasible_path_count", -1)) != 120
        or int(selected.get("exact_oracle_joint_feasible_path_count", -1))
        != 120
        or int(selected.get("reference_feasible_preserved_path_count", -1))
        != 113
        or int(selected.get("actor_floor_recovered_path_count", -1)) != 7
        or int(selected.get(
            "actor_floor_executed_nonzero_path_count", -1
        ))
        != 7
        or float(selected.get("correction_abs_maximum", -1.0))
        != 1.8075526888835756
        or float(selected.get(
            "reference_feasible_correction_rms_maximum", -1.0
        ))
        != 0.2934515838846031
        or float(selected.get(
            "actor_floor_correction_rms_maximum", -1.0
        ))
        != 0.30193708336206443
        or recovery
        != {
            "2802248628": {"recovered": 2, "total": 2},
            "294864529": {"recovered": 5, "total": 5},
        }
        or int(prefix.get("valid_path_count", -1)) != 42
        or int(prefix.get("actor_floor_recovered_path_count", -1)) != 7
        or int(prefix.get("projection_nonconverged_step_count", -1)) != 451
        or set(gate) != expected_gate_keys
        or false_gates
        != {
            "actor_floor_trust_region_gate",
            "global_correction_abs_gate",
            "reference_feasible_trust_region_gate",
        }
        or decision.get("fresh_validation_paths_accessed") is not False
        or decision.get("fresh_path_access_allowed") is not False
        or decision.get("support_gate") is not False
        or report is None
        or "`causal_joint_projection_stops_before_fresh_path_access`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError("MuJoCo v18.3 joint projection decision drifted")
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "path_count": 120,
        "candidate_count": 2,
        "target_labels_accessed": False,
        "selected_direct_joint_feasible_path_count": 120,
        "selected_exact_oracle_joint_feasible_path_count": 120,
        "selected_actor_floor_recovered_path_count": 7,
        "selected_reference_feasible_preserved_path_count": 113,
        "selected_correction_abs_maximum": float(
            selected["correction_abs_maximum"]
        ),
        "selected_reference_correction_rms_maximum": float(
            selected["reference_feasible_correction_rms_maximum"]
        ),
        "selected_actor_floor_correction_rms_maximum": float(
            selected["actor_floor_correction_rms_maximum"]
        ),
        "prefix_ledger_valid_path_count": 42,
        "fresh_validation_paths_accessed": False,
        "eligible_for_fresh_path_validation": False,
        "support_gate": False,
    }


def _mujoco_v18_4_receding_joint_projection_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    scheduler = dict(decision.get("scheduler") or {})
    selected = dict(decision.get("selected_candidate") or {})
    recovery = dict(selected.get("actor_floor_recovery_by_seed") or {})
    candidates = dict(decision.get("candidate_direct_counts") or {})
    gate = dict(decision.get("advancement_gate") or {})
    expected_gate_keys = {
        "actor_floor_trust_region_gate",
        "all_actor_floor_paths_change_executed_action",
        "all_actor_floor_paths_recovered",
        "all_actor_floor_seed_groups_recovered",
        "all_paths_directly_joint_feasible",
        "all_paths_exact_oracle_feasible",
        "all_paths_valid",
        "all_reference_feasible_paths_preserved",
        "global_correction_abs_gate",
        "reference_feasible_trust_region_gate",
        "selected_candidate_exactly_audited_on_all_paths",
    }
    false_gates = {key for key, value in gate.items() if value is False}
    if (
        decision.get("schema_version")
        != MUJOCO_V18_4_RECEDING_JOINT_PROJECTION_SCHEMA_VERSION
        or decision.get("status")
        != "receding_joint_projection_stops_before_fresh_path_access"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != "reused_path_label_free_causal_receding_projection_not_confirmatory"
        or decision.get("development_protocol_version")
        != "mujoco_v18_4_receding_joint_projection_v1"
        or decision.get("frozen_core_revision")
        != "7f649e23a05f3bf142255fc777bb4322931cc617"
        or decision.get("frozen_source_manifest_sha256")
        != "c7f7f0980508d58ea167d9bd420fc88ecefafb207d8735f791169ea822015dff"
        or int(decision.get("path_count", -1)) != 120
        or int(decision.get("candidate_count", -1)) != 4
        or int(decision.get("direct_audit_candidate_count", -1)) != 4
        or int(decision.get("exact_oracle_audit_candidate_count", -1)) != 1
        or decision.get("actor_correction_targets_accessed") is not False
        or decision.get(
            "reference_feasibility_labels_used_for_evaluation"
        ) is not True
        or scheduler.get("task_id") != "t85972"
        or scheduler.get("task_status") != "done"
        or list(scheduler.get("nodes") or []) != ["node003"]
        or int(scheduler.get("cpu_cores", -1)) != 16
        or int(scheduler.get("workers", -1)) != 16
        or int(scheduler.get("peak_ram_mb", -1)) != 1925
        or scheduler.get("slurm_used") is not False
        or selected.get("candidate_id")
        != "joint_mpc_h16_damped_velocity"
        or int(selected.get("valid_path_count", -1)) != 120
        or int(selected.get("direct_joint_feasible_path_count", -1)) != 69
        or int(selected.get("exact_oracle_audited_path_count", -1)) != 120
        or int(selected.get("exact_oracle_joint_feasible_path_count", -1))
        != 120
        or int(selected.get(
            "direct_reference_feasible_preserved_path_count", -1
        ))
        != 67
        or int(selected.get(
            "direct_actor_floor_recovered_path_count", -1
        ))
        != 2
        or int(selected.get(
            "actor_floor_executed_nonzero_path_count", -1
        ))
        != 7
        or int(selected.get("prefix_budget_violation_step_count", -1))
        != 40962
        or float(selected.get("correction_abs_maximum", -1.0))
        != 0.988801906635327
        or float(selected.get(
            "reference_feasible_correction_rms_maximum", -1.0
        ))
        != 0.2507122109046519
        or float(selected.get(
            "actor_floor_correction_rms_maximum", -1.0
        ))
        != 0.25251917036308946
        or recovery
        != {
            "2802248628": {
                "direct_recovered": 1,
                "exact_feasible": 2,
                "total": 2,
            },
            "294864529": {
                "direct_recovered": 1,
                "exact_feasible": 5,
                "total": 5,
            },
        }
        or candidates
        != {
            "joint_mpc_h16_hold": {
                "direct_joint_feasible": 40,
                "actor_floor_recovered": 0,
                "reference_preserved": 40,
            },
            "joint_mpc_h16_damped_velocity": {
                "direct_joint_feasible": 69,
                "actor_floor_recovered": 2,
                "reference_preserved": 67,
            },
            "joint_mpc_h32_hold": {
                "direct_joint_feasible": 40,
                "actor_floor_recovered": 0,
                "reference_preserved": 40,
            },
            "joint_mpc_h32_damped_velocity": {
                "direct_joint_feasible": 40,
                "actor_floor_recovered": 0,
                "reference_preserved": 40,
            },
        }
        or set(gate) != expected_gate_keys
        or false_gates
        != {
            "actor_floor_trust_region_gate",
            "all_actor_floor_paths_recovered",
            "all_actor_floor_seed_groups_recovered",
            "all_paths_directly_joint_feasible",
            "all_reference_feasible_paths_preserved",
            "global_correction_abs_gate",
            "reference_feasible_trust_region_gate",
        }
        or decision.get("fresh_validation_paths_accessed") is not False
        or decision.get("fresh_path_access_allowed") is not False
        or decision.get("support_gate") is not False
        or report is None
        or "`receding_joint_projection_stops_before_fresh_path_access`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError("MuJoCo v18.4 receding projection decision drifted")
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "path_count": 120,
        "candidate_count": 4,
        "direct_audit_candidate_count": 4,
        "exact_oracle_audit_candidate_count": 1,
        "actor_correction_targets_accessed": False,
        "selected_direct_joint_feasible_path_count": 69,
        "selected_exact_oracle_joint_feasible_path_count": 120,
        "selected_actor_floor_recovered_path_count": 2,
        "selected_reference_feasible_preserved_path_count": 67,
        "selected_correction_abs_maximum": float(
            selected["correction_abs_maximum"]
        ),
        "selected_reference_correction_rms_maximum": float(
            selected["reference_feasible_correction_rms_maximum"]
        ),
        "selected_actor_floor_correction_rms_maximum": float(
            selected["actor_floor_correction_rms_maximum"]
        ),
        "selected_prefix_budget_violation_step_count": 40962,
        "offline_exact_online_direct_gap_path_count": 51,
        "fresh_validation_paths_accessed": False,
        "eligible_for_fresh_path_validation": False,
        "support_gate": False,
    }


def _mujoco_v18_5_actor_floor_signal_facts(
    paths: dict[str, Path],
) -> dict[str, Any]:
    decision = _read_json(paths["decision"])
    report = paths.get("report")
    scheduler = dict(decision.get("scheduler") or {})
    selected = dict(decision.get("selected_signal") or {})
    if (
        decision.get("schema_version")
        != MUJOCO_V18_5_ACTOR_FLOOR_SIGNAL_SCHEMA_VERSION
        or decision.get("status")
        != "actor_floor_signal_stops_debt_feedback_direction"
        or decision.get("integrity_status") != "valid"
        or decision.get("evidence_role")
        != "reused_path_causal_actor_floor_signal_diagnostic_only"
        or decision.get("development_protocol_version")
        != "mujoco_v18_5_actor_floor_signal_v1"
        or decision.get("frozen_core_revision")
        != "e97028fd121693c9c5902f2af61c5833006d887f"
        or decision.get("frozen_source_manifest_sha256")
        != "c37f934d1a5fb528b620e27678434148d65b8a727fde04176af5ce58a24b0d08"
        or int(decision.get("path_count", -1)) != 120
        or int(decision.get("candidate_count", -1)) != 2
        or int(decision.get("score_count", -1)) != 6
        or int(decision.get("assessment_count", -1)) != 12
        or decision.get("actor_correction_targets_accessed") is not False
        or scheduler.get("task_id") != "t86055"
        or scheduler.get("task_status") != "done"
        or list(scheduler.get("nodes") or []) != ["node003"]
        or int(scheduler.get("cpu_cores", -1)) != 16
        or int(scheduler.get("workers", -1)) != 16
        or int(scheduler.get("peak_ram_mb", -1)) != 811
        or scheduler.get("slurm_used") is not False
        or selected.get("candidate_id") != "actor_floor_h16_hold"
        or selected.get("score_field") != "floor_power_excess_mean"
        or float(selected.get("global_rank_auc", -1.0))
        != 0.95448798988622
        or float(selected.get(
            "actor_floor_environment_rank_auc", -1.0
        ))
        != 0.8441558441558441
        or int(selected.get("top_7_actor_floor_count", -1)) != 4
        or int(selected.get("top_14_actor_floor_count", -1)) != 5
        or int(selected.get("top_28_actor_floor_count", -1)) != 7
        or int(selected.get("unresolved_v17_14_path_rank", -1)) != 4
        or selected.get("feedback_screen_eligible") is not False
        or int(decision.get("eligible_signal_count", -1)) != 0
        or decision.get("feedback_screen_allowed") is not False
        or decision.get("fresh_validation_paths_accessed") is not False
        or decision.get("fresh_path_access_allowed") is not False
        or decision.get("support_gate") is not False
        or report is None
        or "`actor_floor_signal_stops_debt_feedback_direction`"
        not in report.read_text(encoding="utf-8")
    ):
        raise ValueError("MuJoCo v18.5 actor-floor signal decision drifted")
    return {
        "decision_status": str(decision["status"]),
        "integrity_status": "valid",
        "path_count": 120,
        "candidate_count": 2,
        "score_count": 6,
        "actor_correction_targets_accessed": False,
        "selected_global_rank_auc": float(selected["global_rank_auc"]),
        "selected_actor_floor_environment_rank_auc": float(
            selected["actor_floor_environment_rank_auc"]
        ),
        "selected_top_14_actor_floor_count": 5,
        "selected_unresolved_v17_14_path_rank": 4,
        "eligible_signal_count": 0,
        "feedback_screen_allowed": False,
        "fresh_validation_paths_accessed": False,
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
    "mujoco_v17_zero_dc_plan": _mujoco_v17_zero_dc_plan_facts,
    "mujoco_v17_1_headroom_homotopy": (
        _mujoco_v17_1_headroom_homotopy_facts
    ),
    "mujoco_v17_2_smooth_macro_gauge": (
        _mujoco_v17_2_smooth_macro_gauge_facts
    ),
    "mujoco_v17_3_audit_optimal_macro_gauge": (
        _mujoco_v17_3_audit_optimal_macro_gauge_facts
    ),
    "mujoco_v17_4_streaming_audit_projection": (
        _mujoco_v17_4_streaming_audit_projection_facts
    ),
    "mujoco_v17_5_feasibility_diagnostic": (
        _mujoco_v17_5_feasibility_diagnostic_facts
    ),
    "mujoco_v17_6_full_horizon_oracle": (
        _mujoco_v17_6_full_horizon_oracle_facts
    ),
    "mujoco_v17_8_causal_fir_distillation": (
        _mujoco_v17_8_causal_fir_distillation_facts
    ),
    "mujoco_v17_9_prefix_hpf_fir": (
        _mujoco_v17_9_prefix_hpf_fir_facts
    ),
    "mujoco_v17_10_horizon_reservoir_fir": (
        _mujoco_v17_10_horizon_reservoir_fir_facts
    ),
    "mujoco_v17_11_fractional_reservoir_fir": (
        _mujoco_v17_11_fractional_reservoir_fir_facts
    ),
    "mujoco_v17_12_nearest_feasible_action_oracle": (
        _mujoco_v17_12_nearest_feasible_action_oracle_facts
    ),
    "mujoco_v17_13_causal_actor_adapter": (
        _mujoco_v17_13_causal_actor_adapter_facts
    ),
    "mujoco_v17_14_exhaustive_actor_oracle": (
        _mujoco_v17_14_exhaustive_actor_oracle_facts
    ),
    "mujoco_v18_1_state_actor_dataset": (
        _mujoco_v18_1_state_actor_dataset_facts
    ),
    "mujoco_v18_2_state_conditioned_actor": (
        _mujoco_v18_2_state_conditioned_actor_facts
    ),
    "mujoco_v18_3_causal_joint_projection": (
        _mujoco_v18_3_causal_joint_projection_facts
    ),
    "mujoco_v18_4_receding_joint_projection": (
        _mujoco_v18_4_receding_joint_projection_facts
    ),
    "mujoco_v18_5_actor_floor_signal": (
        _mujoco_v18_5_actor_floor_signal_facts
    ),
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
