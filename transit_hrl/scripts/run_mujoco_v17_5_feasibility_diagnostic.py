#!/usr/bin/env python3
"""Replay v17.4 checkpoints to separate projection regret from policy floors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.mujoco.control_validation import (  # noqa: E402
    RESPONSIBILITY_TRANSFER_ALPHA,
    _model_parameter_sha256,
    load_paired_mujoco_checkpoint,
    rollout_hierarchical,
)
from freq_hrl.rl.smdp_actor_critic import (  # noqa: E402
    FrequencySeparatedActorCriticPPO,
    SMDPPPOConfig,
)
from scripts import (  # noqa: E402
    mujoco_v17_4_streaming_audit_projection_preflight_spec as v17_4,
)


DIAGNOSTIC_PROTOCOL_VERSION = (
    "mujoco_v17_5_feasibility_normalized_projection_diagnostic_v1"
)
EVIDENCE_ROLE = (
    "rejected_v17_4_path_reuse_development_diagnostic_not_confirmatory"
)
ROUTER_MODES = (
    "causal_streaming_audit_projection",
    "causal_feasibility_normalized_audit_projection",
)
TRACE_KEYS = (
    "RewardTraceSHA256",
    "ExecutedActionTraceSHA256",
    "LatentPolicyTraceSHA256",
)
LEGACY_NUMERIC_KEYS = (
    "episode_return",
    "UpperHFPowerAbs",
    "LowerLFDriftAbs",
    "LatentUpperHFPowerAbs",
    "LatentLowerLFDriftAbs",
    "LowerRouterActionReconstructionRMS",
    "ResponsibilityReconstructionRMS",
)
AGGREGATE_RMS_KEYS = (
    "LowerRouterUpperBudgetViolationRMS",
    "LowerRouterLowerBudgetViolationRMS",
    "LowerRouterUnavoidableUpperBudgetViolationRMS",
    "LowerRouterUnavoidableLowerBudgetViolationRMS",
    "LowerRouterBudgetExcessRegretRMS",
)
AGGREGATE_MEAN_KEYS = (
    "episode_return",
    "UpperHFPowerAbs",
    "LowerLFDriftAbs",
    "LowerRouterUpperBudgetFeasibleRate",
    "LowerRouterLowerBudgetSatisfiedRate",
    "LowerRouterJointBudgetFeasibleRate",
)
REGRET_TOLERANCE = 1e-7


def _path_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["disturbance_mode"]), int(row["seed"])


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _model_from_summary(summary: dict[str, Any]) -> FrequencySeparatedActorCriticPPO:
    config = dict(summary.get("config") or {})
    if not config:
        raise ValueError("v17.5 diagnostic checkpoint summary has no model config")
    config["actor_anchor_zero_state_indices"] = tuple(
        config.get("actor_anchor_zero_state_indices") or ()
    )
    return FrequencySeparatedActorCriticPPO(SMDPPPOConfig(**config))


def _legacy_candidate_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    selected = {
        _path_key(row): row
        for row in rows
        if row.get("paired_intervention") == v17_4.CANDIDATE_INTERVENTION
    }
    if len(selected) != v17_4.EXPECTED_PATHS_PER_INTERVENTION:
        raise ValueError("v17.4 checkpoint has an incomplete legacy candidate matrix")
    expected = {
        (mode, int(seed))
        for mode in v17_4.EVALUATION_DISTURBANCE_MODES
        for seed in v17_4.EVALUATION_SEEDS
    }
    if set(selected) != expected:
        raise ValueError("v17.4 checkpoint legacy paths violate the frozen design")
    return selected


def _rollout(
    model: FrequencySeparatedActorCriticPPO,
    *,
    env_id: str,
    disturbance_mode: str,
    seed: int,
    router_mode: str,
) -> dict[str, Any]:
    _, row = rollout_hierarchical(
        model,
        seed=int(seed),
        env_id=str(env_id),
        disturbance_mode=str(disturbance_mode),
        steps=v17_4.STEPS,
        upper_period=v17_4.UPPER_PERIOD,
        frequency_routing=True,
        leakage_constraint=True,
        sample=False,
        upper_action_scale=v17_4.UPPER_ACTION_SCALE,
        lower_action_scale=v17_4.LOWER_ACTION_SCALE,
        upper_action_decoder_mode="causal_smoothstep_plan",
        lower_lf_alpha=RESPONSIBILITY_TRANSFER_ALPHA,
        lower_lf_rms_budget=v17_4.LOWER_LF_RMS_BUDGET,
        leakage_constraint_scope="joint_behavior_latent",
        upper_hf_rms_budget=v17_4.UPPER_HF_RMS_BUDGET,
        upper_hf_penalty_coef=0.0,
        upper_constraint_mode="primal_dual",
        responsibility_mode="additive",
        leakage_cost_mode="power_excess",
        lower_action_router_mode=str(router_mode),
        lower_action_router_alpha=v17_4.ROUTER_ALPHA,
        lower_action_router_strength=1.0,
        lower_action_router_observe_strength=False,
        upper_promotion_gain=0.0,
        method="freq_hrl",
        episode_horizon=v17_4.EPISODE_HORIZON,
    )
    row.update({
        "diagnostic_protocol_version": DIAGNOSTIC_PROTOCOL_VERSION,
        "diagnostic_router_mode": str(router_mode),
        "source_checkpoint_protocol_version": v17_4.FROZEN_CORE_PROTOCOL_VERSION,
    })
    return row


def _legacy_replay_audit(
    legacy: dict[tuple[str, int], dict[str, Any]],
    replay: list[dict[str, Any]],
) -> dict[str, Any]:
    replay_by_path = {_path_key(row): row for row in replay}
    if set(replay_by_path) != set(legacy):
        raise ValueError("v17.5 diagnostic replay paths do not match v17.4")
    trace_mismatches = {
        key: sum(
            str(legacy[path][key]) != str(replay_by_path[path][key])
            for path in legacy
        )
        for key in TRACE_KEYS
    }
    numeric_max_abs_difference = {
        key: max(
            abs(float(legacy[path][key]) - float(replay_by_path[path][key]))
            for path in legacy
        )
        for key in LEGACY_NUMERIC_KEYS
    }
    exact = not any(trace_mismatches.values()) and max(
        numeric_max_abs_difference.values(), default=0.0
    ) <= 1e-12
    return {
        "path_count": len(legacy),
        "trace_mismatches": trace_mismatches,
        "numeric_max_abs_difference": numeric_max_abs_difference,
        "exact_legacy_replay": bool(exact),
    }


def _cross_router_audit(
    rows_by_mode: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    baseline = {_path_key(row): row for row in rows_by_mode[ROUTER_MODES[0]]}
    candidate = {_path_key(row): row for row in rows_by_mode[ROUTER_MODES[1]]}
    return {
        "path_count": len(baseline),
        "trace_mismatches": {
            key: sum(
                str(baseline[path][key]) != str(candidate[path][key])
                for path in baseline
            )
            for key in TRACE_KEYS
        },
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in AGGREGATE_MEAN_KEYS
    }
    result.update({
        key: float(np.sqrt(np.mean(np.square([
            float(row[key]) for row in rows
        ]))))
        for key in AGGREGATE_RMS_KEYS
    })
    result.update({
        f"{key}Max": float(max(float(row[key]) for row in rows))
        for key in AGGREGATE_RMS_KEYS
    })
    return result


def diagnose_projection_limit(
    legacy_audit: dict[str, Any],
    aggregates: dict[str, dict[str, Any]],
) -> str:
    """Classify the next algorithmic bottleneck without making a claim."""

    if not legacy_audit.get("exact_legacy_replay"):
        return "invalid_due_to_v17_4_replay_regression"
    old = aggregates[ROUTER_MODES[0]]
    new = aggregates[ROUTER_MODES[1]]
    old_regret = float(old["LowerRouterBudgetExcessRegretRMSMax"])
    new_regret = float(new["LowerRouterBudgetExcessRegretRMSMax"])
    floor = max(
        float(new["LowerRouterUnavoidableUpperBudgetViolationRMSMax"]),
        float(new["LowerRouterUnavoidableLowerBudgetViolationRMSMax"]),
    )
    if (
        old_regret > REGRET_TOLERANCE
        and new_regret <= max(REGRET_TOLERANCE, 0.25 * old_regret)
    ):
        return "projection_limited_avoidable_budget_regret"
    if floor > REGRET_TOLERANCE:
        return "learned_policy_limited_unavoidable_physical_budget_floor"
    if new_regret > REGRET_TOLERANCE:
        return "projection_limited_residual_budget_regret"
    return "budgets_feasible_on_replayed_paths"


def run_diagnostic(
    *,
    env_id: str,
    optimizer_seed: int,
    checkpoint_dir: Path,
    diagnostic_code_revision: str,
    diagnostic_source_manifest_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = Path(checkpoint_dir) / "cell_summary.json"
    checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
    legacy_rows_path = Path(checkpoint_dir) / "evaluation_rows.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model = _model_from_summary(summary)
    checkpoint_metadata = load_paired_mujoco_checkpoint(
        model,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
        env_id=str(env_id),
        optimizer_seed=int(optimizer_seed),
        expected_code_revision=v17_4.FROZEN_ALGORITHM_REVISION,
        expected_source_manifest_sha256=v17_4.FROZEN_SOURCE_MANIFEST_SHA256,
        expected_method="freq_hrl",
        expected_router_mode="causal_streaming_audit_projection",
        expected_router_strength=0.0,
        expected_router_observe_strength=False,
        expected_responsibility_mode="additive",
        expected_protocol_version=v17_4.FROZEN_CORE_PROTOCOL_VERSION,
    )
    parameter_sha256 = _model_parameter_sha256(model)
    legacy = _legacy_candidate_rows(_read_csv(legacy_rows_path))
    rows_by_mode: dict[str, list[dict[str, Any]]] = {}
    for router_mode in ROUTER_MODES:
        rows_by_mode[router_mode] = [
            _rollout(
                model,
                env_id=str(env_id),
                disturbance_mode=mode,
                seed=seed,
                router_mode=router_mode,
            )
            for mode, seed in sorted(legacy)
        ]
    if parameter_sha256 != _model_parameter_sha256(model):
        raise RuntimeError("v17.5 diagnostic mutated the frozen checkpoint")
    legacy_audit = _legacy_replay_audit(
        legacy, rows_by_mode[ROUTER_MODES[0]]
    )
    aggregates = {
        mode: aggregate_rows(rows) for mode, rows in rows_by_mode.items()
    }
    payload = {
        "status": diagnose_projection_limit(legacy_audit, aggregates),
        "evidence_role": EVIDENCE_ROLE,
        "diagnostic_protocol_version": DIAGNOSTIC_PROTOCOL_VERSION,
        "diagnostic_code_revision": str(diagnostic_code_revision),
        "diagnostic_source_manifest_sha256": str(
            diagnostic_source_manifest_sha256
        ),
        "environment": str(env_id),
        "optimizer_seed": int(optimizer_seed),
        "source_checkpoint": checkpoint_metadata,
        "source_checkpoint_parameter_sha256": parameter_sha256,
        "legacy_replay_audit": legacy_audit,
        "cross_router_audit": _cross_router_audit(rows_by_mode),
        "aggregates": aggregates,
        "claim_boundary": (
            "development diagnosis on already accessed rejected v17.4 paths; "
            "not fresh evidence and not eligible for a performance claim"
        ),
        "artifact_policy": "small_json_and_csv_synced_checkpoint_server_only_v1",
    }
    rows = [
        row
        for router_mode in ROUTER_MODES
        for row in rows_by_mode[router_mode]
    ]
    return payload, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", choices=v17_4.ENVIRONMENTS, required=True)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-code-revision", required=True)
    parser.add_argument("--diagnostic-source-manifest-sha256", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if int(args.optimizer_seed) not in v17_4.OPTIMIZER_SEEDS:
        raise SystemExit("optimizer seed is outside the frozen v17.4 design")
    payload, rows = run_diagnostic(
        env_id=args.env_id,
        optimizer_seed=args.optimizer_seed,
        checkpoint_dir=args.checkpoint_dir,
        diagnostic_code_revision=args.diagnostic_code_revision,
        diagnostic_source_manifest_sha256=(
            args.diagnostic_source_manifest_sha256
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "diagnostic_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "diagnostic_rows.csv", rows)
    print(
        f"DONE v17.5 feasibility diagnostic env={args.env_id} "
        f"status={payload['status']} output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
