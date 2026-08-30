#!/usr/bin/env python3
"""Train one v17.3 policy and evaluate the paired gauge intervention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.mujoco.control_validation import (  # noqa: E402
    RESPONSIBILITY_TRANSFER_ALPHA,
    _model_parameter_sha256,
    rollout_hierarchical,
    train_mujoco_method,
    write_cell,
)
from scripts import (  # noqa: E402
    mujoco_v17_3_audit_optimal_macro_gauge_preflight_spec as spec,
)


EXPORTED_FILES = ("cell_summary.json", "evaluation_rows.csv")
SERVER_ONLY_FILES = ("checkpoint.pt", "training_history.json")
TRACE_KEYS = (
    "RewardTraceSHA256",
    "ExecutedActionTraceSHA256",
    "LatentPolicyTraceSHA256",
)
LATENT_METRICS = (
    "episode_return",
    "LatentUpperHFPowerAbs",
    "LatentLowerLFDriftAbs",
    "AdditiveActionClipExcessMax",
    "AdditiveActionClipExcessRMS",
)


def _path_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["disturbance_mode"]), int(row["seed"])


def paired_intervention_audit(
    control_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Audit exact path identity for two evaluations of one frozen policy."""

    control = {_path_key(row): row for row in control_rows}
    candidate = {_path_key(row): row for row in candidate_rows}
    if len(control) != len(control_rows) or len(candidate) != len(candidate_rows):
        raise ValueError("v17.3 paired rows contain duplicate paths")
    if set(control) != set(candidate):
        raise ValueError("v17.3 interventions do not share heldout paths")
    trace_mismatches = {
        key: sum(
            str(control[path][key]) != str(candidate[path][key])
            for path in control
        )
        for key in TRACE_KEYS
    }
    metric_max_abs_difference = {
        key: max(
            abs(float(control[path][key]) - float(candidate[path][key]))
            for path in control
        )
        for key in LATENT_METRICS
    }
    return {
        "path_count": len(control),
        "trace_mismatches": trace_mismatches,
        "metric_max_abs_difference": metric_max_abs_difference,
        "all_trace_hashes_match": not any(trace_mismatches.values()),
    }


def _candidate_rows(
    model: Any,
    *,
    env_id: str,
    optimizer_seed: int,
    control_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for control in control_rows:
        _, row = rollout_hierarchical(
            model,
            seed=int(control["seed"]),
            env_id=str(env_id),
            disturbance_mode=str(control["disturbance_mode"]),
            steps=spec.STEPS,
            upper_period=spec.UPPER_PERIOD,
            frequency_routing=True,
            leakage_constraint=True,
            sample=False,
            upper_action_scale=spec.UPPER_ACTION_SCALE,
            lower_action_scale=spec.LOWER_ACTION_SCALE,
            upper_action_decoder_mode="causal_smoothstep_plan",
            lower_lf_alpha=RESPONSIBILITY_TRANSFER_ALPHA,
            lower_lf_rms_budget=spec.LOWER_LF_RMS_BUDGET,
            leakage_constraint_scope="joint_behavior_latent",
            upper_hf_rms_budget=spec.UPPER_HF_RMS_BUDGET,
            upper_hf_penalty_coef=0.0,
            upper_constraint_mode="primal_dual",
            responsibility_mode="additive",
            leakage_cost_mode="power_excess",
            lower_action_router_mode="causal_audit_optimal_macro_gauge",
            lower_action_router_alpha=spec.ROUTER_ALPHA,
            lower_action_router_strength=spec.CANDIDATE_STRENGTH,
            lower_action_router_observe_strength=False,
            upper_promotion_gain=0.0,
            method="freq_hrl",
            episode_horizon=spec.EPISODE_HORIZON,
        )
        row.update({
            "training_replicate_seed": int(optimizer_seed),
            "evaluation_role": "heldout_paired_gauge_intervention",
            "protocol_version": spec.FROZEN_CORE_PROTOCOL_VERSION,
            "parameter_count": int(control["parameter_count"]),
            "training_disturbance_mode": "multi_condition",
            "training_disturbance_modes": "|".join(
                spec.TRAINING_DISTURBANCE_MODES
            ),
            "responsibility_mode": "additive",
            "paired_intervention": spec.CANDIDATE_INTERVENTION,
        })
        rows.append(row)
    return rows


def run_cell(
    *,
    env_id: str,
    optimizer_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    payload, control_rows, model = train_mujoco_method(
        method="freq_hrl",
        env_id=str(env_id),
        disturbance_mode="standard",
        train_seeds=spec.TRAIN_SEEDS,
        selection_seeds=spec.SELECTION_SEEDS,
        eval_seeds=spec.EVALUATION_SEEDS,
        steps=spec.STEPS,
        iterations=spec.ITERATIONS,
        optimizer_seed=int(optimizer_seed),
        episode_horizon=spec.EPISODE_HORIZON,
        upper_period=spec.UPPER_PERIOD,
        hidden_dim=spec.HIDDEN_DIM,
        learning_rate=spec.LEARNING_RATE,
        lower_lf_rms_budget=spec.LOWER_LF_RMS_BUDGET,
        checkpoint_smoothing_window=spec.CHECKPOINT_SMOOTHING_WINDOW,
        checkpoint_min_delta=spec.CHECKPOINT_MIN_DELTA,
        checkpoint_minimum_iteration=spec.CHECKPOINT_MINIMUM_ITERATION,
        checkpoint_evaluation_interval=spec.CHECKPOINT_EVALUATION_INTERVAL,
        training_disturbance_modes=spec.TRAINING_DISTURBANCE_MODES,
        evaluation_disturbance_modes=spec.EVALUATION_DISTURBANCE_MODES,
        upper_action_scale=spec.UPPER_ACTION_SCALE,
        lower_action_scale=spec.LOWER_ACTION_SCALE,
        upper_action_decoder_mode="causal_smoothstep_plan",
        upper_promotion_gain=0.0,
        responsibility_mode="additive",
        leakage_constraint_scope="joint_behavior_latent",
        upper_hf_rms_budget=spec.UPPER_HF_RMS_BUDGET,
        upper_hf_penalty_coef=0.0,
        upper_constraint_mode="primal_dual",
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        upper_constraint_update_mode="reward_guarded_adam_projection",
        lower_constraint_update_mode="reward_guarded_adam_projection",
        constraint_dual_normalization="none",
        leakage_cost_mode="power_excess",
        lower_action_router_mode="causal_audit_optimal_macro_gauge",
        lower_action_router_alpha=spec.ROUTER_ALPHA,
        lower_action_router_strength=spec.CONTROL_STRENGTH,
        lower_action_router_training_schedule="constant",
        lower_action_router_warmup_fraction=0.0,
        lower_action_router_ramp_fraction=0.0,
        lower_action_router_observe_strength=False,
        checkpoint_selection_mode=spec.CHECKPOINT_SELECTION_MODE,
        checkpoint_score_mode=spec.CHECKPOINT_SCORE_MODE,
        code_revision=spec.FROZEN_ALGORITHM_REVISION,
        expected_source_manifest_sha256=spec.FROZEN_SOURCE_MANIFEST_SHA256,
        control_protocol_version=spec.FROZEN_CORE_PROTOCOL_VERSION,
    )
    parameter_sha256 = _model_parameter_sha256(model)
    for row in control_rows:
        row["evaluation_role"] = "heldout_paired_gauge_intervention"
        row["paired_intervention"] = spec.CONTROL_INTERVENTION
    candidate_rows = _candidate_rows(
        model,
        env_id=str(env_id),
        optimizer_seed=int(optimizer_seed),
        control_rows=control_rows,
    )
    if parameter_sha256 != _model_parameter_sha256(model):
        raise RuntimeError("v17.3 paired intervention mutated the checkpoint")
    audit = paired_intervention_audit(control_rows, candidate_rows)
    payload.update({
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "evidence_role": spec.EVIDENCE_ROLE,
        "paired_intervention_router_alpha": spec.ROUTER_ALPHA,
        "paired_intervention_control_strength": spec.CONTROL_STRENGTH,
        "paired_intervention_candidate_strength": spec.CANDIDATE_STRENGTH,
        "paired_intervention_audit": audit,
        "heldout_test_access_status": (
            "development_same_paths_loaded_twice_for_paired_intervention"
        ),
        "heldout_evaluation_pass_count": 2,
    })
    return payload, [*control_rows, *candidate_rows], model


def _complete(path: Path) -> bool:
    return all(
        (Path(path) / name).is_file()
        for name in (*EXPORTED_FILES, *SERVER_ONLY_FILES)
    )


def _export(
    full_output_dir: Path,
    export_output_dir: Path,
    *,
    server_full_output_dir: str,
) -> None:
    target = Path(export_output_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name in SERVER_ONLY_FILES:
        stale = target / name
        if stale.exists():
            stale.unlink()
    for name in EXPORTED_FILES:
        shutil.copyfile(Path(full_output_dir) / name, target / name)
    (target / "server_artifact_location.json").write_text(
        json.dumps({
            "artifact_policy": (
                "small_results_synced_full_training_artifacts_server_only_v1"
            ),
            "server_full_output_dir": str(server_full_output_dir),
            "exported_files": list(EXPORTED_FILES),
            "server_only_files": list(SERVER_ONLY_FILES),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", choices=spec.ENVIRONMENTS, required=True)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--full-output-dir", type=Path, required=True)
    parser.add_argument("--export-output-dir", type=Path, required=True)
    parser.add_argument("--server-full-output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if int(args.optimizer_seed) not in spec.OPTIMIZER_SEEDS:
        raise SystemExit("optimizer seed is outside the frozen v17.3 design")
    if _complete(args.full_output_dir):
        print(f"v17.3 reuse=complete_server_cell source={args.full_output_dir}")
    else:
        payload, rows, model = run_cell(
            env_id=args.env_id,
            optimizer_seed=args.optimizer_seed,
        )
        write_cell(args.full_output_dir, payload, rows, model)
    _export(
        args.full_output_dir,
        args.export_output_dir,
        server_full_output_dir=args.server_full_output_dir,
    )
    print(f"DONE mujoco_v17_3_paired_gauge output={args.export_output_dir}")


if __name__ == "__main__":
    main()
