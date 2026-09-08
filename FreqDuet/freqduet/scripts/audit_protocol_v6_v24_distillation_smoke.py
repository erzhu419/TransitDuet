#!/usr/bin/env python3
"""Audit the preregistered non-effect V24 distillation smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.audit_protocol_v6_v23_projection_screen import (
    PASSENGER_REPLAY_TARGET,
    PROJECTION_TOLERANCE,
    REGULARITY_REPLAY_TARGET,
    _all_close,
    _all_zero,
    _finite,
    _numeric,
)


CANDIDATE_SPECS = {
    "F_freqduet_protocol_v6_v24_jointproj_fkl_s1_hiro": (
        "forward_kl_v2", 1),
    "F_freqduet_protocol_v6_v24_jointproj_rkl_s4_hiro": (
        "reverse_kl_v1", 4),
    "F_freqduet_protocol_v6_v24_jointproj_fkl_s4_hiro": (
        "forward_kl_v2", 4),
    "F_freqduet_protocol_v6_v24_jointproj_fkl_s8_hiro": (
        "forward_kl_v2", 8),
}
CANDIDATES = list(CANDIDATE_SPECS)
TRAIN_SEEDS = [29903]
EVAL_SEEDS = [62903]
TRAIN_EPISODES = 2


def _load_training(log_root: Path) -> pd.DataFrame:
    log_root = Path(log_root).resolve()
    frames = []
    for candidate in CANDIDATES:
        path = log_root / f"{candidate}_seed{TRAIN_SEEDS[0]}" / "diagnostics.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing V24 smoke diagnostics: {path}")
        frame = pd.read_csv(path)
        episodes = _numeric(frame, "ep")
        if (len(frame) != TRAIN_EPISODES or not _finite(episodes)
                or set(episodes.astype(int)) != set(range(TRAIN_EPISODES))):
            raise ValueError(
                f"{path}: expected exactly episodes 0--{TRAIN_EPISODES - 1}")
        frame = frame.copy()
        frame["config"] = candidate
        frame["train_seed"] = TRAIN_SEEDS[0]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _candidate_checks(
    rows: pd.DataFrame,
    distillation: str,
    steps: int,
) -> tuple[dict[str, bool], dict[str, object]]:
    applied = rows.loc[
        _numeric(rows, "lower_regularity_projection_applied") == 1.0
    ].copy()
    configured_pre = (
        "lower_regularity_projection_actor_forward_kl_episode_mean"
        if distillation == "forward_kl_v2"
        else "lower_regularity_projection_actor_reverse_kl_episode_mean")
    configured_post = (
        "lower_regularity_projection_actor_post_forward_kl_episode_mean"
        if distillation == "forward_kl_v2"
        else "lower_regularity_projection_actor_post_reverse_kl_episode_mean")
    finite_columns = (
        "lower_regularity_projection_iterations",
        "lower_regularity_projection_valid_count",
        "lower_regularity_projection_target_regularity_cost",
        "lower_regularity_projection_target_passenger_cost",
        "lower_regularity_projection_max_constraint_violation",
        "lower_regularity_projection_actor_reverse_kl_episode_mean",
        "lower_regularity_projection_actor_forward_kl_episode_mean",
        "lower_regularity_projection_actor_post_reverse_kl_episode_mean",
        "lower_regularity_projection_actor_post_forward_kl_episode_mean",
        "lower_regularity_projection_actor_post_regularity_cost_episode_mean",
        "lower_regularity_projection_actor_post_passenger_cost_episode_mean",
        "lower_regularity_projection_target_action_change_abs_mean_s_episode_mean",
        "lower_regularity_projection_actor_post_target_action_change_abs_mean_s_episode_mean",
    )
    pre = _numeric(applied, configured_pre)
    post = _numeric(applied, configured_post)
    checks = {
        "exact_distillation_contract": bool(
            (rows["lower_discrete_critic"].astype(str)
             == "zero_hold_advantage").all()
            and (rows["lower_regularity_policy_mode"].astype(str)
                 == "analytic_two_sided_hf_aggregate_gain_projection_v11").all()
            and (rows["lower_regularity_projection_mode"].astype(str)
                 == "joint_kl_soft_policy_distillation_v2").all()
            and (rows["lower_regularity_projection_distillation"].astype(str)
                 == distillation).all()
            and (_numeric(
                rows, "lower_regularity_projection_distillation_steps")
                 == float(steps)).all()
            and _all_close(
                rows, "lower_regularity_projection_regularity_target",
                REGULARITY_REPLAY_TARGET)
            and _all_close(
                rows, "lower_regularity_projection_passenger_target",
                PASSENGER_REPLAY_TARGET)),
        "projection_observed": len(applied) > 0,
        "every_teacher_projection_converged": bool(
            len(applied) > 0
            and (_numeric(
                applied, "lower_regularity_projection_converged") == 1.0).all()),
        "every_teacher_projection_meets_targets": bool(
            len(applied) > 0
            and (_numeric(
                applied,
                "lower_regularity_projection_target_regularity_cost")
                 <= REGULARITY_REPLAY_TARGET + PROJECTION_TOLERANCE).all()
            and (_numeric(
                applied,
                "lower_regularity_projection_target_passenger_cost")
                 <= PASSENGER_REPLAY_TARGET + PROJECTION_TOLERANCE).all()
            and (_numeric(
                applied,
                "lower_regularity_projection_max_constraint_violation")
                 <= PROJECTION_TOLERANCE).all()),
        "pre_post_diagnostics_finite": bool(
            len(applied) > 0
            and all(_finite(_numeric(applied, column))
                    for column in finite_columns)
            and all((_numeric(applied, column) >= -1e-12).all()
                    for column in finite_columns)),
        "configured_kl_descends_each_episode": bool(
            len(applied) > 0 and _finite(pre) and _finite(post)
            and ((post < pre - 1e-12)
                 | ((pre <= 1e-10) & (post <= pre + 1e-12))).all()),
        "registered_update_count_observed": bool(
            len(applied) > 0
            and (_numeric(
                applied,
                "lower_regularity_projection_actor_distillation_steps")
                 == float(steps)).all()),
        "soft_duals_and_penalties_absent": bool(_all_zero(rows, (
            "lower_regularity_lambda",
            "lower_regularity_passenger_lambda",
            "lower_regularity_policy_penalty",
            "lower_regularity_policy_augmented_penalty",
            "lower_regularity_passenger_actor_penalty",
            "lower_regularity_passenger_actor_augmented_penalty",
        ))),
        "zero_execution_adjustment": bool(_all_zero(
            rows, ("lower_causal_guard_adjustment_mean_s",))),
    }
    diagnostics = {
        "projection_rows": int(len(applied)),
        "configured_kl_pre_mean": float(pre.mean()) if len(applied) else None,
        "configured_kl_post_mean": float(post.mean()) if len(applied) else None,
        "configured_kl_post_pre_ratio": float(
            post.mean() / pre.mean())
        if len(applied) and float(pre.mean()) > 1e-12 else None,
        "post_regularity_cost_max": float(_numeric(
            applied,
            "lower_regularity_projection_actor_post_regularity_cost_episode_mean"
        ).max()) if len(applied) else None,
        "post_passenger_cost_max": float(_numeric(
            applied,
            "lower_regularity_projection_actor_post_passenger_cost_episode_mean"
        ).max()) if len(applied) else None,
        "pre_action_gap_abs_mean_s": float(_numeric(
            applied,
            "lower_regularity_projection_target_action_change_abs_mean_s_episode_mean"
        ).mean()) if len(applied) else None,
        "post_action_gap_abs_mean_s": float(_numeric(
            applied,
            "lower_regularity_projection_actor_post_target_action_change_abs_mean_s_episode_mean"
        ).mean()) if len(applied) else None,
    }
    return checks, diagnostics


def evaluate_v24_distillation_smoke(
    aggregate_dir: Path,
    log_root: Path,
) -> dict[str, object]:
    aggregate_dir = Path(aggregate_dir).resolve()
    manifest_path = aggregate_dir / "matrix_manifest.json"
    per_eval_path = aggregate_dir / "frozen_per_eval.csv"
    if not manifest_path.is_file() or not per_eval_path.is_file():
        raise FileNotFoundError("missing V24 smoke aggregate artifacts")
    manifest = json.loads(manifest_path.read_text())
    per_eval = pd.read_csv(per_eval_path)
    training = _load_training(log_root)
    expected_rollouts = len(CANDIDATES)
    run_git = manifest.get("run_git_provenance", {}) or {}
    aggregate_git = manifest.get("git", {}) or {}
    source_commit = str(run_git.get("commit", ""))
    strict_checks = {
        "strict_complete": manifest.get("strict_complete") is True,
        "run_manifests_verified": manifest.get(
            "run_manifests_verified") is True,
        "common_random_numbers_verified": manifest.get(
            "common_random_numbers_verified") is True,
        "exploratory_nonconfirmation": bool(
            manifest.get("stage") == "exploratory"
            and manifest.get("independent_confirmation") is False),
        "exact_configs": manifest.get("configs") == CANDIDATES,
        "exact_train_seeds": manifest.get("train_seeds") == TRAIN_SEEDS,
        "exact_eval_seeds": manifest.get("eval_seeds") == EVAL_SEEDS,
        "two_training_episodes": bool(
            manifest.get("train_episodes") == TRAIN_EPISODES
            and manifest.get("checkpoint_ep") == TRAIN_EPISODES - 1),
        "reference_is_first_candidate": (
            manifest.get("reference") == CANDIDATES[0]),
        "source_is_clean_and_identified": bool(
            run_git.get("tracked_dirty") is False
            and len(source_commit) == 40
            and all(character in "0123456789abcdef"
                    for character in source_commit.lower())),
        "aggregate_git_matches_runs": bool(
            aggregate_git.get("commit") == source_commit
            and aggregate_git.get("tracked_dirty") is False),
        "expected_rollouts": bool(
            manifest.get("expected_rollouts") == expected_rollouts
            and len(per_eval) == expected_rollouts),
        "unique_rollouts": not per_eval.duplicated(
            ["config", "train_seed", "eval_seed"]).any(),
    }
    if not all(strict_checks.values()):
        raise ValueError(f"V24 smoke strict checks failed: {strict_checks}")

    required_training_columns = {
        "ep", "config", "train_seed", "lower_discrete_critic",
        "lower_regularity_policy_mode", "lower_regularity_projection_mode",
        "lower_regularity_projection_distillation",
        "lower_regularity_projection_distillation_steps",
        "lower_regularity_projection_applied",
        "lower_regularity_projection_converged",
        "lower_regularity_projection_iterations",
        "lower_regularity_projection_valid_count",
        "lower_regularity_projection_regularity_target",
        "lower_regularity_projection_passenger_target",
        "lower_regularity_projection_target_regularity_cost",
        "lower_regularity_projection_target_passenger_cost",
        "lower_regularity_projection_max_constraint_violation",
        "lower_regularity_projection_actor_distillation_steps",
        "lower_regularity_projection_actor_reverse_kl_episode_mean",
        "lower_regularity_projection_actor_forward_kl_episode_mean",
        "lower_regularity_projection_actor_post_reverse_kl_episode_mean",
        "lower_regularity_projection_actor_post_forward_kl_episode_mean",
        "lower_regularity_projection_actor_post_regularity_cost_episode_mean",
        "lower_regularity_projection_actor_post_passenger_cost_episode_mean",
        "lower_regularity_projection_target_action_change_abs_mean_s_episode_mean",
        "lower_regularity_projection_actor_post_target_action_change_abs_mean_s_episode_mean",
        "lower_regularity_lambda", "lower_regularity_passenger_lambda",
        "lower_regularity_policy_penalty",
        "lower_regularity_policy_augmented_penalty",
        "lower_regularity_passenger_actor_penalty",
        "lower_regularity_passenger_actor_augmented_penalty",
        "lower_causal_guard_adjustment_mean_s",
    }
    missing = sorted(required_training_columns - set(training.columns))
    if missing:
        raise ValueError(f"V24 smoke training metrics are missing: {missing}")

    candidate_checks = {}
    diagnostics = {}
    for candidate, (distillation, steps) in CANDIDATE_SPECS.items():
        rows = training.loc[training["config"] == candidate].copy()
        checks, candidate_diagnostics = _candidate_checks(
            rows, distillation, steps)
        candidate_checks[candidate] = checks
        diagnostics[candidate] = candidate_diagnostics

    required_evaluation_columns = {
        "config", "train_seed", "eval_seed", "lower_policy_frozen",
        "lower_critic_frozen", "upper_policy_frozen",
        "lower_regularity_projection_applied",
        "lower_causal_guard_adjustment_mean_s",
        "lower_regularity_policy_evidence_valid_mean",
    }
    missing = sorted(required_evaluation_columns - set(per_eval.columns))
    if missing:
        raise ValueError(f"V24 smoke evaluation metrics are missing: {missing}")
    frozen_projection = _numeric(
        per_eval, "lower_regularity_projection_applied")
    frozen_evidence = _numeric(
        per_eval, "lower_regularity_policy_evidence_valid_mean")
    evaluation_checks = {
        "exact_candidate_rows": bool(
            set(per_eval["config"].astype(str)) == set(CANDIDATES)
            and set(_numeric(per_eval, "train_seed").astype(int))
            == set(TRAIN_SEEDS)
            and set(_numeric(per_eval, "eval_seed").astype(int))
            == set(EVAL_SEEDS)),
        "frozen_policies": bool(
            (_numeric(per_eval, "lower_policy_frozen") == 1.0).all()
            and (_numeric(per_eval, "lower_critic_frozen") == 1.0).all()
            and (_numeric(per_eval, "upper_policy_frozen") == 1.0).all()),
        "projection_disabled_during_frozen_rollout": bool(
            _finite(frozen_projection) and (frozen_projection == 0.0).all()),
        "zero_frozen_execution_adjustment": bool(_all_zero(
            per_eval, ("lower_causal_guard_adjustment_mean_s",))),
        "causal_evidence_present": bool(
            _finite(frozen_evidence) and (frozen_evidence >= 0.50).all()),
    }
    passes = bool(
        all(evaluation_checks.values())
        and all(all(checks.values()) for checks in candidate_checks.values()))
    return {
        "gate_version": "freqduet-v24-distillation-smoke-v1",
        "status": "mechanical_pass" if passes else "no_pass",
        "effect_evidence": False,
        "formal_screen_authorized": passes,
        "strict_checks": strict_checks,
        "candidate_checks": candidate_checks,
        "evaluation_checks": evaluation_checks,
        "diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_dir", type=Path)
    parser.add_argument("--logs-root", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    result = evaluate_v24_distillation_smoke(
        args.aggregate_dir, args.logs_root)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
    print(payload)
    if args.require_pass and not result["formal_screen_authorized"]:
        raise SystemExit("V24 smoke gate did not authorize the formal screen")


if __name__ == "__main__":
    main()
