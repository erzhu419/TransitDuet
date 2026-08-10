#!/usr/bin/env python3
"""Audit the preregistered V12 dimensionless-constraint screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HARD_MAIN = "F_freqduet_protocol_v6_main_hiro"
REFERENCE = "F_freqduet_protocol_v6_noguard_hiro"
MATCHED_CONTEXT = "F_freqduet_protocol_v6_avlcompact_hiro"
CURRENT_MAIN = "F_freqduet_protocol_v6_confirmed_main_hiro"
NO_ENTROPY = {
    "0010": "F_freqduet_protocol_v6_w2actiondual_c0010_hiro",
    "0020": "F_freqduet_protocol_v6_w2actiondual_c0020_hiro",
}
SAME_ENTROPY = {
    "0010": "F_freqduet_protocol_v6_w2adent_e50_c0010_hiro",
    "0020": "F_freqduet_protocol_v6_w2adent_e25_c0020_hiro",
}
LIMIT_SPECS = (("0010", "50"), ("0020", "25"))
INITIAL_SPECS = (("005", 0.05), ("010", 0.10), ("020", 0.20))
CANDIDATE_SPECS = [
    (
        f"F_freqduet_protocol_v6_w2adnorm_l{initial_name}_e{fraction}"
        f"_c{limit}_hiro",
        initial_lambda,
        float(fraction) / 100.0,
        float(limit) / 10_000.0,
        limit,
    )
    for limit, fraction in LIMIT_SPECS
    for initial_name, initial_lambda in INITIAL_SPECS
]
CANDIDATES = [spec[0] for spec in CANDIDATE_SPECS]
CONFIGS = [
    HARD_MAIN,
    REFERENCE,
    MATCHED_CONTEXT,
    CURRENT_MAIN,
    NO_ENTROPY["0010"],
    NO_ENTROPY["0020"],
    SAME_ENTROPY["0010"],
    SAME_ENTROPY["0020"],
    *CANDIDATES,
]
TRAIN_SEEDS = [15013, 15031, 15053, 15077]
EVAL_SEEDS = [48017, 48041, 48059, 48083]
SCENARIO_SHA256 = (
    "45f381a5d79c0cc5ab3e8257c2cf870af62bf076d46563c348eb1194bc116f17")
PRIORITY = [
    f"F_freqduet_protocol_v6_w2adnorm_l{initial}_e{fraction}_c{limit}_hiro"
    for initial, _ in INITIAL_SPECS
    for limit, fraction in (("0020", "25"), ("0010", "50"))
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(values: pd.Series) -> bool:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return bool(numeric.size and np.isfinite(numeric).all())


def _rows(per_eval: pd.DataFrame, config: str) -> pd.DataFrame:
    return per_eval.loc[per_eval["config"] == config].copy()


def _mean(per_eval: pd.DataFrame, config: str, metric: str) -> float:
    values = _rows(per_eval, config)[metric]
    if not _finite(values):
        raise ValueError(f"non-finite metric for {config}: {metric}")
    return float(pd.to_numeric(values).mean())


def _paired_delta(
    per_eval: pd.DataFrame, candidate: str, reference: str, metric: str,
) -> tuple[float, int]:
    keys = ["train_seed", "eval_seed"]
    left = _rows(per_eval, candidate)[[*keys, metric]].rename(
        columns={metric: "candidate"})
    right = _rows(per_eval, reference)[[*keys, metric]].rename(
        columns={metric: "reference"})
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    values = (
        pd.to_numeric(paired["candidate"], errors="coerce")
        - pd.to_numeric(paired["reference"], errors="coerce"))
    if not _finite(values):
        raise ValueError(
            f"non-finite paired delta: {candidate} vs {reference}, {metric}")
    return float(values.mean()), int(len(values))


def evaluate_normalized_constraint_screen(
    aggregate_dir: Path,
) -> dict[str, object]:
    aggregate_dir = Path(aggregate_dir).resolve()
    paths = {
        name: aggregate_dir / filename
        for name, filename in {
            "manifest": "matrix_manifest.json",
            "per_eval": "frozen_per_eval.csv",
            "summary": "frozen_summary.csv",
            "paired": "frozen_paired_deltas.csv",
        }.items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing V12 aggregate artifacts: {missing}")

    manifest = json.loads(paths["manifest"].read_text())
    per_eval = pd.read_csv(paths["per_eval"])
    paired = pd.read_csv(paths["paired"])
    expected_pairs = len(TRAIN_SEEDS) * len(EVAL_SEEDS)
    expected_rollouts = len(CONFIGS) * expected_pairs
    strict_checks = {
        "strict_complete": manifest.get("strict_complete") is True,
        "run_manifests_verified": manifest.get(
            "run_manifests_verified") is True,
        "common_random_numbers_verified": manifest.get(
            "common_random_numbers_verified") is True,
        "exploratory_stage": (
            manifest.get("stage") == "exploratory"
            and manifest.get("independent_confirmation") is False),
        "exact_configs": manifest.get("configs") == CONFIGS,
        "exact_train_seeds": manifest.get("train_seeds") == TRAIN_SEEDS,
        "exact_eval_seeds": manifest.get("eval_seeds") == EVAL_SEEDS,
        "forty_training_episodes": (
            manifest.get("train_episodes") == 40
            and manifest.get("checkpoint_ep") == 39),
        "reference_is_noguard": manifest.get("reference") == REFERENCE,
        "source_is_clean": manifest.get(
            "run_git_provenance", {}).get("tracked_dirty") is False,
        "scenario_unchanged": manifest.get(
            "scenario_contract", {}).get("sha256") == SCENARIO_SHA256,
        "expected_rollouts": (
            manifest.get("expected_rollouts") == expected_rollouts
            and len(per_eval) == expected_rollouts),
        "unique_rollouts": not per_eval.duplicated(
            ["config", "train_seed", "eval_seed"]).any(),
    }
    if not all(strict_checks.values()):
        raise ValueError(f"V12 strict checks failed: {strict_checks}")

    required_columns = {
        "config", "train_seed", "eval_seed", "headway_cv",
        "restricted_total_journey_horizon_min", "holding_vehicle_seconds",
        "fleet_denied_dispatch_events",
        "lower_causal_guard_adjustment_mean_s",
        "lower_regularity_policy_enabled",
        "lower_regularity_policy_constraint_scale_mode",
        "lower_regularity_policy_initial_lambda",
        "lower_regularity_policy_scaled_limit",
        "lower_regularity_policy_action_cost_mean",
        "lower_regularity_policy_oracle_action_cost_mean",
        "lower_regularity_policy_evidence_valid_mean",
        "lower_regularity_entropy_split_enabled",
        "lower_regularity_entropy_target_fraction",
        "lower_regularity_alpha", "lower_alpha",
        "lower_regularity_policy_entropy_valid_mean",
    }
    missing_columns = sorted(required_columns - set(per_eval.columns))
    if missing_columns:
        raise ValueError(f"V12 metrics are missing: {missing_columns}")

    hard_journey = _mean(
        per_eval, HARD_MAIN, "restricted_total_journey_horizon_min")
    reference_holding = _mean(
        per_eval, REFERENCE, "holding_vehicle_seconds")
    reference_denied = _mean(
        per_eval, REFERENCE, "fleet_denied_dispatch_events")
    hard_holding = _mean(per_eval, HARD_MAIN, "holding_vehicle_seconds")
    hard_denied = _mean(per_eval, HARD_MAIN, "fleet_denied_dispatch_events")
    holding_limit = reference_holding + 0.10 * max(
        hard_holding - reference_holding, 0.0)
    denied_limit = reference_denied + 0.10 * max(
        hard_denied - reference_denied, 0.0)

    candidate_results = []
    for (candidate, initial_lambda, entropy_fraction, cost_limit,
         limit_name) in CANDIDATE_SPECS:
        rows = _rows(per_eval, candidate)
        pair = paired.loc[
            (paired["candidate"] == candidate)
            & (paired["reference"] == REFERENCE)]
        if len(rows) != expected_pairs or len(pair) != 1:
            raise ValueError(f"incomplete V12 candidate: {candidate}")
        pair = pair.iloc[0]
        same_entropy = SAME_ENTROPY[limit_name]

        adjustment = pd.to_numeric(
            rows["lower_causal_guard_adjustment_mean_s"], errors="coerce")
        enabled = pd.to_numeric(
            rows["lower_regularity_policy_enabled"], errors="coerce")
        scale_mode = rows[
            "lower_regularity_policy_constraint_scale_mode"].astype(str)
        initial = pd.to_numeric(
            rows["lower_regularity_policy_initial_lambda"], errors="coerce")
        scaled_limit = pd.to_numeric(
            rows["lower_regularity_policy_scaled_limit"], errors="coerce")
        action_cost = pd.to_numeric(
            rows["lower_regularity_policy_action_cost_mean"], errors="coerce")
        oracle_cost = pd.to_numeric(
            rows["lower_regularity_policy_oracle_action_cost_mean"],
            errors="coerce")
        evidence = pd.to_numeric(
            rows["lower_regularity_policy_evidence_valid_mean"],
            errors="coerce")
        split = pd.to_numeric(
            rows["lower_regularity_entropy_split_enabled"], errors="coerce")
        target = pd.to_numeric(
            rows["lower_regularity_entropy_target_fraction"], errors="coerce")
        valid_alpha = pd.to_numeric(
            rows["lower_regularity_alpha"], errors="coerce")
        base_alpha = pd.to_numeric(rows["lower_alpha"], errors="coerce")
        valid_entropy = pd.to_numeric(
            rows["lower_regularity_policy_entropy_valid_mean"],
            errors="coerce")
        baseline_action_cost = _mean(
            per_eval, same_entropy,
            "lower_regularity_policy_action_cost_mean")
        baseline_entropy = _mean(
            per_eval, same_entropy,
            "lower_regularity_policy_entropy_valid_mean")

        context_cv, context_pairs = _paired_delta(
            per_eval, candidate, MATCHED_CONTEXT, "headway_cv")
        context_journey, context_journey_pairs = _paired_delta(
            per_eval, candidate, MATCHED_CONTEXT,
            "restricted_total_journey_horizon_min")
        current_cv, current_pairs = _paired_delta(
            per_eval, candidate, CURRENT_MAIN, "headway_cv")
        current_journey, current_journey_pairs = _paired_delta(
            per_eval, candidate, CURRENT_MAIN,
            "restricted_total_journey_horizon_min")
        baseline_cv, baseline_pairs = _paired_delta(
            per_eval, candidate, same_entropy, "headway_cv")
        baseline_journey, baseline_journey_pairs = _paired_delta(
            per_eval, candidate, same_entropy,
            "restricted_total_journey_horizon_min")

        journey_delta = float(
            pair["delta_restricted_total_journey_horizon_min_mean"])
        cv_delta = float(pair["delta_headway_cv_mean"])
        candidate_journey = _mean(
            per_eval, candidate, "restricted_total_journey_horizon_min")
        candidate_holding = _mean(
            per_eval, candidate, "holding_vehicle_seconds")
        candidate_denied = _mean(
            per_eval, candidate, "fleet_denied_dispatch_events")
        entropy_ceiling = entropy_fraction * float(np.log(7.0)) + 0.15
        gates = {
            "paired_rollouts_complete": (
                int(pair["n_pairs"]) == expected_pairs
                and context_pairs == expected_pairs
                and context_journey_pairs == expected_pairs
                and current_pairs == expected_pairs
                and current_journey_pairs == expected_pairs
                and baseline_pairs == expected_pairs
                and baseline_journey_pairs == expected_pairs),
            "zero_execution_adjustment": (
                _finite(adjustment) and float(adjustment.abs().max()) == 0.0),
            "objective_enabled_every_rollout": (
                _finite(enabled) and bool((enabled == 1.0).all())),
            "dimensionless_constraint_enabled": bool(
                (scale_mode == "cost_limit_ratio_v1").all()),
            "initial_lambda_is_exact": (
                _finite(initial)
                and bool(np.allclose(initial, initial_lambda, atol=1e-12))),
            "scaled_limit_is_one": (
                _finite(scaled_limit)
                and bool(np.allclose(scaled_limit, 1.0, atol=1e-12))),
            "causal_evidence_every_rollout": (
                _finite(evidence) and bool((evidence >= 0.50).all())),
            "discrete_oracle_below_limit": (
                _finite(oracle_cost)
                and bool((oracle_cost <= 0.25 * cost_limit).all())),
            "action_cost_satisfies_limit": (
                _finite(action_cost)
                and bool((action_cost <= cost_limit + 1e-12).all())),
            "action_cost_improves_same_entropy": (
                float(action_cost.mean()) <= baseline_action_cost - 0.00020),
            "conditional_entropy_enabled": (
                _finite(split) and bool((split == 1.0).all())),
            "entropy_target_is_exact": (
                _finite(target)
                and bool(np.allclose(target, entropy_fraction, atol=1e-12))),
            "valid_temperature_is_lower": (
                _finite(valid_alpha) and _finite(base_alpha)
                and bool((valid_alpha < base_alpha).all())),
            "valid_entropy_reaches_target": (
                _finite(valid_entropy)
                and bool((valid_entropy <= entropy_ceiling).all())),
            "entropy_not_reversed_vs_same_entropy": (
                float(valid_entropy.mean()) <= baseline_entropy + 0.05),
            "journey_ci_noninferior_to_noguard": float(pair[
                "delta_restricted_total_journey_horizon_min_ci_high"]) <= 0.15,
            "journey_better_than_hard_main": candidate_journey < hard_journey,
            "cv_improves_over_noguard": cv_delta <= -0.02,
            "cv_ci_excludes_zero": float(
                pair["delta_headway_cv_ci_high"]) < 0.0,
            "holding_gain_preserved": candidate_holding <= holding_limit,
            "denied_dispatch_gain_preserved": candidate_denied <= denied_limit,
            "journey_noninferior_to_context": context_journey <= 0.15,
            "cv_improves_over_context": context_cv <= -0.01,
            "journey_noninferior_to_current_main": current_journey <= 0.15,
            "cv_improves_over_current_main": current_cv <= -0.005,
            "journey_noninferior_to_same_entropy": baseline_journey <= 0.15,
            "cv_improves_over_same_entropy": baseline_cv <= -0.005,
        }
        candidate_results.append({
            "candidate": candidate,
            "passes": bool(all(gates.values())),
            "gates": gates,
            "cost_limit": cost_limit,
            "initial_lambda": initial_lambda,
            "entropy_target_fraction": entropy_fraction,
            "action_cost_mean": float(action_cost.mean()),
            "action_cost_max_rollout_mean": float(action_cost.max()),
            "same_entropy_action_cost_mean": baseline_action_cost,
            "oracle_action_cost_mean": float(oracle_cost.mean()),
            "evidence_coverage_min": float(evidence.min()),
            "valid_alpha_mean": float(valid_alpha.mean()),
            "base_alpha_mean": float(base_alpha.mean()),
            "valid_entropy_mean": float(valid_entropy.mean()),
            "same_entropy_valid_entropy_mean": baseline_entropy,
            "journey_delta_vs_noguard_min": journey_delta,
            "journey_delta_ci_high": float(pair[
                "delta_restricted_total_journey_horizon_min_ci_high"]),
            "headway_cv_delta_vs_noguard": cv_delta,
            "headway_cv_delta_ci_high": float(
                pair["delta_headway_cv_ci_high"]),
            "journey_delta_vs_context_min": context_journey,
            "headway_cv_delta_vs_context": context_cv,
            "journey_delta_vs_current_main_min": current_journey,
            "headway_cv_delta_vs_current_main": current_cv,
            "journey_delta_vs_same_entropy_min": baseline_journey,
            "headway_cv_delta_vs_same_entropy": baseline_cv,
            "holding_delta_vs_noguard_vehicle_s": float(
                pair["delta_holding_vehicle_seconds_mean"]),
            "denied_dispatch_delta_vs_noguard": float(
                pair["delta_fleet_denied_dispatch_events_mean"]),
        })

    passing = {row["candidate"] for row in candidate_results if row["passes"]}
    selected = next((name for name in PRIORITY if name in passing), None)
    return {
        "gate_version": "freqduet-v12-normalized-constraint-screen-v1",
        "status": (
            "exploratory_candidate_selected" if selected else "no_pass"),
        "claim_eligible": False,
        "selected_for_confirmation": selected,
        "passing_candidates": [name for name in PRIORITY if name in passing],
        "selection_priority": PRIORITY,
        "strict_checks": strict_checks,
        "thresholds": {
            "min_evidence_coverage": 0.50,
            "max_oracle_cost_fraction_of_limit": 0.25,
            "min_action_cost_reduction_vs_same_entropy": 0.00020,
            "max_journey_ci_high_vs_noguard_min": 0.15,
            "min_cv_improvement_vs_noguard": 0.02,
            "min_cv_improvement_vs_context": 0.01,
            "min_cv_improvement_vs_current_main": 0.005,
            "min_cv_improvement_vs_same_entropy": 0.005,
            "max_entropy_reversal_vs_same_entropy_nats": 0.05,
            "max_gain_reversal_fraction": 0.10,
        },
        "candidate_results": candidate_results,
        "matrix_provenance": {
            "git": manifest.get("run_git_provenance"),
            "source_fingerprint_sha256": manifest.get(
                "source_fingerprint", {}).get("sha256"),
            "launch_analysis_sha256": manifest.get(
                "launch_analysis_fingerprint", {}).get("sha256"),
            "scenario_contract_sha256": manifest.get(
                "scenario_contract", {}).get("sha256"),
        },
        "input_artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    result = evaluate_normalized_constraint_screen(args.aggregate_dir)
    out = args.out or args.aggregate_dir / "normalized_constraint_screen.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_pass and not result["selected_for_confirmation"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
