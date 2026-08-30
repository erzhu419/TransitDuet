#!/usr/bin/env python3
"""Audit the preregistered V14 capacity-gated regularity-gain screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HARD_MAIN = "F_freqduet_protocol_v6_main_hiro"
REFERENCE = "F_freqduet_protocol_v6_noguard_hiro"
MATCHED_CONTEXT = "F_freqduet_protocol_v6_avlcompact_hiro"
CURRENT_MAIN = "F_freqduet_protocol_v6_confirmed_main_hiro"
SAME_ENTROPY = "F_freqduet_protocol_v6_w2adent_e25_c0020_hiro"
V13_ANCHOR = "F_freqduet_protocol_v6_w2adregret_l001_e25_r00025_hiro"
WEIGHTS = (("0005", 0.005), ("0010", 0.01), ("0020", 0.02))
EXPONENTS = (("1", 1.0), ("2", 2.0))
CANDIDATE_SPECS = [
    (
        "F_freqduet_protocol_v6_w2adcapgain_l001_e25_r00025_"
        f"w{weight_name}_x{exponent_name}_hiro",
        weight,
        exponent,
    )
    for exponent_name, exponent in EXPONENTS
    for weight_name, weight in WEIGHTS
]
CANDIDATES = [name for name, _, _ in CANDIDATE_SPECS]
CONFIGS = [
    HARD_MAIN,
    REFERENCE,
    MATCHED_CONTEXT,
    CURRENT_MAIN,
    SAME_ENTROPY,
    V13_ANCHOR,
    *CANDIDATES,
]
TRAIN_SEEDS = [17013, 17031, 17053, 17077]
EVAL_SEEDS = [50017, 50041, 50059, 50083]
PRIORITY = [
    "F_freqduet_protocol_v6_w2adcapgain_l001_e25_r00025_"
    f"w{weight}_x{exponent}_hiro"
    for weight, exponent in (
        ("0005", "2"), ("0005", "1"),
        ("0010", "2"), ("0010", "1"),
        ("0020", "2"), ("0020", "1"),
    )
]


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
    per_eval: pd.DataFrame,
    candidate: str,
    reference: str,
    metric: str,
) -> tuple[float, int]:
    keys = ["train_seed", "eval_seed"]
    left = _rows(per_eval, candidate)[[*keys, metric]].rename(
        columns={metric: "candidate"})
    right = _rows(per_eval, reference)[[*keys, metric]].rename(
        columns={metric: "reference"})
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    delta = (
        pd.to_numeric(paired["candidate"], errors="coerce")
        - pd.to_numeric(paired["reference"], errors="coerce"))
    if not _finite(delta):
        raise ValueError(
            f"non-finite paired delta: {candidate} vs {reference}, {metric}")
    return float(delta.mean()), int(len(delta))


def evaluate_capacity_gain_screen(aggregate_dir: Path) -> dict[str, object]:
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
        raise FileNotFoundError(f"missing V14 aggregate artifacts: {missing}")

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
        "reference_is_v13_anchor": manifest.get("reference") == V13_ANCHOR,
        "source_is_clean": manifest.get(
            "run_git_provenance", {}).get("tracked_dirty") is False,
        "expected_rollouts": (
            manifest.get("expected_rollouts") == expected_rollouts
            and len(per_eval) == expected_rollouts),
        "unique_rollouts": not per_eval.duplicated(
            ["config", "train_seed", "eval_seed"]).any(),
    }
    if not all(strict_checks.values()):
        raise ValueError(f"V14 strict checks failed: {strict_checks}")

    required_columns = {
        "config", "train_seed", "eval_seed", "headway_cv",
        "restricted_total_journey_horizon_min", "holding_vehicle_seconds",
        "fleet_denied_dispatch_events",
        "lower_causal_guard_adjustment_mean_s",
        "lower_regularity_policy_enabled", "lower_regularity_policy_mode",
        "lower_regularity_policy_constraint_scale_mode",
        "lower_regularity_policy_initial_lambda",
        "lower_regularity_policy_cost_limit",
        "lower_regularity_policy_scaled_limit",
        "lower_regularity_policy_action_regret_mean",
        "lower_regularity_policy_action_regret_max",
        "lower_regularity_policy_evidence_valid_mean",
        "lower_regularity_policy_capacity_gain_enabled",
        "lower_regularity_policy_capacity_gain_weight",
        "lower_regularity_policy_capacity_gain_scale",
        "lower_regularity_policy_capacity_exponent",
        "lower_regularity_policy_capacity_gain_mean",
        "lower_regularity_policy_scaled_capacity_gain_mean",
        "lower_regularity_policy_capacity_gain_bonus",
        "lower_regularity_policy_capacity_gate_mean",
    }
    missing_columns = sorted(required_columns - set(per_eval.columns))
    if missing_columns:
        raise ValueError(f"V14 metrics are missing: {missing_columns}")

    reference_holding = _mean(
        per_eval, REFERENCE, "holding_vehicle_seconds")
    reference_denied = _mean(
        per_eval, REFERENCE, "fleet_denied_dispatch_events")
    hard_holding = _mean(per_eval, HARD_MAIN, "holding_vehicle_seconds")
    hard_denied = _mean(
        per_eval, HARD_MAIN, "fleet_denied_dispatch_events")
    holding_limit = reference_holding + 0.10 * max(
        hard_holding - reference_holding, 0.0)
    denied_limit = reference_denied + 0.10 * max(
        hard_denied - reference_denied, 0.0)

    candidate_results = []
    for candidate, expected_weight, expected_exponent in CANDIDATE_SPECS:
        rows = _rows(per_eval, candidate)
        anchor_pair = paired.loc[
            (paired["candidate"] == candidate)
            & (paired["reference"] == V13_ANCHOR)]
        if len(rows) != expected_pairs or len(anchor_pair) != 1:
            raise ValueError(f"incomplete V14 candidate: {candidate}")
        anchor_pair = anchor_pair.iloc[0]

        adjustment = pd.to_numeric(
            rows["lower_causal_guard_adjustment_mean_s"], errors="coerce")
        enabled = pd.to_numeric(
            rows["lower_regularity_policy_enabled"], errors="coerce")
        mode = rows["lower_regularity_policy_mode"].astype(str)
        scale_mode = rows[
            "lower_regularity_policy_constraint_scale_mode"].astype(str)
        initial = pd.to_numeric(
            rows["lower_regularity_policy_initial_lambda"], errors="coerce")
        cost_limit = pd.to_numeric(
            rows["lower_regularity_policy_cost_limit"], errors="coerce")
        scaled_limit = pd.to_numeric(
            rows["lower_regularity_policy_scaled_limit"], errors="coerce")
        action_regret = pd.to_numeric(
            rows["lower_regularity_policy_action_regret_mean"],
            errors="coerce")
        evidence = pd.to_numeric(
            rows["lower_regularity_policy_evidence_valid_mean"],
            errors="coerce")
        gain_enabled = pd.to_numeric(
            rows["lower_regularity_policy_capacity_gain_enabled"],
            errors="coerce")
        gain_weight = pd.to_numeric(
            rows["lower_regularity_policy_capacity_gain_weight"],
            errors="coerce")
        gain_scale = pd.to_numeric(
            rows["lower_regularity_policy_capacity_gain_scale"],
            errors="coerce")
        gain_exponent = pd.to_numeric(
            rows["lower_regularity_policy_capacity_exponent"],
            errors="coerce")
        realized_gain = pd.to_numeric(
            rows["lower_regularity_policy_capacity_gain_mean"],
            errors="coerce")
        realized_scaled_gain = pd.to_numeric(
            rows["lower_regularity_policy_scaled_capacity_gain_mean"],
            errors="coerce")
        realized_bonus = pd.to_numeric(
            rows["lower_regularity_policy_capacity_gain_bonus"],
            errors="coerce")
        realized_gate = pd.to_numeric(
            rows["lower_regularity_policy_capacity_gate_mean"],
            errors="coerce")

        anchor_cv = float(anchor_pair["delta_headway_cv_mean"])
        anchor_cv_ci_high = float(
            anchor_pair["delta_headway_cv_ci_high"])
        anchor_journey = float(anchor_pair[
            "delta_restricted_total_journey_horizon_min_mean"])
        anchor_journey_ci_high = float(anchor_pair[
            "delta_restricted_total_journey_horizon_min_ci_high"])
        noguard_cv, noguard_pairs = _paired_delta(
            per_eval, candidate, REFERENCE, "headway_cv")
        noguard_journey, noguard_journey_pairs = _paired_delta(
            per_eval, candidate, REFERENCE,
            "restricted_total_journey_horizon_min")
        current_cv, current_pairs = _paired_delta(
            per_eval, candidate, CURRENT_MAIN, "headway_cv")
        current_journey, current_journey_pairs = _paired_delta(
            per_eval, candidate, CURRENT_MAIN,
            "restricted_total_journey_horizon_min")
        v11_cv, v11_pairs = _paired_delta(
            per_eval, candidate, SAME_ENTROPY, "headway_cv")
        v11_journey, v11_journey_pairs = _paired_delta(
            per_eval, candidate, SAME_ENTROPY,
            "restricted_total_journey_horizon_min")
        pairs_complete = all(count == expected_pairs for count in (
            int(anchor_pair["n_pairs"]), noguard_pairs,
            noguard_journey_pairs, current_pairs, current_journey_pairs,
            v11_pairs, v11_journey_pairs,
        ))
        candidate_holding = float(pd.to_numeric(
            rows["holding_vehicle_seconds"], errors="coerce").mean())
        candidate_denied = float(pd.to_numeric(
            rows["fleet_denied_dispatch_events"], errors="coerce").mean())

        mechanism_checks = {
            "paired_rollouts_complete": pairs_complete,
            "zero_execution_adjustment": bool(
                _finite(adjustment) and (adjustment.abs() <= 1e-12).all()),
            "capacity_gain_mode_locked": bool(
                (enabled == 1.0).all()
                and (mode
                     == "analytic_two_sided_capacity_gain_regret_dual_v3").all()),
            "dimensionless_regret_constraint": bool(
                (scale_mode == "cost_limit_ratio_v1").all()
                and np.allclose(initial, 0.01)
                and np.allclose(cost_limit, 0.00025)
                and np.allclose(scaled_limit, 1.0)),
            "capacity_gain_contract_locked": bool(
                (gain_enabled == 1.0).all()
                and np.allclose(gain_weight, expected_weight)
                and np.allclose(gain_scale, 0.002)
                and np.allclose(gain_exponent, expected_exponent)),
            "capacity_gain_active_in_every_rollout": bool(
                _finite(realized_gain) and (realized_gain > 0.0).all()),
            "capacity_gate_active_in_every_rollout": bool(
                _finite(realized_gate)
                and (realized_gate > 0.0).all()
                and (realized_gate <= 1.0).all()),
            "capacity_gain_arithmetic_verified": bool(
                _finite(realized_scaled_gain)
                and _finite(realized_bonus)
                and np.allclose(
                    realized_scaled_gain,
                    realized_gain / 0.002,
                    atol=3e-6,
                    rtol=1e-6,
                )
                and np.allclose(
                    realized_bonus,
                    expected_weight * realized_scaled_gain,
                    atol=1e-7,
                    rtol=1e-6,
                )),
            "regret_limit_satisfied_every_rollout": bool(
                _finite(action_regret)
                and (action_regret <= 0.00025 + 1e-12).all()),
            "causal_evidence_coverage": bool(
                _finite(evidence) and (evidence >= 0.50).all()),
        }
        outcome_checks = {
            "cv_recovers_from_v13_anchor": (
                anchor_cv <= -0.010 and anchor_cv_ci_high < 0.0),
            "journey_preserved_vs_v13_anchor": anchor_journey <= 0.20,
            "journey_beats_noguard": noguard_journey <= -0.25,
            "cv_beats_noguard": noguard_cv <= -0.030,
            "journey_beats_current_main": current_journey <= -0.50,
            "cv_noninferior_to_current_main": current_cv <= 0.005,
            "journey_beats_v11": v11_journey <= -0.25,
            "cv_noninferior_to_v11": v11_cv <= 0.005,
            "holding_gain_preserved": candidate_holding <= holding_limit,
            "denied_dispatch_gain_preserved": candidate_denied <= denied_limit,
        }
        candidate_results.append({
            "config": candidate,
            "expected_gain_weight": expected_weight,
            "expected_capacity_exponent": expected_exponent,
            "mechanism_checks": mechanism_checks,
            "outcome_checks": outcome_checks,
            "passes": (
                all(mechanism_checks.values())
                and all(outcome_checks.values())),
            "anchor_delta_headway_cv_mean": anchor_cv,
            "anchor_delta_headway_cv_ci_high": anchor_cv_ci_high,
            "anchor_delta_journey_min_mean": anchor_journey,
            "anchor_delta_journey_min_ci_high": anchor_journey_ci_high,
            "noguard_delta_headway_cv_mean": noguard_cv,
            "noguard_delta_journey_min_mean": noguard_journey,
            "current_delta_headway_cv_mean": current_cv,
            "current_delta_journey_min_mean": current_journey,
            "v11_delta_headway_cv_mean": v11_cv,
            "v11_delta_journey_min_mean": v11_journey,
            "holding_vehicle_seconds_mean": candidate_holding,
            "fleet_denied_dispatch_events_mean": candidate_denied,
            "action_regret_mean": float(action_regret.mean()),
            "action_regret_event_max_mean": float(pd.to_numeric(
                rows["lower_regularity_policy_action_regret_max"],
                errors="coerce").mean()),
            "causal_evidence_min": float(evidence.min()),
            "realized_capacity_gain_min": float(realized_gain.min()),
            "realized_capacity_gain_mean": float(realized_gain.mean()),
            "realized_capacity_gate_mean": float(realized_gate.mean()),
        })

    passing = {
        result["config"] for result in candidate_results if result["passes"]}
    selected = next((name for name in PRIORITY if name in passing), None)
    return {
        "gate_version": "freqduet-v14-capacity-gain-screen-v1",
        "status": (
            "exploratory_candidate_selected" if selected else "no_pass"),
        "claim_eligible": False,
        "selected_for_confirmation": selected,
        "strict_checks": strict_checks,
        "resource_limits": {
            "holding_vehicle_seconds_mean_max": holding_limit,
            "fleet_denied_dispatch_events_mean_max": denied_limit,
        },
        "candidate_results": candidate_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = evaluate_capacity_gain_screen(args.aggregate_dir)
    out = args.out or args.aggregate_dir / "capacity_gain_screen.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
