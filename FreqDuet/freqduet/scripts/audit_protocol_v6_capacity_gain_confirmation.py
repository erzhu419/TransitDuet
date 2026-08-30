#!/usr/bin/env python3
"""Audit the preregistered V14 capacity-gain 200-episode confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_protocol_v6_capacity_gain_screen import (
    CONFIGS as SCREEN_CONFIGS,
    CURRENT_MAIN,
    EVAL_SEEDS as SCREEN_EVAL_SEEDS,
    HARD_MAIN,
    MATCHED_CONTEXT,
    REFERENCE,
    SAME_ENTROPY,
    TRAIN_SEEDS as SCREEN_TRAIN_SEEDS,
    V13_ANCHOR,
)


SELECTED = (
    "F_freqduet_protocol_v6_w2adcapgain_l001_e25_r00025_w0020_x1_hiro"
)
CONFIGS = [
    HARD_MAIN,
    REFERENCE,
    MATCHED_CONTEXT,
    CURRENT_MAIN,
    SAME_ENTROPY,
    V13_ANCHOR,
    SELECTED,
]
TRAIN_SEEDS = [18013, 18031, 18053, 18077, 18097, 18109, 18143, 18161]
EVAL_SEEDS = [51017, 51041, 51059, 51083, 51101, 51119, 51143, 51167]


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
        - pd.to_numeric(paired["reference"], errors="coerce")
    )
    if not _finite(delta):
        raise ValueError(
            f"non-finite paired delta: {candidate} vs {reference}, {metric}")
    return float(delta.mean()), int(len(delta))


def _negative_train_seed_fraction(
    per_eval: pd.DataFrame,
    candidate: str,
    reference: str,
    metric: str,
) -> tuple[float, dict[int, float]]:
    keys = ["train_seed", "eval_seed"]
    left = _rows(per_eval, candidate)[[*keys, metric]].rename(
        columns={metric: "candidate"})
    right = _rows(per_eval, reference)[[*keys, metric]].rename(
        columns={metric: "reference"})
    paired = left.merge(right, on=keys, how="inner", validate="one_to_one")
    paired["delta"] = (
        pd.to_numeric(paired["candidate"], errors="raise")
        - pd.to_numeric(paired["reference"], errors="raise")
    )
    by_seed = paired.groupby("train_seed")["delta"].mean()
    if len(by_seed) != len(TRAIN_SEEDS):
        raise ValueError("confirmation train-seed direction grid is incomplete")
    values = {int(seed): float(value) for seed, value in by_seed.items()}
    return float((by_seed < 0.0).mean()), values


def evaluate_capacity_gain_confirmation(
    aggregate_dir: Path,
    *,
    screen_dir: Path,
) -> dict[str, object]:
    aggregate_dir = Path(aggregate_dir).resolve()
    screen_dir = Path(screen_dir).resolve()
    paths = {
        "manifest": aggregate_dir / "matrix_manifest.json",
        "per_eval": aggregate_dir / "frozen_per_eval.csv",
        "summary": aggregate_dir / "frozen_summary.csv",
        "paired": aggregate_dir / "frozen_paired_deltas.csv",
        "screen_manifest": screen_dir / "matrix_manifest.json",
        "screen_gate": screen_dir / "capacity_gain_screen.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing V14 confirmation artifacts: {missing}")

    manifest = json.loads(paths["manifest"].read_text())
    screen_manifest = json.loads(paths["screen_manifest"].read_text())
    screen_gate = json.loads(paths["screen_gate"].read_text())
    per_eval = pd.read_csv(paths["per_eval"])
    paired = pd.read_csv(paths["paired"])
    expected_pairs = len(TRAIN_SEEDS) * len(EVAL_SEEDS)
    expected_rollouts = len(CONFIGS) * expected_pairs
    screen_selected = [
        item for item in screen_gate.get("candidate_results", [])
        if item.get("config") == SELECTED
    ]
    screen_selected_passes = bool(
        len(screen_selected) == 1
        and screen_selected[0].get("passes") is True
        and all(screen_selected[0].get("mechanism_checks", {}).values())
        and all(screen_selected[0].get("outcome_checks", {}).values())
    )
    screen_commit = screen_manifest.get("run_git_provenance", {}).get(
        "commit")
    confirmation_commit = manifest.get("run_git_provenance", {}).get(
        "commit")
    strict_checks = {
        "screen_gate_selects_exact_candidate": (
            screen_gate.get("gate_version")
            == "freqduet-v14-capacity-gain-screen-v1"
            and screen_gate.get("status")
            == "exploratory_candidate_selected"
            and screen_gate.get("selected_for_confirmation") == SELECTED
            and screen_gate.get("claim_eligible") is False
            and screen_selected_passes
        ),
        "screen_manifest_is_exact": (
            screen_manifest.get("strict_complete") is True
            and screen_manifest.get("stage") == "exploratory"
            and screen_manifest.get("independent_confirmation") is False
            and screen_manifest.get("configs") == SCREEN_CONFIGS
            and screen_manifest.get("train_seeds") == SCREEN_TRAIN_SEEDS
            and screen_manifest.get("eval_seeds") == SCREEN_EVAL_SEEDS
            and screen_manifest.get("train_episodes") == 40
            and screen_manifest.get("checkpoint_ep") == 39
        ),
        "strict_complete": manifest.get("strict_complete") is True,
        "run_manifests_verified": manifest.get(
            "run_manifests_verified") is True,
        "common_random_numbers_verified": manifest.get(
            "common_random_numbers_verified") is True,
        "independent_confirmation_stage": (
            manifest.get("stage") == "confirmation"
            and manifest.get("independent_confirmation") is True
        ),
        "exact_configs": manifest.get("configs") == CONFIGS,
        "exact_train_seeds": manifest.get("train_seeds") == TRAIN_SEEDS,
        "exact_eval_seeds": manifest.get("eval_seeds") == EVAL_SEEDS,
        "seeds_disjoint_from_screen": (
            not (set(TRAIN_SEEDS) & set(SCREEN_TRAIN_SEEDS))
            and not (set(EVAL_SEEDS) & set(SCREEN_EVAL_SEEDS))
        ),
        "two_hundred_training_episodes": (
            manifest.get("train_episodes") == 200
            and manifest.get("checkpoint_ep") == 199
        ),
        "reference_is_v13_anchor": manifest.get("reference") == V13_ANCHOR,
        "behavior_source_unchanged": (
            bool(screen_commit)
            and confirmation_commit == screen_commit
            and manifest.get("run_source_fingerprint", {}).get("sha256")
            == screen_manifest.get("run_source_fingerprint", {}).get("sha256")
        ),
        "scenario_contract_unchanged": (
            manifest.get("scenario_contract", {}).get("sha256")
            == screen_manifest.get("scenario_contract", {}).get("sha256")
        ),
        "source_is_clean": manifest.get(
            "run_git_provenance", {}).get("tracked_dirty") is False,
        "expected_rollouts": (
            manifest.get("expected_rollouts") == expected_rollouts
            and len(per_eval) == expected_rollouts
        ),
        "unique_rollouts": not per_eval.duplicated(
            ["config", "train_seed", "eval_seed"]).any(),
    }
    if not all(strict_checks.values()):
        raise ValueError(
            f"V14 confirmation strict checks failed: {strict_checks}")

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
        raise ValueError(
            f"V14 confirmation metrics are missing: {missing_columns}")

    rows = _rows(per_eval, SELECTED)
    anchor_pair = paired.loc[
        (paired["candidate"] == SELECTED)
        & (paired["reference"] == V13_ANCHOR)
    ]
    if len(rows) != expected_pairs or len(anchor_pair) != 1:
        raise ValueError("V14 confirmation candidate is incomplete")
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
        rows["lower_regularity_policy_action_regret_mean"], errors="coerce")
    evidence = pd.to_numeric(
        rows["lower_regularity_policy_evidence_valid_mean"], errors="coerce")
    gain_enabled = pd.to_numeric(
        rows["lower_regularity_policy_capacity_gain_enabled"],
        errors="coerce")
    gain_weight = pd.to_numeric(
        rows["lower_regularity_policy_capacity_gain_weight"], errors="coerce")
    gain_scale = pd.to_numeric(
        rows["lower_regularity_policy_capacity_gain_scale"], errors="coerce")
    gain_exponent = pd.to_numeric(
        rows["lower_regularity_policy_capacity_exponent"], errors="coerce")
    realized_gain = pd.to_numeric(
        rows["lower_regularity_policy_capacity_gain_mean"], errors="coerce")
    realized_scaled_gain = pd.to_numeric(
        rows["lower_regularity_policy_scaled_capacity_gain_mean"],
        errors="coerce")
    realized_bonus = pd.to_numeric(
        rows["lower_regularity_policy_capacity_gain_bonus"], errors="coerce")
    realized_gate = pd.to_numeric(
        rows["lower_regularity_policy_capacity_gate_mean"], errors="coerce")
    mechanism_checks = {
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
            and np.allclose(gain_weight, 0.02)
            and np.allclose(gain_scale, 0.002)
            and np.allclose(gain_exponent, 1.0)),
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
                0.02 * realized_scaled_gain,
                atol=1e-7,
                rtol=1e-6,
            )),
        "regret_limit_satisfied_every_rollout": bool(
            _finite(action_regret)
            and (action_regret <= 0.00025 + 1e-12).all()),
        "causal_evidence_coverage": bool(
            _finite(evidence) and (evidence >= 0.50).all()),
    }

    anchor_pairs = int(anchor_pair["n_pairs"])
    anchor_cv = float(anchor_pair["delta_headway_cv_mean"])
    anchor_cv_ci_high = float(anchor_pair["delta_headway_cv_ci_high"])
    anchor_journey = float(anchor_pair[
        "delta_restricted_total_journey_horizon_min_mean"])
    anchor_journey_ci_high = float(anchor_pair[
        "delta_restricted_total_journey_horizon_min_ci_high"])
    noguard_cv, noguard_pairs = _paired_delta(
        per_eval, SELECTED, REFERENCE, "headway_cv")
    noguard_journey, noguard_journey_pairs = _paired_delta(
        per_eval, SELECTED, REFERENCE,
        "restricted_total_journey_horizon_min")
    current_cv, current_pairs = _paired_delta(
        per_eval, SELECTED, CURRENT_MAIN, "headway_cv")
    current_journey, current_journey_pairs = _paired_delta(
        per_eval, SELECTED, CURRENT_MAIN,
        "restricted_total_journey_horizon_min")
    v11_cv, v11_pairs = _paired_delta(
        per_eval, SELECTED, SAME_ENTROPY, "headway_cv")
    v11_journey, v11_journey_pairs = _paired_delta(
        per_eval, SELECTED, SAME_ENTROPY,
        "restricted_total_journey_horizon_min")
    direction_fraction, cv_by_train_seed = _negative_train_seed_fraction(
        per_eval, SELECTED, V13_ANCHOR, "headway_cv")
    paired_complete = all(count == expected_pairs for count in (
        anchor_pairs, noguard_pairs, noguard_journey_pairs,
        current_pairs, current_journey_pairs, v11_pairs, v11_journey_pairs,
    ))

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
    candidate_holding = _mean(
        per_eval, SELECTED, "holding_vehicle_seconds")
    candidate_denied = _mean(
        per_eval, SELECTED, "fleet_denied_dispatch_events")
    outcome_checks = {
        "paired_rollouts_complete": paired_complete,
        "cv_improves_v13_with_ci": (
            anchor_cv <= -0.010 and anchor_cv_ci_high < 0.0),
        "journey_noninferior_to_v13_with_ci": (
            anchor_journey <= 0.20 and anchor_journey_ci_high <= 0.20),
        "cv_direction_consistent_across_train_seeds": (
            direction_fraction >= 0.75),
        "journey_beats_noguard": noguard_journey <= -0.25,
        "cv_beats_noguard": noguard_cv <= -0.030,
        "journey_beats_current_main": current_journey <= -0.50,
        "cv_noninferior_to_current_main": current_cv <= 0.005,
        "journey_beats_v11": v11_journey <= -0.25,
        "cv_noninferior_to_v11": v11_cv <= 0.005,
        "holding_gain_preserved": candidate_holding <= holding_limit,
        "denied_dispatch_gain_preserved": candidate_denied <= denied_limit,
    }
    confirmed = all(mechanism_checks.values()) and all(outcome_checks.values())
    return {
        "gate_version": "freqduet-v14-capacity-gain-confirmation-v1",
        "status": (
            "capacity_gain_confirmed" if confirmed
            else "capacity_gain_not_confirmed"),
        "confirmation_claim_eligible": confirmed,
        "candidate": SELECTED,
        "reference": V13_ANCHOR,
        "strict_checks": strict_checks,
        "mechanism_checks": mechanism_checks,
        "outcome_checks": outcome_checks,
        "thresholds": {
            "max_anchor_headway_cv_mean": -0.010,
            "max_anchor_journey_ci_high_min": 0.20,
            "min_negative_train_seed_fraction": 0.75,
            "max_action_regret_mean": 0.00025,
            "min_causal_evidence_fraction": 0.50,
        },
        "effects": {
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
            "causal_evidence_min": float(evidence.min()),
            "realized_capacity_gain_min": float(realized_gain.min()),
            "realized_capacity_gain_mean": float(realized_gain.mean()),
            "realized_capacity_gate_mean": float(realized_gate.mean()),
            "headway_cv_negative_train_seed_fraction": direction_fraction,
            "headway_cv_delta_by_train_seed": cv_by_train_seed,
        },
        "resource_limits": {
            "holding_vehicle_seconds_mean_max": holding_limit,
            "fleet_denied_dispatch_events_mean_max": denied_limit,
        },
        "screen_artifacts": {
            "directory": str(screen_dir),
            "source_commit": screen_commit,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_dir", type=Path)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    result = evaluate_capacity_gain_confirmation(
        args.aggregate_dir,
        screen_dir=args.screen_dir,
    )
    out = args.out or args.aggregate_dir / "capacity_gain_confirmation.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_pass and not result["confirmation_claim_eligible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
