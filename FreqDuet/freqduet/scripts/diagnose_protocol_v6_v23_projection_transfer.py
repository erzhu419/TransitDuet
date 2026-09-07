#!/usr/bin/env python3
"""Diagnose transfer from the V23 projected teacher to the frozen actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_protocol_v6_v23_projection_screen import (
    CANDIDATE,
    EVAL_SEEDS,
    PROJECTION_TOLERANCE,
    REGULARITY_FROZEN_BUDGET,
    REGULARITY_REPLAY_TARGET,
    TRAIN_SEEDS,
)


TRAINING_COLUMNS = {
    "ep",
    "lower_regularity_policy_cost_mean",
    "lower_regularity_passenger_actor_cost_mean",
    "lower_regularity_projection_applied",
    "lower_regularity_projection_target_regularity_cost",
    "lower_regularity_projection_target_passenger_cost",
    "lower_regularity_projection_actor_reverse_kl",
    "lower_regularity_projection_base_action_mean_s",
    "lower_regularity_projection_target_action_mean_s",
    "lower_regularity_projection_target_action_change_mean_s",
}

FROZEN_COLUMNS = {
    "config",
    "train_seed",
    "eval_seed",
    "headway_cv",
    "restricted_total_journey_horizon_min",
    "lower_action_mean",
    "lower_regularity_gain_floor_expected_aggregate_shortfall_ratio",
    "lower_regularity_passenger_expected_cost_mean",
}


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    source: Path,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")


def _numeric(frame: pd.DataFrame, column: str, source: Path) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{source}: non-finite values in {column}")
    return values.astype(float)


def _mean(values: pd.Series) -> float:
    return float(values.mean())


def _max(values: pd.Series) -> float:
    return float(values.max())


def analyze_v23_projection_transfer(
    logs_root: Path,
    frozen_per_eval_path: Path,
) -> dict[str, object]:
    logs_root = Path(logs_root).resolve()
    frozen_per_eval_path = Path(frozen_per_eval_path).resolve()
    frozen = pd.read_csv(frozen_per_eval_path)
    _require_columns(frozen, FROZEN_COLUMNS, frozen_per_eval_path)
    frozen = frozen.loc[frozen["config"].astype(str) == CANDIDATE].copy()

    expected_pairs = {
        (int(train_seed), int(eval_seed))
        for train_seed in TRAIN_SEEDS
        for eval_seed in EVAL_SEEDS
    }
    actual_pairs = set(zip(
        pd.to_numeric(frozen["train_seed"], errors="coerce").astype(int),
        pd.to_numeric(frozen["eval_seed"], errors="coerce").astype(int),
    ))
    if actual_pairs != expected_pairs or len(frozen) != len(expected_pairs):
        raise ValueError(
            "frozen V23 rows do not match the registered train/eval grid")

    per_seed: list[dict[str, object]] = []
    for train_seed in TRAIN_SEEDS:
        diagnostics_path = (
            logs_root / f"{CANDIDATE}_seed{train_seed}" / "diagnostics.csv")
        if not diagnostics_path.is_file():
            raise ValueError(f"missing diagnostics: {diagnostics_path}")
        training = pd.read_csv(diagnostics_path)
        _require_columns(training, TRAINING_COLUMNS, diagnostics_path)

        episodes = _numeric(training, "ep", diagnostics_path).astype(int)
        if len(training) != 40 or set(episodes) != set(range(40)):
            raise ValueError(
                f"{diagnostics_path}: expected exactly episodes 0--39")
        training = training.assign(ep=episodes).sort_values("ep")
        applied = _numeric(
            training,
            "lower_regularity_projection_applied",
            diagnostics_path,
        )
        active = training.loc[applied == 1.0].copy()
        if len(active) != 40:
            raise ValueError(
                f"{diagnostics_path}: expected projection in all 40 episodes")
        last10 = active.loc[active["ep"] >= 30].copy()
        if len(last10) != 10:
            raise ValueError(
                f"{diagnostics_path}: expected ten active episodes 30--39")

        actor_reg = _numeric(
            last10, "lower_regularity_policy_cost_mean", diagnostics_path)
        target_reg = _numeric(
            last10,
            "lower_regularity_projection_target_regularity_cost",
            diagnostics_path,
        )
        actor_pax = _numeric(
            last10,
            "lower_regularity_passenger_actor_cost_mean",
            diagnostics_path,
        )
        target_pax = _numeric(
            last10,
            "lower_regularity_projection_target_passenger_cost",
            diagnostics_path,
        )
        reverse_kl = _numeric(
            last10,
            "lower_regularity_projection_actor_reverse_kl",
            diagnostics_path,
        )
        action_change = _numeric(
            last10,
            "lower_regularity_projection_target_action_change_mean_s",
            diagnostics_path,
        )
        base_action = _numeric(
            last10,
            "lower_regularity_projection_base_action_mean_s",
            diagnostics_path,
        )
        target_action = _numeric(
            last10,
            "lower_regularity_projection_target_action_mean_s",
            diagnostics_path,
        )

        frozen_seed = frozen.loc[
            pd.to_numeric(frozen["train_seed"], errors="coerce")
            == train_seed
        ].copy()
        frozen_reg = _numeric(
            frozen_seed,
            "lower_regularity_gain_floor_expected_aggregate_shortfall_ratio",
            frozen_per_eval_path,
        )
        frozen_pax = _numeric(
            frozen_seed,
            "lower_regularity_passenger_expected_cost_mean",
            frozen_per_eval_path,
        )
        frozen_cv = _numeric(
            frozen_seed, "headway_cv", frozen_per_eval_path)
        frozen_journey = _numeric(
            frozen_seed,
            "restricted_total_journey_horizon_min",
            frozen_per_eval_path,
        )
        frozen_action = _numeric(
            frozen_seed, "lower_action_mean", frozen_per_eval_path)

        actor_reg_mean = _mean(actor_reg)
        target_reg_mean = _mean(target_reg)
        frozen_reg_mean = _mean(frozen_reg)
        per_seed.append({
            "train_seed": int(train_seed),
            "replay_last10": {
                "actor_regularity_cost_mean": actor_reg_mean,
                "teacher_regularity_cost_mean": target_reg_mean,
                "actor_minus_teacher_regularity_cost": (
                    actor_reg_mean - target_reg_mean),
                "actor_passenger_cost_mean": _mean(actor_pax),
                "teacher_passenger_cost_mean": _mean(target_pax),
                "actor_minus_teacher_passenger_cost": (
                    _mean(actor_pax) - _mean(target_pax)),
                "actor_reverse_kl_mean": _mean(reverse_kl),
                "actor_reverse_kl_max": _max(reverse_kl),
                "teacher_minus_actor_action_mean_s": _mean(action_change),
                "soft_base_action_mean_s": _mean(base_action),
                "teacher_action_mean_s": _mean(target_action),
                "episodes_actor_regularity_above_teacher": int(
                    (actor_reg > target_reg + PROJECTION_TOLERANCE).sum()),
            },
            "frozen_rollouts": {
                "regularity_cost_mean": frozen_reg_mean,
                "regularity_cost_max": _max(frozen_reg),
                "passenger_cost_mean": _mean(frozen_pax),
                "headway_cv_mean": _mean(frozen_cv),
                "journey_min_mean": _mean(frozen_journey),
                "lower_action_mean_s": _mean(frozen_action),
                "frozen_minus_replay_actor_regularity_cost": (
                    frozen_reg_mean - actor_reg_mean),
            },
        })

    replay_excess = np.asarray([
        row["replay_last10"]["actor_minus_teacher_regularity_cost"]
        for row in per_seed
    ], dtype=float)
    frozen_costs = np.asarray([
        row["frozen_rollouts"]["regularity_cost_mean"]
        for row in per_seed
    ], dtype=float)
    frozen_minus_replay = np.asarray([
        row["frozen_rollouts"][
            "frozen_minus_replay_actor_regularity_cost"]
        for row in per_seed
    ], dtype=float)
    teacher_max = max(
        row["replay_last10"]["teacher_regularity_cost_mean"]
        for row in per_seed
    )
    actor_gap_seed_count = int(
        (replay_excess > PROJECTION_TOLERANCE).sum())
    rollout_gap_seed_count = int(
        (frozen_minus_replay > PROJECTION_TOLERANCE).sum())
    frozen_budget_failure_seed_count = int(
        (frozen_costs > REGULARITY_FROZEN_BUDGET).sum())

    teacher_exact = bool(
        teacher_max
        <= REGULARITY_REPLAY_TARGET + PROJECTION_TOLERANCE)
    if teacher_exact and actor_gap_seed_count == len(TRAIN_SEEDS):
        primary_diagnosis = "actor_teacher_distillation_gap_observed"
    elif teacher_exact and rollout_gap_seed_count > 0:
        primary_diagnosis = "frozen_distribution_transfer_gap_observed"
    else:
        primary_diagnosis = "mixed_or_inconclusive"

    return {
        "schema_version": "freqduet-v23-projection-transfer-v1",
        "candidate": CANDIDATE,
        "registered_targets": {
            "replay_regularity": REGULARITY_REPLAY_TARGET,
            "frozen_regularity_budget": REGULARITY_FROZEN_BUDGET,
        },
        "per_seed": per_seed,
        "aggregate": {
            "teacher_target_exact": teacher_exact,
            "actor_gap_seed_count": actor_gap_seed_count,
            "actor_regularity_excess_last10_mean": float(
                replay_excess.mean()),
            "actor_regularity_excess_last10_min": float(
                replay_excess.min()),
            "actor_regularity_excess_last10_max": float(
                replay_excess.max()),
            "rollout_gap_seed_count": rollout_gap_seed_count,
            "frozen_minus_replay_actor_regularity_mean": float(
                frozen_minus_replay.mean()),
            "frozen_budget_failure_seed_count": (
                frozen_budget_failure_seed_count),
            "primary_diagnosis": primary_diagnosis,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs_root", type=Path)
    parser.add_argument("frozen_per_eval", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = analyze_v23_projection_transfer(
        args.logs_root, args.frozen_per_eval)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
