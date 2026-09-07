#!/usr/bin/env python3
"""Summarize the preregistered V22 optimizer training trajectories."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from scripts.audit_protocol_v6_aggregate_gain_screen import (
    FACTORIAL_SPECS,
    TRAIN_SEEDS,
)


METRICS = {
    "regularity_lambda": "lower_regularity_lambda",
    "passenger_lambda": "lower_regularity_passenger_lambda",
    "regularity_scaled_cost": "lower_regularity_policy_scaled_cost_mean",
    "regularity_scaled_gap": (
        "lower_regularity_policy_scaled_constraint_gap"),
    "passenger_scaled_cost": (
        "lower_regularity_passenger_actor_scaled_cost_mean"),
    "passenger_scaled_gap": (
        "lower_regularity_passenger_actor_scaled_constraint_gap"),
    "actor_relative_shortfall": (
        "lower_regularity_gain_floor_actor_expected_shortfall_mean"),
    "actor_aggregate_shortfall_ratio": (
        "lower_regularity_gain_floor_actor_aggregate_shortfall_ratio"),
    "actor_expected_gain_fraction": (
        "lower_regularity_gain_floor_actor_expected_gain_fraction_mean"),
    "action_regret": "lower_regularity_policy_action_regret_mean",
    "evidence_valid": "lower_regularity_policy_evidence_valid_mean",
    "lower_action_mean": "lower_action_mean",
    "holding_vehicle_seconds": "holding_vehicle_seconds",
    "fleet_denied_dispatch_events": "fleet_denied_dispatch_events",
    "policy_entropy": "lower_policy_entropy_mean",
}
DUAL_FLOOR = 0.0001
DUAL_CEILING = 2.0
BOUND_TOLERANCE = 1e-9
CHANGE_TOLERANCE = 1e-12


def _parse_run_name(path: Path) -> tuple[str, int]:
    match = re.fullmatch(r"(.+)_seed([0-9]+)", path.parent.name)
    if match is None:
        raise ValueError(f"cannot parse config and seed from {path.parent}")
    return match.group(1), int(match.group(2))


def _metric_summary(
    frame: pd.DataFrame,
    column: str,
    *,
    window: int,
) -> dict[str, float | int]:
    if column not in frame:
        raise ValueError(f"diagnostics missing required column {column}")
    raw = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite = raw[np.isfinite(raw)]
    if not finite.size:
        raise ValueError(f"diagnostics column {column} has no finite values")
    differences = np.diff(finite)
    moving = differences[np.abs(differences) > CHANGE_TOLERANCE]
    directions = np.sign(moving)
    direction_changes = int(
        np.count_nonzero(directions[1:] != directions[:-1]))
    nonzero = finite[np.abs(finite) > CHANGE_TOLERANCE]
    signs = np.sign(nonzero)
    sign_changes = int(np.count_nonzero(signs[1:] != signs[:-1]))
    width = min(int(window), int(finite.size))
    return {
        "finite_count": int(finite.size),
        "finite_fraction": float(finite.size / max(raw.size, 1)),
        "first": float(finite[0]),
        "final": float(finite[-1]),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
        "mean": float(finite.mean()),
        "standard_deviation": float(finite.std()),
        "early_window_mean": float(finite[:width].mean()),
        "late_window_mean": float(finite[-width:].mean()),
        "late_minus_early": float(
            finite[-width:].mean() - finite[:width].mean()),
        "positive_fraction": float(np.mean(finite > CHANGE_TOLERANCE)),
        "negative_fraction": float(np.mean(finite < -CHANGE_TOLERANCE)),
        "total_variation": float(np.abs(differences).sum()),
        "direction_changes": direction_changes,
        "sign_changes": sign_changes,
    }


def _load_trajectory(
    path: Path,
    *,
    config: str,
    seed: int,
    allocation: str,
    dual_update_mode: str,
    augmented_rho: float,
    expected_episodes: int,
    window: int,
) -> dict[str, object]:
    frame = pd.read_csv(path)
    if "ep" not in frame:
        raise ValueError(f"{path} has no ep column")
    episodes = pd.to_numeric(frame["ep"], errors="coerce")
    if episodes.isna().any() or episodes.duplicated().any():
        raise ValueError(f"{path} has invalid or duplicate episode indexes")
    frame = frame.assign(ep=episodes.astype(int)).sort_values("ep")
    expected = list(range(int(expected_episodes)))
    observed = frame["ep"].tolist()
    if observed != expected:
        raise ValueError(
            f"{path} episodes differ: expected {expected}, observed {observed}")

    string_contract = {
        "lower_regularity_policy_dual_update_mode": dual_update_mode,
        "lower_regularity_passenger_dual_update_mode": dual_update_mode,
        "lower_regularity_policy_constraint_cost_mode": (
            "hf_aggregate_gain_shortfall_v4"
            if allocation == "aggregate"
            else "hf_relative_gain_shortfall_v3"),
    }
    for column, expected_value in string_contract.items():
        if column not in frame or not (
                frame[column].astype(str) == expected_value).all():
            raise ValueError(
                f"{path} violates {column}={expected_value}")
    rho_columns = (
        "lower_regularity_policy_augmented_lagrangian_rho",
        "lower_regularity_passenger_augmented_lagrangian_rho",
    )
    for column in rho_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values).all() or not np.allclose(
                values, augmented_rho):
            raise ValueError(f"{path} violates {column}={augmented_rho}")

    metrics = {
        name: _metric_summary(frame, column, window=window)
        for name, column in METRICS.items()
    }
    for name in ("regularity_lambda", "passenger_lambda"):
        values = pd.to_numeric(
            frame[METRICS[name]], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        metrics[name].update({
            "floor_hit_fraction": float(np.mean(
                finite <= DUAL_FLOOR + BOUND_TOLERANCE)),
            "ceiling_hit_fraction": float(np.mean(
                finite >= DUAL_CEILING - BOUND_TOLERANCE)),
            "final_at_floor": bool(
                finite[-1] <= DUAL_FLOOR + BOUND_TOLERANCE),
            "final_at_ceiling": bool(
                finite[-1] >= DUAL_CEILING - BOUND_TOLERANCE),
        })
    return {
        "config": config,
        "train_seed": int(seed),
        "allocation": allocation,
        "dual_update_mode": dual_update_mode,
        "augmented_lagrangian_rho": float(augmented_rho),
        "episodes": int(len(frame)),
        "diagnostics_path": str(path),
        "metrics": metrics,
    }


def _across_seed_summary(runs: Sequence[dict[str, object]]) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for name in METRICS:
        late = np.asarray([
            run["metrics"][name]["late_window_mean"] for run in runs
        ], dtype=float)
        shifts = np.asarray([
            run["metrics"][name]["late_minus_early"] for run in runs
        ], dtype=float)
        metrics[name] = {
            "late_window_mean_across_seeds": float(late.mean()),
            "late_window_min_across_seeds": float(late.min()),
            "late_window_max_across_seeds": float(late.max()),
            "late_minus_early_mean_across_seeds": float(shifts.mean()),
            "mean_positive_fraction": float(np.mean([
                run["metrics"][name]["positive_fraction"] for run in runs
            ])),
            "mean_negative_fraction": float(np.mean([
                run["metrics"][name]["negative_fraction"] for run in runs
            ])),
            "mean_total_variation": float(np.mean([
                run["metrics"][name]["total_variation"] for run in runs
            ])),
            "mean_direction_changes": float(np.mean([
                run["metrics"][name]["direction_changes"] for run in runs
            ])),
            "mean_sign_changes": float(np.mean([
                run["metrics"][name]["sign_changes"] for run in runs
            ])),
        }
    for name in ("regularity_lambda", "passenger_lambda"):
        metrics[name].update({
            "final_floor_seed_fraction": float(np.mean([
                run["metrics"][name]["final_at_floor"] for run in runs
            ])),
            "final_ceiling_seed_fraction": float(np.mean([
                run["metrics"][name]["final_at_ceiling"] for run in runs
            ])),
            "mean_floor_hit_fraction": float(np.mean([
                run["metrics"][name]["floor_hit_fraction"] for run in runs
            ])),
        })
    return metrics


def _flatten_run(run: dict[str, object]) -> dict[str, object]:
    row = {
        key: run[key]
        for key in (
            "config", "train_seed", "allocation", "dual_update_mode",
            "augmented_lagrangian_rho", "episodes", "diagnostics_path",
        )
    }
    for metric, summary in run["metrics"].items():
        for key, value in summary.items():
            row[f"{metric}_{key}"] = value
    return row


def audit_v22_optimizer_trajectories(
    logs_root: str | Path,
    *,
    specs: Iterable[tuple[str, str, str, float, bool]] = FACTORIAL_SPECS,
    train_seeds: Sequence[int] = TRAIN_SEEDS,
    expected_episodes: int = 40,
    window: int = 5,
) -> dict[str, object]:
    logs_root = Path(logs_root)
    if expected_episodes < 1 or window < 1:
        raise ValueError("expected_episodes and window must be positive")
    specs = list(specs)
    expected_configs = {spec[0] for spec in specs}
    paths: dict[tuple[str, int], Path] = {}
    for path in sorted(logs_root.rglob("diagnostics.csv")):
        config, seed = _parse_run_name(path)
        if config not in expected_configs or seed not in train_seeds:
            continue
        key = (config, seed)
        if key in paths:
            raise ValueError(f"duplicate diagnostics for {config} seed {seed}")
        paths[key] = path

    expected_keys = {
        (config, int(seed))
        for config, *_ in specs
        for seed in train_seeds
    }
    missing = sorted(expected_keys - set(paths))
    unexpected = sorted(set(paths) - expected_keys)
    if missing or unexpected:
        raise ValueError(
            f"trajectory inventory differs: missing={missing}, "
            f"unexpected={unexpected}")

    candidates = []
    flat_rows = []
    for config, allocation, update, rho, promotion_eligible in specs:
        runs = [
            _load_trajectory(
                paths[(config, int(seed))],
                config=config,
                seed=int(seed),
                allocation=allocation,
                dual_update_mode=update,
                augmented_rho=float(rho),
                expected_episodes=expected_episodes,
                window=window,
            )
            for seed in train_seeds
        ]
        flat_rows.extend(_flatten_run(run) for run in runs)
        candidates.append({
            "config": config,
            "allocation": allocation,
            "dual_update_mode": update,
            "augmented_lagrangian_rho": float(rho),
            "promotion_eligible": bool(promotion_eligible),
            "run_count": len(runs),
            "train_seeds": [int(seed) for seed in train_seeds],
            "across_seed_metrics": _across_seed_summary(runs),
            "runs": runs,
        })
    return {
        "schema_version": "freqduet-v22-optimizer-trajectory-v1",
        "logs_root": str(logs_root),
        "expected_episodes": int(expected_episodes),
        "window": int(window),
        "candidate_count": len(candidates),
        "run_count": len(flat_rows),
        "candidates": candidates,
        "flat_rows": flat_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--expected-episodes", type=int, default=40)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()
    result = audit_v22_optimizer_trajectories(
        args.logs_root,
        expected_episodes=args.expected_episodes,
        window=args.window,
    )
    csv_path = args.csv or args.out.with_suffix(".csv")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(result["flat_rows"]).to_csv(csv_path, index=False)
    printable = {key: value for key, value in result.items()
                 if key not in {"candidates", "flat_rows"}}
    printable["json"] = str(args.out)
    printable["csv"] = str(csv_path)
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
