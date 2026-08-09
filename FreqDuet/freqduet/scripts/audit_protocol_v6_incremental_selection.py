#!/usr/bin/env python3
"""Apply the preregistered V6 incremental-regularity selection gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_MAIN = "F_freqduet_protocol_v6_main_hiro"
DEFAULT_REFERENCE = "F_freqduet_protocol_v6_noguard_hiro"
DEFAULT_CANDIDATES = [
    f"F_freqduet_protocol_v6_avlbal_w{weight}_hiro"
    for weight in ("05", "1", "2", "4")
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_row(frame: pd.DataFrame, column: str, value: str) -> pd.Series:
    rows = frame.loc[frame[column] == value]
    if len(rows) != 1:
        raise ValueError(
            f"expected one {column}={value!r} row, observed {len(rows)}")
    return rows.iloc[0]


def _finite(values: Iterable[object]) -> bool:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy()
    return bool(array.size and np.isfinite(array).all())


def evaluate_selection(
    aggregate_dir: Path,
    *,
    candidates: list[str] | None = None,
    main: str = DEFAULT_MAIN,
    reference: str = DEFAULT_REFERENCE,
    max_journey_regression_min: float = 0.15,
    min_headway_cv_improvement: float = 0.02,
    min_follower_coverage: float = 0.50,
    max_gain_reversal_fraction: float = 0.10,
) -> dict[str, object]:
    aggregate_dir = Path(aggregate_dir).resolve()
    candidates = list(candidates or DEFAULT_CANDIDATES)
    if len(candidates) != len(set(candidates)) or not candidates:
        raise ValueError("candidate configs must be non-empty and unique")

    paths = {
        "manifest": aggregate_dir / "matrix_manifest.json",
        "per_eval": aggregate_dir / "frozen_per_eval.csv",
        "summary": aggregate_dir / "frozen_summary.csv",
        "paired": aggregate_dir / "frozen_paired_deltas.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing aggregate artifacts: {missing}")

    manifest = json.loads(paths["manifest"].read_text())
    per_eval = pd.read_csv(paths["per_eval"])
    summary = pd.read_csv(paths["summary"])
    paired = pd.read_csv(paths["paired"])

    expected_rollouts = int(manifest.get("expected_rollouts", -1))
    strict_checks = {
        "strict_complete": manifest.get("strict_complete") is True,
        "common_random_numbers_verified": (
            manifest.get("common_random_numbers_verified") is True),
        "run_manifests_verified": (
            manifest.get("run_manifests_verified") is True),
        "exploratory_stage": manifest.get("stage") == "exploratory",
        "not_labeled_confirmation": (
            manifest.get("independent_confirmation") is False),
        "expected_rollout_count": expected_rollouts == len(per_eval),
        "unique_rollout_keys": not per_eval.duplicated(
            ["config", "train_seed", "eval_seed"]).any(),
    }
    if not all(strict_checks.values()):
        raise ValueError(f"aggregate strict checks failed: {strict_checks}")

    requested = {main, reference, *candidates}
    manifest_configs = set(manifest.get("configs", []))
    if not requested.issubset(manifest_configs):
        raise ValueError(
            "selection configs are absent from the matrix: "
            f"{sorted(requested - manifest_configs)}")
    if str(manifest.get("reference")) != reference:
        raise ValueError("matrix reference does not match the selection gate")

    n_train = len(manifest.get("train_seeds", []))
    n_eval = len(manifest.get("eval_seeds", []))
    expected_candidate_rows = n_train * n_eval
    main_row = _unique_row(summary, "config", main)
    reference_row = _unique_row(summary, "config", reference)

    metric_columns = {
        "journey": "restricted_total_journey_horizon_min_mean",
        "headway_cv": "headway_cv_mean",
        "holding": "holding_vehicle_seconds_mean",
        "denied_dispatch": "fleet_denied_dispatch_events_mean",
    }
    main_values = {
        key: float(main_row[column])
        for key, column in metric_columns.items()
    }
    reference_values = {
        key: float(reference_row[column])
        for key, column in metric_columns.items()
    }

    holding_gain = max(
        0.0, main_values["holding"] - reference_values["holding"])
    denied_gain = max(
        0.0,
        main_values["denied_dispatch"]
        - reference_values["denied_dispatch"],
    )
    holding_limit = (
        reference_values["holding"]
        + max_gain_reversal_fraction * holding_gain)
    denied_limit = (
        reference_values["denied_dispatch"]
        + max_gain_reversal_fraction * denied_gain)

    candidate_results = []
    for candidate in candidates:
        candidate_summary = _unique_row(summary, "config", candidate)
        pair = paired.loc[
            (paired["candidate"] == candidate)
            & (paired["reference"] == reference)
        ]
        if len(pair) != 1:
            raise ValueError(
                f"expected one paired row for {candidate}, observed {len(pair)}")
        pair = pair.iloc[0]
        rollouts = per_eval.loc[per_eval["config"] == candidate].copy()
        if len(rollouts) != expected_candidate_rows:
            raise ValueError(
                f"{candidate}: observed {len(rollouts)} rollouts, expected "
                f"{expected_candidate_rows}")

        adjustment = pd.to_numeric(
            rollouts["lower_causal_guard_adjustment_mean_s"],
            errors="coerce",
        )
        evidence = pd.to_numeric(
            rollouts["lower_departure_regularity_evidence_valid_mean"],
            errors="coerce",
        )
        follower = pd.to_numeric(
            rollouts["lower_departure_regularity_follower_valid_mean"],
            errors="coerce",
        )
        baseline_loss = pd.to_numeric(
            rollouts["lower_departure_regularity_baseline_loss_mean"],
            errors="coerce",
        )
        post_loss = pd.to_numeric(
            rollouts["lower_departure_regularity_post_loss_mean"],
            errors="coerce",
        )

        journey_delta = float(
            pair["delta_restricted_total_journey_horizon_min_mean"])
        cv_delta = float(pair["delta_headway_cv_mean"])
        candidate_holding = float(
            candidate_summary[metric_columns["holding"]])
        candidate_denied = float(
            candidate_summary[metric_columns["denied_dispatch"]])
        gates = {
            "paired_rollouts_complete": (
                int(pair["n_pairs"]) == expected_candidate_rows),
            "zero_execution_adjustment": (
                _finite(adjustment) and float(adjustment.abs().max()) == 0.0),
            "evidence_reported_every_rollout": (
                _finite(evidence) and bool((evidence > 0.0).all())),
            "follower_coverage_every_rollout": (
                _finite(follower)
                and bool((follower >= min_follower_coverage).all())),
            "journey_within_reference_margin": (
                journey_delta <= max_journey_regression_min),
            "journey_better_than_hard_main": (
                float(candidate_summary[metric_columns["journey"]])
                < main_values["journey"]),
            "headway_cv_improvement": (
                cv_delta <= -min_headway_cv_improvement),
            "holding_gain_preserved": candidate_holding <= holding_limit,
            "denied_dispatch_gain_preserved": candidate_denied <= denied_limit,
        }
        loss_improvement = baseline_loss - post_loss
        candidate_results.append({
            "candidate": candidate,
            "passes": bool(all(gates.values())),
            "gates": gates,
            "n_rollouts": int(len(rollouts)),
            "journey_delta_vs_reference_min": journey_delta,
            "journey_delta_ci_low": float(pair[
                "delta_restricted_total_journey_horizon_min_ci_low"]),
            "journey_delta_ci_high": float(pair[
                "delta_restricted_total_journey_horizon_min_ci_high"]),
            "headway_cv_delta_vs_reference": cv_delta,
            "headway_cv_delta_ci_low": float(
                pair["delta_headway_cv_ci_low"]),
            "headway_cv_delta_ci_high": float(
                pair["delta_headway_cv_ci_high"]),
            "holding_delta_vs_reference_vehicle_s": float(
                pair["delta_holding_vehicle_seconds_mean"]),
            "denied_dispatch_delta_vs_reference": float(
                pair["delta_fleet_denied_dispatch_events_mean"]),
            "execution_adjustment_abs_max_s": float(adjustment.abs().max()),
            "evidence_coverage_min": float(evidence.min()),
            "follower_coverage_min": float(follower.min()),
            "regularity_loss_improvement_mean": float(
                loss_improvement.mean()),
            "regularity_loss_improvement_min": float(
                loss_improvement.min()),
        })

    passing = [
        item["candidate"] for item in candidate_results if item["passes"]]
    status = (
        "unique_pass" if len(passing) == 1
        else "no_pass" if not passing
        else "ambiguous_multiple_passes"
    )
    return {
        "gate_version": "freqduet-v6-incremental-selection-v1",
        "status": status,
        "selected_candidate": passing[0] if len(passing) == 1 else None,
        "passing_candidates": passing,
        "main": main,
        "reference": reference,
        "thresholds": {
            "max_journey_regression_min": max_journey_regression_min,
            "min_headway_cv_improvement": min_headway_cv_improvement,
            "min_follower_coverage": min_follower_coverage,
            "max_gain_reversal_fraction": max_gain_reversal_fraction,
        },
        "strict_checks": strict_checks,
        "expected_candidate_rollouts": expected_candidate_rows,
        "main_values": main_values,
        "reference_values": reference_values,
        "holding_reversal_limit": holding_limit,
        "denied_dispatch_reversal_limit": denied_limit,
        "candidate_results": candidate_results,
        "input_artifacts": {
            key: {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for key, path in paths.items()
        },
        "matrix_provenance": {
            "protocol_version": manifest.get("protocol_version"),
            "source_fingerprint_sha256": (
                manifest.get("run_source_fingerprint", {}).get("sha256")),
            "scenario_contract_sha256": (
                manifest.get("scenario_contract", {}).get("sha256")),
            "launch_analysis_sha256": manifest.get(
                "launch_analysis_sha256"),
            "run_git_provenance": manifest.get("run_git_provenance"),
        },
    }


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_dir", type=Path)
    parser.add_argument(
        "--candidates", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--main", default=DEFAULT_MAIN)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--require-unique-pass", action="store_true")
    args = parser.parse_args()

    result = evaluate_selection(
        args.aggregate_dir,
        candidates=parse_csv(args.candidates),
        main=args.main,
        reference=args.reference,
    )
    out = args.out or Path(args.aggregate_dir) / "selection_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_unique_pass and result["status"] != "unique_pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
