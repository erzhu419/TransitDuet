#!/usr/bin/env python3
"""Paired learned-policy comparisons against frozen external baselines."""

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

from scripts.run_freqduet_protocol_v2_matrix import (
    METRIC_DIRECTIONS,
    hierarchical_bootstrap,
    holm_adjusted_pvalues,
    paired_sign_flip_p,
)


METRICS = [
    "service_cost_restricted",
    "service_cost_observed",
    "restricted_wait_horizon_min",
    "avg_wait_observed_min",
    "restricted_in_vehicle_horizon_min",
    "avg_in_vehicle_observed_min",
    "restricted_total_journey_horizon_min",
    "avg_total_journey_observed_min",
    "holding_vehicle_seconds",
    "holding_passenger_seconds",
    "fleet_denied_dispatch_events",
    "fleet_denied_trips",
    "fleet_readiness_delay_mean_s",
    "fleet_readiness_delay_max_s",
    "passenger_unserved_rate",
    "headway_cv",
    "fleet_overshoot",
    "trip_launch_rate",
    "trip_completion_rate",
]


def _require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")


def _validate_tapes(
    learned: pd.DataFrame,
    baseline: pd.DataFrame,
    method: str,
) -> None:
    learned_tapes = learned.groupby("eval_seed")["scenario_tape_id"].nunique()
    if not learned_tapes.eq(1).all():
        raise ValueError("learned policies disagree on scenario tape IDs")
    baseline_tapes = baseline.groupby("eval_seed")["scenario_tape_id"].nunique()
    if not baseline_tapes.eq(1).all():
        raise ValueError(f"{method}: duplicate baseline tape IDs")
    learned_map = learned.groupby("eval_seed")["scenario_tape_id"].first()
    baseline_map = baseline.groupby("eval_seed")["scenario_tape_id"].first()
    if set(learned_map.index) != set(baseline_map.index):
        raise ValueError(f"{method}: evaluation seed set does not match learned")
    mismatch = learned_map.astype(str) != baseline_map.astype(str)
    if mismatch.any():
        raise ValueError(
            f"{method}: common-random-number tape mismatch at "
            f"{learned_map.index[mismatch].tolist()}")


def _load_and_validate_manifests(
    learned_path: Path,
    baseline_path: Path,
    learned_manifest_path: Path | None,
    baseline_manifest_path: Path | None,
) -> dict[str, object]:
    learned_manifest_path = (
        learned_manifest_path or learned_path.with_name("matrix_manifest.json"))
    baseline_manifest_path = (
        baseline_manifest_path
        or baseline_path.with_name("external_baselines_summary.json"))
    if not learned_manifest_path.exists():
        raise ValueError(f"missing learned matrix manifest {learned_manifest_path}")
    if not baseline_manifest_path.exists():
        raise ValueError(
            f"missing external baseline manifest {baseline_manifest_path}")
    learned = json.loads(learned_manifest_path.read_text())
    baseline = json.loads(baseline_manifest_path.read_text())
    required_learned = {
        "protocol_version": "freqduet-eval-v4",
        "strict_complete": True,
        "common_random_numbers_verified": True,
        "run_manifests_verified": True,
    }
    for key, expected in required_learned.items():
        if learned.get(key) != expected:
            raise ValueError(
                f"learned matrix manifest {key}={learned.get(key)!r}, "
                f"expected {expected!r}")
    required_baseline = {
        "direct_scenario_seeds": True,
        "run_manifests_verified": True,
    }
    for key, expected in required_baseline.items():
        if baseline.get(key) != expected:
            raise ValueError(
                f"external manifest {key}={baseline.get(key)!r}, "
                f"expected {expected!r}")
    learned_source = learned.get("run_source_fingerprint") or {}
    learned_sha = str(learned_source.get("sha256", ""))
    baseline_sha = str(baseline.get("core_source_sha256", ""))
    evaluator_sha = str(baseline.get("evaluator_source_sha256", ""))
    if len(learned_sha) != 64 or len(baseline_sha) != 64:
        raise ValueError("comparison manifests contain invalid core source hashes")
    if learned_sha != baseline_sha:
        raise ValueError(
            "learned and external baseline core source fingerprints differ")
    if len(evaluator_sha) != 64:
        raise ValueError("external evaluator fingerprint is missing or invalid")
    return {
        "learned_manifest": str(learned_manifest_path),
        "baseline_manifest": str(baseline_manifest_path),
        "core_source_sha256": learned_sha,
        "external_evaluator_sha256": evaluator_sha,
    }


def compare(
    learned_path: Path,
    baseline_path: Path,
    out_dir: Path,
    learned_config: str,
    baseline_config: str | None = None,
    methods: list[str] | None = None,
    learned_manifest_path: Path | None = None,
    baseline_manifest_path: Path | None = None,
) -> None:
    provenance = _load_and_validate_manifests(
        learned_path,
        baseline_path,
        learned_manifest_path,
        baseline_manifest_path,
    )
    learned_all = pd.read_csv(learned_path)
    baseline_all = pd.read_csv(baseline_path)
    required_common = {"config", "eval_seed", "scenario_tape_id", *METRICS}
    _require_columns(
        learned_all, required_common | {"train_seed"}, learned_path)
    _require_columns(
        baseline_all, required_common | {"method"}, baseline_path)

    learned = learned_all[
        learned_all["config"].astype(str).eq(str(learned_config))
    ].copy()
    baseline_name = str(baseline_config or learned_config)
    baselines = baseline_all[
        baseline_all["config"].astype(str).eq(baseline_name)
    ].copy()
    if learned.empty:
        raise ValueError(f"no learned rows for {learned_config}")
    if baselines.empty:
        raise ValueError(f"no baseline rows for {baseline_name}")
    if learned.duplicated(["train_seed", "eval_seed"]).any():
        raise ValueError("learned matrix has duplicate train/eval pairs")
    if set(learned["protocol_version"].astype(str)) != {"freqduet-eval-v4"}:
        raise ValueError("learned input is not protocol v4")
    if ("protocol_version" not in baselines
            or set(baselines["protocol_version"].astype(str))
            != {"freqduet-eval-v4"}):
        raise ValueError("external baseline input is not protocol v4")

    selected_methods = methods or sorted(baselines["method"].astype(str).unique())
    pair_frames = []
    summary_rows = []
    for method in selected_methods:
        baseline = baselines[baselines["method"].astype(str).eq(method)].copy()
        if baseline.empty:
            raise ValueError(f"missing external baseline method {method}")
        if baseline.duplicated(["eval_seed"]).any():
            raise ValueError(f"{method}: duplicate rows per eval seed")
        _validate_tapes(learned, baseline, method)
        merged = learned.merge(
            baseline,
            on="eval_seed",
            how="inner",
            suffixes=("_learned", "_baseline"),
            validate="many_to_one",
        )
        expected = len(learned)
        if len(merged) != expected:
            raise ValueError(
                f"{method}: paired rows {len(merged)}, expected {expected}")
        pair_frame = merged[["train_seed", "eval_seed"]].copy()
        pair_frame["learned_config"] = learned_config
        pair_frame["baseline_config"] = baseline_name
        pair_frame["baseline_method"] = method
        row = {
            "learned_config": learned_config,
            "baseline_config": baseline_name,
            "baseline_method": method,
            "n_train_seeds": int(learned["train_seed"].nunique()),
            "n_eval_seeds": int(learned["eval_seed"].nunique()),
            "n_pairs": int(len(merged)),
        }
        for metric in METRICS:
            learned_values = pd.to_numeric(
                merged[f"{metric}_learned"], errors="raise")
            baseline_values = pd.to_numeric(
                merged[f"{metric}_baseline"], errors="raise")
            delta_name = f"delta_{metric}"
            pair_frame[delta_name] = learned_values - baseline_values
            bootstrap = hierarchical_bootstrap(pair_frame, delta_name)
            lo, hi = np.percentile(bootstrap, [2.5, 97.5])
            train_delta = pair_frame.groupby(
                "train_seed")[delta_name].mean().to_numpy(dtype=float)
            train_std = (
                float(train_delta.std(ddof=1)) if train_delta.size > 1 else 0.0
            )
            direction = METRIC_DIRECTIONS.get(metric, "descriptive")
            if direction == "min":
                probability_better = float(np.mean(bootstrap < 0.0))
                orientation = -1.0
            elif direction == "max":
                probability_better = float(np.mean(bootstrap > 0.0))
                orientation = 1.0
            else:
                probability_better = float("nan")
                orientation = 1.0
            row[f"{metric}_learned_mean"] = float(learned_values.mean())
            row[f"{metric}_baseline_mean"] = float(baseline_values.mean())
            row[f"{delta_name}_mean"] = float(pair_frame[delta_name].mean())
            row[f"{delta_name}_ci_low"] = float(lo)
            row[f"{delta_name}_ci_high"] = float(hi)
            row[f"{delta_name}_prob_learned_better"] = probability_better
            row[f"{delta_name}_paired_effect_dz"] = (
                orientation * float(train_delta.mean()) / train_std
                if train_std > 0.0 else float("nan")
            )
            row[f"{delta_name}_signflip_p"] = paired_sign_flip_p(train_delta)
        pair_frames.append(pair_frame)
        summary_rows.append(row)

    for metric in METRICS:
        key = f"delta_{metric}_signflip_p"
        adjusted = holm_adjusted_pvalues([
            float(row.get(key, float("nan"))) for row in summary_rows
        ])
        for row, value in zip(summary_rows, adjusted):
            row[f"{key}_holm"] = value

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(pair_frames, ignore_index=True).to_csv(
        out_dir / "learned_vs_external_per_pair.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(
        out_dir / "learned_vs_external_summary.csv", index=False)
    (out_dir / "learned_vs_external_manifest.json").write_text(json.dumps({
        "protocol_version": "freqduet-eval-v4",
        "learned_input": str(learned_path),
        "baseline_input": str(baseline_path),
        "learned_config": learned_config,
        "baseline_config": baseline_name,
        "baseline_methods": selected_methods,
        "metrics": METRICS,
        "delta_definition": "learned minus external baseline",
        "uncertainty": (
            "paired crossed bootstrap over train and eval seed; each draw "
            "shares one evaluation-seed resample across training seeds"
        ),
        "paired_test": "two-sided sign-flip over train-seed mean deltas",
        "multiple_testing": "Holm correction across external methods per metric",
        "common_random_numbers_verified": True,
        "source_provenance": provenance,
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned", required=True)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--learned-config", required=True)
    parser.add_argument("--baseline-config", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--learned-manifest", default=None)
    parser.add_argument("--baseline-manifest", default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    methods = (
        [value.strip() for value in args.methods.split(",") if value.strip()]
        if args.methods else None
    )
    compare(
        learned_path=Path(args.learned),
        baseline_path=Path(args.baselines),
        out_dir=Path(args.out_dir),
        learned_config=args.learned_config,
        baseline_config=args.baseline_config,
        methods=methods,
        learned_manifest_path=(
            Path(args.learned_manifest) if args.learned_manifest else None),
        baseline_manifest_path=(
            Path(args.baseline_manifest) if args.baseline_manifest else None),
    )


if __name__ == "__main__":
    main()
