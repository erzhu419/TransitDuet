#!/usr/bin/env python3
"""Pair frozen learned policies with protocol-v2 external baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_freqduet_protocol_v2_matrix import (
    METRIC_DIRECTIONS,
    PROTOCOL_VERSION,
    hierarchical_bootstrap,
    paired_sign_flip_p,
)


COMPARISON_METRICS = [
    "service_cost",
    "avg_wait_observed_min",
    "restricted_wait_horizon_min",
    "passenger_unserved_rate",
    "headway_cv",
    "fleet_overshoot",
    "trip_launch_rate",
    "trip_completion_rate",
]


def compare(
    learned_path: Path,
    external_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    learned = pd.read_csv(learned_path)
    external = pd.read_csv(external_path)
    required_learned = {
        "protocol_version", "config", "train_seed", "eval_seed",
        "scenario_tape_id", *COMPARISON_METRICS,
    }
    required_external = {
        "protocol_version", "variant", "eval_seed", "scenario_tape_id",
        *COMPARISON_METRICS,
    }
    for label, frame, required in [
        ("learned", learned, required_learned),
        ("external", external, required_external),
    ]:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} input is missing columns {missing}")
        if set(frame["protocol_version"].astype(str)) != {PROTOCOL_VERSION}:
            raise ValueError(f"{label} protocol version mismatch")

    rows = []
    paired_frames = []
    for config, candidate in learned.groupby("config", sort=False):
        for variant, reference in external.groupby("variant", sort=False):
            if reference["eval_seed"].astype(int).duplicated().any():
                raise ValueError(
                    f"external {variant} has duplicate evaluation seeds")
            merged = candidate.merge(
                reference,
                on="eval_seed",
                suffixes=("_candidate", "_reference"),
                validate="many_to_one",
            )
            expected = len(candidate)
            if len(merged) != expected:
                raise ValueError(
                    f"{config} vs {variant}: paired {len(merged)} of "
                    f"{expected} learned rollouts")
            if not merged["scenario_tape_id_candidate"].astype(str).eq(
                    merged["scenario_tape_id_reference"].astype(str)).all():
                raise ValueError(
                    f"{config} vs {variant}: scenario tape mismatch")
            paired = merged[["train_seed", "eval_seed"]].copy()
            paired["config"] = config
            paired["reference_variant"] = variant
            row = {
                "config": config,
                "reference_variant": variant,
                "n_train_seeds": int(candidate["train_seed"].nunique()),
                "n_eval_seeds": int(candidate["eval_seed"].nunique()),
                "n_pairs": int(len(merged)),
            }
            for metric in COMPARISON_METRICS:
                delta = (
                    pd.to_numeric(
                        merged[f"{metric}_candidate"], errors="raise")
                    - pd.to_numeric(
                        merged[f"{metric}_reference"], errors="raise")
                )
                name = f"delta_{metric}"
                paired[name] = delta.to_numpy(dtype=float)
                bootstrap = hierarchical_bootstrap(paired, name)
                lo, hi = np.percentile(bootstrap, [2.5, 97.5])
                direction = METRIC_DIRECTIONS.get(metric, "descriptive")
                probability_better = (
                    float(np.mean(bootstrap < 0.0)) if direction == "min"
                    else float(np.mean(bootstrap > 0.0)) if direction == "max"
                    else float("nan")
                )
                train_deltas = paired.groupby(
                    "train_seed")[name].mean().to_numpy(dtype=float)
                train_std = float(train_deltas.std(ddof=1)) \
                    if train_deltas.size > 1 else 0.0
                orientation = -1.0 if direction == "min" else 1.0
                row.update({
                    f"{name}_mean": float(delta.mean()),
                    f"{name}_ci_low": float(lo),
                    f"{name}_ci_high": float(hi),
                    f"{name}_prob_candidate_better": probability_better,
                    f"{name}_paired_effect_dz": (
                        orientation * float(train_deltas.mean()) / train_std
                        if train_std > 0.0 else float("nan")),
                    f"{name}_signflip_p": paired_sign_flip_p(train_deltas),
                })
            rows.append(row)
            paired_frames.append(paired)

    output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "external_paired_summary.csv", index=False)
    pd.concat(paired_frames, ignore_index=True).to_csv(
        output_dir / "external_paired_rollouts.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-per-eval", required=True)
    parser.add_argument("--external-evaluation", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    result = compare(
        Path(args.learned_per_eval),
        Path(args.external_evaluation),
        Path(args.out_dir),
    )
    columns = [
        "config", "reference_variant", "n_pairs",
        "delta_service_cost_mean", "delta_service_cost_ci_low",
        "delta_service_cost_ci_high",
        "delta_service_cost_prob_candidate_better",
        "delta_service_cost_signflip_p",
    ]
    print(result[columns].to_string(index=False))


if __name__ == "__main__":
    main()
