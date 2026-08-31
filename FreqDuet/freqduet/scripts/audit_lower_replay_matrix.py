#!/usr/bin/env python3
"""Audit HF-attributed lower replay allocation across a trained matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from audit_lower_replay_allocation import (
    audit_lower_replay_allocation,
    resolve_config_path,
)


def _checkpoint_for(
    run_dir: Path,
    config_name: str,
    train_seed: int,
) -> Path:
    pattern = (
        f"logs_shards/*/{config_name}_seed{train_seed}/"
        "checkpoints/training_latest.pt"
    )
    matches = sorted(run_dir.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"expected one checkpoint for {config_name} seed {train_seed}, "
            f"found {len(matches)}"
        )
    return matches[0].resolve()


def _band_columns(
    row: dict[str, object],
    prefix: str,
    bands: dict[str, dict[str, object]],
) -> None:
    metrics = (
        "count",
        "action_mean_s",
        "target_mean_s",
        "action_minus_target_mean_s",
        "target_capture_ratio_mean",
        "signed_regularity_gain_mean",
        "zero_hold_regret_fraction",
    )
    for band_name, summary in bands.items():
        for metric in metrics:
            row[f"{prefix}_{band_name}_{metric}"] = summary.get(metric)


def _flat_row(config_name: str, train_seed: int, audit: dict) -> dict:
    overall = audit["valid_overall"]
    row = {
        "config": config_name,
        "train_seed": train_seed,
        "checkpoint": audit["checkpoint"],
        "checkpoint_episode": audit["checkpoint_episode"],
        "replay_transitions": audit["replay_transitions"],
        "valid_transitions": audit["valid_transitions"],
        "valid_action_mean_s": overall.get("action_mean_s"),
        "valid_target_mean_s": overall.get("target_mean_s"),
        "valid_action_minus_target_mean_s": overall.get(
            "action_minus_target_mean_s"
        ),
        "valid_target_capture_ratio_mean": overall.get(
            "target_capture_ratio_mean"
        ),
        "valid_signed_regularity_gain_mean": overall.get(
            "signed_regularity_gain_mean"
        ),
        "valid_zero_hold_regret_fraction": overall.get(
            "zero_hold_regret_fraction"
        ),
        "valid_local_hf_residual_norm_mean": overall.get(
            "local_hf_residual_norm_mean"
        ),
        "valid_local_hf_energy_norm_mean": overall.get(
            "local_hf_energy_norm_mean"
        ),
        "valid_hf_active_fraction": overall.get("hf_active_fraction"),
        "hf_residual_positive_q33": audit[
            "hf_residual_band_boundaries"
        ]["positive_q33"],
        "hf_residual_positive_q67": audit[
            "hf_residual_band_boundaries"
        ]["positive_q67"],
        "hf_energy_positive_q33": audit[
            "hf_energy_band_boundaries"
        ]["positive_q33"],
        "hf_energy_positive_q67": audit[
            "hf_energy_band_boundaries"
        ]["positive_q67"],
    }
    for name, value in audit["valid_correlations"].items():
        row[f"corr_{name}"] = value
    _band_columns(row, "hf_activity", audit["valid_by_hf_activity"])
    _band_columns(row, "hf_residual", audit["valid_by_hf_residual"])
    _band_columns(row, "hf_energy", audit["valid_by_hf_energy"])
    return row


def audit_lower_replay_matrix(
    run_dir: str | Path,
    configs: list[str],
    train_seeds: list[int],
) -> dict[str, object]:
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    results = []
    for config_name in configs:
        config_path = resolve_config_path(config_name)
        for train_seed in train_seeds:
            checkpoint = _checkpoint_for(run_dir, config_name, train_seed)
            results.append({
                "config": config_name,
                "train_seed": int(train_seed),
                "audit": audit_lower_replay_allocation(
                    checkpoint, config_path
                ),
            })
    return {
        "schema": "freqduet-lower-replay-matrix-audit-v1",
        "run_dir": str(run_dir),
        "configs": list(configs),
        "train_seeds": [int(seed) for seed in train_seeds],
        "expected_checkpoints": len(configs) * len(train_seeds),
        "audited_checkpoints": len(results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--train-seed", action="append", type=int, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    result = audit_lower_replay_matrix(
        args.run_dir, args.config, args.train_seed
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    rows = [
        _flat_row(entry["config"], entry["train_seed"], entry["audit"])
        for entry in result["results"]
    ]
    fieldnames = sorted({key for row in rows for key in row})
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        "Eval complete: FREQDUET_REPLAY_AUDIT_COMPLETE "
        f"checkpoints={result['audited_checkpoints']} "
        f"json={args.out_json} csv={args.out_csv}"
    )


if __name__ == "__main__":
    main()
