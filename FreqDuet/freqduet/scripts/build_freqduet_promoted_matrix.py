#!/usr/bin/env python3
"""Build a promoted-main paper matrix from a validated candidate run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def is_main_config(config: str) -> bool:
    return str(config).endswith("_main_hiro")


def promoted_name(config: str) -> str:
    name = str(config)
    return name.replace("_main_driftcost_hiro", "_main_hiro")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-per-seed", required=True)
    ap.add_argument("--candidate-per-seed", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--candidate-token", default="_main_driftcost_hiro")
    args = ap.parse_args()

    baseline = pd.read_csv(args.baseline_per_seed)
    candidate = pd.read_csv(args.candidate_per_seed)
    if "config" not in baseline.columns or "config" not in candidate.columns:
        raise SystemExit("Both inputs need a config column")

    old_main = baseline["config"].astype(str).map(is_main_config)
    cand_main = candidate["config"].astype(str).str.contains(args.candidate_token, regex=False)
    if not cand_main.all():
        bad = sorted(candidate.loc[~cand_main, "config"].astype(str).unique())
        raise SystemExit(f"Candidate input contains non-candidate configs: {bad}")

    promoted = candidate.copy()
    promoted["source_config"] = promoted["config"]
    promoted["config"] = promoted["config"].astype(str).map(promoted_name)
    combined = pd.concat([baseline.loc[~old_main].copy(), promoted], ignore_index=True, sort=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = out_dir / "freqduet_ablation_per_seed.csv"
    combined.to_csv(per_seed_path, index=False)

    metric_cols = [
        "wait",
        "cv",
        "overshoot",
        "composite",
        "lower_action_mean",
        "lower_drift_penalty_mean",
        "lower_drift_cost_mean",
        "upper_hf_power_ratio",
        "lower_lf_drift_ratio",
    ]
    rows = []
    for config, group in combined.groupby("config", sort=False):
        row = {"config": config, "n_seeds": int(group["seed"].nunique())}
        for col in metric_cols:
            if col not in group.columns:
                continue
            vals = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=0))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "freqduet_ablation_summary.csv", index=False)
    with (out_dir / "promoted_matrix_manifest.json").open("w") as f:
        json.dump({
            "baseline_per_seed": str(args.baseline_per_seed),
            "candidate_per_seed": str(args.candidate_per_seed),
            "old_main_rows_removed": int(old_main.sum()),
            "candidate_rows_promoted": int(len(promoted)),
            "output_per_seed": str(per_seed_path),
        }, f, indent=2)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
