#!/usr/bin/env python3
"""Compare a broad-generalization candidate against existing broad baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_freqduet_broad_generalization import (
    DEFAULT_METRICS,
    DEFAULT_SCENARIOS,
    METHODS,
    bootstrap_ci,
    infer_broad_config,
    parse_csv,
    scenario_family,
    stable_seed,
)


DEFAULT_BASELINES = ("main", "nofreq", "rawhistory", "allfreq", "nopromotion", "noleakage")


def prepare(path: Path, source: str, force_method: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    inferred = df["config"].astype(str).map(infer_broad_config)
    df["scenario"] = [x[0] for x in inferred]
    df["method"] = [x[1] for x in inferred]
    if force_method:
        known = df[df["scenario"] != "unknown"].copy()
        if force_method in set(known["method"].unique()):
            df = known[known["method"].eq(force_method)].copy()
        else:
            df.loc[df["scenario"] != "unknown", "method"] = force_method
    df["family"] = df["scenario"].map(scenario_family)
    df["seed"] = df["seed"].astype(int)
    df["source"] = source
    return df[(df["scenario"] != "unknown") & (df["method"] != "unknown")].copy()


def averaged_scope_rows(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    return df.groupby(["seed", "method"], as_index=False)[metrics].mean()


def paired_delta_rows(
    df: pd.DataFrame,
    candidate_method: str,
    baselines: list[str],
    metrics: list[str],
    scope: str,
    label: str,
    n_boot: int,
) -> list[dict]:
    rows = []
    for metric in metrics:
        pivot = df.pivot_table(index="seed", columns="method", values=metric, aggfunc="mean")
        if candidate_method not in pivot.columns:
            continue
        for baseline in baselines:
            if baseline == candidate_method or baseline not in pivot.columns:
                continue
            pair = pivot[[candidate_method, baseline]].dropna()
            if pair.empty:
                continue
            delta = pair[candidate_method].astype(float) - pair[baseline].astype(float)
            lo, hi = bootstrap_ci(
                delta.to_numpy(),
                n_boot=n_boot,
                seed=stable_seed("broad-candidate", scope, label, candidate_method, baseline, metric),
            )
            rows.append({
                "scope": scope,
                "scenario": label,
                "family": scenario_family(label) if scope == "scenario" else label,
                "candidate": candidate_method,
                "baseline": baseline,
                "metric": metric,
                "n_pairs": int(len(pair)),
                "candidate_mean": float(pair[candidate_method].mean()),
                "baseline_mean": float(pair[baseline].mean()),
                "delta_candidate_minus_baseline": float(delta.mean()),
                "delta_ci95_lo": lo,
                "delta_ci95_hi": hi,
                "candidate_win_rate": float((delta < 0.0).mean()),
                "candidate_tie_rate": float((delta == 0.0).mean()),
            })
    return rows


def method_summary(df: pd.DataFrame, metrics: list[str], n_boot: int) -> pd.DataFrame:
    rows = []
    for (scenario, method), group in df.groupby(["scenario", "method"], sort=False):
        row = {
            "scope": "scenario",
            "scenario": scenario,
            "family": scenario_family(scenario),
            "method": method,
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in metrics:
            vals = group[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=0))
            lo, hi = bootstrap_ci(
                vals, n_boot=n_boot,
                seed=stable_seed("broad-candidate-summary", scenario, method, metric))
            row[f"{metric}_ci95_lo"] = lo
            row[f"{metric}_ci95_hi"] = hi
        rows.append(row)
    for (family, method), group in df.groupby(["family", "method"], sort=False):
        avg = averaged_scope_rows(group, metrics)
        row = {
            "scope": "family",
            "scenario": family,
            "family": family,
            "method": method,
            "n_seeds": int(avg["seed"].nunique()),
        }
        for metric in metrics:
            vals = avg[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=0))
            lo, hi = bootstrap_ci(
                vals, n_boot=n_boot,
                seed=stable_seed("broad-candidate-summary", family, method, metric))
            row[f"{metric}_ci95_lo"] = lo
            row[f"{metric}_ci95_hi"] = hi
        rows.append(row)
    for method, group in df.groupby("method", sort=False):
        avg = averaged_scope_rows(group, metrics)
        row = {
            "scope": "overall",
            "scenario": "overall",
            "family": "overall",
            "method": method,
            "n_seeds": int(avg["seed"].nunique()),
        }
        for metric in metrics:
            vals = avg[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=0))
            lo, hi = bootstrap_ci(
                vals, n_boot=n_boot,
                seed=stable_seed("broad-candidate-summary", "overall", method, metric))
            row[f"{metric}_ci95_lo"] = lo
            row[f"{metric}_ci95_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(
    df: pd.DataFrame,
    candidate_method: str,
    baselines: list[str],
    metrics: list[str],
    n_boot: int,
) -> pd.DataFrame:
    rows = []
    for scenario, group in df.groupby("scenario", sort=False):
        rows.extend(paired_delta_rows(
            group, candidate_method, baselines, metrics,
            "scenario", scenario, n_boot))
    for family, group in df.groupby("family", sort=False):
        avg = averaged_scope_rows(group, metrics)
        rows.extend(paired_delta_rows(
            avg, candidate_method, baselines, metrics,
            "family", family, n_boot))
    avg = averaged_scope_rows(df, metrics)
    rows.extend(paired_delta_rows(
        avg, candidate_method, baselines, metrics,
        "overall", "overall", n_boot))
    return pd.DataFrame(rows)


def print_compact(deltas: pd.DataFrame) -> None:
    rows = deltas[deltas["metric"].eq("composite")].copy()
    if rows.empty:
        print("No composite paired deltas available.")
        return
    print("=" * 118)
    print(f"{'scope':10s} {'scenario':14s} {'baseline':12s} {'n':>4s} {'cand':>9s} {'base':>9s} {'delta':>10s} {'ci95':>23s} {'win':>7s}")
    print("-" * 118)
    for _, row in rows.iterrows():
        print(
            f"{row['scope']:10s} {row['scenario']:14s} {row['baseline']:12s} "
            f"{int(row['n_pairs']):4d} {row['candidate_mean']:9.4f} "
            f"{row['baseline_mean']:9.4f} {row['delta_candidate_minus_baseline']:+10.4f} "
            f"[{row['delta_ci95_lo']:+.4f},{row['delta_ci95_hi']:+.4f}] "
            f"{row['candidate_win_rate']:7.3f}"
        )
    print("=" * 118)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-per-seed", required=True)
    parser.add_argument("--candidate-per-seed", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--candidate-method", required=True)
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES))
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()

    baselines = parse_csv(args.baselines, str)
    scenarios = set(parse_csv(args.scenarios, str))
    metrics = parse_csv(args.metrics, str)

    baseline = prepare(Path(args.baseline_per_seed), "baseline")
    candidate = prepare(
        Path(args.candidate_per_seed), "candidate",
        force_method=args.candidate_method)
    df = pd.concat([baseline, candidate], ignore_index=True, sort=False)
    df = df[df["scenario"].isin(scenarios)].copy()
    metrics = [metric for metric in metrics if metric in df.columns]
    if "composite" not in metrics:
        raise SystemExit("composite metric is required")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = method_summary(df, metrics, args.n_boot)
    deltas = paired_deltas(df, args.candidate_method, baselines, metrics, args.n_boot)
    summary.to_csv(out_dir / "broad_candidate_method_summary.csv", index=False)
    deltas.to_csv(out_dir / "broad_candidate_paired_deltas.csv", index=False)
    with (out_dir / "broad_candidate_comparison.json").open("w", encoding="utf-8") as f:
        json.dump({
            "baseline_per_seed": args.baseline_per_seed,
            "candidate_per_seed": args.candidate_per_seed,
            "candidate_method": args.candidate_method,
            "baselines": baselines,
            "metrics": metrics,
            "n_rows": int(len(df)),
        }, f, indent=2)
    print_compact(deltas)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
