#!/usr/bin/env python3
"""Summarize the broad FreqDuet held-out generalization matrix.

The standard paper matrix summarizer is intentionally tied to the four canonical
domains. This script handles generated configs named
`F_freqduet_broad_<scenario>_<method>_hiro`, then writes scenario-level,
family-level, and overall paired deltas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zlib

import numpy as np
import pandas as pd


METHODS = (
    "main",
    "upperhist_fixedguard",
    "upperhist_safe",
    "upperhist_bestmix",
    "cfvalue_multicand",
    "cfvalue_noisegate",
    "cfvalue_domainmean",
    "upperres_planctx",
    "upperres_reliefguard",
    "upperres_selector",
    "uppervalue_hfgate",
    "termvalselector",
    "headwayplanner",
    "releaseadapt",
    "release20",
    "release15",
    "release10",
    "release5",
    "ctxselector60",
    "upperdisc7",
    "upperdisc5",
    "upperdisc4",
    "upperdisc3",
    "spline2dir_promreplan",
    "spacectx_disc9",
    "gapctx_disc9last",
    "disc9last",
    "spacectx",
    "disc9",
    "nofreq",
    "rawhistory",
    "allfreq",
    "nopromotion",
    "noleakage",
    "promenergy06",
    "promenergy07",
    "promenergy08",
    "histaux3",
    "histaux6",
    "histaux6eg05",
    "histaux6eg06",
    "histaux6eg06upper",
    "sumorl_rawhist_holdrl",
    "sumorl_holdrl",
    "snapshotriskdual_targetnonpos_domainmix",
)
DEFAULT_SCENARIOS = (
    "noise10",
    "noise20",
    "noise40",
    "od20",
    "od50",
    "rush_early",
    "rush_late",
    "rush_extreme",
)
DEFAULT_SEEDS = (
    7, 11, 17, 23, 31, 37, 42, 43, 53, 61,
    71, 83, 97, 109, 123, 127, 149, 456, 789, 2026,
)
DEFAULT_METRICS = (
    "wait",
    "cv",
    "overshoot",
    "composite",
    "lower_action_mean",
    "lower_drift_penalty_mean",
    "lower_drift_cost_mean",
    "upper_residual_selector_active_mean",
    "upper_residual_selector_adjust_mean",
    "freq_high_energy",
    "freq_promotion_active",
    "freq_promotion_ratio",
)


def parse_csv(value: str, cast=str) -> list:
    return [cast(x.strip()) for x in str(value).split(",") if x.strip()]


def scenario_family(scenario: str) -> str:
    if scenario.startswith("noise"):
        return "demand_noise"
    if scenario.startswith("od"):
        return "od_shift"
    if scenario.startswith("rush"):
        return "rush_shift"
    return "other"


def infer_broad_config(config: str) -> tuple[str, str]:
    prefix = "F_freqduet_broad_"
    suffix = "_hiro"
    if not config.startswith(prefix) or not config.endswith(suffix):
        return "unknown", "unknown"
    middle = config[len(prefix):-len(suffix)]
    for method in sorted(METHODS, key=len, reverse=True):
        token = f"_{method}"
        if middle.endswith(token):
            scenario = middle[:-len(token)]
            return scenario or "unknown", method
    return "unknown", "unknown"


def stable_seed(*parts) -> int:
    text = "::".join(str(part) for part in parts)
    return zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF


def bootstrap_ci(values, n_boot=5000, seed=0, alpha=0.05) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        value = float(arr[0])
        return value, value
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def completion_audit(df: pd.DataFrame, scenarios: list[str],
                     methods: list[str], seeds: list[int]) -> pd.DataFrame:
    observed = set(
        tuple(row)
        for row in df[["scenario", "method", "seed"]].itertuples(index=False, name=None)
    )
    rows = []
    for scenario in scenarios:
        for method in methods:
            missing = [
                seed for seed in seeds
                if (scenario, method, int(seed)) not in observed
            ]
            rows.append({
                "scenario": scenario,
                "family": scenario_family(scenario),
                "method": method,
                "expected_seeds": len(seeds),
                "observed_seeds": len(seeds) - len(missing),
                "missing_seeds": ",".join(str(x) for x in missing),
            })
    return pd.DataFrame(rows)


def summarize_group(df: pd.DataFrame, scope: str, scenario: str,
                    metrics: list[str]) -> list[dict]:
    rows = []
    for method, group in df.groupby("method", sort=False):
        if method == "unknown":
            continue
        row = {
            "scope": scope,
            "scenario": scenario,
            "family": scenario_family(scenario) if scope == "scenario" else scenario,
            "method": method,
            "n_seeds": int(group["seed"].nunique()),
            "n_scenarios": int(group["scenario"].nunique()),
        }
        for metric in metrics:
            vals = group[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals, ddof=0))
            ci_lo, ci_hi = bootstrap_ci(vals, seed=stable_seed(scope, scenario, method, metric))
            row[f"{metric}_ci95_lo"] = ci_lo
            row[f"{metric}_ci95_hi"] = ci_hi
        rows.append(row)
    return rows


def averaged_scope_rows(df: pd.DataFrame, scenarios: list[str],
                        metrics: list[str]) -> pd.DataFrame:
    sub = df[df["scenario"].isin(scenarios)].copy()
    if sub.empty:
        return sub
    grouped = sub.groupby(["seed", "method"], as_index=False)[metrics].mean()
    grouped["scenario"] = "scope_average"
    return grouped


def method_summary(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for scenario, group in df.groupby("scenario", sort=False):
        rows.extend(summarize_group(group, "scenario", scenario, metrics))
    for family, group in df.groupby("family", sort=False):
        avg = averaged_scope_rows(group, sorted(group["scenario"].unique()), metrics)
        avg["family"] = family
        rows.extend(summarize_group(avg, "family", family, metrics))
    avg = averaged_scope_rows(df, sorted(df["scenario"].unique()), metrics)
    avg["family"] = "overall"
    rows.extend(summarize_group(avg, "overall", "overall", metrics))
    return pd.DataFrame(rows)


def paired_for_scope(df: pd.DataFrame, scope: str, scenario: str,
                     metrics: list[str]) -> list[dict]:
    rows = []
    for metric in metrics:
        pivot = df.pivot_table(index="seed", columns="method", values=metric, aggfunc="mean")
        if "main" not in pivot.columns:
            continue
        for baseline in METHODS:
            if baseline == "main" or baseline not in pivot.columns:
                continue
            pair = pivot[["main", baseline]].dropna()
            if pair.empty:
                continue
            delta = pair["main"].astype(float) - pair[baseline].astype(float)
            ci_lo, ci_hi = bootstrap_ci(
                delta.to_numpy(),
                seed=stable_seed(scope, scenario, baseline, metric),
            )
            rows.append({
                "scope": scope,
                "scenario": scenario,
                "family": scenario_family(scenario) if scope == "scenario" else scenario,
                "baseline": baseline,
                "metric": metric,
                "n_pairs": int(len(delta)),
                "main_mean": float(pair["main"].mean()),
                "baseline_mean": float(pair[baseline].mean()),
                "delta_main_minus_baseline": float(delta.mean()),
                "delta_ci95_lo": ci_lo,
                "delta_ci95_hi": ci_hi,
                "main_win_rate": float((delta < 0.0).mean()),
                "main_tie_rate": float((delta == 0.0).mean()),
            })
    return rows


def paired_deltas(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for scenario, group in df.groupby("scenario", sort=False):
        rows.extend(paired_for_scope(group, "scenario", scenario, metrics))
    for family, group in df.groupby("family", sort=False):
        avg = averaged_scope_rows(group, sorted(group["scenario"].unique()), metrics)
        rows.extend(paired_for_scope(avg, "family", family, metrics))
    avg = averaged_scope_rows(df, sorted(df["scenario"].unique()), metrics)
    rows.extend(paired_for_scope(avg, "overall", "overall", metrics))
    return pd.DataFrame(rows)


def print_compact(deltas: pd.DataFrame) -> None:
    if deltas.empty:
        print("No paired deltas available.")
        return
    rows = deltas[deltas["metric"].eq("composite")].copy()
    print("=" * 112)
    print(f"{'scope':10s} {'scenario':14s} {'baseline':12s} {'n':>4s} {'delta':>10s} {'ci95':>23s} {'win':>7s}")
    print("-" * 112)
    for _, row in rows.iterrows():
        if row["scope"] == "scenario" and row["scenario"] not in DEFAULT_SCENARIOS:
            continue
        print(
            f"{row['scope']:10s} {row['scenario']:14s} {row['baseline']:12s} "
            f"{int(row['n_pairs']):4d} {row['delta_main_minus_baseline']:+10.4f} "
            f"[{row['delta_ci95_lo']:+.4f},{row['delta_ci95_hi']:+.4f}] "
            f"{row['main_win_rate']:7.3f}"
        )
    print("=" * 112)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-seed", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    args = parser.parse_args()

    per_seed = Path(args.per_seed)
    out_dir = Path(args.out_dir)
    scenarios = parse_csv(args.scenarios, str)
    methods = parse_csv(args.methods, str)
    seeds = parse_csv(args.seeds, int)

    df = pd.read_csv(per_seed)
    inferred = df["config"].astype(str).map(infer_broad_config)
    df["scenario"] = [x[0] for x in inferred]
    df["method"] = [x[1] for x in inferred]
    df["family"] = df["scenario"].map(scenario_family)
    df["seed"] = df["seed"].astype(int)

    unknown = sorted(df[df["scenario"].eq("unknown") | df["method"].eq("unknown")]["config"].unique())
    if unknown:
        print("Warning: ignored unknown config names:")
        for config in unknown:
            print(f"  {config}")
    df = df[~df["scenario"].eq("unknown") & ~df["method"].eq("unknown")].copy()

    metrics = [m for m in parse_csv(args.metrics, str) if m in df.columns]
    if "composite" not in metrics:
        raise SystemExit("composite metric is required for paired deltas")

    out_dir.mkdir(parents=True, exist_ok=True)
    audit = completion_audit(df, scenarios, methods, seeds)
    summary = method_summary(df, metrics)
    deltas = paired_deltas(df, metrics)

    audit.to_csv(out_dir / "broad_completion_audit.csv", index=False)
    summary.to_csv(out_dir / "broad_method_summary.csv", index=False)
    deltas.to_csv(out_dir / "broad_paired_deltas.csv", index=False)

    payload = {
        "source": str(per_seed),
        "scenarios": scenarios,
        "methods": methods,
        "seeds": seeds,
        "metrics": metrics,
        "n_rows": int(len(df)),
        "expected_rows": int(len(scenarios) * len(methods) * len(seeds)),
        "missing_rows": int(audit["expected_seeds"].sum() - audit["observed_seeds"].sum()),
    }
    with (out_dir / "broad_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print_compact(deltas)
    print(
        f"rows={payload['n_rows']} expected={payload['expected_rows']} "
        f"missing={payload['missing_rows']}"
    )
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
