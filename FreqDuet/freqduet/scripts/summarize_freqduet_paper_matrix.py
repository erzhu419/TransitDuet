#!/usr/bin/env python3
"""Summarize FreqDuet paper matrices with paired seed deltas.

Input is the `freqduet_ablation_per_seed.csv` written by
`run_freqduet_ablation.py`. The script infers the paper domain/method from
FreqDuet config names, then writes:

- paper_matrix_method_summary.csv
- paper_matrix_paired_deltas.csv
- paper_matrix_summary.json
"""

import argparse
import json
from pathlib import Path
import zlib

import numpy as np
import pandas as pd


DOMAINS = ("terminal", "highnoise", "odshift", "rushshift")
METHODS = ("main", "nofreq", "rawhistory", "allfreq", "nopromotion", "noleakage")
DEFAULT_METRICS = ("wait", "cv", "overshoot", "composite")


def infer_domain(config):
    if "_gen_highnoise_" in config:
        return "highnoise"
    if "_gen_odshift_" in config:
        return "odshift"
    if "_gen_rushshift_" in config:
        return "rushshift"
    if "_terminal_" in config:
        return "terminal"
    return "unknown"


def infer_method(config):
    if config.endswith("_main_hiro"):
        return "main"
    for method in METHODS:
        if method != "main" and f"_{method}_" in config:
            return method
    return "unknown"


def bootstrap_ci(values, n_boot=5000, seed=0, alpha=0.05):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def stable_seed(*parts):
    text = "::".join(str(part) for part in parts)
    return zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF


def summarize_methods(df, metrics):
    rows = []
    grouped = df.groupby(["domain", "method"], sort=False)
    for (domain, method), group in grouped:
        if domain == "unknown" or method == "unknown":
            continue
        row = {"domain": domain, "method": method, "n_seeds": int(group["seed"].nunique())}
        for metric in metrics:
            vals = group[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals, ddof=0))
            ci_lo, ci_hi = bootstrap_ci(vals, seed=stable_seed(domain, method, metric))
            row[f"{metric}_ci95_lo"] = ci_lo
            row[f"{metric}_ci95_hi"] = ci_hi
        rows.append(row)

    # Overall rows average each seed/method across all available domains. This
    # prevents domains with extra rows from receiving extra weight.
    for method in METHODS:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="seed", columns="domain", values=list(metrics), aggfunc="mean")
        seed_rows = []
        for seed in pivot.index:
            item = {"seed": seed}
            for metric in metrics:
                vals = [
                    pivot.loc[seed, (metric, domain)]
                    for domain in DOMAINS
                    if (metric, domain) in pivot.columns and np.isfinite(pivot.loc[seed, (metric, domain)])
                ]
                if vals:
                    item[metric] = float(np.mean(vals))
            seed_rows.append(item)
        seed_df = pd.DataFrame(seed_rows)
        if seed_df.empty:
            continue
        row = {"domain": "overall", "method": method, "n_seeds": int(seed_df["seed"].nunique())}
        for metric in metrics:
            vals = seed_df[metric].dropna().astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals, ddof=0))
            ci_lo, ci_hi = bootstrap_ci(vals, seed=stable_seed("overall", method, metric))
            row[f"{metric}_ci95_lo"] = ci_lo
            row[f"{metric}_ci95_hi"] = ci_hi
        rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(df, metric):
    rows = []
    domains = list(DOMAINS) + ["overall"]
    for domain in domains:
        if domain == "overall":
            domain_df = (
                df.groupby(["seed", "method"], as_index=False)[metric]
                .mean()
            )
        else:
            domain_df = df[df["domain"] == domain][["seed", "method", metric]]
        pivot = domain_df.pivot_table(index="seed", columns="method", values=metric, aggfunc="mean")
        if "main" not in pivot.columns:
            continue
        for baseline in METHODS:
            if baseline == "main" or baseline not in pivot.columns:
                continue
            pair = pivot[["main", baseline]].dropna()
            if pair.empty:
                continue
            delta = pair["main"].astype(float) - pair[baseline].astype(float)
            ci_lo, ci_hi = bootstrap_ci(delta.to_numpy(), seed=stable_seed(domain, baseline, metric))
            rows.append({
                "domain": domain,
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
    return pd.DataFrame(rows)


def print_compact(deltas):
    if deltas.empty:
        print("No paired deltas available.")
        return
    print("=" * 110)
    print(f"{'domain':12s} {'baseline':12s} {'n':>4s} {'main':>9s} {'base':>9s} {'delta':>10s} {'ci95':>23s} {'win':>7s}")
    print("-" * 110)
    for _, row in deltas.iterrows():
        if row["metric"] != "composite":
            continue
        print(
            f"{row['domain']:12s} {row['baseline']:12s} {int(row['n_pairs']):4d} "
            f"{row['main_mean']:9.4f} {row['baseline_mean']:9.4f} "
            f"{row['delta_main_minus_baseline']:10.4f} "
            f"[{row['delta_ci95_lo']:+.4f},{row['delta_ci95_hi']:+.4f}] "
            f"{row['main_win_rate']:7.3f}"
        )
    print("=" * 110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-seed", required=True, help="freqduet_ablation_per_seed.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--paired-metric", default="composite")
    args = ap.parse_args()

    per_seed = Path(args.per_seed)
    out_dir = Path(args.out_dir)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    df = pd.read_csv(per_seed)
    df["domain"] = df["config"].map(infer_domain)
    df["method"] = df["config"].map(infer_method)
    keep_metrics = [m for m in metrics if m in df.columns]
    if args.paired_metric not in keep_metrics:
        raise SystemExit(f"paired metric {args.paired_metric!r} not in {keep_metrics}")

    unknown = df[(df["domain"] == "unknown") | (df["method"] == "unknown")]["config"].unique()
    if len(unknown):
        print("Warning: ignored unknown config names:")
        for name in sorted(unknown):
            print(f"  {name}")
    df = df[(df["domain"] != "unknown") & (df["method"] != "unknown")].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    method_summary = summarize_methods(df, keep_metrics)
    deltas = paired_deltas(df, args.paired_metric)
    method_summary.to_csv(out_dir / "paper_matrix_method_summary.csv", index=False)
    deltas.to_csv(out_dir / "paper_matrix_paired_deltas.csv", index=False)

    payload = {
        "source": str(per_seed),
        "metrics": keep_metrics,
        "paired_metric": args.paired_metric,
        "domains": list(DOMAINS),
        "methods": list(METHODS),
        "n_rows": int(len(df)),
    }
    with (out_dir / "paper_matrix_summary.json").open("w") as f:
        json.dump(payload, f, indent=2)

    print_compact(deltas)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
