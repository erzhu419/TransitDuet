#!/usr/bin/env python3
"""Build the closest preserved TransitDuet-family baseline table.

The current FreqDuet paper longtrain matrix already contains a no-frequency
HIRO row (`nofreq`) evaluated with the same simulator, domains, seeds, and
training horizon as promoted FreqDuet main. This is not claimed to be the
unmodified original TransitDuet code. It is the closest preserved
TransitDuet-family control within the FreqDuet runner after disabling the
frequency decomposition path.

Outputs:

- transitduet_like_per_seed.csv
- transitduet_like_summary.csv
- transitduet_like_paired_deltas.csv
- transitduet_like_baseline_manifest.json
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


DOMAINS = ("terminal", "highnoise", "odshift", "rushshift")
BASELINE_METHOD = "transitduet_like_nofreq_hiro"
DEFAULT_METRICS = (
    "wait",
    "cv",
    "overshoot",
    "composite",
    "lower_action_mean",
    "lower_drift_penalty_mean",
    "lower_drift_cost_mean",
)


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF


def infer_domain(config: str) -> str:
    if "_gen_highnoise_" in config:
        return "highnoise"
    if "_gen_odshift_" in config:
        return "odshift"
    if "_gen_rushshift_" in config:
        return "rushshift"
    if "_terminal_" in config:
        return "terminal"
    return "unknown"


def infer_method(config: str) -> str:
    if config.endswith("_main_hiro"):
        return "main"
    if "_nofreq_" in config:
        return BASELINE_METHOD
    return "other"


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
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


def read_matrix(path: Path, metrics: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"config", "seed"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"{path} missing required columns: {missing}")

    keep_metrics = [metric for metric in metrics if metric in df.columns]
    if "composite" not in keep_metrics:
        raise SystemExit("source matrix must contain the composite metric")

    out = df.copy()
    out["domain"] = out["config"].astype(str).map(infer_domain)
    out["method"] = out["config"].astype(str).map(infer_method)
    out = out[out["domain"].isin(DOMAINS) & out["method"].isin(("main", BASELINE_METHOD))].copy()
    if out.empty:
        raise SystemExit("no main/nofreq rows found in source matrix")

    columns = [
        "domain",
        "method",
        "config",
        "seed",
        *keep_metrics,
    ]
    optional = [
        "episodes",
        "logs_dir",
        "source_config",
    ]
    columns.extend([name for name in optional if name in out.columns])
    return out[columns].copy()


def summarize(df: pd.DataFrame, metrics: list[str], n_boot: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (domain, method), group in df.groupby(["domain", "method"], sort=False):
        row: dict[str, object] = {
            "domain": domain,
            "method": method,
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in metrics:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            row[f"{metric}_mean"] = float(vals.mean()) if vals.size else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=0)) if vals.size else np.nan
            lo, hi = bootstrap_ci(
                vals,
                n_boot=n_boot,
                seed=stable_seed("summary", domain, method, metric),
            )
            row[f"{metric}_ci95_lo"] = lo
            row[f"{metric}_ci95_hi"] = hi
        rows.append(row)

    for method in ("main", BASELINE_METHOD):
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        row = {
            "domain": "overall",
            "method": method,
            "n_seeds": int(sub["seed"].nunique()),
        }
        for metric in metrics:
            seed_metric = (
                sub.groupby(["seed", "domain"], as_index=False)[metric]
                .mean()
                .groupby("seed", as_index=False)[metric]
                .mean()
            )
            vals = pd.to_numeric(seed_metric[metric], errors="coerce").dropna().to_numpy()
            row[f"{metric}_mean"] = float(vals.mean()) if vals.size else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=0)) if vals.size else np.nan
            lo, hi = bootstrap_ci(
                vals,
                n_boot=n_boot,
                seed=stable_seed("summary", "overall", method, metric),
            )
            row[f"{metric}_ci95_lo"] = lo
            row[f"{metric}_ci95_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(df: pd.DataFrame, metrics: list[str], n_boot: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain in (*DOMAINS, "overall"):
        if domain == "overall":
            domain_df = df.groupby(["seed", "method"], as_index=False)[metrics].mean()
        else:
            domain_df = df[df["domain"] == domain][["seed", "method", *metrics]].copy()

        for metric in metrics:
            pivot = domain_df.pivot_table(
                index="seed",
                columns="method",
                values=metric,
                aggfunc="mean",
            )
            if "main" not in pivot.columns or BASELINE_METHOD not in pivot.columns:
                continue
            pair = pivot[["main", BASELINE_METHOD]].dropna()
            if pair.empty:
                continue
            delta = pair["main"].astype(float) - pair[BASELINE_METHOD].astype(float)
            lo, hi = bootstrap_ci(
                delta.to_numpy(),
                n_boot=n_boot,
                # Match summarize_freqduet_paper_matrix.py for reproducible
                # promoted-matrix nofreq deltas under the renamed baseline.
                seed=stable_seed(domain, "nofreq", metric),
            )
            rows.append({
                "domain": domain,
                "metric": metric,
                "main": "main",
                "baseline": BASELINE_METHOD,
                "n_pairs": int(len(delta)),
                "main_mean": float(pair["main"].mean()),
                "baseline_mean": float(pair[BASELINE_METHOD].mean()),
                "delta_main_minus_transitduet_like": float(delta.mean()),
                "delta_ci95_lo": lo,
                "delta_ci95_hi": hi,
                "main_win_rate": float((delta < 0.0).mean()),
                "main_tie_rate": float((delta == 0.0).mean()),
            })
    return pd.DataFrame(rows)


def print_compact(deltas: pd.DataFrame) -> None:
    rows = deltas[deltas["metric"].eq("composite")].copy()
    if rows.empty:
        print("No composite paired deltas.")
        return
    print("=" * 120)
    print(
        f"{'domain':12s} {'n':>4s} {'main':>9s} {'td_like':>9s} "
        f"{'delta':>10s} {'ci95':>23s} {'win':>7s}"
    )
    print("-" * 120)
    order = {domain: idx for idx, domain in enumerate([*DOMAINS, "overall"])}
    rows["_rank"] = rows["domain"].map(order).fillna(99)
    for _, row in rows.sort_values("_rank").iterrows():
        print(
            f"{row['domain']:12s} {int(row['n_pairs']):4d} "
            f"{row['main_mean']:9.4f} {row['baseline_mean']:9.4f} "
            f"{row['delta_main_minus_transitduet_like']:10.4f} "
            f"[{row['delta_ci95_lo']:+.4f},{row['delta_ci95_hi']:+.4f}] "
            f"{row['main_win_rate']:7.3f}"
        )
    print("=" * 120)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-per-seed", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()

    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    source = Path(args.source_per_seed)
    out_dir = Path(args.out_dir)

    df = read_matrix(source, metrics)
    metrics = [metric for metric in metrics if metric in df.columns]
    summary = summarize(df, metrics, n_boot=args.n_boot)
    deltas = paired_deltas(df, metrics, n_boot=args.n_boot)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "transitduet_like_per_seed.csv", index=False)
    summary.to_csv(out_dir / "transitduet_like_summary.csv", index=False)
    deltas.to_csv(out_dir / "transitduet_like_paired_deltas.csv", index=False)
    with (out_dir / "transitduet_like_baseline_manifest.json").open("w") as f:
        json.dump({
            "source_per_seed": str(source),
            "baseline_method": BASELINE_METHOD,
            "baseline_boundary": (
                "Closest preserved TransitDuet-family control inside the "
                "FreqDuet runner with frequency decomposition disabled; not "
                "the unmodified original TransitDuet repository."
            ),
            "domains": list(DOMAINS),
            "metrics": metrics,
            "n_boot": int(args.n_boot),
            "n_rows": int(len(df)),
        }, f, indent=2)

    print_compact(deltas)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
