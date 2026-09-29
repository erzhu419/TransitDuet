#!/usr/bin/env python3
"""Correlation diagnostics for rolling timetable variation.

Reviewer-facing question: whether dynamic planned timetable/headway variation is
associated with realized headway regularity. This script computes per-seed and
pooled Pearson/Spearman correlations between within-episode planned-headway
standard deviations and two realized regularity measures, then exports a
compact scatter figure for the final rolling-timetable configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METHODS = [
    "H_timetable_v5_stable_300",
    "H_timetable_v4",
    "H_timetable_continuous",
    "H_timetable_no_context",
    "H_fixed_timetable_300",
    "H_fixed_timetable",
    "H_hiro",
]

DEFAULT_SEEDS = ["42", "123", "456", "789", "1001", "1002", "1003", "1004", "1005", "1006"]


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    ok = x.notna() & y.notna()
    x = x[ok]
    y = y[ok]
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(x.corr(y, method=method))


def _method_seed(name: str) -> tuple[str, str] | None:
    if "_seed" not in name:
        return None
    method, seed = name.rsplit("_seed", 1)
    if not seed:
        return None
    return method, seed


def load_runs(log_dirs: list[Path], methods: set[str], seeds: set[str] | None, last_k: int) -> pd.DataFrame:
    rows = []
    seen = set()
    for root in log_dirs:
        if not root.exists():
            continue
        for run_dir in sorted(root.glob("*_seed*")):
            parsed = _method_seed(run_dir.name)
            if parsed is None:
                continue
            method, seed = parsed
            if method not in methods:
                continue
            if seeds is not None and seed not in seeds:
                continue
            key = (method, seed)
            if key in seen:
                continue
            csv = run_dir / "diagnostics.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            if "ep" in df.columns:
                df = df[df["ep"] < 9000]
            if len(df) < max(3, last_k):
                continue
            df = df.iloc[-last_k:].copy()
            df["method"] = method
            df["seed"] = seed
            rows.append(df)
            seen.add(key)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = [
        ("planned_dispatch_headway_std", "headway_cv"),
        ("planned_dispatch_headway_std", "actual_dispatch_headway_cv"),
        ("planned_shift_std", "headway_cv"),
        ("planned_shift_std", "actual_dispatch_headway_cv"),
        ("upper_delta_std", "headway_cv"),
    ]
    per_seed_rows = []
    for (method, seed), g in df.groupby(["method", "seed"], sort=True):
        row = {"method": method, "seed": seed, "n_ep": len(g)}
        for x, y in pairs:
            if x not in g.columns or y not in g.columns:
                continue
            prefix = f"{x}__{y}"
            row[f"{prefix}_pearson"] = _safe_corr(g[x], g[y], "pearson")
            row[f"{prefix}_spearman"] = _safe_corr(g[x], g[y], "spearman")
        per_seed_rows.append(row)
    per_seed = pd.DataFrame(per_seed_rows)

    pooled_rows = []
    for method, g in df.groupby("method", sort=True):
        row = {"method": method, "n_seed": g["seed"].nunique(), "n_ep": len(g)}
        for x, y in pairs:
            if x not in g.columns or y not in g.columns:
                continue
            prefix = f"{x}__{y}"
            row[f"{prefix}_pearson"] = _safe_corr(g[x], g[y], "pearson")
            row[f"{prefix}_spearman"] = _safe_corr(g[x], g[y], "spearman")
            row[f"{x}_mean"] = float(pd.to_numeric(g[x], errors="coerce").mean())
            row[f"{y}_mean"] = float(pd.to_numeric(g[y], errors="coerce").mean())
        pooled_rows.append(row)
    pooled = pd.DataFrame(pooled_rows)
    return per_seed, pooled


def make_figure(df: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    x_col = "planned_dispatch_headway_std"
    method = "H_timetable_v5_stable_300"
    plot_df = df[df["method"] == method].copy()
    if plot_df.empty:
        plot_df = df.copy()
    y_cols = [
        ("headway_cv", "Station-level headway CV"),
        ("actual_dispatch_headway_cv", "Actual dispatch-headway CV"),
    ]
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    if plot_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.45), sharex=True)
    for ax, (y_col, y_label) in zip(axes, y_cols):
        panel = plot_df[[x_col, y_col]].copy()
        panel[y_col] = pd.to_numeric(panel[y_col], errors="coerce")
        panel = panel.dropna()
        panel = panel[(panel[x_col] > 0) & (panel[y_col] >= 0)]
        if panel.empty:
            continue

        x = panel[x_col].to_numpy(dtype=float)
        y = panel[y_col].to_numpy(dtype=float)
        ax.scatter(x, y, s=17, alpha=0.62, color="#1f5a99", edgecolors="none")
        if len(x) >= 3 and np.unique(x).size >= 2:
            coef = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 100)
            ax.plot(xx, coef[0] * xx + coef[1], color="black", linewidth=1.0)

        pearson = _safe_corr(panel[x_col], panel[y_col], "pearson")
        spearman = _safe_corr(panel[x_col], panel[y_col], "spearman")
        ax.text(0.04, 0.95, f"r = {pearson:.2f}\nrho = {spearman:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8)
        ax.set_xlabel("Planned headway std. (s)")
        ax.set_ylabel(y_label)
        ax.grid(True, color="0.9", linewidth=0.6)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", action="append", default=None,
                    help="Run directory root. Can be supplied multiple times.")
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--seeds", default=",".join(DEFAULT_SEEDS),
                    help="Comma-separated seed list; use 'all' to disable filtering.")
    ap.add_argument("--last-k", type=int, default=30)
    ap.add_argument("--out-dir", default="results_remote/timetable_correlation")
    ap.add_argument("--figure", default="../paper/figures/timetable_variance_correlation.pdf")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parents[1]
    logs = [Path(p) for p in (args.logs or ["logs", "logs_remote"])]
    logs = [p if p.is_absolute() else script_dir / p for p in logs]
    methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    seeds = None if args.seeds.strip().lower() == "all" else {
        s.strip() for s in args.seeds.split(",") if s.strip()
    }
    df = load_runs(logs, methods, seeds, args.last_k)
    if df.empty:
        raise SystemExit("No matching diagnostics found.")

    per_seed, pooled = summarize(df)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = script_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed.round(4).to_csv(out_dir / "per_seed.csv", index=False)
    pooled.round(4).to_csv(out_dir / "pooled.csv", index=False)

    fig_path = Path(args.figure)
    if not fig_path.is_absolute():
        fig_path = script_dir / fig_path
    make_figure(df, fig_path)

    cols = [
        "method", "n_seed", "n_ep",
        "planned_dispatch_headway_std__actual_dispatch_headway_cv_pearson",
        "planned_dispatch_headway_std__actual_dispatch_headway_cv_spearman",
        "planned_dispatch_headway_std_mean",
        "actual_dispatch_headway_cv_mean",
    ]
    cols = [c for c in cols if c in pooled.columns]
    print(pooled[cols].round(4).to_string(index=False))
    print(f"Wrote {out_dir / 'per_seed.csv'}")
    print(f"Wrote {out_dir / 'pooled.csv'}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
