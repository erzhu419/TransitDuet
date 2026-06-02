#!/usr/bin/env python3
"""Generate FreqDuet mechanism figures from diagnostics logs.

The older ``make_mechanism_figures.py`` targets the TransitDuet H_hiro logs.
This script is for the FreqDuet paper package: it scans run directories that
contain ``diagnostics.csv`` and writes stable CSV/figure artifacts covering
HF-to-holding credit, LF drift, promotion behavior, leakage, and action/state
frequency spectra.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results_freqduet" / "mechanism_figures"

METHOD_ORDER = [
    "main",
    "main_driftcost",
    "rawhistory",
    "nofreq",
    "allfreq",
    "nopromotion",
    "noleakage",
]
DOMAIN_ORDER = ["terminal", "highnoise", "odshift", "rushshift", "other"]
METHOD_COLORS = {
    "main": "#1f77b4",
    "main_driftcost": "#17becf",
    "rawhistory": "#ff7f0e",
    "nofreq": "#2ca02c",
    "allfreq": "#9467bd",
    "nopromotion": "#8c564b",
    "noleakage": "#d62728",
    "other": "#7f7f7f",
}

METRIC_COLS = [
    "avg_wait_min",
    "headway_cv",
    "fleet_overshoot",
    "composite",
    "lower_action_mean",
    "lower_action_std",
    "lower_lambda",
    "lower_drift_penalty_mean",
    "lower_drift_cost_mean",
    "upper_hf_penalty_mean",
    "upper_hf_power_ratio",
    "lower_lf_drift_ratio",
    "freq_low_demand",
    "freq_low_slope",
    "freq_high_energy",
    "freq_middle_energy",
    "freq_od_high_energy",
    "freq_promotion_active",
    "freq_promotion_persistent",
    "freq_promotion_ratio",
    "freq_promotion_strength",
    "freq_promotion_absorbed",
    "upper_plan_penalty_mean",
    "upper_plan_reuse_ratio",
    "terminal_launch_shift_mean",
    "terminal_shift_cap_mean",
    "freq_wait_lower_penalty_mean",
    "freq_wait_lower_board_credit_mean",
    "freq_wait_lower_high_share_mean",
    "freq_wait_upper_credit_std",
    "demand_attr_mi_score",
    "shock_response_hit_rate",
    "shock_action_mean_s",
]


def parse_run_dir(path: Path) -> tuple[str, int | None]:
    name = path.name
    m = re.match(r"(.+)_seed(-?\d+)$", name)
    if not m:
        return name, None
    return m.group(1), int(m.group(2))


def parse_domain(config: str) -> str:
    if "_gen_highnoise_" in config:
        return "highnoise"
    if "_gen_odshift_" in config:
        return "odshift"
    if "_gen_rushshift_" in config:
        return "rushshift"
    if "_terminal_" in config:
        return "terminal"
    return "other"


def parse_method(config: str) -> str:
    if "driftcost" in config:
        return "main_driftcost"
    if "rawhistory" in config:
        return "rawhistory"
    if "nofreq" in config:
        return "nofreq"
    if "allfreq" in config:
        return "allfreq"
    if "nopromotion" in config:
        return "nopromotion"
    if "noleakage" in config:
        return "noleakage"
    if "_main_" in config or config.endswith("_main_hiro"):
        return "main"
    return "other"


def find_diagnostic_files(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_file() and path.name == "diagnostics.csv":
            found.append(path)
        elif path.is_dir():
            direct = path / "diagnostics.csv"
            if direct.exists():
                found.append(direct)
            found.extend(path.glob("*/diagnostics.csv"))
    return sorted(set(found))


def safe_numeric(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    wait = safe_numeric(out, "avg_wait_min", 0.0)
    cv = safe_numeric(out, "headway_cv", 0.0)
    overshoot = safe_numeric(out, "fleet_overshoot", 0.0)
    n_fleet = safe_numeric(out, "N_fleet", 15.0).replace(0, np.nan).fillna(15.0)
    out["composite"] = wait / 10.0 + (overshoot.clip(lower=0.0) ** 2) / n_fleet + cv
    return out


def load_episode_rows(logs: list[Path], last_k: int | None) -> pd.DataFrame:
    files = find_diagnostic_files(logs)
    rows: list[pd.DataFrame] = []
    for csv_path in files:
        config, seed = parse_run_dir(csv_path.parent)
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[warn] skip unreadable {csv_path}: {exc}")
            continue
        if df.empty:
            continue
        df = add_derived_columns(df)
        if last_k is not None and last_k > 0:
            df = df.tail(last_k).copy()
        df["run_dir"] = str(csv_path.parent)
        df["config"] = config
        df["seed"] = seed
        df["domain"] = parse_domain(config)
        df["method"] = parse_method(config)
        rows.append(df)
    if not rows:
        raise SystemExit("No diagnostics.csv files found")
    data = pd.concat(rows, ignore_index=True, sort=False)
    for col in METRIC_COLS:
        if col not in data.columns:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def mean_ci(vals: pd.Series) -> tuple[float, float, float, int]:
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    n = int(vals.size)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if n > 1 else 0.0
    ci = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return mean, std, ci, n


def write_summary(data: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    seed_rows = []
    for keys, group in data.groupby(["domain", "method", "config", "seed"], dropna=False):
        domain, method, config, seed = keys
        row = {"domain": domain, "method": method, "config": config, "seed": seed}
        for col in METRIC_COLS:
            row[col] = float(pd.to_numeric(group[col], errors="coerce").mean())
        seed_rows.append(row)
    per_seed = pd.DataFrame(seed_rows)
    per_seed.to_csv(out_dir / "mechanism_per_seed.csv", index=False)

    summary_rows = []
    for keys, group in per_seed.groupby(["domain", "method"], dropna=False):
        domain, method = keys
        row = {"domain": domain, "method": method, "n_seeds": int(group["seed"].nunique())}
        for col in METRIC_COLS:
            mean, std, ci, n = mean_ci(group[col])
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_ci95"] = ci
            row[f"{col}_n"] = n
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    domain_rank = {v: i for i, v in enumerate(DOMAIN_ORDER)}
    method_rank = {v: i for i, v in enumerate(METHOD_ORDER)}
    summary["_domain_rank"] = summary["domain"].map(domain_rank).fillna(99)
    summary["_method_rank"] = summary["method"].map(method_rank).fillna(99)
    summary = summary.sort_values(["_domain_rank", "_method_rank"]).drop(
        columns=["_domain_rank", "_method_rank"])
    summary.to_csv(out_dir / "mechanism_summary.csv", index=False)
    return summary


def save(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> None:
    fig.tight_layout()
    for ext in formats:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def boxplot_with_labels(ax: plt.Axes, vals: list[np.ndarray], labels: list[str]) -> dict:
    try:
        return ax.boxplot(vals, tick_labels=labels, showfliers=False, patch_artist=True)
    except TypeError:
        return ax.boxplot(vals, labels=labels, showfliers=False, patch_artist=True)


def ordered_methods(data: pd.DataFrame) -> list[str]:
    present = set(data["method"].dropna().astype(str))
    methods = [m for m in METHOD_ORDER if m in present]
    methods.extend(sorted(present - set(methods)))
    return methods


def figure_hf_action(data: pd.DataFrame, out_dir: Path, formats: list[str], sample: int) -> None:
    subset = data[data["method"].isin(["main", "main_driftcost", "rawhistory", "allfreq"])].copy()
    subset = subset.dropna(subset=["freq_high_energy", "lower_action_mean"])
    if subset.empty:
        return
    if sample > 0 and len(subset) > sample:
        subset = subset.sample(sample, random_state=7)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for method in ordered_methods(subset):
        g = subset[subset["method"] == method]
        ax.scatter(
            g["freq_high_energy"],
            g["lower_action_mean"],
            s=9,
            alpha=0.18,
            color=METHOD_COLORS.get(method, METHOD_COLORS["other"]),
            label=method,
            edgecolors="none",
        )
    main = subset[subset["method"].isin(["main", "main_driftcost"])]
    if len(main) >= 20:
        bins = pd.qcut(main["freq_high_energy"], q=12, duplicates="drop")
        binned = main.groupby(bins, observed=False).agg(
            x=("freq_high_energy", "mean"),
            y=("lower_action_mean", "mean"),
        ).dropna()
        ax.plot(binned["x"], binned["y"], color="black", linewidth=1.6, label="main binned")
    ax.set_xlabel("HF demand energy")
    ax.set_ylabel("Lower holding action mean (s)")
    ax.set_title("HF residual credit to lower holding")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    save(fig, out_dir, "mechanism_hf_energy_to_holding", formats)


def figure_drift_boxes(data: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    methods = ordered_methods(data)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=False)
    metrics = [
        ("lower_lf_drift_ratio", "Lower LF drift ratio"),
        ("lower_drift_penalty_mean", "Lower drift penalty"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        vals = [
            pd.to_numeric(data.loc[data["method"] == m, col], errors="coerce").dropna().values
            for m in methods
        ]
        vals = [v for v in vals if len(v)]
        labels = [m for m in methods if data.loc[data["method"] == m, col].notna().any()]
        if not vals:
            continue
        bp = boxplot_with_labels(ax, vals, labels)
        for patch, label in zip(bp.get("boxes", []), labels):
            patch.set_facecolor(METHOD_COLORS.get(label, METHOD_COLORS["other"]))
            patch.set_alpha(0.35)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.25, axis="y")
    save(fig, out_dir, "mechanism_lower_drift_by_method", formats)


def figure_promotion(data: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    main = data[data["method"].isin(["main", "main_driftcost"])].copy()
    if main.empty or "freq_promotion_active" not in main.columns:
        return
    main["promotion_state"] = np.where(
        pd.to_numeric(main["freq_promotion_active"], errors="coerce").fillna(0.0) > 0.0,
        "active",
        "inactive",
    )
    metrics = [
        ("avg_wait_min", "Wait (min)"),
        ("composite", "Composite"),
        ("fleet_overshoot", "Fleet overshoot"),
        ("upper_plan_penalty_mean", "Upper plan penalty"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(10.5, 3.0))
    for ax, (col, label) in zip(axes, metrics):
        rows = []
        for state in ["inactive", "active"]:
            vals = pd.to_numeric(main.loc[main["promotion_state"] == state, col], errors="coerce").dropna()
            rows.append((state, vals.mean() if len(vals) else np.nan, vals.std() if len(vals) > 1 else 0.0))
        x = np.arange(len(rows))
        ax.bar(x, [r[1] for r in rows], yerr=[r[2] for r in rows], capsize=3,
               color=["#bdbdbd", "#1f77b4"], edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows], rotation=20)
        ax.set_title(label)
        ax.grid(alpha=0.25, axis="y")
    save(fig, out_dir, "mechanism_promotion_active_vs_inactive", formats)


def figure_ablation_bars(summary: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    terminal = summary[summary["domain"].isin(["terminal", "highnoise", "odshift", "rushshift"])].copy()
    if terminal.empty:
        return
    methods = [m for m in METHOD_ORDER if m in set(terminal["method"])]
    metrics = [
        ("composite_mean", "Composite"),
        ("lower_action_mean_mean", "Lower action"),
        ("upper_hf_power_ratio_mean", "Upper HF power"),
        ("lower_lf_drift_ratio_mean", "Lower LF drift"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.8))
    for ax, (col, title) in zip(axes.flat, metrics):
        piv = terminal.pivot_table(index="domain", columns="method", values=col, aggfunc="mean")
        piv = piv.reindex([d for d in DOMAIN_ORDER if d in piv.index])
        width = 0.8 / max(1, len(methods))
        x = np.arange(len(piv.index))
        for i, method in enumerate(methods):
            if method not in piv.columns:
                continue
            ax.bar(
                x + (i - (len(methods) - 1) / 2) * width,
                piv[method].values,
                width=width,
                label=method,
                color=METHOD_COLORS.get(method, METHOD_COLORS["other"]),
                alpha=0.85,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(piv.index, rotation=20)
        ax.grid(alpha=0.25, axis="y")
    axes.flat[0].legend(frameon=False, ncol=3, fontsize=7)
    save(fig, out_dir, "mechanism_domain_method_bars", formats)


def figure_frequency_spectrum(summary: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    rows = summary[summary["method"].isin(["main", "main_driftcost", "rawhistory", "allfreq", "nofreq"])].copy()
    if rows.empty:
        return
    rows["domain_method"] = rows["domain"].astype(str) + "/" + rows["method"].astype(str)
    cols = [
        ("freq_low_demand_mean", "LF demand"),
        ("freq_high_energy_mean", "HF energy"),
        ("freq_middle_energy_mean", "MF energy"),
        ("freq_od_high_energy_mean", "OD HF energy"),
    ]
    available = [(c, l) for c, l in cols if c in rows.columns]
    if not available:
        return
    rows = rows.sort_values(
        by=["domain", "method"],
        key=lambda s: s.map({**{d: i for i, d in enumerate(DOMAIN_ORDER)},
                             **{m: i for i, m in enumerate(METHOD_ORDER)}}).fillna(99),
    )
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    for (col, label), color in zip(available, colors):
        vals = pd.to_numeric(rows[col], errors="coerce").fillna(0.0).clip(lower=0.0).values
        ax.bar(x, vals, bottom=bottom, label=label, color=color, alpha=0.85)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(rows["domain_method"], rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Mean feature energy / magnitude")
    ax.set_title("State frequency spectrum by domain and method")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=False, ncol=4)
    save(fig, out_dir, "mechanism_action_state_spectrum", formats)


def figure_training_drift(data: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    cols = [
        ("lower_action_mean", "Lower action mean"),
        ("lower_lambda", "Lower lambda"),
        ("lower_drift_penalty_mean", "Drift penalty"),
        ("lower_drift_cost_mean", "Drift Lagrangian cost"),
    ]
    methods = [m for m in ["main", "main_driftcost", "nopromotion", "rawhistory"] if m in set(data["method"])]
    if not methods:
        return
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.8), sharex=True)
    for ax, (col, title) in zip(axes.flat, cols):
        if col not in data.columns or not data[col].notna().any():
            ax.set_visible(False)
            continue
        for method in methods:
            g = data[data["method"] == method].copy()
            if g.empty or "ep" not in g.columns:
                continue
            curve = g.groupby("ep")[col].mean().rolling(5, min_periods=1).mean()
            ax.plot(curve.index, curve.values, label=method,
                    color=METHOD_COLORS.get(method, METHOD_COLORS["other"]), linewidth=1.4)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes.flat[0].legend(frameon=False, ncol=2)
    for ax in axes[-1, :]:
        ax.set_xlabel("Episode")
    save(fig, out_dir, "mechanism_longtrain_drift_curves", formats)


def trace_lag_audit(logs: list[Path], out_dir: Path) -> None:
    traces = []
    for path in logs:
        if path.is_file() and path.name == "demand_trace.csv":
            traces.append(path)
        elif path.is_dir():
            direct = path / "demand_trace.csv"
            if direct.exists():
                traces.append(direct)
            traces.extend(path.glob("*/demand_trace.csv"))
    rows = []
    for trace in sorted(set(traces)):
        try:
            df = pd.read_csv(trace)
        except Exception:
            continue
        if "freq_high_energy" not in df.columns:
            continue
        config, seed = parse_run_dir(trace.parent)
        for target in ["lower_action_mean_s", "board_wait_mean_s"]:
            if target not in df.columns:
                continue
            x = pd.to_numeric(df["freq_high_energy"], errors="coerce")
            y = pd.to_numeric(df[target], errors="coerce")
            for lag in range(-8, 9):
                shifted = y.shift(-lag)
                mask = x.notna() & shifted.notna()
                corr = 0.0
                if mask.sum() >= 5 and x[mask].std() > 1e-12 and shifted[mask].std() > 1e-12:
                    corr = float(x[mask].corr(shifted[mask]))
                rows.append({
                    "run_dir": str(trace.parent),
                    "config": config,
                    "seed": seed,
                    "domain": parse_domain(config),
                    "method": parse_method(config),
                    "target": target,
                    "lag": lag,
                    "corr": corr,
                })
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "trace_hf_lag_audit.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True,
                        help="Log roots or run directories containing diagnostics.csv")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--last-k", type=int, default=100,
                        help="Use the last K episodes per run; <=0 uses all episodes")
    parser.add_argument("--sample", type=int, default=12000,
                        help="Maximum points in scatter plots; <=0 disables sampling")
    parser.add_argument("--formats", default="pdf,png",
                        help="Comma-separated figure extensions")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [f.strip().lstrip(".") for f in args.formats.split(",") if f.strip()]
    logs = [Path(p) for p in args.logs]
    last_k = args.last_k if args.last_k and args.last_k > 0 else None

    data = load_episode_rows(logs, last_k)
    data.to_csv(out_dir / "mechanism_episode_rows.csv", index=False)
    summary = write_summary(data, out_dir)
    figure_hf_action(data, out_dir, formats, args.sample)
    figure_drift_boxes(data, out_dir, formats)
    figure_promotion(data, out_dir, formats)
    figure_ablation_bars(summary, out_dir, formats)
    figure_frequency_spectrum(summary, out_dir, formats)
    figure_training_drift(data, out_dir, formats)
    trace_lag_audit(logs, out_dir)
    print(f"loaded {data['run_dir'].nunique()} runs and {len(data)} episode rows")


if __name__ == "__main__":
    main()
