#!/usr/bin/env python3
"""Create paper-grade validation figures for FreqDuet demand decomposers."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frequency import DemandFrequencyTracker, fit_harmonic_prior

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_frequency_modules import eval_method, make_synthetic_stream


METHODS = ["harmonic_prior", "harmonic", "ema", "haar", "raw_history"]
PLOT_METHODS = ["harmonic_prior", "ema", "raw_history"]
METHOD_COLORS = {
    "harmonic_prior": "#1f77b4",
    "harmonic": "#17becf",
    "ema": "#ff7f0e",
    "haar": "#2ca02c",
    "raw_history": "#9467bd",
}


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or x[mask].std() < 1e-9 or y[mask].std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _zscore(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mu = np.nanmean(arr)
    sigma = np.nanstd(arr)
    if not np.isfinite(sigma) or sigma < 1e-9:
        return np.zeros_like(arr)
    return (arr - mu) / sigma


def save(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in formats:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def make_tracker(method: str, low: np.ndarray, args: argparse.Namespace) -> DemandFrequencyTracker:
    tracker_method = "harmonic" if method == "harmonic_prior" else method
    harmonic_prior = None
    harmonic_prior_var = 100.0
    if method == "harmonic_prior":
        harmonic_prior = {
            "global": fit_harmonic_prior(
                low,
                update_interval_s=60.0,
                period_s=args.minutes * 60.0,
                fourier_k=args.fourier_k,
                ridge=args.harmonic_ridge,
            )
        }
        harmonic_prior_var = args.harmonic_prior_var
    return DemandFrequencyTracker(
        method=tracker_method,
        update_interval_s=60.0,
        bin_sec=args.bin_sec if tracker_method == "harmonic" else None,
        low_period_s=args.low_period_min * 60.0,
        fast_period_s=args.fast_period_min * 60.0,
        energy_period_s=args.energy_period_min * 60.0,
        forecast_horizon_s=args.forecast_min * 60.0,
        global_demand_norm=1.0,
        local_demand_norm=1.0,
        slope_norm=1.0,
        od_features=True,
        upper_mode="low",
        lower_mode="high",
        fourier_k=args.fourier_k,
        harmonic_forgetting=args.harmonic_forgetting,
        harmonic_prior_var=harmonic_prior_var,
        harmonic_prior=harmonic_prior,
    )


def trace_method(
    method: str,
    low: np.ndarray,
    station_stream: list[dict],
    od_stream: list[dict],
    args: argparse.Namespace,
) -> pd.DataFrame:
    tracker = make_tracker(method, low, args)
    rows = []
    for t, (st_counts, od_counts) in enumerate(zip(station_stream, od_stream)):
        tracker.update(st_counts, od_counts)
        rows.append({
            "minute": t,
            "method": method,
            "low": float(tracker.global_state.low),
            "high": float(tracker.global_state.high),
            "high_energy": float(tracker.global_state.high_energy),
        })
    return pd.DataFrame(rows)


def synthetic_metric_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed_rows = []
    for seed in range(args.seed, args.seed + args.n_seeds):
        low, high, burst_flag, station_stream, od_stream = make_synthetic_stream(
            minutes=args.minutes,
            seed=seed,
        )
        for method in METHODS:
            row = eval_method(method, low, high, burst_flag, station_stream, od_stream, args)
            row["seed"] = seed
            per_seed_rows.append(row)
    per_seed = pd.DataFrame(per_seed_rows)
    summary_rows = []
    for method, group in per_seed.groupby("method"):
        row = {"method": method, "n_seeds": int(group["seed"].nunique())}
        for col in ["low_rmse", "low_mae", "high_corr", "burst_f1", "burst_precision", "burst_recall"]:
            vals = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=0))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("method")
    return per_seed, summary


def figure_synthetic_components(args: argparse.Namespace, out_dir: Path, formats: list[str]) -> None:
    low, high, burst_flag, station_stream, od_stream = make_synthetic_stream(
        minutes=args.minutes,
        seed=args.plot_seed,
    )
    traces = [trace_method(method, low, station_stream, od_stream, args) for method in PLOT_METHODS]
    pred = pd.concat(traces, ignore_index=True)
    t = np.arange(args.minutes)

    fig, axes = plt.subplots(3, 1, figsize=(10.2, 7.2), sharex=True)
    axes[0].plot(t, low, color="black", linewidth=1.8, label="true LF")
    for method in PLOT_METHODS:
        sub = pred[pred["method"] == method]
        axes[0].plot(sub["minute"], sub["low"], color=METHOD_COLORS[method], alpha=0.9, label=method)
    axes[0].set_ylabel("pax/min")
    axes[0].set_title("Causal LF demand estimate tracks daily structure")
    axes[0].legend(frameon=False, ncol=4)

    axes[1].plot(t, high, color="black", linewidth=1.2, alpha=0.55, label="true HF")
    for method in PLOT_METHODS:
        sub = pred[pred["method"] == method]
        axes[1].plot(sub["minute"], sub["high"], color=METHOD_COLORS[method], alpha=0.9, label=method)
    axes[1].set_ylabel("residual")
    axes[1].set_title("HF residual reacts to local burst shocks")

    for method in PLOT_METHODS:
        sub = pred[pred["method"] == method]
        axes[2].plot(
            sub["minute"],
            np.sqrt(np.maximum(sub["high_energy"].to_numpy(dtype=float), 0.0)),
            color=METHOD_COLORS[method],
            alpha=0.9,
            label=method,
        )
    burst_minutes = t[burst_flag]
    if burst_minutes.size:
        axes[2].scatter(burst_minutes, np.full_like(burst_minutes, 0.0, dtype=float), s=9, color="black", label="true bursts")
    axes[2].set_ylabel("sqrt HF energy")
    axes[2].set_xlabel("minute")
    axes[2].set_title("HF energy is a causal burst detector")
    axes[2].legend(frameon=False, ncol=4)
    for ax in axes:
        ax.grid(alpha=0.25)
    save(fig, out_dir, "decomposer_synthetic_components", formats)


def sensitivity_table(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    prior_vars = [float(x) for x in args.sensitivity_prior_vars.split(",") if x.strip()]
    fast_periods = [float(x) for x in args.sensitivity_fast_periods.split(",") if x.strip()]
    for prior_var in prior_vars:
        for fast_period in fast_periods:
            local = argparse.Namespace(**vars(args))
            local.harmonic_prior_var = prior_var
            local.fast_period_min = fast_period
            metrics = []
            for seed in range(args.seed, args.seed + args.n_seeds):
                low, high, burst_flag, station_stream, od_stream = make_synthetic_stream(
                    minutes=args.minutes,
                    seed=seed,
                )
                metrics.append(eval_method("harmonic_prior", low, high, burst_flag, station_stream, od_stream, local))
            group = pd.DataFrame(metrics)
            rows.append({
                "method": "harmonic_prior",
                "harmonic_prior_var": prior_var,
                "fast_period_min": fast_period,
                "low_rmse_mean": float(group["low_rmse"].mean()),
                "high_corr_mean": float(group["high_corr"].mean()),
                "burst_f1_mean": float(group["burst_f1"].mean()),
            })
    return pd.DataFrame(rows)


def figure_sensitivity(sens: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    metrics = [
        ("low_rmse_mean", "LF RMSE", "lower"),
        ("high_corr_mean", "HF correlation", "higher"),
        ("burst_f1_mean", "Burst F1", "higher"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    for ax, (metric, title, direction) in zip(axes, metrics):
        pivot = sens.pivot(index="harmonic_prior_var", columns="fast_period_min", values=metric).sort_index()
        im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(np.arange(len(pivot.columns)), [f"{x:g}" for x in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)), [f"{x:g}" for x in pivot.index])
        ax.set_xlabel("fast period (min)")
        ax.set_ylabel("harmonic prior var")
        ax.set_title(f"{title} ({direction} is better)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, out_dir, "decomposer_cutoff_window_sensitivity", formats)


def find_trace_files(paths: list[Path]) -> list[Path]:
    traces = []
    for root in paths:
        if root.is_file() and root.name == "demand_trace.csv":
            traces.append(root)
        elif root.is_dir():
            direct = root / "demand_trace.csv"
            if direct.exists():
                traces.append(direct)
            traces.extend(root.glob("*/demand_trace.csv"))
    return sorted(set(traces))


def figure_trace_alignment(traces: list[Path], out_dir: Path, formats: list[str]) -> pd.DataFrame:
    rows = []
    if not traces:
        return pd.DataFrame()
    fig, axes = plt.subplots(len(traces), 2, figsize=(10.8, max(3.0, 2.8 * len(traces))), squeeze=False)
    for idx, trace_path in enumerate(traces):
        df = pd.read_csv(trace_path)
        x = np.arange(len(df))
        label = trace_path.parent.name
        left = axes[idx, 0]
        right = axes[idx, 1]
        for col, name, color in [
            ("arrivals", "arrivals", "#7f7f7f"),
            ("queue_total", "queue", "#ff7f0e"),
            ("freq_low_demand", "LF demand", "#1f77b4"),
        ]:
            if col in df.columns:
                left.plot(x, _zscore(df[col]), label=name, color=color, linewidth=1.2, alpha=0.9)
        left.set_title(f"LF peak alignment: {label}")
        left.set_ylabel("z-score")
        left.grid(alpha=0.25)
        left.legend(frameon=False, ncol=3)

        for col, name, color in [
            ("freq_high_energy", "HF energy", "#d62728"),
            ("board_wait_mean_s", "wait", "#ff7f0e"),
            ("lower_action_mean_s", "holding", "#2ca02c"),
        ]:
            if col in df.columns:
                right.plot(x, _zscore(df[col]), label=name, color=color, linewidth=1.2, alpha=0.9)
        right.set_title(f"HF residual to wait/holding: {label}")
        right.grid(alpha=0.25)
        right.legend(frameon=False, ncol=3)

        rows.append({
            "run_dir": str(trace_path.parent),
            "rows": int(len(df)),
            "corr_lf_queue": _corr(df.get("freq_low_demand", []), df.get("queue_total", [])),
            "corr_hf_wait": _corr(df.get("freq_high_energy", []), df.get("board_wait_mean_s", [])),
            "corr_hf_holding": _corr(df.get("freq_high_energy", []), df.get("lower_action_mean_s", [])),
        })
    axes[-1, 0].set_xlabel("trace step")
    axes[-1, 1].set_xlabel("trace step")
    save(fig, out_dir, "decomposer_trace_alignment", formats)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results_freqduet/decomposer_validation")
    ap.add_argument("--trace-logs", nargs="*", default=["logs_trace_smoke"])
    ap.add_argument("--formats", default="png,pdf")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--plot-seed", type=int, default=7)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--minutes", type=int, default=14 * 60)
    ap.add_argument("--burn-in-min", type=int, default=90)
    ap.add_argument("--bin-sec", type=float, default=60.0)
    ap.add_argument("--low-period-min", type=float, default=60.0)
    ap.add_argument("--fast-period-min", type=float, default=5.0)
    ap.add_argument("--energy-period-min", type=float, default=10.0)
    ap.add_argument("--forecast-min", type=float, default=30.0)
    ap.add_argument("--fourier-k", type=int, default=6)
    ap.add_argument("--harmonic-forgetting", type=float, default=0.9995)
    ap.add_argument("--harmonic-prior-var", type=float, default=0.01)
    ap.add_argument("--harmonic-ridge", type=float, default=1e-2)
    ap.add_argument("--sensitivity-prior-vars", default="0.001,0.01,0.1")
    ap.add_argument("--sensitivity-fast-periods", default="3,5,10")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [x.strip() for x in args.formats.split(",") if x.strip()]

    per_seed, summary = synthetic_metric_tables(args)
    per_seed.to_csv(out_dir / "decomposer_synthetic_per_seed.csv", index=False)
    summary.to_csv(out_dir / "decomposer_synthetic_summary.csv", index=False)
    figure_synthetic_components(args, out_dir, formats)

    sens = sensitivity_table(args)
    sens.to_csv(out_dir / "decomposer_cutoff_window_sensitivity.csv", index=False)
    figure_sensitivity(sens, out_dir, formats)

    traces = find_trace_files([Path(p) for p in args.trace_logs])
    trace_summary = figure_trace_alignment(traces, out_dir, formats)
    if not trace_summary.empty:
        trace_summary.to_csv(out_dir / "decomposer_trace_alignment.csv", index=False)

    payload = {
        "methods": METHODS,
        "plot_methods": PLOT_METHODS,
        "n_seeds": args.n_seeds,
        "trace_logs": [str(p) for p in args.trace_logs],
        "outputs": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
    }
    with (out_dir / "decomposer_validation_manifest.json").open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
