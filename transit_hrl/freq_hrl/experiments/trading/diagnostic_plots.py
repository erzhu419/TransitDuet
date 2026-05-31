"""Generate diagnostic figures for Freq-HRL trading validations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from freq_hrl.domains.trading import TradingFrequencyTracker

from .performance_validation import SCENARIOS, make_synthetic_market


def _load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _first_column(rows: list[dict[str, Any]], candidates: list[str]) -> str | None:
    if not rows:
        return None
    available = set(rows[0].keys())
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_signal_decomposition(output_dir: Path, seed: int, steps: int, assets: int, scenario: str) -> None:
    plt = _matplotlib()
    data = make_synthetic_market(seed=seed, steps=steps, n_assets=assets, scenario=scenario)
    tracker = TradingFrequencyTracker(
        bar_sec=60.0,
        low_period_s=120 * 60.0,
        fast_period_s=5 * 60.0,
        mid_period_s=30 * 60.0,
        energy_period_s=10 * 60.0,
        persistence_period_s=30 * 60.0,
        persistence_threshold=0.0010,
        feature_norm=np.ones(assets) * 0.0015,
        promotion_enable=True,
        promotion_residual_threshold=0.00035,
        promotion_persistence_ratio=0.40,
        promotion_adapt_low=True,
        promotion_adapt_gain=0.25,
    )
    raw, low, mid, high, promote = [], [], [], [], []
    for t in range(steps):
        feats = tracker.update_bar(data["predictor"][t], t=float(t * 60.0))
        raw.append(float(np.asarray(feats["x_raw"]).reshape(-1)[0]))
        low.append(float(np.asarray(feats["x_low"]).reshape(-1)[0]))
        mid.append(float(np.asarray(feats["x_mid"]).reshape(-1)[0]))
        high.append(float(np.asarray(feats["x_high"]).reshape(-1)[0]))
        promote.append(1.0 if dict(feats.get("promotion", {}) or {}).get("promote", False) else 0.0)
    x = np.arange(steps)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, raw, label="raw", linewidth=1.0, alpha=0.65)
    ax.plot(x, low, label="LF", linewidth=1.4)
    ax.plot(x, mid, label="MF", linewidth=1.2)
    ax.plot(x, high, label="HF", linewidth=1.0, alpha=0.75)
    ax.set_title(f"Causal frequency decomposition ({scenario})")
    ax.set_xlabel("bar")
    ax.set_ylabel("signal")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "signal_decomposition.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(x, np.asarray(promote), drawstyle="steps-post", color="tab:red")
    ax.set_title("Promotion gate timeline")
    ax.set_xlabel("bar")
    ax.set_ylabel("promote")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "promotion_timeline.png", dpi=160)
    plt.close(fig)


def plot_performance_bars(performance_dir: Path, output_dir: Path) -> None:
    rows = _load_csv(performance_dir / "summary.csv")
    if not rows:
        return
    plt = _matplotlib()
    rows = sorted(rows, key=lambda row: _float(row, "sharpe_mean"), reverse=True)
    labels = [row["baseline"] for row in rows]
    sharpe = [_float(row, "sharpe_mean") for row in rows]
    focus = [_float(row, "FocusScore_mean") for row in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x, sharpe, color="tab:blue", alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Sharpe")
    ax.set_title("Baseline Sharpe")
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_sharpe_bars.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(focus, sharpe, color="tab:green", alpha=0.85)
    for label, px, py in zip(labels, focus, sharpe):
        ax.annotate(label, (px, py), fontsize=7, alpha=0.75)
    ax.set_xlabel("FocusScore")
    ax.set_ylabel("Sharpe")
    ax.set_title("FocusScore vs task metric")
    fig.tight_layout()
    fig.savefig(output_dir / "focus_score_scatter.png", dpi=160)
    plt.close(fig)


def plot_noleakage_drift(performance_dir: Path, output_dir: Path) -> None:
    rows = _load_csv(performance_dir / "summary.csv")
    metric = _first_column(
        rows,
        [
            "LowerLFDrift_mean",
            "LowerLFInventoryDrift_mean",
            "inventory_drift_mean_mean",
        ],
    )
    if not rows or metric is None:
        return
    priority = {"freq_hrl": 0, "no_leakage": 1}
    rows = sorted(
        rows,
        key=lambda row: (priority.get(row.get("baseline", ""), 2), row.get("baseline", "")),
    )
    labels = [row["baseline"] for row in rows]
    values = [_float(row, metric) for row in rows]
    colors = []
    for label in labels:
        if label == "freq_hrl":
            colors.append("tab:blue")
        elif label == "no_leakage":
            colors.append("tab:red")
        else:
            colors.append("0.70")
    plt = _matplotlib()
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x, values, color=colors, alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(metric.replace("_mean", ""))
    ax.set_title("No-Leakage lower LF drift comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "noleakage_drift_comparison.png", dpi=160)
    plt.close(fig)


def plot_pressure_matrix(pressure_dir: Path, output_dir: Path) -> None:
    rows = _load_csv(pressure_dir / "summary.csv")
    if not rows:
        return
    plt = _matplotlib()
    scenarios = [s for s in SCENARIOS if any(row.get("scenario") == s for row in rows)]
    baselines = sorted({row["baseline"] for row in rows})
    matrix = np.zeros((len(scenarios), len(baselines)), dtype=np.float64)
    for i, scenario in enumerate(scenarios):
        for j, baseline in enumerate(baselines):
            match = [
                row for row in rows
                if row.get("scenario") == scenario and row.get("baseline") == baseline
            ]
            matrix[i, j] = _float(match[0], "sharpe_mean") if match else np.nan
    fig, ax = plt.subplots(figsize=(max(8, len(baselines) * 0.7), 4.5))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(baselines)))
    ax.set_xticklabels(baselines, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels(scenarios)
    ax.set_title("Pressure-test Sharpe by scenario")
    fig.colorbar(im, ax=ax, label="Sharpe")
    fig.tight_layout()
    fig.savefig(output_dir / "pressure_matrix_sharpe.png", dpi=160)
    plt.close(fig)


def plot_promotion_recovery(promotion_dir: Path, output_dir: Path) -> None:
    rows = _load_csv(promotion_dir / "summary.csv")
    if not rows:
        return
    plt = _matplotlib()
    x = [_float(row, "post_shift_delta") for row in rows]
    y = [_float(row, "sharpe_delta") for row in rows]
    c = [_float(row, "recovery_score") for row in rows]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(x, y, c=c, cmap="coolwarm", s=24, alpha=0.8)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xlabel("Post-shift PnL delta")
    ax.set_ylabel("Sharpe delta")
    ax.set_title("Promotion recovery tradeoff")
    fig.colorbar(sc, ax=ax, label="recovery score")
    fig.tight_layout()
    fig.savefig(output_dir / "promotion_recovery_scatter.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--performance-dir", type=Path, default=Path("transit_hrl/results/trading_performance"))
    parser.add_argument("--pressure-dir", type=Path, default=Path("transit_hrl/results/trading_pressure_matrix"))
    parser.add_argument("--promotion-dir", type=Path, default=Path("transit_hrl/results/trading_promotion_recovery_sweep"))
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/trading_diagnostic_plots"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--scenario", choices=SCENARIOS, default="persistent_shift")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_signal_decomposition(args.output_dir, args.seed, args.steps, args.assets, args.scenario)
    plot_performance_bars(args.performance_dir, args.output_dir)
    plot_noleakage_drift(args.performance_dir, args.output_dir)
    plot_pressure_matrix(args.pressure_dir, args.output_dir)
    plot_promotion_recovery(args.promotion_dir, args.output_dir)
    print(f"wrote diagnostic plots to {args.output_dir}")


if __name__ == "__main__":
    main()
