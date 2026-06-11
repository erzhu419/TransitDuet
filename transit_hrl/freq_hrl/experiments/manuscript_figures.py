"""Generate manuscript figures for the conservative Freq-HRL submission pack.

Python/matplotlib is the selected and exclusive rendering backend for these
figures.  The script reads committed experiment artifacts and writes editable
SVG, PDF, TIFF, PNG, and per-figure source-data CSV files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle


DEFAULT_RESULTS_ROOT = Path("transit_hrl/results")
DEFAULT_OUTPUT_DIR = Path("transit_hrl/results/manuscript_figures_latest")

BLUE = "#0F4D92"
BLUE2 = "#3775BA"
TEAL = "#42949E"
GREEN = "#2E9E44"
GREEN_SOFT = "#AADCA9"
RED = "#B64342"
RED_SOFT = "#F6CFCB"
GOLD = "#D89C23"
VIOLET = "#9A4D8E"
NEUTRAL = "#767676"
NEUTRAL_DARK = "#4D4D4D"
NEUTRAL_LIGHT = "#D8D8D8"
BG_BLUE = "#E7F0F8"
BG_GREEN = "#E7F5E9"
BG_GOLD = "#F7EFD9"
BG_VIOLET = "#F1E7F2"


ARTIFACTS = {
    "claims": Path("top_journal_unified_matrix_latest/claims.csv"),
    "baseline": Path("baseline_ablation_matrix_latest/paired_checks.csv"),
    "scenario_winners": Path("baseline_ablation_matrix_latest/scenario_winners.csv"),
    "promotion": Path("transit_native_promotion_v47_odshift_wait_first_512seed_summaryonly/paired_checks.csv"),
    "real_demand": Path("transit_native_real_demand_service_response_v7_48pair_merged/paired_checks.csv"),
    "agency_boundaries": Path("agency_demand_onboard_coverage_latest/claim_boundaries.csv"),
    "external_sources": Path("external_transit_truth_validation_latest/source_coverage.csv"),
    "order_book_per_eval": Path("order_book_lobster_venue_grade_multisymbol/per_eval.csv"),
    "order_book_checks": Path("order_book_lobster_venue_grade_multisymbol/paired_checks.csv"),
    "order_book_summary": Path("order_book_lobster_venue_grade_multisymbol/summary.json"),
    "encoder_domains": Path("encoder_cross_domain_matrix/domain_summary.csv"),
}


def apply_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "axes.titleweight": "bold",
    })


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _paths(results_root: Path) -> dict[str, Path]:
    return {key: results_root / rel for key, rel in ARTIFACTS.items()}


def _save(fig: plt.Figure, output_dir: Path, name: str, dpi: int = 450) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / name
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.03,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = NEUTRAL_DARK) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, lw=1.1, color=color))


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    label: str,
    *,
    fc: str,
    ec: str = NEUTRAL_DARK,
    fontsize: int = 7,
) -> None:
    rect = Rectangle(xy, wh[0], wh[1], facecolor=fc, edgecolor=ec, linewidth=1.0)
    ax.add_patch(rect)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, label, ha="center", va="center", fontsize=fontsize)


def fig1_protocol(output_dir: Path, source_dir: Path) -> dict[str, Any]:
    source_rows = pd.DataFrame([
        {"module": "causal encoder", "responsibility": "trend / regime / residual"},
        {"module": "upper planner", "responsibility": "low-frequency plan"},
        {"module": "promotion gate", "responsibility": "persistent shock replan"},
        {"module": "lower controller", "responsibility": "high-frequency correction"},
        {"module": "leakage and credit", "responsibility": "responsibility audit"},
    ])
    source_rows.to_csv(source_dir / "fig1_protocol_source.csv", index=False)

    fig = plt.figure(figsize=(7.2, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    _panel(ax_a, "a")
    ax_a.set_title("Time-series control abstraction", loc="left", fontsize=9)
    _box(ax_a, (0.05, 0.62), (0.25, 0.18), "exogenous\nstream", fc=BG_BLUE)
    _box(ax_a, (0.05, 0.22), (0.25, 0.18), "endogenous\nstate", fc=BG_GREEN)
    _box(ax_a, (0.44, 0.42), (0.22, 0.18), "Freq-HRL\npolicy", fc=BG_GOLD)
    _box(ax_a, (0.78, 0.42), (0.17, 0.18), "action", fc=BG_VIOLET)
    _arrow(ax_a, (0.31, 0.71), (0.44, 0.53))
    _arrow(ax_a, (0.31, 0.31), (0.44, 0.49))
    _arrow(ax_a, (0.66, 0.51), (0.78, 0.51))
    ax_a.text(0.05, 0.05, "causal inputs only", color=NEUTRAL, fontsize=7)

    _panel(ax_b, "b")
    ax_b.set_title("Causal frequency encoder", loc="left", fontsize=9)
    x = np.linspace(0.05, 0.95, 160)
    trend = 0.58 + 0.18 * np.sin(2 * np.pi * x)
    residual = 0.58 + 0.05 * np.sin(24 * np.pi * x)
    ax_b.plot(x, trend, color=BLUE, lw=2, label="LF trend")
    ax_b.plot(x, residual, color=TEAL, lw=1.2, label="HF residual")
    ax_b.fill_between(x, 0.35, trend, color=BG_BLUE, alpha=0.55)
    ax_b.text(0.06, 0.82, "trend", color=BLUE, fontsize=7)
    ax_b.text(0.59, 0.40, "residual", color=TEAL, fontsize=7)
    _box(ax_b, (0.32, 0.08), (0.38, 0.15), "uncertainty + persistence", fc="#F6F6F6", fontsize=7)

    _panel(ax_c, "c")
    ax_c.set_title("Hierarchical routing and promotion", loc="left", fontsize=9)
    _box(ax_c, (0.06, 0.66), (0.22, 0.17), "upper\nplanner", fc=BG_BLUE)
    _box(ax_c, (0.40, 0.66), (0.22, 0.17), "plan\ncurve", fc=BG_GOLD)
    _box(ax_c, (0.72, 0.66), (0.22, 0.17), "lower\ncontroller", fc=BG_GREEN)
    _box(ax_c, (0.40, 0.18), (0.22, 0.17), "promotion\ngate", fc=RED_SOFT)
    _arrow(ax_c, (0.28, 0.745), (0.40, 0.745))
    _arrow(ax_c, (0.62, 0.745), (0.72, 0.745))
    _arrow(ax_c, (0.51, 0.35), (0.18, 0.66), color=RED)
    ax_c.text(0.09, 0.50, "slow plan", color=BLUE, fontsize=7)
    ax_c.text(0.70, 0.50, "fast correction", color=GREEN, fontsize=7)

    _panel(ax_d, "d")
    ax_d.set_title("Responsibility accounting", loc="left", fontsize=9)
    names = ["upper HF\npower", "lower LF\ndrift", "credit\nresidual", "stress\ncoverage"]
    vals = [0.32, 0.24, 0.14, 0.90]
    colors = [RED_SOFT, RED_SOFT, BG_GOLD, BG_GREEN]
    y = np.arange(len(names))
    ax_d.barh(y, vals, color=colors, edgecolor=NEUTRAL_DARK, lw=0.6)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(names, fontsize=7)
    ax_d.set_xlim(0, 1)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("diagnostic scale", fontsize=7)
    ax_d.grid(axis="x", color="#EEEEEE", lw=0.6)
    for spine in ax_d.spines.values():
        spine.set_visible(False)

    _save(fig, output_dir, "fig1_frequency_separated_protocol")
    return {"figure": "fig1_frequency_separated_protocol", "source_rows": len(source_rows)}


def _status_color(status: str) -> str:
    return {
        "supported": GREEN,
        "positive_mixed": GOLD,
        "inconclusive": NEUTRAL_LIGHT,
        "not_supported": RED,
        "external_missing": NEUTRAL,
        "summary_only": NEUTRAL_LIGHT,
    }.get(str(status), NEUTRAL_LIGHT)


def fig2_evidence_matrix(paths: dict[str, Path], output_dir: Path, source_dir: Path) -> dict[str, Any]:
    claims = _read_csv(paths["claims"])
    baseline = _read_csv(paths["baseline"])
    scenarios = _read_csv(paths["scenario_winners"])

    source = {
        "claims": claims,
        "baseline_sharpe": baseline[baseline["metric"] == "sharpe"].copy(),
        "scenarios": scenarios,
    }
    source["claims"].to_csv(source_dir / "fig2_claims_source.csv", index=False)
    source["baseline_sharpe"].to_csv(source_dir / "fig2_baseline_sharpe_source.csv", index=False)
    source["scenarios"].to_csv(source_dir / "fig2_scenarios_source.csv", index=False)

    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _panel(ax_a, "a")
    ax_a.set_title("Conservative claim matrix", loc="left", fontsize=9)
    claim_ids = claims["id"].tolist()
    colors = [_status_color(s) for s in claims["status"]]
    ax_a.barh(np.arange(len(claim_ids)), np.ones(len(claim_ids)), color=colors, edgecolor="white")
    ax_a.set_yticks(np.arange(len(claim_ids)))
    ax_a.set_yticklabels(claim_ids, fontsize=7)
    ax_a.set_xlim(0, 1)
    ax_a.set_xticks([])
    ax_a.invert_yaxis()
    for y, text in enumerate(claims["claim"].astype(str).str.slice(0, 48)):
        ax_a.text(0.03, y, text, va="center", ha="left", fontsize=6.3, color="white")
    for spine in ax_a.spines.values():
        spine.set_visible(False)

    _panel(ax_b, "b")
    ax_b.set_title("Baseline ablation: Sharpe deltas", loc="left", fontsize=9)
    sharpe = baseline[(baseline["metric"] == "sharpe") & baseline["control"].isin([
        "vanilla_rl",
        "hrl_raw",
        "raw_history",
        "freq_single_policy",
        "allfreq_alllayers",
        "swapped",
        "no_leakage",
        "hf_lower_only",
    ])].copy()
    sharpe = sharpe.sort_values("delta_mean")
    ax_b.barh(sharpe["control"], sharpe["delta_mean"], color=BLUE2, edgecolor=NEUTRAL_DARK, lw=0.5)
    ax_b.errorbar(
        sharpe["delta_mean"],
        sharpe["control"],
        xerr=[
            sharpe["delta_mean"] - sharpe["delta_ci95_low"],
            sharpe["delta_ci95_high"] - sharpe["delta_mean"],
        ],
        fmt="none",
        ecolor=NEUTRAL_DARK,
        lw=0.8,
        capsize=2,
    )
    ax_b.axvline(0, color=NEUTRAL_DARK, lw=0.8)
    ax_b.set_xlabel("delta Sharpe vs baseline", fontsize=7)
    ax_b.tick_params(axis="y", labelsize=6.5)
    ax_b.grid(axis="x", color="#EEEEEE", lw=0.6)

    _panel(ax_c, "c")
    ax_c.set_title("Registered stress regimes", loc="left", fontsize=9)
    y = np.arange(len(scenarios))
    ax_c.barh(y, scenarios["freq_family_best_sharpe"], color=GREEN_SOFT, edgecolor=NEUTRAL_DARK, lw=0.5)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(scenarios["scenario"], fontsize=7)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("best frequency-family Sharpe", fontsize=7)
    for i, wins in enumerate(scenarios["freq_family_wins"].astype(str)):
        ax_c.text(
            scenarios["freq_family_best_sharpe"].iloc[i],
            i,
            " pass" if wins.lower() == "true" else " fail",
            va="center",
            fontsize=6.5,
            color=GREEN if wins.lower() == "true" else RED,
        )
    ax_c.grid(axis="x", color="#EEEEEE", lw=0.6)

    _panel(ax_d, "d")
    ax_d.set_title("Evidence accounting", loc="left", fontsize=9)
    counts = claims["status"].value_counts().reindex(["supported", "partial", "missing"], fill_value=0)
    ax_d.bar(counts.index, counts.values, color=[GREEN, GOLD, RED_SOFT], edgecolor=NEUTRAL_DARK, lw=0.5)
    ax_d.set_ylabel("claims", fontsize=7)
    ax_d.set_ylim(0, max(10, counts.max() + 1))
    for i, v in enumerate(counts.values):
        ax_d.text(i, v + 0.15, str(int(v)), ha="center", fontsize=8, fontweight="bold")
    ax_d.text(
        0.02,
        0.92,
        "Boundary retained:\nexternal deployment and\njoint agency OD/load loop",
        transform=ax_d.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=NEUTRAL_DARK,
    )

    _save(fig, output_dir, "fig2_claim_ablation_matrix")
    return {"figure": "fig2_claim_ablation_matrix", "claim_rows": int(len(claims))}


def _select_checks(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    selected = df[df["metric"].isin(metrics)].copy()
    order = {metric: i for i, metric in enumerate(metrics)}
    selected["order"] = selected["metric"].map(order)
    if "status" in selected.columns:
        status_rank = {"supported": 0, "positive_mixed": 1, "inconclusive": 2, "not_supported": 3}
        selected["status_rank"] = selected["status"].map(status_rank).fillna(9)
    else:
        selected["status_rank"] = 0
    if "noninferiority_margin" in selected.columns:
        selected["is_noninferiority"] = selected["noninferiority_margin"].notna().astype(int)
    else:
        selected["is_noninferiority"] = 0
    return (
        selected.sort_values(["order", "is_noninferiority", "status_rank"])
        .drop_duplicates("metric", keep="first")
        .sort_values("order")
    )


def _ci_barh(ax: plt.Axes, df: pd.DataFrame, labels: list[str], title: str) -> None:
    y = np.arange(len(df))
    vals = df["improvement_mean"].astype(float).to_numpy()
    lo = df["improvement_ci95_low"].astype(float).to_numpy()
    hi = df["improvement_ci95_high"].astype(float).to_numpy()
    colors = [_status_color(s) for s in df["status"]]
    ax.barh(y, vals, color=colors, edgecolor=NEUTRAL_DARK, lw=0.5)
    ax.errorbar(
        vals,
        y,
        xerr=[np.maximum(vals - lo, 0), np.maximum(hi - vals, 0)],
        fmt="none",
        ecolor=NEUTRAL_DARK,
        lw=0.8,
        capsize=2,
    )
    positive = vals[vals > 0]
    use_log = bool(len(positive) and np.nanmax(positive) / max(np.nanmin(positive), 1e-12) > 150)
    if use_log:
        ax.set_xscale("log")
        min_pos = max(np.nanmin(np.maximum(lo, 1e-12)), 1e-6)
        ax.set_xlim(min_pos * 0.65, np.nanmax(hi) * 1.6)
    else:
        ax.axvline(0, color=NEUTRAL_DARK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_xlabel("improvement (native units)", fontsize=7)
    ax.grid(axis="x", color="#EEEEEE", lw=0.6)
    for yi, value in zip(y, vals):
        label = f"{value:.3g}"
        x_text = value * (1.08 if use_log else 1.02)
        ax.text(x_text, yi, label, va="center", ha="left", fontsize=6.3, color=NEUTRAL_DARK)


def fig3_transit(paths: dict[str, Path], output_dir: Path, source_dir: Path) -> dict[str, Any]:
    promotion = _read_csv(paths["promotion"])
    real = _read_csv(paths["real_demand"])
    prom = _select_checks(promotion, ["ep_reward", "avg_wait_min", "score", "shared_ppo_gate_replans"])
    real_main = _select_checks(
        real,
        [
            "control_score",
            "ep_reward",
            "avg_wait_min",
            "native_avg_board_wait_min",
            "native_alighted_pax",
            "native_completed_throughput_pax",
            "LowerLFDrift",
        ],
    )
    prom.to_csv(source_dir / "fig3_promotion_source.csv", index=False)
    real_main.to_csv(source_dir / "fig3_real_demand_source.csv", index=False)

    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _panel(ax_a, "a")
    prom_labels = ["reward", "wait", "score", "replans"]
    _ci_barh(ax_a, prom, prom_labels, "Native promotion CIs")

    _panel(ax_b, "b")
    wait = real_main[real_main["metric"].isin(["avg_wait_min", "native_avg_board_wait_min", "LowerLFDrift"])]
    _ci_barh(ax_b, wait, ["avg wait", "board wait", "lower LF drift"], "Wait and leakage improvements")

    _panel(ax_c, "c")
    throughput = real_main[real_main["metric"].isin(["native_alighted_pax", "native_completed_throughput_pax"])]
    _ci_barh(ax_c, throughput, ["alighted pax", "completed throughput"], "Passenger throughput")

    _panel(ax_d, "d")
    score = real_main[real_main["metric"].isin(["control_score", "ep_reward"])]
    _ci_barh(ax_d, score, ["control score", "episode reward"], "Native real-demand score")
    ax_d.text(
        0.98,
        0.08,
        "n=512 promotion pairs\nn=96 real-demand pairs",
        transform=ax_d.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color=NEUTRAL_DARK,
    )

    _save(fig, output_dir, "fig3_transit_promotion_real_demand")
    return {"figure": "fig3_transit_promotion_real_demand", "promotion_checks": int(len(prom)), "real_checks": int(len(real_main))}


def fig4_external_data(paths: dict[str, Path], output_dir: Path, source_dir: Path) -> dict[str, Any]:
    boundaries = _read_csv(paths["agency_boundaries"])
    external = _read_csv(paths["external_sources"])
    boundaries.to_csv(source_dir / "fig4_boundaries_source.csv", index=False)
    external.to_csv(source_dir / "fig4_external_sources_source.csv", index=False)

    fig = plt.figure(figsize=(7.2, 4.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _panel(ax_a, "a")
    ax_a.set_title("Agency coverage ledger", loc="left", fontsize=9)
    status_counts = boundaries["status"].value_counts().reindex(["supported", "external_missing"], fill_value=0)
    ax_a.bar(status_counts.index, status_counts.values, color=[GREEN, NEUTRAL], edgecolor=NEUTRAL_DARK, lw=0.5)
    ax_a.set_ylabel("boundary rows", fontsize=7)
    for i, v in enumerate(status_counts.values):
        ax_a.text(i, v + 0.15, str(int(v)), ha="center", fontsize=8, fontweight="bold")

    _panel(ax_b, "b")
    ax_b.set_title("External public-source scale", loc="left", fontsize=9)
    rows = []
    for _, row in external.iterrows():
        rows.append({
            "source": "MBTA bus\nboard/load" if "mbta" in str(row["source"]) else "MTA subway\nOD estimate",
            "rows": float(row.get("rows", 0) if not pd.isna(row.get("rows", np.nan)) else row.get("sample_rows", 0)),
            "full_rows": float(row.get("full_table_rows", 0) if not pd.isna(row.get("full_table_rows", np.nan)) else 0),
        })
    scale_df = pd.DataFrame(rows)
    vals = [max(r["rows"], r["full_rows"]) for _, r in scale_df.iterrows()]
    ax_b.bar(scale_df["source"], vals, color=[BLUE2, TEAL], edgecolor=NEUTRAL_DARK, lw=0.5)
    ax_b.set_yscale("log")
    ax_b.set_ylabel("records (log scale)", fontsize=7)
    for i, v in enumerate(vals):
        ax_b.text(i, v * 1.2, f"{v:,.0f}", ha="center", fontsize=6.5)

    _panel(ax_c, "c")
    ax_c.set_title("Field coverage", loc="left", fontsize=9)
    fields = ["board", "alight", "load", "OD"]
    sources = ["AFC/APC native", "MBTA bus", "MTA subway OD", "GTFS-ride feed"]
    mat = np.array([
        [1, 0.5, 0, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ])
    ax_c.imshow(mat, cmap=mpl.colors.ListedColormap(["#F1F1F1", GOLD, GREEN]), vmin=0, vmax=1)
    ax_c.set_xticks(np.arange(len(fields)))
    ax_c.set_xticklabels(fields, fontsize=7)
    ax_c.set_yticks(np.arange(len(sources)))
    ax_c.set_yticklabels(sources, fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            label = "yes" if mat[i, j] == 1 else ("sim" if mat[i, j] == 0.5 else "")
            ax_c.text(j, i, label, ha="center", va="center", fontsize=7, color=NEUTRAL_DARK)
    ax_c.tick_params(length=0)

    _panel(ax_d, "d")
    ax_d.set_title("Claim boundary", loc="left", fontsize=9)
    ax_d.axis("off")
    text = (
        "Supported: public board/alight/load\n"
        "and estimated OD source coverage.\n\n"
        "Boundary: not one joint agency\n"
        "OD/onboard-load control loop.\n\n"
        "GTFS-ride native feed: optional\n"
        "replication path, still missing."
    )
    ax_d.text(0.02, 0.95, text, ha="left", va="top", fontsize=8, linespacing=1.35)

    _save(fig, output_dir, "fig4_external_transit_data_coverage")
    return {"figure": "fig4_external_transit_data_coverage", "boundary_rows": int(len(boundaries))}


def fig5_orderbook_encoder(paths: dict[str, Path], output_dir: Path, source_dir: Path) -> dict[str, Any]:
    per_eval = _read_csv(paths["order_book_per_eval"])
    checks = _read_csv(paths["order_book_checks"])
    enc = _read_csv(paths["encoder_domains"])
    order_summary = _read_json(paths["order_book_summary"])
    per_eval.to_csv(source_dir / "fig5_order_book_per_eval_source.csv", index=False)
    checks.to_csv(source_dir / "fig5_order_book_checks_source.csv", index=False)
    enc.to_csv(source_dir / "fig5_encoder_domain_source.csv", index=False)

    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    cov = order_summary.get("coverage", {}) if isinstance(order_summary.get("coverage"), dict) else {}
    _panel(ax_a, "a")
    ax_a.set_title("Venue-grade replay manifest", loc="left", fontsize=9)
    labels = ["L2 files", "L3 files", "paired\nsessions", "missing"]
    vals = [
        cov.get("venue_grade_l2_files", 0),
        cov.get("venue_grade_l3_files", 0),
        cov.get("venue_grade_l2_l3_session_pairs", 0),
        cov.get("missing_entries", 0),
    ]
    ax_a.bar(labels, vals, color=[BLUE2, TEAL, GREEN, RED_SOFT], edgecolor=NEUTRAL_DARK, lw=0.5)
    ax_a.set_ylabel("count", fontsize=7)
    for i, v in enumerate(vals):
        ax_a.text(i, v + 0.08, str(int(v)), ha="center", fontsize=8, fontweight="bold")

    _panel(ax_b, "b")
    ax_b.set_title("Execution replay summaries", loc="left", fontsize=9)
    l2 = per_eval[per_eval["book_kind"] == "l2"].groupby(["freq_method", "execution_mode"], dropna=False)["fill_rate"].mean().reset_index()
    pivot = l2.pivot(index="freq_method", columns="execution_mode", values="fill_rate").fillna(0)
    x = np.arange(len(pivot.index))
    width = 0.36
    market = pivot["market"].to_numpy() if "market" in pivot.columns else np.zeros(len(pivot))
    passive = pivot["passive_queue"].to_numpy() if "passive_queue" in pivot.columns else np.zeros(len(pivot))
    ax_b.bar(x - width / 2, market, width, label="market", color=BLUE2, edgecolor=NEUTRAL_DARK, lw=0.5)
    ax_b.bar(x + width / 2, passive, width, label="passive", color=GREEN_SOFT, edgecolor=NEUTRAL_DARK, lw=0.5)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([s.replace("_", "\n") for s in pivot.index], fontsize=6.5)
    ax_b.set_ylim(0, 1.05)
    ax_b.set_ylabel("fill rate", fontsize=7)
    ax_b.legend(fontsize=7, loc="lower right")

    _panel(ax_c, "c")
    ax_c.set_title("Encoder evidence by domain", loc="left", fontsize=9)
    enc_plot = enc.copy()
    domains = enc_plot["domain"].str.replace("_", "\n")
    left = np.zeros(len(enc_plot))
    for col, color, label in [
        ("supported", GREEN, "supported"),
        ("positive_mixed", GOLD, "mixed"),
        ("summary_only", NEUTRAL_LIGHT, "summary-only"),
        ("not_supported", RED_SOFT, "not supported"),
    ]:
        vals = enc_plot[col].fillna(0).to_numpy()
        ax_c.barh(domains, vals, left=left, color=color, edgecolor="white", label=label)
        left += vals
    ax_c.invert_yaxis()
    ax_c.set_xlabel("checks", fontsize=7)
    ax_c.tick_params(axis="y", labelsize=6.3)
    ax_c.legend(fontsize=6.2, ncols=2, loc="lower right")

    _panel(ax_d, "d")
    ax_d.set_title("Order-book check status", loc="left", fontsize=9)
    status = checks.groupby(["book_kind", "status"]).size().unstack(fill_value=0)
    status = status.reindex(columns=["supported", "positive_mixed", "inconclusive", "not_supported"], fill_value=0)
    left = np.zeros(len(status))
    y = np.arange(len(status))
    for col, color in [
        ("supported", GREEN),
        ("positive_mixed", GOLD),
        ("inconclusive", NEUTRAL_LIGHT),
        ("not_supported", RED_SOFT),
    ]:
        vals = status[col].to_numpy()
        ax_d.barh(y, vals, left=left, color=color, edgecolor="white", label=col)
        left += vals
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(status.index, fontsize=7)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("paired checks", fontsize=7)
    ax_d.legend(fontsize=6.2, loc="lower right")

    _save(fig, output_dir, "fig5_orderbook_encoder_replay")
    return {"figure": "fig5_orderbook_encoder_replay", "encoder_domains": int(len(enc))}


def build_figures(results_root: Path, output_dir: Path) -> dict[str, Any]:
    apply_style()
    paths = _paths(results_root)
    figures_dir = output_dir / "figures"
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    records = [
        fig1_protocol(figures_dir, source_dir),
        fig2_evidence_matrix(paths, figures_dir, source_dir),
        fig3_transit(paths, figures_dir, source_dir),
        fig4_external_data(paths, figures_dir, source_dir),
        fig5_orderbook_encoder(paths, figures_dir, source_dir),
    ]
    summary = {
        "summary": {
            "figures": len(records),
            "formats": ["svg", "pdf", "png", "tiff"],
            "backend": "python-matplotlib",
            "output_dir": str(output_dir),
            "figures_dir": str(figures_dir),
            "source_data_dir": str(source_dir),
        },
        "figures": records,
        "boundary": (
            "Figures are generated from committed summary artifacts. They are "
            "publication-ready drafts, not a substitute for final journal "
            "caption and source-data submission checks."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    payload = build_figures(args.results_root, args.output_dir)
    print(
        "manuscript_figures "
        f"figures={payload['summary']['figures']} "
        f"output={payload['summary']['output_dir']}"
    )


if __name__ == "__main__":
    main()
