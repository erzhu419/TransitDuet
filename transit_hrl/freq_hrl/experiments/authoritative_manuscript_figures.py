"""Build publication figures from the authoritative Freq-HRL registry only.

Python/matplotlib is the selected and exclusive rendering backend. The package
fails closed if the reportable registry membership changes or if a selected
development record is no longer explicitly development-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from freq_hrl.experiments.authoritative_evidence_registry import (
    DEFAULT_REGISTRY,
    DEFAULT_REPOSITORY_ROOT,
    load_registry,
    validate_registry,
)


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "axes.titleweight": "bold",
})


SCHEMA_VERSION = "freq_hrl_authoritative_manuscript_figures_v1"
DEFAULT_OUTPUT_DIR = Path(
    "transit_hrl/results/authoritative_paper_figures_latest"
)

REPORTABLE_EVIDENCE_IDS = {
    "mujoco_v12_responsibility_confirmatory",
    "mujoco_v13_behavioral_confirmatory",
    "mujoco_v14_29_restoration_portfolio_confirmatory",
    "quant_v74_matched_baseline_confirmatory",
}
DEVELOPMENT_EVIDENCE_IDS = (
    "mujoco_v17_6_full_horizon_oracle_development",
    "mujoco_v17_11_fractional_reservoir_fir_development",
    "mujoco_v17_14_exhaustive_actor_oracle_development",
    "mujoco_v18_2_state_conditioned_actor_development",
    "mujoco_v18_3_causal_joint_projection_development",
    "mujoco_v18_4_receding_joint_projection_development",
    "mujoco_v18_5_actor_floor_signal_development",
)
LEGACY_TOKENS = (
    "top_journal_unified_matrix_latest",
    "freq_hrl_paper_diagnostics",
    "manuscript_figures_latest",
)

INK = "#303038"
NEUTRAL = "#707078"
NEUTRAL_LIGHT = "#D8D8D8"
BASELINE_DARK = "#484878"
BASELINE_MID = "#7884B4"
BASELINE_SOFT = "#B4C0E4"
OURS_BASE = "#E4CCD8"
OURS_STRONG = "#C77C9B"
BG_LILAC = "#E8E8F4"
BG_AQUA = "#E4F2F2"
BG_PEACH = "#F3E5D8"
IMPROVEMENT = "#2C6E9E"
HARM = "#C44E52"
INCONCLUSIVE = "#8A8A8A"


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_figure_evidence(
    *,
    repository_root: Path = DEFAULT_REPOSITORY_ROOT,
    registry_path: Path = DEFAULT_REGISTRY,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(repository_root).resolve()
    registry_file = _resolve(root, Path(registry_path))
    registry = load_registry(registry_file)
    records = validate_registry(registry, root)
    validate_figure_boundary(records)
    return registry, records


def validate_figure_boundary(records: list[dict[str, Any]]) -> None:
    by_id = {row["evidence_id"]: row for row in records}
    actual_reportable = {
        row["evidence_id"] for row in records if row["manuscript_reportable"]
    }
    if actual_reportable != REPORTABLE_EVIDENCE_IDS:
        raise ValueError(
            "reportable evidence membership changed; revise the figure contract"
        )
    missing_development = set(DEVELOPMENT_EVIDENCE_IDS) - set(by_id)
    if missing_development:
        raise ValueError(
            f"development stop-map evidence is missing: {sorted(missing_development)}"
        )
    for evidence_id in DEVELOPMENT_EVIDENCE_IDS:
        row = by_id[evidence_id]
        if (
            row["evidence_stage"] != "development"
            or row["paper_disposition"] != "development_only"
            or row["manuscript_reportable"]
        ):
            raise ValueError(f"{evidence_id}: development boundary changed")


def _by_id(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["evidence_id"]: row for row in records}


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": row["evidence_id"],
        "evidence_stage": row["evidence_stage"],
        "paper_disposition": row["paper_disposition"],
        "manuscript_reportable": str(bool(row["manuscript_reportable"])).lower(),
    }


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    phat = successes / total
    denominator = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        phat * (1.0 - phat) / total + z * z / (4.0 * total * total)
    ) / denominator
    return center - half, center + half


def build_source_rows(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    validate_figure_boundary(records)
    rows = _by_id(records)

    fig1 = [
        {"panel": "a", "element": "upper policy", "estimand": "slow raw action effect"},
        {"panel": "a", "element": "lower policy", "estimand": "fast raw action effect"},
        {"panel": "a", "element": "domain adapter", "estimand": "total executed action effect"},
        {"panel": "b", "element": "raw audit", "estimand": "raw upper/lower component spectra"},
        {"panel": "b", "element": "canonical audit", "estimand": "gauge-fixed responsibility spectra"},
        {"panel": "c", "element": "router candidate", "estimand": "trace-invariant responsibility repair"},
        {"panel": "c", "element": "actor candidate", "estimand": "guarded physical action correction"},
        {"panel": "c", "element": "abstention", "estimand": "counted failure when no candidate passes"},
    ]

    v12 = rows["mujoco_v12_responsibility_confirmatory"]
    v13 = rows["mujoco_v13_behavioral_confirmatory"]
    v14 = rows["mujoco_v14_29_restoration_portfolio_confirmatory"]
    fig2: list[dict[str, Any]] = []
    for environment in v12["facts"]["environments"]:
        fig2.append({
            **_metadata(v12),
            "panel": "a",
            "row_type": "responsibility_drift_reduction",
            "environment": environment["environment"],
            "metric": "responsibility_drift_reduction",
            "estimate": environment["responsibility_drift_reduction"],
            "ci95_lower": environment["familywise_drift_reduction_lower"],
            "ci95_upper": "",
            "status": "supported",
            "independent_optimizer_replicates": v12["facts"][
                "independent_optimizer_replicates_per_arm"
            ],
            "heldout_paths_per_cell": v12["facts"]["heldout_paths_per_cell"],
        })
    gate_order = (
        "return_noninferiority",
        "responsibility_drift",
        "raw_lower_drift",
        "upper_hf_budget",
    )
    for environment in v13["facts"]["environments"]:
        failed = set(environment["failed_gates"])
        for gate in gate_order:
            fig2.append({
                **_metadata(v13),
                "panel": "b",
                "row_type": "behavioral_gate",
                "environment": environment["environment"],
                "metric": gate,
                "estimate": "",
                "ci95_lower": "",
                "ci95_upper": "",
                "status": "fail" if gate in failed else "pass",
                "independent_optimizer_replicates": v13["facts"][
                    "independent_optimizer_replicates_per_arm"
                ],
                "heldout_paths_per_cell": v13["facts"]["heldout_paths_per_cell"],
            })
    for environment, successes in v14["facts"][
        "supported_count_by_environment"
    ].items():
        total = v14["facts"]["optimizer_seed_count_per_environment"]
        lower, upper = _wilson_interval(successes, total)
        registered_lower = v14["facts"]["wilson_lower_by_environment"][environment]
        if not math.isclose(lower, registered_lower, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{environment}: Wilson interval drifted")
        fig2.append({
            **_metadata(v14),
            "panel": "c",
            "row_type": "portfolio_support_rate",
            "environment": environment,
            "metric": "supported_seed_rate",
            "estimate": successes / total,
            "ci95_lower": lower,
            "ci95_upper": upper,
            "status": "supported",
            "independent_optimizer_replicates": total,
            "heldout_paths_per_cell": "nested_frozen_validation_panel",
        })
    for mechanism, count in (
        ("function-preserving router", v14["facts"]["router_selection_count"]),
        ("guarded actor update", v14["facts"]["actor_selection_count"]),
        ("abstention", v14["facts"]["abstention_count"]),
    ):
        fig2.append({
            **_metadata(v14),
            "panel": "d",
            "row_type": "portfolio_mechanism_count",
            "environment": "all",
            "metric": mechanism,
            "estimate": count,
            "ci95_lower": "",
            "ci95_upper": "",
            "status": "count",
            "independent_optimizer_replicates": v14["facts"]["cell_count"],
            "heldout_paths_per_cell": "nested_frozen_validation_panel",
        })

    quant = rows["quant_v74_matched_baseline_confirmatory"]
    fig3 = []
    for primary in quant["facts"]["primary_rows"]:
        fig3.append({
            **_metadata(quant),
            "panel": "a" if primary["metric"] == "total_return" else "b",
            "comparator": primary["comparator"],
            "metric": primary["metric"],
            "directional_improvement_mean": primary[
                "directional_improvement_mean"
            ],
            "ci95_lower": primary["ci95"][0],
            "ci95_upper": primary["ci95"][1],
            "effect_size_dz": primary["effect_size_dz"],
            "holm_p": primary["holm_p"],
            "status": primary["status"],
            "independent_training_replicates": quant["facts"][
                "independent_training_replicates"
            ],
            "heldout_paths_per_replicate": quant["facts"][
                "heldout_paths_per_replicate"
            ],
            "scenario_count": quant["facts"]["scenario_count"],
        })

    development_specs = [
        (
            "mujoco_v17_6_full_horizon_oracle_development",
            "v17.6 acausal oracle",
            "81/88 online failures recoverable; 7 actor-floor paths",
            "diagnosis only: acausal reused paths",
        ),
        (
            "mujoco_v17_11_fractional_reservoir_fir_development",
            "v17.11 causal router",
            "62/81 recovered; Hopper 14/33",
            "router-only FIR line closed",
        ),
        (
            "mujoco_v17_14_exhaustive_actor_oracle_development",
            "v17.14 linear actor FIR",
            "6/7 actor-floor paths; 113/113 feasible preserved",
            "one Hopper ood_chirp unresolved; grid closed",
        ),
        (
            "mujoco_v18_2_state_conditioned_actor_development",
            "v18.2 state MLP",
            "3/7 actor-floor paths",
            "worse than frozen linear FIR",
        ),
        (
            "mujoco_v18_3_causal_joint_projection_development",
            "v18.3 instant projector",
            "7/7 recovered; 120/120 directly feasible",
            "trust failed: RMS 0.2935, abs 1.8076",
        ),
        (
            "mujoco_v18_4_receding_joint_projection_development",
            "v18.4 receding projector",
            "69/120 direct vs 120/120 offline-exact",
            "40,962 prefix violations; no terminal certificate",
        ),
        (
            "mujoco_v18_5_actor_floor_signal_development",
            "v18.5 floor score",
            "5/7 actor-floor paths in global top 14",
            "0 eligible scores; no feedback screen",
        ),
    ]
    fig_s1 = []
    for order, (evidence_id, label, outcome, stop_reason) in enumerate(
        development_specs, start=1
    ):
        row = rows[evidence_id]
        fresh = bool(row["facts"].get("fresh_validation_paths_accessed", False))
        if fresh:
            raise ValueError(f"{evidence_id}: unexpected fresh validation access")
        fig_s1.append({
            **_metadata(row),
            "order": order,
            "label": label,
            "outcome": outcome,
            "stop_reason": stop_reason,
            "path_count": row["facts"].get("path_count", 120),
            "selection_access": row["selection_access"],
            "fresh_validation_paths_accessed": "false",
        })

    return {
        "fig1_protocol_source.csv": fig1,
        "fig2_mujoco_confirmatory_source.csv": fig2,
        "fig3_quant_contrasts_source.csv": fig3,
        "fig_s1_development_stop_map_source.csv": fig_s1,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty source-data file: {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
    )


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = INK,
    fontsize: float = 6.5,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.8,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        transform=ax.transAxes,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = NEUTRAL,
) -> None:
    ax.add_patch(FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=0.9,
        color=color,
        transform=ax.transAxes,
    ))


def _save_figure(fig: plt.Figure, figures_dir: Path, name: str) -> list[str]:
    outputs = []
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 400}),
    ):
        path = figures_dir / f"{name}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs.append(str(Path("figures") / path.name))
    plt.close(fig)
    return outputs


def _figure_1(figures_dir: Path) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.15), constrained_layout=False)
    fig.subplots_adjust(left=0.035, right=0.99, top=0.82, bottom=0.14, wspace=0.30)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    _panel_label(ax, "a")
    ax.set_title("Two-rate policy\none physical effect", loc="left", fontsize=7.5)
    _box(ax, (0.03, 0.64), 0.29, 0.17, "upper policy\n$u_m$ every $K$ steps", facecolor=BG_LILAC, fontsize=5.9)
    _box(ax, (0.03, 0.24), 0.29, 0.17, "lower policy\n$l_t$ every step", facecolor=BG_AQUA, fontsize=5.9)
    _box(ax, (0.48, 0.43), 0.24, 0.20, "domain\nadapter", facecolor=BG_PEACH)
    _box(ax, (0.80, 0.43), 0.17, 0.20, "total\n$a_t$", facecolor=OURS_BASE)
    _arrow(ax, (0.32, 0.72), (0.48, 0.57))
    _arrow(ax, (0.32, 0.32), (0.48, 0.49))
    _arrow(ax, (0.72, 0.53), (0.80, 0.53))
    ax.text(0.05, 0.08, "Clock separation is not spectral ownership.", fontsize=5.8, color=NEUTRAL, transform=ax.transAxes)

    ax = axes[1]
    _panel_label(ax, "b")
    ax.set_title("Raw versus canonical\nresponsibility", loc="left", fontsize=7.5)
    _box(ax, (0.03, 0.72), 0.25, 0.14, "raw $e^U,e^L$", facecolor=BG_PEACH)
    _box(ax, (0.38, 0.72), 0.56, 0.14, "$D_L^{raw}$ and $P_U^{raw}$", facecolor=BG_PEACH)
    _arrow(ax, (0.28, 0.79), (0.38, 0.79))
    _box(ax, (0.03, 0.34), 0.25, 0.14, "total $a_t$", facecolor=OURS_BASE)
    _box(ax, (0.38, 0.34), 0.25, 0.14, "$P(a), a-P(a)$", facecolor=BG_LILAC, fontsize=5.7)
    _box(ax, (0.73, 0.34), 0.21, 0.14, "$D_L^{resp}$", facecolor=BG_AQUA)
    _arrow(ax, (0.28, 0.41), (0.38, 0.41))
    _arrow(ax, (0.63, 0.41), (0.73, 0.41))
    ax.text(0.05, 0.10, "A gauge repair may preserve all actions.", fontsize=5.8, color=NEUTRAL, transform=ax.transAxes)

    ax = axes[2]
    _panel_label(ax, "c")
    ax.set_title("Guarded restoration\ncan abstain", loc="left", fontsize=7.5)
    _box(ax, (0.02, 0.66), 0.27, 0.14, "trace-invariant\nrouters", facecolor=BG_LILAC)
    _box(ax, (0.02, 0.34), 0.27, 0.14, "guarded actor\nupdates", facecolor=OURS_BASE)
    _box(ax, (0.40, 0.47), 0.26, 0.20, "design-fold gates\nfrequency\n+ reward floor", facecolor=BG_PEACH, fontsize=5.8)
    _box(ax, (0.77, 0.62), 0.20, 0.14, "select", facecolor=BG_AQUA)
    _box(ax, (0.77, 0.30), 0.20, 0.14, "abstain", facecolor=NEUTRAL_LIGHT)
    _arrow(ax, (0.29, 0.73), (0.40, 0.60))
    _arrow(ax, (0.29, 0.41), (0.40, 0.54))
    _arrow(ax, (0.66, 0.58), (0.77, 0.69))
    _arrow(ax, (0.66, 0.54), (0.77, 0.37))
    ax.text(0.04, 0.10, "No passing candidate means abstention.", fontsize=5.8, color=NEUTRAL, transform=ax.transAxes)

    return _save_figure(fig, figures_dir, "fig1_protocol_and_estimands")


def _figure_2(
    figures_dir: Path, source_rows: list[dict[str, Any]]
) -> list[str]:
    fig = plt.figure(figsize=(7.2, 3.15), constrained_layout=True)
    grid = fig.add_gridspec(
        2, 3, width_ratios=[1.05, 1.35, 1.10], height_ratios=[2.8, 1.1]
    )
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[:, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    ax_d = fig.add_subplot(grid[1, 2])

    environments = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
    short = {"HalfCheetah-v5": "HalfCheetah", "Hopper-v5": "Hopper", "Walker2d-v5": "Walker2d"}

    rows_a = {
        row["environment"]: row
        for row in source_rows
        if row["panel"] == "a"
    }
    values = [100.0 * float(rows_a[name]["estimate"]) for name in environments]
    lowers = [100.0 * float(rows_a[name]["ci95_lower"]) for name in environments]
    x = np.arange(len(environments))
    ax_a.bar(x, values, color=BASELINE_MID, edgecolor=INK, linewidth=0.6, width=0.62)
    for index, (value, lower) in enumerate(zip(values, lowers)):
        ax_a.plot([index - 0.22, index + 0.22], [lower, lower], color=INK, linewidth=1.2)
        ax_a.text(index, value + 2.2, f"{value:.1f}%", ha="center", va="bottom", fontsize=6.2)
    ax_a.set_ylim(0, 102)
    ax_a.set_ylabel("Responsibility-drift reduction (%)")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([short[name] for name in environments], rotation=25, ha="right")
    ax_a.set_title("v12 responsibility result", loc="left", fontsize=8)
    _panel_label(ax_a, "a")

    rows_b = [row for row in source_rows if row["panel"] == "b"]
    gates = [
        ("return_noninferiority", "Return\nNI"),
        ("responsibility_drift", "Resp.\ndrift"),
        ("raw_lower_drift", "Raw lower\ndrift"),
        ("upper_hf_budget", "Upper HF\nbudget"),
    ]
    lookup = {(row["environment"], row["metric"]): row["status"] for row in rows_b}
    ax_b.set_xlim(-0.5, len(gates) - 0.5)
    ax_b.set_ylim(-0.5, len(environments) - 0.5)
    for yi, environment in enumerate(environments):
        for xi, (gate, _) in enumerate(gates):
            status = lookup[(environment, gate)]
            color = IMPROVEMENT if status == "pass" else HARM
            ax_b.add_patch(Rectangle((xi - 0.42, yi - 0.36), 0.84, 0.72, facecolor=color, edgecolor="white", linewidth=1.0))
            ax_b.text(xi, yi, status.upper(), ha="center", va="center", color="white", fontweight="bold", fontsize=5.8)
    ax_b.set_xticks(range(len(gates)))
    ax_b.set_xticklabels([label for _, label in gates])
    ax_b.set_yticks(range(len(environments)))
    ax_b.set_yticklabels([short[name] for name in environments])
    ax_b.invert_yaxis()
    ax_b.tick_params(length=0)
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    ax_b.set_title("v13 stronger raw claim", loc="left", fontsize=8)
    _panel_label(ax_b, "b")

    rows_c = {
        row["environment"]: row
        for row in source_rows
        if row["panel"] == "c"
    }
    rates = np.array([float(rows_c[name]["estimate"]) for name in environments])
    lower = np.array([float(rows_c[name]["ci95_lower"]) for name in environments])
    upper = np.array([float(rows_c[name]["ci95_upper"]) for name in environments])
    ax_c.errorbar(
        x,
        rates,
        yerr=np.vstack([rates - lower, upper - rates]),
        fmt="o",
        color=BASELINE_DARK,
        ecolor=BASELINE_MID,
        elinewidth=1.2,
        capsize=3,
        markersize=4.5,
    )
    for index, name in enumerate(environments):
        successes = round(rates[index] * 16)
        ax_c.text(index, min(1.055, rates[index] + 0.035), f"{successes}/16", ha="center", fontsize=6.1)
    ax_c.set_ylim(0.65, 1.075)
    ax_c.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax_c.set_ylabel("Supported seed rate")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([short[name] for name in environments], rotation=25, ha="right")
    ax_c.set_title("v14.29 frozen portfolio", loc="left", fontsize=8)
    _panel_label(ax_c, "c")

    rows_d = [row for row in source_rows if row["panel"] == "d"]
    counts = [int(row["estimate"]) for row in rows_d]
    labels = [row["metric"] for row in rows_d]
    colors = [BASELINE_DARK, OURS_STRONG, NEUTRAL_LIGHT]
    left = 0
    for count, label, color in zip(counts, labels, colors):
        ax_d.barh([0], [count], left=left, color=color, edgecolor="white", height=0.52)
        text_color = "white" if color != NEUTRAL_LIGHT else INK
        if count >= 5:
            ax_d.text(left + count / 2, 0, str(count), ha="center", va="center", color=text_color, fontweight="bold", fontsize=6)
        else:
            ax_d.text(left + count + 0.6, 0, str(count), ha="left", va="center", color=INK, fontsize=6)
        left += count
    ax_d.set_xlim(0, 52)
    ax_d.set_yticks([])
    ax_d.set_xticks([])
    for spine in ax_d.spines.values():
        spine.set_visible(False)
    ax_d.set_title("Selected transactions", loc="left", fontsize=7)
    ax_d.text(
        0.0,
        -0.58,
        "router 38  |  actor 9  |  abstain 1",
        transform=ax_d.transAxes,
        ha="left",
        va="center",
        fontsize=5.6,
        color=NEUTRAL,
    )
    _panel_label(ax_d, "d")

    fig.suptitle(
        "Confirmatory MuJoCo evidence separates responsibility success from raw-behavior failure",
        fontsize=9,
        fontweight="bold",
    )
    return _save_figure(fig, figures_dir, "fig2_mujoco_confirmatory_evidence")


def _figure_3(
    figures_dir: Path, source_rows: list[dict[str, Any]]
) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), constrained_layout=False)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.82, bottom=0.26, wspace=0.58)
    comparator_order = [
        "flat_ppo_matched_v7",
        "flat_gru_ppo_matched_v7",
        "generic_hrl_ppo_matched_v7",
        "generic_hrl_gru_ppo_matched_v7",
        "flat_sac_matched_v7",
        "flat_td3_matched_v7",
    ]
    comparator_labels = {
        "flat_ppo_matched_v7": "Flat PPO",
        "flat_gru_ppo_matched_v7": "Flat GRU-PPO",
        "generic_hrl_ppo_matched_v7": "Generic HRL-PPO",
        "generic_hrl_gru_ppo_matched_v7": "Generic HRL-GRU-PPO",
        "flat_sac_matched_v7": "Flat SAC",
        "flat_td3_matched_v7": "Flat TD3",
    }
    metrics = [
        ("total_return", "Directional return difference", "a"),
        ("LowerLFDriftAbs", "Directional lower-LF improvement", "b"),
    ]
    status_style = {
        "supported_improvement": (IMPROVEMENT, "o"),
        "supported_harm": (HARM, "X"),
        "inconclusive": (INCONCLUSIVE, "D"),
    }
    for ax, (metric, title, panel) in zip(axes, metrics):
        metric_rows = {
            row["comparator"]: row
            for row in source_rows
            if row["metric"] == metric
        }
        ordered = [metric_rows[name] for name in comparator_order]
        max_abs = max(
            abs(float(row[key]))
            for row in ordered
            for key in ("ci95_lower", "ci95_upper")
        )
        margin = max_abs * 1.18
        for yi, row in enumerate(reversed(ordered)):
            mean = float(row["directional_improvement_mean"])
            low = float(row["ci95_lower"])
            high = float(row["ci95_upper"])
            color, marker = status_style[row["status"]]
            ax.plot([low, high], [yi, yi], color=color, linewidth=1.5, solid_capstyle="round")
            ax.scatter([mean], [yi], color=color, marker=marker, s=28, zorder=3, edgecolor="white", linewidth=0.4)
        ax.axvline(0.0, color=INK, linewidth=0.8, linestyle="--")
        ax.set_xlim(-margin, margin)
        ax.set_yticks(range(len(comparator_order)))
        ax.set_yticklabels([comparator_labels[name] for name in reversed(comparator_order)])
        ax.tick_params(axis="both", labelsize=6.2)
        ax.set_xlabel(r"Freq-HRL better $\rightarrow$")
        ax.set_title(title, loc="left", fontsize=8)
        ax.grid(axis="x", color="#ECECF0", linewidth=0.6)
        _panel_label(ax, panel)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=IMPROVEMENT, markeredgecolor="white", label="supported improvement"),
        Line2D([0], [0], marker="X", linestyle="none", markerfacecolor=HARM, markeredgecolor="white", label="supported harm"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=INCONCLUSIVE, markeredgecolor="white", label="inconclusive"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.065), fontsize=6.2)
    fig.suptitle(
        "Quant v7.4: all 12 Holm-controlled matched-baseline contrasts",
        fontsize=9,
        fontweight="bold",
    )
    fig.text(0.5, 0.025, "24 independent training replicates; eight held-out paths per replicate; six scenarios", ha="center", fontsize=6.2, color=NEUTRAL)
    return _save_figure(fig, figures_dir, "fig3_quant_matched_baseline_forest")


def _figure_s1(
    figures_dir: Path, source_rows: list[dict[str, Any]]
) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.2, 4.15), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0.02, 0.89), 0.96, 0.08, facecolor=HARM, edgecolor="none"))
    ax.text(
        0.5,
        0.93,
        "DEVELOPMENT ONLY  |  unchanged 120-path panel  |  fresh validation paths accessed: 0",
        color="white",
        ha="center",
        va="center",
        fontsize=7.4,
        fontweight="bold",
    )

    positions = [
        (0.03, 0.58),
        (0.27, 0.58),
        (0.51, 0.58),
        (0.75, 0.58),
        (0.12, 0.17),
        (0.39, 0.17),
        (0.66, 0.17),
    ]
    widths = [0.21, 0.21, 0.21, 0.21, 0.24, 0.24, 0.24]
    for index, (row, xy, width) in enumerate(zip(source_rows, positions, widths)):
        color = BG_LILAC if index == 0 else BG_PEACH
        _box(
            ax,
            xy,
            width,
            0.25,
            f"{row['label']}\n\n{textwrap.fill(row['outcome'], 27)}\n\nSTOP: {textwrap.fill(row['stop_reason'], 27)}",
            facecolor=color,
            fontsize=5.7,
        )
    for left, right in ((0, 1), (1, 2), (2, 3), (4, 5), (5, 6)):
        start_xy, start_width = positions[left], widths[left]
        end_xy = positions[right]
        _arrow(
            ax,
            (start_xy[0] + start_width, start_xy[1] + 0.125),
            (end_xy[0], end_xy[1] + 0.125),
        )
    _arrow(ax, (0.855, 0.58), (0.24, 0.42))
    ax.text(
        0.5,
        0.06,
        "The route closes tested mechanisms; it does not establish fresh-seed control performance or an impossibility theorem.",
        ha="center",
        va="center",
        fontsize=6.4,
        color=NEUTRAL,
    )
    ax.set_title("Why the v17-v18 physical-correction line stopped", loc="left", fontsize=9, fontweight="bold")
    return _save_figure(fig, figures_dir, "fig_s1_development_stop_map")


def _preview_montage(figures_dir: Path, output_dir: Path) -> str:
    names = [
        "fig1_protocol_and_estimands",
        "fig2_mujoco_confirmatory_evidence",
        "fig3_quant_matched_baseline_forest",
        "fig_s1_development_stop_map",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, name in zip(axes.flat, names):
        image = plt.imread(figures_dir / f"{name}.png")
        ax.imshow(image)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    path = output_dir / "figures_preview_montage.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path.name


def build_authoritative_manuscript_figures(
    *,
    repository_root: Path = DEFAULT_REPOSITORY_ROOT,
    registry_path: Path = DEFAULT_REGISTRY,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = _resolve(root, Path(output_dir))
    figures_dir = output / "figures"
    source_dir = output / "source_data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    registry, records = load_figure_evidence(
        repository_root=root,
        registry_path=registry_path,
    )
    source_rows = build_source_rows(records)
    for filename, rows in source_rows.items():
        _write_csv(source_dir / filename, rows)

    figure_outputs = {
        "fig1_protocol_and_estimands": _figure_1(figures_dir),
        "fig2_mujoco_confirmatory_evidence": _figure_2(
            figures_dir, source_rows["fig2_mujoco_confirmatory_source.csv"]
        ),
        "fig3_quant_matched_baseline_forest": _figure_3(
            figures_dir, source_rows["fig3_quant_contrasts_source.csv"]
        ),
        "fig_s1_development_stop_map": _figure_s1(
            figures_dir,
            source_rows["fig_s1_development_stop_map_source.csv"],
        ),
    }
    montage = _preview_montage(figures_dir, output)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "backend": "python_matplotlib",
        "registry_snapshot_date": registry["snapshot_date"],
        "reportable_evidence_ids": sorted(REPORTABLE_EVIDENCE_IDS),
        "development_evidence_ids": list(DEVELOPMENT_EVIDENCE_IDS),
        "main_figure_evidence_policy": "manuscript_reportable_only",
        "supplementary_development_policy": "development_only_reused_paths_explicit",
        "figures": figure_outputs,
        "source_data": [str(Path("source_data") / name) for name in source_rows],
        "preview_montage": montage,
        "export_note": "SVG and PDF are editable submission sources; PNG is a review preview. TIFF is deferred until required by a selected venue.",
    }
    summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if any(token in summary_text for token in LEGACY_TOKENS):
        raise ValueError("legacy figure source leaked into the authoritative package")
    (output / "summary.json").write_text(summary_text, encoding="utf-8")
    (output / "qa_notes.md").write_text(
        "# Figure QA Notes\n\n"
        "- Backend: Python/matplotlib only.\n"
        "- Final width: at most 7.2 inches (183 mm).\n"
        "- SVG text remains editable; PDF uses TrueType text.\n"
        "- Figures 2 and 3 use only manuscript-reportable registry records.\n"
        "- Supplementary Figure 1 is visibly development-only and records zero fresh validation access.\n"
        "- Color is redundant with PASS/FAIL text or distinct markers.\n"
        "- Source-data CSV accompanies every figure.\n"
        "- Visual overlap and final-size readability require manual inspection of the montage before submission.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=DEFAULT_REPOSITORY_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = build_authoritative_manuscript_figures(
        repository_root=args.repository_root,
        registry_path=args.registry,
        output_dir=args.output_dir,
    )
    print(
        "authoritative_manuscript_figures "
        f"snapshot={summary['registry_snapshot_date']} "
        f"main_records={len(summary['reportable_evidence_ids'])} "
        f"development_records={len(summary['development_evidence_ids'])}"
    )


if __name__ == "__main__":
    main()
