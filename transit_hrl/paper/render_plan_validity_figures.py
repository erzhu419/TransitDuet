"""Render the five manuscript figures from completed local evidence only."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.text import Text
import numpy as np

PAPER = Path(__file__).resolve().parent
OUTPUT = PAPER / "figures"
EVIDENCE = "plan_validity_evidence.json"
TEAL, BLUE, GREY, ORANGE = "#167b80", "#527aa3", "#828a94", "#b77431"
INK, LIGHT = "#252d35", "#eef2f4"
MODES = ("fixed", "balanced_jitter", "current_compact_quadratic", "causal_validity_interactions")
LABELS = ("Periodic", "Random offset", "Current-only", "Causal history")
COLORS = (GREY, "#a1a7ad", BLUE, TEAL)
CSV_FIELDS = (
    "panel",
    "kind",
    "comparison",
    "metric",
    "part",
    "root",
    "value_raw",
    "ci_low_raw",
    "ci_high_raw",
    "display_scale",
    "display_unit",
    "n",
    "statistical_unit",
    "source",
)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.titleweight": "bold",
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 7,
    "text.color": INK,
    "axes.edgecolor": GREY,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "legend.frameon": False,
})


def record(
    rows,
    panel,
    kind,
    comparison,
    metric,
    value,
    *,
    source,
    ci=None,
    scale=1,
    unit="",
    n="",
    statistical_unit="",
    part="pooled",
    root="",
):
    rows.append(
        dict(
            zip(
                CSV_FIELDS,
                (
                    panel,
                    kind,
                    comparison,
                    metric,
                    part,
                    root,
                    value,
                    "" if ci is None else ci[0],
                    "" if ci is None else ci[1],
                    scale,
                    unit,
                    n,
                    statistical_unit,
                    source,
                ),
            )
        )
    )


def title(ax, letter, text):
    ax.set_title(text, loc="left", pad=9)
    ax.text(
        -0.02,
        1.13,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def baseline(ax, xlabel):
    ax.axvline(0, color=GREY, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.set_xlabel(xlabel)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=LIGHT, lw=0.7)


def legend_handles(first, second, color):
    return [
        Line2D([], [], color=color, marker="o", ms=3, label=first),
        Line2D([], [], color=color, marker="s", markerfacecolor="white", ms=3, label=second),
    ]


def save(fig, stem, rows):
    OUTPUT.mkdir(exist_ok=True)
    for ax in fig.axes:
        if ax.axison:
            low, high = sorted(ax.get_xlim())
            ax.set_xticks([tick for tick in ax.get_xticks() if low <= tick <= high])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    outside = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
            outside.append(text.get_text())
    if outside:
        raise ValueError(f"Figure text outside canvas: {stem}: {outside}")
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUTPUT / f"{stem}.{ext}", dpi=300, facecolor="white")
    with (OUTPUT / f"{stem}_source.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    fraction = float(np.mean(np.min(rgb, axis=2) < 245))
    if fraction < 0.01:
        raise ValueError(f"Blank figure: {stem}")
    report = {
        "figure": stem,
        "width_mm": round(fig.get_figwidth() * 25.4, 2),
        "height_mm": round(fig.get_figheight() * 25.4, 2),
        "source_rows": len(rows),
        "nonwhite_fraction": fraction,
        "text_within_canvas": True,
    }
    plt.close(fig)
    return report


def figure1(pm):
    fig = plt.figure(figsize=(183 / 25.4, 124 / 25.4))
    grid = fig.add_gridspec(
        2, 1, height_ratios=(0.9, 1.35), left=0.18, right=0.96, bottom=0.16, top=0.90, hspace=0.80
    )
    a, b = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    title(a, "a", "Paired supervision, then frozen deployment")
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.axis("off")
    contract = pm["policy_contract"]
    boxes = [
        (0, "Sample queries", "Causal or uniform\nsimulator paths"),
        (0.37, "Keep / renew", f"Paired {contract['branch_horizon_steps']}-step loss\nClosed-loop lower"),
        (0.74, "Fit predictor", f"{contract['candidate_features']} causal features\nFit-path q75 threshold"),
    ]
    for x, heading, body in boxes:
        a.add_patch(Rectangle((x, 0.17), 0.26, 0.71, facecolor=LIGHT, edgecolor="none"))
        a.text(x + 0.13, 0.73, heading, ha="center", va="center", fontweight="bold")
        a.text(x + 0.13, 0.41, body, ha="center", va="center", linespacing=1.5)
    for x1, x2 in ((0.265, 0.365), (0.635, 0.735)):
        a.add_patch(FancyArrowPatch((x1, 0.54), (x2, 0.54), arrowstyle="->", mutation_scale=9, color=GREY))
    a.text(0, -0.14, "Offline: paired simulator branches (event-stratified or label-free). Online: causal history only.", fontsize=6.6)
    a.text(0, -0.42, "Check every 5 steps; renew at first score >= q, or at the offset-25 deadline.", fontsize=6.6)
    source = "../" + pm["example_schedule"]["source"]
    rows = []
    for key, value in contract.items():
        record(rows, "a", "protocol", "candidate", key, value, source=source)
    title(b, "b", "Observed upper-call timing: first 300 steps")
    example = pm["example_schedule"]
    k, deadline = contract["bin_steps"], contract["max_offset_steps"]
    for start in range(0, 300, k):
        if start // k % 2 == 0:
            b.axvspan(start, start + k, color=LIGHT, zorder=0)
        b.axvline(start, color="#cbd1d6", lw=0.6, zorder=0)
        if start > 0:
            b.axvline(start + deadline, color=GREY, lw=0.65, ls=(0, (1, 3)), zorder=0)
    for i, (mode, label, color) in enumerate(zip(MODES, LABELS, COLORS)):
        y = 3 - i
        b.hlines(y, 0, 300, color="#d5dadd", lw=0.8)
        steps = [s for s in example["decision_steps"][mode] if s < 300]
        b.scatter(steps, [y] * len(steps), color=color, s=21, marker="|", linewidths=1.8, zorder=3)
        for step in steps:
            record(
                rows,
                "b",
                "observed_call",
                mode,
                "decision_step",
                step,
                source=source,
                unit="primitive step",
                n=1,
                statistical_unit="illustrative_path",
                root=example["root"],
            )
    b.set_yticks(range(4), LABELS[::-1])
    b.set_ylim(-0.5, 3.5)
    b.set_xticks(range(0, 301, 50))
    b.set_xlim(-4, 304)
    b.set_xlabel("Primitive step (bin boundaries every 50 steps)")
    b.spines[["left", "right", "top"]].set_visible(False)
    b.tick_params(axis="y", length=0)
    fig.text(
        0.18,
        0.065,
        f"Fixed example: optimizer root {example['root']}, path {example['path_seed']}; not selected for performance.",
        fontsize=6.5,
    )
    fig.text(
        0.18,
        0.025,
        "All modes: 24 calls/episode. Dotted lines: deadlines; lower actions continue at every step.",
        fontsize=6.5,
    )
    return save(fig, "fig1_budgeted_replanning", rows)


def root_forest(ax, section_key, section, names, labels, rows, panel, xlabel):
    roots = sorted(section["root_effects"])
    for y, (name, label) in enumerate(zip(names, labels)):
        item = section["contrasts"][name]
        xs = [section["root_effects"][root][name] for root in roots]
        ax.scatter(xs, y + np.linspace(-0.19, 0.19, len(roots)), color=TEAL, alpha=0.7, s=10, zorder=3)
        ax.plot(item["CI95"], [y, y], color=INK, lw=1.3, zorder=4)
        ax.scatter([item["mean"]], [y], marker="D", color=INK, s=18, zorder=5)
        source = f"{EVIDENCE}#/{section_key}/contrasts/{name}"
        record(
            rows,
            panel,
            "root_t_CI95",
            name,
            name,
            item["mean"],
            ci=item["CI95"],
            source=source,
            unit=xlabel,
            n=8,
            statistical_unit="optimizer_seed_root",
        )
        for root, value in zip(roots, xs):
            record(
                rows,
                panel,
                "root_mean",
                name,
                name,
                value,
                source=f"{EVIDENCE}#/{section_key}/root_effects/{root}/{name}",
                unit=xlabel,
                n=1,
                statistical_unit="optimizer_seed_root",
                root=root,
            )
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.6, -0.6)
    baseline(ax, xlabel)
    low, high = ax.get_xlim()
    ax.set_xlim(min(0, low) - 0.03 * (high - low), high)


def figure2(pm):
    fig = plt.figure(figsize=(183 / 25.4, 139 / 25.4))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.35, 1),
        height_ratios=(1.15, 0.85),
        left=0.20,
        right=0.97,
        bottom=0.16,
        top=0.90,
        wspace=0.57,
        hspace=0.95,
    )
    a, b, c, d = [fig.add_subplot(grid[i, j]) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
    rows = []
    title(a, "a", "Tracking advantage")
    root_forest(
        a,
        "pointmaze_stage9_confirmation",
        pm,
        ("fixed_minus_candidate_ise", "jitter_minus_candidate_ise", "current_minus_candidate_ise"),
        ("Fixed (primary)", "Random offset", "Current-only"),
        rows,
        "a",
        "Comparator - candidate ISE",
    )
    title(b, "b", "Return advantage")
    root_forest(
        b,
        "pointmaze_stage9_confirmation",
        pm,
        ("candidate_minus_fixed_return",),
        ("",),
        rows,
        "b",
        "Candidate - fixed return",
    )
    title(c, "c", "Controller-learning check")
    root_forest(
        c,
        "pointmaze_stage9_confirmation",
        pm,
        ("controller_learning_ise",),
        ("",),
        rows,
        "c",
        "Untrained - trained ISE",
    )
    title(d, "d", "Call count is not total compute")
    d.axis("off")
    d.text(0, 0.96, "Mode", fontweight="bold", va="top")
    d.text(0.60, 0.96, "Calls", fontweight="bold", ha="center", va="top")
    d.text(0.98, 0.96, "Scores", fontweight="bold", ha="right", va="top")
    for i, (mode, label, color) in enumerate(zip(MODES, LABELS, COLORS)):
        y = 0.70 - i * 0.19
        scores = pm["score_checks_by_mode"][mode]
        d.text(0, y, label, color=color)
        d.text(0.60, y, str(pm["upper_calls_per_episode_all_modes"]), ha="center")
        d.text(0.98, y, f"{scores:.2f}", ha="right")
        for metric, value in (("upper_calls", pm["upper_calls_per_episode_all_modes"]), ("score_checks", scores)):
            record(
                rows,
                "d",
                "mean_operation_count",
                mode,
                metric,
                value,
                source=f"{EVIDENCE}#/pointmaze_stage9_confirmation",
                unit="operations/episode",
                n=128,
                statistical_unit="evaluation_path_nested_in_8_roots",
            )
    handles = [
        Line2D([], [], marker="o", ls="none", color=TEAL, ms=3, label="Eight root means"),
        Line2D([], [], marker="D", color=INK, ms=4, label="Mean and 95% root-t CI"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.56, 0.045), ncol=2, fontsize=7)
    fig.text(
        0.20,
        0.02,
        "Same jitter-trained controller across timing modes. Five registered gates passed; total compute is not matched.",
        fontsize=6.3,
    )
    return save(fig, "fig2_pointmaze_confirmation", rows)


def figure3(evidence):
    unpriv = evidence["pointmaze_unprivileged_ablation"]
    pareto = evidence["pointmaze_budget_pareto"]
    fig = plt.figure(figsize=(183 / 25.4, 144 / 25.4))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.08, 1.15),
        height_ratios=(1.05, 1.0),
        left=0.20,
        right=0.97,
        bottom=0.15,
        top=0.89,
        wspace=0.55,
        hspace=0.92,
    )
    a = fig.add_subplot(grid[0, 0])
    b = fig.add_subplot(grid[0, 1])
    c = fig.add_subplot(grid[1, 0])
    d = fig.add_subplot(grid[1, 1])
    rows = []

    title(a, "a", "Label-free acquisition (B = 24)")
    root_forest(
        a,
        "pointmaze_unprivileged_ablation",
        unpriv,
        (
            "periodic_minus_unprivileged_uniform_ise",
            "distance_minus_unprivileged_uniform_ise",
            "privileged_minus_unprivileged_uniform_ise",
        ),
        ("Periodic", "Distance cut", "Event-stratified"),
        rows,
        "a",
        "Comparator - unprivileged ISE",
    )

    title(b, "b", "Multi-budget Pareto frontier")
    budgets = ("12", "15", "20", "24", "30", "40")
    xs = [int(k) for k in budgets]
    series_cfg = (
        ("periodic", "Periodic", GREY, "o", "--"),
        ("distance_threshold", "Distance cut", BLUE, "s", "-."),
        ("unprivileged_uniform", "Unprivileged validity", TEAL, "D", "-"),
    )
    for m_key, m_label, color, marker, ls in series_cfg:
        ys = [pareto["frontier"][k]["mean_ise"][m_key] for k in budgets]
        b.plot(xs, ys, color=color, marker=marker, ms=3.4, ls=ls, lw=1.15, label=m_label, zorder=4)
        for k, x_val, y_val in zip(budgets, xs, ys):
            record(
                rows,
                "b",
                "frontier_mean",
                m_key,
                "mean_ise",
                y_val,
                source=f"{EVIDENCE}#/pointmaze_budget_pareto/frontier/{k}/mean_ise/{m_key}",
                unit="episode ISE",
                n=8,
                statistical_unit="optimizer_seed_root",
                part=f"B={x_val}",
            )
    b.set_xticks(xs)
    b.set_xlabel("Planner calls per episode (B)")
    b.set_ylabel("Mean episode ISE")
    b.grid(True, color=LIGHT, lw=0.7)
    b.legend(loc="upper right", fontsize=6.2)

    title(c, "c", "Gain vs recalibrated distance")
    roots = sorted(pareto["frontier"]["24"]["distance_minus_unprivileged_roots"])
    for y_idx, k in enumerate(budgets):
        item = pareto["frontier"][k]["distance_minus_unprivileged_ise"]
        r_vals = [pareto["frontier"][k]["distance_minus_unprivileged_roots"][r] for r in roots]
        c.scatter(r_vals, y_idx + np.linspace(-0.15, 0.15, len(roots)), color=TEAL, alpha=0.65, s=7, zorder=3)
        c.plot(item["CI95"], [y_idx, y_idx], color=INK, lw=1.2, zorder=4)
        c.scatter([item["mean"]], [y_idx], marker="D", color=INK, s=14, zorder=5)
        record(
            rows,
            "c",
            "root_t_CI95",
            f"B{k}_distance_minus_unprivileged",
            "distance_minus_unprivileged_ise",
            item["mean"],
            ci=item["CI95"],
            source=f"{EVIDENCE}#/pointmaze_budget_pareto/frontier/{k}/distance_minus_unprivileged_ise",
            unit="Distance - unprivileged ISE",
            n=8,
            statistical_unit="optimizer_seed_root",
            part=f"B={k}",
        )
    c.set_yticks(range(len(budgets)), [f"B={k}" for k in budgets])
    c.set_ylim(len(budgets) - 0.5, -0.5)
    baseline(c, "Distance - unprivileged ISE")
    low_c, high_c = c.get_xlim()
    c.set_xlim(min(0, low_c) - 0.04 * (high_c - low_c), high_c)

    title(d, "d", "Fewer calls: paired ISE contrasts")
    cross_specs = (
        ("unprivileged_B20_vs_periodic_B24", "20 vs periodic 24", TEAL),
        ("unprivileged_B20_vs_distance_B24", "20 vs distance 24", BLUE),
        ("unprivileged_B24_vs_periodic_B30", "24 vs periodic 30", TEAL),
        ("unprivileged_B24_vs_distance_B30", "24 vs distance 30", BLUE),
        ("unprivileged_B30_vs_periodic_B40", "30 vs periodic 40", TEAL),
    )
    for y_idx, (c_key, _, color) in enumerate(cross_specs):
        item = pareto["cross_budget_call_savings"][c_key]
        d.plot(item["CI95"], [y_idx, y_idx], color=color, lw=1.3, zorder=4)
        d.scatter([item["mean"]], [y_idx], marker="D", color=color, s=16, zorder=5)
        record(
            rows,
            "d",
            "root_t_CI95",
            c_key,
            c_key,
            item["mean"],
            ci=item["CI95"],
            source=f"{EVIDENCE}#/pointmaze_budget_pareto/cross_budget_call_savings/{c_key}",
            unit="Comparator - fewer-call ridge ISE",
            n=8,
            statistical_unit="optimizer_seed_root",
        )
    d.set_yticks(range(len(cross_specs)), [label for _, label, _ in cross_specs])
    d.set_ylim(len(cross_specs) - 0.6, -0.6)
    baseline(d, "Comparator - fewer-call ridge ISE")
    low_d, high_d = d.get_xlim()
    d.set_xlim(min(0, low_d) - 0.05 * (high_d - low_d), high_d + 0.04 * (high_d - low_d))

    fig.text(
        0.20,
        0.025,
        "Eight frozen roots, 64 fresh paired paths/root (512 paths). Distance threshold recalibrated separately at every budget B.",
        fontsize=6.3,
    )
    return save(fig, "fig3_unprivileged_and_pareto", rows)


def figure4_route_b(evidence):
    sb = evidence["pointmaze_strong_baselines"]
    rp = evidence["reacher_polygonal_confirmation"]
    cem = evidence["pointmaze_expensive_cem_planner"]
    fig = plt.figure(figsize=(183 / 25.4, 152 / 25.4))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.08, 1.08),
        height_ratios=(1.15, 0.98),
        left=0.21,
        right=0.97,
        bottom=0.14,
        top=0.89,
        wspace=0.60,
        hspace=0.92,
    )
    a = fig.add_subplot(grid[0, 0])
    b = fig.add_subplot(grid[0, 1])
    c = fig.add_subplot(grid[1, 0])
    d = fig.add_subplot(grid[1, 1])
    rows = []

    title(a, "a", "PointMaze baselines (B = 24)")
    root_forest(
        a,
        "pointmaze_strong_baselines",
        sb,
        (
            "native_periodic_minus_ridge_ise",
            "temple_minus_ridge_ise",
            "shared_periodic_minus_ridge_ise",
            "generic_mlp_minus_ridge_ise",
            "distance_minus_ridge_ise",
            "deliberation_oc_minus_ridge_ise",
        ),
        (
            "Native periodic",
            "Learned binary",
            "Shared periodic",
            "Generic MLP",
            "Distance cut",
            "Cost difference",
        ),
        rows,
        "a",
        "Comparator - ridge ISE",
    )

    title(b, "b", "Reacher-Polygonal (8 roots)")
    root_forest(
        b,
        "reacher_polygonal_confirmation",
        rp,
        (
            "shared_jitter_minus_ridge_ise",
            "best_clock_minus_ridge_ise",
            "shared_periodic_minus_ridge_ise",
            "temple_minus_ridge_ise",
            "distance_minus_ridge_ise",
            "deliberation_oc_minus_ridge_ise",
        ),
        (
            "Random offset",
            "Best clock",
            "Shared periodic",
            "Learned binary",
            "Distance cut",
            "Cost difference",
        ),
        rows,
        "b",
        "Comparator - ridge ISE",
    )

    title(c, "c", "CEM planner ms vs tracking ISE")
    budgets = ("12", "15", "20", "24", "30", "40")
    series_cfg = (
        ("periodic", "Periodic", GREY, "o", "--"),
        ("distance_threshold", "Distance cut", BLUE, "s", "-."),
        ("unprivileged_uniform", "Unprivileged validity", TEAL, "D", "-"),
    )
    for m_key, m_label, color, marker, ls in series_cfg:
        xs = [cem["frontier"][k]["planner_ms_per_episode"][m_key] for k in budgets]
        ys = [cem["frontier"][k]["mean_ise"][m_key] for k in budgets]
        c.plot(xs, ys, color=color, marker=marker, ms=3.3, ls=ls, lw=1.15, label=m_label, zorder=4)
        for k, x_val, y_val in zip(budgets, xs, ys):
            record(
                rows,
                "c",
                "cem_planner_ms",
                m_key,
                "planner_ms_per_episode",
                x_val,
                source=f"{EVIDENCE}#/pointmaze_expensive_cem_planner/frontier/{k}/planner_ms_per_episode/{m_key}",
                unit="ms/episode",
                n=8,
                statistical_unit="optimizer_seed_root",
                part=f"B={k}",
            )
            record(
                rows,
                "c",
                "cem_frontier_ise",
                m_key,
                "mean_ise",
                y_val,
                source=f"{EVIDENCE}#/pointmaze_expensive_cem_planner/frontier/{k}/mean_ise/{m_key}",
                unit="episode ISE",
                n=8,
                statistical_unit="optimizer_seed_root",
                part=f"B={k}",
            )
    c.set_xlabel("CEM planner time (ms/episode)")
    c.set_ylabel("Mean episode ISE")
    c.grid(True, color=LIGHT, lw=0.7)
    c.legend(loc="upper right", fontsize=6.0)

    title(d, "d", "CEM cross-budget time delta")
    time_specs = (
        ("unprivileged_B20_vs_periodic_B24", "planner_ms_saved", "20 vs Per-24 (plan)", TEAL),
        ("unprivileged_B24_vs_periodic_B30", "planner_ms_saved", "24 vs Per-30 (plan)", TEAL),
        ("unprivileged_B20_vs_periodic_B30", "upper_loop_ms_saved", "20 vs Per-30 (upper)", BLUE),
        ("distance_B24_vs_periodic_B30", "upper_loop_ms_saved", "Dist 24 vs Per-30", BLUE),
        ("unprivileged_B20_vs_periodic_B24", "upper_loop_ms_saved", "20 vs Per-24 (upper)", ORANGE),
    )
    for y_idx, (c_key, sub_key, c_label, col) in enumerate(time_specs):
        item = cem["cross_budget_savings"][c_key][sub_key]
        d.plot(item["CI95"], [y_idx, y_idx], color=col, lw=1.3, zorder=4)
        d.scatter([item["mean"]], [y_idx], marker="D", color=col, s=15, zorder=5)
        record(
            rows,
            "d",
            "root_t_CI95",
            f"{c_key}_{sub_key}",
            sub_key,
            item["mean"],
            ci=item["CI95"],
            source=f"{EVIDENCE}#/pointmaze_expensive_cem_planner/cross_budget_savings/{c_key}/{sub_key}",
            unit="Time saved (ms/episode)",
            n=8,
            statistical_unit="optimizer_seed_root",
        )
    d.set_yticks(range(len(time_specs)), [lbl for _, _, lbl, _ in time_specs])
    d.set_ylim(len(time_specs) - 0.55, -0.55)
    baseline(d, "Periodic - adaptive time (ms)")
    low_d, high_d = d.get_xlim()
    d.set_xlim(low_d - 0.05 * (high_d - low_d), high_d + 0.05 * (high_d - low_d))

    fig.text(
        0.15,
        0.022,
        "Eight roots per study. Panel d separates planner time (plan) from planner plus scoring time (upper).",
        fontsize=6.1,
    )
    return save(fig, "fig4_route_b_baselines_reacher_cem", rows)


def paired_forest(ax, specs, rows, panel, xlabel, color):
    for y, spec in enumerate(specs):
        row, ci_names, source, name, part, metric, n, unit, scale, label = spec
        for offset, marker, ci_name in zip((-0.10, 0.10), ("o", "s"), ci_names):
            mean, ci = row["mean"][metric], row[ci_name][metric]
            ax.errorbar(
                mean * scale,
                y + offset,
                xerr=np.asarray([[mean - ci[0]], [ci[1] - mean]]) * scale,
                fmt=marker,
                ms=3.1,
                color=color,
                markerfacecolor=color if marker == "o" else "white",
                capsize=2,
                elinewidth=1,
                markeredgewidth=0.8,
                zorder=3,
            )
            record(
                rows,
                panel,
                ci_name,
                name,
                metric,
                mean,
                ci=ci,
                source=source,
                scale=scale,
                unit=xlabel,
                n=n,
                statistical_unit=unit,
                part=part,
            )
    ax.set_yticks(range(len(specs)), [s[-1] for s in specs])
    ax.set_ylim(len(specs) - 0.55, -0.65)
    baseline(ax, xlabel)


def figure5_transit(evidence):
    fig = plt.figure(figsize=(183 / 25.4, 190 / 25.4))
    grid = fig.add_gridspec(
        3, 2, left=0.18, right=0.97, bottom=0.17, top=0.91, wspace=0.68, hspace=1.04, height_ratios=(1, 1, 1.08)
    )
    axes = [fig.add_subplot(grid[i, j]) for i in range(3) for j in range(2)]
    rows = []
    titles = (
        "Single intervention: cost",
        "Single intervention: waiting",
        "Two decisions: cost",
        "Two decisions: waiting",
        "Cost components (descriptive)",
        "Half-base waiting margin",
    )
    for ax, letter, text in zip(axes, "abcdef", titles):
        title(ax, letter, text)
    studies = [
        ("transit_stage298_single_intervention", "joint_budgeted_vs_raw", "Cohort A joint"),
        ("transit_stage301_single_intervention", "service_only_vs_raw", "Cohort B service"),
        ("transit_stage301_single_intervention", "joint_budgeted_vs_raw", "Cohort B joint"),
    ]
    for ax, panel, metric, scale, xlabel in (
        (axes[0], "a", "cost", 1000, "Cost change (x 10^-3)"),
        (axes[1], "b", "wait_min", 60, "Wait change (s)"),
    ):
        specs = []
        for study, name, label in studies:
            item = evidence[study]
            specs.append((
                item["comparisons"][name],
                ("prefix_CI95", "conditional_MC_CI95"),
                f"{EVIDENCE}#/{study}/comparisons/{name}",
                name,
                "pooled",
                metric,
                item["prefixes"],
                "native_prefix",
                scale,
                label,
            ))
        paired_forest(ax, specs, rows, panel, xlabel, TEAL)
    native = evidence["transit_stage304_two_decisions"]
    parts = (("all32_scenes", "All 320", 320), ("first16_scenes", "First 160", 160), ("last16_scenes", "Last 160", 160))
    for ax, panel, name, metric, scale, xlabel in (
        (axes[2], "c", "service_only_vs_raw", "cost", 1000, "Cost change (x 10^-3)"),
        (axes[3], "d", "service_only_vs_raw", "wait_min", 60, "Wait change (s)"),
        (axes[5], "f", "service_only_half_base_wait_margin", "wait_min", 60, "Paired waiting margin (s)"),
    ):
        specs = [
            (
                native["comparisons"][part][name],
                ("episode_bootstrap_CI95", "paired_normal_CI95"),
                f"{EVIDENCE}#/transit_stage304_two_decisions/comparisons/{part}/{name}",
                name,
                part,
                metric,
                n,
                "complete_native_scene_episode",
                scale,
                label,
            )
            for part, label, n in parts
        ]
        paired_forest(ax, specs, rows, panel, xlabel, ORANGE)
    comp = native["comparisons"]["all32_scenes"]["service_only_vs_raw"]["mean"]
    e = axes[4]
    for i, (metric, label) in enumerate((("fleet", "Fleet"), ("service", "Service"), ("cost", "Total"))):
        color = (ORANGE, TEAL, GREY)[i]
        e.barh(i, comp[metric] * 1000, height=0.52, color=color)
        record(
            rows,
            "e",
            "descriptive_mean",
            "service_only_vs_raw",
            metric,
            comp[metric],
            source=f"{EVIDENCE}#/transit_stage304_two_decisions/comparisons/all32_scenes/service_only_vs_raw/mean/{metric}",
            scale=1000,
            unit="Cost change (x 10^-3)",
            n=320,
            statistical_unit="complete_native_scene_episode",
        )
    e.set_yticks(range(3), ("Fleet", "Service", "Total"))
    e.set_ylim(2.6, -0.6)
    baseline(e, "Cost component change (x 10^-3)")
    d = native["descriptive"]
    e.text(
        0,
        -0.59,
        f"Peak changes: {d['fleet_peak_increases']} up, {d['fleet_peak_decreases']} down.\nJoint/service: {d['same_metrics_scenes']}/320 outcomes identical.",
        transform=e.transAxes,
        fontsize=6.2,
        va="top",
        linespacing=1.45,
    )
    for key in ("fleet_peak_increases", "fleet_peak_decreases", "same_metrics_scenes", "peak_fleet_difference_scenes"):
        record(
            rows,
            "e",
            "descriptive_count",
            "joint_vs_service" if "scenes" in key else "service_only_vs_raw",
            key,
            d[key],
            source=f"{EVIDENCE}#/transit_stage304_two_decisions/descriptive/{key}",
            n=320,
            statistical_unit="complete_native_scene_episode",
            unit="scenes",
        )
    name = "joint_budgeted_vs_service_only"
    increment = native["comparisons"]["all32_scenes"][name]
    for ci_name in ("episode_bootstrap_CI95", "paired_normal_CI95"):
        record(
            rows,
            "caption",
            ci_name,
            name,
            "cost",
            increment["mean"]["cost"],
            ci=increment[ci_name]["cost"],
            source=f"{EVIDENCE}#/transit_stage304_two_decisions/comparisons/all32_scenes/{name}",
            n=320,
            statistical_unit="complete_native_scene_episode",
            scale=1000,
            unit="Cost change (x 10^-3)",
        )
    fig.legend(
        handles=legend_handles("Prefix bootstrap", "Conditional MC", TEAL),
        loc="upper center",
        bbox_to_anchor=(0.59, 0.99),
        ncol=2,
        fontsize=6.7,
    )
    fig.legend(
        handles=legend_handles("Episode bootstrap", "Paired normal", ORANGE),
        loc="upper center",
        bbox_to_anchor=(0.59, 0.72),
        ncol=2,
        fontsize=6.7,
    )
    fig.text(
        0.18,
        0.033,
        "Negative favors the first controller; all differences are paired. Single and repeated decisions are different cohorts.",
        fontsize=6.1,
    )
    fig.text(
        0.18,
        0.012,
        "Half-base criterion: margin < 0 under both CIs in all three parts. This criterion and the dual-win primary did not pass.",
        fontsize=6.1,
    )
    return save(fig, "fig5_transit_boundary", rows)


def main():
    evidence = json.loads((PAPER / EVIDENCE).read_text())
    reports = [
        figure1(evidence["pointmaze_stage9_confirmation"]),
        figure2(evidence["pointmaze_stage9_confirmation"]),
        figure3(evidence),
        figure4_route_b(evidence),
        figure5_transit(evidence),
    ]
    (OUTPUT / "render_checks.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
