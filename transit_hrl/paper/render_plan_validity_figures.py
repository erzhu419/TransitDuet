"""Render the three manuscript figures from completed local evidence only."""

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
CSV_FIELDS = ("panel", "kind", "comparison", "metric", "part", "root", "value_raw",
              "ci_low_raw", "ci_high_raw", "display_scale", "display_unit", "n",
              "statistical_unit", "source")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
    "axes.titlesize": 8, "axes.titleweight": "bold", "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 7, "text.color": INK,
    "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .6, "lines.linewidth": 1, "svg.fonttype": "none", "pdf.fonttype": 42,
    "legend.frameon": False})


def record(rows, panel, kind, comparison, metric, value, *, source,
           ci=None, scale=1, unit="", n="", statistical_unit="", part="pooled", root=""):
    rows.append(dict(zip(CSV_FIELDS, (panel, kind, comparison, metric, part, root,
        value, "" if ci is None else ci[0], "" if ci is None else ci[1],
        scale, unit, n, statistical_unit, source))))


def title(ax, letter, text):
    ax.set_title(text, loc="left", pad=9)
    ax.text(-.02, 1.13, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="right", va="bottom")


def baseline(ax, xlabel):
    ax.axvline(0, color=GREY, lw=.7, ls=(0, (2, 2)), zorder=1)
    ax.set_xlabel(xlabel)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=LIGHT, lw=.7)


def legend_handles(first, second, color):
    return [Line2D([], [], color=color, marker="o", ms=3, label=first),
            Line2D([], [], color=color, marker="s", markerfacecolor="white", ms=3, label=second)]


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
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width+1 or box.y1 > height+1:
            outside.append(text.get_text())
    if outside:
        raise ValueError(f"Figure text outside canvas: {stem}: {outside}")
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUTPUT / f"{stem}.{ext}", dpi=300, facecolor="white")
    with (OUTPUT / f"{stem}_source.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    fraction = float(np.mean(np.min(rgb, axis=2) < 245))
    if fraction < .01:
        raise ValueError(f"Blank figure: {stem}")
    report = {"figure": stem, "width_mm": round(fig.get_figwidth()*25.4, 2),
        "height_mm": round(fig.get_figheight()*25.4, 2), "source_rows": len(rows),
        "nonwhite_fraction": fraction, "text_within_canvas": True}
    plt.close(fig)
    return report


def figure1(pm):
    fig = plt.figure(figsize=(183/25.4, 124/25.4))
    grid = fig.add_gridspec(2, 1, height_ratios=(.9, 1.35), left=.18, right=.96,
                          bottom=.16, top=.90, hspace=.80)
    a, b = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    title(a, "a", "Paired supervision, then frozen deployment")
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis("off")
    contract = pm["policy_contract"]
    boxes = [(0, "Sample queries", "Event-stratified\nsimulator paths"),
             (.37, "Keep / renew", f"Paired {contract['branch_horizon_steps']}-step loss\nClosed-loop lower"),
             (.74, "Fit predictor", f"{contract['candidate_features']} causal features\nFit-path q75 threshold")]
    for x, heading, body in boxes:
        a.add_patch(Rectangle((x, .17), .26, .71, facecolor=LIGHT, edgecolor="none"))
        a.text(x+.13, .73, heading, ha="center", va="center", fontweight="bold")
        a.text(x+.13, .41, body, ha="center", va="center", linespacing=1.5)
    for x1, x2 in ((.265, .365), (.635, .735)):
        a.add_patch(FancyArrowPatch((x1, .54), (x2, .54), arrowstyle="->", mutation_scale=9, color=GREY))
    a.text(0, -.14, "Offline: event metadata selects queries. Online: observable history only.", fontsize=6.7)
    a.text(0, -.42, "Check every 5 steps; renew at first score >= q, or at the offset-25 deadline.", fontsize=6.7)
    source = "../" + pm["example_schedule"]["source"]
    rows = []
    for key, value in contract.items():
        record(rows, "a", "protocol", "candidate", key, value, source=source)
    title(b, "b", "Observed upper-call timing: first 300 steps")
    example = pm["example_schedule"]
    k, deadline = contract["bin_steps"], contract["max_offset_steps"]
    for start in range(0, 300, k):
        if start//k % 2 == 0:
            b.axvspan(start, start+k, color=LIGHT, zorder=0)
        b.axvline(start, color="#cbd1d6", lw=.6, zorder=0)
        if start > 0:
            b.axvline(start+deadline, color=GREY, lw=.65, ls=(0, (1, 3)), zorder=0)
    for i, (mode, label, color) in enumerate(zip(MODES, LABELS, COLORS)):
        y = 3-i
        b.hlines(y, 0, 300, color="#d5dadd", lw=.8)
        steps = [s for s in example["decision_steps"][mode] if s < 300]
        b.scatter(steps, [y]*len(steps), color=color, s=21, marker="|", linewidths=1.8, zorder=3)
        for step in steps:
            record(rows, "b", "observed_call", mode, "decision_step", step,
                   source=source, unit="primitive step", n=1, statistical_unit="illustrative_path", root=example["root"])
    b.set_yticks(range(4), LABELS[::-1]); b.set_ylim(-.5, 3.5)
    b.set_xticks(range(0, 301, 50)); b.set_xlim(-4, 304)
    b.set_xlabel("Primitive step (bin boundaries every 50 steps)")
    b.spines[["left", "right", "top"]].set_visible(False); b.tick_params(axis="y", length=0)
    fig.text(.18, .065, f"Fixed example: optimizer root {example['root']}, path {example['path_seed']}; not selected for performance.", fontsize=6.5)
    fig.text(.18, .025, "All modes: 24 calls/episode. Dotted lines: deadlines; lower actions continue at every step.", fontsize=6.5)
    return save(fig, "fig1_budgeted_replanning", rows)


def root_forest(ax, pm, names, labels, rows, panel, xlabel):
    roots = sorted(pm["root_effects"])
    for y, (name, label) in enumerate(zip(names, labels)):
        item = pm["contrasts"][name]
        xs = [pm["root_effects"][root][name] for root in roots]
        ax.scatter(xs, y+np.linspace(-.19, .19, len(roots)), color=TEAL, alpha=.7, s=10, zorder=3)
        ax.plot(item["CI95"], [y, y], color=INK, lw=1.3, zorder=4)
        ax.scatter([item["mean"]], [y], marker="D", color=INK, s=18, zorder=5)
        source = f"{EVIDENCE}#/pointmaze_stage9_confirmation/contrasts/{name}"
        record(rows, panel, "root_t_CI95", name, name, item["mean"], ci=item["CI95"],
               source=source, unit=xlabel, n=8, statistical_unit="optimizer_seed_root")
        for root, value in zip(roots, xs):
            record(rows, panel, "root_mean", name, name, value,
                   source=f"{EVIDENCE}#/pointmaze_stage9_confirmation/root_effects/{root}/{name}",
                   unit=xlabel, n=1, statistical_unit="optimizer_seed_root", root=root)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels)-.6, -.6)
    baseline(ax, xlabel)
    low, high = ax.get_xlim()
    ax.set_xlim(min(0, low)-.03*(high-low), high)


def figure2(pm):
    fig = plt.figure(figsize=(183/25.4, 139/25.4))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.35, 1), height_ratios=(1.15, .85),
        left=.20, right=.97, bottom=.16, top=.90, wspace=.57, hspace=.95)
    a, b, c, d = [fig.add_subplot(grid[i, j]) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
    rows = []
    title(a, "a", "Tracking advantage")
    root_forest(a, pm, ("fixed_minus_candidate_ise", "jitter_minus_candidate_ise", "current_minus_candidate_ise"),
        ("Fixed (primary)", "Random offset", "Current-only"), rows, "a", "Comparator - candidate ISE")
    title(b, "b", "Return advantage")
    root_forest(b, pm, ("candidate_minus_fixed_return",), ("",), rows, "b", "Candidate - fixed return")
    title(c, "c", "Controller-learning check")
    root_forest(c, pm, ("controller_learning_ise",), ("",), rows, "c", "Untrained - trained ISE")
    title(d, "d", "Call count is not total compute")
    d.axis("off")
    d.text(0, .96, "Mode", fontweight="bold", va="top")
    d.text(.60, .96, "Calls", fontweight="bold", ha="center", va="top")
    d.text(.98, .96, "Scores", fontweight="bold", ha="right", va="top")
    for i, (mode, label, color) in enumerate(zip(MODES, LABELS, COLORS)):
        y = .70-i*.19
        scores = pm["score_checks_by_mode"][mode]
        d.text(0, y, label, color=color)
        d.text(.60, y, str(pm["upper_calls_per_episode_all_modes"]), ha="center")
        d.text(.98, y, f"{scores:.2f}", ha="right")
        for metric, value in (("upper_calls", pm["upper_calls_per_episode_all_modes"]), ("score_checks", scores)):
            record(rows, "d", "mean_operation_count", mode, metric, value,
                source=f"{EVIDENCE}#/pointmaze_stage9_confirmation", unit="operations/episode",
                n=128, statistical_unit="evaluation_path_nested_in_8_roots")
    handles = [Line2D([], [], marker="o", ls="none", color=TEAL, ms=3, label="Eight root means"),
               Line2D([], [], marker="D", color=INK, ms=4, label="Mean and 95% root-t CI")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.56, .045), ncol=2, fontsize=7)
    fig.text(.20, .025, "Same jitter-trained controller across timing modes. Five registered gates passed; total compute is not matched.", fontsize=6.3)
    return save(fig, "fig2_pointmaze_confirmation", rows)


def paired_forest(ax, specs, rows, panel, xlabel, color):
    for y, spec in enumerate(specs):
        row, ci_names, source, name, part, metric, n, unit, scale, label = spec
        for offset, marker, ci_name in zip((-.10, .10), ("o", "s"), ci_names):
            mean, ci = row["mean"][metric], row[ci_name][metric]
            ax.errorbar(mean*scale, y+offset,
                xerr=np.asarray([[mean-ci[0]], [ci[1]-mean]])*scale,
                fmt=marker, ms=3.1, color=color, markerfacecolor=color if marker == "o" else "white",
                capsize=2, elinewidth=1, markeredgewidth=.8, zorder=3)
            record(rows, panel, ci_name, name, metric, mean, ci=ci, source=source,
                   scale=scale, unit=xlabel, n=n, statistical_unit=unit, part=part)
    ax.set_yticks(range(len(specs)), [s[-1] for s in specs])
    ax.set_ylim(len(specs)-.55, -.65)
    baseline(ax, xlabel)


def figure3(evidence):
    fig = plt.figure(figsize=(183/25.4, 190/25.4))
    grid = fig.add_gridspec(3, 2, left=.18, right=.97, bottom=.17, top=.91,
                          wspace=.68, hspace=1.04, height_ratios=(1, 1, 1.08))
    axes = [fig.add_subplot(grid[i, j]) for i in range(3) for j in range(2)]
    rows = []
    titles = ("Single intervention: cost", "Single intervention: waiting",
              "Two decisions: cost", "Two decisions: waiting",
              "Cost components (descriptive)", "Half-base waiting margin")
    for ax, letter, text in zip(axes, "abcdef", titles):
        title(ax, letter, text)
    studies = [("transit_stage298_single_intervention", "joint_budgeted_vs_raw", "S298 joint"),
               ("transit_stage301_single_intervention", "service_only_vs_raw", "S301 service"),
               ("transit_stage301_single_intervention", "joint_budgeted_vs_raw", "S301 joint")]
    for ax, panel, metric, scale, xlabel in ((axes[0], "a", "cost", 1000, "Cost change (x 10^-3)"),
                                           (axes[1], "b", "wait_min", 60, "Wait change (s)")):
        specs = []
        for study, name, label in studies:
            item = evidence[study]
            specs.append((item["comparisons"][name], ("prefix_CI95", "conditional_MC_CI95"),
                f"{EVIDENCE}#/{study}/comparisons/{name}", name, "pooled", metric, item["prefixes"],
                "native_prefix", scale, label))
        paired_forest(ax, specs, rows, panel, xlabel, TEAL)
    native = evidence["transit_stage304_two_decisions"]
    parts = (("all32_scenes", "All 320", 320), ("first16_scenes", "First 160", 160), ("last16_scenes", "Last 160", 160))
    for ax, panel, name, metric, scale, xlabel in (
            (axes[2], "c", "service_only_vs_raw", "cost", 1000, "Cost change (x 10^-3)"),
            (axes[3], "d", "service_only_vs_raw", "wait_min", 60, "Wait change (s)"),
            (axes[5], "f", "service_only_half_base_wait_margin", "wait_min", 60, "Paired waiting margin (s)")):
        specs = [(native["comparisons"][part][name], ("episode_bootstrap_CI95", "paired_normal_CI95"),
            f"{EVIDENCE}#/transit_stage304_two_decisions/comparisons/{part}/{name}", name, part,
            metric, n, "complete_native_scene_episode", scale, label) for part, label, n in parts]
        paired_forest(ax, specs, rows, panel, xlabel, ORANGE)
    comp = native["comparisons"]["all32_scenes"]["service_only_vs_raw"]["mean"]
    e = axes[4]
    for i, (metric, label) in enumerate((("fleet", "Fleet"), ("service", "Service"), ("cost", "Total"))):
        color = (ORANGE, TEAL, GREY)[i]
        e.barh(i, comp[metric]*1000, height=.52, color=color)
        record(rows, "e", "descriptive_mean", "service_only_vs_raw", metric, comp[metric],
            source=f"{EVIDENCE}#/transit_stage304_two_decisions/comparisons/all32_scenes/service_only_vs_raw/mean/{metric}",
            scale=1000, unit="Cost change (x 10^-3)", n=320, statistical_unit="complete_native_scene_episode")
    e.set_yticks(range(3), ("Fleet", "Service", "Total")); e.set_ylim(2.6, -.6)
    baseline(e, "Cost component change (x 10^-3)")
    d = native["descriptive"]
    e.text(0, -.59, f"Peak changes: {d['fleet_peak_increases']} up, {d['fleet_peak_decreases']} down.\nJoint/service: {d['same_metrics_scenes']}/320 outcomes identical.",
           transform=e.transAxes, fontsize=6.2, va="top", linespacing=1.45)
    for key in ("fleet_peak_increases", "fleet_peak_decreases", "same_metrics_scenes", "peak_fleet_difference_scenes"):
        record(rows, "e", "descriptive_count", "joint_vs_service" if "scenes" in key else "service_only_vs_raw",
               key, d[key], source=f"{EVIDENCE}#/transit_stage304_two_decisions/descriptive/{key}",
               n=320, statistical_unit="complete_native_scene_episode", unit="scenes")
    name = "joint_budgeted_vs_service_only"
    increment = native["comparisons"]["all32_scenes"][name]
    for ci_name in ("episode_bootstrap_CI95", "paired_normal_CI95"):
        record(rows, "caption", ci_name, name, "cost", increment["mean"]["cost"],
            ci=increment[ci_name]["cost"],
            source=f"{EVIDENCE}#/transit_stage304_two_decisions/comparisons/all32_scenes/{name}",
            n=320, statistical_unit="complete_native_scene_episode", scale=1000, unit="Cost change (x 10^-3)")
    fig.legend(handles=legend_handles("Prefix bootstrap", "Conditional MC", TEAL),
               loc="upper center", bbox_to_anchor=(.59, .99), ncol=2, fontsize=6.7)
    fig.legend(handles=legend_handles("Episode bootstrap", "Paired normal", ORANGE),
               loc="upper center", bbox_to_anchor=(.59, .72), ncol=2, fontsize=6.7)
    fig.text(.18, .033, "Negative favors the first controller; all differences are paired. Single and repeated decisions are different cohorts.", fontsize=6.1)
    fig.text(.18, .012, "Half-base criterion: margin < 0 under both CIs in all three parts. This criterion and the dual-win primary did not pass.", fontsize=6.1)
    return save(fig, "fig3_transit_boundary", rows)


def main():
    evidence = json.loads((PAPER / EVIDENCE).read_text())
    reports = [figure1(evidence["pointmaze_stage9_confirmation"]),
               figure2(evidence["pointmaze_stage9_confirmation"]), figure3(evidence)]
    (OUTPUT / "render_checks.json").write_text(json.dumps(reports, indent=2)+"\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
