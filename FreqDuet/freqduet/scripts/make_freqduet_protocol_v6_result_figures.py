#!/usr/bin/env python3
"""Render the Protocol V6 current-best paper result figures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_freqduet_protocol_v6_evidence_package import (
    PAPER_CONTROLLER,
    PROTOCOL,
    refresh_package_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = (
    ROOT / "results_freqduet" / "paper_package" / "protocol_v6_current_best"
)
DEFAULT_FORMATS = ("svg", "pdf", "tiff", "png")
FIGURE_WIDTH_MM = 183.0

TEAL = "#187C73"
ORANGE = "#D97732"
AMBER = "#B9832F"
GRAY = "#68717A"
LIGHT_GRAY = "#D9DEE2"
BLACK = "#202428"


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing figure source table: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty figure source table: {path}")
    return rows


def one_row(rows: Iterable[dict[str, str]], **matches: str) -> dict[str, str]:
    selected = [
        row for row in rows
        if all(row.get(key) == value for key, value in matches.items())
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one figure row for {matches}, found {len(selected)}")
    return selected[0]


def interval(row: dict[str, str], delta_key: str) -> dict[str, float]:
    return {
        "mean": float(row[delta_key]),
        "low": float(row["ci95_low"]),
        "high": float(row["ci95_high"]),
    }


def validate_tables(package_dir: Path) -> dict[str, list[dict[str, str]]]:
    tables = package_dir / "tables"
    sources = {
        "v8": read_rows(tables / "table1_v8_confirmation.csv"),
        "v9": read_rows(tables / "table2_v9_longtrain.csv"),
        "external": read_rows(tables / "table3_v9_external_baselines.csv"),
        "decisions": read_rows(tables / "table4_evidence_decisions.csv"),
    }
    required_metrics = {"passenger_journey_min", "headway_cv"}
    for phase, expected_n in (("v8", 24), ("v9", 64)):
        metrics = {row["metric"] for row in sources[phase]}
        if not required_metrics.issubset(metrics):
            raise ValueError(f"{phase} table lacks a required result metric")
        if any(int(row["n_pairs"]) != expected_n for row in sources[phase]):
            raise ValueError(f"{phase} table has the wrong paired sample size")
        if any(row["paper_controller"] != PAPER_CONTROLLER
               for row in sources[phase]):
            raise ValueError(f"{phase} table uses a different controller")

    expected_baselines = {"fixed_headway", "rule_holding", "rule_mpc"}
    if {row["baseline"] for row in sources["external"]} != expected_baselines:
        raise ValueError("external figure table has the wrong baseline roster")
    if any(int(row["n_pairs"]) != 64 for row in sources["external"]):
        raise ValueError("external figure table has the wrong paired sample size")
    for metric in ("passenger_journey_min", "headway_cv",
                   "restricted_service_cost"):
        rows = [row for row in sources["external"] if row["metric"] == metric]
        if len(rows) != 3:
            raise ValueError(f"external figure table lacks complete {metric} rows")

    decisions = {
        row["phase"]: row["decision"] for row in sources["decisions"]
    }
    if decisions.get("v8_independent_confirmation_ep40") != "primary_confirmed":
        raise ValueError("V8 figure decision is not primary_confirmed")
    if decisions.get("v9_independent_longtrain_ep200") != (
        "longtrain_not_confirmed"
    ):
        raise ValueError("V9 figure decision is not longtrain_not_confirmed")
    return sources


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def significance_color(record: dict[str, float]) -> str:
    if record["high"] < 0:
        return TEAL
    if record["low"] > 0:
        return ORANGE
    return GRAY


def forest_plot(
    ax: plt.Axes,
    records: list[dict[str, float]],
    labels: list[str],
    colors: list[str],
    xlabel: str,
) -> None:
    positions = list(reversed(range(len(records))))
    for y, record, color in zip(positions, records, colors):
        mean = record["mean"]
        ax.errorbar(
            mean,
            y,
            xerr=[[mean - record["low"]], [record["high"] - mean]],
            fmt="o",
            markersize=5.2,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            ecolor=color,
            elinewidth=1.4,
            capsize=2.8,
            capthick=1.0,
            zorder=3,
        )
    ax.axvline(0.0, color=BLACK, linewidth=0.8, linestyle="--", zorder=1)
    ax.set_yticks(positions, labels)
    ax.set_ylim(-0.65, len(records) - 0.35)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(
        0.01,
        -0.31,
        "Lower values favor FreqDuet",
        transform=ax.transAxes,
        color=GRAY,
        fontsize=6,
        ha="left",
        va="top",
    )


def save_figure(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    formats: tuple[str, ...],
) -> list[str]:
    outputs: list[str] = []
    for extension in formats:
        path = out_dir / f"{stem}.{extension}"
        kwargs: dict[str, Any] = {}
        if extension == "tiff":
            kwargs = {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}
        elif extension == "png":
            kwargs = {"dpi": 300}
        fig.savefig(path, **kwargs)
        outputs.append(path.name)
    plt.close(fig)
    return outputs


def confirmation_figure(
    sources: dict[str, list[dict[str, str]]],
    out_dir: Path,
    formats: tuple[str, ...],
) -> list[str]:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH_MM / 25.4, 76.0 / 25.4),
        layout="constrained",
    )
    phase_labels = [
        "V8, 40 episodes\ngate-positive",
        "V9, 200 episodes\nnot confirmed",
    ]
    phase_colors = [TEAL, AMBER]
    metrics = (
        (
            "headway_cv",
            "Headway regularity",
            "Current policy - protocol reference (headway CV)",
        ),
        (
            "passenger_journey_min",
            "Passenger journey",
            "Current policy - protocol reference (min)",
        ),
    )
    for index, (ax, (metric, title, xlabel)) in enumerate(zip(axes, metrics)):
        records = []
        for phase in ("v8", "v9"):
            row = one_row(sources[phase], metric=metric)
            records.append(interval(row, "delta_candidate_minus_reference"))
        forest_plot(ax, records, phase_labels, phase_colors, xlabel)
        ax.set_title(title, loc="left", fontweight="bold", pad=8)
        panel_label(ax, chr(ord("a") + index))
    return save_figure(
        fig,
        out_dir,
        "fig2_protocol_v6_confirmation_robustness",
        formats,
    )


def external_figure(
    sources: dict[str, list[dict[str, str]]],
    out_dir: Path,
    formats: tuple[str, ...],
) -> list[str]:
    fig = plt.figure(
        figsize=(FIGURE_WIDTH_MM / 25.4, 118.0 / 25.4),
        layout="constrained",
    )
    grid = fig.add_gridspec(2, 2, height_ratios=(1.08, 1.0))
    axes = (fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[1, 1]))
    baseline_order = ("fixed_headway", "rule_holding", "rule_mpc")
    baseline_labels = ["Fixed headway", "Rule holding", "Rule MPC"]
    metrics = (
        (
            "passenger_journey_min",
            "Passenger journey",
            "FreqDuet - baseline (min)",
        ),
        ("headway_cv", "Headway regularity", "FreqDuet - baseline (headway CV)"),
        (
            "restricted_service_cost",
            "Restricted service cost",
            "FreqDuet - baseline (cost)",
        ),
    )
    for index, (ax, (metric, title, xlabel)) in enumerate(zip(axes, metrics)):
        records = [
            interval(
                one_row(sources["external"], baseline=baseline, metric=metric),
                "delta_learned_minus_baseline",
            )
            for baseline in baseline_order
        ]
        colors = [significance_color(record) for record in records]
        forest_plot(ax, records, baseline_labels, colors, xlabel)
        ax.set_title(title, loc="left", fontweight="bold", pad=8)
        panel_label(ax, chr(ord("a") + index))
    return save_figure(
        fig,
        out_dir,
        "fig3_protocol_v6_external_tradeoff",
        formats,
    )


def write_notes(out_dir: Path) -> None:
    (out_dir / "captions.md").write_text("""# Protocol V6 Figure Captions

## Figure 2 | Independent confirmation and long-training robustness

Points show paired mean differences between the current policy and the
Protocol V6 reference config named `F_freqduet_protocol_v6_noguard_hiro`; bars
show 95% crossed-bootstrap confidence intervals over training and evaluation
seeds. Both configurations disable the legacy causal holding guard. The
current policy additionally uses compact APC/AVL context and the two-sided
departure-regularity objective, so this is a combined-policy comparison rather
than an isolated guard effect. Lower values favor the current policy. V8 contains 24
paired rollouts (six training seeds by four untouched evaluation seeds) and
passed the registered effect/no-harm gate. Its Holm-adjusted training-seed
sign-flip result was p=0.125, so the figure labels V8 as gate-positive rather
than familywise significant. V9 contains 64 paired rollouts (eight by eight);
its passenger-journey interval favored FreqDuet, but the headway-CV effect did
not meet the registered magnitude and interval gate, so V9 is reported as not
confirmed.

## Figure 3 | External baseline trade-off under the V9 source contract

Points show paired mean differences between FreqDuet and each external
baseline; bars show 95% crossed-bootstrap confidence intervals over eight
training and eight evaluation seeds (64 paired rollouts). Lower values favor
FreqDuet. FreqDuet improved regularity and restricted service cost relative to
fixed headway but increased passenger journey time. It reduced passenger
journey time relative to rule holding and rule MPC. Exact two-sided sign-flip
tests and Holm-adjusted values are provided in the source table and are not
encoded as significance symbols in the figure.
""")
    (out_dir / "figure_qa.md").write_text("""# Protocol V6 Figure QA

- Core conclusion: V8 passes the registered short-horizon effect/no-harm gate,
  but its Holm-adjusted sign-flip p-value is 0.125; V9 does not confirm the
  long-training gate; the external comparison is a service-regularity versus
  passenger-journey trade-off.
- Evidence chain: Figure 2 keeps V8 and V9 separate and labels the historical
  `noguard` config as the protocol reference; Figure 3 uses only the
  source-identical V9 external comparison.
- Archetype: quantitative grid with the confirmation panel as the primary
  evidence and external comparisons as validation.
- Backend: Python with matplotlib only.
- Final size: 183 mm double-column width; 76 mm and 118 mm heights.
- Statistics: paired mean differences and 95% crossed-bootstrap intervals;
  n is 24 pairs for V8 and 64 pairs for V9/external comparisons.
- Source data: `tables/table1_v8_confirmation.csv`,
  `tables/table2_v9_longtrain.csv`, and
  `tables/table3_v9_external_baselines.csv`.
- Integrity: no pooled V8/V9 estimate, no hidden V9 failure, no transformed
  endpoint, no legacy-guard effect claim, and no significance symbol
  substituted for the registered gate.
- Exports: editable-text SVG, TrueType-text PDF, 600 dpi LZW TIFF, and 300 dpi
  PNG review render.
""")


def build_figures(
    package_dir: Path,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
) -> dict[str, Any]:
    unsupported = set(formats) - set(DEFAULT_FORMATS)
    if unsupported or not formats:
        raise ValueError(f"unsupported or empty figure formats: {sorted(unsupported)}")
    status = json.loads((package_dir / "evidence_status.json").read_text())
    if status.get("protocol") != PROTOCOL:
        raise ValueError("figure package is not bound to Protocol V6")
    if status.get("paper_controller") != PAPER_CONTROLLER:
        raise ValueError("figure package uses a different controller")
    if status.get("submission_ready") is not False:
        raise ValueError("current figure package must retain the submission hold")

    sources = validate_tables(package_dir)
    out_dir = package_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in (
        "fig2_protocol_v6_confirmation_robustness",
        "fig3_protocol_v6_external_tradeoff",
    ):
        for extension in DEFAULT_FORMATS:
            (out_dir / f"{stem}.{extension}").unlink(missing_ok=True)
    configure_matplotlib()
    figure2 = confirmation_figure(sources, out_dir, formats)
    figure3 = external_figure(sources, out_dir, formats)
    write_notes(out_dir)
    manifest = {
        "manifest_version": "freqduet-protocol-v6-result-figures-v2",
        "protocol": PROTOCOL,
        "paper_controller": PAPER_CONTROLLER,
        "backend": "python-matplotlib",
        "archetype": "quantitative_grid",
        "submission_ready": False,
        "figures": {
            "figure_2": {
                "stem": "fig2_protocol_v6_confirmation_robustness",
                "width_mm": FIGURE_WIDTH_MM,
                "height_mm": 76.0,
                "outputs": figure2,
                "source_tables": [
                    "tables/table1_v8_confirmation.csv",
                    "tables/table2_v9_longtrain.csv",
                ],
            },
            "figure_3": {
                "stem": "fig3_protocol_v6_external_tradeoff",
                "width_mm": FIGURE_WIDTH_MM,
                "height_mm": 118.0,
                "outputs": figure3,
                "source_tables": ["tables/table3_v9_external_baselines.csv"],
            },
        },
    }
    (out_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    refresh_package_manifest(package_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument(
        "--formats",
        default=",".join(DEFAULT_FORMATS),
        help="comma-separated subset of svg,pdf,tiff,png",
    )
    args = parser.parse_args()
    formats = tuple(item.strip().lower() for item in args.formats.split(",")
                    if item.strip())
    manifest = build_figures(args.package_dir.resolve(), formats)
    print(json.dumps({
        "status": "protocol_v6_figures_complete",
        "package_dir": str(args.package_dir.resolve()),
        "figure_count": len(manifest["figures"]),
        "formats": list(formats),
        "submission_ready": manifest["submission_ready"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
