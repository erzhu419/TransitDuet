#!/usr/bin/env python3
"""Fresh-root analysis of the PointMaze multiscale-HRL factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_pointmaze_exogenous_routing_stage6 import (  # noqa: E402
    METRICS,
    _index_rows,
    _interaction_effect,
    _interval,
    _paired_effect,
    _root_means,
    _validate_cells,
    _validate_row_pairing,
    load_cells,
)
from scripts.pointmaze_exogenous_multiscale_stage7_spec import (  # noqa: E402
    METHODS,
    PROTOCOL,
)


MIN_HRL_MULTISCALE_SUCCESS = 0.50
CONTRASTS = {
    "flat_multiscale_vs_history": (
        "flat_exogenous_multiscale_all", "flat_exogenous_history"
    ),
    "hrl_history_vs_flat_history": (
        "hrl_exogenous_history", "flat_exogenous_history"
    ),
    "hrl_multiscale_vs_history": (
        "hrl_exogenous_multiscale_all", "hrl_exogenous_history"
    ),
    "hrl_multiscale_vs_flat_multiscale": (
        "hrl_exogenous_multiscale_all", "flat_exogenous_multiscale_all"
    ),
}


def analyze_stage7(
    cells: Iterable[dict[str, Any]],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    items = list(cells)
    roots = _validate_cells(items, methods=METHODS)
    final_rows = _index_rows(items, field="evaluation_rows")
    initial_rows = _index_rows(items, field="untrained_evaluation_rows")
    _validate_row_pairing(final_rows, methods=METHODS)
    _validate_row_pairing(initial_rows, methods=METHODS)
    if set(final_rows) != set(initial_rows):
        raise ValueError("Stage-7 trained and untrained rows are not paired")

    final_means = {
        method: {
            metric: _root_means(final_rows, method=method, metric=metric)
            for metric in METRICS
        }
        for method in METHODS
    }
    initial_means = {
        method: {
            metric: _root_means(initial_rows, method=method, metric=metric)
            for metric in METRICS
        }
        for method in METHODS
    }
    absolute = {
        method: {
            metric: _interval(
                values.values(), confidence=confidence, classify=False
            )
            for metric, values in final_means[method].items()
        }
        for method in METHODS
    }
    learning_gain = {
        method: {
            metric: _paired_effect(
                final_means[method][metric],
                initial_means[method][metric],
                higher_is_better=higher_is_better,
                confidence=confidence,
            )
            for metric, higher_is_better in METRICS.items()
        }
        for method in METHODS
    }
    contrasts = {
        name: {
            "candidate": candidate,
            "baseline": baseline,
            **{
                metric: _paired_effect(
                    final_means[candidate][metric],
                    final_means[baseline][metric],
                    higher_is_better=higher_is_better,
                    confidence=confidence,
                )
                for metric, higher_is_better in METRICS.items()
            },
        }
        for name, (candidate, baseline) in CONTRASTS.items()
    }
    interaction = {
        metric: _interaction_effect(
            final_means,
            metric=metric,
            higher_is_better=higher_is_better,
            confidence=confidence,
        )
        for metric, higher_is_better in METRICS.items()
    }
    candidate = "hrl_exogenous_multiscale_all"
    checks = {
        "hrl_multiscale_absolute_success_ci_lower": bool(
            absolute[candidate]["tracking_success_rate"]["ci_lower"]
            >= MIN_HRL_MULTISCALE_SUCCESS
        ),
        "hrl_multiscale_final_vs_untrained_success": (
            learning_gain[candidate]["tracking_success_rate"]["status"]
            == "supported"
        ),
        "hrl_multiscale_final_vs_untrained_return": (
            learning_gain[candidate]["episode_return"]["status"]
            == "supported"
        ),
        "hrl_multiscale_vs_hrl_history_success": (
            contrasts["hrl_multiscale_vs_history"][
                "tracking_success_rate"
            ]["status"] == "supported"
        ),
        "hrl_multiscale_vs_flat_multiscale_success": (
            contrasts["hrl_multiscale_vs_flat_multiscale"][
                "tracking_success_rate"
            ]["status"] == "supported"
        ),
        "hierarchy_x_multiscale_success_interaction": (
            interaction["tracking_success_rate"]["status"] == "supported"
        ),
    }
    return {
        "analysis_version": "pointmaze_exogenous_multiscale_stage7_analysis_v1",
        "protocol_version": PROTOCOL,
        "confidence": float(confidence),
        "cell_count": len(items),
        "evaluation_row_count": len(final_rows),
        "untrained_evaluation_row_count": len(initial_rows),
        "independent_training_replicate_count": len(roots),
        "statistical_unit": "optimizer_seed_root",
        "minimum_hrl_multiscale_success": MIN_HRL_MULTISCALE_SUCCESS,
        "absolute": absolute,
        "learning_gain_vs_untrained": learning_gain,
        "contrasts": contrasts,
        "hierarchy_x_multiscale_interaction": interaction,
        "claim_gate_checks": checks,
        "multiscale_hrl_confirmation_status": (
            "supported" if all(checks.values()) else "not_supported"
        ),
        "root_means": {
            "final": {
                method: {
                    metric: {str(root): value for root, value in values.items()}
                    for metric, values in metrics.items()
                }
                for method, metrics in final_means.items()
            },
            "untrained": {
                method: {
                    metric: {str(root): value for root, value in values.items()}
                    for metric, values in metrics.items()
                }
                for method, metrics in initial_means.items()
            },
        },
    }


def render_report(analysis: dict[str, Any]) -> str:
    lines = [
        "# PointMaze Exogenous Multiscale Stage-7 Analysis",
        "",
        f"Runtime protocol: `{analysis['protocol_version']}`",
        (
            "Independent optimizer roots: "
            f"{analysis['independent_training_replicate_count']}"
        ),
        (
            "Fresh multiscale-HRL confirmation: "
            f"**{analysis['multiscale_hrl_confirmation_status']}**"
        ),
        "",
        "| Method | Tracking success mean [95% CI] | Return | RMSE |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        row = analysis["absolute"][method]
        success = row["tracking_success_rate"]
        lines.append(
            f"| {method} | {success['mean']:.3f} "
            f"[{success['ci_lower']:.3f}, {success['ci_upper']:.3f}] "
            f"| {row['episode_return']['mean']:.3f} "
            f"| {row['tracking_rmse']['mean']:.3f} |"
        )
    lines.extend([
        "",
        "| Root-paired contrast | Success improvement [95% CI] | Status |",
        "|---|---:|---|",
    ])
    for name, row in analysis["contrasts"].items():
        success = row["tracking_success_rate"]
        lines.append(
            f"| {name} | {success['mean_improvement']:.3f} "
            f"[{success['ci_lower']:.3f}, {success['ci_upper']:.3f}] "
            f"| {success['status']} |"
        )
    interaction = analysis["hierarchy_x_multiscale_interaction"][
        "tracking_success_rate"
    ]
    lines.extend([
        "",
        (
            "Hierarchy x multiscale success interaction: "
            f"{interaction['mean_interaction_improvement']:.3f} "
            f"[{interaction['ci_lower']:.3f}, {interaction['ci_upper']:.3f}], "
            f"**{interaction['status']}**."
        ),
        "",
        (
            "The optimizer root is the statistical unit. This fixed 16-root "
            "confirmation cannot be extended after inspecting its result."
        ),
        "",
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analysis = analyze_stage7(
        load_cells(args.inputs), confidence=args.confidence
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_report(analysis), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
