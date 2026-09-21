#!/usr/bin/env python3
"""Analyze the paired Stage-5 optimizer-stability development screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import pointmaze_exogenous_stage5_stability_spec as spec


def _mean(rows: list[dict[str, Any]], metric: str) -> float:
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    if values.size != 16 or not np.all(np.isfinite(values)):
        raise ValueError(f"stability screen requires 16 finite {metric} values")
    return float(np.mean(values))


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records = []
    reverse = {
        (
            str(config["checkpoint_rank_mode"]),
            int(config["train_seed_count"]),
        ): arm
        for arm, config in spec.ARM_CONFIG.items()
    }
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        protocol = payload.get("protocol", {})
        cells = payload.get("cells", [])
        if (
            payload.get("status") != "complete"
            or protocol.get("protocol_version") != spec.BASE_PROTOCOL
            or len(cells) != 1
        ):
            raise ValueError(f"invalid stability result: {path}")
        cell = cells[0]
        signature = (
            str(protocol.get("checkpoint_rank_mode")),
            len(protocol.get("train_seeds", [])),
        )
        arm = reverse.get(signature)
        root = int(protocol.get("optimizer_seed", -1))
        if arm is None or root not in spec.OPTIMIZER_SEEDS:
            raise ValueError(f"unregistered stability cell: {path}")
        final_rows = cell.get("evaluation_rows", [])
        initial_rows = cell.get("untrained_evaluation_rows", [])
        final_seeds = [int(row["seed"]) for row in final_rows]
        initial_seeds = [int(row["seed"]) for row in initial_rows]
        expected = list(spec.seed_roles(root)["evaluation"])
        if final_seeds != expected or initial_seeds != expected:
            raise ValueError(f"stability evaluation seeds drifted: {path}")
        if str(cell.get("policy")) != spec.METHOD:
            raise ValueError(f"stability method drifted: {path}")
        records.append({
            "arm": arm,
            "root": root,
            "success": _mean(final_rows, "tracking_success_rate"),
            "return": _mean(final_rows, "episode_return"),
            "initial_success": _mean(initial_rows, "tracking_success_rate"),
            "initial_return": _mean(initial_rows, "episode_return"),
            "selected_checkpoint_iteration": int(
                cell["selected_checkpoint_iteration"]
            ),
            "runtime_versions": cell.get("runtime_versions"),
        })
    return records


def analyze_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(records)
    indexed = {(str(row["arm"]), int(row["root"])): row for row in items}
    expected = {(arm, root) for arm in spec.ARMS for root in spec.OPTIMIZER_SEEDS}
    if set(indexed) != expected or len(indexed) != len(items):
        raise ValueError("stability screen is incomplete or duplicated")
    runtimes = {
        json.dumps(row.get("runtime_versions"), sort_keys=True) for row in items
    }
    if len(runtimes) != 1 or "null" in runtimes:
        raise ValueError("stability-screen runtime versions differ or are missing")

    summaries: dict[str, Any] = {}
    for arm in spec.ARMS:
        rows = [indexed[(arm, root)] for root in spec.OPTIMIZER_SEEDS]
        success = [float(row["success"]) for row in rows]
        returns = [float(row["return"]) for row in rows]
        success_gain = [
            float(row["success"]) - float(row["initial_success"]) for row in rows
        ]
        return_gain = [
            float(row["return"]) - float(row["initial_return"]) for row in rows
        ]
        summaries[arm] = {
            "root_results": {str(row["root"]): dict(row) for row in rows},
            "worst_root_success": min(success),
            "mean_root_success": float(np.mean(success)),
            "mean_root_return": float(np.mean(returns)),
            "minimum_root_success_gain_vs_untrained": min(success_gain),
            "minimum_root_return_gain_vs_untrained": min(return_gain),
        }

    control = summaries["v1_control"]
    eligible: list[str] = []
    for arm in spec.ARMS[1:]:
        summary = summaries[arm]
        paired_success_gains = {
            str(root): (
                float(indexed[(arm, root)]["success"])
                - float(indexed[("v1_control", root)]["success"])
            )
            for root in spec.OPTIMIZER_SEEDS
        }
        mean_return_gain = (
            float(summary["mean_root_return"])
            - float(control["mean_root_return"])
        )
        summary["paired_success_gain_vs_control"] = paired_success_gains
        summary["mean_return_gain_vs_control"] = mean_return_gain
        summary["eligible"] = bool(
            min(paired_success_gains.values())
            >= spec.MIN_PAIRED_ROOT_SUCCESS_GAIN
            and mean_return_gain >= 0.0
            and float(summary["minimum_root_success_gain_vs_untrained"]) > 0.0
            and float(summary["minimum_root_return_gain_vs_untrained"]) > 0.0
        )
        if summary["eligible"]:
            eligible.append(arm)

    selected = (
        max(
            eligible,
            key=lambda arm: (
                float(summaries[arm]["worst_root_success"]),
                float(summaries[arm]["mean_root_success"]),
                float(summaries[arm]["mean_root_return"]),
                arm,
            ),
        )
        if eligible else None
    )
    return {
        "analysis_version": spec.PROTOCOL + "_analysis_v1",
        "protocol": spec.PROTOCOL,
        "claim_boundary": "post_hoc_development_selection_not_evidence",
        "cell_count": len(items),
        "summaries": summaries,
        "eligible_candidates": eligible,
        "selected_candidate": selected,
        "stage5_v2_freeze_status": "authorized" if selected else "blocked",
    }


def render_report(analysis: dict[str, Any]) -> str:
    lines = [
        "# PointMaze Stage-5 Optimization-Stability Screen",
        "",
        "This is post-hoc development selection, not paper evidence.",
        "",
        "| Arm | Worst-root success | Mean success | Mean return | Eligible |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in spec.ARMS:
        row = analysis["summaries"][arm]
        lines.append(
            f"| {arm} | {row['worst_root_success']:.3f} "
            f"| {row['mean_root_success']:.3f} "
            f"| {row['mean_root_return']:.3f} "
            f"| {row.get('eligible', False)} |"
        )
    lines.extend([
        "",
        f"Selected candidate: `{analysis['selected_candidate']}`",
        f"Stage-5 V2 freeze: **{analysis['stage5_v2_freeze_status']}**",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    analysis = analyze_records(load_records(args.inputs))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_report(analysis), encoding="utf-8"
    )
    print(json.dumps({
        "selected_candidate": analysis["selected_candidate"],
        "stage5_v2_freeze_status": analysis["stage5_v2_freeze_status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
