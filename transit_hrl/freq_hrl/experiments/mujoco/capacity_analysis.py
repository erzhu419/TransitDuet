"""Audit and apply the pre-registered MuJoCo v9 role-capacity gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from freq_hrl.domains.mujoco import DISTURBANCE_MODES
from freq_hrl.experiments.mujoco.control_validation import (
    DEFAULT_ENV_IDS,
    DEFAULT_EVAL_SEEDS,
    MUJOCO_CONTROL_PROTOCOL_VERSION,
    SAFE_SELECTOR_BASELINE_BRANCH,
)
from freq_hrl.experiments.mujoco.pilot_analysis import (
    PilotCell,
    audit_cells,
    load_cells,
)


CAPACITY_ANALYSIS_VERSION = "mujoco_v9_global_role_capacity_gate_v1"
REGISTERED_UPPER_SCALES = (0.35, 0.60, 0.80, 1.00)
EXPECTED_METHODS = ("freq_hrl_safe_selector", "freq_hrl_no_leakage")
EXPECTED_OPTIMIZER_SEEDS = (35207, 35211, 35227)
RETURN_NONINFERIORITY_MARGIN_FRACTION = 0.02
MINIMUM_DRIFT_REDUCTION_FRACTION = 0.10
MINIMUM_CONSTRAINED_REPLICATES = 2


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("capacity analysis requires finite non-empty values")
    return float(np.mean(array))


def _cell_metric(cell: PilotCell, key: str) -> float:
    try:
        return _mean(float(row[key]) for row in cell.rows)
    except KeyError as exc:
        raise ValueError(f"MuJoCo cell is missing metric {key}: {cell.path}") from exc


def summarize_capacity_scale(
    scale: float,
    cells: list[PilotCell],
    *,
    environments: Iterable[str] = DEFAULT_ENV_IDS,
    optimizer_seeds: Iterable[int] = EXPECTED_OPTIMIZER_SEEDS,
) -> tuple[list[dict[str, Any]], list[str]]:
    environments = tuple(map(str, environments))
    optimizer_seeds = tuple(map(int, optimizer_seeds))
    by_key = {cell.key: cell for cell in cells}
    rows: list[dict[str, Any]] = []
    checkpoint_issues: list[str] = []
    for environment in environments:
        safe_cells = [
            by_key[(environment, "freq_hrl_safe_selector", seed)]
            for seed in optimizer_seeds
        ]
        baseline_cells = [
            by_key[(environment, "freq_hrl_no_leakage", seed)]
            for seed in optimizer_seeds
        ]
        safe_returns = []
        baseline_returns = []
        safe_drifts = []
        baseline_drifts = []
        safe_clip_rates = []
        baseline_clip_rates = []
        constrained_replicates = 0
        selected_branches: list[str] = []
        for safe_cell, baseline_cell in zip(safe_cells, baseline_cells):
            safe_scale = float(safe_cell.summary["upper_action_scale"])
            baseline_scale = float(baseline_cell.summary["upper_action_scale"])
            if not (
                np.isclose(safe_scale, scale)
                and np.isclose(baseline_scale, scale)
            ):
                raise ValueError(
                    f"capacity scale mismatch for {environment}: "
                    f"expected={scale} safe={safe_scale} baseline={baseline_scale}"
                )
            internal_baseline_hash = str(
                safe_cell.summary["safe_selector_branch_training"]
                [SAFE_SELECTOR_BASELINE_BRANCH]["selected_parameter_sha256"]
            )
            external_baseline_hash = str(
                baseline_cell.summary["frozen_parameter_sha256"]
            )
            if internal_baseline_hash != external_baseline_hash:
                checkpoint_issues.append(
                    f"{environment}/rep-{safe_cell.optimizer_seed}:"
                    "internal_external_no_leakage_checkpoint_mismatch"
                )
            selected_branch = str(
                safe_cell.summary["safe_selector"]["selected_branch"]
            )
            selected_branches.append(selected_branch)
            constrained_replicates += int(
                selected_branch != SAFE_SELECTOR_BASELINE_BRANCH
            )
            safe_returns.append(_cell_metric(safe_cell, "episode_return"))
            baseline_returns.append(
                _cell_metric(baseline_cell, "episode_return")
            )
            safe_drifts.append(_cell_metric(safe_cell, "LowerLFDriftAbs"))
            baseline_drifts.append(
                _cell_metric(baseline_cell, "LowerLFDriftAbs")
            )
            safe_clip_rates.append(
                _cell_metric(safe_cell, "AdditiveActionClipRate")
            )
            baseline_clip_rates.append(
                _cell_metric(baseline_cell, "AdditiveActionClipRate")
            )

        safe_return = _mean(safe_returns)
        baseline_return = _mean(baseline_returns)
        safe_drift = _mean(safe_drifts)
        baseline_drift = _mean(baseline_drifts)
        return_margin = RETURN_NONINFERIORITY_MARGIN_FRACTION * max(
            abs(baseline_return), 1.0
        )
        return_difference = safe_return - baseline_return
        drift_reduction = (
            (baseline_drift - safe_drift) / baseline_drift
            if baseline_drift > np.finfo(np.float64).eps else 0.0
        )
        return_gate = return_difference >= -return_margin
        drift_gate = drift_reduction >= MINIMUM_DRIFT_REDUCTION_FRACTION
        selection_gate = (
            constrained_replicates >= MINIMUM_CONSTRAINED_REPLICATES
        )
        clip_complete = all(np.isfinite(
            [*safe_clip_rates, *baseline_clip_rates]
        ))
        rows.append({
            "upper_action_scale": float(scale),
            "environment": environment,
            "optimizer_replicates": len(optimizer_seeds),
            "safe_episode_return_mean": safe_return,
            "no_leakage_episode_return_mean": baseline_return,
            "episode_return_difference": return_difference,
            "reward_noninferiority_margin": return_margin,
            "reward_noninferiority_pass": bool(return_gate),
            "safe_LowerLFDriftAbs_mean": safe_drift,
            "no_leakage_LowerLFDriftAbs_mean": baseline_drift,
            "relative_drift_reduction": float(drift_reduction),
            "minimum_drift_reduction_pass": bool(drift_gate),
            "constrained_branch_replicates": constrained_replicates,
            "selected_branches": "|".join(selected_branches),
            "constrained_replicate_gate_pass": bool(selection_gate),
            "safe_AdditiveActionClipRate_mean": _mean(safe_clip_rates),
            "no_leakage_AdditiveActionClipRate_mean": _mean(
                baseline_clip_rates
            ),
            "clip_diagnostic_complete": bool(clip_complete),
            "environment_gate_pass": bool(
                return_gate and drift_gate and selection_gate and clip_complete
            ),
        })
    return rows, checkpoint_issues


def capacity_gate_decision(
    environment_rows: list[dict[str, Any]],
    *,
    scales: Iterable[float] = REGISTERED_UPPER_SCALES,
    environments: Iterable[str] = DEFAULT_ENV_IDS,
) -> dict[str, Any]:
    scales = tuple(map(float, scales))
    environments = tuple(map(str, environments))
    by_scale: dict[float, list[dict[str, Any]]] = {
        scale: [
            row for row in environment_rows
            if np.isclose(float(row["upper_action_scale"]), scale)
        ]
        for scale in scales
    }
    scale_rows = []
    eligible_scales = []
    for scale in scales:
        rows = by_scale[scale]
        if {str(row["environment"]) for row in rows} != set(environments):
            raise ValueError(f"capacity scale {scale} has an incomplete environment set")
        gate_pass = all(bool(row["environment_gate_pass"]) for row in rows)
        worst_drift_reduction = min(
            float(row["relative_drift_reduction"]) for row in rows
        )
        pooled_return_difference = _mean(
            float(row["episode_return_difference"]) for row in rows
        )
        scale_rows.append({
            "upper_action_scale": scale,
            "all_environment_gate_pass": gate_pass,
            "worst_environment_relative_drift_reduction": (
                worst_drift_reduction
            ),
            "equal_environment_episode_return_difference": (
                pooled_return_difference
            ),
        })
        if gate_pass:
            eligible_scales.append(scale)
    selected_scale = (
        max(
            eligible_scales,
            key=lambda scale: (
                next(
                    row["worst_environment_relative_drift_reduction"]
                    for row in scale_rows
                    if np.isclose(row["upper_action_scale"], scale)
                ),
                next(
                    row["equal_environment_episode_return_difference"]
                    for row in scale_rows
                    if np.isclose(row["upper_action_scale"], scale)
                ),
            ),
        )
        if eligible_scales else None
    )
    return {
        "status": (
            "global_capacity_selected"
            if selected_scale is not None else "no_global_scale_passed"
        ),
        "selected_upper_action_scale": selected_scale,
        "registered_upper_action_scales": list(scales),
        "environment_gate_rows": environment_rows,
        "scale_gate_rows": scale_rows,
        "selection_rule": (
            "all_environments_pass_return_2pct_drift_10pct_and_2of3_"
            "constrained; maximize_worst_environment_drift_reduction_then_"
            "equal_environment_return"
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze_capacity_runs(
    runs_by_scale: dict[float, Path],
    output_dir: Path,
    *,
    expected_code_revision: str,
    optimizer_seeds: Iterable[int] = EXPECTED_OPTIMIZER_SEEDS,
) -> dict[str, Any]:
    if set(map(float, runs_by_scale)) != set(REGISTERED_UPPER_SCALES):
        raise ValueError("capacity analysis requires the complete registered scale set")
    optimizer_seeds = tuple(map(int, optimizer_seeds))
    audits = {}
    environment_rows: list[dict[str, Any]] = []
    checkpoint_issues: list[str] = []
    for scale in REGISTERED_UPPER_SCALES:
        run_dir = Path(runs_by_scale[scale])
        cells = load_cells(run_dir)
        audit = audit_cells(
            cells,
            expected_protocol_version=MUJOCO_CONTROL_PROTOCOL_VERSION,
            environments=DEFAULT_ENV_IDS,
            methods=EXPECTED_METHODS,
            optimizer_seeds=optimizer_seeds,
            evaluation_seeds=DEFAULT_EVAL_SEEDS,
            disturbance_modes=DISTURBANCE_MODES,
        )
        if audit["status"] != "valid":
            raise ValueError(f"capacity run {scale} failed integrity audit")
        identities = audit["source_identities"]
        if len(identities) != 1 or str(identities[0][0]) != str(
            expected_code_revision
        ):
            raise ValueError(f"capacity run {scale} source revision drifted")
        rows, issues = summarize_capacity_scale(
            scale,
            cells,
            optimizer_seeds=optimizer_seeds,
        )
        audits[str(scale)] = audit
        environment_rows.extend(rows)
        checkpoint_issues.extend(issues)
    if checkpoint_issues:
        raise ValueError(
            "capacity checkpoint consistency failed: " + checkpoint_issues[0]
        )
    decision = capacity_gate_decision(environment_rows)
    payload = {
        "analysis_version": CAPACITY_ANALYSIS_VERSION,
        "evidence_role": "development_only_not_claim_eligible",
        "expected_code_revision": str(expected_code_revision),
        "protocol_version": MUJOCO_CONTROL_PROTOCOL_VERSION,
        "checkpoint_consistency_status": "verified",
        "integrity_audits": audits,
        **decision,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "environment_gate_rows.csv", environment_rows)
    _write_csv(output / "scale_gate_rows.csv", decision["scale_gate_rows"])
    (output / "decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# MuJoCo v9 Global Role-Capacity Gate",
        "",
        f"- status: `{payload['status']}`",
        f"- selected upper scale: `{payload['selected_upper_action_scale']}`",
        f"- source revision: `{payload['expected_code_revision']}`",
        "- evidence role: `development_only_not_claim_eligible`",
        "",
        "No fresh confirmatory seeds may be used unless one global scale passes.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _parse_run(value: str) -> tuple[float, Path]:
    scale_text, separator, path_text = str(value).partition("=")
    if not separator or not path_text:
        raise argparse.ArgumentTypeError("--run must be SCALE=PATH")
    return float(scale_text), Path(path_text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-code-revision", required=True)
    args = parser.parse_args()
    runs = {float(scale): path for scale, path in args.run}
    payload = analyze_capacity_runs(
        runs,
        args.output_dir,
        expected_code_revision=args.expected_code_revision,
    )
    print(
        f"mujoco_capacity_gate status={payload['status']} "
        f"selected={payload['selected_upper_action_scale']}"
    )


if __name__ == "__main__":
    main()
