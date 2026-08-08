"""Audit and apply the pre-registered MuJoCo v10 responsibility gate."""

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
    RESPONSIBILITY_TRANSFER_ALPHA,
    SAFE_SELECTOR_BASELINE_BRANCH,
)
from freq_hrl.experiments.mujoco.pilot_analysis import (
    PilotCell,
    audit_cells,
    load_cells,
)


ANALYSIS_VERSION = "mujoco_v10_causal_responsibility_transfer_gate_v1"
EXPECTED_METHODS = ("freq_hrl_no_leakage", "freq_hrl_safe_selector")
EXPECTED_OPTIMIZER_SEEDS = (35207, 35211, 35227)
RETURN_NONINFERIORITY_MARGIN_FRACTION = 0.02
MINIMUM_DRIFT_REDUCTION_FRACTION = 0.10
MAXIMUM_RECONSTRUCTION_RMS = 1e-7


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("responsibility analysis requires finite values")
    return float(np.mean(array))


def _maximum(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("responsibility analysis requires finite values")
    return float(np.max(array))


def _cell_metric(cell: PilotCell, key: str) -> float:
    try:
        return _mean(float(row[key]) for row in cell.rows)
    except KeyError as exc:
        raise ValueError(f"MuJoCo cell is missing {key}: {cell.path}") from exc


def _cell_maximum(cell: PilotCell, key: str) -> float:
    try:
        return _maximum(float(row[key]) for row in cell.rows)
    except KeyError as exc:
        raise ValueError(f"MuJoCo cell is missing {key}: {cell.path}") from exc


def summarize_responsibility_pair(
    additive_cells: list[PilotCell],
    transfer_cells: list[PilotCell],
    *,
    environments: Iterable[str] = DEFAULT_ENV_IDS,
    methods: Iterable[str] = EXPECTED_METHODS,
    optimizer_seeds: Iterable[int] = EXPECTED_OPTIMIZER_SEEDS,
) -> tuple[list[dict[str, Any]], list[str]]:
    environments = tuple(map(str, environments))
    methods = tuple(map(str, methods))
    optimizer_seeds = tuple(map(int, optimizer_seeds))
    additive = {cell.key: cell for cell in additive_cells}
    transfer = {cell.key: cell for cell in transfer_cells}
    rows: list[dict[str, Any]] = []
    checkpoint_issues: list[str] = []

    for mode, cells in (("additive", additive), ("causal_lf_transfer", transfer)):
        for environment in environments:
            for seed in optimizer_seeds:
                safe = cells[(environment, "freq_hrl_safe_selector", seed)]
                baseline = cells[(environment, "freq_hrl_no_leakage", seed)]
                internal_hash = str(
                    safe.summary["safe_selector_branch_training"]
                    [SAFE_SELECTOR_BASELINE_BRANCH]["selected_parameter_sha256"]
                )
                external_hash = str(baseline.summary["frozen_parameter_sha256"])
                if internal_hash != external_hash:
                    checkpoint_issues.append(
                        f"{mode}/{environment}/rep-{seed}:"
                        "internal_external_no_leakage_checkpoint_mismatch"
                    )

    for environment in environments:
        for method in methods:
            additive_group = [
                additive[(environment, method, seed)]
                for seed in optimizer_seeds
            ]
            transfer_group = [
                transfer[(environment, method, seed)]
                for seed in optimizer_seeds
            ]
            for cell in additive_group:
                if str(cell.summary.get("responsibility_mode")) != "additive":
                    raise ValueError(f"additive responsibility label drifted: {cell.path}")
            for cell in transfer_group:
                if str(cell.summary.get("responsibility_mode")) != "causal_lf_transfer":
                    raise ValueError(f"transfer responsibility label drifted: {cell.path}")
                if not np.isclose(
                    float(cell.summary.get("responsibility_transfer_alpha", -1.0)),
                    RESPONSIBILITY_TRANSFER_ALPHA,
                ):
                    raise ValueError(f"transfer alpha drifted: {cell.path}")

            additive_return = _mean(
                _cell_metric(cell, "episode_return") for cell in additive_group
            )
            transfer_return = _mean(
                _cell_metric(cell, "episode_return") for cell in transfer_group
            )
            additive_drift = _mean(
                _cell_metric(cell, "LowerLFDriftAbs") for cell in additive_group
            )
            transfer_drift = _mean(
                _cell_metric(cell, "LowerLFDriftAbs") for cell in transfer_group
            )
            return_difference = transfer_return - additive_return
            return_margin = RETURN_NONINFERIORITY_MARGIN_FRACTION * max(
                abs(additive_return), 1.0
            )
            drift_reduction = (
                (additive_drift - transfer_drift) / additive_drift
                if additive_drift > np.finfo(np.float64).eps else 0.0
            )
            reconstruction_max = _maximum(
                _cell_maximum(cell, "ResponsibilityReconstructionRMS")
                for cell in transfer_group
            )
            return_pass = return_difference >= -return_margin
            drift_pass = drift_reduction >= MINIMUM_DRIFT_REDUCTION_FRACTION
            reconstruction_pass = reconstruction_max <= MAXIMUM_RECONSTRUCTION_RMS
            rows.append({
                "environment": environment,
                "method": method,
                "optimizer_replicates": len(optimizer_seeds),
                "additive_episode_return_mean": additive_return,
                "transfer_episode_return_mean": transfer_return,
                "episode_return_difference": return_difference,
                "reward_noninferiority_margin": return_margin,
                "reward_noninferiority_pass": bool(return_pass),
                "additive_LowerLFDriftAbs_mean": additive_drift,
                "transfer_LowerLFDriftAbs_mean": transfer_drift,
                "relative_drift_reduction": float(drift_reduction),
                "minimum_drift_reduction_pass": bool(drift_pass),
                "transfer_RawLowerLFDriftAbs_mean": _mean(
                    _cell_metric(cell, "RawLowerLFDriftAbs")
                    for cell in transfer_group
                ),
                "transfer_ResponsibilityTransferRMS_mean": _mean(
                    _cell_metric(cell, "ResponsibilityTransferRMS")
                    for cell in transfer_group
                ),
                "transfer_headroom_saturation_rate_mean": _mean(
                    _cell_metric(
                        cell,
                        "ResponsibilityTransferHeadroomSaturationRate",
                    )
                    for cell in transfer_group
                ),
                "transfer_lower_contribution_out_of_unit_rate_mean": _mean(
                    _cell_metric(cell, "LowerContributionOutOfUnitRate")
                    for cell in transfer_group
                ),
                "transfer_reconstruction_rms_max": reconstruction_max,
                "reconstruction_gate_pass": bool(reconstruction_pass),
                "environment_gate_pass": bool(
                    return_pass and drift_pass and reconstruction_pass
                ),
            })
    return rows, checkpoint_issues


def responsibility_gate_decision(
    environment_rows: list[dict[str, Any]],
    *,
    environments: Iterable[str] = DEFAULT_ENV_IDS,
) -> dict[str, Any]:
    environments = tuple(map(str, environments))
    expected = {
        (environment, method)
        for environment in environments
        for method in EXPECTED_METHODS
    }
    observed = {
        (str(row["environment"]), str(row["method"]))
        for row in environment_rows
    }
    if observed != expected:
        raise ValueError("responsibility gate environment-method matrix is incomplete")
    one_branch = all(
        bool(row["environment_gate_pass"])
        for row in environment_rows
        if row["method"] == "freq_hrl_no_leakage"
    )
    complete_method = all(
        bool(row["environment_gate_pass"])
        for row in environment_rows
        if row["method"] == "freq_hrl_safe_selector"
    )
    return {
        "status": (
            "causal_transfer_gate_passed"
            if one_branch and complete_method
            else "causal_transfer_gate_failed"
        ),
        "one_branch_structural_gate_pass": bool(one_branch),
        "complete_safe_method_gate_pass": bool(complete_method),
        "environment_gate_rows": environment_rows,
        "selection_rule": (
            "all_environments_reward_noninferior_2pct_drift_reduction_10pct_"
            "and_reconstruction_rms_at_most_1e-7_for_both_methods"
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze_responsibility_runs(
    additive_run: Path,
    transfer_run: Path,
    output_dir: Path,
    *,
    expected_code_revision: str,
    optimizer_seeds: Iterable[int] = EXPECTED_OPTIMIZER_SEEDS,
) -> dict[str, Any]:
    optimizer_seeds = tuple(map(int, optimizer_seeds))
    cells_by_mode = {
        "additive": load_cells(additive_run),
        "causal_lf_transfer": load_cells(transfer_run),
    }
    audits: dict[str, dict[str, Any]] = {}
    identities: set[tuple[str, str, str]] = set()
    for mode, cells in cells_by_mode.items():
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
            raise ValueError(f"responsibility run {mode} failed integrity audit")
        audits[mode] = audit
        identities.update(tuple(item) for item in audit["source_identities"])
    if (
        len(identities) != 1
        or next(iter(identities))[0] != str(expected_code_revision)
        or next(iter(identities))[2] != "verified"
    ):
        raise ValueError("responsibility runs do not share one verified source")

    rows, checkpoint_issues = summarize_responsibility_pair(
        cells_by_mode["additive"],
        cells_by_mode["causal_lf_transfer"],
        optimizer_seeds=optimizer_seeds,
    )
    if checkpoint_issues:
        raise ValueError(
            "responsibility checkpoint consistency failed: "
            + checkpoint_issues[0]
        )
    decision = responsibility_gate_decision(rows)
    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "evidence_role": "development_only_not_claim_eligible",
        "expected_code_revision": str(expected_code_revision),
        "protocol_version": MUJOCO_CONTROL_PROTOCOL_VERSION,
        "checkpoint_consistency_status": "verified",
        "integrity_audits": audits,
        **decision,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "environment_gate_rows.csv", rows)
    (output / "decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "report.md").write_text(
        "\n".join([
            "# MuJoCo v10 Causal Responsibility-Transfer Gate",
            "",
            f"- status: `{payload['status']}`",
            f"- one-branch gate: `{payload['one_branch_structural_gate_pass']}`",
            f"- complete-method gate: `{payload['complete_safe_method_gate_pass']}`",
            f"- source revision: `{payload['expected_code_revision']}`",
            "- evidence role: `development_only_not_claim_eligible`",
            "",
            "Fresh confirmatory seeds remain forbidden unless both gates pass.",
        ]) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--additive-run", type=Path, required=True)
    parser.add_argument("--transfer-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-code-revision", required=True)
    args = parser.parse_args()
    payload = analyze_responsibility_runs(
        args.additive_run,
        args.transfer_run,
        args.output_dir,
        expected_code_revision=args.expected_code_revision,
    )
    print(f"mujoco_responsibility_gate status={payload['status']}")


if __name__ == "__main__":
    main()
