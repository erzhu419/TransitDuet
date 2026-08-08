"""Audit the pre-registered MuJoCo v11 canonical-state gate."""

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


ANALYSIS_VERSION = "mujoco_v11_canonical_policy_state_gate_v1"
EXPECTED_METHODS = ("freq_hrl_no_leakage", "freq_hrl_safe_selector")
EXPECTED_OPTIMIZER_SEEDS = (35207, 35211, 35227)
EXACT_PAIR_TOLERANCE = 1e-7
SAFE_RETURN_MARGIN_FRACTION = 0.02
MINIMUM_DRIFT_REDUCTION_FRACTION = 0.10
MAXIMUM_RECONSTRUCTION_RMS = 1e-7
EXPECTED_POLICY_STATE_CONTRACT = (
    "canonical_raw_lf_and_previous_raw_lower_actor_state_v1"
)
EXPECTED_COST_STATE_CONTRACT = (
    "causal_responsibility_anchor_and_lower_lf_cost_critic_only_v1"
)


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("canonical-state analysis requires finite values")
    return array


def _mean(values: Iterable[float]) -> float:
    return float(np.mean(_finite(values)))


def _maximum(values: Iterable[float]) -> float:
    return float(np.max(_finite(values)))


def _row_index(cell: PilotCell) -> dict[tuple[str, int], dict[str, Any]]:
    rows = {
        (str(row["disturbance_mode"]), int(row["seed"])): row
        for row in cell.rows
    }
    if len(rows) != len(cell.rows):
        raise ValueError(f"duplicate held-out MuJoCo rows: {cell.path}")
    return rows


def _paired_maximum(
    additive: PilotCell,
    transfer: PilotCell,
    metric: str,
) -> float:
    left = _row_index(additive)
    right = _row_index(transfer)
    if set(left) != set(right):
        raise ValueError(f"held-out row pairing drifted: {additive.path}")
    try:
        return _maximum(
            abs(float(left[key][metric]) - float(right[key][metric]))
            for key in sorted(left)
        )
    except KeyError as exc:
        raise ValueError(
            f"paired MuJoCo row is missing {metric}: {additive.path}"
        ) from exc


def _cell_mean(cell: PilotCell, metric: str) -> float:
    try:
        return _mean(float(row[metric]) for row in cell.rows)
    except KeyError as exc:
        raise ValueError(
            f"MuJoCo cell is missing {metric}: {cell.path}"
        ) from exc


def _cell_maximum(cell: PilotCell, metric: str) -> float:
    try:
        return _maximum(float(row[metric]) for row in cell.rows)
    except KeyError as exc:
        raise ValueError(
            f"MuJoCo cell is missing {metric}: {cell.path}"
        ) from exc


def summarize_canonical_pair(
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

    for mode, cells in (("additive", additive), ("transfer", transfer)):
        for environment in environments:
            for seed in optimizer_seeds:
                safe = cells[(environment, "freq_hrl_safe_selector", seed)]
                baseline = cells[(environment, "freq_hrl_no_leakage", seed)]
                internal = str(
                    safe.summary["safe_selector_branch_training"]
                    [SAFE_SELECTOR_BASELINE_BRANCH]
                    ["selected_parameter_sha256"]
                )
                external = str(baseline.summary["frozen_parameter_sha256"])
                if internal != external:
                    checkpoint_issues.append(
                        f"{mode}/{environment}/rep-{seed}:"
                        "internal_external_no_leakage_checkpoint_mismatch"
                    )

    for environment in environments:
        paired_noleak_hashes = []
        for seed in optimizer_seeds:
            left = additive[(environment, "freq_hrl_no_leakage", seed)]
            right = transfer[(environment, "freq_hrl_no_leakage", seed)]
            matched = str(left.summary["frozen_parameter_sha256"]) == str(
                right.summary["frozen_parameter_sha256"]
            )
            paired_noleak_hashes.append(matched)

        for method in methods:
            additive_group = [
                additive[(environment, method, seed)]
                for seed in optimizer_seeds
            ]
            transfer_group = [
                transfer[(environment, method, seed)]
                for seed in optimizer_seeds
            ]
            for cell in additive_group + transfer_group:
                if str(cell.summary.get("policy_filter_state_contract")) != (
                    EXPECTED_POLICY_STATE_CONTRACT
                ):
                    raise ValueError(
                        f"canonical actor-state contract drifted: {cell.path}"
                    )
                if str(cell.summary.get("lower_cost_state_contract")) != (
                    EXPECTED_COST_STATE_CONTRACT
                ):
                    raise ValueError(
                        f"responsibility cost-state contract drifted: {cell.path}"
                    )
            for cell in additive_group:
                if str(cell.summary.get("responsibility_mode")) != "additive":
                    raise ValueError(
                        f"additive responsibility label drifted: {cell.path}"
                    )
            for cell in transfer_group:
                if str(cell.summary.get("responsibility_mode")) != (
                    "causal_lf_transfer"
                ):
                    raise ValueError(
                        f"transfer responsibility label drifted: {cell.path}"
                    )
                if not np.isclose(
                    float(cell.summary.get("responsibility_transfer_alpha", -1)),
                    RESPONSIBILITY_TRANSFER_ALPHA,
                ):
                    raise ValueError(f"transfer alpha drifted: {cell.path}")

            additive_return = _mean(
                _cell_mean(cell, "episode_return") for cell in additive_group
            )
            transfer_return = _mean(
                _cell_mean(cell, "episode_return") for cell in transfer_group
            )
            additive_drift = _mean(
                _cell_mean(cell, "LowerLFDriftAbs") for cell in additive_group
            )
            transfer_drift = _mean(
                _cell_mean(cell, "LowerLFDriftAbs") for cell in transfer_group
            )
            return_difference = transfer_return - additive_return
            drift_reduction = (
                (additive_drift - transfer_drift) / additive_drift
                if additive_drift > np.finfo(np.float64).eps
                else 0.0
            )
            reconstruction_max = _maximum(
                _cell_maximum(cell, "ResponsibilityReconstructionRMS")
                for cell in transfer_group
            )
            paired_return_max = _maximum(
                _paired_maximum(left, right, "episode_return")
                for left, right in zip(additive_group, transfer_group)
            )
            paired_raw_action_max = _maximum(
                _paired_maximum(left, right, "RawLowerActionRMS")
                for left, right in zip(additive_group, transfer_group)
            )
            paired_raw_drift_max = _maximum(
                _paired_maximum(left, right, "RawLowerLFDriftAbs")
                for left, right in zip(additive_group, transfer_group)
            )
            if method == "freq_hrl_no_leakage":
                return_pass = paired_return_max <= EXACT_PAIR_TOLERANCE
                hash_pass = all(paired_noleak_hashes)
                raw_pass = max(
                    paired_raw_action_max, paired_raw_drift_max
                ) <= EXACT_PAIR_TOLERANCE
                return_margin = EXACT_PAIR_TOLERANCE
            else:
                return_margin = SAFE_RETURN_MARGIN_FRACTION * max(
                    abs(additive_return), 1.0
                )
                return_pass = return_difference >= -return_margin
                hash_pass = True
                raw_pass = True
            drift_pass = drift_reduction >= MINIMUM_DRIFT_REDUCTION_FRACTION
            reconstruction_pass = (
                reconstruction_max <= MAXIMUM_RECONSTRUCTION_RMS
            )
            rows.append({
                "environment": environment,
                "method": method,
                "optimizer_replicates": len(optimizer_seeds),
                "additive_episode_return_mean": additive_return,
                "transfer_episode_return_mean": transfer_return,
                "episode_return_difference": return_difference,
                "return_gate_margin": return_margin,
                "return_gate_pass": bool(return_pass),
                "paired_episode_return_absolute_difference_max": (
                    paired_return_max
                ),
                "paired_raw_action_rms_absolute_difference_max": (
                    paired_raw_action_max
                ),
                "paired_raw_drift_absolute_difference_max": (
                    paired_raw_drift_max
                ),
                "paired_checkpoint_hash_pass": bool(hash_pass),
                "paired_raw_metric_pass": bool(raw_pass),
                "additive_LowerLFDriftAbs_mean": additive_drift,
                "transfer_LowerLFDriftAbs_mean": transfer_drift,
                "relative_drift_reduction": float(drift_reduction),
                "minimum_drift_reduction_pass": bool(drift_pass),
                "transfer_reconstruction_rms_max": reconstruction_max,
                "reconstruction_gate_pass": bool(reconstruction_pass),
                "environment_gate_pass": bool(
                    return_pass
                    and hash_pass
                    and raw_pass
                    and drift_pass
                    and reconstruction_pass
                ),
            })
    return rows, checkpoint_issues


def canonical_gate_decision(
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
        raise ValueError("canonical-state gate matrix is incomplete")
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
            "canonical_state_gate_passed"
            if one_branch and complete_method
            else "canonical_state_gate_failed"
        ),
        "one_branch_exact_invariance_gate_pass": bool(one_branch),
        "complete_safe_method_gate_pass": bool(complete_method),
        "environment_gate_rows": environment_rows,
        "selection_rule": (
            "exact_noleak_checkpoint_and_path_pairing_plus_10pct_drift;"
            "safe_method_2pct_noninferiority_plus_10pct_drift;"
            "all_reconstruction_rms_at_most_1e-7"
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze_canonical_runs(
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
            raise ValueError(f"canonical-state run {mode} failed integrity audit")
        audits[mode] = audit
        identities.update(tuple(item) for item in audit["source_identities"])
    if (
        len(identities) != 1
        or next(iter(identities))[0] != str(expected_code_revision)
        or next(iter(identities))[2] != "verified"
    ):
        raise ValueError("canonical-state runs do not share one verified source")

    rows, checkpoint_issues = summarize_canonical_pair(
        cells_by_mode["additive"],
        cells_by_mode["causal_lf_transfer"],
        optimizer_seeds=optimizer_seeds,
    )
    if checkpoint_issues:
        raise ValueError(
            "canonical-state checkpoint consistency failed: "
            + checkpoint_issues[0]
        )
    decision = canonical_gate_decision(rows)
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
            "# MuJoCo v11 Canonical-Policy-State Gate",
            "",
            f"- status: `{payload['status']}`",
            "- one-branch exact invariance gate: "
            f"`{payload['one_branch_exact_invariance_gate_pass']}`",
            "- complete-method gate: "
            f"`{payload['complete_safe_method_gate_pass']}`",
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
    payload = analyze_canonical_runs(
        args.additive_run,
        args.transfer_run,
        args.output_dir,
        expected_code_revision=args.expected_code_revision,
    )
    print(f"mujoco_canonical_state_gate status={payload['status']}")


if __name__ == "__main__":
    main()
