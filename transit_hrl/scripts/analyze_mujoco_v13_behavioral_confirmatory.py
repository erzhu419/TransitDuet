#!/usr/bin/env python3
"""Integrity-first analysis for the preregistered MuJoCo v13 behavioral comparison."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.mujoco.pilot_analysis import (  # noqa: E402
    PilotCell,
    audit_cells,
    load_cells,
)
from freq_hrl.experiments.reproducibility import derive_seed  # noqa: E402
from scripts import mujoco_v13_behavioral_confirmatory_spec as spec  # noqa: E402


ANALYSIS_VERSION = "mujoco_v13_behavioral_confirmatory_analysis_v1"
BASELINE_ARM = "additive_baseline"
FULL_ARM = "transfer_full_method"
BASELINE_METHOD = str(spec.ARMS[BASELINE_ARM]["method"])
FULL_METHOD = str(spec.ARMS[FULL_ARM]["method"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _verify_analysis_revision(expected_revision: str) -> None:
    revision = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    if revision != str(expected_revision).strip().lower():
        raise ValueError(
            f"analysis revision mismatch: expected {expected_revision}, got {revision}"
        )
    git_root = Path(subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()).resolve()
    paths = [Path(__file__).resolve(), Path(spec.__file__).resolve()]
    relative = [str(path.relative_to(git_root)) for path in paths]
    subprocess.run(
        ["git", "-C", str(git_root), "ls-files", "--error-unmatch", *relative],
        check=True,
        capture_output=True,
        text=True,
    )
    clean = subprocess.run(
        ["git", "-C", str(git_root), "diff", "--quiet", "HEAD", "--", *relative]
    )
    if clean.returncode != 0:
        raise ValueError("analysis or frozen spec does not match committed bytes")


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("confirmatory analysis requires finite nonempty values")
    return array


def _cell_mean(cell: PilotCell, metric: str) -> float:
    try:
        return float(np.mean(_finite(float(row[metric]) for row in cell.rows)))
    except KeyError as exc:
        raise ValueError(f"missing {metric} in {cell.path}") from exc


def _cell_max(cell: PilotCell, metric: str) -> float:
    try:
        return float(np.max(_finite(float(row[metric]) for row in cell.rows)))
    except KeyError as exc:
        raise ValueError(f"missing {metric} in {cell.path}") from exc


def _runtime_identity(cell: PilotCell) -> tuple[str, str, str, str]:
    summary = cell.summary
    return (
        str(summary.get("confirmatory_runtime_revision", "")),
        str(summary.get("confirmatory_launcher_sha256", "")),
        str(summary.get("confirmatory_runtime_sha256", "")),
        str(summary.get("confirmatory_spec_sha256", "")),
    )


def _audit_confirmatory_cells(
    cells: list[PilotCell],
    *,
    arm: str,
    expected_runtime_revision: str,
) -> dict[str, Any]:
    arm_spec = spec.ARMS[arm]
    audit = audit_cells(
        cells,
        expected_protocol_version=spec.FROZEN_CORE_PROTOCOL_VERSION,
        environments=spec.ENVIRONMENTS,
        methods=(str(arm_spec["method"]),),
        optimizer_seeds=spec.OPTIMIZER_SEEDS,
        evaluation_seeds=spec.HELDOUT_EVALUATION_SEEDS,
        disturbance_modes=spec.EVALUATION_DISTURBANCE_MODES,
    )
    issues = list(audit["issues"])
    identities = {_runtime_identity(cell) for cell in cells}
    for cell in cells:
        summary = cell.summary
        prefix = "/".join(map(str, cell.key))
        expected_values = {
            "confirmatory_protocol_version": spec.CONFIRMATORY_PROTOCOL_VERSION,
            "confirmatory_runtime_adapter_version": spec.RUNTIME_ADAPTER_VERSION,
            "confirmatory_arm": arm,
            "confirmatory_evidence_role": "fresh_seed_confirmatory_unanalyzed",
            "code_revision": spec.FROZEN_ALGORITHM_REVISION,
            "source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
            "responsibility_mode": arm_spec["responsibility_mode"],
            "method": arm_spec["method"],
        }
        for key, expected in expected_values.items():
            if summary.get(key) != expected:
                issues.append(f"{prefix}:{key}_mismatch")
        if tuple(map(int, summary.get("train_seeds", ()))) != spec.TRAIN_SEEDS:
            issues.append(f"{prefix}:training_seed_registry_mismatch")
        if tuple(map(int, summary.get("selection_seeds", ()))) != (
            spec.CHECKPOINT_SELECTION_SEEDS
        ):
            issues.append(f"{prefix}:checkpoint_seed_registry_mismatch")
        if tuple(map(int, summary.get("eval_seeds", ()))) != (
            spec.HELDOUT_EVALUATION_SEEDS
        ):
            issues.append(f"{prefix}:heldout_seed_registry_mismatch")
        if arm == FULL_ARM and tuple(map(
            int, summary.get("safe_selector_selection_seeds", ())
        )) != spec.SAFETY_SELECTION_SEEDS:
            issues.append(f"{prefix}:safety_seed_registry_mismatch")
    if len(identities) != 1:
        issues.append(f"{arm}:mixed_runtime_identity")
    elif next(iter(identities))[0] != str(expected_runtime_revision).lower():
        issues.append(f"{arm}:runtime_revision_mismatch")
    audit.update({
        "analysis_version": ANALYSIS_VERSION,
        "evidence_role": "fresh_seed_confirmatory",
        "status": "valid" if not issues else "invalid",
        "issues": sorted(set(issues)),
        "runtime_identities": [list(item) for item in sorted(identities)],
    })
    return audit


def _bootstrap_environment(
    *,
    environment: str,
    baseline_return: np.ndarray,
    full_return: np.ndarray,
    baseline_drift: np.ndarray,
    full_drift: np.ndarray,
    baseline_raw_lower_drift: np.ndarray,
    full_raw_lower_drift: np.ndarray,
    full_upper_hf_power: np.ndarray,
) -> dict[str, float | bool]:
    count = len(baseline_return)
    if not (
        count == len(full_return) == len(baseline_drift) == len(full_drift)
        == len(baseline_raw_lower_drift) == len(full_raw_lower_drift)
        == len(full_upper_hf_power)
        == len(spec.OPTIMIZER_SEEDS)
    ):
        raise ValueError("confirmatory paired replicate matrix is incomplete")
    rng = np.random.default_rng(derive_seed(
        "mujoco_v13_behavioral_confirmatory_primary_bootstrap",
        environment,
    ))
    indices = rng.integers(
        0,
        count,
        size=(spec.BOOTSTRAP_DRAWS, count),
        endpoint=False,
    )
    baseline_return_draw = np.mean(baseline_return[indices], axis=1)
    full_return_draw = np.mean(full_return[indices], axis=1)
    return_difference_draw = full_return_draw - baseline_return_draw
    return_margin_draw = spec.RETURN_NONINFERIORITY_MARGIN_FRACTION * np.maximum(
        np.abs(baseline_return_draw), 1.0
    )
    return_ni_statistic_draw = return_difference_draw + return_margin_draw
    baseline_drift_draw = np.mean(baseline_drift[indices], axis=1)
    full_drift_draw = np.mean(full_drift[indices], axis=1)
    if np.any(baseline_drift_draw <= np.finfo(np.float64).eps):
        raise ValueError("baseline LF drift is zero in a bootstrap draw")
    drift_reduction_draw = 1.0 - full_drift_draw / baseline_drift_draw
    baseline_raw_draw = np.mean(baseline_raw_lower_drift[indices], axis=1)
    full_raw_draw = np.mean(full_raw_lower_drift[indices], axis=1)
    if np.any(baseline_raw_draw <= np.finfo(np.float64).eps):
        raise ValueError("baseline raw lower LF drift is zero in a bootstrap draw")
    raw_drift_reduction_draw = 1.0 - full_raw_draw / baseline_raw_draw
    if np.any(full_upper_hf_power < 0.0):
        raise ValueError("upper HF power must be nonnegative")
    upper_hf_rms_draw = np.sqrt(np.mean(full_upper_hf_power[indices], axis=1))
    primary_tail = 1.0 - spec.PER_GATE_ONE_SIDED_CONFIDENCE
    return_ni_lower = float(np.quantile(return_ni_statistic_draw, primary_tail))
    drift_reduction_lower = float(np.quantile(drift_reduction_draw, primary_tail))
    raw_drift_reduction_lower = float(
        np.quantile(raw_drift_reduction_draw, primary_tail)
    )
    upper_hf_rms_familywise_upper = float(
        np.quantile(upper_hf_rms_draw, spec.PER_GATE_ONE_SIDED_CONFIDENCE)
    )
    return_difference = float(np.mean(full_return) - np.mean(baseline_return))
    return_margin = float(
        spec.RETURN_NONINFERIORITY_MARGIN_FRACTION
        * max(abs(float(np.mean(baseline_return))), 1.0)
    )
    drift_reduction = float(
        1.0 - float(np.mean(full_drift)) / float(np.mean(baseline_drift))
    )
    raw_drift_reduction = float(
        1.0
        - float(np.mean(full_raw_lower_drift))
        / float(np.mean(baseline_raw_lower_drift))
    )
    upper_hf_rms = float(np.sqrt(np.mean(full_upper_hf_power)))
    return_ci95 = np.quantile(return_difference_draw, (0.025, 0.975))
    drift_ci95 = np.quantile(drift_reduction_draw, (0.025, 0.975))
    raw_drift_ci95 = np.quantile(raw_drift_reduction_draw, (0.025, 0.975))
    upper_hf_rms_ci95 = np.quantile(upper_hf_rms_draw, (0.025, 0.975))
    return {
        "optimizer_replicates": count,
        "baseline_episode_return_mean": float(np.mean(baseline_return)),
        "full_method_episode_return_mean": float(np.mean(full_return)),
        "episode_return_difference": return_difference,
        "episode_return_difference_ci95_lower": float(return_ci95[0]),
        "episode_return_difference_ci95_upper": float(return_ci95[1]),
        "return_noninferiority_margin": return_margin,
        "return_ni_familywise_lower_bound": return_ni_lower,
        "return_noninferiority_pass": bool(return_ni_lower >= 0.0),
        "exploratory_return_superiority": bool(return_ci95[0] > 0.0),
        "baseline_LowerLFDriftAbs_mean": float(np.mean(baseline_drift)),
        "full_method_LowerLFDriftAbs_mean": float(np.mean(full_drift)),
        "relative_responsibility_drift_reduction": drift_reduction,
        "relative_responsibility_drift_reduction_ci95_lower": float(drift_ci95[0]),
        "relative_responsibility_drift_reduction_ci95_upper": float(drift_ci95[1]),
        "responsibility_drift_reduction_familywise_lower_bound": (
            drift_reduction_lower
        ),
        "minimum_responsibility_drift_reduction_pass": bool(
            drift_reduction_lower
            >= spec.MINIMUM_RESPONSIBILITY_DRIFT_REDUCTION_FRACTION
        ),
        "baseline_RawLowerLFDriftAbs_mean": float(
            np.mean(baseline_raw_lower_drift)
        ),
        "full_method_RawLowerLFDriftAbs_mean": float(
            np.mean(full_raw_lower_drift)
        ),
        "relative_raw_lower_drift_reduction": raw_drift_reduction,
        "relative_raw_lower_drift_reduction_ci95_lower": float(raw_drift_ci95[0]),
        "relative_raw_lower_drift_reduction_ci95_upper": float(raw_drift_ci95[1]),
        "raw_lower_drift_reduction_familywise_lower_bound": (
            raw_drift_reduction_lower
        ),
        "minimum_raw_lower_drift_reduction_pass": bool(
            raw_drift_reduction_lower
            >= spec.MINIMUM_RAW_LOWER_DRIFT_REDUCTION_FRACTION
        ),
        "full_method_upper_hf_rms": upper_hf_rms,
        "full_method_upper_hf_rms_ci95_lower": float(upper_hf_rms_ci95[0]),
        "full_method_upper_hf_rms_ci95_upper": float(upper_hf_rms_ci95[1]),
        "upper_hf_rms_familywise_upper_bound": upper_hf_rms_familywise_upper,
        "upper_hf_budget_pass": bool(
            upper_hf_rms_familywise_upper <= spec.MAXIMUM_UPPER_HF_RMS
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


def analyze(
    baseline_run: Path,
    full_method_run: Path,
    output_dir: Path,
    *,
    expected_runtime_revision: str,
) -> dict[str, Any]:
    _verify_analysis_revision(expected_runtime_revision)
    baseline_cells = load_cells(baseline_run)
    full_cells = load_cells(full_method_run)
    audits = {
        BASELINE_ARM: _audit_confirmatory_cells(
            baseline_cells,
            arm=BASELINE_ARM,
            expected_runtime_revision=expected_runtime_revision,
        ),
        FULL_ARM: _audit_confirmatory_cells(
            full_cells,
            arm=FULL_ARM,
            expected_runtime_revision=expected_runtime_revision,
        ),
    }
    if any(audit["status"] != "valid" for audit in audits.values()):
        raise ValueError("MuJoCo v13 behavioral confirmatory integrity audit failed")
    runtime_identities = {
        _runtime_identity(cell) for cell in baseline_cells + full_cells
    }
    if len(runtime_identities) != 1:
        raise ValueError("confirmatory arms do not share one runtime identity")

    baseline = {cell.key: cell for cell in baseline_cells}
    full = {cell.key: cell for cell in full_cells}
    paired_rows: list[dict[str, Any]] = []
    environment_rows: list[dict[str, Any]] = []
    checkpoint_issues: list[str] = []
    for environment in spec.ENVIRONMENTS:
        baseline_return = []
        full_return = []
        baseline_drift = []
        full_drift = []
        baseline_raw_lower_drift = []
        full_raw_lower_drift = []
        full_upper_hf_power = []
        reconstruction = []
        selected_branches: Counter[str] = Counter()
        for seed in spec.OPTIMIZER_SEEDS:
            baseline_cell = baseline[(environment, BASELINE_METHOD, seed)]
            full_cell = full[(environment, FULL_METHOD, seed)]
            internal_baseline_hash = str(
                full_cell.summary["safe_selector_branch_training"]
                ["no_leakage"]["selected_parameter_sha256"]
            )
            external_baseline_hash = str(
                baseline_cell.summary["frozen_parameter_sha256"]
            )
            if internal_baseline_hash != external_baseline_hash:
                checkpoint_issues.append(
                    f"{environment}/rep-{seed}:internal_external_baseline_mismatch"
                )
            base_return = _cell_mean(baseline_cell, "episode_return")
            proposed_return = _cell_mean(full_cell, "episode_return")
            base_drift = _cell_mean(baseline_cell, "LowerLFDriftAbs")
            proposed_drift = _cell_mean(full_cell, "LowerLFDriftAbs")
            base_raw_drift = _cell_mean(baseline_cell, "RawLowerLFDriftAbs")
            proposed_raw_drift = _cell_mean(full_cell, "RawLowerLFDriftAbs")
            proposed_upper_hf_power = _cell_mean(full_cell, "UpperHFPowerAbs")
            recon = _cell_max(full_cell, "ResponsibilityReconstructionRMS")
            branch = str(full_cell.summary["safe_selector"]["selected_branch"])
            selected_branches[branch] += 1
            baseline_return.append(base_return)
            full_return.append(proposed_return)
            baseline_drift.append(base_drift)
            full_drift.append(proposed_drift)
            baseline_raw_lower_drift.append(base_raw_drift)
            full_raw_lower_drift.append(proposed_raw_drift)
            full_upper_hf_power.append(proposed_upper_hf_power)
            reconstruction.append(recon)
            paired_rows.append({
                "environment": environment,
                "optimizer_seed": seed,
                "baseline_episode_return": base_return,
                "full_method_episode_return": proposed_return,
                "episode_return_difference": proposed_return - base_return,
                "baseline_LowerLFDriftAbs": base_drift,
                "full_method_LowerLFDriftAbs": proposed_drift,
                "relative_responsibility_drift_reduction": (
                    1.0 - proposed_drift / base_drift
                    if base_drift > np.finfo(np.float64).eps else float("nan")
                ),
                "baseline_RawLowerLFDriftAbs": base_raw_drift,
                "full_method_RawLowerLFDriftAbs": proposed_raw_drift,
                "relative_raw_lower_drift_reduction": (
                    1.0 - proposed_raw_drift / base_raw_drift
                    if base_raw_drift > np.finfo(np.float64).eps
                    else float("nan")
                ),
                "full_method_UpperHFPowerAbs": proposed_upper_hf_power,
                "full_method_upper_hf_rms": float(
                    np.sqrt(proposed_upper_hf_power)
                ),
                "selected_branch": branch,
                "reconstruction_rms_max": recon,
                "baseline_checkpoint_match": (
                    internal_baseline_hash == external_baseline_hash
                ),
            })
        stats = _bootstrap_environment(
            environment=environment,
            baseline_return=_finite(baseline_return),
            full_return=_finite(full_return),
            baseline_drift=_finite(baseline_drift),
            full_drift=_finite(full_drift),
            baseline_raw_lower_drift=_finite(baseline_raw_lower_drift),
            full_raw_lower_drift=_finite(full_raw_lower_drift),
            full_upper_hf_power=_finite(full_upper_hf_power),
        )
        reconstruction_max = float(np.max(_finite(reconstruction)))
        reconstruction_pass = (
            reconstruction_max <= spec.MAXIMUM_RECONSTRUCTION_RMS
        )
        row = {
            "environment": environment,
            **stats,
            "selected_branch_counts": json.dumps(
                dict(sorted(selected_branches.items())), sort_keys=True
            ),
            "constrained_branch_selection_rate": float(
                1.0 - selected_branches["no_leakage"] / len(spec.OPTIMIZER_SEEDS)
            ),
            "responsibility_reconstruction_rms_max": reconstruction_max,
            "reconstruction_gate_pass": bool(reconstruction_pass),
        }
        row["environment_primary_gate_pass"] = bool(
            row["return_noninferiority_pass"]
            and row["minimum_responsibility_drift_reduction_pass"]
            and row["minimum_raw_lower_drift_reduction_pass"]
            and row["upper_hf_budget_pass"]
            and reconstruction_pass
        )
        environment_rows.append(row)
    if checkpoint_issues:
        raise ValueError(
            "confirmatory baseline checkpoint consistency failed: "
            + checkpoint_issues[0]
        )

    primary_pass = all(
        bool(row["environment_primary_gate_pass"])
        for row in environment_rows
    )
    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "status": (
            "confirmatory_supported" if primary_pass
            else "confirmatory_primary_gate_failed"
        ),
        "evidence_role": "fresh_seed_confirmatory",
        "confirmatory_protocol_version": spec.CONFIRMATORY_PROTOCOL_VERSION,
        "frozen_algorithm_revision": spec.FROZEN_ALGORITHM_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "runtime_identity": list(next(iter(runtime_identities))),
        "analysis_revision": str(expected_runtime_revision).lower(),
        "analysis_sha256": _sha256(Path(__file__)),
        "spec_sha256": _sha256(Path(spec.__file__)),
        "integrity_status": "valid",
        "integrity_audits": audits,
        "checkpoint_consistency_status": "verified",
        "optimizer_replicates_per_environment_arm": len(spec.OPTIMIZER_SEEDS),
        "heldout_paths_per_cell": (
            len(spec.HELDOUT_EVALUATION_SEEDS)
            * len(spec.EVALUATION_DISTURBANCE_MODES)
        ),
        "family_wise_alpha": spec.FAMILY_WISE_ALPHA,
        "primary_gate_count": spec.PRIMARY_GATE_COUNT,
        "per_gate_one_sided_confidence": spec.PER_GATE_ONE_SIDED_CONFIDENCE,
        "bootstrap_draws": spec.BOOTSTRAP_DRAWS,
        "return_noninferiority_margin_fraction": (
            spec.RETURN_NONINFERIORITY_MARGIN_FRACTION
        ),
        "minimum_responsibility_drift_reduction_fraction": (
            spec.MINIMUM_RESPONSIBILITY_DRIFT_REDUCTION_FRACTION
        ),
        "minimum_raw_lower_drift_reduction_fraction": (
            spec.MINIMUM_RAW_LOWER_DRIFT_REDUCTION_FRACTION
        ),
        "maximum_upper_hf_rms": spec.MAXIMUM_UPPER_HF_RMS,
        "development_disclosure": spec.DEVELOPMENT_DISCLOSURE,
        "primary_gate_pass": bool(primary_pass),
        "environment_rows": environment_rows,
        "selection_rule": (
            "all_three_environments_pass_bonferroni_familywise_return_"
            "noninferiority_responsibility_drift_raw_lower_drift_and_upper_hf_"
            "budget_plus_reconstruction_integrity"
        ),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "paired_replicate_rows.csv", paired_rows)
    _write_csv(output / "environment_gate_rows.csv", environment_rows)
    (output / "decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    table = [
        "| Environment | Return delta [95% CI] | NI | Responsibility drift "
        "reduction | Raw lower drift reduction | Upper HF RMS |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in environment_rows:
        table.append(
            f"| {row['environment']} | {row['episode_return_difference']:.3f} "
            f"[{row['episode_return_difference_ci95_lower']:.3f}, "
            f"{row['episode_return_difference_ci95_upper']:.3f}] | "
            f"{row['return_noninferiority_pass']} | "
            f"{100.0 * row['relative_responsibility_drift_reduction']:.2f}% "
            f"({row['minimum_responsibility_drift_reduction_pass']}) | "
            f"{100.0 * row['relative_raw_lower_drift_reduction']:.2f}% "
            f"({row['minimum_raw_lower_drift_reduction_pass']}) | "
            f"{row['full_method_upper_hf_rms']:.4f} "
            f"({row['upper_hf_budget_pass']}) |"
        )
    (output / "report.md").write_text("\n".join([
        "# MuJoCo v13 behavioral Confirmatory Decision",
        "",
        f"- status: `{payload['status']}`",
        f"- integrity: `{payload['integrity_status']}`",
        f"- primary family-wise gate: `{payload['primary_gate_pass']}`",
        f"- optimizer replicates per environment and arm: `{len(spec.OPTIMIZER_SEEDS)}`",
        f"- held-out paths per cell: `{payload['heldout_paths_per_cell']}`",
        f"- family-wise alpha: `{spec.FAMILY_WISE_ALPHA}` across "
        f"`{spec.PRIMARY_GATE_COUNT}` statistical gates",
        "",
        *table,
        "",
        "Return superiority is exploratory; the preregistered return claim is noninferiority.",
        "",
        f"Development disclosure: {spec.DEVELOPMENT_DISCLOSURE}",
    ]) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--full-method-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-runtime-revision", required=True)
    args = parser.parse_args()
    payload = analyze(
        args.baseline_run,
        args.full_method_run,
        args.output_dir,
        expected_runtime_revision=args.expected_runtime_revision,
    )
    print(f"mujoco_v13_behavioral_confirmatory status={payload['status']}")


if __name__ == "__main__":
    main()
