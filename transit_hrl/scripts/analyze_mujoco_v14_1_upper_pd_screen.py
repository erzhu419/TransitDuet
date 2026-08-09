#!/usr/bin/env python3
"""Analyze the frozen MuJoCo v14.1 screen as development evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.reproducibility import derive_seed  # noqa: E402
from scripts import mujoco_v14_1_upper_pd_screen_spec as spec  # noqa: E402


ANALYSIS_VERSION = "mujoco_v14_1_upper_pd_screen_analysis_v1"
METRICS = (
    "episode_return",
    "LowerLFDriftAbs",
    "RawLowerLFDriftAbs",
    "UpperHFPowerAbs",
    "ResponsibilityReconstructionRMS",
)


def _cell_dir(
    run_dir: Path,
    *,
    environment: str,
    arm: str,
    seed: int,
) -> Path:
    return run_dir / "cells" / environment / arm / f"replicate_{int(seed)}"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = (
        len(spec.DEVELOPMENT_EVALUATION_SEEDS)
        * len(spec.EVALUATION_DISTURBANCE_MODES)
    )
    if len(rows) != expected:
        raise ValueError(
            f"v14.1 cell has {len(rows)} rows; expected {expected}: {path}"
        )
    path_keys = [
        (str(row["disturbance_mode"]), int(row["seed"])) for row in rows
    ]
    expected_keys = {
        (mode, int(seed))
        for mode in spec.EVALUATION_DISTURBANCE_MODES
        for seed in spec.DEVELOPMENT_EVALUATION_SEEDS
    }
    if len(set(path_keys)) != len(path_keys) or set(path_keys) != expected_keys:
        raise ValueError(f"v14.1 evaluation path registry mismatch: {path}")
    return rows


def _finite_mean(rows: list[dict[str, str]], metric: str) -> float:
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite v14.1 metric: {metric}")
    return float(np.mean(values))


def _bounds(
    values: np.ndarray,
    *,
    confidence: float,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("v14.1 bootstrap requires finite optimizer replicates")
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(
        0, array.size, size=(int(draws), array.size)
    )
    means = np.mean(array[indices], axis=1)
    alpha = 1.0 - float(confidence)
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def _input_sha256(run_dir: Path) -> str:
    digest = hashlib.sha256()
    files = [
        run_dir / "preregistration.json",
        run_dir / "merged" / "cell_manifest.json",
    ]
    for environment in spec.ENVIRONMENTS:
        for arm in spec.ARMS:
            for seed in spec.OPTIMIZER_SEEDS:
                cell = _cell_dir(
                    run_dir,
                    environment=environment,
                    arm=arm,
                    seed=seed,
                )
                files.extend((
                    cell / "cell_summary.json",
                    cell / "evaluation_rows.csv",
                ))
    for path in sorted(files):
        relative = path.relative_to(run_dir)
        content = path.read_bytes()
        encoded = str(relative).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def load_replicates(
    run_dir: Path,
) -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
    preregistration = json.loads(
        (run_dir / "preregistration.json").read_text(encoding="utf-8")
    )
    if (
        preregistration.get("status")
        != "frozen_before_v14_1_development_outcome_access"
        or preregistration.get("development_protocol_version")
        != spec.DEVELOPMENT_PROTOCOL_VERSION
        or preregistration.get("frozen_algorithm_revision")
        != spec.FROZEN_ALGORITHM_REVISION
        or preregistration.get("frozen_source_manifest_sha256")
        != spec.FROZEN_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("v14.1 preregistration identity mismatch")
    manifest = json.loads(
        (run_dir / "merged" / "cell_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    expected_cells = (
        len(spec.ENVIRONMENTS) * len(spec.ARMS) * len(spec.OPTIMIZER_SEEDS)
    )
    if (
        manifest.get("status") != "development_screen_complete_unanalyzed"
        or int(manifest.get("cell_count", -1)) != expected_cells
    ):
        raise ValueError("v14.1 analysis requires a complete merged manifest")

    output: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    for environment in spec.ENVIRONMENTS:
        output[environment] = {}
        for arm in spec.ARMS:
            by_mode = {
                mode: {metric: [] for metric in METRICS}
                for mode in spec.EVALUATION_DISTURBANCE_MODES
            }
            for seed in spec.OPTIMIZER_SEEDS:
                rows = _read_rows(
                    _cell_dir(
                        run_dir,
                        environment=environment,
                        arm=arm,
                        seed=seed,
                    ) / "evaluation_rows.csv"
                )
                for mode in spec.EVALUATION_DISTURBANCE_MODES:
                    mode_rows = [
                        row for row in rows
                        if str(row["disturbance_mode"]) == mode
                    ]
                    for metric in METRICS:
                        value = (
                            max(float(row[metric]) for row in mode_rows)
                            if metric == "ResponsibilityReconstructionRMS"
                            else _finite_mean(mode_rows, metric)
                        )
                        by_mode[mode][metric].append(value)
            output[environment][arm] = {
                mode: {
                    metric: np.asarray(values, dtype=np.float64)
                    for metric, values in collected.items()
                }
                for mode, collected in by_mode.items()
            }
    return output


def analyze(run_dir: Path) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    replicates = load_replicates(run)
    rows: list[dict[str, Any]] = []
    arm_status: dict[str, dict[str, Any]] = {}
    for arm in spec.CANDIDATE_ARMS:
        gate_rows = []
        for environment in spec.ENVIRONMENTS:
            for mode in spec.EVALUATION_DISTURBANCE_MODES:
                baseline = replicates[environment][spec.BASELINE_ARM][mode]
                candidate = replicates[environment][arm][mode]
                baseline_return = float(np.mean(baseline["episode_return"]))
                baseline_drift = float(np.mean(baseline["LowerLFDriftAbs"]))
                baseline_raw = float(np.mean(baseline["RawLowerLFDriftAbs"]))
                reward_margin = (
                    spec.RETURN_NONINFERIORITY_MARGIN_FRACTION
                    * max(abs(baseline_return), 1.0)
                )
                required_drift = (
                    spec.MINIMUM_RESPONSIBILITY_DRIFT_REDUCTION_FRACTION
                    * baseline_drift
                )
                required_raw = (
                    spec.MINIMUM_RAW_LOWER_DRIFT_REDUCTION_FRACTION
                    * baseline_raw
                )
                reward_lower, reward_upper = _bounds(
                    candidate["episode_return"] - baseline["episode_return"],
                    confidence=spec.SELECTION_CONFIDENCE,
                    draws=spec.BOOTSTRAP_DRAWS,
                    seed=derive_seed(
                        ANALYSIS_VERSION, arm, environment, mode, "return"
                    ),
                )
                _, drift_upper = _bounds(
                    candidate["LowerLFDriftAbs"]
                    - baseline["LowerLFDriftAbs"],
                    confidence=spec.SELECTION_CONFIDENCE,
                    draws=spec.BOOTSTRAP_DRAWS,
                    seed=derive_seed(
                        ANALYSIS_VERSION,
                        arm,
                        environment,
                        mode,
                        "responsibility",
                    ),
                )
                _, raw_upper = _bounds(
                    candidate["RawLowerLFDriftAbs"]
                    - baseline["RawLowerLFDriftAbs"],
                    confidence=spec.SELECTION_CONFIDENCE,
                    draws=spec.BOOTSTRAP_DRAWS,
                    seed=derive_seed(
                        ANALYSIS_VERSION, arm, environment, mode, "raw_lower"
                    ),
                )
                _, power_upper = _bounds(
                    candidate["UpperHFPowerAbs"],
                    confidence=spec.SELECTION_CONFIDENCE,
                    draws=spec.BOOTSTRAP_DRAWS,
                    seed=derive_seed(
                        ANALYSIS_VERSION, arm, environment, mode, "upper_hf"
                    ),
                )
                upper_hf_rms = float(np.sqrt(np.mean(
                    candidate["UpperHFPowerAbs"]
                )))
                upper_hf_rms_upper = float(np.sqrt(max(power_upper, 0.0)))
                reward_pass = reward_lower >= -reward_margin
                drift_pass = drift_upper <= -required_drift
                raw_pass = raw_upper <= -required_raw
                upper_pass = (
                    upper_hf_rms_upper <= spec.UPPER_HF_REPORTING_GATE
                )
                reconstruction_max = float(np.max(
                    candidate["ResponsibilityReconstructionRMS"]
                ))
                reconstruction_pass = (
                    reconstruction_max <= spec.MAXIMUM_RECONSTRUCTION_RMS
                )
                slacks = (
                    (reward_lower + reward_margin)
                    / max(reward_margin, 1e-12),
                    (-required_drift - drift_upper)
                    / max(required_drift, 1e-12),
                    (-required_raw - raw_upper) / max(required_raw, 1e-12),
                    (spec.UPPER_HF_REPORTING_GATE - upper_hf_rms_upper)
                    / spec.UPPER_HF_REPORTING_GATE,
                )
                row = {
                    "environment": environment,
                    "disturbance_mode": mode,
                    "arm": arm,
                    "upper_constraint_mode": spec.ARMS[arm][
                        "upper_constraint_mode"
                    ],
                    "upper_dual_lr": float(spec.ARMS[arm]["upper_dual_lr"]),
                    "upper_hf_penalty_coef": float(
                        spec.ARMS[arm]["upper_hf_penalty_coef"]
                    ),
                    "replicate_count": len(spec.OPTIMIZER_SEEDS),
                    "baseline_return_mean": baseline_return,
                    "candidate_return_mean": float(np.mean(
                        candidate["episode_return"]
                    )),
                    "return_difference_mean": float(np.mean(
                        candidate["episode_return"]
                        - baseline["episode_return"]
                    )),
                    "return_difference_one_sided_lower": reward_lower,
                    "return_difference_one_sided_upper": reward_upper,
                    "reward_noninferiority_margin": reward_margin,
                    "reward_noninferiority_pass": bool(reward_pass),
                    "responsibility_drift_one_sided_upper": drift_upper,
                    "required_responsibility_drift_reduction": required_drift,
                    "responsibility_drift_pass": bool(drift_pass),
                    "raw_lower_drift_one_sided_upper": raw_upper,
                    "required_raw_lower_drift_reduction": required_raw,
                    "raw_lower_drift_pass": bool(raw_pass),
                    "upper_hf_rms": upper_hf_rms,
                    "upper_hf_rms_one_sided_upper": upper_hf_rms_upper,
                    "upper_hf_budget_pass": bool(upper_pass),
                    "reconstruction_rms_max": reconstruction_max,
                    "reconstruction_integrity_pass": bool(
                        reconstruction_pass
                    ),
                    "minimum_normalized_safety_slack": float(min(slacks)),
                    "condition_gate_pass": bool(
                        reward_pass
                        and drift_pass
                        and raw_pass
                        and upper_pass
                        and reconstruction_pass
                    ),
                }
                rows.append(row)
                gate_rows.append(row)
        arm_status[arm] = {
            "all_environment_condition_gates_pass": bool(all(
                row["condition_gate_pass"] for row in gate_rows
            )),
            "passed_gate_count": int(sum(
                bool(row["condition_gate_pass"]) for row in gate_rows
            )),
            "total_gate_count": len(gate_rows),
            "minimum_normalized_safety_slack": float(min(
                row["minimum_normalized_safety_slack"] for row in gate_rows
            )),
            "mean_return_lower_bound": float(np.mean([
                row["return_difference_one_sided_lower"] for row in gate_rows
            ])),
        }
    eligible = [
        arm for arm, status in arm_status.items()
        if status["all_environment_condition_gates_pass"]
    ]
    selected = (
        max(
            eligible,
            key=lambda arm: (
                arm_status[arm]["minimum_normalized_safety_slack"],
                arm_status[arm]["mean_return_lower_bound"],
                arm,
            ),
        )
        if eligible else None
    )
    return {
        "analysis_version": ANALYSIS_VERSION,
        "status": (
            "development_candidate_selected"
            if selected is not None else "no_behavior_safe_candidate"
        ),
        "evidence_role": "development_screen_not_confirmatory",
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "selected_arm": selected,
        "selected_arm_spec": None if selected is None else spec.ARMS[selected],
        "eligible_arms": eligible,
        "arm_status": arm_status,
        "environment_condition_rows": rows,
        "gate_granularity": "environment_by_disturbance_mode",
        "selection_confidence": spec.SELECTION_CONFIDENCE,
        "bootstrap_draws": spec.BOOTSTRAP_DRAWS,
        "input_sha256": _input_sha256(run),
        "claim_boundary": (
            "The v14.1 screen may select an algorithm arm for a new v15 "
            "confirmation. It cannot support a confirmatory performance or "
            "behavior claim."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_analysis(output_dir: Path, decision: dict[str, Any]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    decision_path = output / "decision.json"
    rendered = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    if decision_path.exists() and decision_path.read_text(
        encoding="utf-8"
    ) != rendered:
        raise RuntimeError("existing v14.1 screen decision differs")
    decision_path.write_text(rendered, encoding="utf-8")
    _write_csv(
        output / "environment_condition_gates.csv",
        decision["environment_condition_rows"],
    )
    lines = [
        "# MuJoCo v14.1 Upper Primal-Dual Screen",
        "",
        f"- Status: `{decision['status']}`",
        f"- Selected arm: `{decision['selected_arm']}`",
        "- Evidence role: development only; not confirmatory.",
        "- Gate granularity: environment by disturbance mode.",
        "",
        "| arm | passed conditions | all pass | minimum slack |",
        "|---|---:|---:|---:|",
    ]
    for arm, status in decision["arm_status"].items():
        lines.append(
            f"| {arm} | {status['passed_gate_count']}/"
            f"{status['total_gate_count']} | "
            f"{status['all_environment_condition_gates_pass']} | "
            f"{status['minimum_normalized_safety_slack']:.6g} |"
        )
    (output / "screen_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    decision = analyze(args.run_dir)
    write_analysis(args.output_dir, decision)
    print(
        f"mujoco_v14_1_screen status={decision['status']} "
        f"selected={decision['selected_arm']}"
    )


if __name__ == "__main__":
    main()
