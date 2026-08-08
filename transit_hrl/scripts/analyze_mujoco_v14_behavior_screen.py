#!/usr/bin/env python3
"""Analyze the frozen MuJoCo v14 behavior screen as development evidence."""

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
from scripts import mujoco_v14_behavior_screen_spec as spec  # noqa: E402
from scripts.submit_mujoco_v14_behavior_screen_scheduleurm import (  # noqa: E402
    cell_relative_dir,
)


ANALYSIS_VERSION = "mujoco_v14_behavior_screen_analysis_v1"
METRICS = (
    "episode_return",
    "LowerLFDriftAbs",
    "RawLowerLFDriftAbs",
    "UpperHFPowerAbs",
    "ResponsibilityReconstructionRMS",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = (
        len(spec.DEVELOPMENT_EVALUATION_SEEDS)
        * len(spec.EVALUATION_DISTURBANCE_MODES)
    )
    if len(rows) != expected:
        raise ValueError(
            f"v14 screen cell has {len(rows)} rows; expected {expected}: {path}"
        )
    return rows


def _finite_mean(rows: list[dict[str, str]], metric: str) -> float:
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite v14 screen metric: {metric}")
    return float(np.mean(values))


def _bounds(
    values: np.ndarray,
    *,
    confidence: float,
    draws: int,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("v14 screen bootstrap requires finite replicates")
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(
        0, array.size, size=(int(draws), array.size)
    )
    means = np.mean(array[indices], axis=1)
    alpha = 1.0 - float(confidence)
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
        means,
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
                cell = run_dir.parent.parent / cell_relative_dir(
                    run_dir.name,
                    environment=environment,
                    arm=arm,
                    optimizer_seed=seed,
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


def _cell_dir(
    run_dir: Path,
    *,
    environment: str,
    arm: str,
    seed: int,
) -> Path:
    return (
        run_dir / "cells" / environment / arm / f"replicate_{int(seed)}"
    )


def load_replicates(run_dir: Path) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    preregistration = json.loads(
        (run_dir / "preregistration.json").read_text(encoding="utf-8")
    )
    if (
        preregistration.get("status")
        != "frozen_before_v14_development_outcome_access"
        or preregistration.get("development_protocol_version")
        != spec.DEVELOPMENT_PROTOCOL_VERSION
        or preregistration.get("frozen_algorithm_revision")
        != spec.FROZEN_ALGORITHM_REVISION
        or preregistration.get("frozen_source_manifest_sha256")
        != spec.FROZEN_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("v14 screen preregistration identity mismatch")
    merged = run_dir / "merged" / "cell_manifest.json"
    manifest = json.loads(merged.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "development_screen_complete_unanalyzed"
        or int(manifest.get("cell_count", -1))
        != len(spec.ENVIRONMENTS) * len(spec.ARMS) * len(spec.OPTIMIZER_SEEDS)
    ):
        raise ValueError("v14 screen requires a complete merged manifest")
    output: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for environment in spec.ENVIRONMENTS:
        output[environment] = {}
        for arm in spec.ARMS:
            collected = {metric: [] for metric in METRICS}
            for seed in spec.OPTIMIZER_SEEDS:
                cell = _cell_dir(
                    run_dir,
                    environment=environment,
                    arm=arm,
                    seed=seed,
                )
                rows = _read_rows(cell / "evaluation_rows.csv")
                for metric in METRICS:
                    if metric == "ResponsibilityReconstructionRMS":
                        value = max(float(row[metric]) for row in rows)
                    else:
                        value = _finite_mean(rows, metric)
                    collected[metric].append(value)
            output[environment][arm] = {
                metric: np.asarray(values, dtype=np.float64)
                for metric, values in collected.items()
            }
    return output


def analyze(run_dir: Path) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    replicates = load_replicates(run)
    rows: list[dict[str, Any]] = []
    arm_status: dict[str, dict[str, Any]] = {}
    for arm in spec.CANDIDATE_ARMS:
        environment_rows = []
        for environment in spec.ENVIRONMENTS:
            baseline = replicates[environment][spec.BASELINE_ARM]
            candidate = replicates[environment][arm]
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
            reward_lower, reward_upper, _ = _bounds(
                candidate["episode_return"] - baseline["episode_return"],
                confidence=spec.SELECTION_CONFIDENCE,
                draws=spec.BOOTSTRAP_DRAWS,
                seed=derive_seed(
                    ANALYSIS_VERSION, arm, environment, "return"
                ),
            )
            drift_lower, drift_upper, _ = _bounds(
                candidate["LowerLFDriftAbs"] - baseline["LowerLFDriftAbs"],
                confidence=spec.SELECTION_CONFIDENCE,
                draws=spec.BOOTSTRAP_DRAWS,
                seed=derive_seed(
                    ANALYSIS_VERSION, arm, environment, "responsibility"
                ),
            )
            raw_lower, raw_upper, _ = _bounds(
                candidate["RawLowerLFDriftAbs"]
                - baseline["RawLowerLFDriftAbs"],
                confidence=spec.SELECTION_CONFIDENCE,
                draws=spec.BOOTSTRAP_DRAWS,
                seed=derive_seed(
                    ANALYSIS_VERSION, arm, environment, "raw_lower"
                ),
            )
            _, power_upper, _ = _bounds(
                candidate["UpperHFPowerAbs"],
                confidence=spec.SELECTION_CONFIDENCE,
                draws=spec.BOOTSTRAP_DRAWS,
                seed=derive_seed(
                    ANALYSIS_VERSION, arm, environment, "upper_hf"
                ),
            )
            upper_hf_rms = float(np.sqrt(np.mean(
                candidate["UpperHFPowerAbs"]
            )))
            upper_hf_rms_upper = float(np.sqrt(max(power_upper, 0.0)))
            reward_pass = reward_lower >= -reward_margin
            drift_pass = drift_upper <= -required_drift
            raw_pass = raw_upper <= -required_raw
            upper_pass = upper_hf_rms_upper <= spec.MAXIMUM_UPPER_HF_RMS
            reconstruction_max = float(np.max(
                candidate["ResponsibilityReconstructionRMS"]
            ))
            reconstruction_pass = (
                reconstruction_max <= spec.MAXIMUM_RECONSTRUCTION_RMS
            )
            slacks = (
                (reward_lower + reward_margin) / max(reward_margin, 1e-12),
                (-required_drift - drift_upper) / max(required_drift, 1e-12),
                (-required_raw - raw_upper) / max(required_raw, 1e-12),
                (spec.MAXIMUM_UPPER_HF_RMS - upper_hf_rms_upper)
                / spec.MAXIMUM_UPPER_HF_RMS,
            )
            row = {
                "environment": environment,
                "arm": arm,
                "upper_hf_penalty_coef": float(
                    spec.ARMS[arm]["upper_hf_penalty_coef"]
                ),
                "replicate_count": len(spec.OPTIMIZER_SEEDS),
                "baseline_return_mean": baseline_return,
                "candidate_return_mean": float(np.mean(
                    candidate["episode_return"]
                )),
                "return_difference_mean": float(np.mean(
                    candidate["episode_return"] - baseline["episode_return"]
                )),
                "return_difference_one_sided_lower": reward_lower,
                "return_difference_one_sided_upper": reward_upper,
                "reward_noninferiority_margin": reward_margin,
                "reward_noninferiority_pass": bool(reward_pass),
                "responsibility_drift_difference_mean": float(np.mean(
                    candidate["LowerLFDriftAbs"]
                    - baseline["LowerLFDriftAbs"]
                )),
                "responsibility_drift_one_sided_lower": drift_lower,
                "responsibility_drift_one_sided_upper": drift_upper,
                "required_responsibility_drift_reduction": required_drift,
                "responsibility_drift_pass": bool(drift_pass),
                "raw_lower_drift_difference_mean": float(np.mean(
                    candidate["RawLowerLFDriftAbs"]
                    - baseline["RawLowerLFDriftAbs"]
                )),
                "raw_lower_drift_one_sided_lower": raw_lower,
                "raw_lower_drift_one_sided_upper": raw_upper,
                "required_raw_lower_drift_reduction": required_raw,
                "raw_lower_drift_pass": bool(raw_pass),
                "upper_hf_rms": upper_hf_rms,
                "upper_hf_rms_one_sided_upper": upper_hf_rms_upper,
                "upper_hf_budget_pass": bool(upper_pass),
                "reconstruction_rms_max": reconstruction_max,
                "reconstruction_integrity_pass": bool(reconstruction_pass),
                "minimum_normalized_safety_slack": float(min(slacks)),
                "environment_gate_pass": bool(
                    reward_pass
                    and drift_pass
                    and raw_pass
                    and upper_pass
                    and reconstruction_pass
                ),
            }
            rows.append(row)
            environment_rows.append(row)
        arm_status[arm] = {
            "all_environment_gates_pass": bool(all(
                row["environment_gate_pass"] for row in environment_rows
            )),
            "minimum_normalized_safety_slack": float(min(
                row["minimum_normalized_safety_slack"]
                for row in environment_rows
            )),
            "mean_return_lower_bound": float(np.mean([
                row["return_difference_one_sided_lower"]
                for row in environment_rows
            ])),
        }
    eligible = [
        arm for arm, status in arm_status.items()
        if status["all_environment_gates_pass"]
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
        "selected_upper_hf_penalty_coef": (
            None if selected is None else float(
                spec.ARMS[selected]["upper_hf_penalty_coef"]
            )
        ),
        "eligible_arms": eligible,
        "arm_status": arm_status,
        "environment_rows": rows,
        "selection_confidence": spec.SELECTION_CONFIDENCE,
        "bootstrap_draws": spec.BOOTSTRAP_DRAWS,
        "input_sha256": _input_sha256(run),
        "claim_boundary": (
            "The screen may select a v14 development coefficient. It cannot "
            "support a confirmatory performance or behavior claim."
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
        raise RuntimeError("existing v14 screen decision differs")
    decision_path.write_text(rendered, encoding="utf-8")
    _write_csv(output / "environment_gates.csv", decision["environment_rows"])
    lines = [
        "# MuJoCo v14 Behavior Screen",
        "",
        f"- Status: `{decision['status']}`",
        f"- Selected arm: `{decision['selected_arm']}`",
        "- Evidence role: development only; not confirmatory.",
        "",
        "| environment | arm | return NI | responsibility | raw lower | upper HF | gate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in decision["environment_rows"]:
        lines.append(
            f"| {row['environment']} | {row['arm']} | "
            f"{row['reward_noninferiority_pass']} | "
            f"{row['responsibility_drift_pass']} | "
            f"{row['raw_lower_drift_pass']} | "
            f"{row['upper_hf_budget_pass']} | "
            f"{row['environment_gate_pass']} |"
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
        f"mujoco_v14_screen status={decision['status']} "
        f"selected={decision['selected_arm']}"
    )


if __name__ == "__main__":
    main()
