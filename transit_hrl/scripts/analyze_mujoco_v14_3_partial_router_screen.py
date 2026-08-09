#!/usr/bin/env python3
"""Analyze the frozen MuJoCo v14.3 screen as development evidence."""

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
from scripts import mujoco_v14_3_partial_router_screen_spec as spec  # noqa: E402


ANALYSIS_VERSION = "mujoco_v14_3_partial_router_screen_analysis_v1"
METRICS = (
    "episode_return",
    "LowerLFDriftAbs",
    "RawLowerLFDriftAbs",
    "LatentLowerLFDriftAbs",
    "RawLowerActionRMS",
    "LowerRouterClipRate",
    "LowerActionRouterStrength",
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
            f"v14.3 cell has {len(rows)} rows; expected {expected}: {path}"
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
        raise ValueError(f"v14.3 evaluation path registry mismatch: {path}")
    return rows


def _finite_mean(rows: list[dict[str, str]], metric: str) -> float:
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite v14.3 metric: {metric}")
    return float(np.mean(values))


def _validate_router_strength(
    rows: list[dict[str, str]],
    *,
    expected: float,
    cell_label: str,
) -> None:
    observed = np.asarray([
        float(row["LowerActionRouterStrength"]) for row in rows
    ], dtype=np.float64)
    if (
        not np.all(np.isfinite(observed))
        or not np.allclose(observed, expected, rtol=0.0, atol=1e-12)
    ):
        raise ValueError(
            f"v14.3 lower-action router strength mismatch: {cell_label}"
        )


def _bounds(
    values: np.ndarray,
    *,
    confidence: float,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("v14.3 bootstrap requires finite optimizer replicates")
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
                    cell / "training_history.json",
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
        != "frozen_before_v14_3_development_outcome_access"
        or preregistration.get("development_protocol_version")
        != spec.DEVELOPMENT_PROTOCOL_VERSION
        or preregistration.get("frozen_algorithm_revision")
        != spec.FROZEN_ALGORITHM_REVISION
        or preregistration.get("frozen_source_manifest_sha256")
        != spec.FROZEN_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("v14.3 preregistration identity mismatch")
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
        raise ValueError("v14.3 analysis requires a complete merged manifest")

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
                expected_strength = float(
                    spec.ARMS[arm]["lower_action_router_strength"]
                )
                _validate_router_strength(
                    rows,
                    expected=expected_strength,
                    cell_label=f"{environment}/{arm}/{seed}",
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


def load_training_diagnostics(run_dir: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for arm, arm_spec in spec.ARMS.items():
        by_environment: dict[str, dict[str, float]] = {}
        all_trained: list[float] = []
        all_upper_saturated: list[float] = []
        all_lower_saturated: list[float] = []
        for environment in spec.ENVIRONMENTS:
            trained: list[float] = []
            upper_saturated: list[float] = []
            lower_saturated: list[float] = []
            for seed in spec.OPTIMIZER_SEEDS:
                cell = _cell_dir(
                    run_dir,
                    environment=environment,
                    arm=arm,
                    seed=seed,
                )
                summary = json.loads(
                    (cell / "cell_summary.json").read_text(encoding="utf-8")
                )
                history = json.loads(
                    (cell / "training_history.json").read_text(encoding="utf-8")
                )
                if not isinstance(history, list) or not history:
                    raise ValueError("v14.3 training history is empty")
                selected = int(summary.get("selected_checkpoint_iteration", -2))
                trained.append(float(selected >= 0))
                upper_values = np.asarray([
                    float(row.get("upper_constraint_lambda", 0.0))
                    for row in history
                ], dtype=np.float64)
                lower_values = np.asarray([
                    float(row.get("constraint_lambda", 0.0))
                    for row in history
                ], dtype=np.float64)
                if (
                    not np.all(np.isfinite(upper_values))
                    or not np.all(np.isfinite(lower_values))
                ):
                    raise ValueError("v14.3 dual history contains non-finite values")
                upper_saturated.append(float(
                    np.max(upper_values) >= spec.DUAL_SATURATION_THRESHOLD
                ))
                lower_saturated.append(float(
                    np.max(lower_values) >= spec.DUAL_SATURATION_THRESHOLD
                ))
            values = {
                "trained_checkpoint_fraction": float(np.mean(trained)),
                "upper_dual_saturation_fraction": float(np.mean(
                    upper_saturated
                )),
                "lower_dual_saturation_fraction": float(np.mean(
                    lower_saturated
                )),
            }
            by_environment[environment] = values
            all_trained.extend(trained)
            all_upper_saturated.extend(upper_saturated)
            all_lower_saturated.extend(lower_saturated)
        constrained = str(arm_spec["method"]) == "freq_hrl"
        output[arm] = {
            "by_environment": by_environment,
            "trained_checkpoint_fraction": float(np.mean(all_trained)),
            "minimum_environment_trained_checkpoint_fraction": float(min(
                row["trained_checkpoint_fraction"]
                for row in by_environment.values()
            )),
            "upper_dual_saturation_fraction": float(np.mean(
                all_upper_saturated
            )),
            "lower_dual_saturation_fraction": float(np.mean(
                all_lower_saturated
            )),
            "maximum_environment_upper_dual_saturation_fraction": float(max(
                row["upper_dual_saturation_fraction"]
                for row in by_environment.values()
            )),
            "maximum_environment_lower_dual_saturation_fraction": float(max(
                row["lower_dual_saturation_fraction"]
                for row in by_environment.values()
            )),
            "dual_saturation_gate_applicable": bool(constrained),
        }
    return output


def analyze(run_dir: Path) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    replicates = load_replicates(run)
    training = load_training_diagnostics(run)
    rows: list[dict[str, Any]] = []
    arm_status: dict[str, dict[str, Any]] = {}
    for arm in spec.CANDIDATE_ARMS:
        gate_rows: list[dict[str, Any]] = []
        for environment in spec.ENVIRONMENTS:
            for mode in spec.EVALUATION_DISTURBANCE_MODES:
                baseline = replicates[environment][spec.BASELINE_ARM][mode]
                candidate = replicates[environment][arm][mode]
                baseline_return = float(np.mean(baseline["episode_return"]))
                reward_margin = (
                    spec.RETURN_NONINFERIORITY_MARGIN_FRACTION
                    * max(abs(baseline_return), 1.0)
                )
                reward_lower, reward_upper = _bounds(
                    candidate["episode_return"] - baseline["episode_return"],
                    confidence=spec.SELECTION_CONFIDENCE,
                    draws=spec.BOOTSTRAP_DRAWS,
                    seed=derive_seed(
                        ANALYSIS_VERSION, arm, environment, mode, "return"
                    ),
                )

                def drift_gate(
                    metric: str,
                    reduction_fraction: float,
                    label: str,
                ) -> dict[str, Any]:
                    baseline_values = baseline[metric]
                    candidate_values = candidate[metric]
                    baseline_mean = float(np.mean(baseline_values))
                    difference = candidate_values - baseline_values
                    _, difference_upper = _bounds(
                        difference,
                        confidence=spec.SELECTION_CONFIDENCE,
                        draws=spec.BOOTSTRAP_DRAWS,
                        seed=derive_seed(
                            ANALYSIS_VERSION,
                            arm,
                            environment,
                            mode,
                            label,
                            "difference",
                        ),
                    )
                    material = baseline_mean > spec.DRIFT_MATERIALITY_FLOOR
                    if material:
                        required = float(reduction_fraction) * baseline_mean
                        passed = difference_upper <= -required
                        slack = (-required - difference_upper) / max(
                            required, spec.DRIFT_MATERIALITY_FLOOR
                        )
                        gate_type = "strict_relative_improvement"
                        candidate_upper = float("nan")
                    else:
                        _, candidate_upper = _bounds(
                            candidate_values,
                            confidence=spec.SELECTION_CONFIDENCE,
                            draws=spec.BOOTSTRAP_DRAWS,
                            seed=derive_seed(
                                ANALYSIS_VERSION,
                                arm,
                                environment,
                                mode,
                                label,
                                "absolute_floor",
                            ),
                        )
                        required = 0.0
                        passed = candidate_upper <= spec.DRIFT_MATERIALITY_FLOOR
                        slack = (
                            spec.DRIFT_MATERIALITY_FLOOR - candidate_upper
                        ) / spec.DRIFT_MATERIALITY_FLOOR
                        gate_type = "absolute_floor_noninferiority"
                    return {
                        "baseline_mean": baseline_mean,
                        "candidate_mean": float(np.mean(candidate_values)),
                        "difference_mean": float(np.mean(difference)),
                        "difference_one_sided_upper": difference_upper,
                        "candidate_one_sided_upper": candidate_upper,
                        "required_reduction": required,
                        "baseline_material": bool(material),
                        "gate_type": gate_type,
                        "pass": bool(passed),
                        "normalized_slack": float(slack),
                    }

                responsibility = drift_gate(
                    "LowerLFDriftAbs",
                    spec.MINIMUM_RESPONSIBILITY_DRIFT_REDUCTION_FRACTION,
                    "responsibility",
                )
                raw_lower = drift_gate(
                    "RawLowerLFDriftAbs",
                    spec.MINIMUM_RAW_LOWER_DRIFT_REDUCTION_FRACTION,
                    "raw_lower",
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
                activity_margin_values = (
                    candidate["RawLowerActionRMS"]
                    - spec.MINIMUM_EFFECTIVE_LOWER_ACTION_RMS_FRACTION
                    * baseline["RawLowerActionRMS"]
                )
                activity_lower, _ = _bounds(
                    activity_margin_values,
                    confidence=spec.SELECTION_CONFIDENCE,
                    draws=spec.BOOTSTRAP_DRAWS,
                    seed=derive_seed(
                        ANALYSIS_VERSION, arm, environment, mode, "activity"
                    ),
                )
                _, router_clip_upper = _bounds(
                    candidate["LowerRouterClipRate"],
                    confidence=spec.SELECTION_CONFIDENCE,
                    draws=spec.BOOTSTRAP_DRAWS,
                    seed=derive_seed(
                        ANALYSIS_VERSION, arm, environment, mode, "router_clip"
                    ),
                )
                reward_pass = reward_lower >= -reward_margin
                upper_pass = (
                    upper_hf_rms_upper <= spec.UPPER_HF_REPORTING_GATE
                )
                activity_pass = activity_lower >= 0.0
                router_clip_pass = (
                    router_clip_upper <= spec.MAXIMUM_ROUTER_CLIP_RATE
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
                    responsibility["normalized_slack"],
                    raw_lower["normalized_slack"],
                    (spec.UPPER_HF_REPORTING_GATE - upper_hf_rms_upper)
                    / spec.UPPER_HF_REPORTING_GATE,
                    activity_lower / max(
                        float(np.mean(baseline["RawLowerActionRMS"])), 1e-12
                    ),
                    (spec.MAXIMUM_ROUTER_CLIP_RATE - router_clip_upper)
                    / spec.MAXIMUM_ROUTER_CLIP_RATE,
                )
                row = {
                    "environment": environment,
                    "disturbance_mode": mode,
                    "arm": arm,
                    "upper_constraint_mode": spec.ARMS[arm][
                        "upper_constraint_mode"
                    ],
                    "upper_dual_lr": float(spec.ARMS[arm]["upper_dual_lr"]),
                    "lower_dual_lr": float(spec.ARMS[arm]["lower_dual_lr"]),
                    "lower_action_router_mode": spec.ARMS[arm][
                        "lower_action_router_mode"
                    ],
                    "lower_action_router_alpha": float(spec.ARMS[arm][
                        "lower_action_router_alpha"
                    ]),
                    "lower_action_router_strength": float(spec.ARMS[arm][
                        "lower_action_router_strength"
                    ]),
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
                    "drift_materiality_floor": spec.DRIFT_MATERIALITY_FLOOR,
                    **{
                        f"responsibility_drift_{key}": value
                        for key, value in responsibility.items()
                    },
                    **{
                        f"raw_lower_drift_{key}": value
                        for key, value in raw_lower.items()
                    },
                    "latent_lower_drift_mean": float(np.mean(
                        candidate["LatentLowerLFDriftAbs"]
                    )),
                    "upper_hf_rms": upper_hf_rms,
                    "upper_hf_rms_one_sided_upper": upper_hf_rms_upper,
                    "upper_hf_budget_pass": bool(upper_pass),
                    "effective_lower_action_rms_mean": float(np.mean(
                        candidate["RawLowerActionRMS"]
                    )),
                    "effective_lower_activity_margin_one_sided_lower": (
                        activity_lower
                    ),
                    "effective_lower_activity_pass": bool(activity_pass),
                    "router_clip_rate_mean": float(np.mean(
                        candidate["LowerRouterClipRate"]
                    )),
                    "router_clip_rate_one_sided_upper": router_clip_upper,
                    "router_clip_pass": bool(router_clip_pass),
                    "reconstruction_rms_max": reconstruction_max,
                    "reconstruction_integrity_pass": bool(
                        reconstruction_pass
                    ),
                    "minimum_normalized_safety_slack": float(min(slacks)),
                    "condition_gate_pass": bool(
                        reward_pass
                        and responsibility["pass"]
                        and raw_lower["pass"]
                        and upper_pass
                        and activity_pass
                        and router_clip_pass
                        and reconstruction_pass
                    ),
                }
                rows.append(row)
                gate_rows.append(row)
        training_status = training[arm]
        minimum_trained_fraction = float(
            training_status["minimum_environment_trained_checkpoint_fraction"]
        )
        trained_pass = (
            minimum_trained_fraction
            >= spec.MINIMUM_TRAINED_CHECKPOINT_FRACTION
        )
        constrained = bool(training_status["dual_saturation_gate_applicable"])
        max_upper_saturation = float(
            training_status[
                "maximum_environment_upper_dual_saturation_fraction"
            ]
        )
        max_lower_saturation = float(
            training_status[
                "maximum_environment_lower_dual_saturation_fraction"
            ]
        )
        dual_pass = bool(
            not constrained
            or (
                max_upper_saturation
                <= spec.MAXIMUM_DUAL_SATURATION_FRACTION
                and max_lower_saturation
                <= spec.MAXIMUM_DUAL_SATURATION_FRACTION
            )
        )
        strict_responsibility = int(sum(
            row["responsibility_drift_gate_type"]
            == "strict_relative_improvement"
            and bool(row["responsibility_drift_pass"])
            for row in gate_rows
        ))
        strict_raw = int(sum(
            row["raw_lower_drift_gate_type"]
            == "strict_relative_improvement"
            and bool(row["raw_lower_drift_pass"])
            for row in gate_rows
        ))
        strict_pass = bool(
            strict_responsibility
            >= spec.MINIMUM_STRICT_RESPONSIBILITY_IMPROVEMENT_CONDITIONS
            and strict_raw
            >= spec.MINIMUM_STRICT_RAW_IMPROVEMENT_CONDITIONS
        )
        all_conditions = bool(all(
            row["condition_gate_pass"] for row in gate_rows
        ))
        global_slacks = [
            (minimum_trained_fraction - spec.MINIMUM_TRAINED_CHECKPOINT_FRACTION)
            / spec.MINIMUM_TRAINED_CHECKPOINT_FRACTION,
            (
                strict_responsibility
                - spec.MINIMUM_STRICT_RESPONSIBILITY_IMPROVEMENT_CONDITIONS
            ) / max(
                spec.MINIMUM_STRICT_RESPONSIBILITY_IMPROVEMENT_CONDITIONS, 1
            ),
            (
                strict_raw - spec.MINIMUM_STRICT_RAW_IMPROVEMENT_CONDITIONS
            ) / max(spec.MINIMUM_STRICT_RAW_IMPROVEMENT_CONDITIONS, 1),
        ]
        if constrained:
            global_slacks.extend((
                (
                    spec.MAXIMUM_DUAL_SATURATION_FRACTION
                    - max_upper_saturation
                ) / spec.MAXIMUM_DUAL_SATURATION_FRACTION,
                (
                    spec.MAXIMUM_DUAL_SATURATION_FRACTION
                    - max_lower_saturation
                ) / spec.MAXIMUM_DUAL_SATURATION_FRACTION,
            ))
        arm_status[arm] = {
            "all_environment_condition_gates_pass": all_conditions,
            "passed_gate_count": int(sum(
                bool(row["condition_gate_pass"]) for row in gate_rows
            )),
            "total_gate_count": len(gate_rows),
            "trained_checkpoint_gate_pass": bool(trained_pass),
            "minimum_environment_trained_checkpoint_fraction": (
                minimum_trained_fraction
            ),
            "dual_saturation_gate_pass": bool(dual_pass),
            "maximum_environment_upper_dual_saturation_fraction": (
                max_upper_saturation
            ),
            "maximum_environment_lower_dual_saturation_fraction": (
                max_lower_saturation
            ),
            "strict_responsibility_improvement_condition_count": (
                strict_responsibility
            ),
            "strict_raw_improvement_condition_count": strict_raw,
            "strict_improvement_gate_pass": strict_pass,
            "development_selection_pass": bool(
                all_conditions and trained_pass and dual_pass and strict_pass
            ),
            "minimum_normalized_safety_slack": float(min([
                row["minimum_normalized_safety_slack"] for row in gate_rows
            ] + global_slacks)),
            "mean_return_lower_bound": float(np.mean([
                row["return_difference_one_sided_lower"] for row in gate_rows
            ])),
            "training_diagnostics": training_status,
        }
    eligible = [
        arm for arm, status in arm_status.items()
        if status["development_selection_pass"]
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
            "The v14.3 screen may select one fully trained, non-saturating "
            "algorithm arm for a new v15 confirmation. Absolute at-floor "
            "conditions establish noninferiority only, not improvement. This "
            "development screen cannot support a confirmatory claim."
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
        raise RuntimeError("existing v14.3 screen decision differs")
    decision_path.write_text(rendered, encoding="utf-8")
    _write_csv(
        output / "environment_condition_gates.csv",
        decision["environment_condition_rows"],
    )
    lines = [
        "# MuJoCo v14.3 Partial-Strength Action-Router Screen",
        "",
        f"- Status: `{decision['status']}`",
        f"- Selected arm: `{decision['selected_arm']}`",
        "- Evidence role: development only; not confirmatory.",
        "- Gate granularity: environment by disturbance mode.",
        "",
        "| arm | conditions | trained | dual | strict | select | min slack |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, status in decision["arm_status"].items():
        lines.append(
            f"| {arm} | {status['passed_gate_count']}/"
            f"{status['total_gate_count']} | "
            f"{status['trained_checkpoint_gate_pass']} | "
            f"{status['dual_saturation_gate_pass']} | "
            f"{status['strict_improvement_gate_pass']} | "
            f"{status['development_selection_pass']} | "
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
        f"mujoco_v14_3_screen status={decision['status']} "
        f"selected={decision['selected_arm']}"
    )


if __name__ == "__main__":
    main()
