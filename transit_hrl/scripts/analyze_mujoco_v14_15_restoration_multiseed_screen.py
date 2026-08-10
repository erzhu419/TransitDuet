#!/usr/bin/env python3
"""Analyze the frozen candidate-fixed MuJoCo v14.15 multiseed screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import mujoco_v14_15_restoration_multiseed_screen_spec as spec  # noqa: E402
from scripts.analyze_mujoco_v14_15_closed_loop_restoration_filter_preflight import (  # noqa: E402
    analyze_preflight,
)


ANALYSIS_VERSION = "mujoco_v14_15_restoration_multiseed_analysis_v1"


def _wilson_lower(successes: int, total: int, z: float) -> float:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("invalid Wilson interval counts")
    p = float(successes) / float(total)
    z2 = float(z) ** 2
    denominator = 1.0 + z2 / total
    center = p + z2 / (2.0 * total)
    radius = float(z) * math.sqrt(
        p * (1.0 - p) / total + z2 / (4.0 * total**2)
    )
    return float((center - radius) / denominator)


def _simultaneous_basic_lower_bounds(
    values: np.ndarray,
    *,
    confidence: float,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    matrix = np.asarray(values, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] < 2
        or matrix.shape[1] < 1
        or not np.all(np.isfinite(matrix))
        or not 0.5 < confidence < 1.0
        or draws < 100
    ):
        raise ValueError("invalid optimizer-level bootstrap matrix")
    observed = np.mean(matrix, axis=0)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0, matrix.shape[0], size=(int(draws), matrix.shape[0])
    )
    bootstrap = np.mean(matrix[indices], axis=1)
    maximum_downward_error = np.max(observed[None, :] - bootstrap, axis=1)
    critical = float(np.quantile(
        maximum_downward_error, confidence, method="higher"
    ))
    simultaneous_lower = observed - critical
    nominal_lower = 2.0 * observed - np.quantile(
        bootstrap, confidence, axis=0, method="higher"
    )
    return {
        "observed": observed,
        "simultaneous_lower": simultaneous_lower,
        "nominal_lower": nominal_lower,
        "critical_value": critical,
    }


def _condition_effects(
    status: dict[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    conditions = list(status.get("conditions") or [])
    if {row.get("disturbance_mode") for row in conditions} != set(
        spec.EVALUATION_DISTURBANCE_MODES
    ):
        raise ValueError("candidate disturbance grid is incomplete")
    per_mode: list[dict[str, Any]] = []
    for row in sorted(
        conditions,
        key=lambda item: spec.EVALUATION_DISTURBANCE_MODES.index(
            str(item["disturbance_mode"])
        ),
    ):
        margin = float(row["reward_noninferiority_margin"])
        denominator = margin / spec.RETURN_NONINFERIORITY_MARGIN_FRACTION
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("invalid paired return normalization")
        effects = {
            "normalized_episode_return": (
                float(row["reward_difference"]) / denominator
            )
        }
        reductions = dict(row["frequency_reduction_fraction"])
        for metric in spec.FREQUENCY_METRICS:
            ratio = max(
                1.0 - float(reductions[metric]), spec.METRIC_EPSILON
            )
            effects[metric] = float(-math.log(ratio))
        per_mode.append({
            "disturbance_mode": str(row["disturbance_mode"]),
            "effects": effects,
        })
    pooled = {
        metric: float(np.mean([
            row["effects"][metric] for row in per_mode
        ]))
        for metric in ("normalized_episode_return", *spec.FREQUENCY_METRICS)
    }
    return pooled, per_mode


def _input_sha256(run: Path, decisions: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for path in (
        run / "preregistration.json",
        run / "merged" / "cell_manifest.json",
        run / "merged" / "run_scoped_result_sync.json",
    ):
        relative = str(path.relative_to(run)).encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    for decision in decisions:
        digest.update(str(decision["environment"]).encode("utf-8"))
        digest.update(int(decision["optimizer_seed"]).to_bytes(8, "big"))
        digest.update(str(decision["input_sha256"]).encode("ascii"))
    return digest.hexdigest()


def analyze(run_dir: Path) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    preregistration = json.loads(
        (run / "preregistration.json").read_text(encoding="utf-8")
    )
    if not (
        preregistration.get("analysis_profile") == spec.ANALYSIS_PROFILE
        and preregistration.get("multiseed_development_protocol_version")
        == spec.DEVELOPMENT_PROTOCOL_VERSION
        and preregistration.get("preselected_candidate_arm")
        == spec.PRESELECTED_CANDIDATE_ARM
        and preregistration.get("selection_source_evidence_id")
        == spec.SELECTION_SOURCE_EVIDENCE_ID
        and preregistration.get("selection_source_decision_sha256")
        == spec.SELECTION_SOURCE_DECISION_SHA256
        and preregistration.get("dispatched_environment_subset")
        == list(spec.ENVIRONMENTS)
        and preregistration.get("dispatched_optimizer_seed_subset")
        == list(spec.OPTIMIZER_SEEDS)
        and preregistration.get("statistical_unit") == "optimizer_seed"
        and preregistration.get("optimizer_seed_reuse_policy")
        == "preflight_selection_seed_excluded"
        and preregistration.get("multiseed_spec_sha256")
        == hashlib.sha256(Path(spec.__file__).read_bytes()).hexdigest()
        and preregistration.get("multiseed_analyzer_sha256")
        == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ):
        raise ValueError("multiseed preregistration identity mismatch")

    decisions: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    mode_observations: dict[tuple[str, str, str], list[float]] = {}
    contrast_by_seed: dict[int, dict[tuple[str, str], float]] = {
        seed: {} for seed in spec.OPTIMIZER_SEEDS
    }
    for environment in spec.ENVIRONMENTS:
        for optimizer_seed in spec.OPTIMIZER_SEEDS:
            decision = analyze_preflight(
                run,
                environment=environment,
                optimizer_seed=optimizer_seed,
                manifest_environments=spec.ENVIRONMENTS,
                manifest_optimizer_seeds=spec.OPTIMIZER_SEEDS,
                expected_algorithm_revision=spec.FROZEN_EXECUTION_REVISION,
                expected_source_manifest_sha256=(
                    spec.FROZEN_EXECUTION_SOURCE_MANIFEST_SHA256
                ),
            )
            decisions.append(decision)
            candidate = decision["arm_status"][spec.PRESELECTED_CANDIDATE_ARM]
            strict = decision["arm_status"][spec.STRICT_ABLATION_ARM]
            pooled, per_mode = _condition_effects(candidate)
            for metric, value in pooled.items():
                contrast_by_seed[optimizer_seed][environment, metric] = value
            for mode_row in per_mode:
                for metric, value in mode_row["effects"].items():
                    mode_observations.setdefault(
                        (environment, mode_row["disturbance_mode"], metric), []
                    ).append(value)
            candidate_guard = candidate["closed_loop_guard"]
            strict_guard = strict["closed_loop_guard"]
            replicate_rows.append({
                "environment": environment,
                "optimizer_seed": optimizer_seed,
                "calibration_pass": bool(decision["calibration_pass"]),
                "candidate_preflight_pass": bool(candidate["preflight_pass"]),
                "candidate_trained_checkpoint_pass": bool(
                    candidate["trained_checkpoint_pass"]
                ),
                "candidate_projection_pass": bool(
                    candidate["deployment_projection"]["pass"]
                ),
                "candidate_replay_pass": bool(
                    candidate["anchor_state_replay"]["pass"]
                ),
                "candidate_trust_pass": bool(
                    candidate["ppo_trust_region"]["pass"]
                ),
                "candidate_guard_pass": bool(candidate_guard["pass"]),
                "candidate_selection_pass": bool(
                    candidate["selection_feasibility_pass"]
                ),
                "candidate_heldout_pass": bool(
                    candidate["all_condition_gates_pass"]
                ),
                "candidate_selected_iteration": int(
                    candidate["selected_checkpoint_iteration"]
                ),
                "candidate_effective_updates": int(
                    candidate_guard["effective_update_count"]
                ),
                "candidate_selected_frequency_violations": int(
                    candidate_guard["selected_frequency_violation_count"]
                ),
                "candidate_selected_reward_violations": int(
                    candidate_guard["selected_reward_violation_count"]
                ),
                "strict_complete_gate_pass": bool(
                    strict["base_preflight_pass"]
                ),
                "strict_effective_updates": int(
                    strict_guard["effective_update_count"]
                ),
                "strict_selected_frequency_violations": int(
                    strict_guard["selected_frequency_violation_count"]
                ),
            })

    values = np.asarray([
        [
            contrast_by_seed[seed][key]
            for key in spec.PRIMARY_CONTRAST_ORDER
        ]
        for seed in spec.OPTIMIZER_SEEDS
    ], dtype=np.float64)
    bootstrap = _simultaneous_basic_lower_bounds(
        values,
        confidence=spec.CONFIDENCE,
        draws=spec.BOOTSTRAP_DRAWS,
        seed=spec.BOOTSTRAP_SEED,
    )
    primary_rows: list[dict[str, Any]] = []
    for index, (environment, metric) in enumerate(spec.PRIMARY_CONTRAST_ORDER):
        threshold = float(spec.PRIMARY_THRESHOLDS[environment, metric])
        lower = float(bootstrap["simultaneous_lower"][index])
        primary_rows.append({
            "environment": environment,
            "metric": metric,
            "effect_scale": (
                "paired_normalized_difference"
                if metric == "normalized_episode_return"
                else "paired_log_baseline_over_candidate"
            ),
            "mean_effect": float(bootstrap["observed"][index]),
            "nominal_lower": float(bootstrap["nominal_lower"][index]),
            "simultaneous_lower": lower,
            "registered_threshold": threshold,
            "pass": bool(lower > threshold),
        })

    mode_rows: list[dict[str, Any]] = []
    for index, key in enumerate(sorted(mode_observations)):
        environment, mode, metric = key
        vector = np.asarray(mode_observations[key], dtype=np.float64)[:, None]
        result = _simultaneous_basic_lower_bounds(
            vector,
            confidence=spec.CONFIDENCE,
            draws=spec.BOOTSTRAP_DRAWS,
            seed=spec.BOOTSTRAP_SEED + index + 1,
        )
        threshold = float(spec.PRIMARY_THRESHOLDS[environment, metric])
        mean_effect = float(result["observed"][0])
        mode_rows.append({
            "environment": environment,
            "disturbance_mode": mode,
            "metric": metric,
            "mean_effect": mean_effect,
            "nominal_lower": float(result["nominal_lower"][0]),
            "registered_threshold": threshold,
            "point_gate_pass": bool(mean_effect > threshold),
            "nominal_gate_pass": bool(
                float(result["nominal_lower"][0]) > threshold
            ),
        })

    environment_gates = []
    for environment in spec.ENVIRONMENTS:
        rows = [
            row for row in replicate_rows
            if row["environment"] == environment
        ]
        complete = sum(bool(row["candidate_preflight_pass"]) for row in rows)
        environment_gates.append({
            "environment": environment,
            "complete_gate_count": complete,
            "replicate_count": len(rows),
            "pass": bool(
                complete >= spec.MINIMUM_ENVIRONMENT_COMPLETE_GATE_COUNT
            ),
        })
    total_complete = sum(
        bool(row["candidate_preflight_pass"]) for row in replicate_rows
    )
    complete_fraction_lower = _wilson_lower(
        total_complete,
        len(replicate_rows),
        spec.WILSON_ONE_SIDED_Z,
    )
    calibration_pass = all(
        bool(row["calibration_pass"]) for row in replicate_rows
    )
    primary_pass = all(bool(row["pass"]) for row in primary_rows)
    mode_point_pass = all(bool(row["point_gate_pass"]) for row in mode_rows)
    environment_gate_pass = all(
        bool(row["pass"]) for row in environment_gates
    )
    complete_fraction_pass = bool(
        complete_fraction_lower
        >= spec.MINIMUM_AGGREGATE_COMPLETE_GATE_FRACTION_LOWER
    )
    ready = bool(
        calibration_pass
        and primary_pass
        and mode_point_pass
        and environment_gate_pass
        and complete_fraction_pass
    )
    return {
        "analysis_version": ANALYSIS_VERSION,
        "status": (
            "candidate_ready_for_fresh_confirmation"
            if ready else "candidate_not_ready_for_confirmation"
        ),
        "evidence_role": "candidate_fixed_multiseed_development_no_confirmation",
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "preselected_candidate_arm": spec.PRESELECTED_CANDIDATE_ARM,
        "strict_ablation_arm": spec.STRICT_ABLATION_ARM,
        "optimizer_seed_count": len(spec.OPTIMIZER_SEEDS),
        "environment_count": len(spec.ENVIRONMENTS),
        "statistical_unit": "optimizer_seed",
        "heldout_paths_are_not_replicates": True,
        "simultaneous_confidence": spec.CONFIDENCE,
        "bootstrap_draws": spec.BOOTSTRAP_DRAWS,
        "bootstrap_seed": spec.BOOTSTRAP_SEED,
        "simultaneous_critical_value": bootstrap["critical_value"],
        "calibration_pass": calibration_pass,
        "primary_contrast_pass": primary_pass,
        "all_mode_point_gates_pass": mode_point_pass,
        "environment_complete_gate_pass": environment_gate_pass,
        "aggregate_complete_gate_count": total_complete,
        "aggregate_replicate_count": len(replicate_rows),
        "aggregate_complete_gate_fraction_lower": complete_fraction_lower,
        "aggregate_complete_gate_fraction_pass": complete_fraction_pass,
        "environment_gates": environment_gates,
        "primary_contrasts": primary_rows,
        "mode_contrasts": mode_rows,
        "replicate_rows": replicate_rows,
        "input_sha256": _input_sha256(run, decisions),
        "claim_boundary": spec.CLAIM_BOUNDARY,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_analysis(output_dir: Path, decision: dict[str, Any]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    decision_path = output / "decision.json"
    if decision_path.exists() and decision_path.read_text(
        encoding="utf-8"
    ) != rendered:
        raise RuntimeError("existing multiseed decision differs")
    decision_path.write_text(rendered, encoding="utf-8")
    _write_csv(output / "primary_contrasts.csv", decision["primary_contrasts"])
    _write_csv(output / "mode_contrasts.csv", decision["mode_contrasts"])
    _write_csv(output / "replicate_rows.csv", decision["replicate_rows"])
    lines = [
        "# MuJoCo v14.15 Restoration Multiseed Development Screen",
        "",
        f"- Status: `{decision['status']}`",
        f"- Candidate: `{decision['preselected_candidate_arm']}`",
        f"- Optimizer seeds: `{decision['optimizer_seed_count']}`",
        f"- Environments: `{decision['environment_count']}`",
        "- Statistical unit: optimizer seed; held-out paths are not replicates.",
        f"- Simultaneous primary gate: `{decision['primary_contrast_pass']}`",
        f"- All mode point gates: `{decision['all_mode_point_gates_pass']}`",
        f"- Complete candidate cells: `{decision['aggregate_complete_gate_count']}/{decision['aggregate_replicate_count']}`",
        f"- One-sided Wilson lower bound: `{decision['aggregate_complete_gate_fraction_lower']:.6f}`",
        "",
        "| environment | metric | mean | simultaneous lower | threshold | pass |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in decision["primary_contrasts"]:
        lines.append(
            f"| {row['environment']} | {row['metric']} | "
            f"{row['mean_effect']:.6f} | {row['simultaneous_lower']:.6f} | "
            f"{row['registered_threshold']:.6f} | {row['pass']} |"
        )
    lines.extend(("", decision["claim_boundary"], ""))
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    decision = analyze(args.run_dir)
    write_analysis(args.output_dir, decision)
    print(
        f"mujoco_v14_15_multiseed status={decision['status']} "
        f"complete={decision['aggregate_complete_gate_count']}/"
        f"{decision['aggregate_replicate_count']}"
    )


if __name__ == "__main__":
    main()
