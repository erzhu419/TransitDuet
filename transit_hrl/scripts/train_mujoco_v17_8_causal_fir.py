#!/usr/bin/env python3
"""Run grouped v17.8 causal FIR selection on server-only reused paths."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.mujoco.full_horizon_responsibility_oracle import (  # noqa: E402
    responsibility_frequency_powers,
)
from freq_hrl.experiments.reproducibility import (  # noqa: E402
    verify_current_freq_hrl_source_identity,
)
from scripts import (  # noqa: E402
    mujoco_v17_8_causal_fir_distillation_spec as spec,
)
from scripts.mujoco_v17_8_causal_fir import (  # noqa: E402
    candidate_id,
    evaluate_causal_fir_split,
    fit_causal_fir,
)


def artifact_path(
    dataset_root: Path, environment: str, mode: str, seed: int
) -> Path:
    return (
        Path(dataset_root) / str(environment) / str(mode)
        / f"seed_{int(seed)}" / "training_path.npz"
    )


def load_reused_panel(dataset_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for environment in spec.ENVIRONMENTS:
        for mode in spec.DISTURBANCE_MODES:
            for seed in spec.REUSED_SELECTION_SEEDS:
                path = artifact_path(dataset_root, environment, mode, seed)
                if not path.is_file():
                    raise FileNotFoundError(f"missing v17.8 dataset path: {path}")
                with np.load(path, allow_pickle=False) as arrays:
                    expected_keys = {
                        "total_action",
                        "baseline_upper_action",
                        "oracle_upper_action",
                    }
                    if set(arrays.files) != expected_keys:
                        raise ValueError(f"invalid v17.8 dataset keys: {path}")
                    total = np.asarray(
                        arrays["total_action"], dtype=np.float64
                    ).copy()
                    baseline_upper = np.asarray(
                        arrays["baseline_upper_action"], dtype=np.float64
                    ).copy()
                    oracle_upper = np.asarray(
                        arrays["oracle_upper_action"], dtype=np.float64
                    ).copy()
                if (
                    total.ndim != 2
                    or not total.size
                    or baseline_upper.shape != total.shape
                    or oracle_upper.shape != total.shape
                    or not np.all(np.isfinite(total))
                    or not np.all(np.isfinite(baseline_upper))
                    or not np.all(np.isfinite(oracle_upper))
                ):
                    raise ValueError(f"invalid v17.8 dataset arrays: {path}")
                baseline_upper_power, baseline_lower_power = (
                    responsibility_frequency_powers(
                        total,
                        baseline_upper,
                        upper_window=spec.UPPER_WINDOW,
                        lower_window=spec.LOWER_WINDOW,
                    )
                )
                oracle_upper_power, oracle_lower_power = (
                    responsibility_frequency_powers(
                        total,
                        oracle_upper,
                        upper_window=spec.UPPER_WINDOW,
                        lower_window=spec.LOWER_WINDOW,
                    )
                )
                upper_budget = spec.UPPER_RMS_BUDGET ** 2
                lower_budget = spec.LOWER_RMS_BUDGET ** 2
                rows.append({
                    "environment": str(environment),
                    "disturbance_mode": str(mode),
                    "evaluation_seed": int(seed),
                    "total_action": total,
                    "baseline_upper_action": baseline_upper,
                    "oracle_upper_action": oracle_upper,
                    "baseline_upper_power": baseline_upper_power,
                    "baseline_lower_power": baseline_lower_power,
                    "baseline_joint_feasible": bool(
                        baseline_upper_power
                        <= upper_budget + spec.POWER_TOLERANCE
                        and baseline_lower_power
                        <= lower_budget + spec.POWER_TOLERANCE
                    ),
                    "oracle_upper_power": oracle_upper_power,
                    "oracle_lower_power": oracle_lower_power,
                    "oracle_joint_feasible": bool(
                        oracle_upper_power
                        <= upper_budget + spec.POWER_TOLERANCE
                        and oracle_lower_power
                        <= lower_budget + spec.POWER_TOLERANCE
                    ),
                })
    if len(rows) != spec.REUSED_EXPECTED_PATH_COUNT:
        raise RuntimeError("v17.8 reused panel path count mismatch")
    return rows


def candidate_configs() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": candidate_id(window, penalty, gain),
            "window": int(window),
            "ridge_penalty": float(penalty),
            "output_gain": float(gain),
        }
        for window in spec.FIR_WINDOWS
        for penalty in spec.RIDGE_PENALTIES
        for gain in spec.OUTPUT_GAINS
    ]


def grouped_out_of_fold_rows(
    panel: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    configs = candidate_configs()
    rows_by_candidate: dict[str, list[dict[str, Any]]] = {
        str(config["candidate_id"]): [] for config in configs
    }
    gains_by_base: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for config in configs:
        gains_by_base[
            (int(config["window"]), float(config["ridge_penalty"]))
        ].append(config)

    for environment in spec.ENVIRONMENTS:
        environment_rows = [
            row for row in panel if row["environment"] == environment
        ]
        for held_seed in spec.REUSED_SELECTION_SEEDS:
            fit_rows = [
                row for row in environment_rows
                if row["evaluation_seed"] != int(held_seed)
            ]
            held_rows = [
                row for row in environment_rows
                if row["evaluation_seed"] == int(held_seed)
            ]
            if (
                len(fit_rows)
                != (len(spec.REUSED_SELECTION_SEEDS) - 1)
                * len(spec.DISTURBANCE_MODES)
                or len(held_rows) != len(spec.DISTURBANCE_MODES)
            ):
                raise RuntimeError("v17.8 grouped fold construction mismatch")
            for (window, penalty), gain_configs in gains_by_base.items():
                model = fit_causal_fir(
                    [row["total_action"] for row in fit_rows],
                    [row["oracle_upper_action"] for row in fit_rows],
                    window=window,
                    ridge_penalty=penalty,
                    feature_scale_floor=spec.FEATURE_SCALE_FLOOR,
                )
                for config in gain_configs:
                    candidate = str(config["candidate_id"])
                    for row in held_rows:
                        metrics = evaluate_causal_fir_split(
                            row["total_action"],
                            model,
                            output_gain=float(config["output_gain"]),
                            upper_action_limit=spec.UPPER_ACTION_LIMIT,
                            lower_action_limit=spec.LOWER_ACTION_LIMIT,
                            upper_window=spec.UPPER_WINDOW,
                            lower_window=spec.LOWER_WINDOW,
                            upper_power_budget=spec.UPPER_RMS_BUDGET ** 2,
                            lower_power_budget=spec.LOWER_RMS_BUDGET ** 2,
                            power_tolerance=spec.POWER_TOLERANCE,
                        )
                        valid = bool(
                            metrics["finite"]
                            and metrics["reconstruction_error_max"]
                            <= spec.RECONSTRUCTION_TOLERANCE
                            and metrics["bound_violation_max"]
                            <= spec.BOUND_TOLERANCE
                        )
                        recoverable = bool(
                            row["oracle_joint_feasible"]
                            and not row["baseline_joint_feasible"]
                        )
                        rows_by_candidate[candidate].append({
                            "candidate_id": candidate,
                            "environment": str(environment),
                            "disturbance_mode": row["disturbance_mode"],
                            "evaluation_seed": int(held_seed),
                            "fit_seed_count": len(
                                spec.REUSED_SELECTION_SEEDS
                            ) - 1,
                            "valid": valid,
                            "baseline_joint_feasible": bool(
                                row["baseline_joint_feasible"]
                            ),
                            "baseline_lower_power": float(
                                row["baseline_lower_power"]
                            ),
                            "oracle_joint_feasible": bool(
                                row["oracle_joint_feasible"]
                            ),
                            "oracle_recoverable_failure": recoverable,
                            "recovers_oracle_recoverable_failure": bool(
                                recoverable and metrics["joint_budget_pass"]
                            ),
                            "preserves_baseline_feasible_path": bool(
                                row["baseline_joint_feasible"]
                                and metrics["joint_budget_pass"]
                            ),
                            **metrics,
                        })
    for candidate, rows in rows_by_candidate.items():
        if len(rows) != spec.REUSED_EXPECTED_PATH_COUNT:
            raise RuntimeError(
                f"v17.8 OOF path count mismatch for {candidate}"
            )
    return rows_by_candidate


def summarize_candidate(
    config: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_environment: dict[str, Any] = {}
    for environment in spec.ENVIRONMENTS:
        selected = [
            row for row in rows if row["environment"] == environment
        ]
        recoverable = [
            row for row in selected if row["oracle_recoverable_failure"]
        ]
        baseline_feasible = [
            row for row in selected if row["baseline_joint_feasible"]
        ]
        recovered = sum(
            bool(row["recovers_oracle_recoverable_failure"])
            for row in selected
        )
        preserved = sum(
            bool(row["preserves_baseline_feasible_path"])
            for row in selected
        )
        mean_lower = float(np.mean([
            row["lower_power"] for row in selected
        ]))
        baseline_mean_lower = float(np.mean([
            row["baseline_lower_power"] for row in selected
        ]))
        by_environment[environment] = {
            "path_count": len(selected),
            "valid_path_count": sum(bool(row["valid"]) for row in selected),
            "upper_budget_path_count": sum(
                bool(row["upper_budget_pass"]) for row in selected
            ),
            "joint_budget_path_count": sum(
                bool(row["joint_budget_pass"]) for row in selected
            ),
            "oracle_recoverable_failure_count": len(recoverable),
            "recovered_failure_count": int(recovered),
            "recovery_rate": (
                float(recovered / len(recoverable)) if recoverable else 1.0
            ),
            "baseline_feasible_path_count": len(baseline_feasible),
            "preserved_baseline_feasible_path_count": int(preserved),
            "mean_lower_power": mean_lower,
            "baseline_mean_lower_power": baseline_mean_lower,
            "mean_lower_power_ratio": float(
                mean_lower / baseline_mean_lower
            ),
        }
    return {
        **config,
        "path_count": len(rows),
        "valid_path_count": sum(bool(row["valid"]) for row in rows),
        "upper_budget_path_count": sum(
            bool(row["upper_budget_pass"]) for row in rows
        ),
        "joint_budget_path_count": sum(
            bool(row["joint_budget_pass"]) for row in rows
        ),
        "oracle_recoverable_failure_count": sum(
            bool(row["oracle_recoverable_failure"]) for row in rows
        ),
        "recovered_failure_count": sum(
            bool(row["recovers_oracle_recoverable_failure"])
            for row in rows
        ),
        "baseline_feasible_path_count": sum(
            bool(row["baseline_joint_feasible"]) for row in rows
        ),
        "preserved_baseline_feasible_path_count": sum(
            bool(row["preserves_baseline_feasible_path"])
            for row in rows
        ),
        "mean_lower_power": float(np.mean([
            row["lower_power"] for row in rows
        ])),
        "by_environment": by_environment,
    }


def selection_key(summary: dict[str, Any]) -> tuple[Any, ...]:
    environments = list(summary["by_environment"].values())
    return (
        int(summary["valid_path_count"]),
        int(summary["upper_budget_path_count"]),
        min(float(row["recovery_rate"]) for row in environments),
        int(summary["recovered_failure_count"]),
        int(summary["preserved_baseline_feasible_path_count"]),
        -max(float(row["mean_lower_power_ratio"]) for row in environments),
        -float(summary["mean_lower_power"]),
    )


def reused_advancement_gate(summary: dict[str, Any]) -> dict[str, bool]:
    by_environment = summary["by_environment"]
    return {
        "expected_oracle_recoverable_denominator": bool(
            summary["oracle_recoverable_failure_count"]
            == spec.REUSED_EXPECTED_ORACLE_RECOVERABLE_FAILURES
        ),
        "all_paths_numerically_and_physically_valid": bool(
            summary["valid_path_count"] == spec.REUSED_EXPECTED_PATH_COUNT
        ),
        "all_paths_meet_endpoint_upper_budget": bool(
            summary["upper_budget_path_count"]
            == spec.REUSED_EXPECTED_PATH_COUNT
        ),
        "total_recovery_gate": bool(
            summary["recovered_failure_count"]
            >= spec.REUSED_RECOVERY_GATE_TOTAL
        ),
        "environment_recovery_gates": all(
            by_environment[environment]["recovered_failure_count"]
            >= minimum
            for environment, minimum in (
                spec.REUSED_RECOVERY_GATE_BY_ENVIRONMENT.items()
            )
        ),
        "expected_walker_baseline_feasible_denominator": bool(
            by_environment["Walker2d-v5"]["baseline_feasible_path_count"]
            == spec.REUSED_EXPECTED_BASELINE_FEASIBLE_WALKER_PATHS
        ),
        "walker_baseline_feasible_preservation_gate": bool(
            by_environment["Walker2d-v5"]
            ["preserved_baseline_feasible_path_count"]
            >= spec.REUSED_PRESERVE_BASELINE_FEASIBLE_WALKER_GATE
        ),
        "mean_lower_power_no_worse_each_environment": all(
            row["mean_lower_power"]
            <= row["baseline_mean_lower_power"] + spec.POWER_TOLERANCE
            for row in by_environment.values()
        ),
    }


def fit_final_models(
    panel: list[dict[str, Any]], selected: dict[str, Any]
) -> dict[str, Any]:
    environment_models = {}
    for environment in spec.ENVIRONMENTS:
        rows = [row for row in panel if row["environment"] == environment]
        model = fit_causal_fir(
            [row["total_action"] for row in rows],
            [row["oracle_upper_action"] for row in rows],
            window=int(selected["window"]),
            ridge_penalty=float(selected["ridge_penalty"]),
            feature_scale_floor=spec.FEATURE_SCALE_FLOOR,
        )
        environment_models[environment] = {
            **{
                key: value for key, value in model.items()
                if key != "coefficients"
            },
            "coefficients": np.asarray(
                model["coefficients"], dtype=np.float64
            ).tolist(),
        }
    return environment_models


def run_selection(dataset_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    panel = load_reused_panel(dataset_root)
    rows_by_candidate = grouped_out_of_fold_rows(panel)
    configs = {row["candidate_id"]: row for row in candidate_configs()}
    summaries = [
        summarize_candidate(configs[candidate], rows)
        for candidate, rows in rows_by_candidate.items()
    ]
    summaries.sort(key=lambda row: str(row["candidate_id"]))
    selected = max(summaries, key=selection_key)
    gate = reused_advancement_gate(selected)
    advances = bool(all(gate.values()))
    selected_rows = rows_by_candidate[str(selected["candidate_id"])]
    summary = {
        "status": (
            "causal_fir_advances_to_fresh_path_validation"
            if advances
            else "causal_fir_stops_before_fresh_path_access"
        ),
        "evidence_role": spec.EVIDENCE_ROLE,
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_core_revision": spec.FROZEN_CORE_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "dataset_root": str(dataset_root),
        "selection_contract": spec.SELECTION_CONTRACT,
        "candidate_count": len(summaries),
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate": selected,
        "advancement_gate": gate,
        "fresh_path_access_allowed": advances,
        "selected_out_of_fold_rows": selected_rows,
        "candidate_summaries": summaries,
        "runtime_seconds": float(time.perf_counter() - started),
        "claim_boundary": spec.SELECTION_CONTRACT["claim_boundary"],
    }
    model = {
        "status": "v17_8_selected_causal_fir_model_fitted",
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_core_revision": spec.FROZEN_CORE_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "selected_candidate_id": selected["candidate_id"],
        "window": int(selected["window"]),
        "ridge_penalty": float(selected["ridge_penalty"]),
        "output_gain": float(selected["output_gain"]),
        "upper_action_limit": spec.UPPER_ACTION_LIMIT,
        "lower_action_limit": spec.LOWER_ACTION_LIMIT,
        "fresh_validation_eligible": advances,
        "environment_models": fit_final_models(panel, selected),
        "claim_boundary": spec.SELECTION_CONTRACT["claim_boundary"],
    }
    return summary, model


def write_outputs(
    summary: dict[str, Any], model: dict[str, Any], output_dir: Path
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "selected_model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_identity = verify_current_freq_hrl_source_identity(
        code_revision=spec.FROZEN_CORE_REVISION,
        expected_source_manifest_sha256=spec.FROZEN_SOURCE_MANIFEST_SHA256,
        require_verified=True,
    )
    summary, model = run_selection(args.dataset_root)
    summary["source_identity"] = source_identity
    model["source_identity"] = source_identity
    write_outputs(summary, model, args.output_dir)
    print(
        f"DONE v17.8 selection status={summary['status']} "
        f"candidate={summary['selected_candidate_id']} "
        f"recovered={summary['selected_candidate']['recovered_failure_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
