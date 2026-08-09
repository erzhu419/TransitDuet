#!/usr/bin/env python3
"""Submit the frozen MuJoCo v14.11 iterative-projection screen."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import mujoco_v14_11_iterative_projection_screen_spec as spec  # noqa: E402
from scripts.submit_hyperparameter_pilot_scheduleurm import (  # noqa: E402
    LINUX_CPU_NODES,
    SCHEDULER,
    STAGE_EXCLUDES,
    default_python_executable,
    execute,
    execute_bulk,
    parse_csv,
    source_identity,
)


MODULE = "freq_hrl.experiments.mujoco.control_validation"
SIGNATURE_VERSION = "mujoco-v14-11-iterative-projection-screen-v1"
ANCHOR_ARM = "iterative_projection_pretrain_anchor"
PHASES = ("anchor", "continuation")
SPEC_PATH = Path(spec.__file__).resolve()
LAUNCHER_PATH = Path(__file__).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_identity() -> dict[str, str]:
    git_root = Path(subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()).resolve()
    revision = subprocess.run(
        ["git", "-C", str(git_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    paths = (LAUNCHER_PATH, SPEC_PATH)
    relatives = [str(path.relative_to(git_root)) for path in paths]
    subprocess.run(
        ["git", "-C", str(git_root), "ls-files", "--error-unmatch", *relatives],
        check=True,
        capture_output=True,
        text=True,
    )
    if subprocess.run(
        ["git", "-C", str(git_root), "diff", "--quiet", "HEAD", "--", *relatives]
    ).returncode != 0:
        raise RuntimeError("v14.11 screen launcher/spec do not match HEAD")
    return {
        "runtime_revision": revision,
        "launcher_sha256": _sha256(LAUNCHER_PATH),
        "spec_sha256": _sha256(SPEC_PATH),
    }


def experiment_cells(
    arms: tuple[str, ...] | list[str] = tuple(spec.ARMS),
    phases: tuple[str, ...] | list[str] = PHASES,
    environments: tuple[str, ...] | list[str] = spec.ENVIRONMENTS,
    optimizer_seeds: tuple[int, ...] | list[int] = spec.OPTIMIZER_SEEDS,
) -> list[tuple[str, str, str, int]]:
    cells: list[tuple[str, str, str, int]] = []
    if "anchor" in phases:
        cells.extend(
            ("anchor", environment, ANCHOR_ARM, int(seed))
            for environment in environments
            for seed in optimizer_seeds
        )
    if "continuation" in phases:
        cells.extend(
            ("continuation", environment, arm, int(seed))
            for environment in environments
            for arm in arms
            for seed in optimizer_seeds
        )
    return cells


def selected_experiment_cells(args: argparse.Namespace) -> list[
    tuple[str, str, str, int]
]:
    cells = experiment_cells(
        args.arms,
        args.phases,
        args.environments,
        args.optimizer_seeds,
    )
    return cells[:int(args.max_cells)] if int(args.max_cells) > 0 else cells


def anchor_relative_dir(
    run_name: str, *, environment: str, optimizer_seed: int
) -> Path:
    return (
        Path("results") / str(run_name) / "anchors" / str(environment)
        / f"replicate_{int(optimizer_seed)}"
    )


def cell_relative_dir(
    run_name: str,
    *,
    environment: str,
    arm: str,
    optimizer_seed: int,
) -> Path:
    return (
        Path("results") / str(run_name) / "cells" / str(environment)
        / str(arm) / f"replicate_{int(optimizer_seed)}"
    )


def task_relative_dir(
    run_name: str,
    *,
    phase: str,
    environment: str,
    arm: str,
    optimizer_seed: int,
) -> Path:
    if str(phase) == "anchor":
        return anchor_relative_dir(
            run_name,
            environment=environment,
            optimizer_seed=optimizer_seed,
        )
    return cell_relative_dir(
        run_name,
        environment=environment,
        arm=arm,
        optimizer_seed=optimizer_seed,
    )


def task_signature(
    run_name: str,
    *,
    phase: str,
    environment: str,
    arm: str,
    optimizer_seed: int,
) -> str:
    return (
        f"Freq-HRL/{SIGNATURE_VERSION}/{run_name}/{phase}/"
        f"{environment}/{arm}/rep-{int(optimizer_seed)}"
    )


def build_training_command(
    args: argparse.Namespace,
    *,
    phase: str,
    environment: str,
    arm: str,
    optimizer_seed: int,
    output_dir: Path,
) -> str:
    is_anchor = str(phase) == "anchor"
    arm_spec = spec.ANCHOR_SPEC if is_anchor else spec.ARMS[str(arm)]
    train_seeds = (
        spec.PRETRAIN_SEEDS if is_anchor else spec.CONTINUATION_TRAIN_SEEDS
    )
    selection_seeds = (
        spec.PRETRAIN_SELECTION_SEEDS
        if is_anchor else spec.CONTINUATION_SELECTION_SEEDS
    )
    iterations = (
        spec.PRETRAIN_ITERATIONS if is_anchor else spec.CONTINUATION_ITERATIONS
    )
    checkpoint_minimum_iteration = (
        spec.PRETRAIN_CHECKPOINT_MINIMUM_ITERATION
        if is_anchor else spec.checkpoint_minimum_iteration(str(arm))
    )
    command = [
        str(args.python_executable), "-u", "-m", MODULE,
        "--method", str(arm_spec["method"]),
        "--env-id", str(environment),
        "--disturbance-mode", "standard",
        "--training-disturbance-modes", *spec.TRAINING_DISTURBANCE_MODES,
        "--evaluation-disturbance-modes", *spec.EVALUATION_DISTURBANCE_MODES,
        "--train-seeds", *(str(seed) for seed in train_seeds),
        "--selection-seeds", *(str(seed) for seed in selection_seeds),
        "--eval-seeds", *(str(seed) for seed in spec.DEVELOPMENT_EVALUATION_SEEDS),
        "--steps", str(spec.STEPS),
        "--episode-horizon", str(spec.EPISODE_HORIZON),
        "--iterations", str(iterations),
        "--optimizer-seed", str(optimizer_seed),
        "--upper-period", str(spec.UPPER_PERIOD),
        "--hidden-dim", str(spec.HIDDEN_DIM),
        "--learning-rate", str(spec.LEARNING_RATE),
        "--lower-lf-rms-budget", str(spec.LOWER_LF_RMS_BUDGET),
        "--upper-hf-rms-budget", str(spec.UPPER_HF_RMS_BUDGET),
        "--upper-hf-penalty-coef", str(arm_spec["upper_hf_penalty_coef"]),
        "--upper-constraint-mode", str(arm_spec["upper_constraint_mode"]),
        "--upper-dual-lr", str(arm_spec["upper_dual_lr"]),
        "--lower-dual-lr", str(arm_spec["lower_dual_lr"]),
        "--upper-deployment-frequency-dual-lr",
        str(arm_spec["upper_deployment_frequency_dual_lr"]),
        "--lower-deployment-frequency-dual-lr",
        str(arm_spec["lower_deployment_frequency_dual_lr"]),
        "--upper-deployment-frequency-lambda-init",
        str(arm_spec["upper_deployment_frequency_lambda_init"]),
        "--lower-deployment-frequency-lambda-init",
        str(arm_spec["lower_deployment_frequency_lambda_init"]),
        "--upper-deployment-frequency-step-scale",
        str(arm_spec["upper_deployment_frequency_step_scale"]),
        "--lower-deployment-frequency-step-scale",
        str(arm_spec["lower_deployment_frequency_step_scale"]),
        "--upper-deployment-frequency-max-projection-steps",
        str(arm_spec["upper_deployment_frequency_max_projection_steps"]),
        "--lower-deployment-frequency-max-projection-steps",
        str(arm_spec["lower_deployment_frequency_max_projection_steps"]),
        "--upper-deployment-frequency-reward-tolerance",
        str(arm_spec["upper_deployment_frequency_reward_tolerance"]),
        "--lower-deployment-frequency-reward-tolerance",
        str(arm_spec["lower_deployment_frequency_reward_tolerance"]),
        "--upper-deployment-frequency-target-tolerance",
        str(arm_spec["upper_deployment_frequency_target_tolerance"]),
        "--lower-deployment-frequency-target-tolerance",
        str(arm_spec["lower_deployment_frequency_target_tolerance"]),
        "--upper-deployment-frequency-rms-budget",
        str(arm_spec["upper_deployment_frequency_rms_budget"]),
        "--lower-deployment-frequency-rms-budget",
        str(arm_spec["lower_deployment_frequency_rms_budget"]),
        "--upper-deployment-frequency-reference-reduction-fraction",
        str(arm_spec[
            "upper_deployment_frequency_reference_reduction_fraction"
        ]),
        "--lower-deployment-frequency-reference-reduction-fraction",
        str(arm_spec[
            "lower_deployment_frequency_reference_reduction_fraction"
        ]),
        "--leakage-cost-mode", str(arm_spec["leakage_cost_mode"]),
        "--upper-constraint-update-mode", spec.UPPER_CONSTRAINT_UPDATE_MODE,
        "--lower-constraint-update-mode", spec.LOWER_CONSTRAINT_UPDATE_MODE,
        "--upper-action-scale", str(spec.UPPER_ACTION_SCALE),
        "--lower-action-scale", str(spec.LOWER_ACTION_SCALE),
        "--responsibility-mode", str(arm_spec["responsibility_mode"]),
        "--lower-action-router-mode", str(arm_spec["lower_action_router_mode"]),
        "--lower-action-router-alpha", str(arm_spec["lower_action_router_alpha"]),
        "--lower-action-router-strength", str(arm_spec["lower_action_router_strength"]),
        "--lower-action-router-training-schedule",
        str(arm_spec["lower_action_router_training_schedule"]),
        "--lower-action-router-warmup-fraction",
        str(arm_spec["lower_action_router_warmup_fraction"]),
        "--lower-action-router-ramp-fraction",
        str(arm_spec["lower_action_router_ramp_fraction"]),
        "--leakage-constraint-scope", str(arm_spec["leakage_constraint_scope"]),
        "--checkpoint-selection-mode", spec.CHECKPOINT_SELECTION_MODE,
        "--checkpoint-score-mode", str(arm_spec["checkpoint_score_mode"]),
        "--checkpoint-constraint-penalty",
        str(arm_spec["checkpoint_constraint_penalty"]),
        "--checkpoint-smoothing-window", str(spec.CHECKPOINT_SMOOTHING_WINDOW),
        "--checkpoint-min-delta", str(spec.CHECKPOINT_MIN_DELTA),
        "--checkpoint-minimum-iteration",
        str(checkpoint_minimum_iteration),
        "--checkpoint-evaluation-interval",
        str(spec.CHECKPOINT_EVALUATION_INTERVAL),
        "--code-revision", spec.FROZEN_ALGORITHM_REVISION,
        "--source-manifest-sha256", spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "--output-dir", str(output_dir),
    ]
    if bool(arm_spec["lower_action_router_observe_strength"]):
        command.append("--lower-action-router-observe-strength")
    if not is_anchor:
        anchor = anchor_relative_dir(
            args.run_name,
            environment=environment,
            optimizer_seed=optimizer_seed,
        )
        command.extend([
            "--initial-checkpoint-path", str(anchor / "checkpoint.pt"),
            "--initial-checkpoint-summary-path",
            str(anchor / "cell_summary.json"),
            "--initial-checkpoint-router-mode",
            str(spec.ANCHOR_SPEC["lower_action_router_mode"]),
            "--upper-actor-anchor-coef",
            str(arm_spec["upper_actor_anchor_coef"]),
            "--lower-actor-anchor-coef",
            str(arm_spec["lower_actor_anchor_coef"]),
        ])
    environment_variables = [
        "MUJOCO_GL=egl",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONPATH=.",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "TORCH_NUM_THREADS=1",
        "CUDA_VISIBLE_DEVICES=",
    ]
    return " ".join([*environment_variables, shlex.join(command)]) + " && echo DONE"


def build_scheduler_spec(
    args: argparse.Namespace,
    *,
    phase: str,
    environment: str,
    arm: str,
    optimizer_seed: int,
) -> dict[str, object]:
    relative = task_relative_dir(
        args.run_name,
        phase=phase,
        environment=environment,
        arm=arm,
        optimizer_seed=optimizer_seed,
    )
    absolute = ROOT / relative
    anchor = ROOT / anchor_relative_dir(
        args.run_name,
        environment=environment,
        optimizer_seed=optimizer_seed,
    )
    return {
        "project": "Freq-HRL-MuJoCo-v14.11-Iterative-Projection-Screen",
        "description": (
            f"Freq-HRL MuJoCo v14.11 {phase} {environment} {arm} "
            f"replicate {optimizer_seed}"
        ),
        "cmd": build_training_command(
            args,
            phase=phase,
            environment=environment,
            arm=arm,
            optimizer_seed=optimizer_seed,
            output_dir=relative,
        ),
        "cwd": str(ROOT),
        "signature": task_signature(
            args.run_name,
            phase=phase,
            environment=environment,
            arm=arm,
            optimizer_seed=optimizer_seed,
        ),
        "resource_family": (
            f"Freq-HRL/{SIGNATURE_VERSION}/{phase}/{arm}/{environment}/cell"
        ),
        "vram": 0,
        "ram_mb": 768,
        "cpu": 1,
        "priority": str(args.priority),
        "ckpt_dir": str(absolute),
        "ckpt_glob": "checkpoint.pt",
        "result_dir": str(absolute),
        "local_result_dir": str(absolute),
        "wait_for_files": (
            [] if str(phase) == "anchor" else [
                str(anchor / "checkpoint.pt"),
                str(anchor / "cell_summary.json"),
            ]
        ),
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Independent source-bound MuJoCo development cells use one core."
        ),
        "reroute_on_node_down": True,
        "node_down_requeue_s": 600,
        "allowed_nodes": list(args.nodes),
        "require_node": None,
        "stage_excludes": list(STAGE_EXCLUDES),
        "stage_input_paths": [],
        "skip_launch_staging": False,
        "allow_duplicate": False,
    }


def _write_preregistration(args: argparse.Namespace) -> None:
    output = ROOT / "results" / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    path = output / "preregistration.json"
    payload: dict[str, Any] = {
        "status": "frozen_before_v14_11_iterative_projection_outcome_access",
        "evidence_role": "development_screen_not_confirmatory",
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "development_disclosure": spec.DEVELOPMENT_DISCLOSURE,
        "anchor_spec": spec.ANCHOR_SPEC,
        "arms": spec.ARMS,
        "base_control_arm": spec.BASE_CONTROL_ARM,
        "calibration_arm": spec.CALIBRATION_ARM,
        "matched_comparator_arm": spec.MATCHED_COMPARATOR_ARM,
        "learned_arms": list(spec.LEARNED_ARMS),
        "legacy_one_step_arm": spec.LEGACY_ONE_STEP_ARM,
        "iterative_arms": list(spec.ITERATIVE_ARMS),
        "environments": list(spec.ENVIRONMENTS),
        "optimizer_seeds": list(spec.OPTIMIZER_SEEDS),
        "dispatched_environment_subset": list(args.environments),
        "dispatched_optimizer_seed_subset": list(args.optimizer_seeds),
        "pretrain_seeds": list(spec.PRETRAIN_SEEDS),
        "pretrain_selection_seeds": list(spec.PRETRAIN_SELECTION_SEEDS),
        "continuation_train_seeds": list(spec.CONTINUATION_TRAIN_SEEDS),
        "continuation_selection_seeds": list(
            spec.CONTINUATION_SELECTION_SEEDS
        ),
        "development_evaluation_seeds": list(
            spec.DEVELOPMENT_EVALUATION_SEEDS
        ),
        "pretrain_iterations": spec.PRETRAIN_ITERATIONS,
        "continuation_iterations": spec.CONTINUATION_ITERATIONS,
        "pretrain_checkpoint_minimum_iteration": (
            spec.PRETRAIN_CHECKPOINT_MINIMUM_ITERATION
        ),
        "continuation_checkpoint_minimum_iteration": (
            spec.CONTINUATION_CHECKPOINT_MINIMUM_ITERATION
        ),
        "checkpoint_minimum_iteration_by_arm": {
            arm: spec.checkpoint_minimum_iteration(arm)
            for arm in spec.ARMS
        },
        "analysis_trained_checkpoint_minimum_iteration": (
            spec.ANALYSIS_TRAINED_CHECKPOINT_MINIMUM_ITERATION
        ),
        "training_disturbance_modes": list(spec.TRAINING_DISTURBANCE_MODES),
        "evaluation_disturbance_modes": list(spec.EVALUATION_DISTURBANCE_MODES),
        "checkpoint_selection_mode": spec.CHECKPOINT_SELECTION_MODE,
        "checkpoint_smoothing_window": spec.CHECKPOINT_SMOOTHING_WINDOW,
        "checkpoint_min_delta": spec.CHECKPOINT_MIN_DELTA,
        "lower_lf_rms_budget": spec.LOWER_LF_RMS_BUDGET,
        "upper_hf_rms_budget": spec.UPPER_HF_RMS_BUDGET,
        "frozen_algorithm_revision": spec.FROZEN_ALGORITHM_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "runtime_revision": args.runtime_revision,
        "launcher_sha256": args.launcher_sha256,
        "spec_sha256": args.spec_sha256,
        "development_selection_gates": {
            "return_noninferiority_margin_fraction": (
                spec.RETURN_NONINFERIORITY_MARGIN_FRACTION
            ),
            "maximum_absolute_return_difference": (
                spec.MAXIMUM_ABSOLUTE_RETURN_DIFFERENCE
            ),
            "calibration_exact_trace_hash_match_required": True,
            "maximum_calibration_actor_rms": (
                spec.MAXIMUM_CALIBRATION_ACTOR_RMS
            ),
            "minimum_projection_lower_reduction_fraction": (
                spec.MINIMUM_PROJECTION_LOWER_REDUCTION_FRACTION
            ),
            "minimum_projection_upper_reduction_fraction": (
                spec.MINIMUM_PROJECTION_UPPER_REDUCTION_FRACTION
            ),
            "minimum_iterative_accepted_steps": (
                spec.MINIMUM_ITERATIVE_ACCEPTED_STEPS
            ),
            "minimum_iterative_multistep_updates": (
                spec.MINIMUM_ITERATIVE_MULTISTEP_UPDATES
            ),
            "minimum_iterative_reduction_gain_over_one_step": (
                spec.MINIMUM_ITERATIVE_REDUCTION_GAIN_OVER_ONE_STEP
            ),
            "maximum_projection_reward_budget_violations": (
                spec.MAXIMUM_PROJECTION_REWARD_BUDGET_VIOLATIONS
            ),
            "minimum_learned_parameter_rms": (
                spec.MINIMUM_LEARNED_PARAMETER_RMS
            ),
            "minimum_changed_action_trace_conditions": (
                spec.MINIMUM_CHANGED_ACTION_TRACE_CONDITIONS
            ),
            "minimum_changed_action_trace_environments": (
                spec.MINIMUM_CHANGED_ACTION_TRACE_ENVIRONMENTS
            ),
            "minimum_strict_reward_improvement_conditions": (
                spec.MINIMUM_STRICT_REWARD_IMPROVEMENT_CONDITIONS
            ),
            "minimum_responsibility_drift_reduction_fraction": (
                spec.MINIMUM_RESPONSIBILITY_DRIFT_REDUCTION_FRACTION
            ),
            "minimum_raw_lower_drift_reduction_fraction": (
                spec.MINIMUM_RAW_LOWER_DRIFT_REDUCTION_FRACTION
            ),
            "minimum_latent_lower_drift_reduction_fraction": (
                spec.MINIMUM_LATENT_LOWER_DRIFT_REDUCTION_FRACTION
            ),
            "minimum_latent_upper_hf_reduction_fraction": (
                spec.MINIMUM_LATENT_UPPER_HF_REDUCTION_FRACTION
            ),
            "maximum_upper_hf_rms": spec.UPPER_HF_REPORTING_GATE,
            "maximum_latent_upper_hf_rms": (
                spec.LATENT_UPPER_HF_REPORTING_GATE
            ),
            "lower_drift_materiality_floor": (
                spec.LOWER_DRIFT_MATERIALITY_FLOOR
            ),
            "upper_hf_materiality_floor": (
                spec.UPPER_HF_MATERIALITY_FLOOR
            ),
            "minimum_effective_lower_action_rms_fraction": (
                spec.MINIMUM_EFFECTIVE_LOWER_ACTION_RMS_FRACTION
            ),
            "maximum_router_clip_rate": spec.MAXIMUM_ROUTER_CLIP_RATE,
            "minimum_trained_checkpoint_fraction": (
                spec.MINIMUM_TRAINED_CHECKPOINT_FRACTION
            ),
            "selection_confidence": spec.SELECTION_CONFIDENCE,
            "bootstrap_draws": spec.BOOTSTRAP_DRAWS,
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise RuntimeError("existing v14.11 preregistration differs")
    path.write_text(rendered, encoding="utf-8")


def _read_cell(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    summary = json.loads((path / "cell_summary.json").read_text(encoding="utf-8"))
    history = json.loads((path / "training_history.json").read_text(encoding="utf-8"))
    with (path / "evaluation_rows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not isinstance(summary, dict) or not isinstance(history, list) or not history:
        raise SystemExit(f"v14.11 cell payload is invalid: {path}")
    return summary, history, rows


def merge_results(args: argparse.Namespace) -> None:
    required = ("cell_summary.json", "training_history.json", "evaluation_rows.csv", "checkpoint.pt")
    tasks = selected_experiment_cells(args)
    sync_manifest_path = (
        ROOT / "results" / args.run_name / "merged"
        / "run_scoped_result_sync.json"
    )
    if not sync_manifest_path.is_file():
        raise SystemExit("v14.11 merge requires run-scoped result sync")
    sync_manifest = json.loads(sync_manifest_path.read_text(encoding="utf-8"))
    if (
        sync_manifest.get("status") != "run_scoped_result_sync_complete"
        or int(sync_manifest.get("cell_count", -1)) != len(tasks)
        or len(sync_manifest.get("task_ids", [])) != len(tasks)
    ):
        raise SystemExit("v14.11 run-scoped result sync manifest is invalid")
    expected = [
        (
            phase,
            environment,
            arm,
            seed,
            ROOT / task_relative_dir(
                args.run_name,
                phase=phase,
                environment=environment,
                arm=arm,
                optimizer_seed=seed,
            ),
        )
        for phase, environment, arm, seed in tasks
    ]
    missing = [
        path for _, _, _, _, path in expected
        if any(not (path / name).is_file() for name in required)
    ]
    if missing:
        raise SystemExit(
            f"cannot merge v14.11 screen: {len(missing)} cells missing; first={missing[0]}"
        )
    expected_selection_paths = (
        len(spec.CONTINUATION_SELECTION_SEEDS)
        * len(spec.TRAINING_DISTURBANCE_MODES)
    )
    expected_rows = (
        len(spec.DEVELOPMENT_EVALUATION_SEEDS)
        * len(spec.EVALUATION_DISTURBANCE_MODES)
    )
    expected_grid = {
        (mode, int(seed))
        for mode in spec.EVALUATION_DISTURBANCE_MODES
        for seed in spec.DEVELOPMENT_EVALUATION_SEEDS
    }
    issues: list[str] = []
    anchors: dict[tuple[str, int], dict[str, Any]] = {}
    cells: list[dict[str, Any]] = []
    for phase, environment, arm, seed, path in expected:
        summary, history, rows = _read_cell(path)
        checkpoint_file_sha256 = _sha256(path / "checkpoint.pt")
        common_checks = {
            "protocol": summary.get("protocol_version")
            == spec.FROZEN_CORE_PROTOCOL_VERSION,
            "environment": summary.get("environment") == environment,
            "optimizer_seed": int(summary.get("optimizer_seed", -1)) == seed,
            "revision": summary.get("code_revision")
            == spec.FROZEN_ALGORITHM_REVISION,
            "manifest": summary.get("source_manifest_sha256")
            == spec.FROZEN_SOURCE_MANIFEST_SHA256,
            "source_verified": summary.get("source_identity_status") == "verified",
            "checkpoint_hash": checkpoint_file_sha256
            == summary.get("checkpoint_file_sha256"),
            "evaluation_rows": len(rows) == expected_rows,
            "evaluation_grid": {
                (str(row["disturbance_mode"]), int(row["seed"])) for row in rows
            } == expected_grid,
            "heldout_once": int(summary.get("heldout_evaluation_pass_count", -1)) == 1,
            "evaluation_protocol_valid": all(
                math.isclose(float(row["protocol_valid"]), 1.0)
                and float(row[
                    "LowerRouterActionReconstructionRMS"
                ]) <= spec.MAXIMUM_RECONSTRUCTION_RMS
                and float(row[
                    "ResponsibilityReconstructionRMS"
                ]) <= spec.MAXIMUM_RECONSTRUCTION_RMS
                and math.isfinite(float(row["LatentLowerLFDriftAbs"]))
                and float(row["LatentLowerLFDriftAbs"]) >= 0.0
                and math.isfinite(float(row["LatentUpperHFPowerAbs"]))
                and float(row["LatentUpperHFPowerAbs"]) >= 0.0
                and len(str(row["RewardTraceSHA256"])) == 64
                and len(str(row["ExecutedActionTraceSHA256"])) == 64
                and len(str(row["LatentPolicyTraceSHA256"])) == 64
                for row in rows
            ),
        }
        if phase == "anchor":
            checks = {
                **common_checks,
                "method": summary.get("method") == spec.ANCHOR_SPEC["method"],
                "joint_router": summary.get(
                    "lower_action_router_mode"
                ) == "causal_joint_band_projection",
                "zero_strength": math.isclose(
                    float(summary.get("lower_action_router_strength", float("nan"))), 0.0
                ),
                "strength_hidden": summary.get(
                    "lower_action_router_observe_strength"
                ) is False,
                "responsibility": summary.get("responsibility_mode")
                == "additive",
                "constraint_scope": summary.get(
                    "leakage_constraint_scope"
                ) == spec.ANCHOR_SPEC["leakage_constraint_scope"],
                "constraint_cost_mode": summary.get(
                    "leakage_constraint_cost_mode"
                ) == spec.ANCHOR_SPEC["leakage_cost_mode"],
                "checkpoint_score_mode": summary.get(
                    "checkpoint_score_mode"
                ) == spec.ANCHOR_SPEC["checkpoint_score_mode"],
                "iterations": int(summary.get("iterations", -1))
                == spec.PRETRAIN_ITERATIONS,
                "train_seeds": summary.get("rollout_seed_roots")
                == list(spec.PRETRAIN_SEEDS),
                "selection_roots": summary.get("checkpoint_selection_seed_roots")
                == list(spec.PRETRAIN_SELECTION_SEEDS),
                "minimum_checkpoint_iteration": int(summary.get(
                    "checkpoint_minimum_eligible_iteration", -2
                )) == spec.PRETRAIN_CHECKPOINT_MINIMUM_ITERATION,
                "trained_checkpoint": int(summary.get(
                    "selected_checkpoint_iteration", -2
                )) >= spec.PRETRAIN_CHECKPOINT_MINIMUM_ITERATION,
                "function_preserving": summary.get(
                    "lower_action_router_function_preserving"
                ) is True,
                "upper_constraint_mode": summary.get(
                    "upper_constraint_mode"
                ) == spec.ANCHOR_SPEC["upper_constraint_mode"],
                "upper_dual_inactive": math.isclose(
                    float(summary.get("upper_dual_lr", float("nan"))), 0.0
                ),
                "lower_dual_inactive": math.isclose(
                    float(summary.get("lower_dual_lr", float("nan"))), 0.0
                ),
                "not_continuation": summary.get(
                    "paired_checkpoint_continuation"
                ) == {"enabled": False},
            }
            anchors[(environment, seed)] = {
                "path": path,
                "summary": summary,
                "checkpoint_file_sha256": checkpoint_file_sha256,
            }
        else:
            arm_spec = spec.ARMS[arm]
            anchor = anchors.get((environment, seed))
            if anchor is None:
                anchor_path = ROOT / anchor_relative_dir(
                    args.run_name,
                    environment=environment,
                    optimizer_seed=seed,
                )
                anchor_summary, _, _ = _read_cell(anchor_path)
                anchor = {
                    "path": anchor_path,
                    "summary": anchor_summary,
                    "checkpoint_file_sha256": _sha256(anchor_path / "checkpoint.pt"),
                }
                anchors[(environment, seed)] = anchor
            continuation = dict(summary.get("paired_checkpoint_continuation") or {})
            target_strength = float(arm_spec["lower_action_router_strength"])
            expected_strengths = spec.expected_router_training_strengths(arm)
            sampled_strengths = [
                float(row.get("sampled_lower_action_router_strength", float("nan")))
                for row in history[1:]
            ]
            selection_strengths = [
                float(row["LowerActionRouterStrength_mean"])
                for row in history
                if "LowerActionRouterStrength_mean" in row
            ]
            checks = {
                **common_checks,
                "method": summary.get("method") == arm_spec["method"],
                "iterations": int(summary.get("iterations", -1))
                == spec.CONTINUATION_ITERATIONS,
                "train_seeds": summary.get("rollout_seed_roots")
                == list(spec.CONTINUATION_TRAIN_SEEDS),
                "selection_roots": summary.get("checkpoint_selection_seed_roots")
                == list(spec.CONTINUATION_SELECTION_SEEDS),
                "selection_paths": int(summary.get(
                    "checkpoint_selection_path_count", -1
                )) == expected_selection_paths,
                "minimum_checkpoint_iteration": int(summary.get(
                    "checkpoint_minimum_eligible_iteration", -2
                )) == spec.checkpoint_minimum_iteration(arm),
                "checkpoint_selection_valid": int(summary.get(
                    "selected_checkpoint_iteration", -2
                )) >= spec.checkpoint_minimum_iteration(arm),
                "router_mode": summary.get("lower_action_router_mode")
                == arm_spec["lower_action_router_mode"],
                "router_strength": math.isclose(
                    float(summary.get("lower_action_router_strength", float("nan"))),
                    target_strength,
                ),
                "router_schedule": summary.get(
                    "lower_action_router_training_schedule"
                ) == arm_spec["lower_action_router_training_schedule"],
                "schedule_payload": all(
                    math.isclose(float(actual), expected, abs_tol=1e-12)
                    for actual, expected in zip(
                        summary.get("lower_action_router_training_strengths_by_iteration", []),
                        expected_strengths,
                        strict=True,
                    )
                ) if len(summary.get(
                    "lower_action_router_training_strengths_by_iteration", []
                )) == len(expected_strengths) else False,
                "schedule_history": len(sampled_strengths) == len(expected_strengths)
                and all(
                    math.isclose(actual, expected, abs_tol=1e-12)
                    for actual, expected in zip(sampled_strengths, expected_strengths)
                ),
                "selection_target": bool(selection_strengths) and all(
                    math.isclose(value, target_strength, abs_tol=1e-12)
                    for value in selection_strengths
                ),
                "evaluation_target": all(
                    math.isclose(
                        float(row["LowerActionRouterStrength"]),
                        target_strength,
                        abs_tol=1e-12,
                    )
                    for row in rows
                ),
                "upper_anchor_coef": math.isclose(
                    float(summary.get("upper_actor_anchor_coef", float("nan"))),
                    float(arm_spec["upper_actor_anchor_coef"]),
                ),
                "lower_anchor_coef": math.isclose(
                    float(summary.get("lower_actor_anchor_coef", float("nan"))),
                    float(arm_spec["lower_actor_anchor_coef"]),
                ),
                "anchor_contract": summary.get("actor_anchor_contract")
                == "frozen_matched_conservative_policy_same_state_analytic_gaussian_kl_v2",
                "same_state_contract": summary.get(
                    "actor_anchor_zero_state_indices"
                ) == [],
                "function_preserving": summary.get(
                    "lower_action_router_function_preserving"
                ) is True,
                "constraint_scope": summary.get(
                    "leakage_constraint_scope"
                ) == arm_spec["leakage_constraint_scope"],
                "constraint_cost_mode": summary.get(
                    "leakage_constraint_cost_mode"
                ) == arm_spec["leakage_cost_mode"],
                "upper_constraint_mode": summary.get(
                    "upper_constraint_mode"
                ) == arm_spec["upper_constraint_mode"],
                "upper_dual_lr": math.isclose(
                    float(summary.get("upper_dual_lr", float("nan"))),
                    float(arm_spec["upper_dual_lr"]),
                ),
                "lower_dual_lr": math.isclose(
                    float(summary.get("lower_dual_lr", float("nan"))),
                    float(arm_spec["lower_dual_lr"]),
                ),
                "upper_deployment_dual_lr": math.isclose(
                    float(summary.get(
                        "upper_deployment_frequency_dual_lr", float("nan")
                    )),
                    float(arm_spec["upper_deployment_frequency_dual_lr"]),
                ),
                "lower_deployment_dual_lr": math.isclose(
                    float(summary.get(
                        "lower_deployment_frequency_dual_lr", float("nan")
                    )),
                    float(arm_spec["lower_deployment_frequency_dual_lr"]),
                ),
                "upper_deployment_lambda_init": math.isclose(
                    float(summary.get(
                        "upper_deployment_frequency_lambda_init", float("nan")
                    )),
                    float(arm_spec["upper_deployment_frequency_lambda_init"]),
                ),
                "lower_deployment_lambda_init": math.isclose(
                    float(summary.get(
                        "lower_deployment_frequency_lambda_init", float("nan")
                    )),
                    float(arm_spec["lower_deployment_frequency_lambda_init"]),
                ),
                "upper_deployment_step_scale": math.isclose(
                    float(summary.get(
                        "upper_deployment_frequency_step_scale", float("nan")
                    )),
                    float(arm_spec["upper_deployment_frequency_step_scale"]),
                ),
                "lower_deployment_step_scale": math.isclose(
                    float(summary.get(
                        "lower_deployment_frequency_step_scale", float("nan")
                    )),
                    float(arm_spec["lower_deployment_frequency_step_scale"]),
                ),
                "upper_deployment_max_projection_steps": int(summary.get(
                    "upper_deployment_frequency_max_projection_steps", -1
                )) == int(arm_spec[
                    "upper_deployment_frequency_max_projection_steps"
                ]),
                "lower_deployment_max_projection_steps": int(summary.get(
                    "lower_deployment_frequency_max_projection_steps", -1
                )) == int(arm_spec[
                    "lower_deployment_frequency_max_projection_steps"
                ]),
                "upper_deployment_reward_tolerance": math.isclose(
                    float(summary.get(
                        "upper_deployment_frequency_reward_tolerance",
                        float("nan"),
                    )),
                    float(arm_spec[
                        "upper_deployment_frequency_reward_tolerance"
                    ]),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                "lower_deployment_reward_tolerance": math.isclose(
                    float(summary.get(
                        "lower_deployment_frequency_reward_tolerance",
                        float("nan"),
                    )),
                    float(arm_spec[
                        "lower_deployment_frequency_reward_tolerance"
                    ]),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                "upper_deployment_target_tolerance": math.isclose(
                    float(summary.get(
                        "upper_deployment_frequency_target_tolerance",
                        float("nan"),
                    )),
                    float(arm_spec[
                        "upper_deployment_frequency_target_tolerance"
                    ]),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                "lower_deployment_target_tolerance": math.isclose(
                    float(summary.get(
                        "lower_deployment_frequency_target_tolerance",
                        float("nan"),
                    )),
                    float(arm_spec[
                        "lower_deployment_frequency_target_tolerance"
                    ]),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                "deployment_projection_contract": summary.get(
                    "deployment_frequency_constraint_contract"
                ) == (
                    "episode_reset_differentiable_actor_mean_tanh_upper_hold_"
                    "hpf8_lower_lpf32_anchor_relative_target_with_absolute_"
                    "floor_and_dimensionless_iterative_cumulative_reward_"
                    "budget_projection_v4"
                ) if (
                    float(arm_spec["upper_deployment_frequency_dual_lr"]) > 0.0
                    or float(arm_spec["lower_deployment_frequency_dual_lr"]) > 0.0
                ) else summary.get(
                    "deployment_frequency_constraint_contract"
                ) == "disabled",
                "upper_deployment_budget": math.isclose(
                    float(summary.get(
                        "upper_deployment_frequency_rms_budget", float("nan")
                    )),
                    float(arm_spec["upper_deployment_frequency_rms_budget"]),
                ),
                "lower_deployment_budget": math.isclose(
                    float(summary.get(
                        "lower_deployment_frequency_rms_budget", float("nan")
                    )),
                    float(arm_spec["lower_deployment_frequency_rms_budget"]),
                ),
                "upper_deployment_reduction": math.isclose(
                    float(summary.get(
                        "upper_deployment_frequency_reference_reduction_fraction",
                        float("nan"),
                    )),
                    float(arm_spec[
                        "upper_deployment_frequency_reference_reduction_fraction"
                    ]),
                ),
                "lower_deployment_reduction": math.isclose(
                    float(summary.get(
                        "lower_deployment_frequency_reference_reduction_fraction",
                        float("nan"),
                    )),
                    float(arm_spec[
                        "lower_deployment_frequency_reference_reduction_fraction"
                    ]),
                ),
                "checkpoint_score_mode": summary.get(
                    "checkpoint_score_mode"
                ) == arm_spec["checkpoint_score_mode"],
                "checkpoint_constraint_penalty": math.isclose(
                    float(summary.get(
                        "checkpoint_constraint_penalty", float("nan")
                    )),
                    float(arm_spec["checkpoint_constraint_penalty"]),
                ),
                "paired_relative_selector": (
                    summary.get("checkpoint_selection_protocol")
                    == "state_aligned_lexicographic_validation_v1"
                    and summary.get("checkpoint_rank_contract")
                    == (
                        "state_aligned_paired_selection_path_reward_floor_and_"
                        "five_frequency_endpoint_relative_feasibility_v1"
                    )
                    and set(dict(summary.get(
                        "checkpoint_selected_rank"
                    ) or {})) == {
                        "negative_worst_paired_relative_violation",
                        "negative_paired_relative_violation_l2",
                        "worst_reward_floor_slack",
                    }
                    and dict(summary.get(
                        "paired_relative_checkpoint_baseline"
                    ) or {}).get("enabled") is True
                    and int(dict(summary.get(
                        "paired_relative_checkpoint_baseline"
                    ) or {}).get("row_count", -1)) == expected_selection_paths
                    and int(dict(summary.get(
                        "paired_relative_checkpoint_baseline"
                    ) or {}).get("heldout_rows_used", -1)) == 0
                )
                if arm_spec["checkpoint_score_mode"]
                == "paired_relative_frequency_feasibility_first"
                else True,
                "continuation_enabled": continuation.get("enabled") is True,
                "anchor_file_hash": continuation.get("checkpoint_file_sha256")
                == anchor["checkpoint_file_sha256"],
                "anchor_parameter_hash": continuation.get(
                    "checkpoint_parameter_sha256"
                ) == anchor["summary"].get("frozen_parameter_sha256"),
                "anchor_environment": continuation.get("checkpoint_environment")
                == environment,
                "anchor_optimizer": int(continuation.get(
                    "checkpoint_optimizer_seed", -1
                )) == seed,
                "anchor_router": continuation.get("checkpoint_router_mode")
                == spec.ANCHOR_SPEC["lower_action_router_mode"],
                "anchor_router_strength_hidden": continuation.get(
                    "checkpoint_router_observe_strength"
                ) is False,
                "anchor_responsibility": continuation.get(
                    "checkpoint_responsibility_mode"
                ) == "additive",
                "anchor_metric_history": all(
                    math.isfinite(float(row.get("upper_actor_anchor_kl", float("nan"))))
                    and math.isfinite(float(row.get("lower_actor_anchor_kl", float("nan"))))
                    for row in history
                ),
            }
        issues.extend(
            f"{phase}/{arm}/{environment}/{seed}:{name}"
            for name, passed in checks.items() if not passed
        )
        cells.append({
            "phase": phase,
            "environment": environment,
            "arm": arm,
            "optimizer_seed": seed,
            "relative_dir": str(path.relative_to(ROOT)),
            "frozen_parameter_sha256": summary.get("frozen_parameter_sha256"),
            "checkpoint_file_sha256": summary.get("checkpoint_file_sha256"),
            "anchor_checkpoint_file_sha256": (
                None if phase == "anchor" else continuation.get(
                    "checkpoint_file_sha256"
                )
            ),
        })
    if issues:
        raise SystemExit("v14.11 screen merge failed: " + ",".join(issues[:20]))
    output = ROOT / "results" / args.run_name / "merged"
    output.mkdir(parents=True, exist_ok=True)
    full_scope = bool(
        set(args.arms) == set(spec.ARMS)
        and set(args.phases) == set(PHASES)
        and set(args.environments) == set(spec.ENVIRONMENTS)
        and set(args.optimizer_seeds) == set(spec.OPTIMIZER_SEEDS)
        and int(args.max_cells) == 0
    )
    (output / "cell_manifest.json").write_text(json.dumps({
        "status": (
            "development_screen_complete_unanalyzed"
            if full_scope else "development_scope_complete_unanalyzed"
        ),
        "evidence_role": "development_screen_not_confirmatory",
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "full_screen_scope": full_scope,
        "environments": list(args.environments),
        "optimizer_seeds": list(args.optimizer_seeds),
        "arms": list(args.arms),
        "phases": list(args.phases),
        "anchor_cell_count": sum(
            cell["phase"] == "anchor" for cell in cells
        ),
        "continuation_cell_count": sum(
            cell["phase"] == "continuation" for cell in cells
        ),
        "cell_count": len(cells),
        "cells": cells,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"merged {len(cells)} frozen MuJoCo v14.11 cells")


def _scheduler_tasks_for_run(run_name: str) -> dict[str, dict[str, Any]]:
    command = [
        sys.executable,
        str(SCHEDULER),
        "status",
        "--all",
        "--json",
        "--brief",
        "--readonly",
        "--lock-timeout",
        "30",
    ]
    completed = None
    for attempt in range(1, 4):
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            break
        if attempt < 3:
            time.sleep(float(attempt))
    if completed is None or completed.returncode != 0:
        raise RuntimeError(
            "v14.11 scheduler snapshot failed after three attempts: "
            + str((completed.stderr if completed else "")[-500:])
        )
    payload = json.loads(completed.stdout)
    prefix = f"Freq-HRL/{SIGNATURE_VERSION}/{run_name}/"
    matches: dict[str, dict[str, Any]] = {}
    for task in payload.get("tasks", []):
        signature = str(task.get("signature", ""))
        if not signature.startswith(prefix):
            continue
        if signature in matches:
            raise RuntimeError(
                f"duplicate scheduler signature during v14.11 sync: {signature}"
            )
        matches[signature] = task
    return matches


def sync_results(args: argparse.Namespace) -> None:
    required = (
        "cell_summary.json",
        "training_history.json",
        "evaluation_rows.csv",
        "checkpoint.pt",
    )
    expected: list[tuple[str, Path, dict[str, Any]]] = []
    scheduler_tasks = _scheduler_tasks_for_run(args.run_name)
    cells = selected_experiment_cells(args)
    for phase, environment, arm, seed in cells:
        signature = task_signature(
            args.run_name,
            phase=phase,
            environment=environment,
            arm=arm,
            optimizer_seed=seed,
        )
        task = scheduler_tasks.get(signature)
        if task is None:
            raise SystemExit(f"v14.11 sync task missing: {signature}")
        if task.get("status") != "done" or not task.get("node"):
            raise SystemExit(
                f"v14.11 sync task is not done: {task.get('id')} "
                f"status={task.get('status')}"
            )
        path = ROOT / task_relative_dir(
            args.run_name,
            phase=phase,
            environment=environment,
            arm=arm,
            optimizer_seed=seed,
        )
        expected.append((signature, path, task))

    scheduler_dir = str(SCHEDULER.parent)
    if scheduler_dir not in sys.path:
        sys.path.insert(0, scheduler_dir)
    import scheduler as scheduler_runtime  # type: ignore  # noqa: E402

    pending = [
        item for item in expected
        if any(not (item[1] / name).is_file() for name in required)
    ]
    errors: dict[str, str] = {}
    for attempt in range(1, 4):
        if not pending:
            break

        def sync_one(
            item: tuple[str, Path, dict[str, Any]],
        ) -> tuple[str, bool, str]:
            signature, path, task = item
            ok, message = scheduler_runtime._sync_one_result({
                "node": task["node"],
                "result_dir": str(path),
                "local_result_dir": str(path),
            })
            return signature, bool(ok), str(message)

        errors = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=int(args.sync_workers)
        ) as executor:
            for signature, ok, message in executor.map(sync_one, pending):
                if not ok:
                    errors[signature] = message
        pending = [
            item for item in pending
            if any(not (item[1] / name).is_file() for name in required)
        ]
        if pending and attempt < 3:
            time.sleep(float(attempt))
    if pending:
        signature, path, _ = pending[0]
        raise SystemExit(
            "v14.11 result sync incomplete: "
            f"{len(pending)} cells; first={signature}; path={path}; "
            f"error={errors.get(signature, 'missing required files')}"
        )

    output = ROOT / "results" / args.run_name / "merged"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "run_scoped_result_sync_complete",
        "run_name": args.run_name,
        "sync_workers": int(args.sync_workers),
        "cell_count": len(expected),
        "environments": list(args.environments),
        "optimizer_seeds": list(args.optimizer_seeds),
        "arms": list(args.arms),
        "phases": list(args.phases),
        "task_ids": [item[2]["id"] for item in expected],
        "node_counts": {
            node: sum(item[2]["node"] == node for item in expected)
            for node in sorted({str(item[2]["node"]) for item in expected})
        },
    }
    (output / "run_scoped_result_sync.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"synced {len(expected)} MuJoCo v14.11 cells with "
        f"{int(args.sync_workers)} workers"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--arms", default=",".join(spec.ARMS))
    parser.add_argument("--phases", default=",".join(PHASES))
    parser.add_argument("--nodes", default=",".join(LINUX_CPU_NODES))
    parser.add_argument("--environments", default=",".join(spec.ENVIRONMENTS))
    parser.add_argument(
        "--optimizer-seeds",
        default=",".join(str(seed) for seed in spec.OPTIMIZER_SEEDS),
    )
    parser.add_argument("--python-executable", default="")
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="normal"
    )
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--skip-complete-cells", action="store_true")
    parser.add_argument("--sync-only", action="store_true")
    parser.add_argument("--sync-workers", type=int, default=4)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.arms = parse_csv(args.arms)
    args.phases = parse_csv(args.phases)
    args.nodes = parse_csv(args.nodes)
    args.environments = parse_csv(args.environments)
    try:
        args.optimizer_seeds = [
            int(seed) for seed in parse_csv(args.optimizer_seeds)
        ]
    except ValueError as exc:
        raise SystemExit("invalid v14.11 optimizer seed subset") from exc
    if not args.arms or not set(args.arms).issubset(spec.ARMS):
        raise SystemExit("invalid v14.11 screen arm registry")
    if not args.phases or not set(args.phases).issubset(PHASES):
        raise SystemExit("invalid v14.11 screen phase registry")
    if (
        not args.environments
        or not set(args.environments).issubset(spec.ENVIRONMENTS)
    ):
        raise SystemExit("invalid v14.11 environment subset")
    if (
        not args.optimizer_seeds
        or not set(args.optimizer_seeds).issubset(spec.OPTIMIZER_SEEDS)
        or len(args.optimizer_seeds) != len(set(args.optimizer_seeds))
    ):
        raise SystemExit("invalid v14.11 optimizer seed subset")
    unknown_nodes = sorted(set(args.nodes) - set(LINUX_CPU_NODES))
    if unknown_nodes:
        raise SystemExit(f"invalid v14.11 screen nodes: {unknown_nodes}")
    if not args.python_executable.strip():
        args.python_executable = default_python_executable(args.nodes)
    if not 1 <= int(args.sync_workers) <= 8:
        raise SystemExit("v14.11 sync workers must be in [1, 8]")
    return args


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    if args.sync_only:
        sync_results(args)
        return
    if args.merge_only:
        merge_results(args)
        return
    identity = runtime_identity()
    args.runtime_revision = identity["runtime_revision"]
    args.launcher_sha256 = identity["launcher_sha256"]
    args.spec_sha256 = identity["spec_sha256"]
    revision, manifest = source_identity(spec.FROZEN_ALGORITHM_REVISION)
    if (
        revision != spec.FROZEN_ALGORITHM_REVISION
        or manifest != spec.FROZEN_SOURCE_MANIFEST_SHA256
    ):
        raise SystemExit("v14.11 frozen algorithm identity mismatch")
    _write_preregistration(args)
    cells = selected_experiment_cells(args)
    if args.skip_complete_cells:
        cells = [
            cell for cell in cells
            if not (
                ROOT / task_relative_dir(
                    args.run_name,
                    phase=cell[0],
                    environment=cell[1],
                    arm=cell[2],
                    optimizer_seed=cell[3],
                ) / "cell_summary.json"
            ).is_file()
        ]
    if not cells:
        print("no v14.11 screen cells require submission")
        return
    print(
        f"run={args.run_name} cells={len(cells)} nodes={','.join(args.nodes)}",
        flush=True,
    )
    execute_bulk([
        build_scheduler_spec(
            args,
            phase=phase,
            environment=environment,
            arm=arm,
            optimizer_seed=seed,
        )
        for phase, environment, arm, seed in cells
    ], dry_run=bool(args.dry_run), intent_label=(
        f"Freq-HRL MuJoCo v14.11 iterative-projection screen {args.run_name}"
    ))
    if args.dispatch and not args.dry_run:
        execute([sys.executable, str(SCHEDULER), "dispatch"], dry_run=False)
    merge_command = [
        sys.executable,
        str(LAUNCHER_PATH),
        "--run-name", args.run_name,
        "--arms", ",".join(args.arms),
        "--phases", ",".join(args.phases),
        "--environments", ",".join(args.environments),
        "--optimizer-seeds", ",".join(
            str(seed) for seed in args.optimizer_seeds
        ),
        "--nodes", ",".join(args.nodes),
        "--merge-only",
    ]
    if int(args.max_cells) > 0:
        merge_command.extend(("--max-cells", str(args.max_cells)))
    print("merge after result sync: " + shlex.join(merge_command))


if __name__ == "__main__":
    main()
