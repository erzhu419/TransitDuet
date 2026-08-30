#!/usr/bin/env python3
"""Submit and synchronize the frozen v16.1 paired audit-gauge preflight."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import mujoco_v16_1_audit_gauge_paired_preflight_spec as spec  # noqa: E402
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
SIGNATURE_VERSION = "mujoco-v16-1-audit-gauge-paired-preflight-v2"
PHASES = ("anchor", "continuation")


def cell_relative_dir(
    run_name: str,
    phase: str,
    environment: str,
    arm: str,
    optimizer_seed: int,
) -> Path:
    return (
        Path("results")
        / str(run_name)
        / "cells"
        / str(phase)
        / str(environment)
        / str(arm)
        / f"replicate_{int(optimizer_seed)}"
    )


def anchor_relative_dir(
    run_name: str,
    environment: str,
    optimizer_seed: int,
) -> Path:
    return cell_relative_dir(
        run_name,
        "anchor",
        environment,
        spec.ANCHOR_ARM,
        optimizer_seed,
    )


def selected_cells(args: argparse.Namespace) -> list[tuple[str, str, str, int]]:
    cells: list[tuple[str, str, str, int]] = []
    if "anchor" in args.phases:
        cells.extend(
            ("anchor", environment, spec.ANCHOR_ARM, int(seed))
            for environment in args.environments
            for seed in args.optimizer_seeds
        )
    if "continuation" in args.phases:
        cells.extend(
            ("continuation", environment, arm, int(seed))
            for environment in args.environments
            for arm in args.arms
            for seed in args.optimizer_seeds
        )
    return cells


def task_signature(
    run_name: str,
    phase: str,
    environment: str,
    arm: str,
    optimizer_seed: int,
) -> str:
    return (
        f"Freq-HRL/{SIGNATURE_VERSION}/{run_name}/{phase}/{environment}/"
        f"{arm}/rep-{int(optimizer_seed)}"
    )


def build_training_command(
    args: argparse.Namespace,
    phase: str,
    environment: str,
    arm: str,
    optimizer_seed: int,
) -> str:
    is_anchor = str(phase) == "anchor"
    arm_spec = spec.ANCHOR_SPEC if is_anchor else spec.ARMS[str(arm)]
    train_seeds = (
        spec.PRETRAIN_SEEDS if is_anchor else spec.CONTINUATION_TRAIN_SEEDS
    )
    selection_seeds = (
        spec.PRETRAIN_SELECTION_SEEDS
        if is_anchor
        else spec.CONTINUATION_SELECTION_SEEDS
    )
    iterations = (
        spec.PRETRAIN_ITERATIONS if is_anchor else spec.CONTINUATION_ITERATIONS
    )
    minimum_iteration = (
        spec.PRETRAIN_CHECKPOINT_MINIMUM_ITERATION
        if is_anchor
        else spec.CONTINUATION_CHECKPOINT_MINIMUM_ITERATION
    )
    output = cell_relative_dir(
        args.run_name, phase, environment, arm, optimizer_seed
    )
    command = [
        str(args.python_executable),
        "-u",
        "-m",
        MODULE,
        "--method",
        str(arm_spec["method"]),
        "--env-id",
        str(environment),
        "--disturbance-mode",
        "standard",
        "--training-disturbance-modes",
        *spec.TRAINING_DISTURBANCE_MODES,
        "--evaluation-disturbance-modes",
        *spec.EVALUATION_DISTURBANCE_MODES,
        "--train-seeds",
        *(str(seed) for seed in train_seeds),
        "--selection-seeds",
        *(str(seed) for seed in selection_seeds),
        "--eval-seeds",
        *(str(seed) for seed in spec.EVALUATION_SEEDS),
        "--steps",
        str(spec.STEPS),
        "--episode-horizon",
        str(spec.EPISODE_HORIZON),
        "--iterations",
        str(iterations),
        "--optimizer-seed",
        str(optimizer_seed),
        "--upper-period",
        str(spec.UPPER_PERIOD),
        "--hidden-dim",
        str(spec.HIDDEN_DIM),
        "--learning-rate",
        str(spec.LEARNING_RATE),
        "--lower-lf-rms-budget",
        str(spec.LOWER_LF_RMS_BUDGET),
        "--upper-hf-rms-budget",
        str(spec.UPPER_HF_RMS_BUDGET),
        "--upper-action-scale",
        str(spec.UPPER_ACTION_SCALE),
        "--lower-action-scale",
        str(spec.LOWER_ACTION_SCALE),
        "--responsibility-mode",
        str(arm_spec["responsibility_mode"]),
        "--lower-action-router-mode",
        str(arm_spec["lower_action_router_mode"]),
        "--lower-action-router-alpha",
        str(arm_spec["lower_action_router_alpha"]),
        "--lower-action-router-strength",
        str(arm_spec["lower_action_router_strength"]),
        "--lower-action-router-training-schedule",
        str(arm_spec["lower_action_router_training_schedule"]),
        "--lower-action-router-warmup-fraction",
        str(arm_spec["lower_action_router_warmup_fraction"]),
        "--lower-action-router-ramp-fraction",
        str(arm_spec["lower_action_router_ramp_fraction"]),
        "--leakage-constraint-scope",
        str(arm_spec["leakage_constraint_scope"]),
        "--leakage-cost-mode",
        str(arm_spec["leakage_cost_mode"]),
        "--upper-constraint-mode",
        str(arm_spec["upper_constraint_mode"]),
        "--upper-hf-penalty-coef",
        str(arm_spec["upper_hf_penalty_coef"]),
        "--upper-dual-lr",
        str(arm_spec["upper_dual_lr"]),
        "--lower-dual-lr",
        str(arm_spec["lower_dual_lr"]),
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
        "--constraint-dual-normalization",
        str(arm_spec["constraint_dual_normalization"]),
        "--constraint-dual-scale-ema-beta",
        str(arm_spec["constraint_dual_scale_ema_beta"]),
        "--constraint-dual-scale-floor",
        str(arm_spec["constraint_dual_scale_floor"]),
        "--upper-constraint-update-mode",
        "reward_guarded_adam_projection",
        "--lower-constraint-update-mode",
        "reward_guarded_adam_projection",
        "--checkpoint-selection-mode",
        spec.CHECKPOINT_SELECTION_MODE,
        "--checkpoint-score-mode",
        str(arm_spec["checkpoint_score_mode"]),
        "--checkpoint-smoothing-window",
        str(spec.CHECKPOINT_SMOOTHING_WINDOW),
        "--checkpoint-min-delta",
        str(spec.CHECKPOINT_MIN_DELTA),
        "--checkpoint-minimum-iteration",
        str(minimum_iteration),
        "--checkpoint-evaluation-interval",
        str(spec.CHECKPOINT_EVALUATION_INTERVAL),
        "--upper-deployment-frequency-rms-budget",
        str(spec.UPPER_HF_RMS_BUDGET),
        "--lower-deployment-frequency-rms-budget",
        str(spec.LOWER_LF_RMS_BUDGET),
        "--upper-deployment-frequency-reference-reduction-fraction",
        str(arm_spec[
            "upper_deployment_frequency_reference_reduction_fraction"
        ]),
        "--lower-deployment-frequency-reference-reduction-fraction",
        str(arm_spec[
            "lower_deployment_frequency_reference_reduction_fraction"
        ]),
        "--control-protocol-version",
        spec.FROZEN_CORE_PROTOCOL_VERSION,
        "--code-revision",
        spec.FROZEN_ALGORITHM_REVISION,
        "--source-manifest-sha256",
        spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "--output-dir",
        str(output),
    ]
    if bool(arm_spec["lower_action_router_observe_strength"]):
        command.append("--lower-action-router-observe-strength")
    if not is_anchor:
        anchor = anchor_relative_dir(
            args.run_name, environment, optimizer_seed
        )
        command.extend([
            "--initial-checkpoint-path",
            str(anchor / "checkpoint.pt"),
            "--initial-checkpoint-summary-path",
            str(anchor / "cell_summary.json"),
            "--initial-checkpoint-router-mode",
            str(spec.ANCHOR_SPEC["lower_action_router_mode"]),
            "--initial-checkpoint-router-strength",
            str(spec.ANCHOR_SPEC["lower_action_router_strength"]),
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
    phase: str,
    environment: str,
    arm: str,
    optimizer_seed: int,
) -> dict[str, object]:
    relative = cell_relative_dir(
        args.run_name, phase, environment, arm, optimizer_seed
    )
    absolute = ROOT / relative
    anchor = ROOT / anchor_relative_dir(
        args.run_name, environment, optimizer_seed
    )
    return {
        "project": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "description": (
            f"Freq-HRL v16.1 {phase} {environment} {arm} {optimizer_seed}"
        ),
        "cmd": build_training_command(
            args, phase, environment, arm, optimizer_seed
        ),
        "cwd": str(ROOT),
        "signature": task_signature(
            args.run_name, phase, environment, arm, optimizer_seed
        ),
        "resource_family": (
            f"Freq-HRL/{SIGNATURE_VERSION}/{phase}/{arm}/{environment}/cell"
        ),
        "vram": 0,
        "ram_mb": 1024,
        "cpu": 1,
        "priority": str(args.priority),
        "ckpt_dir": str(absolute),
        "ckpt_glob": "checkpoint.pt",
        "result_dir": str(absolute),
        "local_result_dir": str(absolute),
        "wait_for_files": (
            []
            if str(phase) == "anchor"
            else [
                str(anchor / "checkpoint.pt"),
                str(anchor / "cell_summary.json"),
            ]
        ),
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Independent paired MuJoCo cells use one physical CPU core."
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
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "status": spec.PREREGISTRATION_STATUS,
        "evidence_role": spec.EVIDENCE_ROLE,
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "launcher_source_revision": revision,
        "frozen_algorithm_revision": spec.FROZEN_ALGORITHM_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "selection_contract": spec.SELECTION_CONTRACT,
        "anchor_spec": spec.ANCHOR_SPEC,
        "continuation_arms": spec.ARMS,
        "environments": list(args.environments),
        "optimizer_seeds": list(args.optimizer_seeds),
        "pretrain_seeds": list(spec.PRETRAIN_SEEDS),
        "pretrain_selection_seeds": list(spec.PRETRAIN_SELECTION_SEEDS),
        "continuation_train_seeds": list(spec.CONTINUATION_TRAIN_SEEDS),
        "continuation_selection_seeds": list(
            spec.CONTINUATION_SELECTION_SEEDS
        ),
        "evaluation_seeds": list(spec.EVALUATION_SEEDS),
        "training_disturbance_modes": list(spec.TRAINING_DISTURBANCE_MODES),
        "evaluation_disturbance_modes": list(spec.EVALUATION_DISTURBANCE_MODES),
        "scheduler_contract": {
            "scheduler": "scheduleurm",
            "allowed_nodes": list(args.nodes),
            "require_node": None,
            "cpu_per_task": 1,
            "ram_mb_per_task": 1024,
            "slurm_used": False,
        },
    }
    (output / "preregistration.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _scheduler_tasks(run_name: str) -> dict[str, dict[str, Any]]:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCHEDULER),
            "status",
            "--all",
            "--json",
            "--brief",
            "--readonly",
            "--lock-timeout",
            "30",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    prefix = f"Freq-HRL/{SIGNATURE_VERSION}/{run_name}/"
    grouped: dict[str, list[dict[str, Any]]] = {}
    for task in json.loads(completed.stdout).get("tasks", []):
        signature = str(task.get("signature", ""))
        if signature.startswith(prefix):
            grouped.setdefault(signature, []).append(dict(task))
    selected: dict[str, dict[str, Any]] = {}
    for signature, attempts in grouped.items():
        done = [task for task in attempts if task.get("status") == "done"]
        if len(done) > 1:
            raise RuntimeError(
                f"multiple successful scheduler attempts for {signature}"
            )
        selected[signature] = done[0] if done else attempts[-1]
    return selected


def sync_results(args: argparse.Namespace) -> None:
    required = (
        "cell_summary.json",
        "training_history.json",
        "evaluation_rows.csv",
        "checkpoint.pt",
    )
    tasks = _scheduler_tasks(args.run_name)
    expected: list[tuple[str, Path, dict[str, Any]]] = []
    for phase, environment, arm, seed in selected_cells(args):
        signature = task_signature(
            args.run_name, phase, environment, arm, seed
        )
        task = tasks.get(signature)
        if task is None:
            raise SystemExit(f"v16.1 sync task missing: {signature}")
        if task.get("status") != "done" or not task.get("node"):
            raise SystemExit(
                f"v16.1 task is not done: {task.get('id')} "
                f"status={task.get('status')}"
            )
        path = ROOT / cell_relative_dir(
            args.run_name, phase, environment, arm, seed
        )
        expected.append((signature, path, task))

    scheduler_dir = str(SCHEDULER.parent)
    if scheduler_dir not in sys.path:
        sys.path.insert(0, scheduler_dir)
    import scheduler as scheduler_runtime  # type: ignore  # noqa: E402

    pending = [
        item
        for item in expected
        if any(not (item[1] / name).is_file() for name in required)
    ]

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

    errors: dict[str, str] = {}
    for attempt in range(1, 4):
        if not pending:
            break
        errors = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=int(args.sync_workers)
        ) as executor:
            for signature, ok, message in executor.map(sync_one, pending):
                if not ok:
                    errors[signature] = message
        pending = [
            item
            for item in pending
            if any(not (item[1] / name).is_file() for name in required)
        ]
        if pending and attempt < 3:
            time.sleep(float(attempt))
    if pending:
        signature, path, _ = pending[0]
        raise SystemExit(
            f"v16.1 result sync incomplete: {len(pending)} cells; "
            f"first={signature}; path={path}; "
            f"error={errors.get(signature, 'missing required files')}"
        )
    manifest = {
        "run_name": args.run_name,
        "cell_count": len(expected),
        "nodes": {
            node: sum(str(item[2]["node"]) == node for item in expected)
            for node in sorted({str(item[2]["node"]) for item in expected})
        },
        "tasks": {
            signature: {
                "task_id": task["id"],
                "node": task["node"],
            }
            for signature, _, task in expected
        },
    }
    output = ROOT / "results" / args.run_name
    (output / "run_scoped_result_sync.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"synced {len(expected)} v16.1 cells")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--phases", default=",".join(PHASES))
    parser.add_argument("--arms", default=",".join(spec.ARMS))
    parser.add_argument("--nodes", default=",".join(LINUX_CPU_NODES))
    parser.add_argument("--environments", default=",".join(spec.ENVIRONMENTS))
    parser.add_argument(
        "--optimizer-seeds",
        default=",".join(map(str, spec.OPTIMIZER_SEEDS)),
    )
    parser.add_argument("--python-executable", default="")
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="normal"
    )
    parser.add_argument("--skip-complete-cells", action="store_true")
    parser.add_argument("--sync-only", action="store_true")
    parser.add_argument("--sync-workers", type=int, default=4)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.phases = parse_csv(args.phases)
    args.arms = parse_csv(args.arms)
    args.nodes = parse_csv(args.nodes)
    args.environments = parse_csv(args.environments)
    try:
        args.optimizer_seeds = [
            int(value) for value in parse_csv(args.optimizer_seeds)
        ]
    except ValueError as exc:
        raise SystemExit("invalid v16.1 optimizer seed subset") from exc
    if not args.phases or not set(args.phases).issubset(PHASES):
        raise SystemExit("invalid v16.1 phase subset")
    if not args.arms or not set(args.arms).issubset(spec.ARMS):
        raise SystemExit("invalid v16.1 continuation arm subset")
    if not args.environments or not set(args.environments).issubset(
        spec.ENVIRONMENTS
    ):
        raise SystemExit("invalid v16.1 environment subset")
    if (
        not args.optimizer_seeds
        or not set(args.optimizer_seeds).issubset(spec.OPTIMIZER_SEEDS)
        or len(args.optimizer_seeds) != len(set(args.optimizer_seeds))
    ):
        raise SystemExit("invalid v16.1 optimizer seed subset")
    if set(args.nodes) - set(LINUX_CPU_NODES):
        raise SystemExit("v16.1 accepts only node001-node006")
    if not 1 <= int(args.sync_workers) <= 8:
        raise SystemExit("v16.1 sync workers must be in [1, 8]")
    if not args.python_executable.strip():
        args.python_executable = default_python_executable(args.nodes)
    return args


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    if args.sync_only:
        sync_results(args)
        return
    revision, manifest = source_identity(spec.FROZEN_ALGORITHM_REVISION)
    if (
        revision != spec.FROZEN_ALGORITHM_REVISION
        or manifest != spec.FROZEN_SOURCE_MANIFEST_SHA256
    ):
        raise SystemExit("v16.1 frozen algorithm identity mismatch")
    _write_preregistration(args)
    cells = selected_cells(args)
    if args.skip_complete_cells:
        cells = [
            cell
            for cell in cells
            if not (
                ROOT / cell_relative_dir(args.run_name, *cell) / "cell_summary.json"
            ).is_file()
        ]
    if not cells:
        print("no v16.1 cells require submission")
        return
    print(
        f"run={args.run_name} cells={len(cells)} "
        f"nodes={','.join(args.nodes)}",
        flush=True,
    )
    execute_bulk(
        [build_scheduler_spec(args, *cell) for cell in cells],
        dry_run=bool(args.dry_run),
        intent_label=(
            f"Freq-HRL {spec.DEVELOPMENT_PROTOCOL_VERSION} {args.run_name}"
        ),
    )
    if args.dispatch and not args.dry_run:
        execute([sys.executable, str(SCHEDULER), "dispatch"], dry_run=False)


if __name__ == "__main__":
    main()
