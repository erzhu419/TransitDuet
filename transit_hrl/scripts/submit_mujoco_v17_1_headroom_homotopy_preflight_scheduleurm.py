#!/usr/bin/env python3
"""Submit and synchronize the frozen MuJoCo v17.1 development preflight."""

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

from scripts import (  # noqa: E402
    mujoco_v17_1_headroom_homotopy_preflight_spec as spec,
)
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


WORKER = Path("scripts/run_mujoco_cell_small_export.py")
SIGNATURE_VERSION = "mujoco-v17-1-headroom-homotopy-preflight-v1"
SMALL_RESULT_FILES = (
    "cell_summary.json",
    "evaluation_rows.csv",
    "server_artifact_location.json",
)


def cell_relative_dir(
    run_name: str,
    environment: str,
    arm: str,
    optimizer_seed: int,
) -> Path:
    return (
        Path("results")
        / str(run_name)
        / "cells"
        / str(environment)
        / str(arm)
        / f"replicate_{int(optimizer_seed)}"
    )


def full_cell_relative_dir(
    run_name: str,
    environment: str,
    arm: str,
    optimizer_seed: int,
) -> Path:
    return (
        Path(".server_artifacts")
        / str(run_name)
        / "cells"
        / str(environment)
        / str(arm)
        / f"replicate_{int(optimizer_seed)}"
    )


def selected_cells(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    return [
        (environment, arm, int(seed))
        for environment in args.environments
        for arm in args.arms
        for seed in args.optimizer_seeds
    ]


def task_signature(
    run_name: str,
    environment: str,
    arm: str,
    optimizer_seed: int,
) -> str:
    return (
        f"Freq-HRL/{SIGNATURE_VERSION}/{run_name}/{environment}/{arm}/"
        f"rep-{int(optimizer_seed)}"
    )


def build_training_command(
    args: argparse.Namespace,
    environment: str,
    arm: str,
    optimizer_seed: int,
) -> str:
    arm_spec = spec.ARMS[str(arm)]
    full_output = full_cell_relative_dir(
        args.run_name, environment, arm, optimizer_seed
    )
    export_output = cell_relative_dir(
        args.run_name, environment, arm, optimizer_seed
    )
    control_args = [
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
        *(str(seed) for seed in spec.TRAIN_SEEDS),
        "--selection-seeds",
        *(str(seed) for seed in spec.SELECTION_SEEDS),
        "--safety-selection-seeds",
        *(str(seed) for seed in spec.SAFETY_SELECTION_SEEDS),
        "--eval-seeds",
        *(str(seed) for seed in spec.EVALUATION_SEEDS),
        "--steps",
        str(spec.STEPS),
        "--episode-horizon",
        str(spec.EPISODE_HORIZON),
        "--iterations",
        str(spec.ITERATIONS),
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
        "--upper-action-decoder-mode",
        str(arm_spec["upper_action_decoder_mode"]),
        "--upper-promotion-gain",
        str(arm_spec["upper_promotion_gain"]),
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
        spec.CHECKPOINT_SCORE_MODE,
        "--checkpoint-smoothing-window",
        str(spec.CHECKPOINT_SMOOTHING_WINDOW),
        "--checkpoint-min-delta",
        str(spec.CHECKPOINT_MIN_DELTA),
        "--checkpoint-minimum-iteration",
        str(spec.CHECKPOINT_MINIMUM_ITERATION),
        "--checkpoint-evaluation-interval",
        str(spec.CHECKPOINT_EVALUATION_INTERVAL),
        "--control-protocol-version",
        str(arm_spec["control_protocol_version"]),
        "--code-revision",
        spec.FROZEN_ALGORITHM_REVISION,
        "--source-manifest-sha256",
        spec.FROZEN_SOURCE_MANIFEST_SHA256,
    ]
    if bool(arm_spec["lower_action_router_observe_strength"]):
        control_args.append("--lower-action-router-observe-strength")
    command = [
        str(args.python_executable),
        "-u",
        str(WORKER),
        "--full-output-dir",
        str(full_output),
        "--export-output-dir",
        str(export_output),
        "--server-full-output-dir",
        str(full_output),
        "--",
        *control_args,
    ]
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
    return " ".join([*environment_variables, shlex.join(command)])


def build_scheduler_spec(
    args: argparse.Namespace,
    environment: str,
    arm: str,
    optimizer_seed: int,
) -> dict[str, object]:
    relative = cell_relative_dir(args.run_name, environment, arm, optimizer_seed)
    full_relative = full_cell_relative_dir(
        args.run_name, environment, arm, optimizer_seed
    )
    export_absolute = ROOT / relative
    full_absolute = ROOT / full_relative
    return {
        "project": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "description": f"Freq-HRL v17.1 {environment} {arm} {optimizer_seed}",
        "cmd": build_training_command(args, environment, arm, optimizer_seed),
        "cwd": str(ROOT),
        "signature": task_signature(
            args.run_name, environment, arm, optimizer_seed
        ),
        "resource_family": (
            f"Freq-HRL/{SIGNATURE_VERSION}/{arm}/{environment}/cell"
        ),
        "vram": 0,
        "ram_mb": 1024,
        "cpu": 1,
        "priority": str(args.priority),
        "ckpt_dir": str(full_absolute),
        "ckpt_glob": "checkpoint.pt",
        "result_dir": str(export_absolute),
        "local_result_dir": str(export_absolute),
        "wait_for_files": [],
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Independent MuJoCo development cells are CPU-bound and single-threaded."
        ),
        "reroute_on_node_down": True,
        "node_down_requeue_s": 600,
        "allowed_nodes": list(args.nodes),
        "require_node": None,
        "stage_excludes": [*STAGE_EXCLUDES, ".server_artifacts"],
        "stage_input_paths": [
            str((ROOT / "scripts").resolve()),
            str((ROOT / "freq_hrl").resolve()),
        ],
        "skip_launch_staging": False,
        "allow_duplicate": False,
    }


def _write_preregistration(args: argparse.Namespace) -> None:
    target = ROOT / "results" / args.run_name
    target.mkdir(parents=True, exist_ok=True)
    launcher_revision = subprocess.run(
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
        "launcher_source_revision": launcher_revision,
        "frozen_algorithm_revision": spec.FROZEN_ALGORITHM_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "environments": list(args.environments),
        "arms": {arm: spec.ARMS[arm] for arm in args.arms},
        "optimizer_seeds": list(args.optimizer_seeds),
        "train_seeds": list(spec.TRAIN_SEEDS),
        "selection_seeds": list(spec.SELECTION_SEEDS),
        "safety_selection_seeds": list(spec.SAFETY_SELECTION_SEEDS),
        "evaluation_seeds": list(spec.EVALUATION_SEEDS),
        "training_disturbance_modes": list(spec.TRAINING_DISTURBANCE_MODES),
        "evaluation_disturbance_modes": list(spec.EVALUATION_DISTURBANCE_MODES),
        "selection_contract": spec.SELECTION_CONTRACT,
        "artifact_contract": {
            "synced": list(SMALL_RESULT_FILES),
            "server_only": ["checkpoint.pt", "training_history.json"],
        },
        "scheduler_contract": {
            "scheduler": "scheduleurm",
            "allowed_nodes": list(args.nodes),
            "require_node": None,
            "cpu_per_task": 1,
            "ram_mb_per_task": 1024,
            "slurm_used": False,
        },
    }
    (target / "preregistration.json").write_text(
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
    tasks = _scheduler_tasks(args.run_name)
    expected: list[tuple[str, Path, dict[str, Any]]] = []
    for environment, arm, seed in selected_cells(args):
        signature = task_signature(args.run_name, environment, arm, seed)
        task = tasks.get(signature)
        if task is None:
            raise SystemExit(f"v17.1 sync task missing: {signature}")
        if task.get("status") != "done" or not task.get("node"):
            raise SystemExit(
                "v17.1 task is not done: "
                f"{task.get('id')} status={task.get('status')}"
            )
        path = ROOT / cell_relative_dir(args.run_name, environment, arm, seed)
        expected.append((signature, path, task))

    scheduler_dir = str(SCHEDULER.parent)
    if scheduler_dir not in sys.path:
        sys.path.insert(0, scheduler_dir)
    import scheduler as scheduler_runtime  # type: ignore  # noqa: E402

    pending = [
        item
        for item in expected
        if any(not (item[1] / name).is_file() for name in SMALL_RESULT_FILES)
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
            if any(not (item[1] / name).is_file() for name in SMALL_RESULT_FILES)
        ]
        if pending and attempt < 3:
            time.sleep(float(attempt))
    if pending:
        signature, path, _ = pending[0]
        raise SystemExit(
            f"v17.1 result sync incomplete: {len(pending)} cells; "
            f"first={signature}; path={path}; "
            f"error={errors.get(signature, 'missing required files')}"
        )
    manifest = {
        "run_name": args.run_name,
        "cell_count": len(expected),
        "artifact_contract": "small_results_only_v1",
        "nodes": {
            node: sum(str(item[2]["node"]) == node for item in expected)
            for node in sorted({str(item[2]["node"]) for item in expected})
        },
        "tasks": {
            signature: {"task_id": task["id"], "node": task["node"]}
            for signature, _, task in expected
        },
    }
    output = ROOT / "results" / args.run_name
    (output / "run_scoped_result_sync.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"synced {len(expected)} v17.1 small-result cells")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
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
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sync-only", action="store_true")
    parser.add_argument(
        "--recovery-only",
        action="store_true",
        help="resubmit failed signatures while reusing completed server cells",
    )
    parser.add_argument("--sync-workers", type=int, default=6)
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.arms = parse_csv(args.arms)
    args.nodes = parse_csv(args.nodes)
    args.environments = parse_csv(args.environments)
    try:
        args.optimizer_seeds = [
            int(value) for value in parse_csv(args.optimizer_seeds)
        ]
    except ValueError as exc:
        raise SystemExit("invalid v17.1 optimizer seed subset") from exc
    if not args.arms or not set(args.arms).issubset(spec.ARMS):
        raise SystemExit("invalid v17.1 arm subset")
    if not args.environments or not set(args.environments).issubset(
        spec.ENVIRONMENTS
    ):
        raise SystemExit("invalid v17.1 environment subset")
    if (
        not args.optimizer_seeds
        or not set(args.optimizer_seeds).issubset(spec.OPTIMIZER_SEEDS)
        or len(args.optimizer_seeds) != len(set(args.optimizer_seeds))
    ):
        raise SystemExit("invalid v17.1 optimizer seed subset")
    if set(args.nodes) - set(LINUX_CPU_NODES):
        raise SystemExit("v17.1 accepts only node001-node006")
    if not args.python_executable.strip():
        args.python_executable = default_python_executable(args.nodes)
    if int(args.sync_workers) < 1:
        raise SystemExit("v17.1 sync workers must be positive")
    return args


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    revision, manifest = source_identity(spec.FROZEN_ALGORITHM_REVISION)
    if (
        revision != spec.FROZEN_ALGORITHM_REVISION
        or manifest != spec.FROZEN_SOURCE_MANIFEST_SHA256
    ):
        raise SystemExit("v17.1 frozen algorithm identity mismatch")
    if args.sync_only:
        sync_results(args)
        return
    if not args.recovery_only:
        _write_preregistration(args)
    cells = selected_cells(args)
    print(
        f"run={args.run_name} cells={len(cells)} nodes={','.join(args.nodes)} "
        f"recovery_only={bool(args.recovery_only)}",
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
