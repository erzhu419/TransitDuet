#!/usr/bin/env python3
"""Submit source-bound MuJoCo development cells through scheduleurm."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.domains.mujoco import DISTURBANCE_MODES  # noqa: E402
from freq_hrl.experiments.mujoco import control_validation as validation  # noqa: E402
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


PILOT_OPTIMIZER_SEEDS = (35207, 35211, 35227)
PREFLIGHT_OPTIMIZER_SEED = 35233
MODULE = "freq_hrl.experiments.mujoco.control_validation"
SIGNATURE_VERSION = "mujoco-shared-core-pilot-v9"
SUBMIT_SCRIPT_PATH = Path(__file__).resolve()


def experiment_cells(
    *,
    stage: str,
    environments: list[str],
    methods: list[str],
) -> list[tuple[str, str, int]]:
    if str(stage) == "preflight":
        environment = environments[0]
        return [
            (environment, method, PREFLIGHT_OPTIMIZER_SEED)
            for method in methods
        ]
    return [
        (environment, method, int(seed))
        for environment in environments
        for method in methods
        for seed in PILOT_OPTIMIZER_SEEDS
    ]


def cell_relative_dir(
    run_name: str,
    environment: str,
    method: str,
    optimizer_seed: int,
) -> Path:
    return (
        Path("results") / str(run_name) / "cells" / str(environment)
        / str(method) / f"replicate_{int(optimizer_seed)}"
    )


def build_training_command(
    args: argparse.Namespace,
    *,
    environment: str,
    method: str,
    optimizer_seed: int,
    output_dir: Path,
) -> str:
    command = [
        str(args.python_executable), "-u", "-m", MODULE,
        "--method", str(method),
        "--env-id", str(environment),
        "--disturbance-mode", "standard",
        "--training-disturbance-modes", *args.training_disturbance_modes,
        "--evaluation-disturbance-modes", *args.evaluation_disturbance_modes,
        "--train-seeds", *(str(seed) for seed in args.train_seeds),
        "--selection-seeds", *(str(seed) for seed in args.selection_seeds),
        "--safety-selection-seeds",
        *(str(seed) for seed in args.safety_selection_seeds),
        "--eval-seeds", *(str(seed) for seed in args.eval_seeds),
        "--steps", str(args.steps),
        "--episode-horizon", str(args.episode_horizon),
        "--iterations", str(args.iterations),
        "--optimizer-seed", str(optimizer_seed),
        "--upper-period", str(args.upper_period),
        "--hidden-dim", str(args.hidden_dim),
        "--learning-rate", str(args.learning_rate),
        "--lower-lf-rms-budget", str(args.lower_lf_rms_budget),
        "--upper-action-scale", str(args.upper_action_scale),
        "--lower-action-scale", str(args.lower_action_scale),
        "--lower-constraint-update-mode",
        str(args.lower_constraint_update_mode),
        "--checkpoint-smoothing-window", str(args.checkpoint_smoothing_window),
        "--checkpoint-min-delta", str(args.checkpoint_min_delta),
        "--checkpoint-evaluation-interval",
        str(args.checkpoint_evaluation_interval),
        "--code-revision", str(args.code_revision),
        "--source-manifest-sha256", str(args.source_manifest_sha256),
        "--output-dir", str(output_dir),
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
    run = " ".join([*environment_variables, shlex.join(command)]) + " && echo DONE"
    return f"cd .. && {run}" if args.launch_subdir == "scripts" else run


def build_scheduler_spec(
    args: argparse.Namespace,
    *,
    environment: str,
    method: str,
    optimizer_seed: int,
) -> dict[str, object]:
    relative = cell_relative_dir(
        args.run_name, environment, method, optimizer_seed
    )
    absolute = ROOT / relative
    return {
        "project": str(args.project),
        "description": (
            f"Freq-HRL MuJoCo {args.stage} {environment} {method} "
            f"replicate {optimizer_seed}"
        ),
        "cmd": build_training_command(
            args,
            environment=environment,
            method=method,
            optimizer_seed=optimizer_seed,
            output_dir=relative,
        ),
        "cwd": str(ROOT / str(args.launch_subdir)),
        "signature": (
            f"Freq-HRL/{SIGNATURE_VERSION}/{args.stage}/{args.run_name}/"
            f"{environment}/{method}/rep-{optimizer_seed}"
        ),
        "resource_family": f"Freq-HRL/{SIGNATURE_VERSION}/{method}/cell",
        "vram": 0,
        "ram_mb": int(args.ram_mb),
        "cpu": 1,
        "priority": str(args.priority),
        "ckpt_dir": str(absolute),
        "ckpt_glob": "checkpoint.pt",
        "result_dir": str(absolute),
        "local_result_dir": str(absolute),
        "wait_for_files": list(args.wait_for_files),
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Independent headless MuJoCo PPO cells use one physical CPU core."
        ),
        "reroute_on_node_down": True,
        "node_down_requeue_s": 600,
        "allowed_nodes": list(args.nodes),
        "require_node": None,
        "stage_excludes": list(STAGE_EXCLUDES),
        "stage_input_paths": list(args.stage_input_paths),
        "skip_launch_staging": bool(args.skip_launch_staging),
        "allow_duplicate": bool(args.allow_duplicate),
    }


def expected_cell_dirs(args: argparse.Namespace) -> list[Path]:
    return [
        ROOT / cell_relative_dir(args.run_name, *cell)
        for cell in experiment_cells(
            stage=args.stage,
            environments=args.environments,
            methods=args.methods,
        )
    ]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def merge_results(args: argparse.Namespace) -> None:
    directories = expected_cell_dirs(args)
    missing = [
        path for path in directories
        if not (path / "cell_summary.json").exists()
        or not (path / "training_history.json").exists()
        or not (path / "evaluation_rows.csv").exists()
    ]
    if missing:
        raise SystemExit(
            f"cannot merge MuJoCo pilot: {len(missing)} cells missing; "
            f"first={missing[0]}"
        )
    summaries = [
        json.loads((path / "cell_summary.json").read_text(encoding="utf-8"))
        for path in directories
    ]
    rows = [row for path in directories for row in _read_rows(
        path / "evaluation_rows.csv"
    )]
    if {
        str(summary["protocol_version"]) for summary in summaries
    } != {validation.MUJOCO_CONTROL_PROTOCOL_VERSION}:
        raise SystemExit("MuJoCo pilot protocol versions are mixed")
    source_pairs = {
        (
            str(summary["code_revision"]),
            str(summary["source_manifest_sha256"]),
            str(summary["source_identity_status"]),
        )
        for summary in summaries
    }
    if len(source_pairs) != 1 or next(iter(source_pairs))[2] != "verified":
        raise SystemExit("MuJoCo pilot source identity is not uniform and verified")
    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (
            str(row["environment"]),
            str(row["method"]),
            str(row["disturbance_mode"]),
        )
        groups.setdefault(key, []).append(float(row["episode_return"]))
    leaderboard = [
        {
            "environment": key[0],
            "method": key[1],
            "disturbance_mode": key[2],
            "path_row_count": len(values),
            "episode_return_mean": float(sum(values) / len(values)),
        }
        for key, values in sorted(groups.items())
    ]
    output = ROOT / "results" / args.run_name / "merged"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "leaderboard.csv", leaderboard)
    (output / "summary.json").write_text(json.dumps({
        "status": "development_pilot_only",
        "stage": args.stage,
        "cell_count": len(summaries),
        "evaluation_row_count": len(rows),
        "code_revision": next(iter(source_pairs))[0],
        "source_manifest_sha256": next(iter(source_pairs))[1],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"merged {len(summaries)} MuJoCo pilot cells into {output}")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0]) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", choices=("preflight", "pilot"), default="preflight")
    parser.add_argument("--environments", default=",".join(validation.DEFAULT_ENV_IDS))
    parser.add_argument("--methods", default=",".join(validation.METHODS))
    parser.add_argument(
        "--training-disturbance-modes",
        default=",".join(validation.DEFAULT_TRAINING_DISTURBANCE_MODES),
    )
    parser.add_argument("--evaluation-disturbance-modes", default=",".join(DISTURBANCE_MODES))
    parser.add_argument("--train-seeds", default=",".join(map(str, validation.DEFAULT_TRAIN_SEEDS)))
    parser.add_argument("--selection-seeds", default=",".join(map(str, validation.DEFAULT_SELECTION_SEEDS)))
    parser.add_argument(
        "--safety-selection-seeds",
        default=",".join(map(str, validation.DEFAULT_SAFETY_SELECTION_SEEDS)),
    )
    parser.add_argument("--eval-seeds", default=",".join(map(str, validation.DEFAULT_EVAL_SEEDS)))
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--episode-horizon", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--upper-period", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lower-lf-rms-budget", type=float, default=0.05)
    parser.add_argument("--upper-action-scale", type=float, default=1.0)
    parser.add_argument("--lower-action-scale", type=float, default=1.0)
    parser.add_argument(
        "--lower-constraint-update-mode",
        choices=(
            "scalarized",
            "reward_guarded_projection",
            "reward_guarded_adam_projection",
        ),
        default="reward_guarded_adam_projection",
    )
    parser.add_argument("--checkpoint-smoothing-window", type=int, default=8)
    parser.add_argument("--checkpoint-min-delta", type=float, default=1e-3)
    parser.add_argument("--checkpoint-evaluation-interval", type=int, default=4)
    parser.add_argument("--nodes", default=",".join(LINUX_CPU_NODES))
    parser.add_argument("--python-executable", default="")
    parser.add_argument("--launch-subdir", choices=(".", "scripts"), default=".")
    parser.add_argument("--source-code-revision", default="")
    parser.add_argument("--project", default="Freq-HRL-MuJoCo-Pilot")
    parser.add_argument("--ram-mb", type=int, default=2048)
    parser.add_argument("--priority", choices=("low", "normal", "high"), default="normal")
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--skip-complete-cells", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-duplicate", action="store_true")
    parser.add_argument("--skip-launch-staging", action="store_true")
    parser.add_argument("--stage-input-path", action="append", default=[])
    parser.add_argument("--wait-for-file", action="append", default=[])
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.environments = parse_csv(args.environments)
    args.methods = parse_csv(args.methods)
    args.training_disturbance_modes = parse_csv(
        args.training_disturbance_modes
    )
    args.evaluation_disturbance_modes = parse_csv(
        args.evaluation_disturbance_modes
    )
    args.train_seeds = parse_csv(args.train_seeds, int)
    args.selection_seeds = parse_csv(args.selection_seeds, int)
    args.safety_selection_seeds = parse_csv(args.safety_selection_seeds, int)
    args.eval_seeds = parse_csv(args.eval_seeds, int)
    args.nodes = parse_csv(args.nodes)
    args.stage_input_paths = [
        str(Path(path).expanduser().resolve())
        for path in args.stage_input_path if str(path).strip()
    ]
    args.wait_for_files = [
        str(Path(path).expanduser().resolve())
        for path in args.wait_for_file if str(path).strip()
    ]
    if not args.environments or not set(args.environments).issubset(
        validation.DEFAULT_ENV_IDS
    ):
        raise SystemExit("invalid MuJoCo environment registry")
    if not args.methods or not set(args.methods).issubset(validation.METHODS):
        raise SystemExit("invalid MuJoCo method registry")
    if (
        not args.training_disturbance_modes
        or not set(args.training_disturbance_modes).issubset(DISTURBANCE_MODES)
        or not args.evaluation_disturbance_modes
        or not set(args.evaluation_disturbance_modes).issubset(DISTURBANCE_MODES)
    ):
        raise SystemExit("invalid MuJoCo disturbance registry")
    unknown_nodes = sorted(set(args.nodes) - set(LINUX_CPU_NODES))
    if unknown_nodes:
        raise SystemExit(f"invalid MuJoCo nodes: {unknown_nodes}")
    if args.stage == "preflight":
        args.steps = min(int(args.steps), 64)
        args.episode_horizon = min(int(args.episode_horizon), 64)
        args.iterations = min(int(args.iterations), 2)
        args.training_disturbance_modes = ["standard"]
        args.evaluation_disturbance_modes = ["standard"]
        args.train_seeds = args.train_seeds[:1]
        args.selection_seeds = args.selection_seeds[:1]
        args.safety_selection_seeds = args.safety_selection_seeds[:1]
        args.eval_seeds = args.eval_seeds[:1]
        args.checkpoint_smoothing_window = 1
        args.checkpoint_min_delta = 0.0
        args.checkpoint_evaluation_interval = 1
    if not args.python_executable.strip():
        args.python_executable = default_python_executable(args.nodes)
    if float(args.lower_lf_rms_budget) <= 0.0:
        raise SystemExit("lower LF RMS budget must be positive")
    if not 0.0 <= float(args.upper_action_scale) <= 1.0:
        raise SystemExit("upper action scale must be in [0, 1]")
    if not 0.0 < float(args.lower_action_scale) <= 1.0:
        raise SystemExit("lower action scale must be in (0, 1]")
    return args


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    if args.merge_only:
        merge_results(args)
        return
    try:
        args.code_revision, args.source_manifest_sha256 = source_identity(
            args.source_code_revision
        )
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"cannot bind MuJoCo pilot source: {exc}") from exc
    cells = experiment_cells(
        stage=args.stage,
        environments=args.environments,
        methods=args.methods,
    )
    if args.skip_complete_cells:
        cells = [
            cell for cell in cells
            if not (
                ROOT / cell_relative_dir(args.run_name, *cell)
                / "cell_summary.json"
            ).exists()
        ]
    if args.max_cells > 0:
        cells = cells[:args.max_cells]
    if not cells:
        print("no MuJoCo pilot cells require submission")
        return
    print(
        f"run={args.run_name} stage={args.stage} cells={len(cells)} "
        f"nodes={','.join(args.nodes)}",
        flush=True,
    )
    execute_bulk([
        build_scheduler_spec(
            args,
            environment=environment,
            method=method,
            optimizer_seed=seed,
        )
        for environment, method, seed in cells
    ], dry_run=bool(args.dry_run), intent_label=(
        f"Freq-HRL MuJoCo {args.stage} {args.run_name}"
    ))
    if args.dispatch and not args.dry_run:
        execute([sys.executable, str(SCHEDULER), "dispatch"], dry_run=False)
    print("merge after result sync: " + shlex.join([
        sys.executable,
        str(SUBMIT_SCRIPT_PATH),
        "--run-name", args.run_name,
        "--stage", args.stage,
        "--merge-only",
    ]))


if __name__ == "__main__":
    main()
