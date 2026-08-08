#!/usr/bin/env python3
"""Submit the frozen MuJoCo v12 confirmatory matrix through scheduleurm."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import mujoco_v12_confirmatory_spec as spec  # noqa: E402
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


RUNTIME_PATH = Path(__file__).with_name("mujoco_v12_confirmatory_runtime.py")
SPEC_PATH = Path(spec.__file__).resolve()
SIGNATURE_VERSION = "mujoco-v12-full-method-confirmatory-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _git_runtime_identity() -> dict[str, str]:
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
    tracked_paths = (Path(__file__).resolve(), RUNTIME_PATH.resolve(), SPEC_PATH)
    relative_paths = [str(path.relative_to(git_root)) for path in tracked_paths]
    subprocess.run(
        ["git", "-C", str(git_root), "ls-files", "--error-unmatch", *relative_paths],
        check=True,
        capture_output=True,
        text=True,
    )
    clean = subprocess.run(
        ["git", "-C", str(git_root), "diff", "--quiet", "HEAD", "--", *relative_paths]
    )
    if clean.returncode != 0:
        raise RuntimeError("confirmatory runtime files do not match HEAD")
    return {
        "runtime_revision": revision,
        "launcher_sha256": _sha256(Path(__file__)),
        "runtime_sha256": _sha256(RUNTIME_PATH),
        "spec_sha256": _sha256(SPEC_PATH),
    }


def experiment_cells() -> list[tuple[str, int]]:
    return [
        (environment, int(seed))
        for environment in spec.ENVIRONMENTS
        for seed in spec.OPTIMIZER_SEEDS
    ]


def cell_relative_dir(
    run_name: str,
    *,
    arm: str,
    environment: str,
    optimizer_seed: int,
) -> Path:
    method = str(spec.ARMS[arm]["method"])
    return (
        Path("results") / str(run_name) / "cells" / str(environment)
        / method / f"replicate_{int(optimizer_seed)}"
    )


def build_training_command(
    args: argparse.Namespace,
    *,
    environment: str,
    optimizer_seed: int,
    output_dir: Path,
) -> str:
    arm_spec = spec.ARMS[args.arm]
    command = [
        str(args.python_executable), "-u", str(RUNTIME_PATH.relative_to(ROOT)),
        "--confirmatory-arm", str(args.arm),
        "--confirmatory-runtime-revision", str(args.runtime_revision),
        "--confirmatory-launcher-sha256", str(args.launcher_sha256),
        "--confirmatory-runtime-sha256", str(args.runtime_sha256),
        "--confirmatory-spec-sha256", str(args.spec_sha256),
        "--method", str(arm_spec["method"]),
        "--env-id", str(environment),
        "--disturbance-mode", "standard",
        "--training-disturbance-modes", *spec.TRAINING_DISTURBANCE_MODES,
        "--evaluation-disturbance-modes", *spec.EVALUATION_DISTURBANCE_MODES,
        "--train-seeds", *(str(seed) for seed in spec.TRAIN_SEEDS),
        "--selection-seeds",
        *(str(seed) for seed in spec.CHECKPOINT_SELECTION_SEEDS),
        "--safety-selection-seeds",
        *(str(seed) for seed in spec.SAFETY_SELECTION_SEEDS),
        "--eval-seeds", *(str(seed) for seed in spec.HELDOUT_EVALUATION_SEEDS),
        "--steps", str(spec.STEPS),
        "--episode-horizon", str(spec.EPISODE_HORIZON),
        "--iterations", str(spec.ITERATIONS),
        "--optimizer-seed", str(optimizer_seed),
        "--upper-period", str(spec.UPPER_PERIOD),
        "--hidden-dim", str(spec.HIDDEN_DIM),
        "--learning-rate", str(spec.LEARNING_RATE),
        "--lower-lf-rms-budget", str(spec.LOWER_LF_RMS_BUDGET),
        "--upper-action-scale", str(spec.UPPER_ACTION_SCALE),
        "--lower-action-scale", str(spec.LOWER_ACTION_SCALE),
        "--responsibility-mode", str(arm_spec["responsibility_mode"]),
        "--lower-constraint-update-mode", spec.LOWER_CONSTRAINT_UPDATE_MODE,
        "--checkpoint-smoothing-window", str(spec.CHECKPOINT_SMOOTHING_WINDOW),
        "--checkpoint-min-delta", str(spec.CHECKPOINT_MIN_DELTA),
        "--checkpoint-evaluation-interval",
        str(spec.CHECKPOINT_EVALUATION_INTERVAL),
        "--code-revision", spec.FROZEN_ALGORITHM_REVISION,
        "--source-manifest-sha256", spec.FROZEN_SOURCE_MANIFEST_SHA256,
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
    return " ".join([*environment_variables, shlex.join(command)]) + " && echo DONE"


def build_scheduler_spec(
    args: argparse.Namespace,
    *,
    environment: str,
    optimizer_seed: int,
) -> dict[str, object]:
    relative = cell_relative_dir(
        args.run_name,
        arm=args.arm,
        environment=environment,
        optimizer_seed=optimizer_seed,
    )
    absolute = ROOT / relative
    method = str(spec.ARMS[args.arm]["method"])
    return {
        "project": "Freq-HRL-MuJoCo-Confirmatory",
        "description": (
            f"Freq-HRL MuJoCo v12 {args.arm} {environment} replicate "
            f"{optimizer_seed}"
        ),
        "cmd": build_training_command(
            args,
            environment=environment,
            optimizer_seed=optimizer_seed,
            output_dir=relative,
        ),
        "cwd": str(ROOT),
        "signature": (
            f"Freq-HRL/{SIGNATURE_VERSION}/{args.run_name}/{args.arm}/"
            f"{environment}/{method}/rep-{optimizer_seed}"
        ),
        "resource_family": (
            f"Freq-HRL/{SIGNATURE_VERSION}/{args.arm}/{environment}/cell"
        ),
        "vram": 0,
        "ram_mb": 2048,
        "cpu": 1,
        "priority": str(args.priority),
        "ckpt_dir": str(absolute),
        "ckpt_glob": "checkpoint.pt",
        "result_dir": str(absolute),
        "local_result_dir": str(absolute),
        "wait_for_files": [],
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Independent frozen MuJoCo PPO replicates use one physical core."
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


def _expected_dirs(args: argparse.Namespace) -> list[Path]:
    return [
        ROOT / cell_relative_dir(
            args.run_name,
            arm=args.arm,
            environment=environment,
            optimizer_seed=seed,
        )
        for environment, seed in experiment_cells()
    ]


def _write_preregistration(args: argparse.Namespace) -> None:
    output = ROOT / "results" / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    path = output / "preregistration.json"
    payload: dict[str, Any] = {
        "confirmatory_protocol_version": spec.CONFIRMATORY_PROTOCOL_VERSION,
        "status": "frozen_before_heldout_access",
        "arm": args.arm,
        "arm_spec": spec.ARMS[args.arm],
        "environments": list(spec.ENVIRONMENTS),
        "optimizer_seeds": list(spec.OPTIMIZER_SEEDS),
        "train_seeds": list(spec.TRAIN_SEEDS),
        "checkpoint_selection_seeds": list(spec.CHECKPOINT_SELECTION_SEEDS),
        "safety_selection_seeds": list(spec.SAFETY_SELECTION_SEEDS),
        "heldout_evaluation_seeds": list(spec.HELDOUT_EVALUATION_SEEDS),
        "training_disturbance_modes": list(spec.TRAINING_DISTURBANCE_MODES),
        "evaluation_disturbance_modes": list(spec.EVALUATION_DISTURBANCE_MODES),
        "frozen_algorithm_revision": spec.FROZEN_ALGORITHM_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "runtime_revision": args.runtime_revision,
        "launcher_sha256": args.launcher_sha256,
        "runtime_sha256": args.runtime_sha256,
        "spec_sha256": args.spec_sha256,
        "primary_gates": {
            "return_noninferiority_margin_fraction": (
                spec.RETURN_NONINFERIORITY_MARGIN_FRACTION
            ),
            "minimum_drift_reduction_fraction": (
                spec.MINIMUM_DRIFT_REDUCTION_FRACTION
            ),
            "family_wise_alpha": spec.FAMILY_WISE_ALPHA,
            "primary_gate_count": spec.PRIMARY_GATE_COUNT,
            "per_gate_one_sided_confidence": (
                spec.PER_GATE_ONE_SIDED_CONFIDENCE
            ),
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise RuntimeError("existing confirmatory preregistration differs")
    path.write_text(rendered, encoding="utf-8")


def merge_results(args: argparse.Namespace) -> None:
    directories = _expected_dirs(args)
    required = ("cell_summary.json", "training_history.json", "evaluation_rows.csv", "checkpoint.pt")
    missing = [
        path for path in directories
        if any(not (path / name).is_file() for name in required)
    ]
    if missing:
        raise SystemExit(
            f"cannot merge MuJoCo v12 confirmatory arm: {len(missing)} cells "
            f"missing; first={missing[0]}"
        )
    summaries = [
        json.loads((path / "cell_summary.json").read_text(encoding="utf-8"))
        for path in directories
    ]
    arm_spec = spec.ARMS[args.arm]
    issues = []
    runtime_identities = set()
    for summary in summaries:
        if summary.get("confirmatory_protocol_version") != spec.CONFIRMATORY_PROTOCOL_VERSION:
            issues.append("confirmatory_protocol_version_mismatch")
        if summary.get("confirmatory_arm") != args.arm:
            issues.append("confirmatory_arm_mismatch")
        if summary.get("method") != arm_spec["method"]:
            issues.append("method_mismatch")
        if summary.get("responsibility_mode") != arm_spec["responsibility_mode"]:
            issues.append("responsibility_mode_mismatch")
        if summary.get("code_revision") != spec.FROZEN_ALGORITHM_REVISION:
            issues.append("algorithm_revision_mismatch")
        if summary.get("source_manifest_sha256") != spec.FROZEN_SOURCE_MANIFEST_SHA256:
            issues.append("source_manifest_mismatch")
        runtime_identities.add((
            str(summary.get("confirmatory_runtime_revision", "")),
            str(summary.get("confirmatory_launcher_sha256", "")),
            str(summary.get("confirmatory_runtime_sha256", "")),
            str(summary.get("confirmatory_spec_sha256", "")),
        ))
    if len(runtime_identities) != 1:
        issues.append("mixed_runtime_identity")
    if issues:
        raise SystemExit("confirmatory merge failed: " + ",".join(sorted(set(issues))))
    output = ROOT / "results" / args.run_name / "merged"
    output.mkdir(parents=True, exist_ok=True)
    (output / "arm_manifest.json").write_text(json.dumps({
        "status": "confirmatory_complete_unanalyzed",
        "evidence_role": "fresh_seed_confirmatory_unanalyzed",
        "confirmatory_protocol_version": spec.CONFIRMATORY_PROTOCOL_VERSION,
        "arm": args.arm,
        "arm_spec": arm_spec,
        "cell_count": len(summaries),
        "evaluation_rows_per_cell": (
            len(spec.HELDOUT_EVALUATION_SEEDS)
            * len(spec.EVALUATION_DISTURBANCE_MODES)
        ),
        "runtime_identity": list(next(iter(runtime_identities))),
        "frozen_algorithm_revision": spec.FROZEN_ALGORITHM_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"merged {len(summaries)} frozen MuJoCo v12 cells into {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--arm", choices=tuple(spec.ARMS), required=True)
    parser.add_argument("--nodes", default=",".join(LINUX_CPU_NODES))
    parser.add_argument("--python-executable", default="")
    parser.add_argument("--priority", choices=("low", "normal", "high"), default="normal")
    parser.add_argument("--skip-complete-cells", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.nodes = parse_csv(args.nodes)
    unknown = sorted(set(args.nodes) - set(LINUX_CPU_NODES))
    if unknown or not args.nodes:
        raise SystemExit(f"invalid confirmatory Linux node pool: {unknown}")
    if not args.python_executable.strip():
        args.python_executable = default_python_executable(args.nodes)
    return args


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    if args.merge_only:
        merge_results(args)
        return
    runtime = _git_runtime_identity()
    for key, value in runtime.items():
        setattr(args, key, value)
    revision, manifest = source_identity(spec.FROZEN_ALGORITHM_REVISION)
    if (
        revision != spec.FROZEN_ALGORITHM_REVISION
        or manifest != spec.FROZEN_SOURCE_MANIFEST_SHA256
    ):
        raise SystemExit("frozen MuJoCo v12 algorithm identity mismatch")
    _write_preregistration(args)
    cells = experiment_cells()
    if args.skip_complete_cells:
        cells = [
            cell for cell in cells
            if not (
                ROOT / cell_relative_dir(
                    args.run_name,
                    arm=args.arm,
                    environment=cell[0],
                    optimizer_seed=cell[1],
                ) / "cell_summary.json"
            ).is_file()
        ]
    if not cells:
        print("no frozen MuJoCo v12 cells require submission")
        return
    print(
        f"run={args.run_name} arm={args.arm} cells={len(cells)} "
        f"nodes={','.join(args.nodes)} runtime={args.runtime_revision}",
        flush=True,
    )
    execute_bulk([
        build_scheduler_spec(
            args,
            environment=environment,
            optimizer_seed=seed,
        )
        for environment, seed in cells
    ], dry_run=bool(args.dry_run), intent_label=(
        f"Freq-HRL MuJoCo v12 confirmatory {args.arm}"
    ))
    if args.dispatch and not args.dry_run:
        execute([sys.executable, str(SCHEDULER), "dispatch"], dry_run=False)


if __name__ == "__main__":
    main()
