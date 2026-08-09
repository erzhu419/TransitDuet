#!/usr/bin/env python3
"""Submit the frozen MuJoCo v14.1 development screen via scheduleurm."""

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

from scripts import mujoco_v14_1_upper_pd_screen_spec as spec  # noqa: E402
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
SIGNATURE_VERSION = "mujoco-v14-1-upper-pd-screen-v1"
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
        raise RuntimeError("v14.1 screen launcher/spec do not match HEAD")
    return {
        "runtime_revision": revision,
        "launcher_sha256": _sha256(LAUNCHER_PATH),
        "spec_sha256": _sha256(SPEC_PATH),
    }


def experiment_cells(
    arms: tuple[str, ...] | list[str] = tuple(spec.ARMS),
) -> list[tuple[str, str, int]]:
    return [
        (environment, arm, int(seed))
        for environment in spec.ENVIRONMENTS
        for arm in arms
        for seed in spec.OPTIMIZER_SEEDS
    ]


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


def build_training_command(
    args: argparse.Namespace,
    *,
    environment: str,
    arm: str,
    optimizer_seed: int,
    output_dir: Path,
) -> str:
    arm_spec = spec.ARMS[str(arm)]
    command = [
        str(args.python_executable), "-u", "-m", MODULE,
        "--method", str(arm_spec["method"]),
        "--env-id", str(environment),
        "--disturbance-mode", "standard",
        "--training-disturbance-modes", *spec.TRAINING_DISTURBANCE_MODES,
        "--evaluation-disturbance-modes", *spec.EVALUATION_DISTURBANCE_MODES,
        "--train-seeds", *(str(seed) for seed in spec.TRAIN_SEEDS),
        "--selection-seeds",
        *(str(seed) for seed in spec.CHECKPOINT_SELECTION_SEEDS),
        "--eval-seeds",
        *(str(seed) for seed in spec.DEVELOPMENT_EVALUATION_SEEDS),
        "--steps", str(spec.STEPS),
        "--episode-horizon", str(spec.EPISODE_HORIZON),
        "--iterations", str(spec.ITERATIONS),
        "--optimizer-seed", str(optimizer_seed),
        "--upper-period", str(spec.UPPER_PERIOD),
        "--hidden-dim", str(spec.HIDDEN_DIM),
        "--learning-rate", str(spec.LEARNING_RATE),
        "--lower-lf-rms-budget", str(spec.LOWER_LF_RMS_BUDGET),
        "--upper-hf-rms-budget", str(spec.UPPER_HF_RMS_BUDGET),
        "--upper-hf-penalty-coef", str(arm_spec["upper_hf_penalty_coef"]),
        "--upper-constraint-mode", str(arm_spec["upper_constraint_mode"]),
        "--upper-dual-lr", str(arm_spec["upper_dual_lr"]),
        "--upper-constraint-update-mode", spec.UPPER_CONSTRAINT_UPDATE_MODE,
        "--upper-action-scale", str(spec.UPPER_ACTION_SCALE),
        "--lower-action-scale", str(spec.LOWER_ACTION_SCALE),
        "--responsibility-mode", str(arm_spec["responsibility_mode"]),
        "--leakage-constraint-scope",
        str(arm_spec["leakage_constraint_scope"]),
        "--lower-constraint-update-mode", spec.LOWER_CONSTRAINT_UPDATE_MODE,
        "--checkpoint-selection-mode", spec.CHECKPOINT_SELECTION_MODE,
        "--checkpoint-score-mode", spec.CHECKPOINT_SCORE_MODE,
        "--checkpoint-constraint-penalty",
        str(spec.CHECKPOINT_CONSTRAINT_PENALTY),
        "--checkpoint-smoothing-window",
        str(spec.CHECKPOINT_SMOOTHING_WINDOW),
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
    arm: str,
    optimizer_seed: int,
) -> dict[str, object]:
    relative = cell_relative_dir(
        args.run_name,
        environment=environment,
        arm=arm,
        optimizer_seed=optimizer_seed,
    )
    absolute = ROOT / relative
    return {
        "project": "Freq-HRL-MuJoCo-v14.1-Upper-PD-Screen",
        "description": (
            f"Freq-HRL MuJoCo v14.1 {environment} {arm} "
            f"replicate {optimizer_seed}"
        ),
        "cmd": build_training_command(
            args,
            environment=environment,
            arm=arm,
            optimizer_seed=optimizer_seed,
            output_dir=relative,
        ),
        "cwd": str(ROOT),
        "signature": (
            f"Freq-HRL/{SIGNATURE_VERSION}/{args.run_name}/{environment}/"
            f"{arm}/rep-{optimizer_seed}"
        ),
        "resource_family": (
            f"Freq-HRL/{SIGNATURE_VERSION}/{arm}/{environment}/cell"
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


def _expected_dirs(args: argparse.Namespace) -> list[tuple[str, str, int, Path]]:
    return [
        (
            environment,
            arm,
            seed,
            ROOT / cell_relative_dir(
                args.run_name,
                environment=environment,
                arm=arm,
                optimizer_seed=seed,
            ),
        )
        for environment, arm, seed in experiment_cells(tuple(spec.ARMS))
    ]


def _write_preregistration(args: argparse.Namespace) -> None:
    output = ROOT / "results" / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    path = output / "preregistration.json"
    payload: dict[str, Any] = {
        "status": "frozen_before_v14_1_development_outcome_access",
        "evidence_role": "development_screen_not_confirmatory",
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "development_disclosure": spec.DEVELOPMENT_DISCLOSURE,
        "arms": spec.ARMS,
        "environments": list(spec.ENVIRONMENTS),
        "optimizer_seeds": list(spec.OPTIMIZER_SEEDS),
        "train_seeds": list(spec.TRAIN_SEEDS),
        "checkpoint_selection_seeds": list(spec.CHECKPOINT_SELECTION_SEEDS),
        "development_evaluation_seeds": list(
            spec.DEVELOPMENT_EVALUATION_SEEDS
        ),
        "training_disturbance_modes": list(spec.TRAINING_DISTURBANCE_MODES),
        "evaluation_disturbance_modes": list(spec.EVALUATION_DISTURBANCE_MODES),
        "checkpoint_selection_mode": spec.CHECKPOINT_SELECTION_MODE,
        "checkpoint_score_mode": spec.CHECKPOINT_SCORE_MODE,
        "checkpoint_constraint_penalty": (
            spec.CHECKPOINT_CONSTRAINT_PENALTY
        ),
        "frozen_algorithm_revision": spec.FROZEN_ALGORITHM_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "runtime_revision": args.runtime_revision,
        "launcher_sha256": args.launcher_sha256,
        "spec_sha256": args.spec_sha256,
        "development_selection_gates": {
            "return_noninferiority_margin_fraction": (
                spec.RETURN_NONINFERIORITY_MARGIN_FRACTION
            ),
            "minimum_responsibility_drift_reduction_fraction": (
                spec.MINIMUM_RESPONSIBILITY_DRIFT_REDUCTION_FRACTION
            ),
            "minimum_raw_lower_drift_reduction_fraction": (
                spec.MINIMUM_RAW_LOWER_DRIFT_REDUCTION_FRACTION
            ),
            "maximum_upper_hf_rms": spec.UPPER_HF_REPORTING_GATE,
            "selection_confidence": spec.SELECTION_CONFIDENCE,
            "bootstrap_draws": spec.BOOTSTRAP_DRAWS,
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != rendered:
        raise RuntimeError("existing v14.1 preregistration differs")
    path.write_text(rendered, encoding="utf-8")


def merge_results(args: argparse.Namespace) -> None:
    required = (
        "cell_summary.json",
        "training_history.json",
        "evaluation_rows.csv",
        "checkpoint.pt",
    )
    expected = _expected_dirs(args)
    missing = [
        path for _, _, _, path in expected
        if any(not (path / name).is_file() for name in required)
    ]
    if missing:
        raise SystemExit(
            f"cannot merge v14.1 screen: {len(missing)} cells missing; "
            f"first={missing[0]}"
        )
    issues: list[str] = []
    cells = []
    expected_paths = (
        len(spec.CHECKPOINT_SELECTION_SEEDS)
        * len(spec.TRAINING_DISTURBANCE_MODES)
    )
    expected_rows = (
        len(spec.DEVELOPMENT_EVALUATION_SEEDS)
        * len(spec.EVALUATION_DISTURBANCE_MODES)
    )
    for environment, arm, seed, path in expected:
        summary = json.loads(
            (path / "cell_summary.json").read_text(encoding="utf-8")
        )
        arm_spec = spec.ARMS[arm]
        expected_scope = (
            str(arm_spec["leakage_constraint_scope"])
            if str(arm_spec["method"]) == "freq_hrl" else "disabled"
        )
        checks = {
            "protocol": summary.get("protocol_version")
            == spec.FROZEN_CORE_PROTOCOL_VERSION,
            "environment": summary.get("environment") == environment,
            "method": summary.get("method") == arm_spec["method"],
            "responsibility": summary.get("responsibility_mode")
            == arm_spec["responsibility_mode"],
            "scope": summary.get("leakage_constraint_scope") == expected_scope,
            "upper_mode": summary.get("upper_constraint_mode")
            == arm_spec["upper_constraint_mode"],
            "upper_coef": float(summary.get(
                "upper_hf_penalty_coef", float("nan")
            )) == float(arm_spec["upper_hf_penalty_coef"]),
            "upper_dual": float(summary.get(
                "upper_dual_lr", float("nan")
            )) == float(arm_spec["upper_dual_lr"]),
            "checkpoint_selection": summary.get("checkpoint_selection_mode")
            == spec.CHECKPOINT_SELECTION_MODE,
            "checkpoint_score": summary.get("checkpoint_score_mode")
            == spec.CHECKPOINT_SCORE_MODE,
            "checkpoint_paths": int(summary.get(
                "checkpoint_selection_path_count", -1
            )) == expected_paths,
            "heldout_once": int(summary.get(
                "heldout_evaluation_pass_count", -1
            )) == 1,
            "evaluation_rows": int(summary.get(
                "evaluation_row_count", -1
            )) == expected_rows,
            "revision": summary.get("code_revision")
            == spec.FROZEN_ALGORITHM_REVISION,
            "manifest": summary.get("source_manifest_sha256")
            == spec.FROZEN_SOURCE_MANIFEST_SHA256,
            "source_verified": summary.get("source_identity_status")
            == "verified",
        }
        issues.extend(
            f"{arm}/{environment}/{seed}:{name}"
            for name, passed in checks.items() if not passed
        )
        cells.append({
            "environment": environment,
            "arm": arm,
            "optimizer_seed": seed,
            "relative_dir": str(path.relative_to(ROOT)),
            "frozen_parameter_sha256": summary.get(
                "frozen_parameter_sha256"
            ),
            "checkpoint_file_sha256": summary.get("checkpoint_file_sha256"),
        })
    if issues:
        raise SystemExit("v14.1 screen merge failed: " + ",".join(issues[:20]))
    output = ROOT / "results" / args.run_name / "merged"
    output.mkdir(parents=True, exist_ok=True)
    (output / "cell_manifest.json").write_text(json.dumps({
        "status": "development_screen_complete_unanalyzed",
        "evidence_role": "development_screen_not_confirmatory",
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "cell_count": len(cells),
        "cells": cells,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"merged {len(cells)} frozen MuJoCo v14.1 development cells")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--arms", default=",".join(spec.ARMS))
    parser.add_argument("--nodes", default=",".join(LINUX_CPU_NODES))
    parser.add_argument("--python-executable", default="")
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="normal"
    )
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--skip-complete-cells", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.arms = parse_csv(args.arms)
    args.nodes = parse_csv(args.nodes)
    if not args.arms or not set(args.arms).issubset(spec.ARMS):
        raise SystemExit("invalid v14.1 screen arm registry")
    unknown_nodes = sorted(set(args.nodes) - set(LINUX_CPU_NODES))
    if unknown_nodes:
        raise SystemExit(f"invalid v14.1 screen nodes: {unknown_nodes}")
    if not args.python_executable.strip():
        args.python_executable = default_python_executable(args.nodes)
    return args


def main() -> None:
    args = normalize_args(build_parser().parse_args())
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
        raise SystemExit("v14.1 frozen algorithm identity mismatch")
    _write_preregistration(args)
    cells = experiment_cells(args.arms)
    if args.skip_complete_cells:
        cells = [
            cell for cell in cells
            if not (
                ROOT / cell_relative_dir(
                    args.run_name,
                    environment=cell[0],
                    arm=cell[1],
                    optimizer_seed=cell[2],
                ) / "cell_summary.json"
            ).is_file()
        ]
    if int(args.max_cells) > 0:
        cells = cells[:int(args.max_cells)]
    if not cells:
        print("no v14.1 screen cells require submission")
        return
    print(
        f"run={args.run_name} cells={len(cells)} "
        f"nodes={','.join(args.nodes)}",
        flush=True,
    )
    execute_bulk([
        build_scheduler_spec(
            args,
            environment=environment,
            arm=arm,
            optimizer_seed=seed,
        )
        for environment, arm, seed in cells
    ], dry_run=bool(args.dry_run), intent_label=(
        f"Freq-HRL MuJoCo v14.1 upper-PD screen {args.run_name}"
    ))
    if args.dispatch and not args.dry_run:
        execute([sys.executable, str(SCHEDULER), "dispatch"], dry_run=False)
    print("merge after result sync: " + shlex.join([
        sys.executable,
        str(LAUNCHER_PATH),
        "--run-name", args.run_name,
        "--merge-only",
    ]))


if __name__ == "__main__":
    main()
