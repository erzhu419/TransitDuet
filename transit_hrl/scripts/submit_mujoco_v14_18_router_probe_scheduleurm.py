#!/usr/bin/env python3
"""Submit and synchronize a frozen MuJoCo router-probe protocol."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import mujoco_v14_18_router_mechanism_screen_spec as spec  # noqa: E402
from scripts.analyze_mujoco_v14_18_router_mechanism_screen import (  # noqa: E402
    analyze_run,
    cell_relative_dir,
)
from scripts.submit_hyperparameter_pilot_scheduleurm import (  # noqa: E402
    LINUX_CPU_NODES,
    SCHEDULER,
    STAGE_EXCLUDES,
    default_python_executable,
    execute,
    execute_bulk,
    parse_csv,
)


SIGNATURE_VERSION = "mujoco-v14-18-router-mechanism-screen-v1"
LAUNCHER_PATH = Path(__file__).resolve()


def anchor_relative_dir(anchor_run_name: str, environment: str, seed: int) -> Path:
    return (
        Path("results") / anchor_run_name / "anchors" / environment
        / f"replicate_{seed}"
    )


def selected_cells() -> list[tuple[str, int]]:
    return [
        (environment, int(seed))
        for environment in spec.ENVIRONMENTS
        for seed in spec.OPTIMIZER_SEEDS
    ]


def task_signature(run_name: str, environment: str, seed: int) -> str:
    return f"Freq-HRL/{SIGNATURE_VERSION}/{run_name}/{environment}/rep-{seed}"


def build_probe_command(
    args: argparse.Namespace, environment: str, seed: int
) -> str:
    anchor = anchor_relative_dir(args.anchor_run_name, environment, seed)
    output = cell_relative_dir(args.run_name, environment, seed) / "probe.json"
    command = [
        str(args.python_executable),
        "scripts/probe_mujoco_radial_restoration.py",
        "--checkpoint", str(anchor / "checkpoint.pt"),
        "--summary", str(anchor / "cell_summary.json"),
        "--output", str(output),
        "--gains", ",".join(map(str, spec.ACTOR_GAINS)),
        "--router-strengths", ",".join(map(str, spec.ROUTER_STRENGTHS)),
        "--profile", spec.PROFILE,
        "--episode-horizon", str(spec.EPISODE_HORIZON),
        "--leakage-cost-mode", spec.LEAKAGE_COST_MODE,
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
    args: argparse.Namespace, environment: str, seed: int
) -> dict[str, object]:
    anchor = ROOT / anchor_relative_dir(args.anchor_run_name, environment, seed)
    output = ROOT / cell_relative_dir(args.run_name, environment, seed)
    return {
        "project": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "description": (
            f"Freq-HRL {spec.DEVELOPMENT_PROTOCOL_VERSION} "
            f"{environment} replicate {seed}"
        ),
        "cmd": build_probe_command(args, environment, seed),
        "cwd": str(ROOT),
        "signature": task_signature(args.run_name, environment, seed),
        "resource_family": (
            f"Freq-HRL/{SIGNATURE_VERSION}/{environment}/single-cell"
        ),
        "vram": 0,
        "ram_mb": int(getattr(spec, "RAM_MB_PER_TASK", 768)),
        "cpu": int(getattr(spec, "CPU_PER_TASK", 1)),
        "priority": str(args.priority),
        "ckpt_dir": str(output),
        "ckpt_glob": "probe.json",
        "skip_resume_scan": True,
        "result_dir": str(output),
        "local_result_dir": str(output),
        "wait_for_files": [
            str(anchor / "checkpoint.pt"),
            str(anchor / "cell_summary.json"),
        ],
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Independent deterministic MuJoCo guard probes use declared CPU cores."
        ),
        "reroute_on_node_down": True,
        "node_down_requeue_s": 600,
        "allowed_nodes": list(args.nodes),
        "require_node": None,
        "stage_excludes": list(STAGE_EXCLUDES),
        "stage_input_paths": [str(anchor)],
        "skip_launch_staging": False,
        "allow_duplicate": False,
    }


def _write_preregistration(args: argparse.Namespace) -> None:
    target = ROOT / "results" / args.run_name
    target.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, capture_output=True,
    ).stdout.strip()
    payload = {
        "status": "frozen_before_cross_environment_router_outcome_access",
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "evidence_role": spec.EVIDENCE_ROLE,
        "source_revision": revision,
        "anchor_run_name": args.anchor_run_name,
        "environments": list(spec.ENVIRONMENTS),
        "optimizer_seeds": list(spec.OPTIMIZER_SEEDS),
        "router_strengths": list(spec.ROUTER_STRENGTHS),
        "actor_gains": list(spec.ACTOR_GAINS),
        "profile": spec.PROFILE,
        "selection_contract": spec.SELECTION_CONTRACT,
        "scheduler_contract": {
            "scheduler": "scheduleurm",
            "allowed_nodes": list(args.nodes),
            "require_node": None,
            "cpu_per_task": 1,
            "ram_mb_per_task": 768,
            "slurm_used": False,
        },
    }
    (target / "preregistration.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _scheduler_tasks(run_name: str) -> dict[str, dict[str, Any]]:
    process = subprocess.run([
        sys.executable, str(SCHEDULER), "status", "--all", "--json", "--readonly",
    ], check=True, text=True, capture_output=True)
    tasks = json.loads(process.stdout).get("tasks", [])
    prefix = f"Freq-HRL/{SIGNATURE_VERSION}/{run_name}/"
    matches: dict[str, dict[str, Any]] = {}
    for task in tasks:
        signature = str(task.get("signature", ""))
        if not signature.startswith(prefix):
            continue
        if signature in matches:
            raise RuntimeError(f"duplicate scheduler task signature: {signature}")
        matches[signature] = dict(task)
    return matches


def sync_results(args: argparse.Namespace) -> None:
    tasks = _scheduler_tasks(args.run_name)
    expected = []
    for environment, seed in selected_cells():
        signature = task_signature(args.run_name, environment, seed)
        task = tasks.get(signature)
        if task is None:
            raise SystemExit(
                f"{spec.DEVELOPMENT_PROTOCOL_VERSION} scheduler task missing: "
                f"{signature}"
            )
        if task.get("status") != "done" or not task.get("node"):
            raise SystemExit(
                f"{spec.DEVELOPMENT_PROTOCOL_VERSION} task is not done: "
                f"{task.get('id')} "
                f"status={task.get('status')}"
            )
        expected.append((
            ROOT / cell_relative_dir(args.run_name, environment, seed), task
        ))

    scheduler_dir = str(SCHEDULER.parent)
    if scheduler_dir not in sys.path:
        sys.path.insert(0, scheduler_dir)
    import scheduler as scheduler_runtime  # type: ignore  # noqa: E402

    pending = [item for item in expected if not (item[0] / "probe.json").is_file()]
    for attempt in range(1, 4):
        if not pending:
            break

        def sync_one(item: tuple[Path, dict[str, Any]]) -> tuple[bool, str]:
            path, task = item
            return scheduler_runtime._sync_one_result({
                "node": task["node"],
                "result_dir": str(path),
                "local_result_dir": str(path),
            })

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=int(args.sync_workers)
        ) as executor:
            outcomes = list(executor.map(sync_one, pending))
        pending = [
            item for item in pending if not (item[0] / "probe.json").is_file()
        ]
        if pending and attempt < 3:
            time.sleep(float(attempt))
    if pending:
        raise SystemExit(
            f"{spec.DEVELOPMENT_PROTOCOL_VERSION} result sync incomplete: "
            f"{len(pending)} cells; "
            f"last_outcomes={outcomes}"
        )
    result = analyze_run(args.run_name)
    print(
        f"synced and analyzed {len(expected)} "
        f"{spec.DEVELOPMENT_PROTOCOL_VERSION} cells: "
        f"{result['status']}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Submit and synchronize frozen MuJoCo router probes for "
            f"{spec.DEVELOPMENT_PROTOCOL_VERSION}."
        )
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--anchor-run-name", default=spec.ANCHOR_RUN_NAME)
    parser.add_argument("--nodes", default=",".join(LINUX_CPU_NODES))
    parser.add_argument("--python-executable", default="")
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="normal"
    )
    parser.add_argument("--sync-workers", type=int, default=4)
    parser.add_argument("--skip-complete-cells", action="store_true")
    parser.add_argument("--sync-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.nodes = parse_csv(args.nodes)
    unknown_nodes = sorted(set(args.nodes) - set(LINUX_CPU_NODES))
    if not args.nodes or unknown_nodes:
        raise SystemExit(
            f"invalid {spec.DEVELOPMENT_PROTOCOL_VERSION} scheduler nodes: "
            f"{unknown_nodes}"
        )
    if not args.python_executable.strip():
        args.python_executable = default_python_executable(args.nodes)
    if not 1 <= int(args.sync_workers) <= 8:
        raise SystemExit(
            f"{spec.DEVELOPMENT_PROTOCOL_VERSION} sync workers must be in [1, 8]"
        )
    for environment, seed in selected_cells():
        anchor = ROOT / anchor_relative_dir(args.anchor_run_name, environment, seed)
        missing = [
            name for name in ("checkpoint.pt", "cell_summary.json")
            if not (anchor / name).is_file()
        ]
        if missing:
            raise SystemExit(
                f"{spec.DEVELOPMENT_PROTOCOL_VERSION} anchor incomplete: "
                f"{anchor}: {missing}"
            )
    return args


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    if args.sync_only:
        sync_results(args)
        return
    if args.analyze_only:
        result = analyze_run(args.run_name)
        print(json.dumps({"status": result["status"]}, sort_keys=True))
        return
    _write_preregistration(args)
    cells = selected_cells()
    if args.skip_complete_cells:
        cells = [
            cell for cell in cells
            if not (
                ROOT / cell_relative_dir(args.run_name, cell[0], cell[1])
                / "probe.json"
            ).is_file()
        ]
    if not cells:
        print(
            f"no {spec.DEVELOPMENT_PROTOCOL_VERSION} router probes "
            "require submission"
        )
        return
    print(
        f"run={args.run_name} cells={len(cells)} nodes={','.join(args.nodes)}",
        flush=True,
    )
    execute_bulk([
        build_scheduler_spec(args, environment, seed)
        for environment, seed in cells
    ], dry_run=bool(args.dry_run), intent_label=(
        f"Freq-HRL {spec.DEVELOPMENT_PROTOCOL_VERSION} {args.run_name}"
    ))
    if args.dispatch and not args.dry_run:
        execute([sys.executable, str(SCHEDULER), "dispatch"], dry_run=False)
    print(
        "sync after completion: "
        + shlex.join([
            sys.executable, str(LAUNCHER_PATH),
            "--run-name", args.run_name,
            "--anchor-run-name", args.anchor_run_name,
            "--sync-only",
        ])
    )


if __name__ == "__main__":
    main()
