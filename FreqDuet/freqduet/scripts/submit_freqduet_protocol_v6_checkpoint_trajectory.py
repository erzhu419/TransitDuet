#!/usr/bin/env python3
"""Dispatch V34 trajectories and their dependent server-side aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_freqduet_protocol_v6_checkpoint_trajectory import (
    CONFIGS, RUN_NAME, TRAIN_SEEDS,
)


SCHEDULER = Path("/home/erzhu419/.claude/skills/scheduler/scheduler.py")
REMOTE_ROOT = Path("/home/zhengliang01/scheduleurm_work") / ROOT.relative_to(
    "/home/erzhu419/mine_code"
)
PYTHON = "/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python"
NODES = [f"node{index:03d}" for index in range(1, 7)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-task", required=True)
    args = parser.parse_args()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise SystemExit("Commit the V34 diagnostic source before dispatch")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    remote_output = REMOTE_ROOT / "results_freqduet" / RUN_NAME
    local_output = ROOT / "results_freqduet" / RUN_NAME
    queue = json.loads((Path.home() / ".claude/scheduler/queue.json").read_text())
    smoke = next(task for task in queue["tasks"] if task["id"] == args.smoke_task)
    if smoke["signature"] != f"FreqDuet/{RUN_NAME}/smoke" or not smoke.get("exit_status_path"):
        raise SystemExit("The V34 smoke must be launched before queuing its dependents")
    smoke_file = REMOTE_ROOT / "results_freqduet" / f"{RUN_NAME}_smoke/smoke_pass.json"
    source = remote_output / "source/run_checkpoint_trajectory.py"
    smoke_ready = remote_output / "smoke_ready"
    q = shlex.quote
    prepare = (
        f"set -eu; mkdir -p {q(str(source.parent))} {q(str(smoke_ready))}; "
        f"cp {q(str(REMOTE_ROOT / 'scripts/run_freqduet_protocol_v6_checkpoint_trajectory.py'))} "
        f"{q(str(source))}; "
        f"while test ! -s {q(str(smoke_file))}; do "
        f"if test -s {q(smoke['exit_status_path'])}; then "
        "echo 'Smoke ended without passing checkpoint equivalence'; exit 1; fi; sleep 5; done; "
        f"cp {q(str(smoke_file))} {q(str(smoke_ready / 'smoke_pass.json'))}; "
        "echo TRAJECTORY_SMOKE_GATE_READY"
    )
    smoke_prerequisite = str(local_output / "smoke_ready/smoke_pass.json")
    command = [
        "env", "PATH=/usr/local/bin:/usr/bin:/bin", "PYTHONPATH=.",
        "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
        "FREQDUET_TORCH_THREADS=1", f"FREQDUET_DIAGNOSTIC_COMMIT={commit}",
        PYTHON, "-u", str(source),
    ]
    common = {
        "project": "FreqDuet", "cwd": str(ROOT), "vram": 0, "cpu": 2,
        "ram_mb": 4096, "priority": "normal", "allowed_nodes": NODES,
        "stage_excludes": ["results_freqduet", "**/__pycache__"],
        "skip_launch_staging": True,
        "skip_resume_scan": True,
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Historical transit simulator checkpoint diagnostics run on CPU "
            "in the independent freqduet-cpu-py310 environment."
        ),
    }
    specs = [{
        **common, "description": "V34 await checkpoint-save smoke and freeze diagnostic script",
        "signature": f"FreqDuet/{RUN_NAME}/smoke_gate",
        "require_node": smoke["node"], "cmd": prepare, "cpu": 1, "ram_mb": 64,
        "skip_launch_staging": False, "stage_input_paths": [str(ROOT / "scripts")],
        "result_dir": str(smoke_ready),
        "local_result_dir": str(local_output / "smoke_ready"),
    }]
    prerequisites = []
    for config in CONFIGS:
        for seed in TRAIN_SEEDS:
            name = f"{config}_seed{seed}"
            ready = Path("ready") / name
            prerequisites.append(str(local_output / ready / "ready.json"))
            specs.append({
                **common,
                "description": f"V34 checkpoint trajectory {config} seed {seed}",
                "signature": f"FreqDuet/{RUN_NAME}/{name}",
                "preferred_node": NODES[len(specs) % len(NODES)],
                "wait_for_files": [smoke_prerequisite],
                "cmd": shlex.join([*command, "run", "--out-root", str(remote_output),
                                   "--config", config, "--train-seed", str(seed)]),
                "result_dir": str(remote_output / ready),
                "local_result_dir": str(local_output / ready),
            })
    specs.append({
        **common,
        "description": "V34 paired checkpoint and cohort aggregation",
        "signature": f"FreqDuet/{RUN_NAME}/aggregate",
        "preferred_node": "node004",
        "cmd": shlex.join([*command, "aggregate", "--out-root", str(remote_output)]),
        "wait_for_files": prerequisites,
        "result_dir": str(remote_output / "summary"),
        "local_result_dir": str(local_output / "summary"),
    })
    process = subprocess.run(
        [sys.executable, str(SCHEDULER), "submit-jsonl", "--stdin", "--trusted",
         "--json", "--lock-timeout", "60", "--intent-label", f"FreqDuet/{RUN_NAME}"],
        input=json.dumps(specs), capture_output=True, text=True, check=True,
    )
    payload = json.loads(process.stdout)
    tasks = payload["submitted"]
    local_output.mkdir(parents=True, exist_ok=True)
    (local_output / "dispatch.json").write_text(json.dumps({
        "diagnostic_commit": commit, "submitted": tasks,
        "source_commit": "673355cae10640b2737b6d51d541a9685059c342",
    }, indent=2) + "\n")
    print(json.dumps(payload), flush=True)
    dispatch = [sys.executable, str(SCHEDULER), "dispatch", "--lock-timeout", "60"]
    for task in tasks:
        dispatch.extend(["--task-id", task["id"]])
    subprocess.run(dispatch, check=True)


if __name__ == "__main__":
    main()
