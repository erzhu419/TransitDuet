#!/usr/bin/env python3
"""Bulk-submit the frozen V30 discovery labels to HPC CPU nodes."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_protocol_v6_v30_expanded_prefix_common import (
    CHECKPOINT_EP,
    CONFIG,
    DISCOVERY_DECISION_INDICES,
    DISCOVERY_EVAL_EPISODE,
    DISCOVERY_REPLAY_SEED,
    DISCOVERY_SCENARIO_SEEDS,
    DISCOVERY_TRAIN_SEEDS,
    LABEL_PROTOCOL_VERSION,
    OFFSETS_S,
    discovery_checkpoint_dir,
    discovery_job_name,
    expected_discovery_jobs,
)


SCHEDULER = Path("/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py")
REMOTE_PYTHON = Path(
    "/home/zhengliang01/scheduleurm_work/conda_envs/"
    "freqduet-cpu-py310/bin/python"
)
REMOTE_RESULTS = Path("/home/zhengliang01/scheduleurm_work/results")
DEFAULT_NODES = [
    "node001", "node002", "node003", "node004", "node005", "node006",
]
CPU_JUSTIFICATION = (
    "Frozen-checkpoint V30 exact-prefix replay is a CPU-only transit "
    "simulation in the isolated FreqDuet conda environment."
)


def csv(values) -> str:
    return ",".join(str(value) for value in values)


def source_commit() -> str:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        [
            "git", "-C", str(ROOT), "status", "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()
    if dirty:
        raise SystemExit("V30 submission requires a clean tracked source snapshot")
    if len(commit) != 40:
        raise SystemExit(f"expected a full Git commit, got {commit!r}")
    return commit


def build_job_command(
    *,
    commit: str,
    run_name: str,
    train_seed: int,
    scenario_seed: int,
    decision_index: int,
) -> tuple[str, str]:
    name = discovery_job_name(train_seed, scenario_seed, decision_index)
    out_dir = REMOTE_RESULTS / run_name / "jobs" / name
    env = {
        "PYTHONPATH": ".",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TORCH_NUM_THREADS": "1",
        "FREQDUET_TORCH_THREADS": "1",
        "FREQDUET_SOURCE_COMMIT": commit,
        "FREQDUET_SOURCE_BRANCH": "codex/freqduet-v6-causal-protocol",
        "FREQDUET_SOURCE_TRACKED_DIRTY": "0",
    }
    command = [
        str(REMOTE_PYTHON),
        "-u",
        "scripts/audit_freqduet_prefix_counterfactual.py",
        "--protocol-version", LABEL_PROTOCOL_VERSION,
        "--config", CONFIG,
        "--train-seed", str(train_seed),
        "--checkpoint-dir", str(discovery_checkpoint_dir(train_seed)),
        "--checkpoint-ep", str(CHECKPOINT_EP),
        "--eval-episode", str(DISCOVERY_EVAL_EPISODE),
        "--scenario-seed", str(scenario_seed),
        "--replay-seed", str(DISCOVERY_REPLAY_SEED),
        "--decision-index", str(decision_index),
        f"--offsets-s={csv(OFFSETS_S)}",
        "--worker-threads", "1",
        "--expected-source-commit", commit,
        "--out-dir", str(out_dir),
    ]
    exports = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in env.items()
    )
    return f"export {exports} && {shlex.join(command)}", str(out_dir)


def build_specs(
    *,
    commit: str,
    run_name: str,
    nodes: list[str],
    ram_mb: int,
    priority: str,
    allow_duplicate: bool,
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for index, (train_seed, scenario_seed, decision_index) in enumerate(
            expected_discovery_jobs()):
        node = nodes[index % len(nodes)]
        name = discovery_job_name(train_seed, scenario_seed, decision_index)
        command, _ = build_job_command(
            commit=commit,
            run_name=run_name,
            train_seed=train_seed,
            scenario_seed=scenario_seed,
            decision_index=decision_index,
        )
        specs.append({
            "project": "FreqDuet",
            "description": f"FreqDuet V30 expanded prefix {name}",
            "cmd": command,
            "cwd": str(ROOT),
            "signature": f"FreqDuet/{run_name}/{name}",
            "vram": 0,
            "ram_mb": int(ram_mb),
            "cpu": 1,
            "priority": priority,
            "require_node": node,
            "skip_resume_scan": True,
            "allow_cpu_training": True,
            "cpu_training_justification": CPU_JUSTIFICATION,
            "allow_remote_large_data": True,
            "reroute_on_node_down": True,
            "node_down_requeue_s": 900,
            "allow_duplicate": bool(allow_duplicate),
        })
    return specs


def execute(command: list[str], *, input_text: str | None = None) -> str:
    process = subprocess.run(
        command,
        text=True,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        detail = (process.stdout or "") + (process.stderr or "")
        raise RuntimeError(detail.strip() or f"command failed: {shlex.join(command)}")
    return process.stdout or ""


def submit(args: argparse.Namespace) -> int:
    commit = source_commit()
    if args.expected_commit and commit != args.expected_commit:
        raise SystemExit(f"source commit {commit} != expected {args.expected_commit}")
    nodes = [item.strip() for item in args.nodes.split(",") if item.strip()]
    unknown = sorted(set(nodes) - set(DEFAULT_NODES))
    if not nodes or unknown:
        raise SystemExit(
            f"nodes must be a nonempty subset of {DEFAULT_NODES}; got {nodes}"
        )
    specs = build_specs(
        commit=commit,
        run_name=args.run_name,
        nodes=nodes,
        ram_mb=args.ram_mb,
        priority=args.priority,
        allow_duplicate=args.allow_duplicate,
    )
    assignments = Counter(str(spec["require_node"]) for spec in specs)
    print(
        f"V30 frozen discovery commit={commit} jobs={len(specs)} "
        f"branches/job=7 train={DISCOVERY_TRAIN_SEEDS} "
        f"scenarios={DISCOVERY_SCENARIO_SEEDS} "
        f"decisions={DISCOVERY_DECISION_INDICES} offsets={OFFSETS_S} "
        f"nodes={dict(assignments)}",
        flush=True,
    )
    if args.dry_run:
        return 0

    output = execute(
        [
            sys.executable,
            str(SCHEDULER),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--lock-timeout", "600",
            "--intent-label", f"FreqDuet/{args.run_name}",
        ],
        input_text=json.dumps(specs),
    )
    try:
        payload = json.loads(output)
        task_ids = [str(item["id"]) for item in payload.get("submitted", [])]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("scheduler returned an invalid bulk-submit payload") from exc
    if len(task_ids) != len(specs):
        raise RuntimeError(
            f"scheduler submitted {len(task_ids)} of {len(specs)} V30 jobs"
        )
    print(
        f"submitted {len(task_ids)} V30 jobs: {task_ids[0]}..{task_ids[-1]}",
        flush=True,
    )
    if args.dispatch:
        dispatch = [
            sys.executable,
            str(SCHEDULER),
            "dispatch",
            "--lock-timeout", "600",
            "--intent-label", f"FreqDuet/{args.run_name}",
        ]
        for task_id in task_ids:
            dispatch.extend(["--task-id", task_id])
        dispatch_output = execute(dispatch)
        if dispatch_output.strip():
            print(dispatch_output.strip(), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--nodes", default=csv(DEFAULT_NODES))
    parser.add_argument("--ram-mb", type=int, default=1536)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="high"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--allow-duplicate", action="store_true")
    return submit(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
