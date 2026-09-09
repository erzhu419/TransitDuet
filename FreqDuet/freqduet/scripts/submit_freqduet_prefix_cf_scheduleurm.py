#!/usr/bin/env python3
"""Submit the frozen V28 exact-prefix matrix to HPC CPU nodes."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_protocol_v6_v28_prefix_common import (
    CHECKPOINT_EP,
    CONFIG,
    DECISION_INDICES,
    EVAL_EPISODE,
    EVAL_SEEDS,
    OFFSETS_S,
    REPLAY_SEED,
    TRAIN_SEEDS,
    checkpoint_dir,
    expected_jobs,
    job_name,
)


SCHEDULER = Path("/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py")
REMOTE_PYTHON = Path(
    "/home/zhengliang01/scheduleurm_work/conda_envs/"
    "freqduet-cpu-py310/bin/python"
)
DEFAULT_NODES = [
    "node001", "node002", "node003", "node004", "node005", "node006"
]
CPU_JUSTIFICATION = (
    "Frozen-checkpoint V28 counterfactual replay is a registered CPU-only "
    "simulation workload in the isolated FreqDuet conda environment."
)


def csv(values) -> str:
    return ",".join(str(value) for value in values)


def source_commit() -> str:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if dirty:
        raise SystemExit("V28 submission requires a clean tracked source snapshot")
    if len(commit) != 40:
        raise SystemExit(f"expected a full Git commit, got {commit!r}")
    return commit


def run(command: list[str], *, dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if dry_run:
        return
    proc = subprocess.run(command, text=True, capture_output=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        if "duplicate" in output.lower() or "already queued" in output.lower():
            print(output.strip(), flush=True)
            return
        print(output, file=sys.stderr)
        proc.check_returncode()
    if proc.stdout:
        print(proc.stdout.strip(), flush=True)


def build_job_command(
    *, commit: str, run_name: str, train_seed: int, eval_seed: int,
    decision_index: int,
) -> tuple[str, str]:
    name = job_name(train_seed, eval_seed, decision_index)
    out_dir = (
        Path("/home/zhengliang01/scheduleurm_work/results")
        / run_name
        / "jobs"
        / name
    )
    env = [
        "PYTHONPATH=.",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "TORCH_NUM_THREADS=1",
        "FREQDUET_TORCH_THREADS=1",
        f"FREQDUET_SOURCE_COMMIT={commit}",
        "FREQDUET_SOURCE_BRANCH=codex/freqduet-v6-causal-protocol",
        "FREQDUET_SOURCE_TRACKED_DIRTY=0",
    ]
    command = [
        str(REMOTE_PYTHON),
        "-u",
        "scripts/audit_freqduet_prefix_counterfactual.py",
        "--config", CONFIG,
        "--train-seed", str(train_seed),
        "--checkpoint-dir", str(checkpoint_dir(train_seed)),
        "--checkpoint-ep", str(CHECKPOINT_EP),
        "--eval-episode", str(EVAL_EPISODE),
        "--scenario-seed", str(eval_seed),
        "--replay-seed", str(REPLAY_SEED),
        "--decision-index", str(decision_index),
        f"--offsets-s={csv(OFFSETS_S)}",
        "--worker-threads", "1",
        "--expected-source-commit", commit,
        "--out-dir", str(out_dir),
    ]
    return "export " + " ".join(env) + " && " + shlex.join(command), str(out_dir)


def submit(args: argparse.Namespace) -> int:
    commit = source_commit()
    if args.expected_commit and commit != args.expected_commit:
        raise SystemExit(f"source commit {commit} != expected {args.expected_commit}")
    nodes = [item.strip() for item in args.nodes.split(",") if item.strip()]
    unknown = sorted(set(nodes) - set(DEFAULT_NODES))
    if not nodes or unknown:
        raise SystemExit(f"nodes must be a nonempty subset of {DEFAULT_NODES}; got {nodes}")

    jobs = expected_jobs()
    print(
        f"V28 frozen matrix commit={commit} jobs={len(jobs)} branches/job=7 "
        f"train={TRAIN_SEEDS} eval={EVAL_SEEDS} decisions={DECISION_INDICES} "
        f"offsets={OFFSETS_S}",
        flush=True,
    )
    for index, (train_seed, eval_seed, decision_index) in enumerate(jobs):
        node = nodes[index % len(nodes)]
        name = job_name(train_seed, eval_seed, decision_index)
        inner, _ = build_job_command(
            commit=commit,
            run_name=args.run_name,
            train_seed=train_seed,
            eval_seed=eval_seed,
            decision_index=decision_index,
        )
        command = [
            sys.executable,
            str(SCHEDULER),
            "submit",
            "--project", "FreqDuet",
            "--description", f"FreqDuet V28 exact-prefix {name}",
            "--cmd", inner,
            "--cwd", str(ROOT),
            "--signature", f"FreqDuet/{args.run_name}/{name}",
            "--vram", "0",
            "--ram-mb", str(args.ram_mb),
            "--cpu", "1",
            "--priority", args.priority,
            "--require-node", node,
            "--allow-cpu-training",
            "--cpu-training-justification", CPU_JUSTIFICATION,
            "--allow-no-ckpt",
            "--allow-no-resume",
            "--allow-remote-large-data",
            "--reroute-on-node-down",
            "--node-down-requeue-s", "900",
        ]
        if args.allow_duplicate:
            command.append("--allow-duplicate")
        run(command, dry_run=args.dry_run)

    if args.dispatch and not args.dry_run:
        run([sys.executable, str(SCHEDULER), "dispatch"], dry_run=False)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--nodes", default=csv(DEFAULT_NODES))
    parser.add_argument("--ram-mb", type=int, default=1536)
    parser.add_argument("--priority", choices=("low", "normal", "high"), default="high")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--allow-duplicate", action="store_true")
    return submit(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
