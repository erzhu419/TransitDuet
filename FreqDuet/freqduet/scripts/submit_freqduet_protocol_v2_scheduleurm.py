#!/usr/bin/env python3
"""Submit protocol-v2 train/frozen-evaluation jobs directly to CPU nodes."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = Path("/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py")
REMOTE_ROOT = Path("/home/zhengliang01/scheduleurm_work/TransitDuet/FreqDuet/freqduet")
REMOTE_PYTHON = Path("/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python")
DEFAULT_CONFIGS = [
    "F_freqduet_protocol_v2_main_hiro",
    "F_freqduet_protocol_v2_upperdisc_hiro",
    "F_freqduet_protocol_v2_upperhist_hiro",
    "F_freqduet_protocol_v2_upperdisc_hist_hiro",
]
DEFAULT_TRAIN_SEEDS = [7, 17, 31, 42]
DEFAULT_EVAL_SEEDS = [10001, 10007, 10009, 10037, 10039, 10061, 10067, 10069]
DEFAULT_NODES = ["node001", "node002", "node003", "node004", "node005", "node006"]
CPU_JUSTIFICATION = (
    "FreqDuet protocol-v2 transit simulation and reinforcement learning are "
    "CPU-only and run in the isolated freqduet-cpu-py310 environment."
)


def parse_csv(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def ranges(total: int, size: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + max(1, int(size)), total))
        for start in range(0, total, max(1, int(size)))
    ]


def resolve_reference(configs: list[str], reference: str | None) -> str:
    if not configs:
        raise ValueError("at least one config is required")
    requested = Path(reference or configs[0]).stem
    for config in configs:
        if Path(config).stem == requested:
            return config
    raise ValueError("reference config must be included in --configs")


def execute(command: list[str], dry_run: bool) -> None:
    print(shlex.join(command))
    if dry_run:
        return
    process = subprocess.run(command, text=True, capture_output=True)
    output = (process.stdout or "") + (process.stderr or "")
    if process.returncode != 0:
        if "duplicate" not in output.lower() and "already queued" not in output.lower():
            print(output, file=sys.stderr)
            process.check_returncode()
    if output.strip():
        print(output.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument(
        "--reference",
        default=None,
        help="Reference config for paired aggregation; defaults to the first config.",
    )
    parser.add_argument("--train-seeds", default=",".join(map(str, DEFAULT_TRAIN_SEEDS)))
    parser.add_argument("--eval-seeds", default=",".join(map(str, DEFAULT_EVAL_SEEDS)))
    parser.add_argument("--train-episodes", type=int, default=60)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--shard-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--ram-mb", type=int, default=32768)
    parser.add_argument("--nodes", default=",".join(DEFAULT_NODES))
    parser.add_argument("--priority", choices=["low", "normal", "high"], default="normal")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-duplicate", action="store_true")
    args = parser.parse_args()

    configs = parse_csv(args.configs)
    try:
        reference = resolve_reference(configs, args.reference)
    except ValueError as exc:
        parser.error(str(exc))
    train_seeds = parse_csv(args.train_seeds, int)
    eval_seeds = parse_csv(args.eval_seeds, int)
    nodes = parse_csv(args.nodes)
    total = len(configs) * len(train_seeds)
    shards = ranges(total, args.shard_size)
    result_base = f"results_freqduet/{args.run_name}"
    print(
        f"protocol-v2 run={args.run_name} jobs={total} shards={len(shards)} "
        f"train_episodes={args.train_episodes} eval_seeds={len(eval_seeds)}")

    for index, (start, end) in enumerate(shards):
        shard_id = f"{start:04d}_{end:04d}"
        logs_dir = f"{result_base}/logs_shards/shard_{shard_id}"
        out_dir = f"{result_base}/shard_summaries/shard_{shard_id}"
        inner = [
            "PYTHONPATH=.",
            "OMP_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
            "TORCH_NUM_THREADS=1",
            "FREQDUET_TORCH_THREADS=1",
            shlex.join([
                str(REMOTE_PYTHON),
                "-u",
                "scripts/run_freqduet_protocol_v2_matrix.py",
                "--configs", ",".join(configs),
                "--reference", reference,
                "--train-seeds", ",".join(map(str, train_seeds)),
                "--eval-seeds", ",".join(map(str, eval_seeds)),
                "--train-episodes", str(args.train_episodes),
                "--workers", str(args.workers),
                "--worker-threads", "1",
                "--logs-dir", logs_dir,
                "--out-dir", out_dir,
                "--job-start", str(start),
                "--job-end", str(end),
                "--skip-existing",
                "--suppress-heavy-artifacts",
            ]),
        ]
        node = nodes[index % len(nodes)]
        command = [
            sys.executable,
            str(SCHEDULER),
            "submit",
            "--project", "FreqDuet",
            "--description", f"FreqDuet protocol-v2 {args.run_name} shard {index + 1}/{len(shards)}",
            "--cmd", " ".join(inner),
            "--cwd", str(ROOT),
            "--signature", f"FreqDuet/{args.run_name}/shard_{shard_id}",
            "--vram", "0",
            "--ram-mb", str(args.ram_mb),
            "--cpu", str(args.cpu),
            "--priority", args.priority,
            "--require-node", node,
            "--result-dir", str(REMOTE_ROOT / logs_dir),
            "--local-result-dir", str(ROOT / logs_dir),
            "--allow-cpu-training",
            "--cpu-training-justification", CPU_JUSTIFICATION,
            "--allow-no-resume",
            "--allow-remote-large-data",
            "--reroute-on-node-down",
            "--node-down-requeue-s", "900",
        ]
        if args.allow_duplicate:
            command.append("--allow-duplicate")
        execute(command, args.dry_run)

    print("\nAggregate after scheduler sync:")
    print(
        "python3 scripts/run_freqduet_protocol_v2_matrix.py --aggregate-only "
        f"--configs {shlex.quote(','.join(configs))} "
        f"--reference {shlex.quote(reference)} "
        f"--train-seeds {shlex.quote(','.join(map(str, train_seeds)))} "
        f"--eval-seeds {shlex.quote(','.join(map(str, eval_seeds)))} "
        f"--logs-dir {result_base}/logs_shards/shard_0000_0000 "
        f"--aggregate-logs-dirs \"$(find {result_base}/logs_shards -mindepth 1 -maxdepth 1 -type d | sort | paste -sd, -)\" "
        f"--out-dir {result_base}/combined_summary")
    if args.dispatch and not args.dry_run:
        execute([sys.executable, str(SCHEDULER), "dispatch"], False)


if __name__ == "__main__":
    main()
