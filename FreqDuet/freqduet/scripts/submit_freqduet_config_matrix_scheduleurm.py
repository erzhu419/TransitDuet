#!/usr/bin/env python3
"""Submit an arbitrary FreqDuet config x seed matrix to scheduleurm.

This is intentionally scheduler-direct: shards are hard-pinned to
node001-node006 so they appear in tui-top as normal scheduleurm tasks instead
of being auto-routed through the zhengliang-hpc Slurm backend.
"""

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
DEFAULT_SEEDS = [
    7, 11, 17, 23, 31, 37, 42, 43, 53, 61,
    71, 83, 97, 109, 123, 127, 149, 456, 789, 2026,
]
DEFAULT_NODES = ["node001", "node002", "node003", "node004", "node005", "node006"]
CPU_JUSTIFICATION = (
    "FreqDuet transit-control experiments are CPU-only on zhengliang-hpc CPU "
    "nodes, using an isolated conda environment and no GPU kernels."
)


def parse_csv(value: str, cast=str) -> list:
    return [cast(x.strip()) for x in str(value).split(",") if x.strip()]


def shard_ranges(total: int, shard_size: int) -> list[tuple[int, int]]:
    shard_size = max(1, int(shard_size))
    return [
        (start, min(start + shard_size, total))
        for start in range(0, total, shard_size)
    ]


def run_command(command: list[str], dry_run: bool) -> None:
    print(shlex.join(command))
    if dry_run:
        return
    proc = subprocess.run(command, text=True, capture_output=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        if "duplicate" in output.lower() or "already queued" in output.lower():
            print(output.strip())
            return
        print(output, file=sys.stderr)
        proc.check_returncode()
    if proc.stdout:
        print(proc.stdout.strip())


def build_inner_cmd(
    args: argparse.Namespace,
    configs_csv: str,
    seeds_csv: str,
    start: int,
    end: int,
    logs_dir: str,
    out_dir: str,
) -> str:
    env_bits = [
        "PYTHONPATH=.",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "VECLIB_MAXIMUM_THREADS=1",
        "TORCH_NUM_THREADS=1",
        "FREQDUET_TORCH_THREADS=1",
    ]
    if args.suppress_heavy_artifacts:
        env_bits.append("FREQDUET_SUPPRESS_HEAVY_ARTIFACTS=1")
    cmd = [
        str(REMOTE_PYTHON),
        "-u",
        "scripts/run_freqduet_ablation.py",
        "--configs", configs_csv,
        "--seeds", seeds_csv,
        "--episodes", str(args.episodes),
        "--last-k", str(args.last_k),
        "--workers", str(args.workers),
        "--worker-threads", "1",
        "--logs-dir", logs_dir,
        "--out-dir", out_dir,
        "--job-start", str(start),
        "--job-end", str(end),
        "--skip-existing",
        "--no-aggregate",
    ]
    if args.upper_warmup_eps is not None:
        cmd.extend(["--upper-warmup-eps", str(args.upper_warmup_eps)])
    return " ".join(env_bits + [shlex.join(cmd)])


def submit_shard(
    args: argparse.Namespace,
    configs_csv: str,
    seeds_csv: str,
    shard_index: int,
    num_shards: int,
    start: int,
    end: int,
    node: str,
) -> None:
    shard_id = f"{start:04d}_{end:04d}"
    result_base = f"results_freqduet/{args.run_name}"
    logs_dir = f"{result_base}/logs_shards/shard_{shard_id}"
    out_dir = f"{result_base}/shard_summaries/shard_{shard_id}"
    local_result_dir = str(ROOT / logs_dir)
    remote_result_dir = str(REMOTE_ROOT / logs_dir)
    cmd = build_inner_cmd(args, configs_csv, seeds_csv, start, end, logs_dir, out_dir)
    command = [
        sys.executable, str(SCHEDULER), "submit",
        "--project", "FreqDuet",
        "--description",
        f"FreqDuet config matrix {args.run_name} shard {shard_index + 1}/{num_shards}",
        "--cmd", cmd,
        "--cwd", str(ROOT),
        "--signature", f"FreqDuet/{args.run_name}/shard_{shard_id}",
        "--vram", "0",
        "--ram-mb", str(args.ram_mb),
        "--cpu", str(args.cpu),
        "--priority", args.priority,
        "--require-node", node,
        "--result-dir", remote_result_dir,
        "--local-result-dir", local_result_dir,
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
    run_command(command, args.dry_run)


def print_aggregate_hint(run_name: str, configs: list[str],
                         seeds: list[int], last_k: int) -> None:
    configs_csv = ",".join(configs)
    seeds_csv = ",".join(str(s) for s in seeds)
    logs_expr = (
        f"$(find results_freqduet/{run_name}/logs_shards "
        "-mindepth 1 -maxdepth 1 -type d | sort | paste -sd, -)"
    )
    print("\nAggregate after result sync:")
    print(
        "  python3 scripts/run_freqduet_ablation.py "
        f"--configs {shlex.quote(configs_csv)} "
        f"--seeds {shlex.quote(seeds_csv)} "
        "--aggregate-only "
        f"--aggregate-logs-dirs \"{logs_expr}\" "
        f"--last-k {int(last_k)} "
        f"--out-dir results_freqduet/{run_name}/combined_summary"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--last-k", type=int, default=50)
    parser.add_argument("--upper-warmup-eps", type=int, default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--shard-size", type=int, default=30)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--cpu", type=int, default=30)
    parser.add_argument("--ram-mb", type=int, default=32768)
    parser.add_argument("--nodes", default=",".join(DEFAULT_NODES))
    parser.add_argument("--priority", choices=["low", "normal", "high"], default="normal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--allow-duplicate", action="store_true")
    parser.add_argument(
        "--suppress-heavy-artifacts",
        action="store_true",
        help="Disable trip_details.csv and checkpoint writes for large paper matrices.",
    )
    args = parser.parse_args()

    configs = parse_csv(args.configs, str)
    seeds = parse_csv(args.seeds, int)
    nodes = parse_csv(args.nodes, str)
    total_jobs = len(configs) * len(seeds)
    ranges = shard_ranges(total_jobs, args.shard_size)
    print(
        f"Submitting config matrix run={args.run_name}; configs={len(configs)}; "
        f"seeds={len(seeds)}; total jobs={total_jobs}; shards={len(ranges)}; "
        f"nodes={','.join(nodes)}"
    )
    print_aggregate_hint(args.run_name, configs, seeds, args.last_k)

    configs_csv = ",".join(configs)
    seeds_csv = ",".join(str(x) for x in seeds)
    for i, (start, end) in enumerate(ranges):
        node = nodes[i % len(nodes)]
        submit_shard(
            args=args,
            configs_csv=configs_csv,
            seeds_csv=seeds_csv,
            shard_index=i,
            num_shards=len(ranges),
            start=start,
            end=end,
            node=node,
        )

    if args.dispatch and not args.dry_run:
        run_command([sys.executable, str(SCHEDULER), "dispatch"], False)


if __name__ == "__main__":
    main()
