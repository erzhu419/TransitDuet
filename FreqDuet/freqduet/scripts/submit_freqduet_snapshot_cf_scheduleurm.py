#!/usr/bin/env python3
"""Submit FreqDuet snapshot counterfactual audits to scheduleurm CPU nodes."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from run_freqduet_snapshot_counterfactual_matrix import (
    DEFAULT_CONFIGS,
    DEFAULT_SEEDS,
    parse_csv,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = Path("/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py")
REMOTE_ROOT = Path("/home/zhengliang01/scheduleurm_work/TransitDuet/FreqDuet/freqduet")
REMOTE_PYTHON = Path("/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python")
DEFAULT_NODES = ["node001", "node002", "node003", "node004", "node005", "node006"]
CPU_JUSTIFICATION = (
    "FreqDuet snapshot counterfactual audits are CPU-only on zhengliang-hpc CPU "
    "nodes, using the isolated freqduet-cpu-py310 conda environment."
)


def shard_ranges(total: int, shard_size: int) -> list[tuple[int, int]]:
    shard_size = max(1, int(shard_size))
    return [(start, min(start + shard_size, total)) for start in range(0, total, shard_size)]


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
    cmd = [
        str(REMOTE_PYTHON),
        "-u",
        "scripts/run_freqduet_snapshot_counterfactual_matrix.py",
        "--configs", configs_csv,
        "--seeds", seeds_csv,
        "--episodes", str(int(args.episodes)),
        "--burn-in-episodes", str(int(args.burn_in_episodes)),
        "--upper-warmup-eps", str(int(args.upper_warmup_eps)),
        "--max-snapshots", str(int(args.max_snapshots)),
        "--start-dispatch", str(int(args.start_dispatch)),
        "--snapshot-stride", str(int(args.snapshot_stride)),
        "--horizon-s", str(float(args.horizon_s)),
        f"--deltas-s={args.deltas_s}",
        "--modes", args.modes,
        "--candidate-frame", args.candidate_frame,
        "--terminal-hold-s", str(float(args.terminal_hold_s)),
        "--terminal-min-s", str(float(args.terminal_min_s)),
        "--terminal-floor-ratio", str(float(args.terminal_floor_ratio)),
        "--terminal-floor-min-s", str(float(args.terminal_floor_min_s)),
        "--risk-proxy-wait-weight", str(float(args.risk_proxy_wait_weight)),
        "--risk-proxy-cv-weight", str(float(args.risk_proxy_cv_weight)),
        "--risk-proxy-overshoot-sq-weight", str(float(args.risk_proxy_overshoot_sq_weight)),
        "--risk-proxy-overshoot-mean-weight", str(float(args.risk_proxy_overshoot_mean_weight)),
        "--risk-proxy-holding-weight", str(float(args.risk_proxy_holding_weight)),
        "--risk-proxy-cv-excess-target", str(float(args.risk_proxy_cv_excess_target)),
        "--risk-proxy-cv-excess-weight", str(float(args.risk_proxy_cv_excess_weight)),
        "--risk-proxy-launch-delay-weight", str(float(args.risk_proxy_launch_delay_weight)),
        "--risk-proxy-positive-offset-weight", str(float(args.risk_proxy_positive_offset_weight)),
        "--workers", str(int(args.workers)),
        "--worker-threads", "1",
        "--job-start", str(int(start)),
        "--job-end", str(int(end)),
        "--out-dir", out_dir,
        "--skip-existing",
        "--no-aggregate",
    ]
    if args.stochastic_lower:
        cmd.append("--stochastic-lower")
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
    result_base = f"results_freqduet/{args.run_name}/shards/shard_{shard_id}"
    remote_result_dir = str(REMOTE_ROOT / result_base)
    local_result_dir = str(ROOT / result_base)
    cmd = build_inner_cmd(args, configs_csv, seeds_csv, start, end, result_base)
    command = [
        sys.executable, str(SCHEDULER), "submit",
        "--project", "FreqDuet",
        "--description",
        f"FreqDuet snapshot CF {args.run_name} shard {shard_index + 1}/{num_shards}",
        "--cmd", cmd,
        "--cwd", str(ROOT),
        "--signature", f"FreqDuet/{args.run_name}/snapshot_cf_shard_{shard_id}",
        "--vram", "0",
        "--ram-mb", str(int(args.ram_mb)),
        "--cpu", str(int(args.cpu)),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--burn-in-episodes", type=int, default=0)
    parser.add_argument("--upper-warmup-eps", type=int, default=0)
    parser.add_argument("--max-snapshots", type=int, default=20)
    parser.add_argument("--start-dispatch", type=int, default=4)
    parser.add_argument("--snapshot-stride", type=int, default=4)
    parser.add_argument("--horizon-s", type=float, default=600.0)
    parser.add_argument("--modes", default="target,terminalhold45")
    parser.add_argument("--deltas-s", default="-20,0,20")
    parser.add_argument(
        "--candidate-frame",
        choices=["absolute", "actor_relative"],
        default="absolute",
    )
    parser.add_argument("--terminal-hold-s", type=float, default=45.0)
    parser.add_argument("--terminal-min-s", type=float, default=0.0)
    parser.add_argument("--terminal-floor-ratio", type=float, default=0.0)
    parser.add_argument("--terminal-floor-min-s", type=float, default=0.0)
    parser.add_argument("--stochastic-lower", action="store_true")
    parser.add_argument("--risk-proxy-wait-weight", type=float, default=1.0)
    parser.add_argument("--risk-proxy-cv-weight", type=float, default=1.0)
    parser.add_argument("--risk-proxy-overshoot-sq-weight", type=float, default=1.0)
    parser.add_argument("--risk-proxy-overshoot-mean-weight", type=float, default=0.0)
    parser.add_argument("--risk-proxy-holding-weight", type=float, default=0.10)
    parser.add_argument("--risk-proxy-cv-excess-target", type=float, default=0.44)
    parser.add_argument("--risk-proxy-cv-excess-weight", type=float, default=0.0)
    parser.add_argument("--risk-proxy-launch-delay-weight", type=float, default=0.0)
    parser.add_argument("--risk-proxy-positive-offset-weight", type=float, default=0.0)
    parser.add_argument("--shard-size", type=int, default=14)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--cpu", type=int, default=14)
    parser.add_argument("--ram-mb", type=int, default=49152)
    parser.add_argument("--nodes", default=",".join(DEFAULT_NODES))
    parser.add_argument("--priority", choices=["low", "normal", "high"], default="normal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--allow-duplicate", action="store_true")
    parser.add_argument("--baseline-method", default="target_0")
    parser.add_argument("--summary-metric", default="proxy_cost")
    args = parser.parse_args()

    configs = parse_csv(args.configs, str)
    seeds = parse_csv(args.seeds, int)
    nodes = parse_csv(args.nodes, str)
    ranges = shard_ranges(len(configs) * len(seeds), int(args.shard_size))
    print(
        f"Submitting snapshot CF run={args.run_name}; configs={len(configs)}; "
        f"seeds={len(seeds)}; total jobs={len(configs) * len(seeds)}; "
        f"shards={len(ranges)}; nodes={','.join(nodes)}"
    )

    configs_csv = ",".join(configs)
    seeds_csv = ",".join(str(seed) for seed in seeds)
    for i, (start, end) in enumerate(ranges):
        submit_shard(
            args=args,
            configs_csv=configs_csv,
            seeds_csv=seeds_csv,
            shard_index=i,
            num_shards=len(ranges),
            start=start,
            end=end,
            node=nodes[i % len(nodes)],
        )

    if args.dispatch and not args.dry_run:
        run_command([sys.executable, str(SCHEDULER), "dispatch"], False)
    print("\nAggregate after sync:")
    print(
        "  python3 scripts/run_freqduet_snapshot_counterfactual_matrix.py "
        f"--aggregate-only --out-dir results_freqduet/{args.run_name}/shards "
        f"--baseline-method {shlex.quote(args.baseline_method)} "
        f"--summary-metric {shlex.quote(args.summary_metric)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
