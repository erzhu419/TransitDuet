#!/usr/bin/env python3
"""Sync, aggregate, and summarize a broad FreqDuet scheduleurm run."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from submit_freqduet_broad_generalization_scheduleurm import (
    DEFAULT_SEEDS,
    DEFAULT_METHODS,
    METHOD_PARENTS,
    REMOTE_ROOT,
    SCENARIOS,
    config_name,
    parse_csv,
    write_configs,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "zhengliang01@202.197.46.16"
DEFAULT_JUMP = "jtl110gpu2"


def run(cmd: list[str], dry_run: bool = False) -> None:
    print(shlex.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def rsync_remote(args: argparse.Namespace) -> None:
    remote_root = Path(args.remote_root)
    remote_result = remote_root / "results_freqduet" / args.run_name
    local_result = ROOT / "results_freqduet" / args.run_name
    local_result.mkdir(parents=True, exist_ok=True)
    ssh_cmd = f"ssh -J {args.jump_host}"
    run(
        [
            "rsync",
            "-az",
            "--partial",
            "-e",
            ssh_cmd,
            f"{args.host}:{remote_result}/",
            f"{local_result}/",
        ],
        dry_run=args.dry_run,
    )


def aggregate(args: argparse.Namespace, configs: list[str], seeds: list[int]) -> None:
    logs_base = ROOT / "results_freqduet" / args.run_name / "logs_shards"
    logs_dirs = sorted(str(p.relative_to(ROOT)) for p in logs_base.glob("shard_*") if p.is_dir())
    if not logs_dirs:
        raise SystemExit(f"no shard log directories found under {logs_base}")
    cmd = [
        sys.executable,
        "scripts/run_freqduet_ablation.py",
        "--configs", ",".join(configs),
        "--seeds", ",".join(str(x) for x in seeds),
        "--aggregate-only",
        "--aggregate-logs-dirs", ",".join(logs_dirs),
        "--last-k", str(args.last_k),
        "--out-dir", f"results_freqduet/{args.run_name}/combined_summary",
    ]
    run(cmd, dry_run=args.dry_run)


def summarize(args: argparse.Namespace, scenarios: list[str],
              methods: list[str], seeds: list[int]) -> None:
    per_seed = (
        ROOT / "results_freqduet" / args.run_name /
        "combined_summary" / "freqduet_ablation_per_seed.csv"
    )
    cmd = [
        sys.executable,
        "scripts/summarize_freqduet_broad_generalization.py",
        "--per-seed", str(per_seed.relative_to(ROOT)),
        "--out-dir", f"results_freqduet/{args.run_name}/paper_summary",
        "--scenarios", ",".join(scenarios),
        "--methods", ",".join(methods),
        "--seeds", ",".join(str(x) for x in seeds),
    ]
    run(cmd, dry_run=args.dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="broad_generalization_ep100_wu10")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS.keys()))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--last-k", type=int, default=50)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--jump-host", default=DEFAULT_JUMP)
    parser.add_argument("--remote-root", default=str(REMOTE_ROOT))
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scenarios = parse_csv(args.scenarios, SCENARIOS.keys())
    methods = parse_csv(args.methods, METHOD_PARENTS.keys())
    seeds = [int(x) for x in parse_csv(args.seeds)]
    configs = write_configs(scenarios, methods)

    if not args.skip_sync:
        rsync_remote(args)
    if not args.skip_aggregate:
        aggregate(args, configs, seeds)
    if not args.skip_summary:
        summarize(args, scenarios, methods, seeds)


if __name__ == "__main__":
    main()
