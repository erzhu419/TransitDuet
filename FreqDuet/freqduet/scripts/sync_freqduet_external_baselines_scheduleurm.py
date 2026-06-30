#!/usr/bin/env python3
"""Sync and aggregate FreqDuet external-baseline scheduleurm runs."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from submit_freqduet_config_matrix_scheduleurm import (
    DEFAULT_SEEDS,
    REMOTE_ROOT,
    parse_csv,
    parse_csv_file,
)
from submit_freqduet_external_baselines_scheduleurm import DEFAULT_VARIANTS


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


def aggregate(
    args: argparse.Namespace,
    configs: list[str],
    variants: list[str],
    seeds: list[int],
) -> None:
    logs_base = ROOT / "results_freqduet" / args.run_name / "logs_shards"
    logs_dirs = sorted(
        str(p.relative_to(ROOT))
        for p in logs_base.glob("shard_*")
        if p.is_dir()
    )
    if not logs_dirs:
        raise SystemExit(f"no shard log directories found under {logs_base}")
    cmd = [
        sys.executable,
        "scripts/run_freqduet_external_baselines.py",
        "--variants", ",".join(variants),
        "--seeds", ",".join(str(x) for x in seeds),
        "--aggregate-only",
        "--aggregate-logs-dirs", ",".join(logs_dirs),
        "--last-k", str(args.last_k),
        "--out-dir", args.out_dir or f"results_freqduet/{args.run_name}/combined_summary",
    ]
    if args.configs_file:
        cmd.extend(["--configs-file", args.configs_file])
    else:
        cmd.extend(["--configs", ",".join(configs)])
    run(cmd, dry_run=args.dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--configs", default=None)
    parser.add_argument("--configs-file", default=None)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--last-k", type=int, default=50)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--jump-host", default=DEFAULT_JUMP)
    parser.add_argument("--remote-root", default=str(REMOTE_ROOT))
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.configs_file:
        configs = parse_csv_file(args.configs_file, str)
    elif args.configs:
        configs = parse_csv(args.configs, str)
    else:
        raise SystemExit("Either --configs or --configs-file is required")
    variants = parse_csv(args.variants, str)
    seeds = parse_csv(args.seeds, int)
    if not args.skip_sync:
        rsync_remote(args)
    if not args.skip_aggregate:
        aggregate(args, configs, variants, seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
