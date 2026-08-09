#!/usr/bin/env python3
"""Submit FreqDuet external baseline matrices to scheduleurm CPU nodes."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from submit_freqduet_config_matrix_scheduleurm import (
    CPU_JUSTIFICATION,
    DEFAULT_NODES,
    DEFAULT_SEEDS,
    REMOTE_PYTHON,
    REMOTE_ROOT as LEGACY_REMOTE_ROOT,
    SCHEDULER,
    parse_csv,
    parse_csv_file,
    run_command,
    shard_ranges,
)
from submit_freqduet_protocol_v2_scheduleurm import (
    git_output,
    preflight_source,
    protocol_label,
)


ROOT = Path(__file__).resolve().parents[1]
LOCAL_WORKSPACE_ROOT = Path("/home/erzhu419/mine_code")
REMOTE_WORKSPACE_ROOT = Path("/home/zhengliang01/scheduleurm_work")
try:
    REMOTE_ROOT = REMOTE_WORKSPACE_ROOT / ROOT.relative_to(LOCAL_WORKSPACE_ROOT)
except ValueError:
    REMOTE_ROOT = LEGACY_REMOTE_ROOT
DEFAULT_VARIANTS = ["fixed_headway", "rule_holding", "rule_mpc"]


def build_inner_cmd(
    args: argparse.Namespace,
    configs_csv: str,
    variants_csv: str,
    seeds_csv: str,
    start: int,
    end: int,
    logs_dir: str,
    out_dir: str,
    source_commit: str,
    source_branch: str,
    source_tracked_dirty: bool,
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
        f"FREQDUET_SOURCE_COMMIT={shlex.quote(source_commit)}",
        f"FREQDUET_SOURCE_BRANCH={shlex.quote(source_branch)}",
        "FREQDUET_SOURCE_TRACKED_DIRTY="
        f"{int(source_tracked_dirty)}",
    ]
    cmd = [
        str(REMOTE_PYTHON),
        "-u",
        "scripts/run_freqduet_external_baselines.py",
        "--variants", variants_csv,
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
    if args.configs_file:
        cmd.extend(["--configs-file", args.configs_file])
    else:
        cmd.extend(["--configs", configs_csv])
    if args.headway_policy_csv:
        cmd.extend(["--headway-policy-csv", args.headway_policy_csv])
    if args.policy_default_headway is not None:
        cmd.extend(["--policy-default-headway", str(args.policy_default_headway)])
    if args.direct_scenario_seeds:
        cmd.append("--direct-scenario-seeds")
    return " ".join(env_bits + [shlex.join(cmd)])


def submit_shard(
    args: argparse.Namespace,
    configs_csv: str,
    variants_csv: str,
    seeds_csv: str,
    shard_index: int,
    num_shards: int,
    start: int,
    end: int,
    node: str,
    source_commit: str,
    source_branch: str,
    source_tracked_dirty: bool,
) -> None:
    shard_id = f"{start:04d}_{end:04d}"
    result_base = f"results_freqduet/{args.run_name}"
    logs_dir = f"{result_base}/logs_shards/shard_{shard_id}"
    out_dir = f"{result_base}/shard_summaries/shard_{shard_id}"
    remote_result_dir = str(Path(args.remote_root) / logs_dir)
    local_result_dir = str(ROOT / logs_dir)
    cmd = build_inner_cmd(
        args,
        configs_csv,
        variants_csv,
        seeds_csv,
        start,
        end,
        logs_dir,
        out_dir,
        source_commit,
        source_branch,
        source_tracked_dirty,
    )
    command = [
        sys.executable, str(SCHEDULER), "submit",
        "--project", "FreqDuet",
        "--description",
        f"FreqDuet external baselines {args.run_name} shard {shard_index + 1}/{num_shards}",
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


def print_aggregate_hint(
    run_name: str,
    configs: list[str],
    variants: list[str],
    seeds: list[int],
    last_k: int,
    configs_file: str | None = None,
) -> None:
    logs_expr = (
        f"$(find results_freqduet/{run_name}/logs_shards "
        "-mindepth 1 -maxdepth 1 -type d | sort | paste -sd, -)"
    )
    print("\nAggregate after result sync:")
    config_arg = (
        f"--configs-file {shlex.quote(configs_file)}"
        if configs_file else f"--configs {shlex.quote(','.join(configs))}"
    )
    print(
        "  python3 scripts/run_freqduet_external_baselines.py "
        f"{config_arg} --variants {shlex.quote(','.join(variants))} "
        f"--seeds {shlex.quote(','.join(str(s) for s in seeds))} "
        "--aggregate-only "
        f"--aggregate-logs-dirs \"{logs_expr}\" "
        f"--last-k {int(last_k)} "
        f"--out-dir results_freqduet/{run_name}/combined_summary"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=None)
    parser.add_argument("--configs-file", default=None,
                        help="file containing config names; same path must exist on remote cwd")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--last-k", type=int, default=50)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--headway-policy-csv", default=None,
                        help="CSV mapping route policy variants/configs[/seeds] to target headways")
    parser.add_argument("--policy-default-headway", type=float, default=None,
                        help="fallback target for route policy variants missing from the CSV")
    parser.add_argument("--shard-size", type=int, default=30)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--cpu", type=int, default=30)
    parser.add_argument("--ram-mb", type=int, default=32768)
    parser.add_argument("--nodes", default=",".join(DEFAULT_NODES))
    parser.add_argument("--remote-root", default=str(REMOTE_ROOT))
    parser.add_argument("--direct-scenario-seeds", action="store_true")
    parser.add_argument("--require-clean-source", action="store_true")
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--priority", choices=["low", "normal", "high"], default="normal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--allow-duplicate", action="store_true")
    args = parser.parse_args()

    if args.configs_file:
        configs = parse_csv_file(args.configs_file, str)
    elif args.configs:
        configs = parse_csv(args.configs, str)
    else:
        raise SystemExit("Either --configs or --configs-file is required")
    variants = parse_csv(args.variants, str)
    seeds = parse_csv(args.seeds, int)
    nodes = parse_csv(args.nodes, str)
    if args.direct_scenario_seeds and int(args.episodes) != 1:
        raise SystemExit("--direct-scenario-seeds requires --episodes 1")
    protocol = protocol_label(configs)
    try:
        commit = preflight_source(
            configs, protocol, args.require_clean_source,
            args.expected_commit)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    total_jobs = len(configs) * len(variants) * len(seeds)
    ranges = shard_ranges(total_jobs, args.shard_size)
    print(
        f"Submitting external baseline run={args.run_name}; configs={len(configs)}; "
        f"variants={len(variants)}; seeds={len(seeds)}; total jobs={total_jobs}; "
        f"shards={len(ranges)}; nodes={','.join(nodes)}"
    )
    print(
        f"source_commit={commit} local_root={ROOT} "
        f"remote_root={Path(args.remote_root)}")
    source_branch = git_output("rev-parse", "--abbrev-ref", "HEAD")
    source_tracked_dirty = bool(git_output(
        "status", "--porcelain", "--untracked-files=no"))
    print_aggregate_hint(
        args.run_name, configs, variants, seeds, args.last_k,
        configs_file=args.configs_file)

    configs_csv = ",".join(configs)
    variants_csv = ",".join(variants)
    seeds_csv = ",".join(str(x) for x in seeds)
    for i, (start, end) in enumerate(ranges):
        submit_shard(
            args=args,
            configs_csv=configs_csv,
            variants_csv=variants_csv,
            seeds_csv=seeds_csv,
            shard_index=i,
            num_shards=len(ranges),
            start=start,
            end=end,
            node=nodes[i % len(nodes)],
            source_commit=commit,
            source_branch=source_branch,
            source_tracked_dirty=source_tracked_dirty,
        )

    if args.dispatch and not args.dry_run:
        run_command([sys.executable, str(SCHEDULER), "dispatch"], False)


if __name__ == "__main__":
    main()
