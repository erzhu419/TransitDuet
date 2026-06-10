#!/usr/bin/env python3
"""Generate and submit the broad FreqDuet generalization matrix to scheduleurm.

The matrix is intentionally scheduler-direct: shards are hard-pinned to
node001-node006 so they appear in tui-top as normal scheduleurm tasks instead of
being auto-routed through the zhengliang-hpc Slurm backend.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = Path("/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py")
REMOTE_ROOT = Path("/home/zhengliang01/scheduleurm_work/TransitDuet/FreqDuet/freqduet")
REMOTE_PYTHON = Path("/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python")
CONFIG_DIR = ROOT / "configs_freqduet" / "paper_generalization"
DEFAULT_SEEDS = [
    7, 11, 17, 23, 31, 37, 42, 43, 53, 61,
    71, 83, 97, 109, 123, 127, 149, 456, 789, 2026,
]
DEFAULT_NODES = ["node001", "node002", "node003", "node004", "node005", "node006"]
CPU_JUSTIFICATION = (
    "FreqDuet DSAC/SUMO-style transit experiments are intentionally CPU-only on "
    "zhengliang-hpc CPU nodes, using an isolated conda environment and no GPU kernels."
)


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    env: dict


METHOD_PARENTS = {
    "main": "../F_freqduet_terminal_main_hiro.yaml",
    "nofreq": "../F_freqduet_terminal_final_nofreq_hiro.yaml",
    "rawhistory": "../F_freqduet_terminal_final_rawhistory_hiro.yaml",
    "allfreq": "../F_freqduet_terminal_final_allfreq_hiro.yaml",
    "nopromotion": "../F_freqduet_terminal_final_nopromotion_hiro.yaml",
    "noleakage": "../F_freqduet_terminal_final_noleakage_hiro.yaml",
    "promenergy06": "../F_freqduet_terminal_main_hiro.yaml",
    "promenergy07": "../F_freqduet_terminal_main_hiro.yaml",
    "promenergy08": "../F_freqduet_terminal_main_hiro.yaml",
    "histaux3": "../F_freqduet_terminal_main_histaux3_hiro.yaml",
    "histaux6": "../F_freqduet_terminal_main_histaux6_hiro.yaml",
    "histaux6eg05": "../F_freqduet_terminal_main_histaux6eg05_hiro.yaml",
    "histaux6eg06": "../F_freqduet_terminal_main_histaux6eg06_hiro.yaml",
    "histaux6eg06upper": "../F_freqduet_terminal_main_histaux6eg06upper_hiro.yaml",
}

DEFAULT_METHODS = [
    "main",
    "nofreq",
    "rawhistory",
    "allfreq",
    "nopromotion",
    "noleakage",
]

METHOD_OVERRIDES = {
    "promenergy06": {
        "frequency": {
            "promotion": {
                "adapt_high_energy_min": 0.06,
            },
        },
    },
    "promenergy07": {
        "frequency": {
            "promotion": {
                "adapt_high_energy_min": 0.07,
            },
        },
    },
    "promenergy08": {
        "frequency": {
            "promotion": {
                "adapt_high_energy_min": 0.08,
            },
        },
    },
}

SCENARIOS = {
    "noise10": Scenario(
        key="noise10",
        label="mild demand noise",
        env={"demand_noise": 0.10},
    ),
    "noise20": Scenario(
        key="noise20",
        label="moderate demand noise",
        env={"demand_noise": 0.20},
    ),
    "noise40": Scenario(
        key="noise40",
        label="strong demand noise",
        env={"demand_noise": 0.40},
    ),
    "od20": Scenario(
        key="od20",
        label="mild OD multiplier shift",
        env={"demand_noise": 0.15, "od_noise": 0.20, "od_noise_clip": [0.5, 1.6]},
    ),
    "od50": Scenario(
        key="od50",
        label="strong OD multiplier shift",
        env={"demand_noise": 0.15, "od_noise": 0.50, "od_noise_clip": [0.3, 2.0]},
    ),
    "rush_early": Scenario(
        key="rush_early",
        label="early peak lookup shift",
        env={
            "demand_noise": 0.0,
            "peak_shift_choices": [-2, -1],
            "peak_shift_probs": [0.5, 0.5],
        },
    ),
    "rush_late": Scenario(
        key="rush_late",
        label="late peak lookup shift",
        env={
            "demand_noise": 0.0,
            "peak_shift_choices": [1, 2],
            "peak_shift_probs": [0.5, 0.5],
        },
    ),
    "rush_extreme": Scenario(
        key="rush_extreme",
        label="extreme early/late peak lookup shift",
        env={
            "demand_noise": 0.0,
            "peak_shift_choices": [-3, 3],
            "peak_shift_probs": [0.5, 0.5],
        },
    ),
}


def parse_csv(value: str, valid: Iterable[str] | None = None) -> list[str]:
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    if valid is not None:
        valid_set = set(valid)
        bad = sorted(set(items) - valid_set)
        if bad:
            raise SystemExit(f"unknown item(s): {', '.join(bad)}")
    return items


def config_name(scenario_key: str, method: str) -> str:
    return f"F_freqduet_broad_{scenario_key}_{method}_hiro"


def deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def write_configs(scenarios: list[str], methods: list[str]) -> list[str]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for scenario_key in scenarios:
        scenario = SCENARIOS[scenario_key]
        for method in methods:
            name = config_name(scenario_key, method)
            payload = {
                "_extends": METHOD_PARENTS[method],
                "_name": name,
                "_paper_generalization": {
                    "scenario": scenario.key,
                    "label": scenario.label,
                    "method": method,
                },
                "env": scenario.env,
            }
            deep_merge(payload, METHOD_OVERRIDES.get(method, {}))
            path = CONFIG_DIR / f"{name}.yaml"
            with path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(payload, f, sort_keys=False)
            names.append(name)
    return names


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


def build_inner_cmd(args: argparse.Namespace, configs_csv: str,
                    seeds_csv: str, start: int, end: int,
                    logs_dir: str, out_dir: str) -> str:
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
        "scripts/run_freqduet_ablation.py",
        "--configs", configs_csv,
        "--seeds", seeds_csv,
        "--episodes", str(args.episodes),
        "--last-k", str(args.last_k),
        "--workers", str(args.workers),
        "--worker-threads", "1",
        "--upper-warmup-eps", str(args.upper_warmup_eps),
        "--logs-dir", logs_dir,
        "--out-dir", out_dir,
        "--job-start", str(start),
        "--job-end", str(end),
        "--skip-existing",
        "--no-aggregate",
    ]
    return " ".join(env_bits + [shlex.join(cmd)])


def submit_shard(args: argparse.Namespace, configs_csv: str, seeds_csv: str,
                 shard_index: int, num_shards: int, start: int, end: int,
                 node: str) -> None:
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
        f"FreqDuet broad generalization {args.run_name} shard {shard_index + 1}/{num_shards}",
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


def print_aggregate_hint(run_name: str, configs: list[str], seeds: list[int],
                         last_k: int) -> None:
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
    parser.add_argument("--scenarios", default=",".join(SCENARIOS.keys()))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--last-k", type=int, default=50)
    parser.add_argument("--upper-warmup-eps", type=int, default=10)
    parser.add_argument("--run-name", default="broad_generalization_ep100_wu10")
    parser.add_argument("--shard-size", type=int, default=30)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--cpu", type=int, default=30)
    parser.add_argument("--ram-mb", type=int, default=32768)
    parser.add_argument("--nodes", default=",".join(DEFAULT_NODES))
    parser.add_argument("--priority", choices=["low", "normal", "high"], default="normal")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--allow-duplicate", action="store_true")
    args = parser.parse_args()

    scenarios = parse_csv(args.scenarios, SCENARIOS.keys())
    methods = parse_csv(args.methods, METHOD_PARENTS.keys())
    nodes = parse_csv(args.nodes)
    seeds = [int(x) for x in parse_csv(args.seeds)]

    configs = write_configs(scenarios, methods)
    total_jobs = len(configs) * len(seeds)
    ranges = shard_ranges(total_jobs, args.shard_size)
    print(
        f"Generated {len(configs)} configs in {CONFIG_DIR.relative_to(ROOT)}; "
        f"total jobs={total_jobs}; shards={len(ranges)}; nodes={','.join(nodes)}"
    )
    print_aggregate_hint(args.run_name, configs, seeds, args.last_k)
    if args.generate_only:
        return

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
