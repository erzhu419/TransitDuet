#!/usr/bin/env python3
"""Submit or merge the confirmatory Freq-HRL learned-baseline matrix."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.trading.strong_learned_baseline_validation import (  # noqa: E402
    ALL_POLICY_MODES,
    DEFAULT_EVAL_SEEDS,
    DEFAULT_OPTIMIZER_SEEDS,
    DEFAULT_POLICY_MODES,
    DEFAULT_ROLLOUT_SEED_ROOTS,
    DEFAULT_SCENARIOS,
    DEFAULT_VALIDATION_SEEDS,
    SCENARIOS,
    merge_strong_learned_baseline_shards,
    write_outputs,
)


SCHEDULER = Path("/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py")
DEFAULT_NODES = ("jtl110cpu", "jtl110cpu2")
CPU_JUSTIFICATION = (
    "The synthetic trading environment and small actor-critic models are "
    "CPU-bound; one independent training replicate runs single-threaded."
)
STAGE_EXCLUDES = (
    "results",
    "data",
    "freq_transitduet",
    "scheduler_results",
    "**/__pycache__",
)


def parse_csv(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def experiment_cells(
    scenarios: list[str],
    policy_modes: list[str],
    optimizer_seeds: list[int],
) -> list[tuple[str, str, int]]:
    return [
        (scenario, mode, int(seed))
        for scenario in scenarios
        for mode in policy_modes
        for seed in optimizer_seeds
    ]


def cell_relative_dir(
    run_name: str,
    scenario: str,
    mode: str,
    replicate_seed: int,
) -> Path:
    return (
        Path("results")
        / run_name
        / "cells"
        / scenario
        / mode
        / f"replicate_{int(replicate_seed)}"
    )


def build_training_command(
    args: argparse.Namespace,
    *,
    scenario: str,
    mode: str,
    replicate_seed: int,
    output_dir: Path,
) -> str:
    command = [
        "python3",
        "-u",
        "-m",
        "freq_hrl.experiments.trading.strong_learned_baseline_validation",
        "--scenarios",
        scenario,
        "--policy-modes",
        mode,
        "--train-seeds",
        *(str(seed) for seed in args.train_seeds),
        "--validation-seeds",
        *(str(seed) for seed in args.validation_seeds),
        "--eval-seeds",
        *(str(seed) for seed in args.eval_seeds),
        "--steps",
        str(args.steps),
        "--assets",
        str(args.assets),
        "--iterations",
        str(args.iterations),
        "--optimizer-seeds",
        str(replicate_seed),
        "--min-pairs",
        str(args.min_pairs),
        "--ppo-hidden-dim",
        str(args.ppo_hidden_dim),
        "--ppo-learning-rate",
        str(args.ppo_learning_rate),
        "--ppo-epochs",
        str(args.ppo_epochs),
        "--ppo-minibatch-size",
        str(args.ppo_minibatch_size),
        "--ppo-init-log-std",
        str(args.ppo_init_log_std),
        "--offpolicy-hidden-dim",
        str(args.offpolicy_hidden_dim),
        "--offpolicy-replay-capacity",
        str(args.offpolicy_replay_capacity),
        "--offpolicy-warmup-steps",
        str(args.offpolicy_warmup_steps),
        "--offpolicy-batch-size",
        str(args.offpolicy_batch_size),
        "--offpolicy-updates-per-step",
        str(args.offpolicy_updates_per_step),
        "--output-dir",
        str(output_dir),
    ]
    env = [
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONPATH=.",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "TORCH_NUM_THREADS=1",
    ]
    return " ".join([*env, shlex.join(command)])


def build_scheduler_command(
    args: argparse.Namespace,
    *,
    scenario: str,
    mode: str,
    replicate_seed: int,
) -> list[str]:
    relative_dir = cell_relative_dir(
        args.run_name, scenario, mode, replicate_seed
    )
    absolute_dir = ROOT / relative_dir
    command = [
        sys.executable,
        str(SCHEDULER),
        "submit",
        "--project",
        "Freq-HRL",
        "--description",
        f"Freq-HRL strong v2 {scenario} {mode} replicate {replicate_seed}",
        "--cmd",
        build_training_command(
            args,
            scenario=scenario,
            mode=mode,
            replicate_seed=replicate_seed,
            output_dir=relative_dir,
        ),
        "--cwd",
        str(ROOT),
        "--signature",
        f"Freq-HRL/strong-v2/{args.run_name}/{scenario}/{mode}/rep-{replicate_seed}",
        "--resource-family",
        "Freq-HRL/strong-v2/single-cell",
        "--vram",
        "0",
        "--ram-mb",
        str(args.ram_mb),
        "--cpu",
        str(args.cpu),
        "--priority",
        args.priority,
        "--ckpt-dir",
        str(absolute_dir / "checkpoints"),
        "--result-dir",
        str(absolute_dir),
        "--local-result-dir",
        str(absolute_dir),
        "--allow-cpu-training",
        "--cpu-training-justification",
        CPU_JUSTIFICATION,
        "--allow-no-resume",
        "--reroute-on-node-down",
        "--node-down-requeue-s",
        "600",
    ]
    for node in args.nodes:
        command.extend(["--allowed-node", node])
    for excluded in STAGE_EXCLUDES:
        command.extend(["--stage-exclude", excluded])
    if args.skip_launch_staging:
        command.append("--skip-launch-staging")
    if args.allow_duplicate:
        command.append("--allow-duplicate")
    return command


def execute(command: list[str], *, dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if dry_run:
        return
    process = subprocess.run(command, text=True, capture_output=True)
    output = (process.stdout or "") + (process.stderr or "")
    if process.returncode != 0:
        lowered = output.lower()
        if "duplicate" not in lowered and "already queued" not in lowered:
            if output.strip():
                print(output.strip(), file=sys.stderr)
            process.check_returncode()
    if output.strip():
        print(output.strip(), flush=True)


def expected_cell_dirs(args: argparse.Namespace) -> list[Path]:
    return [
        ROOT / cell_relative_dir(args.run_name, scenario, mode, seed)
        for scenario, mode, seed in experiment_cells(
            args.scenarios, args.policy_modes, args.optimizer_seeds
        )
    ]


def merge_results(args: argparse.Namespace) -> None:
    directories = expected_cell_dirs(args)
    missing = [path for path in directories if not (path / "per_seed.csv").exists()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise SystemExit(
            f"cannot merge: {len(missing)} expected cells are missing\n{preview}"
        )
    payload = merge_strong_learned_baseline_shards(
        directories, min_pairs=int(args.min_pairs)
    )
    output_dir = ROOT / "results" / args.run_name / "merged"
    write_outputs(output_dir, payload)
    print(f"merged {len(directories)} cells into {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--policy-modes", default=",".join(DEFAULT_POLICY_MODES))
    parser.add_argument(
        "--optimizer-seeds", default=",".join(map(str, DEFAULT_OPTIMIZER_SEEDS))
    )
    parser.add_argument(
        "--train-seeds", default=",".join(map(str, DEFAULT_ROLLOUT_SEED_ROOTS))
    )
    parser.add_argument(
        "--validation-seeds", default=",".join(map(str, DEFAULT_VALIDATION_SEEDS))
    )
    parser.add_argument("--eval-seeds", default=",".join(map(str, DEFAULT_EVAL_SEEDS)))
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--min-pairs", type=int, default=10)
    parser.add_argument("--ppo-hidden-dim", type=int, default=64)
    parser.add_argument("--ppo-learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch-size", type=int, default=512)
    parser.add_argument("--ppo-init-log-std", type=float, default=-1.0)
    parser.add_argument("--offpolicy-hidden-dim", type=int, default=64)
    parser.add_argument("--offpolicy-replay-capacity", type=int, default=100_000)
    parser.add_argument("--offpolicy-warmup-steps", type=int, default=2048)
    parser.add_argument("--offpolicy-batch-size", type=int, default=64)
    parser.add_argument("--offpolicy-updates-per-step", type=int, default=1)
    parser.add_argument("--nodes", default=",".join(DEFAULT_NODES))
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--ram-mb", type=int, default=2048)
    parser.add_argument(
        "--priority", choices=["low", "normal", "high"], default="normal"
    )
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-duplicate", action="store_true")
    parser.add_argument("--skip-launch-staging", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.scenarios = parse_csv(args.scenarios)
    args.policy_modes = parse_csv(args.policy_modes)
    args.optimizer_seeds = parse_csv(args.optimizer_seeds, int)
    args.train_seeds = parse_csv(args.train_seeds, int)
    args.validation_seeds = parse_csv(args.validation_seeds, int)
    args.eval_seeds = parse_csv(args.eval_seeds, int)
    args.nodes = parse_csv(args.nodes)
    unknown_modes = sorted(set(args.policy_modes) - set(ALL_POLICY_MODES))
    unknown_scenarios = sorted(set(args.scenarios) - set(SCENARIOS))
    unknown_nodes = sorted(set(args.nodes) - set(DEFAULT_NODES))
    if unknown_modes or unknown_scenarios or unknown_nodes:
        raise SystemExit(
            "invalid matrix selection: "
            f"modes={unknown_modes}, scenarios={unknown_scenarios}, nodes={unknown_nodes}"
        )
    if args.smoke:
        args.scenarios = ["persistent_shift"]
        args.policy_modes = ["freq_hrl"]
        args.optimizer_seeds = [args.optimizer_seeds[0]]
        args.train_seeds = [args.train_seeds[0]]
        args.validation_seeds = [args.validation_seeds[0]]
        args.eval_seeds = [args.eval_seeds[0]]
        args.steps = min(int(args.steps), 32)
        args.iterations = 1
        args.min_pairs = 1
    return args


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    if args.merge_only:
        merge_results(args)
        return
    cells = experiment_cells(
        args.scenarios, args.policy_modes, args.optimizer_seeds
    )
    if int(args.max_cells) > 0:
        cells = cells[: int(args.max_cells)]
    print(
        f"run={args.run_name} cells={len(cells)} nodes={','.join(args.nodes)} "
        f"iterations={args.iterations} steps={args.steps}",
        flush=True,
    )
    for scenario, mode, seed in cells:
        execute(
            build_scheduler_command(
                args,
                scenario=scenario,
                mode=mode,
                replicate_seed=seed,
            ),
            dry_run=bool(args.dry_run),
        )
    if args.dispatch and not args.dry_run:
        execute(
            [sys.executable, str(SCHEDULER), "dispatch"], dry_run=False
        )
    print(
        "merge after all cells sync: "
        + shlex.join([
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-name",
            args.run_name,
            "--scenarios",
            ",".join(args.scenarios),
            "--policy-modes",
            ",".join(args.policy_modes),
            "--optimizer-seeds",
            ",".join(map(str, args.optimizer_seeds)),
            "--merge-only",
        ])
    )


if __name__ == "__main__":
    main()
