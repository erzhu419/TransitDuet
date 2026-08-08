#!/usr/bin/env python3
"""Submit or merge the source-bound Freq-HRL v7.3.2 budget ladder."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.trading import (  # noqa: E402
    full_method_budget_plan_v732 as plan,
)
from freq_hrl.experiments.trading import (  # noqa: E402
    full_method_budget_validation_v732 as validation,
)
from freq_hrl.experiments.trading import full_method_hpo_v7 as hpo  # noqa: E402
from scripts.submit_hyperparameter_pilot_scheduleurm import (  # noqa: E402
    LINUX_CPU_NODES,
    SCHEDULER,
    STAGE_EXCLUDES,
    default_python_executable,
    execute,
    execute_bulk,
    parse_csv,
    source_identity,
)


NODE_CPU_CAPACITY = 192
POOL_CPU_CAPACITY = NODE_CPU_CAPACITY * len(LINUX_CPU_NODES)
HPO_MODULE = "freq_hrl.experiments.trading.full_method_hpo_v7"
SIGNATURE_VERSION = "budget-v7-3-2-source-bound"
SUBMIT_SCRIPT_PATH = Path(__file__).resolve()


def experiment_cells(
    budgets: list[int],
) -> list[tuple[int, str, str, int]]:
    return plan.experiment_cells(budgets)


def cell_relative_dir(
    run_name: str,
    budget: int,
    variant_id: str,
    candidate_id: str,
    replicate_seed: int,
) -> Path:
    return (
        Path("results") / run_name / "budgets" / f"iterations_{int(budget)}"
        / "cells" / variant_id / candidate_id
        / f"replicate_{int(replicate_seed)}"
    )


def build_training_command(
    args: argparse.Namespace,
    *,
    budget: int,
    variant_id: str,
    candidate_id: str,
    replicate_seed: int,
    output_dir: Path,
) -> str:
    command = [
        str(args.python_executable), "-u", "-m", HPO_MODULE,
        "--candidate-id", candidate_id,
        "--variant-id", variant_id,
        "--training-replicate-seed", str(replicate_seed),
        "--train-seeds", *(str(seed) for seed in args.train_seeds),
        "--promotion-calibration-seeds",
        *(str(seed) for seed in args.promotion_calibration_seeds),
        "--checkpoint-validation-seeds",
        *(str(seed) for seed in args.checkpoint_validation_seeds),
        "--tuning-validation-seeds",
        *(str(seed) for seed in args.tuning_validation_seeds),
        "--steps", str(args.steps),
        "--assets", str(args.assets),
        "--iterations", str(budget),
        "--code-revision", str(args.code_revision),
        "--source-manifest-sha256", str(args.source_manifest_sha256),
        "--output-dir", str(output_dir),
    ]
    env = [
        "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=.",
        "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1", "TORCH_NUM_THREADS=1",
        "CUDA_VISIBLE_DEVICES=",
    ]
    run = " ".join([*env, shlex.join(command)]) + " && echo DONE"
    return f"cd .. && {run}" if args.launch_subdir == "scripts" else run


def _ram_mb(args: argparse.Namespace, variant_id: str) -> int:
    return int(
        args.offpolicy_ram_mb
        if hpo.VARIANTS_BY_ID[variant_id].trainer_family == "offpolicy"
        else args.ppo_ram_mb
    )


def build_scheduler_spec(
    args: argparse.Namespace,
    *,
    budget: int,
    variant_id: str,
    candidate_id: str,
    replicate_seed: int,
) -> dict[str, object]:
    relative = cell_relative_dir(
        args.run_name, budget, variant_id, candidate_id, replicate_seed
    )
    absolute = ROOT / relative
    return {
        "project": str(args.project),
        "description": (
            f"Freq-HRL v7.3.2 budget {budget} {variant_id} "
            f"{candidate_id} replicate {replicate_seed}"
        ),
        "cmd": build_training_command(
            args,
            budget=budget,
            variant_id=variant_id,
            candidate_id=candidate_id,
            replicate_seed=replicate_seed,
            output_dir=relative,
        ),
        "cwd": str(ROOT / str(args.launch_subdir)),
        "signature": (
            f"Freq-HRL/{SIGNATURE_VERSION}/{plan.plan_sha256()[:16]}/"
            f"{args.run_name}/iterations-{budget}/{variant_id}/"
            f"{candidate_id}/rep-{replicate_seed}"
        ),
        "resource_family": (
            f"Freq-HRL/{SIGNATURE_VERSION}/"
            f"{hpo.VARIANTS_BY_ID[variant_id].trainer_family}/cell"
        ),
        "vram": 0,
        "ram_mb": _ram_mb(args, variant_id),
        "cpu": 1,
        "priority": str(args.priority),
        "ckpt_dir": str(absolute),
        "ckpt_glob": "checkpoint.pt",
        "result_dir": str(absolute),
        "local_result_dir": str(absolute),
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Budget cells are independent one-core training replicates; the "
            "iteration ladder is a source-bound development protocol."
        ),
        "reroute_on_node_down": True,
        "node_down_requeue_s": 600,
        "allowed_nodes": list(args.nodes),
        "require_node": None,
        "stage_excludes": list(STAGE_EXCLUDES),
        "stage_input_paths": list(args.stage_input_paths),
        "skip_launch_staging": bool(args.skip_launch_staging),
        "allow_duplicate": bool(args.allow_duplicate),
    }


def expected_cell_dirs(args: argparse.Namespace) -> list[Path]:
    return [
        ROOT / cell_relative_dir(args.run_name, *cell)
        for cell in experiment_cells(args.budgets)
    ]


def merge_results(args: argparse.Namespace) -> None:
    directories = expected_cell_dirs(args)
    missing = [
        path for path in directories
        if not (path / "cell_summary.json").exists()
    ]
    if missing:
        raise SystemExit(
            f"cannot merge budget ladder: {len(missing)} cells missing; "
            f"first={missing[0]}"
        )
    payload = validation.summarize_budget_cells(
        directories, expected_budgets=args.budgets
    )
    output = ROOT / "results" / args.run_name / "merged"
    validation.write_budget_decision(output, payload)
    print(
        f"merged {len(directories)} budget cells into {output}; "
        f"status={payload['status']} selected={payload['selected_iterations']}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--budgets", default=",".join(map(str, plan.MANDATORY_BUDGETS))
    )
    parser.add_argument(
        "--train-seeds", default=",".join(map(str, hpo.DEFAULT_TRAIN_SEEDS))
    )
    parser.add_argument(
        "--promotion-calibration-seeds",
        default=",".join(map(str, hpo.DEFAULT_PROMOTION_CALIBRATION_SEEDS)),
    )
    parser.add_argument(
        "--checkpoint-validation-seeds",
        default=",".join(map(str, hpo.DEFAULT_CHECKPOINT_VALIDATION_SEEDS)),
    )
    parser.add_argument(
        "--tuning-validation-seeds",
        default=",".join(map(str, hpo.DEFAULT_TUNING_SEEDS)),
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--nodes", default=",".join(LINUX_CPU_NODES))
    parser.add_argument("--python-executable", default="")
    parser.add_argument("--launch-subdir", choices=(".", "scripts"), default=".")
    parser.add_argument("--source-code-revision", default="")
    parser.add_argument("--project", default="Freq-HRL-v7.3.2-Budget")
    parser.add_argument("--ppo-ram-mb", type=int, default=1024)
    parser.add_argument("--offpolicy-ram-mb", type=int, default=1536)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="normal"
    )
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--skip-complete-cells", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-duplicate", action="store_true")
    parser.add_argument("--skip-launch-staging", action="store_true")
    parser.add_argument("--stage-input-path", action="append", default=[])
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.budgets = parse_csv(args.budgets, int)
    args.train_seeds = parse_csv(args.train_seeds, int)
    args.promotion_calibration_seeds = parse_csv(
        args.promotion_calibration_seeds, int
    )
    args.checkpoint_validation_seeds = parse_csv(
        args.checkpoint_validation_seeds, int
    )
    args.tuning_validation_seeds = parse_csv(
        args.tuning_validation_seeds, int
    )
    args.nodes = parse_csv(args.nodes)
    args.stage_input_paths = [
        str(Path(path).expanduser().resolve())
        for path in args.stage_input_path if str(path).strip()
    ]
    if tuple(sorted(set(args.budgets))) != tuple(args.budgets):
        raise SystemExit("budget ladder must be unique and increasing")
    if not set(args.budgets).issubset(plan.BUDGET_LADDER):
        raise SystemExit("unregistered iteration budget")
    unknown_nodes = sorted(set(args.nodes) - set(LINUX_CPU_NODES))
    if unknown_nodes:
        raise SystemExit(f"invalid budget nodes: {unknown_nodes}")
    if not args.python_executable.strip():
        args.python_executable = default_python_executable(args.nodes)
    return args


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    if args.merge_only:
        merge_results(args)
        return
    try:
        args.code_revision, args.source_manifest_sha256 = source_identity(
            args.source_code_revision
        )
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"cannot bind v7.3.2 budget source: {exc}") from exc
    cells = experiment_cells(args.budgets)
    if args.skip_complete_cells:
        cells = [
            cell for cell in cells
            if not (
                ROOT / cell_relative_dir(args.run_name, *cell)
                / "cell_summary.json"
            ).exists()
        ]
    if args.max_cells > 0:
        cells = cells[:args.max_cells]
    if not cells:
        print("no v7.3.2 budget cells require submission")
        return
    print(
        f"run={args.run_name} cells={len(cells)} budgets={args.budgets} "
        f"nodes={','.join(args.nodes)} pool_cores={POOL_CPU_CAPACITY} "
        f"plan={plan.plan_sha256()[:16]}",
        flush=True,
    )
    execute_bulk(
        [
            build_scheduler_spec(
                args,
                budget=budget,
                variant_id=variant,
                candidate_id=candidate,
                replicate_seed=seed,
            )
            for budget, variant, candidate, seed in cells
        ],
        dry_run=bool(args.dry_run),
        intent_label=f"Freq-HRL v7.3.2 budget ladder {args.run_name}",
    )
    if args.dispatch and not args.dry_run:
        execute([sys.executable, str(SCHEDULER), "dispatch"], dry_run=False)
    print(
        "merge after all summaries sync: " + shlex.join([
            sys.executable,
            str(SUBMIT_SCRIPT_PATH),
            "--run-name", args.run_name,
            "--budgets", ",".join(map(str, args.budgets)),
            "--merge-only",
        ])
    )


if __name__ == "__main__":
    main()
