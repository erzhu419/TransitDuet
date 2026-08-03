#!/usr/bin/env python3
"""Submit, preflight, or merge full-method HPO through scheduleurm."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.trading.full_method_hpo import (  # noqa: E402
    ALL_VARIANT_IDS,
    CANDIDATES_BY_ID,
    DEFAULT_PILOT_SCENARIOS,
    DEFAULT_TUNING_SEEDS,
    FULL_METHOD_TUNING_PROTOCOL_VERSION,
    VARIANTS_BY_ID,
    candidate_ids_for_variant,
    merge_hpo_cells,
    write_hpo_merge,
)
from freq_hrl.experiments.trading.strong_learned_baseline_validation import (  # noqa: E402
    DEFAULT_OPTIMIZER_SEEDS,
    DEFAULT_ROLLOUT_SEED_ROOTS,
    DEFAULT_SCENARIOS,
    DEFAULT_VALIDATION_SEEDS,
)
from scripts.submit_hyperparameter_pilot_scheduleurm import (  # noqa: E402
    DEFAULT_LINUX_PYTHON,
    LINUX_CPU_NODES,
    SCHEDULER,
    STAGE_EXCLUDES,
    default_python_executable,
    execute,
    execute_bulk,
    parse_csv,
    source_identity,
)


DEFAULT_NODES = LINUX_CPU_NODES
NODE_CPU_CAPACITY = 192
POOL_CPU_CAPACITY = NODE_CPU_CAPACITY * len(DEFAULT_NODES)
CPU_JUSTIFICATION = (
    "Full-method nested-validation cells are independent, CPU-bound, and "
    "explicitly single-threaded. scheduleurm dynamically packs them across "
    "the six-node 1152-core Linux CPU pool."
)


def experiment_cells(
    variant_ids: list[str],
    candidate_ids: list[str],
    scenarios: list[str],
    replicate_seeds: list[int],
) -> list[tuple[str, str, str, int]]:
    return [
        (variant_id, candidate_id, scenario, int(seed))
        for variant_id in variant_ids
        for candidate_id in candidate_ids
        if candidate_id in candidate_ids_for_variant(variant_id)
        for scenario in scenarios
        for seed in replicate_seeds
    ]


def cell_relative_dir(
    run_name: str,
    variant_id: str,
    candidate_id: str,
    scenario: str,
    replicate_seed: int,
) -> Path:
    return (
        Path("results")
        / run_name
        / "cells"
        / variant_id
        / candidate_id
        / scenario
        / f"replicate_{int(replicate_seed)}"
    )


def preflight_relative_dir(run_name: str, node: str) -> Path:
    return Path("results") / run_name / "environment_preflight" / str(node)


def build_training_command(
    args: argparse.Namespace,
    *,
    variant_id: str,
    candidate_id: str,
    scenario: str,
    replicate_seed: int,
    output_dir: Path,
) -> str:
    command = [
        str(args.python_executable),
        "-u",
        "-m",
        "freq_hrl.experiments.trading.full_method_hpo",
        "--candidate-id",
        str(candidate_id),
        "--variant-id",
        str(variant_id),
        "--scenario",
        str(scenario),
        "--training-replicate-seed",
        str(replicate_seed),
        "--train-seeds",
        *(str(seed) for seed in args.train_seeds),
        "--checkpoint-validation-seeds",
        *(str(seed) for seed in args.checkpoint_validation_seeds),
        "--tuning-validation-seeds",
        *(str(seed) for seed in args.tuning_validation_seeds),
        "--steps",
        str(args.steps),
        "--assets",
        str(args.assets),
        "--iterations",
        str(args.iterations),
        "--code-revision",
        str(args.code_revision),
        "--source-manifest-sha256",
        str(args.source_manifest_sha256),
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
        "CUDA_VISIBLE_DEVICES=",
    ]
    command_text = " ".join([*env, shlex.join(command)]) + " && echo DONE"
    if str(args.launch_subdir) == "scripts":
        return f"cd .. && {command_text}"
    return command_text


def build_preflight_command(
    args: argparse.Namespace,
    *,
    node: str,
    output_dir: Path,
) -> str:
    code = (
        "import json,os,platform;"
        "from pathlib import Path;"
        "import numpy,torch;"
        "from freq_hrl.experiments.trading.full_method_hpo import "
        "FULL_METHOD_TUNING_PROTOCOL_VERSION;"
        f"p=Path({str(output_dir)!r});p.mkdir(parents=True,exist_ok=True);"
        "payload={'status':'ready','node':platform.node(),"
        f"'requested_node':{str(node)!r},"
        "'python':platform.python_version(),'python_executable':os.sys.executable,"
        "'numpy':numpy.__version__,'torch':torch.__version__,"
        "'visible_cpu_count':os.cpu_count(),"
        "'torch_num_threads':torch.get_num_threads(),"
        "'protocol':FULL_METHOD_TUNING_PROTOCOL_VERSION};"
        "(p/'environment.json').write_text(json.dumps(payload,sort_keys=True));"
        "print(json.dumps(payload,sort_keys=True))"
    )
    command = [str(args.python_executable), "-c", code]
    env = [
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONPATH=.",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "TORCH_NUM_THREADS=1",
        "CUDA_VISIBLE_DEVICES=",
    ]
    command_text = " ".join([*env, shlex.join(command)]) + " && echo READY"
    if str(args.launch_subdir) == "scripts":
        return f"cd .. && {command_text}"
    return command_text


def _ram_mb(args: argparse.Namespace, variant_id: str) -> int:
    family = VARIANTS_BY_ID[variant_id].trainer_family
    return int(args.offpolicy_ram_mb if family == "offpolicy" else args.ppo_ram_mb)


def build_scheduler_spec(
    args: argparse.Namespace,
    *,
    variant_id: str,
    candidate_id: str,
    scenario: str,
    replicate_seed: int,
) -> dict[str, object]:
    relative_dir = cell_relative_dir(
        args.run_name,
        variant_id,
        candidate_id,
        scenario,
        replicate_seed,
    )
    absolute_dir = ROOT / relative_dir
    return {
        "project": str(args.project),
        "description": (
            f"Freq-HRL full HPO {variant_id} {candidate_id} "
            f"{scenario} replicate {replicate_seed}"
        ),
        "cmd": build_training_command(
            args,
            variant_id=variant_id,
            candidate_id=candidate_id,
            scenario=scenario,
            replicate_seed=replicate_seed,
            output_dir=relative_dir,
        ),
        "cwd": str(ROOT / str(args.launch_subdir)),
        "signature": (
            f"Freq-HRL/full-hpo-v2/{args.run_name}/{variant_id}/"
            f"{candidate_id}/{scenario}/rep-{replicate_seed}"
        ),
        "resource_family": (
            f"Freq-HRL/full-hpo-v2/{VARIANTS_BY_ID[variant_id].trainer_family}/cell"
        ),
        "vram": 0,
        "ram_mb": _ram_mb(args, variant_id),
        "cpu": 1,
        "priority": str(args.priority),
        "ckpt_dir": str(absolute_dir),
        "ckpt_glob": "checkpoint.pt",
        "result_dir": str(absolute_dir),
        "local_result_dir": str(absolute_dir),
        "allow_cpu_training": True,
        "cpu_training_justification": CPU_JUSTIFICATION,
        "reroute_on_node_down": True,
        "node_down_requeue_s": 600,
        "allowed_nodes": list(args.nodes),
        "require_node": None,
        "stage_excludes": list(STAGE_EXCLUDES),
        "stage_input_paths": list(args.stage_input_paths),
        "skip_launch_staging": bool(args.skip_launch_staging),
        "allow_duplicate": bool(args.allow_duplicate),
    }


def build_preflight_spec(
    args: argparse.Namespace,
    *,
    node: str,
) -> dict[str, object]:
    relative_dir = preflight_relative_dir(args.run_name, node)
    absolute_dir = ROOT / relative_dir
    return {
        "project": str(args.project),
        "description": f"Freq-HRL full HPO environment preflight on {node}",
        "cmd": build_preflight_command(
            args,
            node=node,
            output_dir=relative_dir,
        ),
        "cwd": str(ROOT / str(args.launch_subdir)),
        "signature": f"Freq-HRL/full-hpo-v2/{args.run_name}/preflight/{node}",
        "resource_family": "Freq-HRL/full-hpo-v2/environment-preflight",
        "vram": 0,
        "ram_mb": 512,
        "cpu": 1,
        "priority": "high",
        "ckpt_dir": str(absolute_dir),
        "ckpt_glob": "environment.json",
        "result_dir": str(absolute_dir),
        "local_result_dir": str(absolute_dir),
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "One lightweight preflight is required on each compute node before "
            "the full CPU experiment matrix is launched."
        ),
        "reroute_on_node_down": False,
        "node_down_requeue_s": 600,
        "allowed_nodes": [str(node)],
        "require_node": None,
        "stage_excludes": list(STAGE_EXCLUDES),
        "stage_input_paths": list(args.stage_input_paths),
        "skip_launch_staging": bool(args.skip_launch_staging),
        "allow_duplicate": bool(args.allow_duplicate),
    }


def expected_cell_dirs(args: argparse.Namespace) -> list[Path]:
    return [
        ROOT / cell_relative_dir(args.run_name, variant, candidate, scenario, seed)
        for variant, candidate, scenario, seed in experiment_cells(
            args.variant_ids,
            args.candidate_ids,
            args.scenarios,
            args.optimizer_seeds,
        )
    ]


def cells_without_local_summary(
    cells: list[tuple[str, str, str, int]],
    *,
    run_name: str,
    root: Path = ROOT,
) -> list[tuple[str, str, str, int]]:
    return [
        cell
        for cell in cells
        if not (
            Path(root)
            / cell_relative_dir(
                run_name,
                cell[0],
                cell[1],
                cell[2],
                cell[3],
            )
            / "cell_summary.json"
        ).exists()
    ]


def validate_preflight_results(args: argparse.Namespace) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for node in args.nodes:
        path = ROOT / preflight_relative_dir(args.run_name, node) / "environment.json"
        if not path.exists():
            raise SystemExit(f"missing environment preflight result for {node}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "ready":
            raise SystemExit(f"environment preflight failed for {node}: {payload}")
        if payload.get("protocol") != FULL_METHOD_TUNING_PROTOCOL_VERSION:
            raise SystemExit(f"environment protocol mismatch for {node}: {payload}")
        if int(payload.get("torch_num_threads", 0)) != 1:
            raise SystemExit(f"environment thread isolation failed for {node}: {payload}")
        payloads.append(payload)
    return payloads


def merge_results(args: argparse.Namespace) -> None:
    directories = expected_cell_dirs(args)
    missing = [path for path in directories if not (path / "cell_summary.json").exists()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise SystemExit(
            f"cannot merge: {len(missing)} full-method HPO cells are missing\n{preview}"
        )
    payload = merge_hpo_cells(
        directories,
        expected_variant_ids=args.variant_ids,
        expected_candidate_ids=args.candidate_ids,
        expected_scenarios=args.scenarios,
        expected_replicate_seeds=args.optimizer_seeds,
        top_k=int(args.top_k),
        stage=str(args.stage),
    )
    output_dir = ROOT / "results" / args.run_name / "merged"
    write_hpo_merge(output_dir, payload)
    print(f"merged {len(directories)} full-method HPO cells into {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", choices=("pilot", "final"), default="pilot")
    parser.add_argument("--variant-ids", default=",".join(ALL_VARIANT_IDS))
    parser.add_argument("--candidate-ids", default=",".join(sorted(CANDIDATES_BY_ID)))
    parser.add_argument("--scenarios", default=None)
    parser.add_argument("--optimizer-seeds", default=None)
    parser.add_argument(
        "--train-seeds", default=",".join(map(str, DEFAULT_ROLLOUT_SEED_ROOTS))
    )
    parser.add_argument(
        "--checkpoint-validation-seeds",
        default=",".join(map(str, DEFAULT_VALIDATION_SEEDS)),
    )
    parser.add_argument(
        "--tuning-validation-seeds",
        default=",".join(map(str, DEFAULT_TUNING_SEEDS)),
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--project", default="Freq-HRL-Full")
    parser.add_argument("--nodes", default=",".join(DEFAULT_NODES))
    parser.add_argument("--python-executable", default="")
    parser.add_argument("--launch-subdir", choices=(".", "scripts"), default=".")
    parser.add_argument("--source-code-revision", default="")
    parser.add_argument("--ppo-ram-mb", type=int, default=512)
    parser.add_argument("--offpolicy-ram-mb", type=int, default=1024)
    parser.add_argument("--priority", choices=("low", "normal", "high"), default="normal")
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--environment-preflight", action="store_true")
    parser.add_argument("--validate-preflight", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-duplicate", action="store_true")
    parser.add_argument("--skip-launch-staging", action="store_true")
    parser.add_argument("--stage-input-path", action="append", default=[])
    parser.add_argument("--skip-complete-cells", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.variant_ids = parse_csv(args.variant_ids)
    args.candidate_ids = parse_csv(args.candidate_ids)
    args.scenarios = parse_csv(
        args.scenarios
        or ",".join(
            DEFAULT_SCENARIOS if args.stage == "final" else DEFAULT_PILOT_SCENARIOS
        )
    )
    default_replicates = (
        DEFAULT_OPTIMIZER_SEEDS[:5]
        if args.stage == "final" else DEFAULT_OPTIMIZER_SEEDS[:3]
    )
    args.optimizer_seeds = parse_csv(
        args.optimizer_seeds or ",".join(map(str, default_replicates)), int
    )
    args.train_seeds = parse_csv(args.train_seeds, int)
    args.checkpoint_validation_seeds = parse_csv(
        args.checkpoint_validation_seeds, int
    )
    args.tuning_validation_seeds = parse_csv(args.tuning_validation_seeds, int)
    args.nodes = parse_csv(args.nodes)
    args.stage_input_paths = [
        str(Path(path).expanduser().resolve())
        for path in args.stage_input_path
        if str(path).strip()
    ]
    if args.skip_launch_staging and not args.stage_input_paths:
        args.stage_input_paths = [str((ROOT / "freq_hrl").resolve())]
    missing_inputs = [
        path for path in args.stage_input_paths if not Path(path).is_dir()
    ]
    if missing_inputs:
        raise SystemExit(
            "stage input directories do not exist: " + ",".join(missing_inputs)
        )
    if args.iterations is None:
        args.iterations = 32 if args.stage == "final" else 8
    unknown_variants = sorted(set(args.variant_ids) - set(ALL_VARIANT_IDS))
    unknown_candidates = sorted(set(args.candidate_ids) - set(CANDIDATES_BY_ID))
    unknown_scenarios = sorted(set(args.scenarios) - set(DEFAULT_SCENARIOS))
    unknown_nodes = sorted(set(args.nodes) - set(LINUX_CPU_NODES))
    if unknown_variants or unknown_candidates or unknown_scenarios or unknown_nodes:
        raise SystemExit(
            "invalid full-method HPO matrix: "
            f"variants={unknown_variants}, candidates={unknown_candidates}, "
            f"scenarios={unknown_scenarios}, nodes={unknown_nodes}"
        )
    if not str(args.python_executable).strip():
        try:
            args.python_executable = default_python_executable(args.nodes)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if int(args.ppo_ram_mb) < 256 or int(args.offpolicy_ram_mb) < 512:
        raise SystemExit("RAM requests are below the safe full-method minimum")
    if args.smoke:
        args.variant_ids = ["freq_hrl_full_v4"]
        args.candidate_ids = [candidate_ids_for_variant("freq_hrl_full_v4")[0]]
        args.scenarios = ["persistent_shift"]
        args.optimizer_seeds = [args.optimizer_seeds[0]]
        args.train_seeds = [args.train_seeds[0]]
        args.checkpoint_validation_seeds = [args.checkpoint_validation_seeds[0]]
        args.tuning_validation_seeds = [args.tuning_validation_seeds[0]]
        args.steps = min(int(args.steps), 32)
        args.iterations = 1
    return args


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    if args.validate_preflight:
        payloads = validate_preflight_results(args)
        print(
            f"validated {len(payloads)} compute-node environments; "
            f"declared_pool_cores={POOL_CPU_CAPACITY}"
        )
        return
    if args.merge_only:
        merge_results(args)
        return
    try:
        args.code_revision, args.source_manifest_sha256 = source_identity(
            args.source_code_revision
        )
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"cannot freeze full-method HPO source identity: {exc}") from exc
    if args.environment_preflight:
        specs = [build_preflight_spec(args, node=node) for node in args.nodes]
        execute_bulk(
            specs,
            dry_run=bool(args.dry_run),
            intent_label=f"Freq-HRL full HPO environment preflight {args.run_name}",
        )
        if args.dispatch and not args.dry_run:
            execute([sys.executable, str(SCHEDULER), "dispatch"], dry_run=False)
        return
    cells = experiment_cells(
        args.variant_ids,
        args.candidate_ids,
        args.scenarios,
        args.optimizer_seeds,
    )
    if args.skip_complete_cells:
        requested_count = len(cells)
        cells = cells_without_local_summary(cells, run_name=args.run_name)
        print(
            f"skipped {requested_count - len(cells)} cells with local summaries",
            flush=True,
        )
    if int(args.max_cells) > 0:
        cells = cells[:int(args.max_cells)]
    if not cells:
        print("no full-method HPO cells require submission", flush=True)
        return
    print(
        f"run={args.run_name} stage={args.stage} cells={len(cells)} "
        f"nodes={','.join(args.nodes)} pool_cores={POOL_CPU_CAPACITY} "
        f"iterations={args.iterations} steps={args.steps}",
        flush=True,
    )
    specs = [
        build_scheduler_spec(
            args,
            variant_id=variant,
            candidate_id=candidate,
            scenario=scenario,
            replicate_seed=seed,
        )
        for variant, candidate, scenario, seed in cells
    ]
    execute_bulk(
        specs,
        dry_run=bool(args.dry_run),
        intent_label=f"Freq-HRL full HPO {args.run_name}",
    )
    if args.dispatch and not args.dry_run:
        execute([sys.executable, str(SCHEDULER), "dispatch"], dry_run=False)
    print(
        "merge after all result summaries are synced: "
        + shlex.join([
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-name",
            args.run_name,
            "--stage",
            args.stage,
            "--variant-ids",
            ",".join(args.variant_ids),
            "--candidate-ids",
            ",".join(args.candidate_ids),
            "--scenarios",
            ",".join(args.scenarios),
            "--optimizer-seeds",
            ",".join(map(str, args.optimizer_seeds)),
            "--merge-only",
        ])
    )


if __name__ == "__main__":
    main()
