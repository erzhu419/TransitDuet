#!/usr/bin/env python3
"""Submit or merge support-only Freq-HRL v6 HPO through scheduleurm."""

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

from freq_hrl.experiments.trading import full_method_hpo_v6 as hpo  # noqa: E402
from freq_hrl.experiments.trading.strong_learned_baseline_validation import (  # noqa: E402
    DEFAULT_OPTIMIZER_SEEDS,
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
HPO_MODULE = "freq_hrl.experiments.trading.full_method_hpo_v6"
HPO_SIGNATURE_VERSION = "full-hpo-v6-support-only"
SUBMIT_SCRIPT_PATH = Path(__file__).resolve()
CPU_JUSTIFICATION = (
    "Each v6 HPO cell trains one frozen checkpoint and is explicitly "
    "single-threaded. scheduleurm dynamically packs independent cells across "
    "the six-node 1152-physical-core Linux pool."
)


def experiment_cells(
    variant_ids: list[str],
    candidate_ids: list[str],
    replicate_seeds: list[int],
) -> list[tuple[str, str, int]]:
    return [
        (variant_id, candidate_id, int(seed))
        for variant_id in variant_ids
        for candidate_id in candidate_ids
        if candidate_id in hpo.candidate_ids_for_variant(variant_id)
        for seed in replicate_seeds
    ]


def cell_relative_dir(
    run_name: str,
    variant_id: str,
    candidate_id: str,
    replicate_seed: int,
) -> Path:
    return (
        Path("results") / run_name / "cells" / variant_id / candidate_id
        / f"replicate_{int(replicate_seed)}"
    )


def preflight_relative_dir(run_name: str, node: str) -> Path:
    return Path("results") / run_name / "environment_preflight" / str(node)


def build_training_command(
    args: argparse.Namespace,
    *,
    variant_id: str,
    candidate_id: str,
    replicate_seed: int,
    output_dir: Path,
) -> str:
    command = [
        str(args.python_executable), "-u", "-m", HPO_MODULE,
        "--candidate-id", str(candidate_id),
        "--variant-id", str(variant_id),
        "--training-replicate-seed", str(replicate_seed),
        "--train-seeds", *(str(seed) for seed in args.train_seeds),
        "--checkpoint-validation-seeds",
        *(str(seed) for seed in args.checkpoint_validation_seeds),
        "--tuning-validation-seeds",
        *(str(seed) for seed in args.tuning_validation_seeds),
        "--steps", str(args.steps), "--assets", str(args.assets),
        "--iterations", str(args.iterations),
        "--code-revision", str(args.code_revision),
        "--source-manifest-sha256", str(args.source_manifest_sha256),
        "--output-dir", str(output_dir),
    ]
    env = [
        "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=.",
        "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1", "TORCH_NUM_THREADS=1", "CUDA_VISIBLE_DEVICES=",
    ]
    command_text = " ".join([*env, shlex.join(command)]) + " && echo DONE"
    return f"cd .. && {command_text}" if args.launch_subdir == "scripts" else command_text


def build_preflight_command(
    args: argparse.Namespace, *, node: str, output_dir: Path
) -> str:
    code = (
        "import json,os,platform;from pathlib import Path;import numpy,torch;"
        f"from {HPO_MODULE} import FULL_METHOD_TUNING_PROTOCOL_VERSION;"
        f"p=Path({str(output_dir)!r});p.mkdir(parents=True,exist_ok=True);"
        "x={'status':'ready','node':platform.node(),"
        f"'requested_node':{str(node)!r},'python':platform.python_version(),"
        "'python_executable':os.sys.executable,'numpy':numpy.__version__,"
        "'torch':torch.__version__,'visible_cpu_count':os.cpu_count(),"
        "'torch_num_threads':torch.get_num_threads(),"
        "'protocol':FULL_METHOD_TUNING_PROTOCOL_VERSION};"
        "(p/'environment.json').write_text(json.dumps(x,sort_keys=True));"
        "print(json.dumps(x,sort_keys=True))"
    )
    env = [
        "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=.",
        "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1", "TORCH_NUM_THREADS=1", "CUDA_VISIBLE_DEVICES=",
    ]
    command_text = " ".join([
        *env, shlex.join([str(args.python_executable), "-c", code])
    ]) + " && echo DONE"
    return f"cd .. && {command_text}" if args.launch_subdir == "scripts" else command_text


def _ram_mb(args: argparse.Namespace, variant_id: str) -> int:
    return int(
        args.offpolicy_ram_mb
        if hpo.VARIANTS_BY_ID[variant_id].trainer_family == "offpolicy"
        else args.ppo_ram_mb
    )


def build_scheduler_spec(
    args: argparse.Namespace,
    *,
    variant_id: str,
    candidate_id: str,
    replicate_seed: int,
) -> dict[str, object]:
    relative = cell_relative_dir(
        args.run_name, variant_id, candidate_id, replicate_seed
    )
    absolute = ROOT / relative
    return {
        "project": str(args.project),
        "description": (
            f"Freq-HRL v6 support HPO {variant_id} {candidate_id} "
            f"replicate {replicate_seed}"
        ),
        "cmd": build_training_command(
            args, variant_id=variant_id, candidate_id=candidate_id,
            replicate_seed=replicate_seed, output_dir=relative,
        ),
        "cwd": str(ROOT / str(args.launch_subdir)),
        "signature": (
            f"Freq-HRL/{HPO_SIGNATURE_VERSION}/{args.run_name}/{variant_id}/"
            f"{candidate_id}/rep-{replicate_seed}"
        ),
        "resource_family": (
            f"Freq-HRL/{HPO_SIGNATURE_VERSION}/"
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


def build_preflight_spec(args: argparse.Namespace, *, node: str) -> dict[str, object]:
    relative = preflight_relative_dir(args.run_name, node)
    absolute = ROOT / relative
    return {
        "project": str(args.project),
        "description": f"Freq-HRL v6 environment preflight on {node}",
        "cmd": build_preflight_command(args, node=node, output_dir=relative),
        "cwd": str(ROOT / str(args.launch_subdir)),
        "signature": f"Freq-HRL/{HPO_SIGNATURE_VERSION}/{args.run_name}/preflight/{node}",
        "resource_family": f"Freq-HRL/{HPO_SIGNATURE_VERSION}/environment-preflight",
        "vram": 0, "ram_mb": 512, "cpu": 1, "priority": "high",
        "ckpt_dir": str(absolute), "ckpt_glob": "environment.json",
        "result_dir": str(absolute), "local_result_dir": str(absolute),
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "A lightweight import and CPU-visibility check runs on each compute "
            "node; no package installation occurs on the login node."
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
        ROOT / cell_relative_dir(args.run_name, variant, candidate, seed)
        for variant, candidate, seed in experiment_cells(
            args.variant_ids, args.candidate_ids, args.optimizer_seeds
        )
    ]


def cells_without_local_summary(
    cells: list[tuple[str, str, int]], *, run_name: str
) -> list[tuple[str, str, int]]:
    return [
        cell for cell in cells
        if not (
            ROOT / cell_relative_dir(run_name, *cell) / "cell_summary.json"
        ).exists()
    ]


def validate_preflight_results(args: argparse.Namespace) -> list[dict[str, object]]:
    payloads = []
    for node in args.nodes:
        path = ROOT / preflight_relative_dir(args.run_name, node) / "environment.json"
        if not path.exists():
            raise SystemExit(f"missing environment preflight for {node}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "ready":
            raise SystemExit(f"environment preflight failed for {node}: {payload}")
        if payload.get("protocol") != hpo.FULL_METHOD_TUNING_PROTOCOL_VERSION:
            raise SystemExit(f"environment protocol mismatch for {node}")
        if int(payload.get("torch_num_threads", 0)) != 1:
            raise SystemExit(f"thread isolation failed for {node}")
        if int(payload.get("visible_cpu_count", 0)) < NODE_CPU_CAPACITY:
            raise SystemExit(f"{node} exposes fewer than {NODE_CPU_CAPACITY} CPUs")
        payloads.append(payload)
    return payloads


def merge_results(args: argparse.Namespace) -> None:
    directories = expected_cell_dirs(args)
    missing = [path for path in directories if not (path / "cell_summary.json").exists()]
    if missing:
        raise SystemExit(
            f"cannot merge: {len(missing)} cells missing; first={missing[0]}"
        )
    payload = hpo.merge_hpo_cells(
        directories,
        expected_variant_ids=args.variant_ids,
        expected_candidate_ids=args.candidate_ids,
        expected_replicate_seeds=args.optimizer_seeds,
        top_k=int(args.top_k), stage=str(args.stage),
    )
    output = ROOT / "results" / args.run_name / "merged"
    hpo.write_hpo_merge(output, payload)
    print(f"merged {len(directories)} v6 HPO cells into {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", choices=("pilot", "final"), default="pilot")
    parser.add_argument("--variant-ids", default=",".join(hpo.HPO_VARIANT_IDS))
    parser.add_argument("--candidate-ids", default=",".join(sorted(hpo.CANDIDATES_BY_ID)))
    parser.add_argument("--optimizer-seeds", default=None)
    parser.add_argument("--train-seeds", default=",".join(map(str, hpo.DEFAULT_TRAIN_SEEDS)))
    parser.add_argument("--checkpoint-validation-seeds", default=",".join(map(str, hpo.DEFAULT_CHECKPOINT_VALIDATION_SEEDS)))
    parser.add_argument("--tuning-validation-seeds", default=",".join(map(str, hpo.DEFAULT_TUNING_SEEDS)))
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--project", default="Freq-HRL-v6")
    parser.add_argument("--nodes", default=",".join(DEFAULT_NODES))
    parser.add_argument("--python-executable", default="")
    parser.add_argument("--launch-subdir", choices=(".", "scripts"), default=".")
    parser.add_argument("--source-code-revision", default="")
    parser.add_argument("--ppo-ram-mb", type=int, default=768)
    parser.add_argument("--offpolicy-ram-mb", type=int, default=1536)
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
    default_replicates = (
        DEFAULT_OPTIMIZER_SEEDS[:5]
        if args.stage == "final" else DEFAULT_OPTIMIZER_SEEDS[:3]
    )
    args.optimizer_seeds = parse_csv(
        args.optimizer_seeds or ",".join(map(str, default_replicates)), int
    )
    args.train_seeds = parse_csv(args.train_seeds, int)
    args.checkpoint_validation_seeds = parse_csv(args.checkpoint_validation_seeds, int)
    args.tuning_validation_seeds = parse_csv(args.tuning_validation_seeds, int)
    args.nodes = parse_csv(args.nodes)
    args.stage_input_paths = [
        str(Path(path).expanduser().resolve())
        for path in args.stage_input_path if str(path).strip()
    ]
    if args.skip_launch_staging and not args.stage_input_paths:
        args.stage_input_paths = [str((ROOT / "freq_hrl").resolve())]
    if args.iterations is None:
        args.iterations = 32 if args.stage == "final" else 8
    unknown_variants = sorted(set(args.variant_ids) - set(hpo.HPO_VARIANT_IDS))
    unknown_candidates = sorted(set(args.candidate_ids) - set(hpo.CANDIDATES_BY_ID))
    unknown_nodes = sorted(set(args.nodes) - set(LINUX_CPU_NODES))
    if unknown_variants or unknown_candidates or unknown_nodes:
        raise SystemExit(
            f"invalid v6 matrix: variants={unknown_variants}, "
            f"candidates={unknown_candidates}, nodes={unknown_nodes}"
        )
    if not args.python_executable.strip():
        args.python_executable = default_python_executable(args.nodes)
    if args.smoke:
        args.variant_ids = [hpo.ABLATION_PARENT_VARIANT]
        args.candidate_ids = [hpo.candidate_ids_for_variant(hpo.ABLATION_PARENT_VARIANT)[0]]
        args.optimizer_seeds = [args.optimizer_seeds[0]]
        args.train_seeds = [args.train_seeds[0]]
        args.checkpoint_validation_seeds = [args.checkpoint_validation_seeds[0]]
        args.tuning_validation_seeds = [args.tuning_validation_seeds[0]]
        args.steps = min(int(args.steps), 24)
        args.iterations = 1
    return args


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    if args.validate_preflight:
        payloads = validate_preflight_results(args)
        print(
            f"validated {len(payloads)} compute nodes; "
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
        raise SystemExit(f"cannot freeze v6 HPO source identity: {exc}") from exc
    if args.environment_preflight:
        execute_bulk(
            [build_preflight_spec(args, node=node) for node in args.nodes],
            dry_run=bool(args.dry_run),
            intent_label=f"Freq-HRL v6 environment preflight {args.run_name}",
        )
        if args.dispatch and not args.dry_run:
            execute([sys.executable, str(SCHEDULER), "dispatch"], dry_run=False)
        return
    cells = experiment_cells(
        args.variant_ids, args.candidate_ids, args.optimizer_seeds
    )
    if args.skip_complete_cells:
        cells = cells_without_local_summary(cells, run_name=args.run_name)
    if args.max_cells > 0:
        cells = cells[:args.max_cells]
    if not cells:
        print("no v6 HPO cells require submission")
        return
    print(
        f"run={args.run_name} cells={len(cells)} nodes={','.join(args.nodes)} "
        f"pool_cores={POOL_CPU_CAPACITY} iterations={args.iterations} steps={args.steps}",
        flush=True,
    )
    execute_bulk(
        [
            build_scheduler_spec(
                args, variant_id=variant, candidate_id=candidate,
                replicate_seed=seed,
            )
            for variant, candidate, seed in cells
        ],
        dry_run=bool(args.dry_run),
        intent_label=f"Freq-HRL v6 support HPO {args.run_name}",
    )
    if args.dispatch and not args.dry_run:
        execute([sys.executable, str(SCHEDULER), "dispatch"], dry_run=False)
    print(
        "merge after all summaries sync: " + shlex.join([
            sys.executable, str(SUBMIT_SCRIPT_PATH), "--run-name", args.run_name,
            "--stage", args.stage, "--variant-ids", ",".join(args.variant_ids),
            "--candidate-ids", ",".join(args.candidate_ids),
            "--optimizer-seeds", ",".join(map(str, args.optimizer_seeds)),
            "--merge-only",
        ])
    )


if __name__ == "__main__":
    main()
