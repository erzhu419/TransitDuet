#!/usr/bin/env python3
"""Submit or merge source-bound Freq-HRL v7.3.2 confirmatory cells."""

from __future__ import annotations

import argparse
import base64
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.trading import (  # noqa: E402
    full_method_confirmatory_plan_v732 as plan,
)
from freq_hrl.experiments.trading import (  # noqa: E402
    full_method_confirmatory_v732 as confirm,
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
CONFIRMATORY_MODULE = (
    "freq_hrl.experiments.trading.full_method_confirmatory_v732"
)
SIGNATURE_VERSION = "confirmatory-v7-3-2-source-bound"
SUBMIT_SCRIPT_PATH = Path(__file__).resolve()


def experiment_cells(
    variant_ids: list[str], training_replicates: list[int]
) -> list[tuple[str, int]]:
    return [
        (variant_id, int(replicate))
        for variant_id in variant_ids for replicate in training_replicates
    ]


def cell_relative_dir(run_name: str, variant_id: str, replicate: int) -> Path:
    return (
        Path("results") / run_name / "cells" / variant_id
        / f"replicate_{int(replicate)}"
    )


def _runtime_frozen_path(
    args: argparse.Namespace, variant_id: str, replicate: int
) -> str:
    return (
        f"/tmp/freq_hrl_v732_{args.frozen_config_sha256[:16]}_"
        f"{variant_id}_{int(replicate)}.json"
    )


def build_training_command(
    args: argparse.Namespace,
    *,
    variant_id: str,
    replicate: int,
    output_dir: Path,
) -> str:
    frozen_path = _runtime_frozen_path(args, variant_id, replicate)
    materialize_code = (
        "import base64;from pathlib import Path;"
        f"Path({frozen_path!r}).write_bytes(base64.b64decode("
        f"{args.frozen_config_b64!r}))"
    )
    materialize = shlex.join([
        str(args.python_executable), "-c", materialize_code
    ])
    command = [
        str(args.python_executable), "-u", "-m", CONFIRMATORY_MODULE,
        "--frozen-config", frozen_path,
        "--variant-id", str(variant_id),
        "--training-replicate-seed", str(replicate),
        "--heldout-seeds", *(str(seed) for seed in args.heldout_seeds),
        "--output-dir", str(output_dir),
    ]
    if args.save_checkpoints:
        command.append("--save-checkpoint")
    env = [
        "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=.",
        "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1", "TORCH_NUM_THREADS=1",
        "CUDA_VISIBLE_DEVICES=",
    ]
    run = " ".join([*env, shlex.join(command)])
    command_text = (
        f"{materialize} && {run} && rm -f {shlex.quote(frozen_path)} "
        "&& echo DONE"
    )
    return (
        f"cd .. && {command_text}"
        if args.launch_subdir == "scripts" else command_text
    )


def _ram_mb(args: argparse.Namespace, variant_id: str) -> int:
    return int(
        args.offpolicy_ram_mb
        if hpo.VARIANTS_BY_ID[variant_id].trainer_family == "offpolicy"
        else args.ppo_ram_mb
    )


def build_scheduler_spec(
    args: argparse.Namespace, *, variant_id: str, replicate: int
) -> dict[str, object]:
    relative = cell_relative_dir(args.run_name, variant_id, replicate)
    absolute = ROOT / relative
    return {
        "project": str(args.project),
        "description": (
            f"Freq-HRL v7.3.2 confirmatory {variant_id} replicate {replicate}"
        ),
        "cmd": build_training_command(
            args,
            variant_id=variant_id,
            replicate=replicate,
            output_dir=relative,
        ),
        "cwd": str(ROOT / str(args.launch_subdir)),
        "signature": (
            f"Freq-HRL/{SIGNATURE_VERSION}/{plan.plan_sha256()[:16]}/"
            f"{args.frozen_config_sha256[:16]}/{args.run_name}/"
            f"{variant_id}/rep-{replicate}"
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
        "ckpt_glob": (
            "checkpoint.pt" if args.save_checkpoints else "cell_summary.json"
        ),
        "result_dir": str(absolute),
        "local_result_dir": str(absolute),
        "allow_cpu_training": True,
        "cpu_training_justification": (
            "Each source-bound confirmatory replicate is independent and "
            "single-threaded; paths and scenarios are repeated measures."
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
        ROOT / cell_relative_dir(args.run_name, variant, replicate)
        for variant, replicate in experiment_cells(
            args.variant_ids, args.training_replicates
        )
    ]


def merge_results(args: argparse.Namespace) -> None:
    directories = expected_cell_dirs(args)
    missing = [
        path for path in directories
        if not (path / "cell_summary.json").exists()
    ]
    if missing:
        raise SystemExit(
            f"cannot merge: {len(missing)} cells missing; first={missing[0]}"
        )
    payload = confirm.merge_confirmatory_cells(
        directories,
        expected_variant_ids=args.variant_ids,
        expected_training_replicates=args.training_replicates,
        expected_heldout_seeds=args.heldout_seeds,
    )
    output = ROOT / "results" / args.run_name / "merged"
    confirm.write_confirmatory_merge(output, payload)
    print(f"merged {len(directories)} confirmatory cells into {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--variant-ids", default=",".join(hpo.ALL_VARIANT_IDS))
    parser.add_argument(
        "--training-replicates",
        default=",".join(map(str, plan.DEFAULT_CONFIRMATORY_REPLICATES)),
    )
    parser.add_argument(
        "--heldout-seeds",
        default=",".join(map(str, plan.DEFAULT_HELDOUT_SEEDS)),
    )
    parser.add_argument("--nodes", default=",".join(LINUX_CPU_NODES))
    parser.add_argument("--python-executable", default="")
    parser.add_argument("--launch-subdir", choices=(".", "scripts"), default=".")
    parser.add_argument("--project", default="Freq-HRL-v7.3.2-Confirmatory")
    parser.add_argument("--ppo-ram-mb", type=int, default=1024)
    parser.add_argument("--offpolicy-ram-mb", type=int, default=1536)
    parser.add_argument(
        "--priority", choices=("low", "normal", "high"), default="normal"
    )
    parser.add_argument("--save-checkpoints", action="store_true")
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
    args.variant_ids = parse_csv(args.variant_ids)
    args.training_replicates = parse_csv(args.training_replicates, int)
    args.heldout_seeds = parse_csv(args.heldout_seeds, int)
    args.nodes = parse_csv(args.nodes)
    args.stage_input_paths = [
        str(Path(path).expanduser().resolve())
        for path in args.stage_input_path if str(path).strip()
    ]
    if tuple(args.variant_ids) != tuple(hpo.ALL_VARIANT_IDS):
        raise SystemExit("confirmatory scheduler requires all registered variants")
    if tuple(args.training_replicates) != tuple(
        plan.DEFAULT_CONFIRMATORY_REPLICATES
    ):
        raise SystemExit("confirmatory training replicate registry drifted")
    if tuple(args.heldout_seeds) != tuple(plan.DEFAULT_HELDOUT_SEEDS):
        raise SystemExit("confirmatory held-out registry drifted")
    unknown_nodes = sorted(set(args.nodes) - set(LINUX_CPU_NODES))
    if unknown_nodes:
        raise SystemExit(f"invalid confirmatory nodes: {unknown_nodes}")
    frozen_path = Path(args.frozen_config).expanduser().resolve()
    try:
        frozen, audit = hpo.load_frozen_config(frozen_path)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"invalid frozen v7.3.2 config: {exc}") from exc
    try:
        _, current_manifest = source_identity()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"cannot register confirmatory source: {exc}") from exc
    if current_manifest != frozen["source_manifest_sha256"]:
        raise SystemExit("current Freq-HRL source differs from the HPO freeze")
    if frozen["confirmatory_plan_sha256"] != plan.plan_sha256():
        raise SystemExit("frozen config has a different confirmatory plan")
    args.frozen_config = frozen_path
    args.frozen_config_sha256 = str(audit["sha256"])
    args.frozen_config_b64 = base64.b64encode(
        frozen_path.read_bytes()
    ).decode("ascii")
    if not args.python_executable.strip():
        args.python_executable = default_python_executable(args.nodes)
    return args


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    if args.merge_only:
        merge_results(args)
        return
    cells = experiment_cells(args.variant_ids, args.training_replicates)
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
        print("no confirmatory v7.3.2 cells require submission")
        return
    print(
        f"run={args.run_name} cells={len(cells)} nodes={','.join(args.nodes)} "
        f"pool_cores={POOL_CPU_CAPACITY} plan={plan.plan_sha256()[:16]} "
        f"frozen={args.frozen_config_sha256[:16]}",
        flush=True,
    )
    execute_bulk(
        [
            build_scheduler_spec(
                args, variant_id=variant, replicate=replicate
            )
            for variant, replicate in cells
        ],
        dry_run=bool(args.dry_run),
        intent_label=f"Freq-HRL v7.3.2 confirmatory {args.run_name}",
    )
    if args.dispatch and not args.dry_run:
        execute([sys.executable, str(SCHEDULER), "dispatch"], dry_run=False)
    print(
        "merge after all summaries sync: " + shlex.join([
            sys.executable,
            str(SUBMIT_SCRIPT_PATH),
            "--run-name",
            args.run_name,
            "--frozen-config",
            str(args.frozen_config),
            "--merge-only",
        ])
    )


if __name__ == "__main__":
    main()
