#!/usr/bin/env python3
"""Submit or merge nested-validation HPO cells through scheduleurm."""

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

from freq_hrl.experiments.trading.hyperparameter_pilot import (  # noqa: E402
    ALL_POLICY_MODES,
    CANDIDATES_BY_ID,
    DEFAULT_PILOT_SCENARIOS,
    DEFAULT_TUNING_SEEDS,
    candidate_ids_for_mode,
    merge_hpo_cells,
    write_hpo_merge,
)
from freq_hrl.experiments.trading.strong_learned_baseline_validation import (  # noqa: E402
    DEFAULT_OPTIMIZER_SEEDS,
    DEFAULT_ROLLOUT_SEED_ROOTS,
    DEFAULT_SCENARIOS,
    DEFAULT_VALIDATION_SEEDS,
)
from freq_hrl.experiments.reproducibility import (  # noqa: E402
    git_source_manifest_sha256,
    is_hex_digest,
    registered_git_source_identity,
)


SCHEDULER = Path("/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py")
WINDOWS_CPU_NODES = ("jtl110cpu", "jtl110cpu2")
LINUX_CPU_NODES = tuple(f"node{index:03d}" for index in range(1, 7))
SUPPORTED_NODES = WINDOWS_CPU_NODES + LINUX_CPU_NODES
DEFAULT_NODES = LINUX_CPU_NODES
DEFAULT_LINUX_PYTHON = (
    "/home/zhengliang01/scheduleurm_work/conda_envs/"
    "csbapr-gpu-py310/bin/python"
)
CPU_JUSTIFICATION = (
    "Nested-validation actor-critic HPO cells are independent, CPU-bound, and "
    "single-threaded; scheduleurm dynamically places each cell across the "
    "declared physical-core CPU pool."
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


def source_identity(revision_override: str = "") -> tuple[str, str]:
    current_revision, working_manifest = registered_git_source_identity(
        ROOT, Path("freq_hrl")
    )
    requested_revision = str(revision_override).strip().lower()
    if not requested_revision:
        return current_revision, working_manifest
    if not is_hex_digest(requested_revision, length=40):
        raise ValueError("source code revision override must be a full Git SHA")
    requested_manifest = git_source_manifest_sha256(
        ROOT,
        Path("freq_hrl"),
        revision=requested_revision,
    )
    if requested_manifest != working_manifest:
        raise RuntimeError(
            "requested source revision does not contain the staged Freq-HRL source"
        )
    return requested_revision, working_manifest


def default_python_executable(nodes: list[str]) -> str:
    selected = set(nodes)
    if selected and selected <= set(LINUX_CPU_NODES):
        return DEFAULT_LINUX_PYTHON
    if selected and selected <= set(WINDOWS_CPU_NODES):
        return "python3"
    raise ValueError(
        "mixed Linux/Windows node pools require an explicit portable "
        "--python-executable"
    )


def experiment_cells(
    policy_modes: list[str],
    candidate_ids: list[str],
    scenarios: list[str],
    replicate_seeds: list[int],
) -> list[tuple[str, str, str, int]]:
    return [
        (mode, candidate_id, scenario, int(seed))
        for mode in policy_modes
        for candidate_id in candidate_ids
        if candidate_id in candidate_ids_for_mode(mode)
        for scenario in scenarios
        for seed in replicate_seeds
    ]


def cell_relative_dir(
    run_name: str,
    policy_mode: str,
    candidate_id: str,
    scenario: str,
    replicate_seed: int,
) -> Path:
    return (
        Path("results")
        / run_name
        / "cells"
        / policy_mode
        / candidate_id
        / scenario
        / f"replicate_{int(replicate_seed)}"
    )


def build_training_command(
    args: argparse.Namespace,
    *,
    policy_mode: str,
    candidate_id: str,
    scenario: str,
    replicate_seed: int,
    output_dir: Path,
) -> str:
    command = [
        str(args.python_executable),
        "-u",
        "-m",
        "freq_hrl.experiments.trading.hyperparameter_pilot",
        "--candidate-id",
        candidate_id,
        "--policy-mode",
        policy_mode,
        "--scenario",
        scenario,
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
        str(getattr(args, "code_revision", "")),
        "--source-manifest-sha256",
        str(getattr(args, "source_manifest_sha256", "")),
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


def build_scheduler_command(
    args: argparse.Namespace,
    *,
    policy_mode: str,
    candidate_id: str,
    scenario: str,
    replicate_seed: int,
) -> list[str]:
    spec = build_scheduler_spec(
        args,
        policy_mode=policy_mode,
        candidate_id=candidate_id,
        scenario=scenario,
        replicate_seed=replicate_seed,
    )
    command = [
        sys.executable,
        str(SCHEDULER),
        "submit",
        "--project",
        str(spec["project"]),
        "--description",
        str(spec["description"]),
        "--cmd",
        str(spec["cmd"]),
        "--cwd",
        str(spec["cwd"]),
        "--signature",
        str(spec["signature"]),
        "--resource-family",
        str(spec["resource_family"]),
        "--vram",
        str(spec["vram"]),
        "--ram-mb",
        str(spec["ram_mb"]),
        "--cpu",
        str(spec["cpu"]),
        "--priority",
        str(spec["priority"]),
        "--ckpt-dir",
        str(spec["ckpt_dir"]),
        "--ckpt-glob",
        str(spec["ckpt_glob"]),
        "--result-dir",
        str(spec["result_dir"]),
        "--local-result-dir",
        str(spec["local_result_dir"]),
        "--allow-cpu-training",
        "--cpu-training-justification",
        str(spec["cpu_training_justification"]),
        "--allow-no-resume",
        "--reroute-on-node-down",
        "--node-down-requeue-s",
        str(spec["node_down_requeue_s"]),
    ]
    for node in spec["allowed_nodes"]:
        command.extend(["--allowed-node", str(node)])
    for excluded in spec["stage_excludes"]:
        command.extend(["--stage-exclude", str(excluded)])
    if spec["skip_launch_staging"]:
        command.append("--skip-launch-staging")
    if spec["allow_duplicate"]:
        command.append("--allow-duplicate")
    return command


def build_scheduler_spec(
    args: argparse.Namespace,
    *,
    policy_mode: str,
    candidate_id: str,
    scenario: str,
    replicate_seed: int,
) -> dict[str, object]:
    relative_dir = cell_relative_dir(
        args.run_name,
        policy_mode,
        candidate_id,
        scenario,
        replicate_seed,
    )
    absolute_dir = ROOT / relative_dir
    return {
        "project": str(args.project),
        "description": (
            f"Freq-HRL nested HPO {policy_mode} {candidate_id} "
            f"{scenario} replicate {replicate_seed}"
        ),
        "cmd": build_training_command(
            args,
            policy_mode=policy_mode,
            candidate_id=candidate_id,
            scenario=scenario,
            replicate_seed=replicate_seed,
            output_dir=relative_dir,
        ),
        "cwd": str(ROOT / str(args.launch_subdir)),
        "signature": (
            f"Freq-HRL/hpo-v1/{args.run_name}/{policy_mode}/{candidate_id}/"
            f"{scenario}/rep-{replicate_seed}"
        ),
        "resource_family": "Freq-HRL/hpo-v1/single-cell",
        "vram": 0,
        "ram_mb": int(args.ram_mb),
        "cpu": int(args.cpu),
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
        "stage_excludes": list(STAGE_EXCLUDES),
        "skip_launch_staging": bool(args.skip_launch_staging),
        "allow_duplicate": bool(args.allow_duplicate),
    }


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


def execute_bulk(
    specs: list[dict[str, object]],
    *,
    dry_run: bool,
    intent_label: str,
) -> None:
    if dry_run:
        print(
            f"dry-run bulk submit: {len(specs)} tasks; "
            f"first={specs[0]['signature'] if specs else 'none'}",
            flush=True,
        )
        return
    command = [
        sys.executable,
        str(SCHEDULER),
        "submit-jsonl",
        "--stdin",
        "--trusted",
        "--json",
        "--intent-label",
        str(intent_label),
    ]
    process = subprocess.run(
        command,
        input=json.dumps(specs, ensure_ascii=True, separators=(",", ":")),
        text=True,
        capture_output=True,
    )
    if process.returncode != 0:
        if process.stdout.strip():
            print(process.stdout.strip(), file=sys.stderr)
        if process.stderr.strip():
            print(process.stderr.strip(), file=sys.stderr)
        process.check_returncode()
    payload = json.loads(process.stdout)
    submitted = list(payload.get("submitted", []))
    first_id = submitted[0]["id"] if submitted else "none"
    last_id = submitted[-1]["id"] if submitted else "none"
    print(
        f"bulk submitted {payload.get('count', 0)} tasks "
        f"ids={first_id}..{last_id}",
        flush=True,
    )


def expected_cell_dirs(args: argparse.Namespace) -> list[Path]:
    return [
        ROOT / cell_relative_dir(args.run_name, mode, candidate, scenario, seed)
        for mode, candidate, scenario, seed in experiment_cells(
            args.policy_modes,
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


def merge_results(args: argparse.Namespace) -> None:
    directories = expected_cell_dirs(args)
    missing = [path for path in directories if not (path / "cell_summary.json").exists()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise SystemExit(f"cannot merge: {len(missing)} HPO cells are missing\n{preview}")
    payload = merge_hpo_cells(
        directories,
        expected_policy_modes=args.policy_modes,
        expected_candidate_ids=args.candidate_ids,
        expected_scenarios=args.scenarios,
        expected_replicate_seeds=args.optimizer_seeds,
        top_k=int(args.top_k),
        stage=str(args.stage),
    )
    output_dir = ROOT / "results" / args.run_name / "merged"
    write_hpo_merge(output_dir, payload)
    print(f"merged {len(directories)} HPO cells into {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage", choices=("pilot", "final"), default="pilot")
    parser.add_argument("--policy-modes", default=",".join(ALL_POLICY_MODES))
    parser.add_argument("--candidate-ids", default=",".join(sorted(CANDIDATES_BY_ID)))
    parser.add_argument("--scenarios", default=None)
    parser.add_argument("--optimizer-seeds", default=None)
    parser.add_argument("--train-seeds", default=",".join(map(str, DEFAULT_ROLLOUT_SEED_ROOTS)))
    parser.add_argument(
        "--checkpoint-validation-seeds",
        default=",".join(map(str, DEFAULT_VALIDATION_SEEDS)),
    )
    parser.add_argument(
        "--tuning-validation-seeds",
        default=",".join(map(str, DEFAULT_TUNING_SEEDS)),
    )
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--project", default="Freq-HRL")
    parser.add_argument("--nodes", default=",".join(DEFAULT_NODES))
    parser.add_argument("--python-executable", default="")
    parser.add_argument("--launch-subdir", choices=(".", "scripts"), default=".")
    parser.add_argument("--source-code-revision", default="")
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--ram-mb", type=int, default=2048)
    parser.add_argument("--priority", choices=("low", "normal", "high"), default="normal")
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-duplicate", action="store_true")
    parser.add_argument("--skip-launch-staging", action="store_true")
    parser.add_argument("--skip-complete-cells", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.policy_modes = parse_csv(args.policy_modes)
    args.candidate_ids = parse_csv(args.candidate_ids)
    args.scenarios = parse_csv(
        args.scenarios
        or ",".join(DEFAULT_SCENARIOS if args.stage == "final" else DEFAULT_PILOT_SCENARIOS)
    )
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
    if args.iterations is None:
        args.iterations = 64 if args.stage == "final" else 16
    unknown_modes = sorted(set(args.policy_modes) - set(ALL_POLICY_MODES))
    unknown_candidates = sorted(set(args.candidate_ids) - set(CANDIDATES_BY_ID))
    unknown_scenarios = sorted(set(args.scenarios) - set(DEFAULT_SCENARIOS))
    unknown_nodes = sorted(set(args.nodes) - set(SUPPORTED_NODES))
    if unknown_modes or unknown_candidates or unknown_scenarios or unknown_nodes:
        raise SystemExit(
            "invalid HPO matrix selection: "
            f"modes={unknown_modes}, candidates={unknown_candidates}, "
            f"scenarios={unknown_scenarios}, nodes={unknown_nodes}"
        )
    if not str(args.python_executable).strip():
        try:
            args.python_executable = default_python_executable(args.nodes)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    if args.smoke:
        args.policy_modes = ["freq_hrl"]
        args.candidate_ids = [candidate_ids_for_mode("freq_hrl")[0]]
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
    if args.merge_only:
        merge_results(args)
        return
    try:
        args.code_revision, args.source_manifest_sha256 = source_identity(
            args.source_code_revision
        )
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"cannot freeze HPO source identity: {exc}") from exc
    cells = experiment_cells(
        args.policy_modes,
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
        cells = cells[: int(args.max_cells)]
    if not cells:
        print("no HPO cells require submission", flush=True)
        return
    print(
        f"run={args.run_name} stage={args.stage} cells={len(cells)} "
        f"nodes={','.join(args.nodes)} iterations={args.iterations} steps={args.steps}",
        flush=True,
    )
    specs = [
        build_scheduler_spec(
            args,
            policy_mode=mode,
            candidate_id=candidate,
            scenario=scenario,
            replicate_seed=seed,
        )
        for mode, candidate, scenario, seed in cells
    ]
    execute_bulk(
        specs,
        dry_run=bool(args.dry_run),
        intent_label=f"Freq-HRL HPO {args.run_name}",
    )
    if args.dispatch and not args.dry_run:
        execute([sys.executable, str(SCHEDULER), "dispatch"], dry_run=False)
    print(
        "merge after every result is synced: "
        + shlex.join([
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-name",
            args.run_name,
            "--stage",
            args.stage,
            "--policy-modes",
            ",".join(args.policy_modes),
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
