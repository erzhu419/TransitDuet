#!/usr/bin/env python3
"""Submit or merge the confirmatory Freq-HRL learned-baseline matrix."""

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

from freq_hrl.experiments.trading.strong_learned_baseline_validation import (  # noqa: E402
    ALL_POLICY_MODES,
    DEFAULT_EVAL_SEEDS,
    DEFAULT_OPTIMIZER_SEEDS,
    DEFAULT_POLICY_MODES,
    DEFAULT_ROLLOUT_SEED_ROOTS,
    DEFAULT_SCENARIOS,
    DEFAULT_VALIDATION_SEEDS,
    POLICY_MODES,
    SCENARIOS,
    merge_strong_learned_baseline_shards,
    write_outputs,
)
from freq_hrl.experiments.trading.hyperparameter_pilot import (  # noqa: E402
    frozen_config_sha256,
    load_frozen_config,
)
from freq_hrl.experiments.reproducibility import (  # noqa: E402
    git_source_manifest_sha256,
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
    "Confirmatory actor-critic cells are independent, CPU-bound, and "
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


def source_identity() -> tuple[str, str]:
    return registered_git_source_identity(ROOT, Path("freq_hrl"))


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


def resolved_hyperparameters(
    args: argparse.Namespace,
    mode: str,
) -> tuple[dict[str, object], str]:
    selected = getattr(args, "frozen_selected", {}) or {}
    if mode in selected:
        entry = selected[mode]
        return dict(entry["parameters"]), str(entry["candidate_id"])
    if mode in POLICY_MODES:
        return {
            "hidden_dim": int(args.ppo_hidden_dim),
            "learning_rate": float(args.ppo_learning_rate),
            "epochs": int(args.ppo_epochs),
            "minibatch_size": int(args.ppo_minibatch_size),
            "init_log_std": float(args.ppo_init_log_std),
            "reward_scale": float(args.training_reward_scale),
        }, "exploratory_manual"
    return {
        "hidden_dim": int(args.offpolicy_hidden_dim),
        "learning_rate": float(args.offpolicy_learning_rate),
        "replay_capacity": int(args.offpolicy_replay_capacity),
        "warmup_steps": int(args.offpolicy_warmup_steps),
        "batch_size": int(args.offpolicy_batch_size),
        "updates_per_step": int(args.offpolicy_updates_per_step),
        "reward_scale": float(args.training_reward_scale),
    }, "exploratory_manual"


def build_training_command(
    args: argparse.Namespace,
    *,
    scenario: str,
    mode: str,
    replicate_seed: int,
    output_dir: Path,
) -> str:
    parameters, candidate_id = resolved_hyperparameters(args, mode)
    candidate_parameters_sha256 = frozen_config_sha256(parameters)
    is_ppo = mode in POLICY_MODES
    ppo_parameters = parameters if is_ppo else {
        "hidden_dim": args.ppo_hidden_dim,
        "learning_rate": args.ppo_learning_rate,
        "epochs": args.ppo_epochs,
        "minibatch_size": args.ppo_minibatch_size,
        "init_log_std": args.ppo_init_log_std,
    }
    offpolicy_parameters = parameters if not is_ppo else {
        "hidden_dim": args.offpolicy_hidden_dim,
        "learning_rate": args.offpolicy_learning_rate,
        "replay_capacity": args.offpolicy_replay_capacity,
        "warmup_steps": args.offpolicy_warmup_steps,
        "batch_size": args.offpolicy_batch_size,
        "updates_per_step": args.offpolicy_updates_per_step,
    }
    command = [
        str(args.python_executable),
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
        str(ppo_parameters["hidden_dim"]),
        "--ppo-learning-rate",
        str(ppo_parameters["learning_rate"]),
        "--ppo-epochs",
        str(ppo_parameters["epochs"]),
        "--ppo-minibatch-size",
        str(ppo_parameters["minibatch_size"]),
        "--ppo-init-log-std",
        str(ppo_parameters["init_log_std"]),
        "--training-reward-scale",
        str(parameters.get("reward_scale", args.training_reward_scale)),
        "--offpolicy-hidden-dim",
        str(offpolicy_parameters["hidden_dim"]),
        "--offpolicy-learning-rate",
        str(offpolicy_parameters["learning_rate"]),
        "--offpolicy-replay-capacity",
        str(offpolicy_parameters["replay_capacity"]),
        "--offpolicy-warmup-steps",
        str(offpolicy_parameters["warmup_steps"]),
        "--offpolicy-batch-size",
        str(offpolicy_parameters["batch_size"]),
        "--offpolicy-updates-per-step",
        str(offpolicy_parameters["updates_per_step"]),
        "--hyperparameter-source",
        (
            "frozen_nested_validation"
            if getattr(args, "confirmatory", False)
            else "exploratory_unfrozen"
        ),
        "--frozen-config-sha256",
        str(getattr(args, "frozen_config_sha256", "")),
        "--selected-candidate-id",
        candidate_id,
        "--frozen-candidate-parameters-sha256",
        candidate_parameters_sha256,
        "--code-revision",
        str(getattr(args, "code_revision", "")),
        "--source-manifest-sha256",
        str(getattr(args, "source_manifest_sha256", "")),
        "--output-dir",
        str(output_dir),
    ]
    if getattr(args, "confirmatory", False):
        command.append("--confirmatory")
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
    scenario: str,
    mode: str,
    replicate_seed: int,
) -> list[str]:
    spec = build_scheduler_spec(
        args,
        scenario=scenario,
        mode=mode,
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
    scenario: str,
    mode: str,
    replicate_seed: int,
) -> dict[str, object]:
    relative_dir = cell_relative_dir(
        args.run_name, scenario, mode, replicate_seed
    )
    absolute_dir = ROOT / relative_dir
    return {
        "project": str(args.project),
        "description": (
            f"Freq-HRL confirmatory {scenario} {mode} replicate {replicate_seed}"
        ),
        "cmd": build_training_command(
            args,
            scenario=scenario,
            mode=mode,
            replicate_seed=replicate_seed,
            output_dir=relative_dir,
        ),
        "cwd": str(ROOT / str(args.launch_subdir)),
        "signature": (
            f"Freq-HRL/strong-v3/{args.run_name}/"
            f"{getattr(args, 'frozen_config_sha256', '')[:12] or getattr(args, 'source_manifest_sha256', '')[:12] or 'exploratory'}/"
            f"{scenario}/{mode}/rep-{replicate_seed}"
        ),
        "resource_family": "Freq-HRL/strong-v3/single-cell",
        "vram": 0,
        "ram_mb": int(args.ram_mb),
        "cpu": int(args.cpu),
        "priority": str(args.priority),
        "ckpt_dir": str(absolute_dir / "checkpoints"),
        "ckpt_glob": "*.pt",
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
        ROOT / cell_relative_dir(args.run_name, scenario, mode, seed)
        for scenario, mode, seed in experiment_cells(
            args.scenarios, args.policy_modes, args.optimizer_seeds
        )
    ]


def cells_without_local_results(
    cells: list[tuple[str, str, int]],
    *,
    run_name: str,
    root: Path = ROOT,
) -> list[tuple[str, str, int]]:
    return [
        cell
        for cell in cells
        if not (
            Path(root)
            / cell_relative_dir(run_name, cell[0], cell[1], cell[2])
            / "per_seed.csv"
        ).exists()
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
    parser.add_argument("--training-reward-scale", type=float, default=100.0)
    parser.add_argument("--offpolicy-hidden-dim", type=int, default=64)
    parser.add_argument("--offpolicy-learning-rate", type=float, default=3e-4)
    parser.add_argument("--offpolicy-replay-capacity", type=int, default=100_000)
    parser.add_argument("--offpolicy-warmup-steps", type=int, default=2048)
    parser.add_argument("--offpolicy-batch-size", type=int, default=64)
    parser.add_argument("--offpolicy-updates-per-step", type=int, default=1)
    parser.add_argument(
        "--frozen-config",
        type=Path,
        help="Final nested-validation frozen_config.json; required outside smoke mode.",
    )
    parser.add_argument("--nodes", default=",".join(DEFAULT_NODES))
    parser.add_argument("--python-executable", default="")
    parser.add_argument("--launch-subdir", choices=(".", "scripts"), default="scripts")
    parser.add_argument("--project", default="Freq-HRL-Confirmatory")
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
    parser.add_argument("--skip-complete-cells", action="store_true")
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
    unknown_nodes = sorted(set(args.nodes) - set(SUPPORTED_NODES))
    if unknown_modes or unknown_scenarios or unknown_nodes:
        raise SystemExit(
            "invalid matrix selection: "
            f"modes={unknown_modes}, scenarios={unknown_scenarios}, nodes={unknown_nodes}"
        )
    if not str(args.python_executable).strip():
        try:
            args.python_executable = default_python_executable(args.nodes)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    current_revision = ""
    current_manifest = ""
    if not args.merge_only:
        try:
            current_revision, current_manifest = source_identity()
        except (
            OSError,
            RuntimeError,
            ValueError,
            subprocess.CalledProcessError,
        ) as exc:
            raise SystemExit(
                f"cannot register learned-baseline source identity: {exc}"
            ) from exc
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
        args.confirmatory = False
        args.frozen_selected = {}
        args.frozen_config_sha256 = ""
        args.code_revision = current_revision
        args.source_manifest_sha256 = current_manifest
    elif not args.merge_only:
        if args.frozen_config is None:
            raise SystemExit(
                "confirmatory submission requires --frozen-config from final nested HPO"
            )
        path = Path(args.frozen_config).expanduser()
        if not path.is_absolute():
            path = path.resolve() if path.exists() else (ROOT / path).resolve()
        try:
            _, audit = load_frozen_config(
                path, required_policy_modes=args.policy_modes
            )
        except (OSError, ValueError) as exc:
            raise SystemExit(f"invalid frozen config: {exc}") from exc
        args.confirmatory = True
        args.frozen_selected = audit["selected"]
        args.frozen_config_sha256 = str(audit["sha256"])
        frozen_revision = str(audit["code_revision"])
        frozen_manifest = str(audit["source_manifest_sha256"])
        try:
            committed_manifest = git_source_manifest_sha256(
                ROOT,
                Path("freq_hrl"),
                revision=frozen_revision,
            )
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            raise SystemExit(
                f"frozen source revision is unavailable or invalid: {exc}"
            ) from exc
        if committed_manifest != frozen_manifest:
            raise SystemExit(
                "frozen source manifest does not match its registered Git revision"
            )
        if current_manifest != frozen_manifest:
            raise SystemExit(
                "current freq_hrl source bytes differ from the nested-HPO freeze"
            )
        args.code_revision = frozen_revision
        args.source_manifest_sha256 = frozen_manifest
        args.frozen_config = path
    else:
        args.confirmatory = False
        args.frozen_selected = {}
        args.frozen_config_sha256 = ""
        args.code_revision = ""
        args.source_manifest_sha256 = ""
    return args


def main() -> None:
    args = normalize_args(build_parser().parse_args())
    if args.merge_only:
        merge_results(args)
        return
    cells = experiment_cells(
        args.scenarios, args.policy_modes, args.optimizer_seeds
    )
    if args.skip_complete_cells:
        requested_count = len(cells)
        cells = cells_without_local_results(cells, run_name=args.run_name)
        print(
            f"skipped {requested_count - len(cells)} cells with local results",
            flush=True,
        )
    if int(args.max_cells) > 0:
        cells = cells[: int(args.max_cells)]
    if not cells:
        print("no confirmatory cells require submission", flush=True)
        return
    print(
        f"run={args.run_name} cells={len(cells)} nodes={','.join(args.nodes)} "
        f"iterations={args.iterations} steps={args.steps} "
        f"frozen={args.frozen_config_sha256[:12] or 'exploratory'}",
        flush=True,
    )
    specs = [
        build_scheduler_spec(
            args,
            scenario=scenario,
            mode=mode,
            replicate_seed=seed,
        )
        for scenario, mode, seed in cells
    ]
    execute_bulk(
        specs,
        dry_run=bool(args.dry_run),
        intent_label=f"Freq-HRL confirmatory {args.run_name}",
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
