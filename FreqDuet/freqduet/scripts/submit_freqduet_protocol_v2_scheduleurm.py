#!/usr/bin/env python3
"""Submit FreqDuet train/frozen-evaluation jobs directly to CPU nodes."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = Path("/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py")
LOCAL_WORKSPACE_ROOT = Path("/home/erzhu419/mine_code")
REMOTE_WORKSPACE_ROOT = Path("/home/zhengliang01/scheduleurm_work")
try:
    REMOTE_ROOT = REMOTE_WORKSPACE_ROOT / ROOT.relative_to(LOCAL_WORKSPACE_ROOT)
except ValueError:
    REMOTE_ROOT = Path(
        "/home/zhengliang01/scheduleurm_work/TransitDuet/FreqDuet/freqduet")
REMOTE_PYTHON = Path("/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python")
DEFAULT_CONFIGS = [
    f"F_freqduet_protocol_v6_{name}_hiro"
    for name in (
        "main", "nofreq", "rawhistory", "allfreq", "upperonly",
        "loweronly", "swapped", "nobudget", "noguard", "noloadcost",
        "waitonlycredit", "csac",
    )
]
DEFAULT_TRAIN_SEEDS = [503, 521, 541, 557]
DEFAULT_EVAL_SEEDS = [41011, 41017, 41023, 41039]
DEFAULT_NODES = ["node001", "node002", "node003", "node004", "node005", "node006"]
CPU_JUSTIFICATION = (
    "FreqDuet transit simulation and reinforcement learning are "
    "CPU-only and run in the isolated freqduet-cpu-py310 environment."
)
SUBMITTED_TASK_RE = re.compile(r"\bsubmitted\s+(t\d+)\b", re.IGNORECASE)


def parse_csv(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def ranges(total: int, size: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + max(1, int(size)), total))
        for start in range(0, total, max(1, int(size)))
    ]


def resolve_reference(configs: list[str], reference: str | None) -> str:
    if not configs:
        raise ValueError("at least one config is required")
    requested = Path(reference or configs[0]).stem
    for config in configs:
        if Path(config).stem == requested:
            return config
    raise ValueError("reference config must be included in --configs")


def protocol_label(configs: list[str]) -> str:
    versions = {
        match.group(1)
        for config in configs
        if (match := re.search(r"protocol_v(\d+)", Path(config).stem))
    }
    if len(versions) == 1:
        return f"protocol-v{versions.pop()}"
    return "mixed-protocol"


def execute(
    command: list[str],
    dry_run: bool,
    input_text: str | None = None,
) -> str:
    print(shlex.join(command))
    if dry_run:
        return ""
    process = subprocess.run(
        command,
        text=True,
        input=input_text,
        capture_output=True,
    )
    output = (process.stdout or "") + (process.stderr or "")
    if process.returncode != 0:
        if "duplicate" not in output.lower() and "already queued" not in output.lower():
            print(output, file=sys.stderr)
            process.check_returncode()
    if output.strip():
        print(output.strip())
    return output


def git_output(*args: str) -> str:
    process = subprocess.run(
        ["git", *args], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "git command failed")
    return process.stdout.strip()


def preflight_source(
    configs: list[str], protocol: str, require_clean: bool,
    expected_commit: str | None, allow_experimental_configs: bool = False,
) -> str:
    commit = git_output("rev-parse", "HEAD")
    if expected_commit and commit != str(expected_commit).strip():
        raise RuntimeError(
            f"source commit {commit} does not match {expected_commit}")
    if require_clean:
        status = git_output("status", "--porcelain")
        if status:
            raise RuntimeError(
                "submission source is dirty; use an immutable worktree snapshot")
    validator = None
    if protocol == "protocol-v4":
        validator = "validate_freqduet_protocol_v4_configs.py"
    elif protocol == "protocol-v6":
        validator = "validate_freqduet_protocol_v6_configs.py"
    if validator is not None:
        names = [
            name if str(name).endswith(".yaml") else f"{name}.yaml"
            for name in configs
        ]
        if protocol == "protocol-v6":
            names = [Path(name).stem for name in names]
        validator_command = [
            sys.executable,
            str(ROOT / "scripts" / validator),
        ]
        if protocol == "protocol-v6" and allow_experimental_configs:
            validator_command.append("--allow-experimental")
        validator_command.extend(names)
        process = subprocess.run(
            validator_command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"{protocol} config preflight failed:\n"
                + (process.stdout or ""))
        if process.stdout.strip():
            print(process.stdout.strip())
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument(
        "--reference",
        default=None,
        help="Reference config for paired aggregation; defaults to the first config.",
    )
    parser.add_argument("--train-seeds", default=",".join(map(str, DEFAULT_TRAIN_SEEDS)))
    parser.add_argument("--eval-seeds", default=",".join(map(str, DEFAULT_EVAL_SEEDS)))
    parser.add_argument("--train-episodes", type=int, default=40)
    parser.add_argument(
        "--stage",
        choices=["development", "confirmation", "exploratory"],
        default="development",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--shard-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--ram-mb", type=int, default=32768)
    parser.add_argument("--nodes", default=",".join(DEFAULT_NODES))
    parser.add_argument("--remote-root", default=str(REMOTE_ROOT))
    parser.add_argument("--require-clean-source", action="store_true")
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--priority", choices=["low", "normal", "high"], default="normal")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument(
        "--serial-submit",
        action="store_true",
        help="Use one scheduler lock per shard instead of submit-jsonl.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-duplicate", action="store_true")
    parser.add_argument(
        "--allow-experimental-configs",
        action="store_true",
        help="Allow registered V6 exploratory configs; requires --stage exploratory.",
    )
    args = parser.parse_args()

    configs = parse_csv(args.configs)
    try:
        reference = resolve_reference(configs, args.reference)
    except ValueError as exc:
        parser.error(str(exc))
    train_seeds = parse_csv(args.train_seeds, int)
    eval_seeds = parse_csv(args.eval_seeds, int)
    nodes = parse_csv(args.nodes)
    for label, values in [
        ("configs", configs),
        ("train seeds", train_seeds),
        ("eval seeds", eval_seeds),
        ("nodes", nodes),
    ]:
        if not values:
            parser.error(f"{label} must not be empty")
        if len(values) != len(set(values)):
            parser.error(f"{label} must be unique")
    for label, value in [
        ("train episodes", args.train_episodes),
        ("shard size", args.shard_size),
        ("workers", args.workers),
        ("cpu", args.cpu),
        ("RAM", args.ram_mb),
    ]:
        if int(value) <= 0:
            parser.error(f"{label} must be positive")
    if protocol_label(configs) == "protocol-v6" and set(train_seeds) & set(
            eval_seeds):
        parser.error("V6 train and evaluation seed sets must be disjoint")
    if args.allow_experimental_configs and args.stage != "exploratory":
        parser.error(
            "--allow-experimental-configs requires --stage exploratory")
    total = len(configs) * len(train_seeds)
    shards = ranges(total, args.shard_size)
    result_base = f"results_freqduet/{args.run_name}"
    protocol = protocol_label(configs)
    remote_root = Path(args.remote_root)
    try:
        commit = preflight_source(
            configs, protocol, args.require_clean_source,
            args.expected_commit,
            allow_experimental_configs=args.allow_experimental_configs,
        )
    except RuntimeError as exc:
        parser.error(str(exc))
    print(
        f"{protocol} run={args.run_name} jobs={total} shards={len(shards)} "
        f"train_episodes={args.train_episodes} eval_seeds={len(eval_seeds)}")
    print(f"source_commit={commit} local_root={ROOT} remote_root={remote_root}")
    source_branch = git_output("rev-parse", "--abbrev-ref", "HEAD")
    source_tracked_dirty = bool(git_output(
        "status", "--porcelain", "--untracked-files=no"))

    submitted_task_ids: list[str] = []
    bulk_specs: list[dict[str, object]] = []
    for index, (start, end) in enumerate(shards):
        shard_id = f"{start:04d}_{end:04d}"
        logs_dir = f"{result_base}/logs_shards/shard_{shard_id}"
        out_dir = f"{result_base}/shard_summaries/shard_{shard_id}"
        inner = [
            "PYTHONPATH=.",
            "OMP_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
            "TORCH_NUM_THREADS=1",
            "FREQDUET_TORCH_THREADS=1",
            f"FREQDUET_SOURCE_COMMIT={shlex.quote(commit)}",
            f"FREQDUET_SOURCE_BRANCH={shlex.quote(source_branch)}",
            "FREQDUET_SOURCE_TRACKED_DIRTY="
            f"{int(source_tracked_dirty)}",
            shlex.join([
                str(REMOTE_PYTHON),
                "-u",
                "scripts/run_freqduet_protocol_v2_matrix.py",
                "--configs", ",".join(configs),
                "--reference", reference,
                "--train-seeds", ",".join(map(str, train_seeds)),
                "--eval-seeds", ",".join(map(str, eval_seeds)),
                "--train-episodes", str(args.train_episodes),
                "--stage", args.stage,
                "--workers", str(args.workers),
                "--worker-threads", "1",
                "--logs-dir", logs_dir,
                "--out-dir", out_dir,
                "--job-start", str(start),
                "--job-end", str(end),
                "--skip-existing",
                "--suppress-heavy-artifacts",
            ]),
        ]
        node = nodes[index % len(nodes)]
        command = [
            sys.executable,
            str(SCHEDULER),
            "submit",
            "--project", "FreqDuet",
            "--description", f"FreqDuet {protocol} {args.run_name} shard {index + 1}/{len(shards)}",
            "--cmd", " ".join(inner),
            "--cwd", str(ROOT),
            "--signature", f"FreqDuet/{args.run_name}/shard_{shard_id}",
            "--vram", "0",
            "--ram-mb", str(args.ram_mb),
            "--cpu", str(args.cpu),
            "--priority", args.priority,
            "--require-node", node,
            "--result-dir", str(remote_root / logs_dir),
            "--local-result-dir", str(ROOT / logs_dir),
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
        bulk_specs.append({
            "project": "FreqDuet",
            "description": (
                f"FreqDuet {protocol} {args.run_name} "
                f"shard {index + 1}/{len(shards)}"
            ),
            "cmd": " ".join(inner),
            "cwd": str(ROOT),
            "signature": f"FreqDuet/{args.run_name}/shard_{shard_id}",
            "vram": 0,
            "ram_mb": int(args.ram_mb),
            "cpu": int(args.cpu),
            "priority": args.priority,
            "require_node": node,
            "result_dir": str(remote_root / logs_dir),
            "local_result_dir": str(ROOT / logs_dir),
            "skip_resume_scan": True,
            "allow_cpu_training": True,
            "cpu_training_justification": CPU_JUSTIFICATION,
            "allow_remote_large_data": True,
            "reroute_on_node_down": True,
            "node_down_requeue_s": 900,
            "allow_duplicate": bool(args.allow_duplicate),
        })
        if args.dry_run or args.serial_submit:
            output = execute(command, args.dry_run)
            submitted_task_ids.extend(SUBMITTED_TASK_RE.findall(output))

    if not args.dry_run and not args.serial_submit:
        output = execute(
            [
                sys.executable,
                str(SCHEDULER),
                "submit-jsonl",
                "--stdin",
                "--trusted",
                "--json",
                "--lock-timeout", "600",
                "--intent-label", f"FreqDuet/{args.run_name}",
            ],
            False,
            input_text=json.dumps(bulk_specs),
        )
        try:
            payload = json.loads(output)
            submitted_task_ids = [
                str(item["id"]) for item in payload.get("submitted", [])
            ]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(
                "scheduler submit-jsonl did not return a valid task-id payload"
            ) from exc
        if len(submitted_task_ids) != len(bulk_specs):
            raise RuntimeError(
                "scheduler submit-jsonl count mismatch: "
                f"expected {len(bulk_specs)}, got {len(submitted_task_ids)}"
            )

    print("\nAggregate after scheduler sync:")
    print(
        "python3 scripts/run_freqduet_protocol_v2_matrix.py --aggregate-only "
        f"--configs {shlex.quote(','.join(configs))} "
        f"--reference {shlex.quote(reference)} "
        f"--train-seeds {shlex.quote(','.join(map(str, train_seeds)))} "
        f"--eval-seeds {shlex.quote(','.join(map(str, eval_seeds)))} "
        f"--train-episodes {int(args.train_episodes)} "
        f"--stage {shlex.quote(args.stage)} "
        f"--logs-dir {result_base}/logs_shards/shard_0000_0000 "
        f"--aggregate-logs-dirs \"$(find {result_base}/logs_shards -mindepth 1 -maxdepth 1 -type d | sort | paste -sd, -)\" "
        f"--out-dir {result_base}/combined_summary")
    if args.dispatch and not args.dry_run:
        if not submitted_task_ids:
            print("No newly submitted task IDs to dispatch; scheduler watch will handle active duplicates.")
            return
        dispatch = [
            sys.executable,
            str(SCHEDULER),
            "dispatch",
            "--lock-timeout", "600",
            "--intent-label", f"FreqDuet/{args.run_name}",
        ]
        for task_id in submitted_task_ids:
            dispatch.extend(["--task-id", task_id])
        execute(dispatch, False)


if __name__ == "__main__":
    main()
