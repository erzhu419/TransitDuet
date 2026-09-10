#!/usr/bin/env python3
"""Submit the frozen V33 timescale development matrix through scheduleurm."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_protocol_v6_v33_timescale_screen import (  # noqa: E402
    DISCOVERY_EVAL_SEEDS,
    DISCOVERY_TRAIN_SEEDS,
    EXPECTED_CONFIGS,
    RUN_NAME,
    TRAIN_EPISODES,
)


GENERIC_SUBMITTER = ROOT / "scripts/submit_freqduet_protocol_v2_scheduleurm.py"
DEFAULT_NODES = [
    "node001", "node002", "node003", "node004", "node005", "node006",
]


def csv(values) -> str:
    return ",".join(str(value) for value in values)


def source_commit() -> str:
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        [
            "git", "-C", str(ROOT), "status", "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()
    if dirty:
        raise SystemExit("V33 submission requires a clean tracked source snapshot")
    return commit


def build_submit_command(
    *,
    commit: str,
    nodes: list[str],
    dispatch: bool,
    dry_run: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(GENERIC_SUBMITTER),
        "--configs", csv(EXPECTED_CONFIGS),
        "--reference", EXPECTED_CONFIGS[1],
        "--train-seeds", csv(DISCOVERY_TRAIN_SEEDS),
        "--eval-seeds", csv(DISCOVERY_EVAL_SEEDS),
        "--train-episodes", str(TRAIN_EPISODES),
        "--stage", "exploratory",
        "--run-name", RUN_NAME,
        "--shard-size", "8",
        "--workers", "8",
        "--cpu", "8",
        "--ram-mb", "24576",
        "--nodes", csv(nodes),
        "--result-sync", "summary",
        "--priority", "normal",
        "--require-clean-source",
        "--expected-commit", commit,
        "--allow-experimental-configs",
    ]
    if dispatch:
        command.append("--dispatch")
    if dry_run:
        command.append("--dry-run")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default=csv(DEFAULT_NODES))
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    nodes = [item.strip() for item in args.nodes.split(",") if item.strip()]
    unknown = sorted(set(nodes) - set(DEFAULT_NODES))
    if not nodes or unknown:
        parser.error(f"nodes must be a nonempty subset of {DEFAULT_NODES}")
    commit = source_commit()
    if args.expected_commit and commit != args.expected_commit:
        parser.error(f"source commit {commit} != {args.expected_commit}")
    command = build_submit_command(
        commit=commit,
        nodes=nodes,
        dispatch=args.dispatch,
        dry_run=args.dry_run,
    )
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
