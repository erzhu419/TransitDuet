#!/usr/bin/env python3
"""Submit the single authorized V33 confirmation through scheduleurm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_protocol_v6_v33_timescale_screen import (  # noqa: E402
    CONFIRMATION_EVAL_SEEDS,
    CONFIRMATION_RUN_NAME,
    CONFIRMATION_TRAIN_SEEDS,
    DEFAULT_REFERENCE,
    TRAIN_EPISODES,
    confirmation_configs,
    validate_development_authorization,
)
from scripts.submit_freqduet_protocol_v6_v33_timescale_scheduleurm import (  # noqa: E402
    DEFAULT_NODES,
    GENERIC_SUBMITTER,
    csv,
    source_commit,
)


def authorized_candidate(gate_path: Path) -> str:
    gate_path = Path(gate_path).resolve()
    try:
        gate = json.loads(gate_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read V33 development gate: {exc}") from exc
    return validate_development_authorization(gate)


def build_submit_command(
    *,
    commit: str,
    gate_path: Path,
    nodes: list[str],
    dispatch: bool,
    dry_run: bool,
) -> list[str]:
    selected = authorized_candidate(gate_path)
    configs = confirmation_configs(selected)
    command = [
        sys.executable,
        str(GENERIC_SUBMITTER),
        "--configs", csv(configs),
        "--reference", DEFAULT_REFERENCE,
        "--train-seeds", csv(CONFIRMATION_TRAIN_SEEDS),
        "--eval-seeds", csv(CONFIRMATION_EVAL_SEEDS),
        "--train-episodes", str(TRAIN_EPISODES),
        "--stage", "confirmation",
        "--run-name", CONFIRMATION_RUN_NAME,
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
        "--v33-development-gate", str(Path(gate_path).resolve()),
    ]
    if dispatch:
        command.append("--dispatch")
    if dry_run:
        command.append("--dry-run")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-gate", type=Path, required=True)
    parser.add_argument("--nodes", default=csv(DEFAULT_NODES))
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    nodes = [item.strip() for item in args.nodes.split(",") if item.strip()]
    unknown = sorted(set(nodes) - set(DEFAULT_NODES))
    if not nodes or unknown:
        parser.error(f"nodes must be a nonempty subset of {DEFAULT_NODES}")
    try:
        authorized_candidate(args.development_gate)
    except ValueError as exc:
        parser.error(str(exc))
    commit = source_commit()
    if args.expected_commit and commit != args.expected_commit:
        parser.error(f"source commit {commit} != {args.expected_commit}")
    command = build_submit_command(
        commit=commit,
        gate_path=args.development_gate,
        nodes=nodes,
        dispatch=args.dispatch,
        dry_run=args.dry_run,
    )
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
