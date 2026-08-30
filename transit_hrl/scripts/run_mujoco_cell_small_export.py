#!/usr/bin/env python3
"""Run one MuJoCo cell while exporting only analysis-sized artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


MODULE = "freq_hrl.experiments.mujoco.control_validation"
EXPORTED_FILES = ("cell_summary.json", "evaluation_rows.csv")
SERVER_ONLY_FILES = ("checkpoint.pt", "training_history.json")


def export_small_cell(
    full_output_dir: Path,
    export_output_dir: Path,
    *,
    server_full_output_dir: str,
) -> None:
    source = Path(full_output_dir)
    target = Path(export_output_dir)
    missing = [
        name
        for name in (*EXPORTED_FILES, *SERVER_ONLY_FILES)
        if not (source / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"MuJoCo cell did not produce required artifacts: {missing}")
    target.mkdir(parents=True, exist_ok=True)
    for name in SERVER_ONLY_FILES:
        stale = target / name
        if stale.exists():
            stale.unlink()
    for name in EXPORTED_FILES:
        shutil.copyfile(source / name, target / name)
    location = {
        "artifact_policy": "small_results_synced_full_training_artifacts_server_only_v1",
        "server_full_output_dir": str(server_full_output_dir),
        "exported_files": list(EXPORTED_FILES),
        "server_only_files": list(SERVER_ONLY_FILES),
    }
    (target / "server_artifact_location.json").write_text(
        json.dumps(location, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-output-dir", type=Path, required=True)
    parser.add_argument("--export-output-dir", type=Path, required=True)
    parser.add_argument("--server-full-output-dir", required=True)
    parser.add_argument("control_args", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    control_args = list(args.control_args)
    if control_args and control_args[0] == "--":
        control_args = control_args[1:]
    if not control_args or "--output-dir" in control_args:
        raise SystemExit("control arguments must be nonempty and must not set --output-dir")
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            MODULE,
            *control_args,
            "--output-dir",
            str(args.full_output_dir),
        ],
        check=True,
    )
    export_small_cell(
        args.full_output_dir,
        args.export_output_dir,
        server_full_output_dir=args.server_full_output_dir,
    )
    print(
        "mujoco_small_export status=valid "
        f"output={args.export_output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
