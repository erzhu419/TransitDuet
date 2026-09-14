#!/usr/bin/env python3
"""Profile one MuJoCo cell while exporting only compact timing summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pstats
import subprocess
import sys
import time


MODULE = "freq_hrl.experiments.mujoco.control_validation"


def _function_rows(stats: pstats.Stats) -> list[dict[str, object]]:
    rows = []
    for (filename, line, function), values in stats.stats.items():
        primitive_calls, total_calls, self_seconds, cumulative_seconds, _ = values
        rows.append({
            "file": filename,
            "line": int(line),
            "function": function,
            "primitive_calls": int(primitive_calls),
            "total_calls": int(total_calls),
            "self_seconds": float(self_seconds),
            "cumulative_seconds": float(cumulative_seconds),
        })
    return rows


def _top(rows: list[dict[str, object]], key: str, limit: int = 80) -> list[dict[str, object]]:
    return sorted(rows, key=lambda row: float(row[key]), reverse=True)[:limit]


def _write_summary(profile_path: Path, export_dir: Path, wall_seconds: float) -> None:
    stats = pstats.Stats(str(profile_path))
    rows = _function_rows(stats)
    payload = {
        "evidence_role": "efficiency_diagnostic_only_not_algorithm_evidence",
        "profile_file_server_only": str(profile_path),
        "wall_seconds": float(wall_seconds),
        "profile_total_seconds": float(stats.total_tt),
        "total_calls": int(stats.total_calls),
        "primitive_calls": int(stats.prim_calls),
        "top_cumulative": _top(rows, "cumulative_seconds"),
        "top_self": _top(rows, "self_seconds"),
    }
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "profile_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        f"wall_seconds={wall_seconds:.6f}",
        f"profile_total_seconds={stats.total_tt:.6f}",
        f"calls={stats.total_calls} primitive_calls={stats.prim_calls}",
        "",
        "top cumulative",
    ]
    for row in payload["top_cumulative"][:40]:
        lines.append(
            f"{row['cumulative_seconds']:12.6f} {row['self_seconds']:12.6f} "
            f"{row['total_calls']:10d} {row['file']}:{row['line']}({row['function']})"
        )
    lines.extend(("", "top self"))
    for row in payload["top_self"][:40]:
        lines.append(
            f"{row['self_seconds']:12.6f} {row['cumulative_seconds']:12.6f} "
            f"{row['total_calls']:10d} {row['file']}:{row['line']}({row['function']})"
        )
    (export_dir / "profile_top.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-output-dir", type=Path, required=True)
    parser.add_argument("--export-output-dir", type=Path, required=True)
    parser.add_argument("control_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    control_args = list(args.control_args)
    if control_args and control_args[0] == "--":
        control_args = control_args[1:]
    if not control_args or "--output-dir" in control_args:
        raise SystemExit("control arguments must be nonempty and must not set --output-dir")
    args.full_output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = args.full_output_dir / "cell.prof"
    cell_output = args.full_output_dir / "cell"
    started = time.perf_counter()
    subprocess.run([
        sys.executable, "-u", "-m", "cProfile", "-o", str(profile_path),
        "-m", MODULE, *control_args, "--output-dir", str(cell_output),
    ], check=True)
    wall_seconds = time.perf_counter() - started
    _write_summary(profile_path, args.export_output_dir, wall_seconds)
    source_summary = cell_output / "cell_summary.json"
    if not source_summary.is_file():
        raise RuntimeError("profiled cell did not produce cell_summary.json")
    (args.export_output_dir / "cell_summary.json").write_bytes(source_summary.read_bytes())
    print(f"DONE mujoco_cprofile wall_seconds={wall_seconds:.3f}", flush=True)


if __name__ == "__main__":
    main()
