"""Merge sharded trading pressure-test matrix outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .pressure_test_matrix import summarize, write_csv, write_report


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _coerce_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key in {"baseline", "scenario"}:
            out[key] = value
        elif key == "seed":
            out[key] = int(float(value))
        else:
            out[key] = float(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/trading_pressure_matrix"))
    parser.add_argument("--steps", type=int, default=720)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for input_dir in args.input_dirs:
        per_seed = input_dir / "per_seed.csv"
        if not per_seed.exists():
            raise FileNotFoundError(f"missing pressure shard output: {per_seed}")
        rows.extend(_coerce_row(row) for row in read_rows(per_seed))
    if not rows:
        raise ValueError("no pressure-test rows found")

    seeds = sorted({int(row["seed"]) for row in rows})
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_seed.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"per_seed": rows, "summary": summary}, f, indent=2)
    write_report(args.output_dir / "report.md", summary, seeds, args.steps)
    print(f"merged {len(args.input_dirs)} shards into {args.output_dir}")


if __name__ == "__main__":
    main()
