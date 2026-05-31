"""Merge sharded promotion-sweep outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .promotion_sweep import write_report


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def float_key(row: dict[str, Any], key: str) -> float:
    return float(row.get(key, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/trading_promotion_recovery_sweep"))
    args = parser.parse_args()

    per_seed: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for directory in args.input_dirs:
        per_seed.extend(read_csv(directory / "per_seed.csv"))
        summary.extend(read_csv(directory / "summary.csv"))
    if not summary:
        raise ValueError("no promotion summary rows to merge")

    per_seed.sort(key=lambda row: (
        float_key(row, "promotion_threshold"),
        float_key(row, "promotion_ratio"),
        float_key(row, "promotion_mid_gain"),
        float_key(row, "promotion_adapt_gain"),
        int(float(row.get("seed", 0))),
        str(row.get("baseline", "")),
    ))
    summary.sort(key=lambda row: (
        float_key(row, "promotion_threshold"),
        float_key(row, "promotion_ratio"),
        float_key(row, "promotion_mid_gain"),
        float_key(row, "promotion_adapt_gain"),
    ))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_seed.csv", per_seed)
    write_csv(args.output_dir / "summary.csv", summary)
    best = max(summary, key=lambda row: float_key(row, "sharpe_delta"))
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "best": best}, f, indent=2)

    seeds = sorted({int(float(row.get("seed", 0))) for row in per_seed})
    scenarios = sorted({str(row.get("scenario", "persistent_shift")) for row in summary})
    scenario = scenarios[0] if len(scenarios) == 1 else "mixed"
    report_rows = [
        {key: float(value) if key not in {"scenario"} else value for key, value in row.items()}
        for row in summary
    ]
    write_report(args.output_dir / "report.md", report_rows, seeds, args.steps, scenario)
    print(f"wrote {args.output_dir}")
    print(
        "promotion_merge "
        f"rows={len(summary)} best_threshold={float(best['promotion_threshold']):.5f} "
        f"best_sharpe_delta={float(best['sharpe_delta']):+.3f}"
    )


if __name__ == "__main__":
    main()
