"""Merge FreqDuet ablation shard CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EXCLUDE_SUMMARY_FIELDS = {"config", "seed", "logs_dir", "episodes"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    configs = []
    for row in rows:
        cfg = row.get("config", "")
        if cfg and cfg not in configs:
            configs.append(cfg)
    metrics = [
        key for key in rows[0]
        if key not in EXCLUDE_SUMMARY_FIELDS and as_float(rows[0].get(key)) is not None
    ]
    out: list[dict[str, object]] = []
    for cfg in configs:
        group = [row for row in rows if row.get("config") == cfg]
        seeds = sorted({int(float(row["seed"])) for row in group if row.get("seed")})
        item: dict[str, object] = {"config": cfg, "n_seeds": len(seeds)}
        for metric in metrics:
            vals = [as_float(row.get(metric)) for row in group]
            arr = np.asarray([v for v in vals if v is not None], dtype=np.float64)
            if arr.size == 0:
                continue
            item[f"{metric}_mean"] = float(arr.mean())
            item[f"{metric}_std"] = float(arr.std(ddof=0))
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for directory in args.input_dirs:
        path = directory / "freqduet_ablation_per_seed.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(read_rows(path))
    if not rows:
        raise ValueError("no per-seed rows to merge")

    rows.sort(key=lambda row: (str(row.get("config", "")), int(float(row.get("seed", 0)))))
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "freqduet_ablation_per_seed.csv", rows)
    write_csv(args.output_dir / "freqduet_ablation_summary.csv", summary)
    with (args.output_dir / "freqduet_ablation_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_seed": rows}, f, indent=2)
    print(f"wrote {args.output_dir}")
    print(f"merged_configs={len(summary)} merged_rows={len(rows)}")


if __name__ == "__main__":
    main()
