#!/usr/bin/env python3
"""Aggregate compact per-seed held-out JSON summaries for the final paper."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from scripts.heldout_final_eval import MAIN_METHODS, METRIC_KEYS, PROTOCOL_VERSION, SEEDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results_remote/revision3_heldout")
    parser.add_argument("--output-dir", default="results_remote/revision3_heldout")
    args = parser.parse_args()

    input_dir = SCRIPT_DIR / args.input_dir
    output_dir = SCRIPT_DIR / args.output_dir
    rows = []
    for method in MAIN_METHODS:
        payloads = []
        for seed in SEEDS:
            path = input_dir / method / f"seed{seed}.json"
            if not path.exists():
                raise FileNotFoundError(f"missing held-out summary: {path}")
            payload = json.loads(path.read_text())
            if payload.get("protocol_version") != PROTOCOL_VERSION:
                raise ValueError(f"unexpected protocol in {path}")
            if payload.get("training_seed") != seed or payload.get("method") != method:
                raise ValueError(f"mismatched held-out summary: {path}")
            payloads.append(payload)
        row = {"method": method, "n_seeds": len(payloads),
               "n_test_episodes_per_seed": payloads[0]["n_test_episodes"]}
        for metric in METRIC_KEYS:
            values = np.asarray([payload["metrics"][f"{metric}_mean"] for payload in payloads],
                                dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_seed_sd"] = float(values.std(ddof=1))
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "main_results_heldout.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "n_methods": len(rows),
        "n_seeds": len(SEEDS),
        "n_test_episodes_per_seed": rows[0]["n_test_episodes_per_seed"],
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in rows:
        print(
            f"{row['method']}: composite={row['composite_mean']:.3f} "
            f"+/- {row['composite_seed_sd']:.3f}, "
            f"wait={row['wait_mean']:.2f} +/- {row['wait_seed_sd']:.2f}"
        )


if __name__ == "__main__":
    main()
