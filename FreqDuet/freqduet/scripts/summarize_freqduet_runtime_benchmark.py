#!/usr/bin/env python3
"""Summarize the six original/optimized frozen-rollout benchmark pairs."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    root = parser.parse_args().root
    rows = []
    for path in sorted((root / "optimized").glob("*/summary.json")):
        item = json.loads(path.read_text())
        comparison = item["comparison"]
        if not comparison["exact_metrics_and_action_reward_traces"]:
            raise ValueError(f"rollout differs: {path}")
        rows.append(dict(config=item["config"], eval_seed=item["eval_seed"],
                         original_s=comparison["reference_elapsed_s"],
                         optimized_s=item["elapsed_s"], speedup=comparison["speedup"],
                         exact_columns=comparison["metric_columns"]))
    if len(rows) != 6:
        raise ValueError(f"expected six benchmark pairs, found {len(rows)}")
    result = dict(cases=rows, all_exact=True,
                  total_original_s=sum(row["original_s"] for row in rows),
                  total_optimized_s=sum(row["optimized_s"] for row in rows))
    result["aggregate_speedup"] = result["total_original_s"] / result["total_optimized_s"]
    (root / "benchmark_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
