"""Summarize only compact frozen-pilot diagnostic outputs; never promote the recipe."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize(manifest, *, root):
    cells, missing = [], []
    for expected in manifest["cells"]:
        path = root / expected["output_directory"] / "public/result.json"
        if not path.exists():
            missing.append({"task": expected["task"], "optimizer_seed": expected["optimizer_seed"], "status": "missing"})
            continue
        row = json.loads(path.read_text())
        if (row["run_name"] != manifest["run_name"] or row["task"] != expected["task"]
            or row["optimizer_seed"] != expected["optimizer_seed"] or row["evaluation_seeds"] != expected["evaluation_seeds"]
            or row["probe_seeds"] != expected["probe_seeds"]
            or row["source_run_ids"] != {m: s["run_id"] for m, s in expected["sources"].items()}):
            raise ValueError("diagnostic output differs from its frozen sources or path roster")
        if row["status"] != "complete":
            missing.append({"task": row["task"], "optimizer_seed": row["optimizer_seed"],
                            "status": row["status"], "error": row.get("error")})
            continue
        cells.append({"source_file": str(path), **{k: row[k] for k in
            ("task", "optimizer_seed", "episode_panels", "local_errors", "cross_future_attribution",
             "termination_development_eligible", "evaluation_cost")},
            "occupancy": row["occupancy"]})
    task_means = {}
    for task in sorted({c["task"] for c in manifest["cells"]}):
        rows = [c for c in cells if c["task"] == task]
        expected_count = sum(c["task"] == task for c in manifest["cells"])
        if len(rows) != expected_count:
            task_means[task] = None
            continue
        panels = {}
        for panel in ("all", "first_half", "second_half"):
            means = {m: sum(c["episode_panels"][panel]["mean_ise"][m] for c in rows) / len(rows)
                     for m in ("history", "threshold", "termination_history")}
            panels[panel] = {"mean_ise": means, "history_minus_comparator": {
                m: means["history"] - means[m] for m in ("threshold", "termination_history")}}
        task_means[task] = {"optimizer_roots": len(rows), "episode_panels": panels}
    return {"run_name": manifest["run_name"], "status": "complete" if not missing else "incomplete_or_failed",
        "scope": manifest["scope"], "statistical_unit": "optimizer_root_not_probes_or_future_replicas",
        "completed_root_cells": len(cells), "missing_or_failed": missing, "cells": cells,
        "task_means": task_means,
        "completed_evaluation_cpu_seconds": sum(c["evaluation_cost"]["cpu_seconds"] for c in cells),
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "lifecycle_costs_complete": False, "performance_claim_supported": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(json.loads(args.manifest.read_text()), root=ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
