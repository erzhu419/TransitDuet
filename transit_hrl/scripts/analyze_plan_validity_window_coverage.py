"""Compact root-preserving readout of the frozen W coverage factorial."""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def summarize(manifest, *, root=ROOT):
    rows, missing, failed = [], [], []
    for cell in manifest["cells"]:
        path = root / cell["output_directory"] / "run/public/result.json"
        if not path.exists():
            missing.append([cell["task"], cell["optimizer_seed"]])
            continue
        row = json.loads(path.read_text())
        if (row["run_name"] != manifest["run_name"] or row["task"] != cell["task"]
            or row["optimizer_seed"] != cell["optimizer_seed"] or row["phase"] != "run"
            or row["held_prefix_seeds"] != cell["held_prefix_seeds"]
            or row["new_training_prefix_seeds"] != cell["new_training_prefix_seeds"]):
            raise ValueError("result differs from the frozen factorial cell")
        rows.append(row)
        if row["status"] != "complete":
            failed.append([cell["task"], cell["optimizer_seed"]])
    tasks = {}
    for task in sorted(set(c["task"] for c in manifest["cells"])):
        group = [r for r in rows if r["task"] == task and r["status"] == "complete"]
        expected = sum(c["task"] == task for c in manifest["cells"])
        tasks[task] = None if len(group) != expected else {"trained_roots": len(group), "panels": {panel: {
            "mean_ise": {method: float(np.mean([r["episode_panels"][panel]["mean_ise"][method] for r in group]))
                for method in ("dense_df2", "dense_df8", "wide_df2", "wide_df8", "history", "L", "threshold", "termination_history")},
            "root_factorial_effects": {contrast: [r["episode_panels"][panel]["factorial_contrasts"][contrast]["mean"] for r in group]
                for contrast in ("coverage_df2", "coverage_df8", "complexity_dense", "complexity_wide", "interaction")}}
            for panel in ("all", "first_half", "second_half")}}
    return {"run_name": manifest["run_name"], "status": "complete" if not missing and not failed else "incomplete_or_failed",
        "missing": missing, "failed": failed, "root_readouts": rows, "task_means": tasks,
        "known_main_cpu_seconds": sum(r["measured_cost"]["cpu_seconds"] for r in rows),
        "cost_scope": "marginal_main_cost_not_total_lifecycle_budget_match",
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "lifecycle_costs_complete": False, "performance_claim_supported": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(json.loads(args.manifest.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "missing", "failed", "candidate_recipe_decision")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
