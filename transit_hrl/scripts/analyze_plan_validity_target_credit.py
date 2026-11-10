"""Read only compact matched-target results, retaining every task/root."""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def summarize(manifest, *, root=ROOT):
    rows, missing, failed = [], [], []
    for cell in manifest["cells"]:
        path = root / cell["output_directory"] / "run/public/result.json"
        key = f"{cell['task']}/{cell['optimizer_seed']}"
        if not path.exists():
            missing.append(key)
            continue
        row = json.loads(path.read_text())
        if (row["run_name"] != manifest["run_name"] or row["task"] != cell["task"]
            or row["optimizer_seed"] != cell["optimizer_seed"] or row["phase"] != "run"
            or row["training_prefix_seeds"] != cell["training_prefix_seeds"]
            or row["held_prefix_seeds"] != cell["held_prefix_seeds"]):
            raise ValueError("readout does not match the frozen target-credit cell")
        rows.append(row)
        if row["status"] != "complete":
            failed.append(key)
    tasks = {}
    for task in sorted(set(c["task"] for c in manifest["cells"])):
        group = [r for r in rows if r["task"] == task and r["status"] == "complete"]
        expected = sum(c["task"] == task for c in manifest["cells"])
        tasks[task] = None if len(group) != expected else {
            "trained_roots": len(group), "episode_panels": {panel: {
                "mean_ise": {method: float(np.mean([r["episode_panels"][panel]["mean_ise"][method] for r in group]))
                    for method in ("L", "W", "G", "history", "threshold", "termination_history")},
                "root_differences_from_L": {method: [r["episode_panels"][panel]["paired_difference"][method]["L"]["mean"]
                    for r in group] for method in ("W", "G")}}
                for panel in ("all", "first_half", "second_half")}}
    return {"run_name": manifest["run_name"], "status": "complete" if not missing and not failed else "incomplete_or_failed",
        "missing": missing, "failed": failed, "root_readouts": rows, "task_means": tasks,
        "known_main_cpu_seconds": sum(r["measured_cost"]["cpu_seconds"] for r in rows),
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "lifecycle_costs_complete": False, "performance_claim_supported": False,
        "interval_scope": "within_frozen_root_paths_descriptive_not_confirmatory_cross_root",
        "cost_scope": "main_only_prechecks_and_old_lifecycle_costs_reported_separately"}


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
