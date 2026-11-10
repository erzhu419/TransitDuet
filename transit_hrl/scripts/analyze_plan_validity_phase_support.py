"""Root-preserving readout of the frozen sampling/clock factorial."""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("start_plain", "start_clock", "checks_plain", "checks_clock", "history", "L", "prior_wide_df2", "threshold", "termination_history")
CONTRASTS = ("sampling_plain", "sampling_clock", "clock_start", "clock_checks", "interaction")


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
            or row["fit_prefix_seeds"] != cell["fit_prefix_seeds"] or row["held_prefix_seeds"] != cell["held_prefix_seeds"]):
            raise ValueError("result differs from the frozen phase-support cell")
        rows.append(row)
        if row["status"] != "complete":
            failed.append([cell["task"], cell["optimizer_seed"]])
    tasks = {}
    for task in sorted(set(c["task"] for c in manifest["cells"])):
        group = [r for r in rows if r["task"] == task and r["status"] == "complete"]
        expected = sum(c["task"] == task for c in manifest["cells"])
        tasks[task] = None if len(group) != expected else {"trained_roots": len(group), "panels": {panel: {
            "mean_ise": {m: float(np.mean([r["episode_panels"][panel]["mean_ise"][m] for r in group])) for m in METHODS},
            "root_factorial_effects": {c: [r["episode_panels"][panel]["factorial_contrasts"][c]["mean"] for r in group] for c in CONTRASTS}}
            for panel in ("all", "first_half", "second_half")}}
    return {"run_name": manifest["run_name"], "status": "complete" if not missing and not failed else "incomplete_or_failed",
        "missing": missing, "failed": failed, "root_readouts": rows, "task_means": tasks,
        "known_main_cpu_seconds": sum(r["measured_cost"]["cpu_seconds"] for r in rows),
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "lifecycle_costs_complete": False, "performance_claim_supported": False,
        "matching_scope": "same_fitting_count_and_effective_df_not_equal_total_cpu"}


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
