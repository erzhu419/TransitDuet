"""Descriptive episode sensitivity; leave the frozen primary cohort unchanged."""

import argparse
import json
from pathlib import Path


def sensitivity(episodes, candidate, comparator):
    effects = [{"optimizer_seed": root, "seed": r["seed"], "difference": r[candidate] - r[comparator]}
               for root, r in episodes]
    total = sum(r["difference"] for r in effects)
    best, worst = min(effects, key=lambda r: r["difference"]), max(effects, key=lambda r: r["difference"])
    return {"episodes": len(effects), "primary_mean_difference": total / len(effects),
        "improved_paths": sum(r["difference"] < 0 for r in effects),
        "largest_benefit_path": best, "largest_harm_path": worst,
        "descriptive_mean_without_largest_benefit": (total - best["difference"]) / (len(effects) - 1),
        "descriptive_mean_without_largest_harm": (total - worst["difference"]) / (len(effects) - 1),
        "paired_effects": effects}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = json.loads(args.summary.read_text())
    if summary["status"] != "complete":
        raise ValueError("sensitivity requires the complete frozen cohort")
    tasks = {}
    for task in summary["task_means"]:
        episodes = [(r["optimizer_seed"], e) for r in summary["root_readouts"] if r["task"] == task for e in r["episodes"]]
        tasks[task] = {label: sensitivity(episodes, a, b) for label, a, b in (
            ("joint_vs_start", "checks_clock", "start_plain"),
            ("clock_vs_start", "start_clock", "start_plain"),
            ("joint_vs_history", "checks_clock", "history"))}
    result = {"run_name": summary["run_name"], "analysis_role": "posthoc_descriptive_not_a_new_primary_cohort",
        "primary_exclusions": [], "policy_selection": "none", "tasks": tasks}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
