"""Reconcile frozen nested-credit moments without raw draws, fitting or replay."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.plan_validity_boundary_credit import summarize_boundary_reports
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready


def readout(reports):
    summary = summarize_boundary_reports(reports)
    for target, aggregate in summary["targets"].items():
        within = aggregate["within_window"]
        total = aggregate["total_corrected"]
        projection = aggregate["equal_pair_count_independent_projection"]
        aggregate["suffix_fraction_of_total_corrected"] = within / total if total > 0 else None
        aggregate["nested_over_independent_variance_projection"] = (
            aggregate["nested_mean_variance"] / projection if projection > 0 else None)
        noise = np.asarray([r["targets"][target]["nested_mean_variance"] for r in reports])
        aggregate["largest_state_share_of_nested_variance"] = float(noise.max() / noise.sum()) if noise.sum() > 0 else None
        aggregate["half_means"] = {axis: np.mean([
            r["targets"][target][axis + "_half_means"] for r in reports], axis=0).tolist()
            for axis in ("window", "suffix")}
    return summary


def analyze(manifest, records):
    expected = {(c["task"], c["optimizer_seed"], s) for c in manifest["cells"] for s in range(2)}
    keyed = {(r["task"], r["optimizer_seed"], r["shard_index"]): r for r in records}
    if len(keyed) != len(records) or set(keyed) != expected:
        raise ValueError("need exactly the eight frozen root/shard reports")
    cells, total_counts = [], {}
    for cell in manifest["cells"]:
        rows = []
        for shard in range(2):
            record = keyed[cell["task"], cell["optimizer_seed"], shard]
            if (record["run_name"], record["status"], record["stage"]) != (
                manifest["run_name"], "complete", "boundary_variance_complete"):
                raise ValueError("a matching completed variance report is required")
            seeds = cell["development_prefix_seeds"][4 * shard:4 * (shard + 1)]
            states = record["state_reports"]
            roster = [(r["seed"], r["opportunity_step"]) for r in states]
            if (record["sampled_seeds"] != seeds or len(states) != 12 or len(set(roster)) != 12
                or {r["seed"] for r in states} != set(seeds)
                or any(sum(r["seed"] == s for r in states) != 3 for s in seeds)):
                raise ValueError("state/prefix coverage differs from the frozen shard")
            if any((r["windows"], r["suffixes"], r["boundary_step"] - r["opportunity_step"]) != (
                manifest["window_draws"], manifest["suffix_draws_per_window"], manifest["window_steps"])
                for r in states):
                raise ValueError("nested dimensions or window boundary changed")
            counts = record["measured_cost"]["counts"]
            if (record["new_environment_steps"] != manifest["per_job_environment_steps"]
                or counts.get("environment_steps") != manifest["per_job_environment_steps"]
                or any(counts.get(k, 0) for k in ("ridge_fit_calls", "controller_optimizer_updates", "termination_optimizer_updates"))):
                raise ValueError("the no-fit native replay budget did not close")
            if record["confirmation_released"] or record["performance_claim_supported"]:
                raise ValueError("a variance diagnostic cannot release a performance claim")
            for key, value in counts.items():
                total_counts[key] = total_counts.get(key, 0) + value
            rows.extend(states)
        cells.append({"task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
            "summary": readout(rows), "prefix_reports": [
                {"seed": seed, **readout([r for r in rows if r["seed"] == seed])}
                for seed in cell["development_prefix_seeds"]]})
    if total_counts["environment_steps"] != manifest["total_environment_steps"]:
        raise ValueError("total native budget did not close")
    return {"run_name": manifest["run_name"], "status": "complete", "cells": cells,
        "measured_counts": total_counts,
        "acquisition_cpu_seconds": sum(r["measured_cost"]["cpu_seconds"] for r in records),
        "max_shard_wall_seconds": max(r["measured_cost"]["wall_seconds"] for r in records),
        "peak_process_rss_mb": max(r["peak_process_rss_mb"] for r in records),
        "new_analysis_environment_steps": 0, "new_model_fits": 0,
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "performance_claim_supported": False, "lifecycle_costs_complete": False,
        "scope": "descriptive_fixed_state_moments_not_policy_effects_or_learnability",
        "projection_scope": "analytical_equal_full_replay_pair_count_not_sampled_control_or_total_CPU",
        "credit_sign": "positive_favors_now_over_wait_one_check_not_improvement_over_C0"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    inputs = json.loads(args.inputs.read_text())
    result = analyze(json.loads(args.manifest.read_text()), inputs["records"])
    result["retrieved_public_bytes"] = inputs["retrieved_public_bytes"]
    args.output.write_text(json.dumps(_json_ready(result), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
