"""Audit all fresh shards before fixed whole-prefix bootstrap; public moments only."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.compute_budget import ComputeLedger
from freq_hrl.experiments.plan_validity_boundary_conditioning_confirmation import COMPARISONS, bootstrap_root
from scripts.analyze_plan_validity_boundary_information_readout import merge_root


def aggregate_records(plan, release, records):
    if len(records) != 16 or len(release["shards"]) != 16:
        raise ValueError("all sixteen completed shards are required before any bootstrap")
    panels, costs, counts = [[] for _ in plan["cells"]], [], {}
    for shard, r in zip(release["shards"], records):
        cell = plan["cells"][shard["cell_index"]]
        if (r["run_name"], r["task"], r["optimizer_seed"], r["shard_index"], r["status"], r["stage"],
            r["sampled_prefix_seeds"], r["completed_states"], r["completed_pairs"], r["new_environment_steps"],
            r["new_fits"], r["all_models_frozen"], r["evaluation_prefixes_accessed"]) != (
            plan["run_name"], cell["task"], cell["optimizer_seed"], shard["shard_index"], "complete",
            "boundary_conditioning_confirmation_shard_complete", shard["prefix_seeds"], 12, 384, 926400, 0, True, True):
            raise ValueError("confirmation shard identity, roster or completion differs")
        cost, panel = r["measured_cost"], r["readout"]
        c = cost["counts"]
        if (c.get("environment_steps") != shard["environment_steps"] or c.get("frozen_model_window_predictions") != 336
            or any(c.get(k, 0) for k in ("ridge_fit_calls", "constant_fit_calls", "controller_optimizer_updates", "termination_optimizer_updates"))):
            raise ValueError("confirmation native/scoring/no-fit ledger differs")
        if (panel["prefixes"], panel["states"], panel["windows"], panel["E_pairs"]) != (4, 12, 48, 384) or set(panel["MSE_to_E_mean"]) != set(plan["methods"]):
            raise ValueError("confirmation public moments differ")
        for a, b in COMPARISONS:
            if [p["seed"] for p in panel["comparisons"][a + "_minus_" + b]["prefix_reports"]] != shard["prefix_seeds"]:
                raise ValueError("confirmation prefix moments differ from their shard")
        panels[shard["cell_index"]].append(panel)
        costs.append(cost)
        for key, value in c.items():
            counts[key] = counts.get(key, 0) + value
    if counts["environment_steps"] != release["released_environment_steps"] or counts["environment_steps"] != plan["proposed_environment_steps"]:
        raise ValueError("confirmation native budget did not close")
    # Check every root before releasing any final statistical readout.
    merged = []
    for cell, parts in zip(plan["cells"], panels):
        readout = merge_root(parts, comparisons=COMPARISONS)
        if len(parts) != 4 or any([p["seed"] for p in c["prefix_reports"]] != cell["readout_prefix_seeds"] for c in readout["comparisons"].values()):
            raise ValueError("confirmation root is not the declared full cohort")
        merged.append(readout)
    cells = [{"task": cell["task"], "optimizer_seed": cell["optimizer_seed"], "readout": bootstrap_root(readout, cell, plan)}
        for cell, readout in zip(plan["cells"], merged)]
    primary, = [c for c in cells if (c["task"], c["optimizer_seed"]) == (plan["primary_root"]["task"], plan["primary_root"]["optimizer_seed"])]
    return {"cells": cells, "primary_prediction_rule_passed": primary["readout"]["prediction_confirmation_rule_passed"],
        "completed_shards": 16, "completed_prefixes": 64, "completed_states": 192, "E_pairs": 6144,
        "evaluation_environment_steps": counts["environment_steps"], "evaluation_measured_cost": {"counts": counts,
            "cpu_seconds": sum(c["cpu_seconds"] for c in costs),
            "sum_shard_wall_seconds_not_parallel_makespan": sum(c["wall_seconds"] for c in costs)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("preserve completed confirmation analyses")
    release = json.loads(args.manifest.read_text())
    plan = json.loads((ROOT / release["plan_manifest"]).read_text())
    ledger = ComputeLedger()
    result = {"run_name": plan["run_name"], "status": "running", "candidate_recipe_decision": "PILOT_RECIPE_HOLD",
        "confirmation_released": False, "performance_claim_supported": False, "lifecycle_costs_complete": False}
    with ledger.recording("confirmation_public_moments_and_prefix_bootstrap"):
        records = [json.loads((ROOT / s["output_directory"] / "public/result.json").read_text()) for s in release["shards"]]
        result.update(aggregate_records(plan, release, records))
    if ledger.summary()["counts"] != {"bootstrap_prefix_resamples": 4 * plan["bootstrap_replicates"]}:
        raise RuntimeError("confirmation analysis performed unexpected counted work")
    result.update(status="complete", stage="boundary_conditioning_confirmation_merged", new_fits=0, new_environment_steps=0,
        measured_cost=ledger.summary(), scope=plan["scope"], bound_scope=plan["bound_scope"],
        source="sixteen_completed_public_moment_JSONs_no_raw_trajectory_model_or_checkpoint_reads")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "stage", "evaluation_environment_steps", "primary_prediction_rule_passed")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
