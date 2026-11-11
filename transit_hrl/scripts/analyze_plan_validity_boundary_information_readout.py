"""Merge completed E public moments, without raw trajectories or new scoring."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.compute_budget import ComputeLedger
from freq_hrl.experiments.plan_validity_boundary_information_readout import COMPARISONS


def merge_loss(parts, weights):
    result = {k: float(sum(w * p[k] for w, p in zip(weights, parts)))
        for k in ("MSE_delta", "squared_prediction_term", "alignment_term")}
    result["conditional_suffix_MC_SE"] = float(np.sqrt(sum((w * p["conditional_suffix_MC_SE"]) ** 2
        for w, p in zip(weights, parts))))
    return result


def merge_root(panels):
    total = sum(p["windows"] for p in panels)
    weights = [p["windows"] / total for p in panels]
    result = {k: sum(p[k] for p in panels) for k in ("prefixes", "states", "windows", "E_pairs")}
    result.update(MSE_to_E_mean={m: float(sum(w * p["MSE_to_E_mean"][m] for w, p in zip(weights, panels)))
        for m in panels[0]["MSE_to_E_mean"]}, empirical_E_mean_noise_variance=float(sum(
            w * p["empirical_E_mean_noise_variance"] for w, p in zip(weights, panels))), comparisons={})
    for candidate, reference in COMPARISONS:
        name = candidate + "_minus_" + reference
        parts = [p["comparisons"][name] for p in panels]
        prefixes = sorted([r for p in parts for r in p["prefix_reports"]], key=lambda r: r["seed"])
        deltas = np.asarray([r["MSE_delta"] for r in prefixes])
        result["comparisons"][name] = {**merge_loss(parts, weights), "prefix_reports": prefixes,
            "improving_prefixes": int(np.sum(deltas < 0)), "prefix_delta_SD_descriptive": float(np.std(deltas, ddof=1)),
            **{half: [merge_loss([p[half][j] for p in parts], weights) for j in range(2)]
                for half in ("suffix_halves", "window_halves")}}
    return result


def aggregate_records(design, release, records):
    if len(records) != 8 or len(release["shards"]) != 8:
        raise ValueError("all eight completed E shards are required")
    costs, panels = [], [[] for _ in design["cells"]]
    counts = {}
    for shard, r in zip(release["shards"], records):
        cell = design["cells"][shard["cell_index"]]
        if (r["run_name"], r["task"], r["optimizer_seed"], r["shard_index"], r["status"], r["stage"],
            r["sampled_prefix_seeds"], r["completed_states"], r["completed_pairs"], r["new_environment_steps"],
            r["new_fits"], r["all_models_frozen"], r["evaluation_prefixes_accessed"]) != (
            design["run_name"], cell["task"], cell["optimizer_seed"], shard["shard_index"], "complete",
            "boundary_information_readout_complete", shard["prefix_seeds"], 12, shard["new_pairs"],
            shard["environment_steps"], 0, True, True):
            raise ValueError("E shard identity, roster or completion differs")
        cost = r["measured_cost"]
        c, p = cost["counts"], r["readout"]
        if (c.get("environment_steps") != shard["environment_steps"] or c.get("frozen_model_window_predictions") != 288
            or any(c.get(k, 0) for k in ("ridge_fit_calls", "constant_fit_calls", "controller_optimizer_updates", "termination_optimizer_updates"))):
            raise ValueError("E native or no-fit ledger differs")
        if (p["prefixes"], p["states"], p["windows"], p["E_pairs"]) != (4, 12, 48, 384):
            raise ValueError("E public moments have the wrong sample counts")
        for name in (a + "_minus_" + b for a, b in COMPARISONS):
            if [r["seed"] for r in p["comparisons"][name]["prefix_reports"]] != shard["prefix_seeds"]:
                raise ValueError("E prefix moments do not match their shard")
        panels[shard["cell_index"]].append(p)
        costs.append(cost)
        for key, value in c.items():
            counts[key] = counts.get(key, 0) + value
    if counts["environment_steps"] != design["formal_readout_total_steps"]:
        raise ValueError("E total native budget did not close")
    cells = []
    for cell, parts in zip(design["cells"], panels):
        prefixes = [s for p in parts for s in p["comparisons"]["base_minus_zero"]["prefix_reports"]]
        if len(parts) != 2 or [r["seed"] for r in prefixes] != cell["readout_prefix_seeds"]:
            raise ValueError("root does not contain the complete eight-prefix E roster")
        cells.append({"task": cell["task"], "optimizer_seed": cell["optimizer_seed"], "readout": merge_root(parts)})
    return {"cells": cells, "evaluation_measured_cost": {"counts": counts,
        "cpu_seconds": sum(c["cpu_seconds"] for c in costs),
        "sum_shard_wall_seconds_not_parallel_makespan": sum(c["wall_seconds"] for c in costs)},
        "completed_shards": 8, "evaluation_environment_steps": counts["environment_steps"],
        "completed_prefixes": sum(c["readout"]["prefixes"] for c in cells),
        "completed_states": sum(c["readout"]["states"] for c in cells),
        "E_pairs": sum(c["readout"]["E_pairs"] for c in cells)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("preserve completed readout summaries")
    release = json.loads(args.manifest.read_text())
    design = json.loads((ROOT / release["design_manifest"]).read_text())
    ledger = ComputeLedger()
    result = {"run_name": design["run_name"], "status": "running", "candidate_recipe_decision": "PILOT_RECIPE_HOLD",
        "confirmation_released": False, "performance_claim_supported": False, "lifecycle_costs_complete": False}
    with ledger.recording("merge_completed_E_public_moments"):
        records = [json.loads((ROOT / s["output_directory"] / "public/result.json").read_text()) for s in release["shards"]]
        result.update(aggregate_records(design, release, records))
    if ledger.summary()["counts"]:
        raise RuntimeError("public-moment aggregation unexpectedly performed counted work")
    result.update(status="complete", stage="boundary_information_E_readout_merged", new_fits=0, new_environment_steps=0,
        measured_cost=ledger.summary(), source="eight_existing_public_result_JSONs_only_no_raw_or_private_reads",
        statistics="paired_MSE_deltas_and_quadrature_suffix_MC_SE_prefix_SD_descriptive_no_pseudoseed_CI",
        scope="new_prefix_boundary_prediction_given_frozen_roots_not_episode_policy_performance")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "stage", "completed_shards", "evaluation_environment_steps", "E_pairs", "new_fits", "new_environment_steps")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
