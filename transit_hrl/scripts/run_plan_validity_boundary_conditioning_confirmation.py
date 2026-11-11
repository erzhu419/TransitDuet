"""Sample fresh frozen-predictor confirmation; no fitting or early decisions."""

import argparse
import json
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.compute_budget import ComputeLedger, count_work, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_boundary_conditioning import METHODS, predict_frozen_bundle
from freq_hrl.experiments.plan_validity_boundary_conditioning_confirmation import COMPARISONS
from freq_hrl.experiments.plan_validity_boundary_information import acquire_windows, fresh_reference
from freq_hrl.experiments.plan_validity_boundary_information_readout import evaluation_data, summarize
from freq_hrl.experiments.plan_validity_phase_support import observe
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from scripts.run_plan_validity_boundary_information_precheck import load_frozen_C0


def load_snapshot(plan, design):
    freeze = json.loads((ROOT / plan["freeze_manifest"]).read_text())
    directory = ROOT / freeze["output_directory"]
    report = json.loads((directory / "public/result.json").read_text())
    if (report["run_name"], report["status"], report["stage"], report["all_models_frozen"], report["new_ridge_fits"],
        report["baseline_refits"], report["new_environment_steps"], report["evaluation_prefixes_accessed"],
        report["original_models_modified"]) != (freeze["run_name"], "complete", "boundary_conditioning_full_A_freeze_complete",
        True, 8, 0, 0, False, False) or report["measured_cost"]["counts"] != freeze["expected_counts"]:
        raise ValueError("completed global full-A freeze is required before fresh prefix access")
    bundle = json.loads((directory / "private/frozen_models.json").read_text())
    if bundle["run_name"] != freeze["run_name"] or bundle["evaluation_prefixes_accessed"] or len(bundle["cells"]) != 4:
        raise ValueError("complete global frozen bundle differs")
    result = {}
    for cell, source, saved in zip(plan["cells"], design["cells"], bundle["cells"]):
        key = (cell["task"], cell["optimizer_seed"])
        if any(key != (c["task"], c["optimizer_seed"]) for c in (source, saved)):
            raise ValueError("confirmation and frozen root identities differ")
        models = saved["models"]
        if set(models) != set(plan["methods"]) - {"zero"}:
            raise ValueError("frozen control/method set differs")
        roster = saved["training_roster"]
        if (len(roster["window_keys"]) != 384 or sum(roster["label_counts"]) != 768
            or set(r[0] for r in roster["window_keys"]) & set(cell["readout_prefix_seeds"])
            or any(len(models[m]["weights"]) != 240 or abs(models[m]["total_effective_df"] - 3) > 1e-10 for m in METHODS)):
            raise ValueError("training roster, fixed DF or fresh-prefix separation differs")
        result[key] = models
    return result


def run_shard(plan, design, release, shard, directory, publish):
    cell = design["cells"][shard["cell_index"]]
    with measured_phase("load_global_freeze_and_unchanged_C0"):
        models = load_snapshot(plan, design)[cell["task"], cell["optimizer_seed"]]
        controller, rule, args, spec = load_frozen_C0(cell, design)
    publish({"stage": "global_freeze_verified_before_fresh_prefix_access", "all_models_frozen": True})
    rows = []
    with measured_phase("native_fresh_prefix_acquisition"):
        for seed in shard["prefix_seeds"]:
            observation = observe(controller, rule, spec=spec, args=args, seed=seed, probes_per_path=plan["checks_per_prefix"])
            for step in sorted(observation["check_steps"]):
                reference = fresh_reference(observation, step, args=args, windows=plan["windows_per_state"])
                indices = {j: plan["suffix_indices"] for j in range(plan["windows_per_state"])}
                rows.append(acquire_windows(controller, rule, reference, indices, spec=spec, args=args))
                (directory / "private/rows.json").write_text(json.dumps(_json_ready(rows)) + "\n")
                publish({"stage": "sampling_fresh_confirmation", "completed_states": len(rows),
                    "completed_pairs": sum(len(w["A_replicas"]) for r in rows for w in r["windows"])})
    with measured_phase("seven_frozen_predictors_and_paired_moments"):
        evaluation_design = {**design, "readout": {"suffix_indices": plan["suffix_indices"]}}
        data = evaluation_data(rows, cell, evaluation_design, shard["prefix_seeds"])
        predictions = predict_frozen_bundle(models, data)
        if set(predictions) != set(plan["methods"]):
            raise ValueError("scored predictor set differs")
        count_work("frozen_model_window_predictions", sum(len(p) for p in predictions.values()))
        (directory / "private/predictions.json").write_text(json.dumps(_json_ready({
            "window_keys": data["window_keys"], "predictions": predictions})) + "\n")
        readout = summarize(data, predictions, comparisons=COMPARISONS)
    if readout["E_pairs"] != shard["new_pairs"]:
        raise ValueError("fresh pair roster differs from its shard budget")
    return {"stage": "boundary_conditioning_confirmation_shard_complete", "sampled_prefix_seeds": shard["prefix_seeds"],
        "completed_states": len(rows), "completed_pairs": readout["E_pairs"], "readout": readout,
        "all_models_frozen": True, "evaluation_prefixes_accessed": True, "new_fits": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args(argv)
    release = json.loads(args.manifest.read_text())
    plan = json.loads((ROOT / release["plan_manifest"]).read_text())
    design = json.loads((ROOT / release["design_manifest"]).read_text())
    shard = release["shards"][args.shard_index]
    cell = plan["cells"][shard["cell_index"]]
    directory = ROOT / shard["output_directory"]
    if directory.exists():
        raise FileExistsError("preserve paid fresh-confirmation attempts")
    for part in ("public", "private"):
        (directory / part).mkdir(parents=True)
    ledger = ComputeLedger()
    result = {"run_name": plan["run_name"], "task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
        "shard_index": shard["shard_index"], "status": "running", "candidate_recipe_decision": "PILOT_RECIPE_HOLD",
        "confirmation_released": False, "performance_claim_supported": False, "lifecycle_costs_complete": False}

    def publish(update):
        result.update(update, measured_cost=ledger.summary())
        (directory / "public/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
        print(f"{cell['task']} {cell['optimizer_seed']}/{shard['shard_index']}: {result.get('stage', result['status'])} {result.get('completed_states', '')}", flush=True)

    try:
        with ledger.recording("boundary_conditioning_confirmation"), single_thread_runtime() as profile:
            if profile != design["source_hardware_profile"]:
                raise ValueError("runtime differs from the frozen source")
            result["hardware_profile"] = profile
            summary = run_shard(plan, design, release, shard, directory, publish)
        c = ledger.summary()["counts"]
        if (c.get("environment_steps") != shard["environment_steps"] or c.get("frozen_model_window_predictions") != 336
            or any(c.get(k, 0) for k in ("ridge_fit_calls", "constant_fit_calls", "controller_optimizer_updates", "termination_optimizer_updates"))):
            raise RuntimeError("confirmation exceeded its fixed no-fit native/scoring budget")
        result.update(summary, status="complete")
    except Exception as exc:
        result.update(status="failed", stage="failed", error_type=type(exc).__name__, error=str(exc))
    result.update(measured_cost=ledger.summary(), new_environment_steps=ledger.summary()["counts"].get("environment_steps", 0),
        peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (directory / "private/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
    publish({})
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
