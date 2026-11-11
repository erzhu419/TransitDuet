"""Acquire reserved E windows only after the global eight-model freeze."""

import argparse
import json
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.compute_budget import ComputeLedger, count_work, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_boundary_information import acquire_windows, fresh_reference
from freq_hrl.experiments.plan_validity_boundary_information_fit import frozen_predictions
from freq_hrl.experiments.plan_validity_boundary_information_readout import evaluation_data, summarize
from freq_hrl.experiments.plan_validity_phase_support import observe
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from scripts.run_plan_validity_boundary_information_precheck import load_frozen_C0


def load_global_freeze(design, release):
    manifest = json.loads((ROOT / release["fit_manifest"]).read_text())
    directory = ROOT / manifest["output_directory"]
    report = json.loads((directory / "public/result.json").read_text())
    if (report["run_name"], report["status"], report["stage"], report["all_models_frozen"],
        report["new_ridge_fits"], report["new_environment_steps"], report["evaluation_prefixes_accessed"]) != (
        design["run_name"], "complete", "boundary_information_fit_complete", True, 8, 0, False):
        raise ValueError("complete global eight-model freeze is required before any E access")
    if report["measured_cost"]["counts"] != {"ridge_fit_calls": 8}:
        raise ValueError("source fitting budget differs")
    saved = json.loads((directory / "private/frozen_models.json").read_text())
    if saved["run_name"] != design["run_name"] or saved["evaluation_prefixes_accessed"] or len(saved["cells"]) != 4:
        raise ValueError("global frozen bundle differs")
    models = {}
    for cell, frozen in zip(design["cells"], saved["cells"]):
        key = (cell["task"], cell["optimizer_seed"])
        if key != (frozen["task"], frozen["optimizer_seed"]) or set(frozen["models"]) != {"base", "constant", "causal_t", "dense", "wide"}:
            raise ValueError("all four matching frozen root models are required")
        for method in ("dense", "wide"):
            roster = frozen["training_rosters"][method]
            if (len(roster["window_keys"]) != design[method]["prefixes"] * 12
                or sum(roster["label_counts"]) != design[method]["training_pairs"]
                or set(k[0] for k in roster["window_keys"]) & set(cell["readout_prefix_seeds"])
                or abs(frozen["models"][method]["total_effective_df"] - 3) > 1e-10):
                raise ValueError("frozen training roster or fixed DF differs")
        models[key] = frozen["models"]
    return models


def run_shard(cell, design, release, shard, directory, publish):
    with measured_phase("load_global_freeze_and_unchanged_C0"):
        models = load_global_freeze(design, release)[cell["task"], cell["optimizer_seed"]]
        controller, rule, args, spec = load_frozen_C0(cell, design)
    publish({"stage": "global_freeze_verified_before_E_access", "all_models_frozen": True})
    rows = []
    with measured_phase("native_E_acquisition"):
        for seed in shard["prefix_seeds"]:
            observation = observe(controller, rule, spec=spec, args=args, seed=seed, probes_per_path=design["checks_per_prefix"])
            for step in sorted(observation["check_steps"]):
                reference = fresh_reference(observation, step, args=args, windows=design["windows_per_state"])
                indices = {j: design["readout"]["suffix_indices"] for j in range(design["windows_per_state"])}
                rows.append(acquire_windows(controller, rule, reference, indices, spec=spec, args=args))
                (directory / "private/rows.json").write_text(json.dumps(_json_ready(rows)) + "\n")
                publish({"stage": "sampling_frozen_E", "completed_states": len(rows),
                    "completed_pairs": sum(len(w["A_replicas"]) for r in rows for w in r["windows"])})
    with measured_phase("frozen_prediction_and_E_readout"):
        data = evaluation_data(rows, cell, design, shard["prefix_seeds"])
        predictions = frozen_predictions(data, models)
        count_work("frozen_model_window_predictions", sum(len(p) for p in predictions.values()))
        (directory / "private/predictions.json").write_text(json.dumps(_json_ready({
            "window_keys": data["window_keys"], "predictions": predictions})) + "\n")
        readout = summarize(data, predictions)
    if readout["E_pairs"] != shard["new_pairs"]:
        raise RuntimeError("E pair roster differs from its frozen budget")
    return {"stage": "boundary_information_readout_complete", "sampled_prefix_seeds": shard["prefix_seeds"],
        "completed_states": len(rows), "completed_pairs": readout["E_pairs"], "readout": readout,
        "all_models_frozen": True, "evaluation_prefixes_accessed": True, "new_fits": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args(argv)
    release = json.loads(args.manifest.read_text())
    design = json.loads((ROOT / release["design_manifest"]).read_text())
    shard = release["shards"][args.shard_index]
    cell = design["cells"][shard["cell_index"]]
    directory = ROOT / shard["output_directory"]
    if directory.exists():
        raise FileExistsError("preserve paid E readout attempts")
    for name in ("public", "private"):
        (directory / name).mkdir(parents=True)
    ledger = ComputeLedger()
    result = {"run_name": design["run_name"], "task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
        "shard_index": shard["shard_index"], "status": "running", "candidate_recipe_decision": "PILOT_RECIPE_HOLD",
        "confirmation_released": False, "performance_claim_supported": False, "lifecycle_costs_complete": False}

    def publish(update):
        result.update(update, measured_cost=ledger.summary())
        (directory / "public/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
        print(f"{cell['task']} {cell['optimizer_seed']}/{shard['shard_index']}: {result.get('stage', result['status'])} {result.get('completed_states', '')}", flush=True)

    try:
        with ledger.recording("boundary_information_readout"), single_thread_runtime() as profile:
            if profile != design["source_hardware_profile"]:
                raise ValueError("runtime differs from the frozen acquisition profile")
            result["hardware_profile"] = profile
            summary = run_shard(cell, design, release, shard, directory, publish)
        counts = ledger.summary()["counts"]
        if (counts.get("environment_steps") != shard["environment_steps"] or any(counts.get(k, 0)
            for k in ("ridge_fit_calls", "constant_fit_calls", "controller_optimizer_updates", "termination_optimizer_updates"))):
            raise RuntimeError("E readout exceeded its no-fit native budget")
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
