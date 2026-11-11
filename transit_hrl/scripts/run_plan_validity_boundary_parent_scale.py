"""Refit only boundary potentials; reuse every frozen baseline prediction."""

import argparse
import json
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.compute_budget import ComputeLedger, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_boundary_parent_scale import parent_dataset, verify_prior, fit_parent_probe
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from scripts.run_plan_validity_boundary_tail import load_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    cell = manifest["cells"][args.cell_index]
    directory = ROOT / cell["output_directory"]
    if directory.exists():
        raise FileExistsError("preserve measured geometry-repair attempts")
    for name in ("public", "private"):
        (directory / name).mkdir(parents=True)
    ledger = ComputeLedger()
    result = {"run_name": manifest["run_name"], "task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
        "status": "running", "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "performance_claim_supported": False, "lifecycle_costs_complete": False}

    def publish(update):
        result.update(update, measured_cost=ledger.summary())
        (directory / "public/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
        print(f"{cell['task']} {cell['optimizer_seed']}: {result.get('stage', result['status'])} {result.get('completed_prefix_folds', '')}", flush=True)

    try:
        with ledger.recording("boundary_parent_scale"), single_thread_runtime() as profile:
            if profile != manifest["source_hardware_profile"]:
                raise ValueError("runtime differs from the frozen pilot")
            result["hardware_profile"] = profile
            publish({"stage": "verify_frozen_cache_and_predictions"})
            with measured_phase("resident_cache_and_baseline_reproduction"):
                prior = ROOT / cell["prior_probe_directory"]
                saved = json.loads((prior / "public/result.json").read_text())
                if (saved["run_name"], saved["task"], saved["optimizer_seed"], saved["status"], saved["stage"], saved["matched_total_effective_df"]) != (
                    manifest["prior_probe_run_name"], cell["task"], cell["optimizer_seed"], "complete", "boundary_tail_probe_complete", manifest["total_effective_df"]):
                    raise ValueError("a matching completed prior probe is required")
                previous = json.loads((prior / "private/models_and_predictions.json").read_text())
                data = parent_dataset(load_cache(cell, manifest), horizon=manifest["horizon"], period=manifest["period"])
                verify_prior(data, previous, saved)
            with measured_phase("A_only_parent_scaled_potential_fits"):
                public, private = fit_parent_probe(data, previous, total_df=manifest["total_effective_df"], publish=publish)
                (directory / "private/models_and_predictions.json").write_text(json.dumps(_json_ready(private)) + "\n")
        counts = ledger.summary()["counts"]
        if counts != {"ridge_fit_calls": manifest["ridge_fits_per_root"]}:
            raise RuntimeError("geometry repair exceeded its potential-only no-simulation budget")
        result.update(public, status="complete", stage="boundary_parent_scale_complete",
            frozen_baseline_losses_reproduced=True,
            prior_boundary_delta={k: saved["panels"][k]["boundary_delta"] for k in ("training", "prefix_oof")})
    except Exception as exc:
        result.update(status="failed", stage="failed", error_type=type(exc).__name__, error=str(exc))
    result.update(measured_cost=ledger.summary(), new_environment_steps=ledger.summary()["counts"].get("environment_steps", 0),
        peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (directory / "private/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
    publish({})
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
