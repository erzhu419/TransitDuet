"""Fit fixed-capacity boundary-tail probes from server caches; no simulation."""

import argparse
import json
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.compute_budget import ComputeLedger, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_boundary_tail import dataset, fit_probe
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready


def load_cache(cell, manifest):
    rows = []
    for shard in (0, 1):
        directory = ROOT / cell["cache_directory"] / f"shard{shard}"
        record = json.loads((directory / "public/result.json").read_text())
        seeds = cell["development_prefix_seeds"][4 * shard:4 * (shard + 1)]
        if (record["run_name"], record["task"], record["optimizer_seed"], record["status"], record["stage"], record["sampled_seeds"]) != (
            manifest["source_run_name"], cell["task"], cell["optimizer_seed"], "complete", "boundary_variance_complete", seeds):
            raise ValueError("a matching completed boundary bank is required")
        raw = json.loads((directory / "private/rows.json").read_text())
        if (len(raw) != 12 or {r["seed"] for r in raw} != set(seeds)
            or len({(r["seed"], r["opportunity_step"]) for r in raw}) != 12
            or any(sum(r["seed"] == s for r in raw) != 3 for s in seeds)
            or any(len(r["windows"]) != 4 or r["boundary_step"] - r["opportunity_step"] != manifest["period"]
                or any(len(w["suffix_draws"]) != 4 for w in r["windows"]) for r in raw)):
            raise ValueError("cached prefix, window or suffix roster differs")
        rows.extend(raw)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    cell = manifest["cells"][args.cell_index]
    directory = ROOT / cell["output_directory"]
    if directory.exists():
        raise FileExistsError("preserve paid cache-probe attempts")
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
        with ledger.recording("boundary_tail_probe"), single_thread_runtime() as profile:
            if profile != manifest["source_hardware_profile"]:
                raise ValueError("runtime differs from the frozen pilot")
            result["hardware_profile"] = profile
            publish({"stage": "loading_resident_cache"})
            with measured_phase("cache_and_features"):
                data = dataset(load_cache(cell, manifest), horizon=manifest["horizon"], period=manifest["period"])
            with measured_phase("A_only_fixed_capacity_fits_and_B_readout"):
                public, private = fit_probe(data, total_df=manifest["total_effective_df"], publish=publish)
                (directory / "private/models_and_predictions.json").write_text(json.dumps(_json_ready(private)) + "\n")
        counts = ledger.summary()["counts"]
        if (counts.get("ridge_fit_calls", 0) != manifest["ridge_fits_per_root"]
            or counts.get("constant_fit_calls", 0) != manifest["constant_fits_per_root"]
            or any(counts.get(k, 0) for k in ("environment_opens", "environment_steps", "controller_optimizer_updates", "termination_optimizer_updates"))):
            raise RuntimeError("cache-only probe exceeded its frozen fitting budget")
        result.update(public, status="complete", stage="boundary_tail_probe_complete")
    except Exception as exc:
        result.update(status="failed", stage="failed", error_type=type(exc).__name__, error=str(exc))
    result.update(measured_cost=ledger.summary(), new_environment_steps=ledger.summary()["counts"].get("environment_steps", 0),
        peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (directory / "private/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
    publish({})
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
