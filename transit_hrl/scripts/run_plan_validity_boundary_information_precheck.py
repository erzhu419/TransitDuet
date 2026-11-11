"""Precheck cached-window extensions and new-prefix acquisition; no fitting."""

import argparse
import json
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.compute_budget import ComputeLedger, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_boundary_information import native_precheck
from freq_hrl.experiments.plan_validity_transfer import PlanValidityTaskSpec
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts.run_plan_validity_boundary_tail import load_cache
from scripts.run_plan_validity_target_credit import load_sources


def load_frozen_C0(cell, manifest):
    variance = json.loads((ROOT / manifest["source_variance_manifest"]).read_text())
    old_manifest = json.loads((ROOT / variance["source_manifest"]).read_text())
    old, = [c for c in old_manifest["cells"] if (c["task"], c["optimizer_seed"]) == (cell["task"], cell["optimizer_seed"])]
    with measured_phase("load_unchanged_C0"):
        _, controllers, _, args = load_sources(old)
        rule = TimingRule(**json.loads((ROOT / old["phase_cache_directory"] / "frozen_rules.json").read_text())["checks_clock"])
        if args.horizon != manifest["horizon"]:
            raise ValueError("frozen horizon differs from the information budget")
    spec = PlanValidityTaskSpec(cell["task"], _task_options(args), args.env_id)
    return controllers["history"], rule, args, spec


def run_cell(cell, manifest):
    source = json.loads((ROOT / manifest["source_manifest"]).read_text())
    with measured_phase("read_existing_training_windows"):
        rows = load_cache(cell, source)
        cached = min(rows, key=lambda r: (r["seed"], r["opportunity_step"]))
    controller, rule, args, spec = load_frozen_C0(cell, manifest)
    with measured_phase("cached_and_fresh_native_precheck"):
        return native_precheck(controller, rule, cached, spec=spec, args=args, seed=cell["precheck_seed"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    cell = manifest["cells"][args.cell_index]
    directory = ROOT / cell["output_directory"] / "precheck"
    if directory.exists():
        raise FileExistsError("preserve measured information prechecks")
    for name in ("public", "private"):
        (directory / name).mkdir(parents=True)
    ledger = ComputeLedger()
    result = {"run_name": manifest["run_name"], "task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
        "status": "running", "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "performance_claim_supported": False, "lifecycle_costs_complete": False}
    raw = None
    try:
        with ledger.recording("boundary_information_precheck"), single_thread_runtime() as profile:
            if profile != manifest["source_hardware_profile"]:
                raise ValueError("runtime differs from the frozen pilot")
            result["hardware_profile"] = profile
            raw = run_cell(cell, manifest)
        counts = ledger.summary()["counts"]
        if (counts.get("environment_steps") != manifest["precheck_steps_per_root"] or any(counts.get(k, 0)
            for k in ("ridge_fit_calls", "constant_fit_calls", "controller_optimizer_updates", "termination_optimizer_updates"))):
            raise RuntimeError("information precheck exceeded its replay-only budget")
        result.update(status="complete", stage="boundary_information_precheck_passed", factual_C0_replay=True,
            cached_windows_unchanged=True, fresh_windows_unchanged=True, G_equals_W_plus_tail=True,
            cached_prefix=raw["cached_extension"]["seed"], cached_step=raw["cached_extension"]["opportunity_step"],
            fresh_prefix=raw["fresh_prefix"]["seed"], fresh_step=raw["fresh_prefix"]["opportunity_step"],
            new_training_and_readout_cohorts_accessed=False, fits=0)
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    result.update(measured_cost=ledger.summary(), new_environment_steps=ledger.summary()["counts"].get("environment_steps", 0),
        peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (directory / "private/result.json").write_text(json.dumps(_json_ready({**result, "precheck_rows": raw}), indent=2) + "\n")
    (directory / "public/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
    print(f"{cell['task']} {cell['optimizer_seed']}: {result.get('stage', result['status'])}", flush=True)
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
