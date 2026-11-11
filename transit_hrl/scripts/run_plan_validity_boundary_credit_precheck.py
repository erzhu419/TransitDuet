"""Check paired boundary redraws on the frozen C0 controllers, without fitting."""

import argparse
import json
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.compute_budget import ComputeLedger, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_boundary_credit import collect_boundary_pair
from freq_hrl.experiments.plan_validity_phase_support import observe
from freq_hrl.experiments.plan_validity_transfer import PlanValidityTaskSpec
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts.run_plan_validity_target_credit import load_sources


def run_cell(cell, source, manifest):
    old, = [c for c in source["cells"] if (c["task"], c["optimizer_seed"]) == (cell["task"], cell["optimizer_seed"])]
    with measured_phase("load_frozen_C0"):
        _, controllers, _, args = load_sources(old)
        rule = TimingRule(**json.loads((ROOT / old["phase_cache_directory"] / "frozen_rules.json").read_text())["checks_clock"])
        if args.horizon != manifest["horizon"]:
            raise ValueError("source horizon differs from the precheck budget")
    spec = PlanValidityTaskSpec(cell["task"], _task_options(args), args.env_id)
    with measured_phase("nested_boundary_pair"):
        observation = observe(controllers["history"], rule, spec=spec, args=args,
            seed=cell["precheck_seed"], probes_per_path=1)
        step = next((s for s in observation["available_steps"] if s % 50 > 0), None)
        if step is None:
            raise RuntimeError("precheck prefix has no nonzero optional phase")
        result = collect_boundary_pair(controllers["history"], rule, observation, step,
            spec=spec, args=args, windows=manifest["window_draws"], suffixes=manifest["suffix_draws_per_window"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    cell = manifest["cells"][args.cell_index]
    source = json.loads((ROOT / manifest["source_manifest"]).read_text())
    directory = ROOT / cell["output_directory"]
    if directory.exists():
        raise FileExistsError("preserve measured boundary-credit attempts")
    for name in ("public", "private"):
        (directory / name).mkdir(parents=True)
    ledger = ComputeLedger()
    result = {"run_name": manifest["run_name"], "task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
        "precheck_seed": cell["precheck_seed"], "status": "running", "candidate_recipe_decision": "PILOT_RECIPE_HOLD",
        "confirmation_released": False, "performance_claim_supported": False, "lifecycle_costs_complete": False}
    raw = None
    try:
        with ledger.recording("boundary_credit_precheck"), single_thread_runtime() as profile:
            if profile != source["source_hardware_profile"]:
                raise ValueError("runtime differs from the frozen pilot")
            result["hardware_profile"] = profile
            raw = run_cell(cell, source, manifest)
        counts = ledger.summary()["counts"]
        if counts.get("environment_steps", 0) != manifest["per_job_environment_steps"] or any(
            counts.get(k, 0) for k in ("ridge_fit_calls", "controller_optimizer_updates", "termination_optimizer_updates")
        ):
            raise RuntimeError("boundary precheck exceeded its replay-only contract")
        result.update(status="complete", stage="boundary_precheck_passed",
            intervention_step=raw["opportunity_step"], boundary_step=raw["boundary_step"],
            window_draws=len(raw["windows"]), suffix_draws_per_window=len(raw["windows"][0]["suffix_draws"]),
            causal_feature_count=len(raw["causal_features"]), factual_C0_replay_passed=True,
            fixed_window_and_boundary_context_passed=True, G_equals_W_plus_tail_passed=True)
    except Exception as exc:
        result.update(status="failed", stage="failed", error_type=type(exc).__name__, error=str(exc))
    cost = ledger.summary()
    result.update(measured_cost=cost, new_environment_steps=cost["counts"].get("environment_steps", 0),
        peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (directory / "private/result.json").write_text(json.dumps(_json_ready({**result, "boundary_pair": raw}), indent=2) + "\n")
    (directory / "public/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
    print(f"{cell['task']} {cell['optimizer_seed']}: {result['stage']}", flush=True)
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
