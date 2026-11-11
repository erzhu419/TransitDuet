"""Fixed-budget nested credit variance diagnostic; no model or policy fitting."""

import argparse
import json
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.compute_budget import ComputeLedger, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_boundary_credit import collect_boundary_pair, boundary_state_report, summarize_boundary_reports
from freq_hrl.experiments.plan_validity_phase_support import observe
from freq_hrl.experiments.plan_validity_transfer import PlanValidityTaskSpec
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts.run_plan_validity_target_credit import load_sources


def run_cell(cell, manifest, shard, directory, publish):
    precheck = json.loads((ROOT / manifest["precheck_manifest"]).read_text())
    for c in precheck["cells"]:
        saved = json.loads((ROOT / c["output_directory"] / "public/result.json").read_text())
        if (saved["run_name"], saved["task"], saved["optimizer_seed"], saved["status"], saved["stage"]) != (
            precheck["run_name"], c["task"], c["optimizer_seed"], "complete", "boundary_precheck_passed"
        ):
            raise ValueError("all four matching boundary prechecks must pass")
    source = json.loads((ROOT / manifest["source_manifest"]).read_text())
    old, = [c for c in source["cells"] if (c["task"], c["optimizer_seed"]) == (cell["task"], cell["optimizer_seed"])]
    with measured_phase("load_frozen_C0"):
        _, controllers, _, args = load_sources(old)
        rule = TimingRule(**json.loads((ROOT / old["phase_cache_directory"] / "frozen_rules.json").read_text())["checks_clock"])
        if args.horizon != manifest["horizon"]:
            raise ValueError("source horizon differs from the variance budget")
    spec = PlanValidityTaskSpec(cell["task"], _task_options(args), args.env_id)
    seeds = cell["development_prefix_seeds"][4 * shard:4 * (shard + 1)]
    rows, reports = [], []
    with measured_phase("nested_boundary_sampling"):
        for seed in seeds:
            observation = observe(controllers["history"], rule, spec=spec, args=args, seed=seed, probes_per_path=3)
            for step in observation["check_steps"]:
                row = collect_boundary_pair(controllers["history"], rule, observation, step,
                    spec=spec, args=args, windows=manifest["window_draws"], suffixes=manifest["suffix_draws_per_window"])
                rows.append(row)
                reports.append(boundary_state_report(row))
                (directory / "private/rows.json").write_text(json.dumps(_json_ready(rows)) + "\n")
                publish({"stage": "sampling_boundary_variance", "completed_states": len(rows)})
    roster = [(r["seed"], r["opportunity_step"]) for r in rows]
    if (len(rows) != 12 or len(set(roster)) != 12 or set(r["seed"] for r in rows) != set(seeds)
        or any(sum(r["seed"] == s for r in rows) != 3 for s in seeds)):
        raise RuntimeError("development-prefix roster differs from the frozen plan")
    with measured_phase("variance_readout"):
        summary = summarize_boundary_reports(reports)
        by_prefix = [{"seed": seed, **summarize_boundary_reports([r for r in reports if r["seed"] == seed])} for seed in seeds]
    return {"stage": "boundary_variance_complete", "sampled_seeds": seeds, "summary": summary,
        "prefix_reports": by_prefix, "state_reports": reports,
        "scope": "descriptive_nested_variance_on_new_development_prefixes_no_prediction_or_policy_validation"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    parser.add_argument("--shard-index", type=int, choices=(0, 1), required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    cell = manifest["cells"][args.cell_index]
    source = json.loads((ROOT / manifest["source_manifest"]).read_text())
    directory = ROOT / cell["output_directory"] / f"shard{args.shard_index}"
    if directory.exists():
        raise FileExistsError("preserve paid boundary-variance attempts")
    for name in ("public", "private"):
        (directory / name).mkdir(parents=True)
    ledger = ComputeLedger()
    result = {"run_name": manifest["run_name"], "task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
        "shard_index": args.shard_index, "status": "running", "candidate_recipe_decision": "PILOT_RECIPE_HOLD",
        "confirmation_released": False, "performance_claim_supported": False, "lifecycle_costs_complete": False}

    def publish(update):
        result.update(update, measured_cost=ledger.summary())
        (directory / "public/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
        print(f"{cell['task']} {cell['optimizer_seed']} shard{args.shard_index}: {result.get('stage', result['status'])} {result.get('completed_states', '')}", flush=True)

    try:
        with ledger.recording("boundary_variance"), single_thread_runtime() as profile:
            if profile != source["source_hardware_profile"]:
                raise ValueError("runtime differs from the frozen pilot")
            result["hardware_profile"] = profile
            publish({"stage": "checking_prechecks_and_frozen_sources"})
            summary = run_cell(cell, manifest, args.shard_index, directory, publish)
        counts = ledger.summary()["counts"]
        if counts.get("environment_steps", 0) != manifest["per_job_environment_steps"] or any(
            counts.get(k, 0) for k in ("ridge_fit_calls", "controller_optimizer_updates", "termination_optimizer_updates")
        ):
            raise RuntimeError("boundary variance exceeded its no-fit replay budget")
        result.update(summary, status="complete")
    except Exception as exc:
        result.update(status="failed", stage="failed", error_type=type(exc).__name__, error=str(exc))
    cost = ledger.summary()
    result.update(measured_cost=cost, new_environment_steps=cost["counts"].get("environment_steps", 0),
        peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (directory / "private/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
    publish({})
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
