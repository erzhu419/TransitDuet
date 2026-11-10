"""Read frozen pilot artifacts and diagnose local-to-episode transfer, without fitting."""

import argparse
import json
from pathlib import Path
import resource
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.compute_budget import ComputeLedger, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_reliability import cross_future_attribution, evaluate_profile
from freq_hrl.experiments.plan_validity_transfer import PlanValidityTaskSpec, run_episode, time_contract
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts.run_plan_validity_reliability import load_controller, load_record


def run_cell(cell, manifest, publish):
    expected = cell["expected_configuration"]
    sources, controllers, rules, args_by_method = {}, {}, {}, {}
    for method, source in cell["sources"].items():
        row = load_record(Path(source["report"]), source["run_id"])
        if (row["task"] != cell["task"] or row["optimizer_seed"] != cell["optimizer_seed"]
            or row["method"] != method or row["configuration"] != {**expected, "method": method}
            or {k: v for k, v in row["selected_rule"].items() if k != "model"} != source["selected_rule_summary"]):
            raise ValueError("frozen source parameters, rule or identity changed")
        exposed = set().union(*(row["configuration"][k] for k in
            ("train_seeds", "selection_seeds", "branch_fit_seeds", "timing_selection_seeds")))
        if (cell["evaluation_seeds"] != row["configuration"]["trigger_eval_seeds"]
            or exposed.intersection(cell["evaluation_seeds"])):
            raise ValueError("diagnostic must use only the reserved pilot evaluation role")
        sources[method] = row
        with measured_phase("load_artifacts"):
            controllers[method], args_by_method[method] = load_controller(row)
        rules[method] = TimingRule(**row["selected_rule"])
    args = args_by_method["history"]
    scale = time_contract(args)
    spec = PlanValidityTaskSpec(cell["task"], _task_options(args), args.env_id)
    replay = {}
    with measured_phase("native_reproduction_precheck"):
        for method in sources:
            scores = [run_episode(controllers[method], task_spec=spec, seed=seed, args=args,
                time_scale=scale, rule=rules[method])[1]["tracking_squared_error_integral"]
                for seed in args.timing_selection_seeds]
            replay[method] = float(np.mean(scores))
            if not np.isclose(replay[method], cell["sources"][method]["validation_ise"], rtol=1e-10, atol=1e-10):
                raise RuntimeError("native reproduction differs from the original selected artifact")
    publish({"stage": "native_precheck_passed", "validation_reproduction": replay})
    fit_rows = json.loads(Path(sources["history"]["fit_cache_file"]).read_text())["branch_labels"]
    with measured_phase("frozen_diagnostic"):
        profile = evaluate_profile(controllers["history"], controllers["threshold"], rules["history"], rules["threshold"],
            task_spec=spec, args=args, seeds=cell["evaluation_seeds"], probe_seeds=cell["probe_seeds"], fit_rows=fit_rows,
            probes_per_episode=manifest["probes_per_episode"], risk_spec=manifest["risk_spec"],
            conditional_replicates=manifest["conditional_replicates"])
        termination = [run_episode(controllers["termination_history"], task_spec=spec, seed=seed, args=args,
            time_scale=scale, rule=rules["termination_history"])[1]["tracking_squared_error_integral"]
            for seed in cell["evaluation_seeds"]]
    episode_rows = [{"seed": row["seed"], "history": row["candidate"]["tracking_squared_error_integral"],
        "threshold": row["comparator"]["tracking_squared_error_integral"], "termination_history": value}
        for row, value in zip(profile["episodes"], termination)]
    halves = np.array_split(np.arange(len(episode_rows)), 2)
    panels = {}
    for name, indices in (("all", np.arange(len(episode_rows))), ("first_half", halves[0]), ("second_half", halves[1])):
        means = {m: float(np.mean([episode_rows[i][m] for i in indices]))
                 for m in ("history", "threshold", "termination_history")}
        panels[name] = {"paths": len(indices), "mean_ise": means,
            "history_minus_comparator": {m: means["history"] - means[m] for m in ("threshold", "termination_history")}}
    summary = {"stage": "diagnostic_complete", "validation_reproduction": replay, "episode_panels": panels,
        "episodes": episode_rows, "local_errors": profile["local_errors"], "occupancy": profile["occupancy"],
        "probe_sampling": profile["probe_sampling"],
        "cross_future_attribution": cross_future_attribution(profile["probes"], threshold=rules["history"].threshold),
        "termination_development_eligible": sources["termination_history"]["termination_readiness"]["eligible_for_fresh_confirmation"]}
    return summary, profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    options = parser.parse_args(argv)
    manifest = json.loads(options.manifest.read_text())
    cell = manifest["cells"][options.cell_index]
    directory = ROOT / cell["output_directory"]
    private, public = directory / "private/result.json", directory / "public/result.json"
    if private.exists() or public.exists():
        raise FileExistsError("preserve this diagnostic attempt and its measured costs")
    private.parent.mkdir(parents=True, exist_ok=True)
    public.parent.mkdir(parents=True, exist_ok=True)
    ledger = ComputeLedger()
    result = {"status": "running", "run_name": manifest["run_name"], "task": cell["task"],
        "optimizer_seed": cell["optimizer_seed"], "scope": manifest["scope"],
        "evaluation_seeds": cell["evaluation_seeds"], "probe_seeds": cell["probe_seeds"],
        "probes_per_episode": manifest["probes_per_episode"], "conditional_replicates": manifest["conditional_replicates"],
        "source_run_ids": {name: source["run_id"] for name, source in cell["sources"].items()},
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "lifecycle_costs_complete": False, "performance_claim_supported": False}

    def publish(update):
        result.update(update, evaluation_cost=ledger.summary())
        public.write_text(json.dumps(_json_ready(result), indent=2) + "\n")
        print(f"{cell['task']} {cell['optimizer_seed']}: {result.get('stage', result['status'])}", flush=True)

    publish({"stage": "loading_frozen_artifacts"})
    profile = None
    try:
        with ledger.recording("development_diagnostic"), single_thread_runtime() as hardware:
            result["hardware_profile"] = hardware
            if hardware != manifest["source_hardware_profile"]:
                raise ValueError("diagnostic runtime differs from the measured pilot runtime")
            summary, profile = run_cell(cell, manifest, publish)
        counts = ledger.summary()["counts"]
        if any(counts.get(k, 0) for k in ("controller_optimizer_updates", "ridge_fit_calls", "termination_optimizer_updates")):
            raise RuntimeError("frozen diagnostic unexpectedly performed fitting")
        result.update(summary, status="complete")
    except Exception as exc:
        result.update(status="failed", stage="failed", error_type=type(exc).__name__, error=str(exc))
    result.update(evaluation_cost=ledger.summary(), peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    private.write_text(json.dumps(_json_ready({**result, "raw_profile": profile}), indent=2) + "\n")
    publish({})
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
