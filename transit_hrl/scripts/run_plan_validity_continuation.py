"""Read frozen timing policies and cross their prefix and continuation behavior."""

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
from freq_hrl.experiments.plan_validity_continuation import (
    POLICIES, factual_pair, occupancy, paired_probe, probe_summary, query_support, selected_checks,
)
from freq_hrl.experiments.plan_validity_transfer import PlanValidityTaskSpec, run_episode, time_contract
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts.run_plan_validity_target_credit import load_sources


def run_cell(cell, manifest, phase, shard, directory, publish):
    with measured_phase("load_frozen_artifacts"):
        records, controllers, original, args = load_sources(cell)
        source = json.loads((ROOT / cell["phase_public_result"]).read_text())
        private = ROOT / cell["phase_cache_directory"]
        saved = json.loads((private / "frozen_rules.json").read_text())
        if source["status"] != "complete" or source["optimizer_seed"] != cell["optimizer_seed"]:
            raise ValueError("phase-support source is incomplete or mismatched")
        rules = {"reference": original["history"], "candidate": TimingRule(**saved["checks_clock"])}
        fit_rows = json.loads((private / "checks_bank.json").read_text())
        controller = controllers["history"]
    spec = PlanValidityTaskSpec(cell["task"], _task_options(args), args.env_id)
    if phase == "precheck":
        with measured_phase("native_source_reproduction"):
            for name, method in (("reference", "history"), ("candidate", "checks_clock")):
                by_seed = {r["seed"]: r[method] for r in source["episodes"]}
                if sorted(by_seed) != sorted(cell["previous_evaluation_seeds"]):
                    raise ValueError("source evaluation cohort differs from the frozen manifest")
                for seed in cell["previous_evaluation_seeds"]:
                    row = run_episode(controller, task_spec=spec, seed=seed, args=args,
                        time_scale=time_contract(args), rule=rules[name])[1]
                    if not np.isclose(row["tracking_squared_error_integral"], by_seed[seed], rtol=1e-10, atol=1e-10):
                        raise RuntimeError("native frozen source episode does not reproduce")
        with measured_phase("native_continuation_precheck"):
            later = []
            for seed in cell["fit_prefix_seeds"][:4]:
                factual = factual_pair(controller, rules, spec=spec, args=args, seed=seed)
                common = query_support(factual, horizon=args.horizon)["common"]
                later = sorted(s for s in common if s % 50 > 0)
                if later:
                    break
            if not later:
                raise RuntimeError("precheck prefix has no common later optional check")
            probe = paired_probe(controller, rules, factual, spec=spec, args=args, seed=seed, step=later[0], replicas=2)
        return {"stage": "precheck_passed", "source_episodes_reproduced": True,
            "same_policy_factual_replay": True, "cross_continuation_prefix_preserved": True,
            "nonzero_phase": later[0] % 50}, {"probe": probe}
    gate = json.loads((ROOT / cell["output_directory"] / "precheck/public/result.json").read_text())
    if (gate["status"] != "complete" or gate["stage"] != "precheck_passed"
        or gate["run_name"] != manifest["run_name"] or gate["optimizer_seed"] != cell["optimizer_seed"]):
        raise ValueError("matching continuation precheck required before new diagnostic paths")
    seeds = cell["evaluation_seeds"][shard * 4:(shard + 1) * 4]
    probes, path_rows, factual_rows = [], [], []
    with measured_phase("matched_prefix_continuation_sampling"):
        for seed in seeds:
            factual = factual_pair(controller, rules, spec=spec, args=args, seed=seed)
            factual_rows.append(factual)
            steps, support = selected_checks(factual, args=args, seed=seed, probes_per_path=manifest["probes_per_path"])
            path_probes = []
            for step in steps:
                probe = paired_probe(controller, rules, factual, spec=spec, args=args, seed=seed,
                    step=step, replicas=manifest["conditional_replicates"])
                probes.append(probe)
                path_probes.append(probe)
                (directory / "private/probes.json").write_text(json.dumps(_json_ready(probes)) + "\n")
                publish({"stage": "matched_prefix_continuation_sampling", "completed_probes": len(probes)})
            path_rows.append({"seed": seed, "support_counts": support["counts"], "support_phase_counts": support["phase_counts"],
                "episodes": {p: factual[p]["tracking_squared_error_integral"] for p in POLICIES},
                "probes": len(path_probes), "blocks": probe_summary(path_probes)})
            (directory / "private/paths.json").write_text(json.dumps(_json_ready(path_rows)) + "\n")
    with measured_phase("occupancy_readout"):
        distribution = occupancy(fit_rows, factual_rows, horizon=args.horizon)
    return {"stage": "continuation_diagnostic_complete", "paths": path_rows,
        "evaluation_seeds": seeds, "occupancy": distribution,
        "termination_development_eligible": records["termination_history"]["termination_readiness"]["eligible_for_fresh_confirmation"]}, \
        {"probes": probes, "source_phase_cache": cell["phase_cache_directory"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    parser.add_argument("--phase", choices=("precheck", "run"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    options = parser.parse_args(argv)
    if options.shard_index not in ((-1,) if options.phase == "precheck" else (0, 1)):
        raise ValueError("precheck uses shard -1; the main run uses shards 0 and 1")
    manifest = json.loads(options.manifest.read_text())
    cell = manifest["cells"][options.cell_index]
    directory = ROOT / cell["output_directory"] / ("precheck" if options.phase == "precheck" else f"run/shard{options.shard_index}")
    if directory.exists():
        raise FileExistsError("preserve paid attempts and samples")
    for name in ("private", "public"):
        (directory / name).mkdir(parents=True)
    ledger = ComputeLedger()
    result = {"run_name": manifest["run_name"], "task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
        "phase": options.phase, "shard_index": options.shard_index, "status": "running",
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "lifecycle_costs_complete": False, "performance_claim_supported": False}

    def publish(update):
        result.update(update, measured_cost=ledger.summary())
        (directory / "public/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
        print(f"{cell['task']} {cell['optimizer_seed']} shard{options.shard_index}: {result.get('stage',result['status'])} {result.get('completed_probes','')}", flush=True)

    raw = None
    try:
        with ledger.recording("continuation_" + options.phase), single_thread_runtime() as profile:
            if profile != manifest["source_hardware_profile"]:
                raise ValueError("runtime differs from the frozen pilot")
            result["hardware_profile"] = profile
            publish({"stage": "loading_frozen_artifacts"})
            summary, raw = run_cell(cell, manifest, options.phase, options.shard_index, directory, publish)
        if any(ledger.summary()["counts"].get(k, 0) for k in
            ("controller_optimizer_updates", "termination_optimizer_updates", "ridge_fit_calls")):
            raise RuntimeError("frozen diagnostic unexpectedly performed fitting")
        result.update(summary, status="complete")
    except Exception as exc:
        result.update(status="failed", stage="failed", error_type=type(exc).__name__, error=str(exc))
    result.update(measured_cost=ledger.summary(), peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (directory / "private/result.json").write_text(json.dumps(_json_ready({**result, "raw_data": raw}), indent=2) + "\n")
    publish({})
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
