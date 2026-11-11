"""Acquire only frozen Dense/Wide A labels; no fitting or evaluation access."""

from collections import Counter
import argparse
import json
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.compute_budget import ComputeLedger, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_boundary_information import acquire_windows, fresh_reference
from freq_hrl.experiments.plan_validity_phase_support import observe
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from scripts.run_plan_validity_boundary_information_precheck import load_frozen_C0
from scripts.run_plan_validity_boundary_tail import load_cache


def check_prechecks(design):
    for cell in design["cells"]:
        r = json.loads((ROOT / cell["output_directory"] / "precheck/public/result.json").read_text())
        if (r["run_name"], r["task"], r["optimizer_seed"], r["status"], r["stage"],
            r["fresh_prefix"], r["new_environment_steps"], r["fits"]) != (
            design["run_name"], cell["task"], cell["optimizer_seed"], "complete", "boundary_information_precheck_passed",
            cell["precheck_seed"], design["precheck_steps_per_root"], 0):
            raise ValueError("all four matching information prechecks must pass")


def sample_rows(cell, design, shard, controller, rule, args, spec, cached, persist):
    if shard["method"] == "dense":
        extras = {(k["seed"], k["state_index"], k["window_index"]) for k in cell["dense_extra_windows"]}
        for seed in shard["prefix_seeds"]:
            states = sorted((r for r in cached if r["seed"] == seed), key=lambda r: r["opportunity_step"])
            for i, row in enumerate(states):
                indices = {j: [*design["dense"]["new_A_indices"], *(
                    [design["dense"]["extra_A_index"]] if (seed, i, j) in extras else [])]
                    for j in range(design["windows_per_state"])}
                persist(acquire_windows(controller, rule, row, indices, spec=spec, args=args))
    else:
        for seed in shard["prefix_seeds"]:
            observation = observe(controller, rule, spec=spec, args=args, seed=seed, probes_per_path=design["checks_per_prefix"])
            for step in sorted(observation["check_steps"]):
                reference = fresh_reference(observation, step, args=args, windows=design["windows_per_state"])
                indices = {j: design["wide"]["new_A_indices"] for j in range(design["windows_per_state"])}
                persist(acquire_windows(controller, rule, reference, indices, spec=spec, args=args))


def run_shard(cell, design, shard, directory, publish):
    check_prechecks(design)
    cached = None
    if shard["method"] == "dense":
        source = json.loads((ROOT / design["source_manifest"]).read_text())
        with measured_phase("read_existing_training_windows"):
            cached = load_cache(cell, source)
    controller, rule, args, spec = load_frozen_C0(cell, design)
    rows = []

    def persist(row):
        rows.append(row)
        (directory / "private/rows.json").write_text(json.dumps(_json_ready(rows)) + "\n")
        counts = Counter(r["seed"] for r in rows)
        publish({"stage": "sampling_" + shard["method"], "completed_states": len(rows),
            "completed_prefixes": sum(n == design["checks_per_prefix"] for n in counts.values()),
            "completed_pairs": sum(len(w["A_replicas"]) for r in rows for w in r["windows"])})

    with measured_phase("native_A_acquisition"):
        sample_rows(cell, design, shard, controller, rule, args, spec, cached, persist)
    roster = [(r["seed"], r["opportunity_step"]) for r in rows]
    pairs = sum(len(w["A_replicas"]) for r in rows for w in r["windows"])
    if (len(rows) != 12 or len(set(roster)) != 12 or set(r["seed"] for r in rows) != set(shard["prefix_seeds"])
        or any(sum(r["seed"] == s for r in rows) != 3 for s in shard["prefix_seeds"])
        or any(len(r["windows"]) != 4 for r in rows) or pairs != shard["new_pairs"]):
        raise RuntimeError("sampled prefix/state/window/pair roster differs from the frozen shard")
    return {"stage": "boundary_information_sample_complete", "sampled_prefix_seeds": shard["prefix_seeds"],
        "completed_states": len(rows), "completed_prefixes": 4, "completed_pairs": pairs,
        "evaluation_prefixes_accessed": False, "fits": 0}


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
        raise FileExistsError("preserve paid information-sampling attempts")
    for name in ("public", "private"):
        (directory / name).mkdir(parents=True)
    ledger = ComputeLedger()
    result = {"run_name": design["run_name"], "task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
        "method": shard["method"], "shard_index": shard["shard_index"], "status": "running",
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "performance_claim_supported": False, "lifecycle_costs_complete": False}

    def publish(update):
        result.update(update, measured_cost=ledger.summary())
        (directory / "public/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
        print(f"{cell['task']} {cell['optimizer_seed']} {shard['method']}/{shard['shard_index']}: {result.get('stage', result['status'])} {result.get('completed_states', '')}", flush=True)

    try:
        with ledger.recording("boundary_information_sample"), single_thread_runtime() as profile:
            if profile != design["source_hardware_profile"]:
                raise ValueError("runtime differs from the frozen pilot")
            result["hardware_profile"] = profile
            publish({"stage": "checking_prechecks_and_frozen_sources"})
            sampled = run_shard(cell, design, shard, directory, publish)
        counts = ledger.summary()["counts"]
        if (counts.get("environment_steps") != shard["environment_steps"] or any(counts.get(k, 0)
            for k in ("ridge_fit_calls", "constant_fit_calls", "controller_optimizer_updates", "termination_optimizer_updates"))):
            raise RuntimeError("training acquisition exceeded the no-fit native budget")
        result.update(sampled, status="complete")
    except Exception as exc:
        result.update(status="failed", stage="failed", error_type=type(exc).__name__, error=str(exc))
    result.update(measured_cost=ledger.summary(), new_environment_steps=ledger.summary()["counts"].get("environment_steps", 0),
        peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (directory / "private/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
    publish({})
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
