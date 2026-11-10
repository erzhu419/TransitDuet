"""Native paired supervision, training-only calibration, then one-step readout."""

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
from freq_hrl.experiments.plan_validity_label_continuation import (
    CONTINUATIONS, fit_labels, sample_labels, single_step_readout, validate_banks,
)
from freq_hrl.experiments.plan_validity_phase_support import observe, phase_counts, predict
from freq_hrl.experiments.plan_validity_transfer import PlanValidityTaskSpec
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts.run_plan_validity_target_credit import load_sources


def phase_directory(cell, phase, shard=-1):
    path = ROOT / cell["output_directory"] / phase
    return path if shard < 0 else path / f"shard{shard}"


def complete_phase(cell, manifest, phase, shard=-1):
    directory = phase_directory(cell, phase, shard)
    row = json.loads((directory / "public/result.json").read_text())
    if (row["status"] != "complete" or row["run_name"] != manifest["run_name"]
        or row["task"] != cell["task"] or row["optimizer_seed"] != cell["optimizer_seed"]
        or row["phase"] != phase or row["shard_index"] != shard):
        raise ValueError(f"matching completed {phase} phase required")
    return directory, row


def run_cell(cell, manifest, phase, shard, directory, publish):
    with measured_phase("load_frozen_sources"):
        records, controllers, original, args = load_sources(cell)
        source = json.loads((ROOT / cell["phase_public_result"]).read_text())
        if source["status"] != "complete" or source["optimizer_seed"] != cell["optimizer_seed"]:
            raise ValueError("frozen phase-support source is incomplete or mismatched")
        saved = json.loads((ROOT / cell["phase_cache_directory"] / "frozen_rules.json").read_text())
        rules = {"reference": original["history"], "candidate": TimingRule(**saved["checks_clock"])}
        controller = controllers["history"]
    spec = PlanValidityTaskSpec(cell["task"], _task_options(args), args.env_id)

    def observation(seed):
        return observe(controller, rules["candidate"], spec=spec, args=args, seed=seed,
            probes_per_path=manifest["probes_per_path"])

    def acquire(obs, step, a, b, tag):
        return sample_labels(controller, rules, obs, step, spec=spec, args=args, a=a, b=b, tag=tag)

    if phase == "precheck":
        with measured_phase("native_candidate_label_precheck"):
            for seed in cell["fit_prefix_seeds"][:4]:
                obs = observation(seed)
                later = [s for s in obs["available_steps"] if s % 50 > 0]
                if later:
                    break
            if not later:
                raise RuntimeError("candidate prefixes have no later optional check")
            step = later[0]
            pair = acquire(obs, step, 1, 1, manifest["training_future_tag"])
            banks = {c: [pair[c]] for c in CONTINUATIONS}
            validate_banks(banks, seeds=[seed], probes=1, a=1, b=1)
            native_score = next(q["score"] for q in obs["factual"]["queries"] if q["step"] == step)
            if not np.isclose(predict(rules["candidate"], banks["candidate"])[0], native_score, rtol=1e-10, atol=1e-10):
                raise RuntimeError("native and fitting feature/clock paths differ")
        return {"stage": "precheck_passed", "candidate_factual_replayed": True,
            "shared_state_and_future_pairing": True, "native_fitting_score_matches": True,
            "nonzero_phase": step % 50, "upper_calls": obs["factual"]["upper_decision_count"]}, banks

    complete_phase(cell, manifest, "precheck")
    if phase == "fit":
        with measured_phase("merge_paid_training_banks"):
            banks = {c: [] for c in CONTINUATIONS}
            for index in range(manifest["shards_per_root"]):
                previous, row = complete_phase(cell, manifest, "sample", index)
                if row["sampled_seeds"] != cell["fit_prefix_seeds"][16 * index:16 * (index + 1)]:
                    raise ValueError("sample shard changes its training roster")
                saved_banks = json.loads((previous / "private/label_banks.json").read_text())
                for c in CONTINUATIONS:
                    banks[c].extend(saved_banks[c])
            validate_banks(banks, seeds=cell["fit_prefix_seeds"], probes=manifest["probes_per_path"],
                a=manifest["fitting_futures"], b=manifest["independent_B_futures"])
        with measured_phase("paired_DF8_training_A_fits"):
            heads, reports = fit_labels(banks, effective_df=manifest["effective_df"], threshold_quantile=manifest["threshold_quantile"])
            (directory / "private/frozen_rules.json").write_text(json.dumps(_json_ready({k: v.to_dict() for k, v in heads.items()}), indent=2) + "\n")
        return {"stage": "heads_frozen_before_new_paths", "training_readout": reports,
            "training_phase_counts": phase_counts(banks["candidate"]), "training_paths": len(cell["fit_prefix_seeds"]),
            "training_states_per_head": len(banks["candidate"]),
            "fitting_pairs_per_head": sum(len(r["A_replicas"]) for r in banks["candidate"])}, None

    if phase == "sample":
        seeds = cell["fit_prefix_seeds"][16 * shard:16 * (shard + 1)]
        a, b, tag = manifest["fitting_futures"], manifest["independent_B_futures"], manifest["training_future_tag"]
        active_rules = None
    else:
        # All roots are frozen before any new diagnostic-prefix evaluation.
        for other in manifest["cells"]:
            complete_phase(other, manifest, "fit")
        previous = phase_directory(cell, "fit")
        heads = {k: TimingRule(**v) for k, v in json.loads((previous / "private/frozen_rules.json").read_text()).items()}
        active_rules = {"source_R": rules["reference"], "source_C": rules["candidate"], **heads}
        seeds = cell["evaluation_seeds"][4 * shard:4 * (shard + 1)]
        a, b, tag = manifest["evaluation_A_futures"], manifest["evaluation_B_futures"], manifest["evaluation_future_tag"]

    banks, paths, available = {c: [] for c in CONTINUATIONS}, [], []
    with measured_phase("paired_candidate_states_" + phase):
        for seed in seeds:
            obs = observation(seed)
            available.extend(obs["available_steps"])
            path_banks = {c: [] for c in CONTINUATIONS}
            for step in obs["check_steps"]:
                pair = acquire(obs, step, a, b, tag)
                for c in CONTINUATIONS:
                    banks[c].append(pair[c])
                    path_banks[c].append(pair[c])
                (directory / "private/label_banks.json").write_text(json.dumps(_json_ready(banks)) + "\n")
                publish({"stage": "paired_candidate_states_" + phase, "completed_probes": len(banks["candidate"])})
            if phase == "probe":
                paths.append({"seed": seed, "blocks": single_step_readout(path_banks, active_rules)})
                (directory / "private/path_readouts.json").write_text(json.dumps(_json_ready(paths)) + "\n")
    validate_banks(banks, seeds=seeds, probes=manifest["probes_per_path"], a=a, b=b)
    public = {"stage": phase + "_complete", "sampled_seeds": seeds,
        "states_per_continuation": len(banks["candidate"]), "phase_counts": phase_counts(banks["candidate"]),
        "available_candidate_phase_counts": {str(k): sum(s % 50 == k for s in available) for k in (0, 5, 10, 15, 20)},
        "termination_development_eligible": records["termination_history"]["termination_readiness"]["eligible_for_fresh_confirmation"]}
    if phase == "probe":
        public["paths"] = paths
    return public, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    parser.add_argument("--phase", choices=("precheck", "sample", "fit", "probe"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    options = parser.parse_args(argv)
    if options.shard_index not in ((-1,) if options.phase in ("precheck", "fit") else (0, 1)):
        raise ValueError("unsharded precheck/fit or two sampling/probe shards required")
    manifest = json.loads(options.manifest.read_text())
    cell = manifest["cells"][options.cell_index]
    directory = phase_directory(cell, options.phase, options.shard_index)
    if directory.exists():
        raise FileExistsError("preserve paid label-continuation attempts")
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
        print(f"{cell['task']} {cell['optimizer_seed']} {options.phase}/{options.shard_index}: {result.get('stage','running')} {result.get('completed_probes','')}", flush=True)

    raw = None
    try:
        with ledger.recording("label_continuation_" + options.phase), single_thread_runtime() as profile:
            if profile != manifest["source_hardware_profile"]:
                raise ValueError("runtime differs from the frozen pilot")
            result["hardware_profile"] = profile
            publish({"stage": "loading_frozen_sources"})
            summary, raw = run_cell(cell, manifest, options.phase, options.shard_index, directory, publish)
        counts = ledger.summary()["counts"]
        if any(counts.get(k, 0) for k in ("controller_optimizer_updates", "termination_optimizer_updates")):
            raise RuntimeError("label study unexpectedly updated a controller or termination policy")
        if options.phase != "fit" and counts.get("ridge_fit_calls", 0):
            raise RuntimeError("sampling unexpectedly fitted a head")
        if options.phase == "fit" and counts.get("environment_steps", 0):
            raise RuntimeError("training-only fit unexpectedly accessed an environment")
        result.update(summary, status="complete")
    except Exception as exc:
        result.update(status="failed", stage="failed", error_type=type(exc).__name__, error=str(exc))
    result.update(measured_cost=ledger.summary(), peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (directory / "private/result.json").write_text(json.dumps(_json_ready({**result, "raw_data": raw}), indent=2) + "\n")
    publish({})
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
