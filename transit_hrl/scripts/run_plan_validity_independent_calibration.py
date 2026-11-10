"""Precheck, independent calibration sampling, cut freeze, then fresh-prefix readout."""

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
from freq_hrl.experiments.plan_validity_independent_calibration import cut_readout, select_cuts, validate_rows
from freq_hrl.experiments.plan_validity_phase_support import observe, phase_counts, predict, sample
from freq_hrl.experiments.plan_validity_transfer import PlanValidityTaskSpec
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts.run_plan_validity_target_credit import load_sources


def phase_directory(cell, phase, shard=-1):
    directory = ROOT / cell["output_directory"] / phase
    return directory if shard < 0 else directory / f"shard{shard}"


def completed(cell, manifest, phase, shard=-1):
    directory = phase_directory(cell, phase, shard)
    saved = json.loads((directory / "public/result.json").read_text())
    if (saved["status"] != "complete" or saved["run_name"] != manifest["run_name"]
        or saved["task"] != cell["task"] or saved["optimizer_seed"] != cell["optimizer_seed"]
        or saved["phase"] != phase or saved["shard_index"] != shard):
        raise ValueError("matching completed " + phase + " required")
    return directory, saved


def run_cell(cell, manifest, phase, shard, directory, publish):
    # Gate new readout paths before loading any native controller or task.
    if phase != "precheck":
        for other in manifest["cells"]:
            completed(other, manifest, "select" if phase == "readout" else "precheck")
    with measured_phase("load_unchanged_artifacts"):
        _, controllers, _, args = load_sources(cell)
        c0 = TimingRule(**json.loads((ROOT / cell["phase_cache_directory"] / "frozen_rules.json").read_text())["checks_clock"])
        frozen_text = (ROOT / cell["fit_directory"] / "private/frozen_rules.json").read_text()
        rules = {k: TimingRule(**v) for k, v in json.loads(frozen_text).items()}
        previous = json.loads((ROOT / cell["previous_calibration_directory"] / "private/frozen_cuts.json").read_text())["frozen_full_scores"]
        if args.horizon != manifest["horizon"]:
            raise ValueError("native horizon differs from the frozen budget")
    controller = controllers["history"]
    spec = PlanValidityTaskSpec(cell["task"], _task_options(args), args.env_id)

    def observation(seed):
        return observe(controller, c0, spec=spec, args=args, seed=seed, probes_per_path=3)

    def acquire(obs, steps, a, b):
        return sample(controller, c0, obs, steps, spec=spec, args=args, fitting_futures=a, evaluation_futures=b)

    if phase == "precheck":
        with measured_phase("new_prefix_native_precheck"):
            for seed in cell["precheck_seeds"]:
                obs = observation(seed)
                later = [s for s in obs["available_steps"] if s % 50 > 0]
                if later:
                    break
            if not later:
                raise RuntimeError("precheck has no later optional check")
            rows = acquire(obs, later[:1], 1, 1)
            validate_rows(rows, seeds=[seed], probes=1, a=1, b=1)
            row = rows[0]
            query = next(q for q in obs["factual"]["queries"] if q["step"] == later[0])
            if not np.isclose(predict(c0, rows)[0], query["score"], rtol=1e-10, atol=1e-10):
                raise RuntimeError("C0 native and cached scores differ")
            for rule in rules.values():
                native = rule.score(tuple(row["feature_names"]), row["causal_features"], row["offset_fraction"], row["remaining_fraction"])
                if not np.isclose(predict(rule, rows)[0], native, rtol=1e-10, atol=1e-10):
                    raise RuntimeError("frozen head native and full score differ")
        return {"stage": "precheck_passed", "sampled_seeds": [seed], "nonzero_phase": later[0] % 50,
            "factual_replay_and_conditional_pairing_passed": True, "unchanged_head_score_path_passed": True,
            "upper_calls": obs["factual"]["upper_decision_count"]}, rows

    if phase == "select":
        rows = []
        with measured_phase("read_paid_independent_calibration"):
            for index in (0, 1):
                path, saved = completed(cell, manifest, "sample", index)
                if saved["sampled_seeds"] != cell["calibration_prefix_seeds"][8 * index:8 * (index + 1)]:
                    raise ValueError("calibration shard roster differs")
                rows.extend(json.loads((path / "private/rows.json").read_text()))
            validate_rows(rows, seeds=cell["calibration_prefix_seeds"], probes=3, a=4, b=4)
        with measured_phase("freeze_A_only_cuts_on_fixed_full_scores"):
            selected = select_cuts(rows, rules)
            (directory / "private/frozen_cuts.json").write_text(json.dumps(_json_ready(selected), indent=2) + "\n")
        publish({"stage": "both_cuts_frozen_before_calibration_B", "selected": selected})
        with measured_phase("calibration_B_readout"):
            report = cut_readout(rows, rules, selected, previous, fresh_prefix=False)
        return {"stage": "cuts_frozen", "selected": selected, "calibration_B": report,
            "states": len(rows), "prefixes": 16}, None

    if phase == "readout":
        path, _ = completed(cell, manifest, "select")
        selected = json.loads((path / "private/frozen_cuts.json").read_text())
        seeds = cell["readout_prefix_seeds"]
    else:
        seeds = cell["calibration_prefix_seeds"][8 * shard:8 * (shard + 1)]
    rows = []
    with measured_phase("new_prefix_" + phase):
        for seed in seeds:
            obs = observation(seed)
            rows.extend(acquire(obs, obs["check_steps"], 4, 4))
            (directory / "private/rows.json").write_text(json.dumps(_json_ready(rows)) + "\n")
            publish({"stage": "sampling_" + phase, "completed_prefixes": len(rows) // 3})
    validate_rows(rows, seeds=seeds, probes=3, a=4, b=4)
    result = {"stage": phase + "_complete", "sampled_seeds": seeds, "states": len(rows), "phase_counts": phase_counts(rows)}
    if phase == "readout":
        with measured_phase("frozen_fresh_prefix_E8_readout"):
            result["fresh_prefix_E8"] = cut_readout(rows, rules, selected, previous, fresh_prefix=True)
    return result, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    parser.add_argument("--phase", choices=("precheck", "sample", "select", "readout"), required=True)
    parser.add_argument("--shard-index", type=int, default=-1)
    args = parser.parse_args(argv)
    if args.shard_index not in ((0, 1) if args.phase == "sample" else (-1,)):
        raise ValueError("only calibration sampling has two shards")
    manifest = json.loads(args.manifest.read_text())
    cell = manifest["cells"][args.cell_index]
    directory = phase_directory(cell, args.phase, args.shard_index)
    if directory.exists():
        raise FileExistsError("preserve paid independent-calibration attempts")
    for name in ("public", "private"):
        (directory / name).mkdir(parents=True)
    ledger = ComputeLedger()
    result = {"run_name": manifest["run_name"], "task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
        "phase": args.phase, "shard_index": args.shard_index, "status": "running",
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "performance_claim_supported": False, "lifecycle_costs_complete": False}

    def publish(update):
        result.update(update, measured_cost=ledger.summary())
        (directory / "public/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
        print(f"{cell['task']} {cell['optimizer_seed']} {args.phase}/{args.shard_index}: {result.get('stage', result['status'])}", flush=True)

    raw = None
    try:
        with ledger.recording("independent_calibration_" + args.phase), single_thread_runtime() as profile:
            if profile != manifest["source_hardware_profile"]:
                raise ValueError("runtime differs from the frozen pilot")
            result["hardware_profile"] = profile
            summary, raw = run_cell(cell, manifest, args.phase, args.shard_index, directory, publish)
        counts = ledger.summary()["counts"]
        if any(counts.get(k, 0) for k in ("ridge_fit_calls", "controller_optimizer_updates", "termination_optimizer_updates")):
            raise RuntimeError("independent calibration unexpectedly fitted a model")
        steps = counts.get("environment_steps", 0)
        budget = manifest["per_job_environment_steps"][args.phase]
        if steps > budget or (args.phase != "precheck" and steps != budget):
            raise RuntimeError("native work differs from the frozen phase budget")
        result.update(summary, status="complete")
    except Exception as exc:
        result.update(status="failed", stage="failed", error_type=type(exc).__name__, error=str(exc))
    cost = ledger.summary()
    result.update(measured_cost=cost, new_environment_steps=cost["counts"].get("environment_steps", 0),
        peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (directory / "private/result.json").write_text(json.dumps(_json_ready({**result, "precheck_rows": raw}), indent=2) + "\n")
    publish({})
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
