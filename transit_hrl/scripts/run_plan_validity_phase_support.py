"""Native 2x2 sampling-support/clock-information W experiment."""

import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.compute_budget import ComputeLedger, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_phase_support import (
    ARMS, clock_row, deployment_counts, fit, held_readout, observe, phase_counts, predict, sample, training_readout,
)
from freq_hrl.experiments.plan_validity_reliability import paired_mean_interval
from freq_hrl.experiments.plan_validity_transfer import PlanValidityTaskSpec, run_episode, time_contract
from freq_hrl.experiments.plan_validity_window_coverage import cached_rows
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts.run_plan_validity_target_credit import load_sources


def episode_panels(rows):
    output = {}
    for panel, subset in (("all", rows), ("first_half", rows[:4]), ("second_half", rows[4:])):
        contrasts = {"sampling_plain": [r["checks_plain"] - r["start_plain"] for r in subset],
            "sampling_clock": [r["checks_clock"] - r["start_clock"] for r in subset],
            "clock_start": [r["start_clock"] - r["start_plain"] for r in subset],
            "clock_checks": [r["checks_clock"] - r["checks_plain"] for r in subset],
            "interaction": [(r["checks_clock"] - r["checks_plain"]) - (r["start_clock"] - r["start_plain"]) for r in subset]}
        output[panel] = {"paths": len(subset),
            "mean_ise": {m: float(np.mean([r[m] for r in subset])) for m in rows[0] if m != "seed"},
            "factorial_contrasts": {k: paired_mean_interval(v) for k, v in contrasts.items()},
            "against_controls": {m: {c: paired_mean_interval([r[m] - r[c] for r in subset])
                for c in ("history", "threshold", "L", "prior_wide_df2", "termination_history")} for m in ARMS}}
    return output


def run_cell(cell, manifest, phase, directory, publish):
    with measured_phase("load_frozen_artifacts"):
        records, controllers, original, args = load_sources(cell)
        coverage = ROOT / cell["coverage_cache_directory"]
        previous = json.loads((coverage / "result.json").read_text())
        public = json.loads((ROOT / cell["coverage_public_result"]).read_text())
        target = ROOT / cell["target_cache_directory"]
        old = json.loads((target / "result.json").read_text())
        if (previous["status"] != "complete" or public["status"] != "complete"
            or public["optimizer_seed"] != cell["optimizer_seed"]):
            raise ValueError("source coverage cell is incomplete or mismatched")
        saved_coverage = {k: TimingRule(**v) for k, v in json.loads((coverage / "frozen_rules.json").read_text()).items()}
        saved_target = {k: TimingRule(**v) for k, v in json.loads((target / "frozen_rules.json").read_text()).items()}
        starts = [clock_row(r, horizon=args.horizon) for r in
            cached_rows(old["raw_banks"]["training_bank"], 1) + previous["raw_banks"]["new_training_bank"]]
        if len(starts) != 96 or set(r["seed"] for r in starts) != set(cell["fit_prefix_seeds"]):
            raise ValueError("the frozen 32-prefix start bank changed")
    spec = PlanValidityTaskSpec(cell["task"], _task_options(args), args.env_id)
    scale = time_contract(args)
    with measured_phase("start_fits_and_reproduction"):
        rules, oofs = {}, {}
        for name in ("start_plain", "start_clock"):
            rules[name], oofs[name] = fit(starts, clock=name.endswith("clock"), effective_df=manifest["effective_df"],
                                          threshold_quantile=manifest["threshold_quantile"])
        if (not np.allclose(predict(rules["start_plain"], starts), predict(saved_coverage["wide_df8"], starts), rtol=1e-12, atol=1e-12)
            or not np.isclose(rules["start_plain"].threshold, saved_coverage["wide_df8"].threshold, rtol=1e-12, atol=1e-12)):
            raise RuntimeError("start/plain no longer reproduces previous wide DF8")

    def observation(seed):
        return observe(controllers["history"], original["history"], spec=spec, args=args, seed=seed,
                       probes_per_path=manifest["probes_per_path"])

    def acquire(obs, steps, a, b):
        return sample(controllers["history"], original["history"], obs, steps, spec=spec, args=args,
                      fitting_futures=a, evaluation_futures=b)

    if phase == "precheck":
        with measured_phase("native_phase_precheck"):
            values = [run_episode(controllers["history"], task_spec=spec, seed=s, args=args, time_scale=scale,
                rule=original["history"])[1]["tracking_squared_error_integral"] for s in args.timing_selection_seeds]
            if not np.isclose(np.mean(values), cell["sources"]["history"]["validation_ise"], rtol=1e-10, atol=1e-10):
                raise RuntimeError("original history native replay changed")
            observations = [observation(s) for s in cell["fit_prefix_seeds"][:4]]
            late = [(obs, s) for obs in observations for s in obs["available_steps"] if s % 50 > 0]
            if not late:
                raise RuntimeError("reference prefixes have no later optional checks")
            obs, step = late[0]
            bank = acquire(obs, [step], 1, 1)
            phase_rollout = run_episode(controllers["history"], task_spec=spec, seed=cell["fit_prefix_seeds"][0],
                args=args, time_scale=scale, rule=rules["start_clock"])[1]
        return {"stage": "precheck_passed", "start_plain_reproduced": True,
            "nonzero_phase_probe": step % 50, "phase_rule_upper_calls": phase_rollout["upper_decision_count"],
            "available_reference_phase_counts": dict(Counter(str(s % 50) for o in observations for s in o["available_steps"]))}, \
            {"paired_late_bank": bank}
    gate = json.loads((directory.parent / "precheck/public/result.json").read_text())
    if (gate["status"] != "complete" or gate["stage"] != "precheck_passed"
        or gate["run_name"] != manifest["run_name"] or gate["optimizer_seed"] != cell["optimizer_seed"]):
        raise ValueError("successful matching phase precheck required")

    checks = []
    available_counts = Counter()
    with measured_phase("reachable_training_supervision"):
        for seed in cell["fit_prefix_seeds"]:
            obs = observation(seed)
            available_counts.update(str(s % 50) for s in obs["available_steps"])
            by_step = {r["opportunity_step"]: r for r in starts if r["seed"] == seed}
            features = {r["opportunity_step"]: r for r in obs["factual"]["feature_rows"]}
            if (set(by_step) != set(obs["start_steps"]) or any(not np.array_equal(
                r["causal_features"], features[s]["causal_features"]) for s, r in by_step.items())):
                raise RuntimeError("reference bin-start states changed from the source cache")
            for step in obs["check_steps"]:
                checks.extend([by_step[step]] if step in by_step else acquire(obs, [step], 1, 4))
            (directory / "private/checks_bank.json").write_text(json.dumps(_json_ready(checks)) + "\n")
            publish({"stage": "reachable_training_supervision", "completed_paths": len(checks) // 3})
    with measured_phase("reachable_fits"):
        for name in ("checks_plain", "checks_clock"):
            rules[name], oofs[name] = fit(checks, clock=name.endswith("clock"), effective_df=manifest["effective_df"],
                threshold_quantile=manifest["threshold_quantile"])
        banks = {"start": starts, "checks": checks}
        if any(sum(len(r["A_replicas"]) for r in bank) != manifest["fitting_state_future_pairs_per_arm"] for bank in banks.values()):
            raise RuntimeError("sampling arms have different fitting budgets")
        (directory / "private/frozen_rules.json").write_text(json.dumps(_json_ready({k: v.to_dict() for k, v in rules.items()}), indent=2) + "\n")
    publish({"stage": "all_four_rules_frozen_before_holdout"})
    with measured_phase("independent_B_readout"):
        trained = {k: training_readout(banks[k.split("_")[0]], r, oofs[k]) for k, r in rules.items()}
    held = {"start": [], "checks": []}
    with measured_phase("paired_held_supervision"):
        for seed in cell["held_prefix_seeds"]:
            obs = observation(seed)
            rows = acquire(obs, sorted(set(obs["start_steps"] + obs["check_steps"])), 4, 4)
            by_step = {r["opportunity_step"]: r for r in rows}
            held["start"].extend(by_step[s] for s in obs["start_steps"])
            held["checks"].extend(by_step[s] for s in obs["check_steps"])
            (directory / "private/held_bank.json").write_text(json.dumps(_json_ready(held)) + "\n")
            publish({"stage": "paired_held_supervision", "completed_paths": len(held["start"]) // 3})
    held["later_checks"] = [r for r in held["checks"] if r["opportunity_step"] % 50 > 0]
    with measured_phase("held_error_readout"):
        held_errors = {panel: {k: held_readout(rows, rule) for k, rule in rules.items()} for panel, rows in held.items()}
    episodes, deployment = [], {name: {"queries": Counter(), "renewals": Counter()}
        for name in (*original, "L", "prior_wide_df2", *rules)}
    with measured_phase("complete_episode_evaluation"):
        for seed in cell["held_prefix_seeds"]:
            row = {"seed": seed}
            for name, rule in {**original, "L": saved_target["L"], "prior_wide_df2": saved_coverage["wide_df2"], **rules}.items():
                result = run_episode(controllers.get(name, controllers["history"]), task_spec=spec, seed=seed,
                    args=args, time_scale=scale, rule=rule)[1]
                row[name] = result["tracking_squared_error_integral"]
                for key, counts in deployment_counts(result).items():
                    deployment[name][key].update(counts)
            episodes.append(row)
    return {"stage": "phase_factorial_complete", "start_plain_reproduced": True,
        "training_readout": trained, "held_readout": held_errors,
        "training_available_phase_counts": dict(available_counts),
        "held_phase_counts": {k: phase_counts(v) for k, v in held.items()},
        "deployment_phase_counts": deployment, "episodes": episodes, "episode_panels": episode_panels(episodes),
        "termination_development_eligible": records["termination_history"]["termination_readiness"]["eligible_for_fresh_confirmation"]}, \
        {"source_cache": cell["coverage_cache_directory"], "checks_bank": checks, "held_bank": held}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    parser.add_argument("--phase", choices=("precheck", "run"), required=True)
    options = parser.parse_args(argv)
    manifest = json.loads(options.manifest.read_text())
    cell = manifest["cells"][options.cell_index]
    directory = ROOT / cell["output_directory"] / options.phase
    if directory.exists():
        raise FileExistsError("preserve measured attempts and paid samples")
    for name in ("private", "public"):
        (directory / name).mkdir(parents=True)
    ledger = ComputeLedger()
    result = {"run_name": manifest["run_name"], "task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
        "phase": options.phase, "status": "running", "fit_prefix_seeds": cell["fit_prefix_seeds"],
        "held_prefix_seeds": cell["held_prefix_seeds"], "candidate_recipe_decision": "PILOT_RECIPE_HOLD",
        "confirmation_released": False, "lifecycle_costs_complete": False, "performance_claim_supported": False}

    def publish(update):
        result.update(update, measured_cost=ledger.summary())
        (directory / "public/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
        print(f"{cell['task']} {cell['optimizer_seed']}: {result.get('stage', result['status'])} {result.get('completed_paths', '')}", flush=True)

    raw = None
    try:
        with ledger.recording("phase_support_" + options.phase), single_thread_runtime() as profile:
            if profile != manifest["source_hardware_profile"]:
                raise ValueError("runtime differs from the frozen pilot")
            result["hardware_profile"] = profile
            publish({"stage": "loading_frozen_artifacts"})
            summary, raw = run_cell(cell, manifest, options.phase, directory, publish)
        if any(ledger.summary()["counts"].get(k, 0) for k in ("controller_optimizer_updates", "termination_optimizer_updates")):
            raise RuntimeError("phase support unexpectedly trained a controller or termination head")
        result.update(summary, status="complete")
    except Exception as exc:
        result.update(status="failed", stage="failed", error_type=type(exc).__name__, error=str(exc))
    result.update(measured_cost=ledger.summary(), peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (directory / "private/result.json").write_text(json.dumps(_json_ready({**result, "raw_banks": raw}), indent=2) + "\n")
    publish({})
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
