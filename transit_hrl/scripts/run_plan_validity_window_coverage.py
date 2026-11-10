"""Native W coverage/complexity prechecks and frozen development readout."""

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
from freq_hrl.experiments.plan_validity_reliability import paired_mean_interval
from freq_hrl.experiments.plan_validity_target_credit import predict
from freq_hrl.experiments.plan_validity_transfer import PlanValidityTaskSpec, run_episode, time_contract
from freq_hrl.experiments.plan_validity_window_coverage import (
    ARMS, cached_rows, collect_window_bank, fit_window, held_readout, training_readout,
)
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts.run_plan_validity_target_credit import load_sources


def episode_panels(rows):
    panels = {}
    for panel, selected in (("all", rows), ("first_half", rows[:4]), ("second_half", rows[4:])):
        methods = [k for k in rows[0] if k != "seed"]
        contrasts = {"coverage_df2": [r["wide_df2"] - r["dense_df2"] for r in selected],
            "coverage_df8": [r["wide_df8"] - r["dense_df8"] for r in selected],
            "complexity_dense": [r["dense_df2"] - r["dense_df8"] for r in selected],
            "complexity_wide": [r["wide_df2"] - r["wide_df8"] for r in selected],
            "interaction": [(r["wide_df2"] - r["dense_df2"]) - (r["wide_df8"] - r["dense_df8"]) for r in selected]}
        panels[panel] = {"paths": len(selected),
            "mean_ise": {m: float(np.mean([r[m] for r in selected])) for m in methods},
            "factorial_contrasts": {k: paired_mean_interval(v) for k, v in contrasts.items()},
            "against_controls": {m: {c: paired_mean_interval([r[m] - r[c] for r in selected])
                for c in ("history", "L", "threshold", "termination_history")} for m in ARMS}}
    return panels


def run_cell(cell, manifest, phase, directory, publish):
    with measured_phase("load_frozen_artifacts"):
        records, controllers, original, args = load_sources(cell)
        source_public = json.loads((ROOT / cell["source_public_result"]).read_text())
        source = json.loads((ROOT / cell["source_cache_directory"] / "result.json").read_text())
        saved = {k: TimingRule(**v) for k, v in json.loads(
            (ROOT / cell["source_cache_directory"] / "frozen_rules.json").read_text()).items()}
        if (source_public["status"] != "complete" or source["status"] != "complete"
            or source_public["optimizer_seed"] != cell["optimizer_seed"]
            or source_public["training_prefix_seeds"] != cell["training_prefix_seeds"]):
            raise ValueError("source cache differs from the completed training cohort")
        old = source["raw_banks"]["training_bank"]
        if (len(old) != 24 or set(r["seed"] for r in old) != set(cell["training_prefix_seeds"])
            or any(len(r["conditional_replicas"]) != 8 for r in old)):
            raise ValueError("the cached 8-prefix/8-future bank changed")
        dense, initial_wide = cached_rows(old, 4), cached_rows(old, 1)
    scale = time_contract(args)
    spec = PlanValidityTaskSpec(cell["task"], _task_options(args), args.env_id)
    with measured_phase("dense_reproduction"):
        dense8, dense_oof = fit_window(dense, effective_df=8, threshold_quantile=manifest["threshold_quantile"])
        if (not np.allclose(predict(dense8, dense), predict(saved["W"], dense), rtol=1e-12, atol=1e-12)
            or not np.isclose(dense8.threshold, saved["W"].threshold, rtol=1e-12, atol=1e-12)):
            raise RuntimeError("dense df8 does not reproduce the frozen W head")
    collect = lambda seeds, probes, a, b: collect_window_bank(controllers["history"], original["history"],
        spec=spec, args=args, seeds=seeds, probes_per_path=probes, fitting_futures=a, evaluation_futures=b)
    if phase == "precheck":
        with measured_phase("native_precheck"):
            ise = np.mean([run_episode(controllers["history"], task_spec=spec, seed=seed, args=args,
                time_scale=scale, rule=original["history"])[1]["tracking_squared_error_integral"]
                for seed in args.timing_selection_seeds])
            if not np.isclose(ise, cell["sources"]["history"]["validation_ise"], rtol=1e-10, atol=1e-10):
                raise RuntimeError("original native history replay changed")
            fit_window(dense, effective_df=2, threshold_quantile=manifest["threshold_quantile"])
            bank = collect(cell["new_training_prefix_seeds"][:1], 1, 1, 4)
        return {"stage": "precheck_passed", "dense_df8_reproduced": True,
            "native_history_reproduced": True, "paired_prefix_and_factual_replay": True,
            "new_training_probe_states": len(bank)}, {"precheck_bank": bank}
    gate = json.loads((directory.parent / "precheck/public/result.json").read_text())
    if (gate["status"] != "complete" or gate["stage"] != "precheck_passed"
        or gate["run_name"] != manifest["run_name"] or gate["optimizer_seed"] != cell["optimizer_seed"]):
        raise ValueError("successful matching native precheck required")

    def acquire(seeds, *, phase_name, a, b):
        rows = []
        with measured_phase(phase_name):
            for seed in seeds:
                rows.extend(collect([seed], manifest["probes_per_path"], a, b))
                (directory / "private" / (phase_name + ".json")).write_text(json.dumps(_json_ready(rows)) + "\n")
                publish({"stage": phase_name, "completed_paths": len(rows) // manifest["probes_per_path"]})
        return rows

    new = acquire(cell["new_training_prefix_seeds"], phase_name="new_training_bank", a=1, b=4)
    banks = {"dense": dense, "wide": initial_wide + new}
    rules, oof = {"dense_df8": dense8}, {"dense_df8": dense_oof}
    with measured_phase("factorial_fit"):
        for name in ARMS:
            coverage, df = name.split("_df")
            if name not in rules:
                rules[name], oof[name] = fit_window(banks[coverage], effective_df=int(df),
                                                  threshold_quantile=manifest["threshold_quantile"])
        if any(sum(len(r["A_replicas"]) for r in bank) != manifest["fitting_state_future_pairs_per_arm"]
               for bank in banks.values()):
            raise RuntimeError("factorial fitting-future budgets differ")
        (directory / "private/frozen_rules.json").write_text(
            json.dumps(_json_ready({k: v.to_dict() for k, v in rules.items()}), indent=2) + "\n")
    publish({"stage": "all_four_rules_frozen_before_holdout"})
    with measured_phase("independent_B_readout"):
        fitted = {k: training_readout(banks[k.split("_df")[0]], rule, oof[k]) for k, rule in rules.items()}
    held = acquire(cell["held_prefix_seeds"], phase_name="held_bank", a=4, b=4)
    with measured_phase("held_error_readout"):
        held_errors = {k: held_readout(held, rule) for k, rule in rules.items()}
    episodes = []
    with measured_phase("complete_episode_evaluation"):
        for seed in cell["held_prefix_seeds"]:
            row = {"seed": seed}
            for name, rule in {**original, "L": saved["L"], **rules}.items():
                result = run_episode(controllers.get(name, controllers["history"]), task_spec=spec,
                    seed=seed, args=args, time_scale=scale, rule=rule)[1]
                row[name] = result["tracking_squared_error_integral"]
            episodes.append(row)
    return {"stage": "factorial_readout_complete", "dense_df8_reproduced": True,
        "training_readout": fitted, "held_readout": held_errors,
        "episodes": episodes, "episode_panels": episode_panels(episodes),
        "termination_development_eligible": records["termination_history"]["termination_readiness"]["eligible_for_fresh_confirmation"]}, \
        {"source_cache": cell["source_cache_directory"], "new_training_bank": new, "held_bank": held}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    parser.add_argument("--phase", choices=("precheck", "run"), required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    cell = manifest["cells"][args.cell_index]
    directory = ROOT / cell["output_directory"] / args.phase
    if directory.exists():
        raise FileExistsError("preserve the measured attempt and its paid samples")
    for name in ("private", "public"):
        (directory / name).mkdir(parents=True)
    ledger = ComputeLedger()
    result = {"run_name": manifest["run_name"], "task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
        "phase": args.phase, "status": "running", "new_training_prefix_seeds": cell["new_training_prefix_seeds"],
        "held_prefix_seeds": cell["held_prefix_seeds"], "candidate_recipe_decision": "PILOT_RECIPE_HOLD",
        "confirmation_released": False, "lifecycle_costs_complete": False, "performance_claim_supported": False}

    def publish(update):
        result.update(update, measured_cost=ledger.summary())
        (directory / "public/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
        print(f"{cell['task']} {cell['optimizer_seed']}: {result.get('stage', result['status'])} {result.get('completed_paths', '')}", flush=True)

    raw = None
    try:
        with ledger.recording("window_coverage_" + args.phase), single_thread_runtime() as profile:
            if profile != manifest["source_hardware_profile"]:
                raise ValueError("runtime differs from the source pilot")
            result["hardware_profile"] = profile
            publish({"stage": "loading_frozen_artifacts"})
            summary, raw = run_cell(cell, manifest, args.phase, directory, publish)
        if any(ledger.summary()["counts"].get(k, 0) for k in ("controller_optimizer_updates", "termination_optimizer_updates")):
            raise RuntimeError("coverage diagnostic unexpectedly trained a controller or termination head")
        result.update(summary, status="complete")
    except Exception as exc:
        result.update(status="failed", stage="failed", error_type=type(exc).__name__, error=str(exc))
    result.update(measured_cost=ledger.summary(), peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (directory / "private/result.json").write_text(json.dumps(_json_ready({**result, "raw_banks": raw}), indent=2) + "\n")
    publish({})
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
