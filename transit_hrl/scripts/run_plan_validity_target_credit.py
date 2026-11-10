"""Matched target repair: native precheck, training-only fit, then fresh holdout."""

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
from freq_hrl.experiments.plan_validity_target_credit import (
    collect_bank, fit_targets, fitting_readout, held_readout, label_uncertainty,
)
from freq_hrl.experiments.plan_validity_transfer import PlanValidityTaskSpec, run_episode, time_contract
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts.run_plan_validity_reliability import load_controller, load_record


def load_sources(cell):
    records, controllers, rules, arguments = {}, {}, {}, {}
    for method, source in cell["sources"].items():
        record = load_record(Path(source["report"]), source["run_id"])
        if (record["configuration"] != {**cell["expected_configuration"], "method": method}
            or {k: v for k, v in record["selected_rule"].items() if k != "model"} != source["selected_rule_summary"]):
            raise ValueError("frozen source configuration or rule changed")
        records[method] = record
        controllers[method], arguments[method] = load_controller(record)
        rules[method] = TimingRule(**record["selected_rule"])
    return records, controllers, rules, arguments["history"]


def run_cell(cell, manifest, phase, directory, publish):
    with measured_phase("load_artifacts"):
        records, controllers, original, args = load_sources(cell)
    scale = time_contract(args)
    spec = PlanValidityTaskSpec(cell["task"], _task_options(args), args.env_id)
    replay = {}
    with measured_phase("native_reproduction"):
        for method in original:
            scores = [run_episode(controllers[method], task_spec=spec, seed=seed, args=args,
                time_scale=scale, rule=original[method])[1]["tracking_squared_error_integral"]
                for seed in args.timing_selection_seeds]
            replay[method] = float(np.mean(scores))
            if not np.isclose(replay[method], cell["sources"][method]["validation_ise"], rtol=1e-10, atol=1e-10):
                raise RuntimeError("native selection replay differs from frozen source")
    publish({"stage": "native_reproduction_passed", "selection_reproduction": replay})
    collect = lambda seeds, probes, replicas: collect_bank(controllers["history"], original["history"],
        spec=spec, args=args, seeds=seeds, probes_per_path=probes, replicas=replicas)
    if phase == "precheck":
        with measured_phase("native_paired_precheck"):
            bank = collect(cell["training_prefix_seeds"][:1], 1, 2)
        return {"stage": "precheck_passed", "paired_states": len(bank),
                "prefix_and_factual_replay_passed": True}, {"precheck_bank": bank}
    gate = json.loads((directory.parent / "precheck/public/result.json").read_text())
    if (gate["status"] != "complete" or gate["run_name"] != manifest["run_name"]
        or gate["optimizer_seed"] != cell["optimizer_seed"] or gate["stage"] != "precheck_passed"):
        raise ValueError("a successful matching native precheck is required")
    with measured_phase("shared_training_supervision"):
        training = collect(cell["training_prefix_seeds"], manifest["probes_per_path"], manifest["conditional_replicates"])
    # This file closes the fit before any independent-future or fresh-prefix readout.
    with measured_phase("matched_target_fit"):
        rules, oof = fit_targets(training, effective_df=manifest["effective_df"],
                                 threshold_quantile=manifest["threshold_quantile"])
        frozen = directory / "private/frozen_rules.json"
        frozen.write_text(json.dumps(_json_ready({k: v.to_dict() for k, v in rules.items()}), indent=2) + "\n")
    publish({"stage": "rules_frozen_before_holdout", "training_states": len(training)})
    with measured_phase("independent_future_readout"):
        fit_report = fitting_readout(training, rules, oof)
    with measured_phase("fresh_prefix_supervision"):
        held = collect(cell["held_prefix_seeds"], manifest["probes_per_path"], manifest["conditional_replicates"])
        held_report = held_readout(held, rules)
    episode_rows = []
    with measured_phase("complete_episode_evaluation"):
        for seed in cell["held_prefix_seeds"]:
            row = {"seed": seed}
            for name, rule in {**original, **rules}.items():
                controller = controllers.get(name, controllers["history"])
                outcome = run_episode(controller, task_spec=spec, seed=seed, args=args,
                                      time_scale=scale, rule=rule)[1]
                row[name] = outcome["tracking_squared_error_integral"]
            episode_rows.append(row)
    panels = {}
    for name, indices in (("all", range(8)), ("first_half", range(4)), ("second_half", range(4, 8))):
        rows = [episode_rows[i] for i in indices]
        panels[name] = {"paths": len(rows),
            "mean_ise": {m: float(np.mean([r[m] for r in rows])) for m in (*original, *rules)},
            "paired_difference": {m: {c: paired_mean_interval([r[m] - r[c] for r in rows])
                for c in ("L", "history", "threshold", "termination_history")}
                for m in rules}}
    return {"stage": "repair_readout_complete", "selection_reproduction": replay,
        "fit_readout": fit_report, "held_readout": held_report,
        "label_uncertainty": {"training": label_uncertainty(training), "held": label_uncertainty(held)},
        "episode_panels": panels, "episodes": episode_rows,
        "termination_development_eligible": records["termination_history"]["termination_readiness"]["eligible_for_fresh_confirmation"]}, \
        {"training_bank": training, "held_bank": held}


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
        raise FileExistsError("preserve the measured attempt")
    for name in ("private", "public"):
        (directory / name).mkdir(parents=True)
    ledger = ComputeLedger()
    result = {"status": "running", "run_name": manifest["run_name"], "task": cell["task"],
        "optimizer_seed": cell["optimizer_seed"], "phase": options.phase,
        "training_prefix_seeds": cell["training_prefix_seeds"], "held_prefix_seeds": cell["held_prefix_seeds"],
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "lifecycle_costs_complete": False, "performance_claim_supported": False}

    def publish(update):
        result.update(update, measured_cost=ledger.summary())
        (directory / "public/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
        print(f"{cell['task']} {cell['optimizer_seed']}: {result.get('stage', result['status'])}", flush=True)

    raw = None
    try:
        with ledger.recording("target_credit_" + options.phase), single_thread_runtime() as hardware:
            if hardware != manifest["source_hardware_profile"]:
                raise ValueError("runtime differs from the frozen pilot")
            result["hardware_profile"] = hardware
            publish({"stage": "loading_frozen_artifacts"})
            summary, raw = run_cell(cell, manifest, options.phase, directory, publish)
        if any(ledger.summary()["counts"].get(k, 0) for k in ("controller_optimizer_updates", "termination_optimizer_updates")):
            raise RuntimeError("matched-target diagnostic unexpectedly trained a controller or termination head")
        result.update(summary, status="complete")
    except Exception as exc:
        result.update(status="failed", stage="failed", error_type=type(exc).__name__, error=str(exc))
    result.update(measured_cost=ledger.summary(),
        peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (directory / "private/result.json").write_text(json.dumps(_json_ready({**result, "raw_banks": raw}), indent=2) + "\n")
    publish({})
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
