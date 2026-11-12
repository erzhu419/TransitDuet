"""Fit matched timing heads on resident labels, then evaluate a frozen cohort."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from freq_hrl.experiments.compute_budget import ComputeLedger, ControlLatency, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_reliability import paired_mean_interval
from freq_hrl.experiments.plan_validity_transfer import PlanValidityTaskSpec, run_episode, time_contract
from freq_hrl.experiments.pointmaze_compact_plan_validity import _base_matrix, _quadratic_features, _select_named_features
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule, fit_grouped_ridge
from scripts.prepare_stage9_strong_timing import MANIFEST
from scripts.run_plan_validity_reliability import load_controller
from scripts.run_stage9_distance_diagnostic import historical_path_seeds
from scripts.run_stage9_strong_timing import panels
from scripts.run_stage9_strong_timing_confirmation import directory, emit
from scripts.run_stage9_timing_portfolio import RUN as PORTFOLIO_RUN
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON


RUN = "pointmaze_stage9_scale_ablation_20261112"
PLAN = Path("protocols/plan_validity_stage9_scale_ablation_2026-11-12.json")
RUNNER = "scripts/run_stage9_scale_ablation.py"
COMMON = ("target_error_0", "target_error_1", "waypoint_error_0", "waypoint_error_1",
          "tracking_distance", "waypoint_distance", "plan_age_fraction", "force_current_norm")
VIEWS = {
    "current": (*COMMON, "target_current_0", "target_current_1", "force_current_0", "force_current_1"),
    "single_scale": (*COMMON, "target_velocity_010_0", "target_velocity_010_1", "force_rms_010_0", "force_rms_010_1"),
    "mixed_scale": (*COMMON, "target_velocity_050_0", "target_velocity_050_1", "force_rms_010_0", "force_rms_010_1"),
}
METHODS = (*VIEWS, "stage9")
CONTRASTS = (("mixed_scale", "single_scale"), ("mixed_scale", "current"), ("single_scale", "current"))


def prepare(old):
    if not json.loads((ROOT / "results" / PORTFOLIO_RUN / "summary.json").read_text())["joint_gate_passed"]:
        raise ValueError("the timing utility study has not passed")
    used = historical_path_seeds(old)
    cells = []
    for c in old["confirmation_reservation"]:
        root = c["optimizer_seed"]
        seeds = list(range(root + 9000, root + 9064))
        if used.intersection(seeds):
            raise ValueError("new path role overlaps an existing role")
        used.update(seeds)
        cells.append({"optimizer_seed": root, "branch_fit_seeds": c["branch_fit_seeds"],
                      "evaluation_seeds": seeds, "public": f"results/{RUN}/{root}/public"})

    def job(phase, public, label, extra=()):
        spec = deepcopy(old["jobs"]["audit"][0])
        cmd = [DEFAULT_LINUX_PYTHON, "-u", RUNNER, phase, *extra]
        spec.update(description=f"Stage9 scale ablation {label}", signature=f"Freq-HRL/stage9-scale-ablation/{RUN}/{label}",
            cmd=spec["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(cmd),
            result_dir=str(ROOT / public), local_result_dir=str(ROOT / public),
            wait_for_files=[str(ROOT / p) for p in (PLAN, MANIFEST)])
        return spec

    return {"run_name": RUN, "cells": cells, "views": {k: list(v) for k, v in VIEWS.items()},
        "source_manifest": str(MANIFEST), "ridge_df": 8., "threshold_quantile": .75, "minimum_relative_gain": .02,
        "steps_per_root": len(METHODS) * 64 * 1200, "calls_per_episode": 24, "ridge_fits": 216,
        "fit_public": f"results/{RUN}/fit/public",
        "jobs": {"fit": [job("fit", f"results/{RUN}/fit/public", "fit")],
                 "evaluate": [job("evaluate", c["public"], str(c["optimizer_seed"]), ("--cell", str(i)))
                              for i, c in enumerate(cells)]},
        "evaluation_gate": "all_24_heads_frozen_before_any_new_path",
        "analysis": "paired_optimizer_root_nominal_t95_lower_gt_0_and_pooled_2pct_and_both_halves_positive",
        "scope": "timing_head_feature_ablation_on_shared_frozen_PointMaze_controllers_not_LF_HF_routing",
        "original_confirmation_released": False}


def load_source(old, root):
    home = directory(old, root)
    report = json.loads((home / "build/private/history.json").read_text())
    if ((report["optimizer_seed"], report["method"], report["status"]) != (root, "history", "validation_complete")
        or (home / "evaluate").exists() or not Path(report["controller_checkpoint"]).is_file()):
        raise ValueError("original build or HOLD boundary differs")
    return report


def fit_heads(rows, plan):
    schema, raw = _base_matrix(rows)
    y = np.array([r["renew_ise_advantage"] for r in rows])
    groups = np.array([r["seed"] for r in rows])
    result = {}
    for name, inputs in plan["views"].items():
        x, names = _quadratic_features(_select_named_features(schema, raw, inputs), inputs)
        model, cut = fit_grouped_ridge(x, names, y, groups, effective_df=plan["ridge_df"],
                                      threshold_quantile=plan["threshold_quantile"])
        result[name] = TimingRule(name, "named_ridge", threshold=cut,
                                  model={**model, "input_names": inputs}).to_dict()
    return result


def fit(plan, old, ledger):
    frozen, public = {}, []
    for cell in plan["cells"]:
        root = cell["optimizer_seed"]
        source = load_source(old, root)
        rows = json.loads(Path(source["fit_cache_file"]).read_text())["branch_labels"]
        if (len(rows) != 192 or {r["seed"] for r in rows} != set(cell["branch_fit_seeds"])
            or len({(r["seed"], r["opportunity_step"]) for r in rows}) != len(rows)):
            raise ValueError("branch label cache roster differs")
        with measured_phase(f"root_{root}"):
            heads = fit_heads(rows, plan)
        frozen[str(root)] = heads
        public.append({"optimizer_seed": root, "labels": len(rows), "heads": {
            name: {"coefficient_count": rule["model"]["coefficient_count"],
                   "effective_df": rule["model"]["effective_df_slopes"],
                   "OOF_MSE": rule["model"]["oof_mse"], "threshold": rule["threshold"]}
            for name, rule in heads.items()}})
    count = ledger.summary()["counts"]
    if count.get("ridge_fit_calls") != plan["ridge_fits"] or count.get("environment_steps", 0):
        raise RuntimeError("fit accounting differs from the fixed no-simulation recipe")
    path = ROOT / "results" / RUN / "fit/private"
    path.mkdir(exist_ok=False)
    emit(path / "rules.json", frozen)
    return {"roots": public, "heads_frozen": 24, "evaluation_paths_accessed": False}


def evaluate(plan, old, cell, ledger):
    fit_report = json.loads((ROOT / plan["fit_public"] / "result.json").read_text())
    bundle = json.loads((ROOT / "results" / RUN / "fit/private/rules.json").read_text())
    if (fit_report["status"] != "complete" or fit_report["heads_frozen"] != 24
        or fit_report["evaluation_paths_accessed"] or set(bundle) != {str(c["optimizer_seed"]) for c in plan["cells"]}
        or any(set(heads) != set(VIEWS) for heads in bundle.values())):
        raise ValueError("global fit freeze incomplete")
    root = cell["optimizer_seed"]
    source = load_source(old, root)
    controller, args = load_controller(source)
    seeds = cell["evaluation_seeds"]
    if set(seeds).intersection(args.train_seeds + args.selection_seeds + args.branch_fit_seeds
                               + args.timing_selection_seeds + args.trigger_eval_seeds):
        raise ValueError("evaluation overlaps an original role")
    spec, scale = PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id), time_contract(args)
    rows, costs = {}, {}
    for name, saved in {**bundle[str(root)], "stage9": source["selected_rule"]}.items():
        rule, latency = TimingRule(**saved), ControlLatency()
        with measured_phase(name):
            rows[name] = [run_episode(controller, task_spec=spec, seed=s, args=args,
                time_scale=scale, rule=rule, latency_recorder=latency)[1] for s in seeds]
        costs[name] = latency.summary(episodes=len(seeds), deadline_seconds=.01)
    count = ledger.summary()["counts"]
    if (count["environment_steps"] != plan["steps_per_root"] or count["planner_calls"] != len(METHODS) * len(seeds) * 24
        or count.get("ridge_fit_calls", 0) or count.get("controller_optimizer_updates", 0)):
        raise ValueError("evaluation did not close its native/call/no-fit budget")
    return {"evaluation_seeds": seeds, "panels": panels(rows, methods=METHODS), "latency": costs}


def readout(plan, records):
    if len(records) != len(plan["cells"]) or any(r["status"] != "complete" or
        (r["optimizer_seed"], r["evaluation_seeds"], r["measured_cost"]["counts"]["environment_steps"]) !=
        (c["optimizer_seed"], c["evaluation_seeds"], plan["steps_per_root"]) for c, r in zip(plan["cells"], records)):
        raise ValueError("all registered paired roots are required")

    def means(panel, method):
        return np.array([r["panels"][panel][method]["mean_ise"] for r in records])

    contrasts = {}
    for candidate, ref in CONTRASTS:
        a, b = means("all", candidate), means("all", ref)
        gain = float(1 - a.mean() / b.mean())
        ci = paired_mean_interval(b - a)
        halves = [float(1 - means(p, candidate).mean() / means(p, ref).mean()) for p in ("first", "last")]
        contrasts[f"{candidate}_vs_{ref}"] = {"gain": gain, "absolute_gain": ci, "half_gains": halves,
            "root_gains": ((b - a) / b).tolist(), "passed": ci["interval"][0] > 0 and
                gain >= plan["minimum_relative_gain"] and min(halves) > 0}
    return {"roots": records, "mean_ise": {m: float(means("all", m).mean()) for m in METHODS},
        "comparisons": contrasts, "mixed_scale_supported": all(contrasts[f"mixed_scale_vs_{r}"]["passed"]
            for r in ("current", "single_scale")),
        "history_supported": contrasts["single_scale_vs_current"]["passed"],
        "original_confirmation_released": False, "scope": plan["scope"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "fit", "evaluate", "readout"))
    parser.add_argument("--cell", type=int)
    args = parser.parse_args()
    old = json.loads((ROOT / MANIFEST).read_text())
    if args.phase == "prepare":
        emit(ROOT / PLAN, prepare(old))
        return 0
    plan = json.loads((ROOT / PLAN).read_text())
    if args.phase == "readout":
        reports = [json.loads((ROOT / c["public"] / "result.json").read_text()) for c in plan["cells"]]
        print(json.dumps(readout(plan, reports), separators=(",", ":")))
        return 0
    fitting = args.phase == "fit"
    cell = None if fitting else plan["cells"][args.cell]
    path = ROOT / (plan["fit_public"] if fitting else cell["public"])
    path.mkdir(parents=True, exist_ok=False)
    ledger = ComputeLedger()
    try:
        with single_thread_runtime() as profile, ledger.recording(args.phase):
            if profile != old["hardware_profile"]:
                raise ValueError("runtime differs from original build")
            result = fit(plan, old, ledger) if fitting else evaluate(plan, old, cell, ledger)
        result.update(status="complete")
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    emit(path / "result.json", {**result, "optimizer_seed": None if fitting else cell["optimizer_seed"],
                               "measured_cost": ledger.summary()})
    print(args.phase, result["status"], result.get("error", ""), flush=True)
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
