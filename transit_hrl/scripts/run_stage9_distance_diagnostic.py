"""Frozen Stage9 versus distance threshold on new development paths."""

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

from freq_hrl.experiments.compute_budget import ComputeLedger, ControlLatency, single_thread_runtime
from freq_hrl.experiments.plan_validity_development import load_plan
from freq_hrl.experiments.plan_validity_reliability import paired_mean_interval
from freq_hrl.experiments.plan_validity_transfer import PlanValidityTaskSpec, run_episode, time_contract
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts import pointmaze_budgeted_trigger_stage9_spec as stage9
from scripts.prepare_plan_validity_boundary_conditioning_confirmation import historical_seeds
from scripts.prepare_stage9_strong_timing import MANIFEST, training_seeds
from scripts.run_plan_validity_reliability import load_controller
from scripts.run_stage9_strong_timing import ISE
from scripts.run_stage9_strong_timing_confirmation import directory, emit
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON


RUN = "pointmaze_stage9_distance_diagnostic_20261112"
PLAN = Path("protocols/plan_validity_stage9_distance_diagnostic_2026-11-12.json")
METHODS = ("history", "threshold")


def prepare():
    old = json.loads((ROOT / MANIFEST).read_text())
    folder = ROOT / "results" / old["run_name"]
    builds = json.loads((folder / "confirmation_builds.json").read_text())["roots"]
    decision = json.loads((folder / "confirmation_decision.json").read_text())
    if decision["status"] != "BUILD_GATE_HOLD" or decision["evaluation_task_ids"]:
        raise ValueError("this is a separate diagnostic after the original HOLD")
    protocols = ROOT / "protocols"
    used = historical_seeds(protocols, load_plan(protocols / "plan_validity_preregistration_2026-10-22.json"))
    for root in (*stage9.OPTIMIZER_SEEDS, *stage9.CONFIRMATION_OPTIMIZER_SEEDS, *stage9.PREFLIGHT_OPTIMIZER_SEEDS):
        roles = stage9.seed_roles(root)
        used.update(s for seeds in roles.values() for s in seeds)
        used.update(training_seeds(root, roles["train"], stage9.ITERATIONS))
    for c in old["confirmation_reservation"]:
        used.update(training_seeds(c["optimizer_seed"], c["train_seeds"], stage9.ITERATIONS))
    historical_count = len(used)
    cells, jobs = [], []
    for i, c in enumerate(old["confirmation_reservation"]):
        root = c["optimizer_seed"]
        report, = [r for r in builds if r["optimizer_seed"] == root]
        if report["status"] != "complete" or report["test_accessed"] or not all(report["budgets"][m]["admitted"] for m in METHODS):
            raise ValueError("each existing candidate and distance baseline must have closed its build")
        start = root + 7000
        seeds = list(range(start, start + 32))
        if used.intersection(seeds):
            raise ValueError("diagnostic paths overlap reserved or historical seeds")
        used.update(seeds)
        public = Path("results") / RUN / str(root) / "public"
        cells.append({"optimizer_seed": root, "diagnostic_seeds": seeds, "public": str(public)})
        job = deepcopy(old["jobs"]["audit"][0])
        cmd = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_stage9_distance_diagnostic.py", "run", "--cell", str(i)]
        job.update(description=f"Frozen Stage9 versus distance, development root {root}",
            signature=f"Freq-HRL/stage9-distance-diagnostic/{RUN}/{root}",
            cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(cmd),
            result_dir=str(ROOT / public), local_result_dir=str(ROOT / public),
            wait_for_files=[str(ROOT / p) for p in (PLAN, MANIFEST)])
        jobs.append(job)
    return {"run_name": RUN, "source_manifest": str(MANIFEST), "cells": cells, "jobs": jobs,
        "steps_per_root": 76800, "historical_seed_exclusion_count": historical_count,
        "development_signal_minimum_relative_gain": .02, "new_fits": 0,
        "scope": "post_hold_development_only_all_eight_roots_no_original_test_access",
        "original_confirmation_released": False}


def run(plan, cell, ledger):
    old = json.loads((ROOT / plan["source_manifest"]).read_text())
    root = cell["optimizer_seed"]
    home = directory(old, root)
    gate = json.loads((home / "build/public/result.json").read_text())
    if (gate["optimizer_seed"], gate["status"], gate["test_accessed"]) != (root, "complete", False):
        raise ValueError("source build not frozen")
    frozen = {m: json.loads((home / f"build/private/{m}.json").read_text()) for m in METHODS}
    if any(r["optimizer_seed"] != root or r["status"] != "validation_complete"
           or r["controller_checkpoint"] != frozen["history"]["controller_checkpoint"] for r in frozen.values()):
        raise ValueError("methods do not share one frozen controller")
    rules = {m: TimingRule(**r["selected_rule"]) for m, r in frozen.items()}
    if (rules["history"].kind, rules["history"].view, rules["threshold"].kind, rules["threshold"].feature) != (
        "legacy", "causal_validity_interactions", "threshold", "tracking_distance"):
        raise ValueError("not the frozen Stage9/distance-threshold contrast")
    controller, args = load_controller(frozen["history"])
    seeds = cell["diagnostic_seeds"]
    if set(seeds).intersection(args.train_seeds + args.selection_seeds + args.branch_fit_seeds
                               + args.timing_selection_seeds + args.trigger_eval_seeds):
        raise ValueError("diagnostic paths overlap source roles")
    spec = PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id)
    rows, costs = {}, {}
    for method in METHODS:
        latency = ControlLatency()
        with ledger.recording(method):
            rows[method] = [run_episode(controller, task_spec=spec, seed=s, args=args, time_scale=time_contract(args),
                rule=rules[method], latency_recorder=latency, risk_spec={"distance": 1., "duration_steps": 25})[1] for s in seeds]
        costs[method] = latency.summary(episodes=len(seeds), deadline_seconds=.01)
    counts = ledger.summary()["counts"]
    if counts["environment_steps"] != plan["steps_per_root"] or counts["planner_calls"] != len(seeds) * 48:
        raise ValueError("diagnostic did not complete its fixed steps and calls")
    return {"paths": [{"seed": seed, **{m: rows[m][i][ISE] for m in METHODS},
        "schedule_differs": rows["history"][i]["decision_steps"] != rows["threshold"][i]["decision_steps"]}
        for i, seed in enumerate(seeds)],
        "sustained_failure_counts": {m: sum(r["reliability"]["sustained_failure"] for r in rows[m]) for m in METHODS},
        "latency": costs}


def summarize(plan, records):
    if len(records) != len(plan["cells"]) or any(
        r["status"] != "complete" or
        (r["optimizer_seed"], [p["seed"] for p in r["paths"]], r["measured_cost"]["counts"]["environment_steps"]) !=
        (c["optimizer_seed"], c["diagnostic_seeds"], plan["steps_per_root"])
        for c, r in zip(plan["cells"], records)):
        raise ValueError("the entire ordered development cohort is required")
    h, b = (np.array([[p[m] for p in r["paths"]] for r in records]) for m in METHODS)
    delta = b - h
    relative = float(delta.mean() / b.mean())
    halves = [float(d.mean() / t.mean()) for d, t in zip(np.split(delta, 2, axis=1), np.split(b, 2, axis=1))]
    ci = paired_mean_interval(delta.mean(axis=1))
    return {"roots": [{"optimizer_seed": r["optimizer_seed"], "mean_ise": dict(zip(METHODS, [float(h[i].mean()), float(b[i].mean())])),
        "relative_gain": float(delta[i].mean() / b[i].mean()), "differing_schedules": sum(p["schedule_differs"] for p in r["paths"]),
        "sustained_failure_counts": r["sustained_failure_counts"], "latency": r["latency"],
        "measured_cost": r["measured_cost"]} for i, r in enumerate(records)],
        "paired_ise_gain": ci, "relative_gain": relative, "relative_gain_halves": halves,
        "development_signal": relative >= plan["development_signal_minimum_relative_gain"] and min(halves) > 0
            and ci["interval"][0] > 0, "original_confirmation_released": False, "scope": plan["scope"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "readout"))
    parser.add_argument("--cell", type=int)
    opts = parser.parse_args()
    if opts.phase == "prepare":
        emit(ROOT / PLAN, prepare())
        return 0
    plan = json.loads((ROOT / PLAN).read_text())
    if opts.phase == "readout":
        records = [json.loads((ROOT / c["public"] / "result.json").read_text()) for c in plan["cells"]]
        print(json.dumps(summarize(plan, records), separators=(",", ":")))
        return 0
    cell = plan["cells"][opts.cell]
    output = ROOT / cell["public"] / "result.json"
    output.parent.mkdir(parents=True, exist_ok=False)
    ledger = ComputeLedger()
    old = json.loads((ROOT / plan["source_manifest"]).read_text())
    try:
        with single_thread_runtime() as profile:
            if profile != old["hardware_profile"]:
                raise ValueError("runtime differs from source builds")
            result = {**run(plan, cell, ledger), "status": "complete"}
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    emit(output, {**result, "optimizer_seed": cell["optimizer_seed"], "measured_cost": ledger.summary()})
    print(cell["optimizer_seed"], result["status"], result.get("error", ""), flush=True)
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
