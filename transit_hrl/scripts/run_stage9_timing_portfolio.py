"""Freeze, run and read out the Stage9 timing portfolio comparison."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from freq_hrl.experiments.compute_budget import ComputeLedger, ControlLatency, single_thread_runtime
from freq_hrl.experiments.plan_validity_reliability import paired_mean_interval
from freq_hrl.experiments.plan_validity_transfer import PlanValidityTaskSpec, run_episode, time_contract
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts.prepare_stage9_strong_timing import MANIFEST
from scripts.run_plan_validity_reliability import load_controller
from scripts.run_stage9_distance_diagnostic import RUN as DISTANCE_RUN, historical_path_seeds
from scripts.run_stage9_strong_timing import panels
from scripts.run_stage9_strong_timing_confirmation import directory, emit
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON


RUN = "pointmaze_stage9_timing_portfolio_20261112"
PLAN = Path("protocols/plan_validity_stage9_timing_portfolio_2026-11-12.json")
RUNNER = "scripts/run_stage9_timing_portfolio.py"
METHODS = ("history", "threshold", "portfolio")
SOURCES = ("history", "threshold", "termination_history", "best_clock")


def winner(ise):
    if not all(math.isfinite(ise[m]) for m in SOURCES[-2:]):
        raise ValueError("selection ISE is not finite")
    return "termination_history" if ise["termination_history"] < ise["best_clock"] else "best_clock"


def cost_pair(cost):
    return cost["cpu_seconds"], cost["counts"].get("environment_steps", 0)


def within_budget(charge, latency, budget):
    cpu = charge["offline_cpu_seconds"] + budget["deployments"] * latency["cpu_seconds_per_episode"]
    return {**charge, "lifecycle_cpu_seconds": cpu, "p99_seconds": latency["wall_p99_seconds"],
            "admitted": cpu <= budget["cpu_seconds"] and charge["offline_steps"] <= budget["offline_steps"]
                and latency["wall_p99_seconds"] <= budget["p99_seconds"]}


def prepare(old, records, dev, distance, metadata, used):
    if len(records) != 8 or len(metadata) != 8 or not distance["development_signal"]:
        raise ValueError("the paid eight-build cohort and completed distance study are required")
    cells, jobs = [], []
    distance_costs = [cost_pair(r["measured_cost"]) for r in distance["roots"]]
    clock_dev_costs = [cost_pair(r["measured_cost"]["phases"]["preflight/frozen_selection_reproduction"])
                       for r in dev["preflight"]]
    # The termination ledger already includes its own clock search. The separately
    # paid clock run and old clock preflight are additional, not free duplicates.
    common_cpu, common_steps = np.sum(distance_costs, axis=0) / len(records)
    clock_cpu, clock_steps = np.sum(clock_dev_costs, axis=0) / len(records)
    for i, (old_cell, build, meta) in enumerate(zip(old["confirmation_reservation"], records, metadata)):
        root = old_cell["optimizer_seed"]
        if (build["status"], build["optimizer_seed"], build["test_accessed"], meta["optimizer_seed"], meta["original_test_directory_exists"]) != (
            "complete", root, False, root, False) or not all(b["admitted"] for b in build["budgets"].values()):
            raise ValueError("all four source budgets must close on each unchanged root")
        seeds = list(range(root + 8000, root + 8064))
        if set(seeds) & used:
            raise ValueError("portfolio paths overlap a historical or reserved role")
        used.update(seeds)
        choice = winner(meta["selection_ise"])
        underlying = {"history": "history", "threshold": "threshold", "portfolio": choice}
        clock_fit_cpu, clock_fit_steps = cost_pair(build["budgets"]["best_clock"]["fit_cost"])
        admission = {}
        for m in METHODS:
            source = build["budgets"]["termination_history" if m == "portfolio" else m]
            charge = {"offline_cpu_seconds": float(source["charged_cpu_seconds"] + common_cpu),
                      "offline_steps": int(source["charged_steps"] + common_steps)}
            if m == "portfolio":
                charge["offline_cpu_seconds"] += float(clock_fit_cpu + clock_cpu)
                charge["offline_steps"] += int(clock_fit_steps + clock_steps)
            admission[m] = within_budget(charge, build["budgets"][underlying[m]]["validation_latency"], old["budget"])
        if not all(v["admitted"] for v in admission.values()):
            raise ValueError(f"portfolio study budget not met on root {root}")
        public = Path("results") / RUN / str(root) / "public"
        cells.append({**meta, "portfolio_choice": choice, "evaluation_seeds": seeds, "public": str(public),
                      "frozen_budget_admission": admission})
        job = deepcopy(old["jobs"]["audit"][0])
        cmd = [DEFAULT_LINUX_PYTHON, "-u", RUNNER, "run", "--cell", str(i)]
        job.update(description=f"Stage9 frozen timing portfolio root {root}",
            signature=f"Freq-HRL/stage9-timing-portfolio/{RUN}/{root}",
            cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(cmd),
            result_dir=str(ROOT / public), local_result_dir=str(ROOT / public),
            wait_for_files=[str(ROOT / p) for p in (PLAN, MANIFEST)])
        jobs.append(job)
    return {"run_name": RUN, "source_manifest": str(MANIFEST), "cells": cells, "jobs": jobs,
        "budget": old["budget"], "steps_per_root": 64 * 3 * 1200, "calls_per_episode": 24,
        "minimum_ise_gain": .02, "new_fits": 0, "original_confirmation_released": False,
        "selection": "lowest_old_timing_selection_ISE_termination_vs_best_clock_ties_clock",
        "accounting": "retain_source_costs_share_distance_diagnostic_over_8_add_actual_standalone_clock_searches_to_termination",
        "scope": "fresh_paths_existing_eight_built_PointMaze_roots_not_new_recipe_or_learned_only_confirmation",
        "primary": "both_threshold_and_portfolio_root_t95_lower_above_zero_2pct_pooled_both_halves_positive_all_budgets"}


def sources(old, root):
    home = directory(old, root)
    records = {m: json.loads((home / f"build/private/{m}.json").read_text()) for m in SOURCES}
    if any((r["optimizer_seed"], r["method"], r["status"], r["controller_checkpoint"]) !=
           (root, m, "validation_complete", records["history"]["controller_checkpoint"]) for m, r in records.items()):
        raise ValueError("source rules must share one frozen controller and root")
    return records, {"optimizer_seed": root,
        "selection_ise": {m: r["validation_ise"] for m, r in records.items()},
        "rule_names": {m: r["selected_rule"]["name"] for m, r in records.items()},
        "original_test_directory_exists": (home / "evaluate").exists()}


def run(plan, cell, old, ledger):
    root = cell["optimizer_seed"]
    frozen, meta = sources(old, root)
    if any(cell[k] != v for k, v in meta.items()) or winner(meta["selection_ise"]) != cell["portfolio_choice"]:
        raise ValueError("saved portfolio selection changed after the global freeze")
    chosen = cell["portfolio_choice"]
    controller, args = load_controller(frozen["history"])
    seeds = cell["evaluation_seeds"]
    if set(seeds).intersection(args.train_seeds + args.selection_seeds + args.branch_fit_seeds
                               + args.timing_selection_seeds + args.trigger_eval_seeds):
        raise ValueError("evaluation overlaps fitting, selection or original reserved paths")
    spec, scale = PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id), time_contract(args)
    rows, budgets, latencies = {}, {}, {}
    for method, source in (("history", "history"), ("threshold", "threshold"), ("portfolio", chosen)):
        rule = TimingRule(**frozen[source]["selected_rule"])
        latency = ControlLatency()
        with ledger.recording(method):
            rows[method] = [run_episode(controller, task_spec=spec, seed=s, args=args, time_scale=scale, rule=rule,
                latency_recorder=latency, risk_spec={"distance": 1., "duration_steps": 25})[1] for s in seeds]
        summary = latencies[method] = latency.summary(episodes=len(seeds), deadline_seconds=.01)
        charge = {k: cell["frozen_budget_admission"][method][k] for k in ("offline_cpu_seconds", "offline_steps")}
        budgets[method] = within_budget(charge, summary, plan["budget"])
    counts = ledger.summary()["counts"]
    if counts["environment_steps"] != plan["steps_per_root"] or counts["planner_calls"] != 3 * len(seeds) * plan["calls_per_episode"]:
        raise ValueError("fixed evaluation steps/calls did not close")
    return {"evaluation_seeds": seeds, "portfolio_choice": chosen, "panels": panels(rows, methods=METHODS),
        "evaluation_budget_admission": budgets, "latency": latencies, "new_fits": 0,
        "sustained_failure_counts": {m: sum(r["reliability"]["sustained_failure"] for r in rs) for m, rs in rows.items()}}


def readout(plan, records):
    if len(records) != len(plan["cells"]) or any(r["status"] != "complete" or
        (r["optimizer_seed"], r["evaluation_seeds"], r["portfolio_choice"], r["new_fits"],
         r["measured_cost"]["counts"]["environment_steps"]) !=
        (c["optimizer_seed"], c["evaluation_seeds"], c["portfolio_choice"], 0, plan["steps_per_root"])
        for c, r in zip(plan["cells"], records)):
        raise ValueError("readout needs every completed paired root")
    def means(panel, method):
        return np.array([r["panels"][panel][method]["mean_ise"] for r in records])

    contrasts = {}
    for ref in METHODS[1:]:
        b, h = means("all", ref), means("all", "history")
        interval = paired_mean_interval(b - h)
        relative = float((b - h).mean() / b.mean())
        halves = [float(1 - means(p, "history").mean() / means(p, ref).mean()) for p in ("first", "last")]
        contrasts[ref] = {"paired_ise_gain": interval, "relative_gain": relative, "half_relative_gains": halves,
            "root_relative_gains": ((b - h) / b).tolist(),
            "passed": interval["interval"][0] > 0 and relative >= plan["minimum_ise_gain"] and min(halves) > 0}
    admitted = all(all(b["admitted"] for b in r["evaluation_budget_admission"].values()) for r in records)
    return {"roots": records, "mean_ise": {m: float(means("all", m).mean()) for m in METHODS},
        "comparisons": contrasts, "learned_choices": sum(r["portfolio_choice"] == "termination_history" for r in records),
        "budget_passed": admitted, "joint_gate_passed": admitted and all(v["passed"] for v in contrasts.values()),
        "original_confirmation_released": False, "scope": plan["scope"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("sources", "prepare", "run", "readout"))
    parser.add_argument("--cell", type=int)
    parser.add_argument("--source-metadata", type=Path)
    opts = parser.parse_args()
    old = json.loads((ROOT / MANIFEST).read_text())
    if opts.phase == "sources":
        print(json.dumps([sources(old, c["optimizer_seed"])[1] for c in old["confirmation_reservation"]]))
        return 0
    if opts.phase == "prepare":
        previous = ROOT / "results" / old["run_name"]
        if json.loads((previous / "confirmation_decision.json").read_text())["status"] != "BUILD_GATE_HOLD":
            raise ValueError("the original learned-only HOLD must remain unchanged")
        plan = prepare(old, json.loads((previous / "confirmation_builds.json").read_text())["roots"],
            json.loads((previous / "development_summary.json").read_text()),
            json.loads((ROOT / "results" / DISTANCE_RUN / "summary.json").read_text()),
            json.loads(opts.source_metadata.read_text()), historical_path_seeds(old))
        emit(ROOT / PLAN, plan)
        return 0
    plan = json.loads((ROOT / PLAN).read_text())
    if opts.phase == "readout":
        reports = [json.loads((ROOT / c["public"] / "result.json").read_text()) for c in plan["cells"]]
        print(json.dumps(readout(plan, reports), separators=(",", ":")))
        return 0
    cell = plan["cells"][opts.cell]
    output = ROOT / cell["public"] / "result.json"
    output.parent.mkdir(parents=True, exist_ok=False)
    ledger = ComputeLedger()
    try:
        with single_thread_runtime() as profile:
            if profile != old["hardware_profile"]:
                raise ValueError("runtime differs from original build")
            result = {**run(plan, cell, old, ledger), "status": "complete"}
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    emit(output, {**result, "optimizer_seed": cell["optimizer_seed"], "measured_cost": ledger.summary()})
    print(cell["optimizer_seed"], result["status"], result.get("error", ""), flush=True)
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
