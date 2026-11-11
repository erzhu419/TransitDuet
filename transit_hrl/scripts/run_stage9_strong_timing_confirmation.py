"""Conditional fresh-root build and frozen PointMaze evaluation for Stage9."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from freq_hrl.experiments.compute_budget import ComputeLedger, ControlLatency, single_thread_runtime
from freq_hrl.experiments.plan_validity_reliability import paired_mean_interval
from freq_hrl.experiments.plan_validity_transfer import PlanValidityTaskSpec, run_episode, time_contract, train_shared_controller
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts.run_plan_validity_compute_budget import run_trial
from scripts.run_plan_validity_reliability import load_controller
from scripts.run_stage9_strong_timing import ISE, METHODS, panels, readout


def emit(path, payload):
    with path.open("x") as out:
        json.dump(_json_ready(payload), out, indent=2)
        out.write("\n")


def development_reports(plan):
    return [json.loads((ROOT / c["directory"] / "audit/public/result.json").read_text())
            for c in plan["development_cells"]]


def directory(plan, root):
    return ROOT / "results" / plan["run_name"] / "confirmation" / str(root)


def require_development(plan):
    reports = development_reports(plan)
    if not readout(plan, reports)["confirmation_builds_allowed"]:
        raise ValueError("fresh roots are not released by the frozen development gate")
    return reports


def build(plan, cell, workdir, shared, records):
    reports = require_development(plan)
    params = {**plan["development_cells"][0]["expected_configuration"], **cell}
    args = SimpleNamespace(**params)
    spec = PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id)
    budgets = {}
    with single_thread_runtime() as profile:
        if profile != plan["hardware_profile"]:
            raise ValueError("confirmation must use the registered hardware/runtime")
        print(cell["optimizer_seed"], "development gate passed; training one shared controller", flush=True)
        with shared.recording("shared_controller"):
            payload, controller = train_shared_controller(spec, args, time_contract(args))
            checkpoint = workdir / "private/controller.pt"
            torch.save(controller.state_dict(), checkpoint)
        common = shared.summary()
        for method in METHODS:
            print(cell["optimizer_seed"], "building frozen timing rule", method, flush=True)
            args = SimpleNamespace(**{**params, "method": method, "output": workdir / f"private/{method}.json"})
            record = run_trial(args, "pointmaze", shared_training=(payload, controller, checkpoint))
            records[method] = record
            emit(args.output, record)
            if record["status"] != "validation_complete" or not record["accounting_complete"]:
                raise RuntimeError(f"fresh build failed: {method}: {record.get('error')}")
            local = record["offline_cost"]
            cohorts = len(plan["confirmation_reservation"])
            charged_cpu = (common["cpu_seconds"] + local["cpu_seconds"] +
                sum(r["budgets"][method]["offline_cpu_seconds"] + r["measured_cost"]["cpu_seconds"] for r in reports) / cohorts)
            charged_steps = (common["counts"]["environment_steps"] + local["counts"].get("environment_steps", 0) +
                sum(r["budgets"][method]["offline_steps"] + r["measured_cost"]["counts"]["environment_steps"] for r in reports) / cohorts)
            cap, latency = plan["budget"], record["validation_latency"]
            lifecycle_cpu = charged_cpu + cap["deployments"] * latency["cpu_seconds_per_episode"]
            budgets[method] = {"charged_steps": charged_steps, "charged_cpu_seconds": charged_cpu,
                "lifecycle_cpu_seconds": lifecycle_cpu, "validation_latency": latency, "fit_cost": local,
                "admitted": charged_steps <= cap["offline_steps"] and lifecycle_cpu <= cap["cpu_seconds"]
                    and latency["wall_p99_seconds"] <= cap["p99_seconds"]}
    readiness = records["termination_history"]["termination_readiness"]
    return {"status": "complete", "hardware_profile": profile, "test_accessed": False,
            "termination_readiness": readiness, "budgets": budgets, "shared_cost": common,
            "admitted": readiness["eligible_for_fresh_confirmation"] and all(r["admitted"] for r in budgets.values())}


def require_global_freeze(plan):
    for cell in plan["confirmation_reservation"]:
        root = cell["optimizer_seed"]
        home = directory(plan, root)
        report = json.loads((home / "build/public/result.json").read_text())
        if (report["status"], report["optimizer_seed"], report["admitted"], report["test_accessed"]) != (
            "complete", root, True, False):
            raise ValueError("all fresh builds must be frozen and eligible before any test access")
        for path in ("controller.pt", *(m + ".json" for m in METHODS)):
            if not (home / "build/private" / path).is_file():
                raise ValueError("incomplete global artifact freeze")


def evaluate(plan, cell, workdir, ledger):
    require_global_freeze(plan)
    source = directory(plan, cell["optimizer_seed"]) / "build/private"
    frozen = {m: json.loads((source / f"{m}.json").read_text()) for m in METHODS}
    controller, args = load_controller(frozen["history"])
    rules = {m: TimingRule(**r["selected_rule"]) for m, r in frozen.items()}
    spec, scale = PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id), time_contract(args)
    rows, costs = {}, {}
    seeds = cell["trigger_eval_seeds"]
    if set(seeds).intersection(args.train_seeds + args.selection_seeds + args.branch_fit_seeds + args.timing_selection_seeds):
        raise ValueError("confirmation evaluation overlaps fitting or selection")
    with single_thread_runtime() as profile:
        if profile != plan["hardware_profile"]:
            raise ValueError("test hardware/runtime differs")
        for method, rule in rules.items():
            latency = ControlLatency()
            with ledger.recording(method):
                rows[method] = [run_episode(controller, task_spec=spec, seed=seed, args=args,
                    time_scale=scale, rule=rule, latency_recorder=latency,
                    risk_spec={"distance": 1., "duration_steps": 25})[1] for seed in seeds]
            costs[method] = latency.summary(episodes=len(seeds), deadline_seconds=.01)
    count = ledger.summary()["counts"]
    if count["environment_steps"] != len(METHODS) * len(seeds) * 1200 or count["planner_calls"] != len(METHODS) * len(seeds) * 24:
        raise RuntimeError("test native or planner accounting differs")
    emit(workdir / "private/episode_metrics.json", {m: [{k: r[k] for k in
        ("seed", ISE, "episode_return", "decision_steps", "reliability")} for r in rs] for m, rs in rows.items()})
    return {"status": "complete", "evaluation_seeds": seeds, "panels": panels(rows),
        "sustained_failure_counts": {m: sum(r["reliability"]["sustained_failure"] for r in rs) for m, rs in rows.items()},
        "latency": costs, "measured_cost": ledger.summary(), "new_fits": 0}


def confirmation_readout(plan, reports):
    cells = plan["confirmation_reservation"]
    if len(reports) != len(cells) or any((r["status"], r["optimizer_seed"], r["evaluation_seeds"]) !=
        ("complete", c["optimizer_seed"], c["trigger_eval_seeds"]) for c, r in zip(cells, reports)):
        raise ValueError("analysis requires all eight complete registered evaluations")
    contrasts = {}
    for method in METHODS[1:]:
        def effects(panel):
            h = np.asarray([r["panels"][panel]["history"]["mean_ise"] for r in reports])
            b = np.asarray([r["panels"][panel][method]["mean_ise"] for r in reports])
            return b - h, float((b - h).mean() / b.mean())

        paired, relative = effects("all")
        interval = paired_mean_interval(paired)
        halves = [effects(p)[1] for p in ("first", "last")]
        passed = relative >= plan["minimum_ise_gain"]
        if method != "best_clock":
            passed = passed and interval["interval"][0] > 0 and min(halves) > 0
        contrasts[method] = {**interval, "relative_gain": relative, "root_gains": paired.tolist(),
                             "relative_gain_halves": halves, "passed": passed}
    return {"pointmaze_strong_timing_supported": all(r["passed"] for r in contrasts.values()),
            "contrasts": contrasts, "roots": reports, "original_two_task_confirmation_released": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=("build", "evaluate", "readout"), required=True)
    parser.add_argument("--cell", type=int)
    opts = parser.parse_args()
    plan = json.loads(opts.manifest.read_text())
    require_development(plan)
    if opts.phase == "readout":
        reports = [json.loads((directory(plan, c["optimizer_seed"]) / "evaluate/public/result.json").read_text())
                   for c in plan["confirmation_reservation"]]
        print(json.dumps(confirmation_readout(plan, reports), separators=(",", ":")))
        return 0
    cell = plan["confirmation_reservation"][opts.cell]
    root = cell["optimizer_seed"]
    workdir = directory(plan, root) / opts.phase
    (workdir / "public").mkdir(parents=True, exist_ok=False)
    (workdir / "private").mkdir()
    ledger, records = ComputeLedger(), {}
    try:
        result = (build(plan, cell, workdir, ledger, records) if opts.phase == "build"
                  else evaluate(plan, cell, workdir, ledger))
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "measured_cost": ledger.summary(), "measured_fit_costs": {m: r["offline_cost"] for m, r in records.items()}}
    emit(workdir / "public/result.json", {"optimizer_seed": root, **result})
    print(root, opts.phase, result["status"], result.get("error", ""), flush=True)
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
