"""Test frozen Reacher waypoint authority and episode-level timing opportunity."""

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

from freq_hrl.domains.mujoco.reacher_tracking import ReacherTrackingDriver
from freq_hrl.experiments.compute_budget import ComputeLedger, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_reliability import paired_mean_interval
from freq_hrl.experiments.plan_validity_transfer import PlanValidityTaskSpec, run_episode, time_contract
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from scripts.prepare_stage9_strong_timing import MANIFEST
from scripts.run_plan_validity_reliability import load_controller, load_record
from scripts.run_stage9_distance_diagnostic import historical_path_seeds
from scripts.run_stage9_strong_timing import ISE
from scripts.run_stage9_strong_timing_confirmation import emit
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON


RUN = "reacher_timing_applicability_20261112"
PLAN = Path("protocols/plan_validity_reacher_timing_applicability_2026-11-12.json")
RUNNER = "scripts/run_reacher_timing_applicability.py"
OFFSETS = (0, 5, 10, 15, 20, 25)
METHODS = ("best_clock", "worst_clock", "lagged_waypoint", "event_aligned", "rotated_event")


def clock_schedule(horizon, offset):
    return (0, *(start + offset for start in range(50, horizon, 50)))


def event_schedules(horizon, offset, events):
    """Privileged: align to the first switch reachable in each bin."""
    result = list(clock_schedule(horizon, offset))
    used = set()
    for event in sorted(events):
        bin_index, within = divmod(event, 50)
        if 0 < bin_index < len(result) and within <= 25 and bin_index not in used:
            result[bin_index] = 50 * bin_index + 5 * ((within + 4) // 5)
            used.add(bin_index)
    offsets = [step % 50 for step in result[1:]]
    shifted = offsets[-7:] + offsets[:-7]
    rotated = (0, *(50 * j + o for j, o in enumerate(shifted, start=1)))
    return tuple(result), rotated


def prepare():
    old = json.loads((ROOT / MANIFEST).read_text())
    parent = json.loads((ROOT / old["source_manifest"]).read_text())
    pilot = json.loads((ROOT / "results/plan_validity_dev_20261028_r2/analysis/complete_pilot_readout_20261029.json").read_text())
    used = historical_path_seeds(old)
    cells, jobs = [], []
    for i, row in enumerate(c for c in parent["cells"] if c["task"] == "reacher_tracking"):
        root = row["optimizer_seed"]
        prior, = [c for c in pilot["paired_cells"] if c["optimizer_seed"] == root and c["task"] == "reacher_tracking"]
        seeds = list(range(root + 98000, root + 98024))
        if set(seeds) & used:
            raise ValueError("applicability paths overlap another role")
        used.update(seeds)
        public = f"results/{RUN}/{root}/public"
        cells.append({"optimizer_seed": root, "source": row["sources"]["history"],
            "selection_seeds": row["expected_configuration"]["timing_selection_seeds"],
            "old_best_clock_ise": prior["best_clock_ise_derived_from_readiness"],
            "development_seeds": seeds, "public": public})
        job = deepcopy(old["jobs"]["audit"][0])
        cmd = [DEFAULT_LINUX_PYTHON, "-u", RUNNER, "run", "--cell", str(i)]
        job.update(description=f"Reacher timing applicability {root}",
            signature=f"Freq-HRL/reacher-timing-applicability/{RUN}/{root}",
            cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(cmd),
            result_dir=str(ROOT / public), local_result_dir=str(ROOT / public),
            wait_for_files=[str(ROOT / PLAN)])
        jobs.append(job)
    return {"run_name": RUN, "cells": cells, "jobs": jobs, "offsets": list(OFFSETS),
        "hardware_profile": parent["source_hardware_profile"], "minimum_gain": .02,
        "steps_per_root": 201600, "calls_per_episode": 24, "new_fits": 0,
        "gate": "both_roots_current_vs_lag_total_and_late_and_event_vs_best_and_rotated_each_pass",
        "scope": "two_frozen_Reacher_roots_development_only_privileged_timing_probe_not_policy_transfer",
        "original_confirmation_released": False}


def evaluate(plan, cell):
    ref = cell["source"]
    source = load_record(Path(ref["report"]), ref["run_id"])
    if (source["task"], source["optimizer_seed"], source["method"]) != (
        "reacher_tracking", cell["optimizer_seed"], "history"):
        raise ValueError("expected the unchanged Reacher history-controller artifact")
    controller, args = load_controller(source)
    if cell["selection_seeds"] != args.timing_selection_seeds:
        raise ValueError("clock selection role changed")
    if set(cell["development_seeds"]).intersection(args.train_seeds + args.selection_seeds + args.branch_fit_seeds
                                                    + args.timing_selection_seeds + args.trigger_eval_seeds):
        raise ValueError("new paths overlap a source role")
    spec = PlanValidityTaskSpec("reacher_tracking", _task_options(args), args.env_id)
    scale = time_contract(args)

    def play(seed, schedule, audit=None):
        return run_episode(controller, task_spec=spec, seed=seed, args=args, time_scale=scale,
                           schedule=schedule, waypoint_audit=audit)[1]

    with measured_phase("reproduce_old_clock_selection"):
        means = {o: float(np.mean([play(s, clock_schedule(args.horizon, o))[ISE] for s in cell["selection_seeds"]]))
                 for o in plan["offsets"]}
        best = min(means, key=lambda o: (means[o], o))
        worst = max(means, key=lambda o: (means[o], -o))
        if not np.isclose(means[best], cell["old_best_clock_ise"], rtol=1e-10, atol=1e-10):
            raise ValueError("saved best clock did not reproduce; do not open new development paths")
    paths = []
    with measured_phase("paired_development_paths"):
        for seed in cell["development_seeds"]:
            driver = ReacherTrackingDriver(seed=seed, horizon=args.horizon, dt_seconds=scale.dt_seconds, **spec.options)
            event, rotated = event_schedules(args.horizon, best, driver.regime_change_steps)
            clock = clock_schedule(args.horizon, best)
            rows = {"best_clock": play(seed, clock, "current"),
                "worst_clock": play(seed, clock_schedule(args.horizon, worst)),
                "lagged_waypoint": play(seed, clock, "lag_one_bin"),
                "event_aligned": play(seed, event), "rotated_event": play(seed, rotated)}
            paths.append({"seed": seed, "ise": {m: rows[m][ISE] for m in METHODS},
                "late_ise": {m: rows[m]["late_option_ise"] for m in ("best_clock", "lagged_waypoint")},
                "aligned_changes_from_clock": sum(a != b for a, b in zip(event, clock)),
                "aligned_changes_from_rotated": sum(a != b for a, b in zip(event, rotated))})
    return {"selection_mean_ise": means, "best_offset": best, "worst_offset": worst, "paths": paths}


def contrast(paths, field, candidate, reference, floor):
    a, b = (np.array([p[field][m] for p in paths]) for m in (candidate, reference))
    delta = b - a
    relative = float(delta.mean() / b.mean())
    halves = [float(d.mean() / v.mean()) for d, v in zip(np.split(delta, 2), np.split(b, 2))]
    interval = paired_mean_interval(delta)
    return {"relative_gain": relative, "absolute_gain": interval, "half_gains": halves,
        "passed": relative >= floor and min(halves) > 0 and interval["interval"][0] > 0}


def readout(plan, records):
    if len(records) != len(plan["cells"]) or any(r["status"] != "complete" or
        (r["optimizer_seed"], [p["seed"] for p in r["paths"]], r["measured_cost"]["counts"]["environment_steps"]) !=
        (c["optimizer_seed"], c["development_seeds"], plan["steps_per_root"]) for c, r in zip(plan["cells"], records)):
        raise ValueError("requires both complete development roots")
    roots = []
    for r in records:
        paths = r["paths"]
        tests = {name: contrast(paths, field, a, b, plan["minimum_gain"]) for name, field, a, b in (
            ("waypoint_total", "ise", "best_clock", "lagged_waypoint"),
            ("waypoint_late", "late_ise", "best_clock", "lagged_waypoint"),
            ("event_vs_best", "ise", "event_aligned", "best_clock"),
            ("event_vs_rotated", "ise", "event_aligned", "rotated_event"),
            ("clock_sensitivity", "ise", "best_clock", "worst_clock"))}
        roots.append({"optimizer_seed": r["optimizer_seed"], "best_offset": r["best_offset"], "worst_offset": r["worst_offset"],
            "mean_ise": {m: float(np.mean([p["ise"][m] for p in paths])) for m in METHODS},
            "mean_changed_calls": {k: float(np.mean([p[k] for p in paths]))
                                    for k in ("aligned_changes_from_clock", "aligned_changes_from_rotated")},
            "contrasts": tests, "measured_cost": r["measured_cost"]})
    authority = all(r["contrasts"][k]["passed"] for r in roots for k in ("waypoint_total", "waypoint_late"))
    timing = all(r["contrasts"][k]["passed"] for r in roots for k in ("event_vs_best", "event_vs_rotated"))
    return {"roots": roots, "waypoint_authority_passed": authority, "event_timing_passed": timing,
            "development_gate_passed": authority and timing, "original_confirmation_released": False, "scope": plan["scope"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "readout"))
    parser.add_argument("--cell", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        emit(ROOT / PLAN, prepare())
        return 0
    plan = json.loads((ROOT / PLAN).read_text())
    if args.phase == "readout":
        records = [json.loads((ROOT / c["public"] / "result.json").read_text()) for c in plan["cells"]]
        print(json.dumps(readout(plan, records), separators=(",", ":")))
        return 0
    cell = plan["cells"][args.cell]
    out = ROOT / cell["public"]
    out.mkdir(parents=True, exist_ok=False)
    ledger = ComputeLedger()
    try:
        with single_thread_runtime() as profile, ledger.recording("applicability"):
            if profile != plan["hardware_profile"]:
                raise ValueError("runtime differs from original build")
            result = evaluate(plan, cell)
        count = ledger.summary()["counts"]
        if (count["environment_steps"] != plan["steps_per_root"] or
            count["planner_calls"] != plan["steps_per_root"] // 1200 * plan["calls_per_episode"]):
            raise ValueError("native steps or fixed calls did not close")
        result.update(status="complete")
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    emit(out / "result.json", {**result, "optimizer_seed": cell["optimizer_seed"], "measured_cost": ledger.summary()})
    print(cell["optimizer_seed"], result["status"], result.get("error", ""), flush=True)
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
