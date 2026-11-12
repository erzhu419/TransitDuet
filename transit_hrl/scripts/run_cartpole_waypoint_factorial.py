"""Cross frozen cart-pole gains and clock offsets on common fresh paths."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from freq_hrl.experiments.compute_budget import ComputeLedger, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_reliability import paired_mean_interval
from freq_hrl.experiments.plan_validity_transfer import run_episode, time_contract
from scripts.prepare_stage9_strong_timing import MANIFEST
from scripts.run_cartpole_timing_applicability import (
    PLAN as PARENT, RUN as PARENT_RUN, CartpoleQualificationController, CartpoleSpec, clock_schedule,
)
from scripts.run_stage9_distance_diagnostic import historical_path_seeds
from scripts.run_stage9_strong_timing import ISE
from scripts.run_stage9_strong_timing_confirmation import emit
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON


RUN = "cartpole_waypoint_factorial_20261112"
PLAN = Path("protocols/plan_validity_cartpole_waypoint_factorial_2026-11-12.json")
RUNNER = "scripts/run_cartpole_waypoint_factorial.py"
MODES = ("current", "lag_one_bin")


def prepare(parent, summary, used):
    weights = [c["position_weight"] for c in parent["cells"]]
    offsets = sorted(c["best_offset"] for c in summary["configurations"])
    if summary["qualification_passed"] or weights != [10., 30.] or offsets != [5, 20]:
        raise ValueError("expected the preserved parent qualification and its two clocks")
    seeds = list(range(1981730200, 1981730232))
    if used.intersection(seeds):
        raise ValueError("factorial paths overlap a registered role")
    steps = parent["config"]["horizon"] * len(seeds) * len(offsets) * len(MODES)
    cells, jobs = [], []
    for i, (source, weight) in enumerate(zip(parent["jobs"], weights)):
        public = f"results/{RUN}/gain{i}/public"
        cells.append({"configuration": i, "position_weight": weight, "public": public})
        job = deepcopy(source)
        command = [DEFAULT_LINUX_PYTHON, "-u", RUNNER, "run", "--cell", str(i)]
        job.update(description=f"Cart-pole waypoint factorial LQR weight {weight:g}",
            signature=f"Freq-HRL/cartpole-waypoint-factorial/{RUN}/{i}",
            cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(command),
            result_dir=str(ROOT / public), local_result_dir=str(ROOT / public),
            wait_for_files=[str(ROOT / PLAN)])
        jobs.append(job)
    return {"run_name": RUN, "source_plan": str(PARENT), "config": parent["config"],
        "hardware_profile": parent["hardware_profile"], "cells": cells, "offsets": offsets,
        "development_seeds": seeds, "steps_per_gain": steps, "calls_per_episode": 24,
        "linearization_steps_per_gain": 10, "new_policy_fits": 0, "jobs": jobs,
        "inference": "whole_path_paired_nominal_t95_gain_and_offset_marginal_contrasts_and_interaction",
        "scope": "diagnostic_only_no_qualification_or_transfer_release",
        "parent_qualification_passed": False, "original_confirmation_released": False}


def evaluate(plan, cell):
    args = SimpleNamespace(**plan["config"])
    spec, scale = CartpoleSpec(), time_contract(args)
    with measured_phase("same_lqr_design"):
        task = spec.open(seed=0, horizon=args.horizon)
        try:
            controller = CartpoleQualificationController(task.environment.unwrapped.model, cell["position_weight"])
        finally:
            task.environment.close()
    paths = [{"seed": seed, "offsets": {}} for seed in plan["development_seeds"]]
    for offset in plan["offsets"]:
        schedule = clock_schedule(args.horizon, offset)
        with measured_phase(f"offset_{offset}"):
            for row in paths:
                result = row["offsets"][str(offset)] = {}
                for mode in MODES:
                    episode = run_episode(controller, task_spec=spec, seed=row["seed"], args=args, time_scale=scale,
                        schedule=schedule, waypoint_audit=mode)[1]
                    if episode["decision_steps"] != list(schedule):
                        raise RuntimeError("waypoint intervention changed the clock")
                    result[mode] = {"episode_ise": episode[ISE], "late_ise": episode["late_option_ise"]}
    return {"paths": paths, "lqr_spectral_radius": controller.spectral_radius}


def effect(values):
    return {**paired_mean_interval(values), "half_means": [float(v.mean()) for v in np.split(values, 2)]}


def readout(plan, records):
    if len(records) != 2 or any(
        r["status"] != "complete" or r["configuration"] != c["configuration"]
        or [p["seed"] for p in r["paths"]] != plan["development_seeds"]
        or r["measured_cost"]["counts"]["environment_steps"] != plan["steps_per_gain"]
        for c, r in zip(plan["cells"], records)):
        raise ValueError("factorial needs both gains and the identical ordered seed roster")
    results = {}
    for metric in ("episode_ise", "late_ise"):
        timely, lagged = (np.asarray([[[p["offsets"][str(offset)][mode][metric] for p in r["paths"]]
            for offset in plan["offsets"]] for r in records]) for mode in MODES)
        d = lagged - timely
        cells = [{"position_weight": c["position_weight"], "offset": offset,
            "mean_current": float(timely[i, j].mean()), "mean_lagged": float(lagged[i, j].mean()),
            "relative_gain": float(d[i, j].mean() / lagged[i, j].mean()), "absolute_gain": effect(d[i, j])}
            for i, c in enumerate(plan["cells"]) for j, offset in enumerate(plan["offsets"])]
        gain = d[1] - d[0]
        offset = d[:, 1] - d[:, 0]
        results[metric] = {"cells": cells, "gain_30_minus_10": effect(gain.mean(axis=0)),
            "offset_20_minus_5": effect(offset.mean(axis=0)), "interaction": effect(gain[1] - gain[0]),
            "gain_effect_by_offset": [effect(v) for v in gain], "offset_effect_by_gain": [effect(v) for v in offset]}
    return {"metrics": results, "paired_paths": len(plan["development_seeds"]),
        "workers": [{k: r[k] for k in ("configuration", "lqr_spectral_radius", "measured_cost")} for r in records],
        "scope": plan["scope"], "parent_qualification_passed": False, "original_confirmation_released": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "readout"))
    parser.add_argument("--cell", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        parent = json.loads((ROOT / PARENT).read_text())
        summary = json.loads((ROOT / "results" / PARENT_RUN / "summary.json").read_text())
        source = json.loads((ROOT / MANIFEST).read_text())
        emit(ROOT / PLAN, prepare(parent, summary, historical_path_seeds(source)))
        return 0
    plan = json.loads((ROOT / PLAN).read_text())
    if args.phase == "readout":
        rows = [json.loads((ROOT / c["public"] / "result.json").read_text()) for c in plan["cells"]]
        print(json.dumps(readout(plan, rows), separators=(",", ":")))
        return 0
    cell = plan["cells"][args.cell]
    path = ROOT / cell["public"]
    path.mkdir(parents=True, exist_ok=False)
    ledger = ComputeLedger()
    try:
        with single_thread_runtime() as profile, ledger.recording("waypoint_factorial"):
            if profile != plan["hardware_profile"]:
                raise ValueError("runtime differs from original qualification")
            result = evaluate(plan, cell)
        counts = ledger.summary()["counts"]
        if (counts["environment_steps"], counts["planner_calls"], counts["linearization_native_steps"]) != (
            plan["steps_per_gain"], plan["steps_per_gain"] // 50, plan["linearization_steps_per_gain"]):
            raise RuntimeError("factorial simulator or call ledger did not close")
        result["status"] = "complete"
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    emit(path / "result.json", {**result, "configuration": cell["configuration"], "measured_cost": ledger.summary()})
    print(cell["configuration"], result["status"], result.get("error", ""), flush=True)
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
