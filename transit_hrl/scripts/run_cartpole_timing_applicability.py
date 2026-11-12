"""Qualification of visible setpoint timing on a stabilized native cart-pole."""

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

from freq_hrl.domains.mujoco import RelativeSubgoalAdapter
from freq_hrl.domains.mujoco.cartpole_setpoint import CartpoleSetpointTask, MAX_GOAL_DELTA, command_path
from freq_hrl.experiments.compute_budget import ComputeLedger, count_work, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_transfer import run_episode, time_contract
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from scripts.prepare_stage9_strong_timing import MANIFEST
from scripts.run_reacher_timing_applicability import clock_schedule, contrast, OFFSETS
from scripts.run_stage9_distance_diagnostic import historical_path_seeds
from scripts.run_stage9_strong_timing import ISE
from scripts.run_stage9_strong_timing_confirmation import emit
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON


RUN = "cartpole_timing_applicability_20261112"
PLAN = Path("protocols/plan_validity_cartpole_timing_applicability_2026-11-12.json")
RUNNER = "scripts/run_cartpole_timing_applicability.py"
METHODS = ("best_clock", "lagged_waypoint", "visible_event", "rotated_event")


class CartpoleSpec:
    name = "cartpole_setpoint"

    def open(self, *, seed, horizon):
        count_work("environment_opens")
        return CartpoleSetpointTask(seed=seed, horizon=horizon)

    def adapter(self, task, *, maximum_subgoal_delta):
        return RelativeSubgoalAdapter(maximum_delta=np.full(2, maximum_subgoal_delta))


def linearize(model):
    import mujoco
    from scipy.optimize import brentq

    data = mujoco.MjData(model)

    def slope(theta):
        data.qpos[:] = [0., theta]
        mujoco.mj_forward(model, data)
        return float(data.qacc[1])

    upright = brentq(slope, -.03, .03)

    def one_step(z):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = [z[0], upright + z[1]]
        data.qvel[:] = z[2:4]
        data.ctrl[0] = z[4]
        mujoco.mj_step(model, data)
        count_work("linearization_native_steps")
        return np.r_[data.qpos, data.qvel]

    d = 1e-5 * np.eye(5)
    jacobian = np.column_stack([(one_step(e) - one_step(-e)) / 2e-5 for e in d])
    return upright, jacobian[:, :4], jacobian[:, 4:]


class CartpoleQualificationController:
    """Fixed reference-following upper and model-derived LQR lower."""

    config = SimpleNamespace(gamma=.995)

    def __init__(self, model, position_weight):
        from scipy.linalg import solve_discrete_are

        self.upright, a, b = linearize(model)
        p = solve_discrete_are(a, b, np.diag([position_weight, 100., .2, .15]), np.eye(1))
        self.gain = np.linalg.solve(np.eye(1) + b.T @ p @ b, b.T @ p @ a)
        self.spectral_radius = float(np.max(np.abs(np.linalg.eigvals(a - b @ self.gain))))
        if self.spectral_radius >= 1:
            raise RuntimeError("nominal LQR is not stable")

    def reset_recurrent_inference(self):
        pass

    def plan_goal(self, state, *, sample):
        return {"action": state[4:6] / MAX_GOAL_DELTA}

    def act_conditioned(self, state, *, sample):
        x = np.array([-state[4], state[1] - self.upright, state[2], state[3]])
        control = -self.gain @ x
        return {"action": np.arctanh(np.clip(control / 3., -.9999, .9999))}


def visible_schedule(horizon, command_at):
    calls = [0]
    issued = command_at(0)
    for start in range(50, horizon, 50):
        for offset in OFFSETS:
            step = start + offset
            value = command_at(step)
            if value != issued or offset == OFFSETS[-1]:
                calls.append(step)
                issued = value
                break
    return tuple(calls)


def rotate_schedule(schedule):
    offsets = [s % 50 for s in schedule[1:]]
    return (0, *(50 * j + offset for j, offset in enumerate(offsets[-7:] + offsets[:-7], start=1)))


def prepare():
    source = json.loads((ROOT / MANIFEST).read_text())
    used = historical_path_seeds(source)
    cells, jobs = [], []
    for i, weight in enumerate((10., 30.)):
        base = 1981700000 + 10000 * i
        selection = list(range(base + 100, base + 108))
        evaluation = list(range(base + 200, base + 224))
        if used.intersection(selection + evaluation):
            raise ValueError("cart-pole development path seed collision")
        used.update(selection + evaluation)
        public = f"results/{RUN}/config{i}/public"
        cells.append({"configuration": i, "position_weight": weight, "clock_selection_seeds": selection,
                      "development_seeds": evaluation, "public": public})
        job = deepcopy(source["jobs"]["audit"][0])
        cmd = [DEFAULT_LINUX_PYTHON, "-u", RUNNER, "run", "--cell", str(i)]
        job.update(description=f"Cart-pole timing qualification config{i}",
            signature=f"Freq-HRL/cartpole-timing-applicability/{RUN}/config{i}",
            cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(cmd),
            result_dir=str(ROOT / public), local_result_dir=str(ROOT / public), wait_for_files=[str(ROOT / PLAN)])
        jobs.append(job)
    return {"run_name": RUN, "cells": cells, "jobs": jobs, "hardware_profile": source["hardware_profile"],
        "config": {"horizon": 1200, "upper_period_seconds": .5, "history_seconds": .64, "fast_period_seconds": .04,
                   "maximum_subgoal_delta": MAX_GOAL_DELTA, "max_offset_steps": 25, "check_stride_steps": 5},
        "steps_per_config": 172800, "linearization_steps_per_config": 10, "minimum_gain": .02, "new_policy_fits": 0,
        "scope": "designed_native_cartpole_setpoint_task_two_fixed_LQR_configs_not_policy_transfer",
        "gate": "both_waypoint_contrasts_and_both_event_contrasts_pass_in_each_configuration",
        "original_confirmation_released": False}


def evaluate(plan, cell):
    args = SimpleNamespace(**plan["config"])
    spec, scale = CartpoleSpec(), time_contract(args)
    # Linearize at the model equilibrium, not along any selection or evaluation path.
    task = spec.open(seed=0, horizon=args.horizon)
    try:
        with measured_phase("fixed_model_linearization"):
            controller = CartpoleQualificationController(task.environment.unwrapped.model, cell["position_weight"])
    finally:
        task.environment.close()

    def play(seed, schedule, audit=None):
        return run_episode(controller, task_spec=spec, seed=seed, args=args, time_scale=scale,
                           schedule=schedule, waypoint_audit=audit)[1]

    with measured_phase("six_clock_selection"):
        costs = {offset: float(np.mean([play(seed, clock_schedule(args.horizon, offset))[ISE]
                    for seed in cell["clock_selection_seeds"]])) for offset in OFFSETS}
        best = min(costs, key=lambda offset: (costs[offset], offset))
    clock = clock_schedule(args.horizon, best)
    paths = []
    with measured_phase("paired_development"):
        for seed in cell["development_seeds"]:
            target = command_path(seed, args.horizon)
            visible = visible_schedule(args.horizon, target.__getitem__)
            rotated = rotate_schedule(visible)
            rows = {"best_clock": play(seed, clock, "current"), "lagged_waypoint": play(seed, clock, "lag_one_bin"),
                    "visible_event": play(seed, visible), "rotated_event": play(seed, rotated)}
            paths.append({"seed": seed, "ise": {m: rows[m][ISE] for m in METHODS},
                "late": {m: rows[m]["late_option_ise"] for m in METHODS[:2]},
                "event_changes": sum(a != b for a, b in zip(visible, clock)),
                "rotated_changes": sum(a != b for a, b in zip(visible, rotated))})
    return {"best_offset": best, "selection_ise": costs, "lqr_spectral_radius": controller.spectral_radius, "paths": paths}


def readout(plan, records):
    if len(records) != len(plan["cells"]) or any(r["status"] != "complete" or
        (r["configuration"], [p["seed"] for p in r["paths"]], r["measured_cost"]["counts"]["environment_steps"]) !=
        (c["configuration"], c["development_seeds"], plan["steps_per_config"]) for c, r in zip(plan["cells"], records)):
        raise ValueError("qualification needs both completed configurations")
    configs = []
    for r in records:
        paths = r["paths"]
        contrasts = {label: contrast(paths, field, candidate, reference, plan["minimum_gain"])
                     for label, field, candidate, reference in (
                         ("waypoint_total", "ise", "best_clock", "lagged_waypoint"),
                         ("waypoint_late", "late", "best_clock", "lagged_waypoint"),
                         ("event_vs_best", "ise", "visible_event", "best_clock"),
                         ("event_vs_rotated", "ise", "visible_event", "rotated_event"))}
        configs.append({**{k: v for k, v in r.items() if k != "paths"}, "contrasts": contrasts,
            "mean_ise": {m: float(np.mean([p["ise"][m] for p in paths])) for m in METHODS},
            "mean_changed_calls": {k: float(np.mean([p[k] for p in paths])) for k in ("event_changes", "rotated_changes")}})
    return {"configurations": configs, "qualification_passed": all(c["passed"] for r in configs
            for c in r["contrasts"].values()), "scope": plan["scope"], "original_confirmation_released": False}


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
        rows = [json.loads((ROOT / c["public"] / "result.json").read_text()) for c in plan["cells"]]
        print(json.dumps(_json_ready(readout(plan, rows)), separators=(",", ":")))
        return 0
    cell = plan["cells"][args.cell]
    path = ROOT / cell["public"]
    path.mkdir(parents=True, exist_ok=False)
    ledger = ComputeLedger()
    try:
        with single_thread_runtime() as profile, ledger.recording("qualification"):
            if profile != plan["hardware_profile"]:
                raise ValueError("runtime differs from frozen scheduler profile")
            result = evaluate(plan, cell)
        count = ledger.summary()["counts"]
        if (count["environment_steps"], count["planner_calls"], count["linearization_native_steps"]) != (
            plan["steps_per_config"], plan["steps_per_config"] // 50, plan["linearization_steps_per_config"]):
            raise RuntimeError("native qualification ledger differs")
        result["status"] = "complete"
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    emit(path / "result.json", {**result, "configuration": cell["configuration"], "measured_cost": ledger.summary()})
    print(cell["configuration"], result["status"], result.get("error", ""), flush=True)
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
