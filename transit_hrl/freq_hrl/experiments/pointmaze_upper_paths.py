"""Factor the sampled upper residual into position and velocity pathways."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
import time

import numpy as np
import torch

from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO
from . import pointmaze_learned_plan as learned
from . import pointmaze_native_direction as native
from . import pointmaze_crossed_direction as previous
from .pointmaze_root_response import write_json
from scripts import pointmaze_upper_paths_stage75_spec as spec


class PathFactorPlan(learned.ResidualPlan):
    def __init__(self, predictor, period, scale, mode):
        super().__init__(predictor, period, scale)
        if mode not in spec.MODES:
            raise ValueError("unregistered Stage75 path intervention")
        self.mode, self.proposed_actions = mode, []
        self.reference_target_squared_error_integral = 0.
        self.reference_residual_squared_integral = 0.
        self.velocity_residual_squared_integral = 0.

    def decode(self, *, action, **kwargs):
        self.proposed_actions.append(action.copy())
        super().decode(action=action, **kwargs)
        self.reference_points = self.points if self.mode[1] == "1" else self.base_points
        self.velocity_points = self.points if self.mode[3] == "1" else self.base_points
        return self.reference_points[0].copy()

    def __call__(self, *, age, observation, **kwargs):
        self.calls += 1
        reference = self.reference_points[age]
        dt = learned.forecast.spec.DT_SECONDS
        self.reference_target_squared_error_integral += float(np.square(
            reference.astype(np.float64) - observation.task_measurement[:2]).sum() * dt)
        self.reference_residual_squared_integral += float(np.square(
            reference.astype(np.float64) - self.base_points[age]).sum() * dt)
        return reference.copy()

    def velocity(self, age):
        return (self.velocity_points[age + 1] - self.velocity_points[age]) / learned.forecast.spec.DT_SECONDS

    def actor_context(self, *, age, step, horizon):
        self.context_calls += 1
        velocity = self.velocity(age)
        dt = learned.forecast.spec.DT_SECONDS
        base_velocity = (self.base_points[age + 1] - self.base_points[age]) / dt
        self.velocity_residual_squared_integral += float(np.square(velocity.astype(np.float64) - base_velocity).sum() * dt)
        return velocity

    def value_context(self, *, age, step, horizon):
        return np.r_[self.velocity(age), learned.clocks.time_context(
            age=age, step=step, horizon=horizon, clock=True)].astype(np.float32)


_WORKER = None
PLANNING_KEYS = ("plan_ols_fits", "plan_ridge_predictions", "reference_evaluations", "actor_context_evaluations")
PAIR_KEYS = ("policy_seed", "lower_seed", "decision_steps", "upper_proposed_actions")


def init_worker(config, args):
    global _WORKER
    torch.set_num_threads(1)
    native.init_worker(config, args)
    _WORKER = FrequencySeparatedActorCriticPPO(config), args


def worker_native(job):
    weights, seed, mode, period, predictor = job
    model, args = _WORKER
    model.load_state_dict(weights)
    policy_seed = native.native.spec.policy_seed(args.optimizer_seed, seed)
    torch.manual_seed(policy_seed)
    plan = PathFactorPlan(predictor, period, args.maximum_subgoal_delta, mode)
    kwargs = native.native.spec.rollout_arguments(args.optimizer_seed, seed, phase="train", mode="training")
    kwargs["sample"] = False
    batch, row, raw = native.joint.rollout(model, args, f"fixed{period}", seed=seed, capture=False,
        lower_credit="task_option", upper_plan_decoder=plan.decode, lower_reference_builder=plan,
        lower_actor_context_builder=plan.actor_context, lower_value_context_builder=plan.value_context, **kwargs)
    if batch is not None or raw is not None or not (row["upper_sample"] and row["lower_sample"]):
        raise ValueError("Stage75 changed actor sampling or materialized training data")
    torch.testing.assert_close(native.joint.inference_weights(model), weights, atol=0, rtol=0)
    return {"seed": seed, "mode": mode, **{k: row[k] for k in (*spec.METRICS, "episode_length", "decision_steps", "lower_seed")},
        "upper_calls": row["upper_inference_calls"], "lower_calls": row["lower_inference_calls"],
        "policy_seed": policy_seed, "upper_proposed_actions": np.asarray(plan.proposed_actions).tolist(),
        "network_check": "passed", "plan_ols_fits": plan.ols_fits, "plan_ridge_predictions": plan.ridge_predictions,
        "reference_evaluations": plan.calls, "actor_context_evaluations": plan.context_calls,
        "reference_target_squared_error_integral": plan.reference_target_squared_error_integral,
        "reference_residual_squared_integral": plan.reference_residual_squared_integral,
        "velocity_residual_squared_integral": plan.velocity_residual_squared_integral,
        "proposed_action_rms": float(np.sqrt(np.square(np.asarray(plan.proposed_actions, dtype=np.float64)).mean()))}


def paired_endpoints(period, evaluation, seeds):
    if set(evaluation) != set(spec.MODES) or any([r["seed"] for r in rows] != seeds for rows in evaluation.values()):
        raise ValueError("Stage75 mode or seed roster changed")
    anchor = evaluation["R0V0"]
    for rows in evaluation.values():
        for row, base in zip(rows, anchor):
            if any(row[k] != base[k] for k in PAIR_KEYS):
                raise ValueError("Stage75 common-noise or upper-proposal pairing changed")
    effects = {}
    for metric in spec.METRICS:
        a, b, c, d = [np.asarray([r[metric] for r in evaluation[m]], dtype=np.float64) for m in spec.MODES]
        for name, values in zip(spec.CONTRASTS, (b-a, d-c, c-a, d-b, d-a, d-b-c+a)):
            if not np.isfinite(values).all():
                raise ValueError("Stage75 native endpoint is nonfinite")
            effects[f"{period}/{metric}/{name}"] = float(values.mean())
    return effects


def check_production(evaluation, production, seeds):
    if set(production) != {"R0V0", "R1V1"} or any([r["seed"] for r in rows] != seeds for rows in production.values()):
        raise ValueError("Stage75 preflight production replay roster changed")
    keys = (*PAIR_KEYS, "episode_return", "episode_length", "upper_calls", "lower_calls", *PLANNING_KEYS, "network_check")
    for mode, rows in production.items():
        for row, expected in zip(evaluation[mode], rows):
            if any(row[k] != expected[k] for k in keys):
                raise ValueError("Stage75 aligned path differs from original native execution")


def check_row(row, period, horizon):
    if (row["episode_length"] != horizon or row["lower_calls"] != horizon or row["upper_calls"] != horizon // period
            or row["decision_steps"] != list(range(0, horizon, period)) or row["network_check"] != "passed"
            or row["reference_evaluations"] != horizon or row["actor_context_evaluations"] != horizon
            or row["plan_ols_fits"] != horizon // period - 1 or row["plan_ridge_predictions"] != horizon // period - 1):
        raise ValueError("Stage75 native inference, planning or source weights changed")


def run(root, *, preflight, output):
    prerequisite = json.loads(spec.source_result(root, preflight=preflight).read_text())
    previous.qualify(prerequisite, preflight=preflight)
    if prerequisite["root"] != root:
        raise ValueError("Stage75 prerequisite root changed")
    clones, predictor, initialization = native.native.load_source(root, preflight=preflight)
    opt, args, roles = spec.options(preflight=preflight), spec.arguments(root, preflight=preflight), spec.seed_roles(root, preflight=preflight)
    cost, groups, started = dict.fromkeys(spec.budget(preflight=preflight), 0), {}, time.monotonic()
    cost.update(source_clone_loads=len(clones), forecaster_loads=1)
    planning = dict.fromkeys(PLANNING_KEYS, 0)
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"), initializer=init_worker,
            initargs=(clones[str(spec.PERIODS[0])].config, args)) as pool:
        for period in spec.PERIODS:
            clone = clones[str(period)]
            snapshot, weights = copy.deepcopy(clone.state_dict()), native.joint.inference_weights(clone)
            evaluation, production = {}, {}
            for mode in spec.MODES:
                evaluation[mode] = list(pool.map(worker_native, [(weights, s, mode, period, predictor) for s in roles["native_evaluation"]]))
            if preflight:
                for mode, arm in (("R0V0", "zero_train"), ("R1V1", "joint_ppo")):
                    production[mode] = list(pool.map(native.worker_native, [(weights, s, arm, period, predictor) for s in roles["native_evaluation"]]))
                check_production(evaluation, production, roles["native_evaluation"])
                cost["production_equivalence_checks"] += sum(map(len, production.values()))
            for rows in [*evaluation.values(), *production.values()]:
                for row in rows:
                    check_row(row, period, args.horizon)
                    cost["native_episodes"] += 1
                    cost["native_steps"] += row["episode_length"]
                    cost["native_lower_calls"] += row["lower_calls"]
                    cost["native_upper_calls"] += row["upper_calls"]
                    cost["native_network_checks"] += 1
                    for key in planning:planning[key] += row[key]
            effects = paired_endpoints(period, evaluation, roles["native_evaluation"])
            cost["native_pair_checks"] += len(roles["native_evaluation"])
            native.independent.assert_frozen(clone, snapshot)
            cost["frozen_model_checks"] += 1
            groups[str(period)] = {"evaluation": evaluation, "production_replays": production, "effects": effects,
                "pairing": "passed", "source_and_Adam_unchanged": "passed"}
            print(f"upper paths {root}/{period}: all four frozen modes complete", flush=True)
    cell = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "seed_roles": roles, "cost": cost, "native_planning_cost": planning,
        "groups": groups, "source_initialization": initialization, "optimizer_steps": 0, "critic_fits": 0,
        "forecaster_fits": 0, "checkpoint_writes": 0, "native_trace_writes": 0, "wall_seconds": time.monotonic() - started}
    qualify(cell, preflight=preflight)
    write_json(output, cell)
    write_json(output.parent / "completion" / "ready.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return cell


def qualify(cell, *, preflight):
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or cell["root"] not in spec.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["cost"] != spec.budget(preflight=preflight) or cell["seed_roles"] != spec.seed_roles(cell["root"], preflight=preflight)
            or any(cell[k] for k in ("optimizer_steps", "critic_fits", "forecaster_fits", "checkpoint_writes", "native_trace_writes"))
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}):
        raise ValueError("Stage75 frozen protocol or budget changed")
    h = spec.arguments(cell["root"], preflight=preflight).horizon
    planning = dict.fromkeys(PLANNING_KEYS, 0)
    for p, group in cell["groups"].items():
        if group["pairing"] != "passed" or group["source_and_Adam_unchanged"] != "passed":
            raise ValueError("Stage75 source or pairing changed")
        if group["effects"] != paired_endpoints(p, group["evaluation"], cell["seed_roles"]["native_evaluation"]):
            raise ValueError("Stage75 paired contrast accounting changed")
        if preflight:
            check_production(group["evaluation"], group["production_replays"], cell["seed_roles"]["native_evaluation"])
        elif group["production_replays"]:
            raise ValueError("Stage75 full run repeats settled production-equivalence checks")
        for mode, rows in group["evaluation"].items():
            if any(row["mode"] != mode for row in rows):raise ValueError("Stage75 intervention label changed")
        for rows in [*group["evaluation"].values(), *group["production_replays"].values()]:
            for row in rows:
                check_row(row, int(p), h)
                for key in planning:planning[key] += row[key]
    if cell["native_planning_cost"] != planning:
        raise ValueError("Stage75 planning accounting changed")
    return cell


def aggregate(cells, *, preflight):
    if len({c["root"] for c in cells}) != len(cells) or {c["root"] for c in cells} != set(spec.roots(preflight=preflight)):
        raise ValueError("Stage75 requires every frozen root")
    rows = [qualify(c, preflight=preflight) for c in sorted(cells, key=lambda c: c["root"])]
    effects = [{k: v for g in c["groups"].values() for k, v in g["effects"].items()} for c in rows]
    x = np.asarray([[e[k] for k in spec.ENDPOINTS] for e in effects])
    endpoints = {k: {"mean": float(x[:, i].mean())} for i, k in enumerate(spec.ENDPOINTS)}
    if not preflight:
        idx = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED)).integers(0, len(rows), (spec.BOOTSTRAP_DRAWS, len(rows)))
        tail = .05 / (2 * len(spec.ENDPOINTS))
        bounds = np.quantile(x[idx].mean(1), [tail, 1-tail], axis=0)
        for i, k in enumerate(spec.ENDPOINTS):
            endpoints[k].update(ci=bounds[:, i].tolist(), effect="positive" if bounds[0, i] > 0 else "negative" if bounds[1, i] < 0 else "inconclusive")
    return {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "mechanical_gate": "passed", "root_rows": rows, "endpoints": endpoints,
        "cost": {k: sum(c["cost"][k] for c in rows) for k in spec.budget(preflight=preflight)},
        "native_planning_cost": {k: sum(c["native_planning_cost"][k] for c in rows) for k in PLANNING_KEYS},
        "native_trial_prerequisite": "hold_Stage67_credit_gate_unchanged", "performance_claim": "none_path_diagnosis_only"}
