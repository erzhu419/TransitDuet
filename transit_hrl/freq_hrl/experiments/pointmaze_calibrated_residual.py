"""Shrink a coherent residual curve using historical cloned-command fidelity."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
import time
from types import SimpleNamespace

import numpy as np
import torch

from . import pointmaze_upper_paths as paths
from . import pointmaze_velocity_support as support
from .pointmaze_root_response import write_json
from scripts import pointmaze_calibrated_residual_stage77_spec as spec

init_worker = paths.init_worker


def blend(base, original, alpha):
    if alpha == 0.:return base.copy()
    if alpha == 1.:return original.copy()
    return (base.astype(np.float64) + alpha * (original.astype(np.float64) - base)).astype(np.float32)


def calibration_alpha(bc_mse, original_change_rms):
    return 1. if original_change_rms == 0. else min(1., float(np.sqrt(bc_mse) / original_change_rms))


class CalibratedPlan(paths.PathFactorPlan):
    def __init__(self, predictor, period, scale, alpha, envelope):
        super().__init__(predictor, period, scale, "R1V1")
        self.alpha, self.envelope = alpha, envelope
        self.speed_tail_rows = self.axis_outside_rows = 0

    def decode(self, **kwargs):
        before = self.executed_delta_squared_sum
        super().decode(**kwargs)
        self.points = blend(self.base_points, self.points, self.alpha)
        self.reference_points = self.velocity_points = self.points
        self.executed_delta_squared_sum = before + float(np.square(self.points.astype(np.float64) - self.base_points).sum())
        return self.points[0].copy()

    def actor_context(self, **kwargs):
        velocity = super().actor_context(**kwargs)
        self.speed_tail_rows += int(np.linalg.norm(velocity.astype(np.float64)) > self.envelope["velocity_speed_q99"])
        self.axis_outside_rows += int(np.any((velocity < self.envelope["axis_min"]) | (velocity > self.envelope["axis_max"])))
        return velocity


def historical_curves(raw, model, predictor, args, root, seed, period, bounds):
    torch.manual_seed(spec.source.counterfactual_seed(root, seed, period))
    std = model.upper_actor.log_std.detach().exp().clamp(1e-4, 3.)
    actions = torch.distributions.Normal(torch.zeros(args.horizon // period, 4), std).sample().numpy()
    plan = support.learned.ResidualPlan(predictor, period, args.maximum_subgoal_delta)
    base, original = [], []
    for i, start in enumerate(range(0, args.horizon, period)):
        prefix = raw["measurement"][max(0, start - 63):start + 1]
        plan.decode(action=actions[i], observation=SimpleNamespace(task_measurement=raw["measurement"][start]),
            history=SimpleNamespace(history=prefix.reshape(-1)), step=start, world_low=bounds[0], world_high=bounds[1])
        base.append(plan.base_points.copy());original.append(plan.points.copy())
    return np.asarray(base), np.asarray(original), len(actions)


def curve_states(states, achieved, points):
    changed = states.copy()
    changed[:, 4:6] = points[:, :-1].reshape(-1, 2) - achieved
    changed[:, -2:] = (np.diff(points, axis=1) / support.learned.forecast.spec.DT_SECONDS).reshape(-1, 2)
    return changed


def commands(actor, states):
    with torch.inference_mode():
        return np.concatenate([actor.distribution(torch.as_tensor(states[i:i + spec.CHUNK_SIZE])).mean.tanh().numpy()
            for i in range(0, len(states), spec.CHUNK_SIZE)])


def response(command, base_command, target):
    return {"bc_command_mse": float(np.square(command.astype(np.float64) - target).mean()),
        "command_change_rms": float(np.sqrt(np.square(command.astype(np.float64) - base_command).mean()))}


def prepare_calibration(root, period, *, model, predictor, args, roles, bounds, saved_bc_mse, cost, preflight):
    cache = []
    for seed in roles["calibration_labels"]:
        with np.load(spec.source.label_archive(root, period, seed, preflight=preflight)) as a:
            raw = {k: a[k] for k in support.RAW_KEYS}
        states = support.label_states(raw, args)
        base, original, n = historical_curves(raw, model, predictor, args, root, seed, period, bounds)
        np.testing.assert_array_equal(curve_states(states, raw["achieved_before"], base), states)
        base_command = commands(model.lower_actor, states)
        original_command = commands(model.lower_actor, curve_states(states, raw["achieved_before"], original))
        cache.append((states, raw["achieved_before"], raw["action"], base, original, base_command, original_command))
        for k in ("label_archive_loads", "label_plan_checks"):cost[k] += 1
        cost["label_state_rows"] += len(states)
        cost["calibration_curve_decodes"] += n
        cost["calibration_upper_proposals"] += n
    target = np.concatenate([c[2] for c in cache])
    base_command, original_command = [np.concatenate([c[i] for c in cache]) for i in (5, 6)]
    responses = {"zero": response(base_command, base_command, target),
        "original": response(original_command, base_command, target)}
    np.testing.assert_allclose(responses["zero"]["bc_command_mse"], saved_bc_mse, atol=1e-7, rtol=1e-5)
    cost["bc_mse_reproductions"] += 1
    cost["offline_actor_rows"] += 2 * len(target)
    cost["offline_actor_forward_batches"] += 2 * len(cache) * int(np.ceil(args.horizon / spec.CHUNK_SIZE))
    return cache, target, base_command, responses


def evaluate_curve(actor, cache, alpha, cost):
    command = np.concatenate([commands(actor, curve_states(s, a, blend(b, o, alpha)))
        for s, a, _, b, o, _, _ in cache])
    cost["offline_actor_rows"] += len(command)
    cost["offline_actor_forward_batches"] += sum(int(np.ceil(len(c[0]) / spec.CHUNK_SIZE)) for c in cache)
    return command


def calibrate(root, period, *, model, predictor, args, roles, bounds, old, cost, preflight):
    saved = old["bc_mse_reproduction"]["saved"]
    cache, target, base_command, responses = prepare_calibration(root, period, model=model, predictor=predictor,
        args=args, roles=roles, bounds=bounds, saved_bc_mse=saved, cost=cost, preflight=preflight)
    alpha = calibration_alpha(responses["zero"]["bc_command_mse"], responses["original"]["command_change_rms"])
    calibrated_command = evaluate_curve(model.lower_actor, cache, alpha, cost)
    responses["calibrated"] = response(calibrated_command, base_command, target)
    return {"alpha": alpha, "saved_bc_mse": saved, "responses": responses,
        "calibrated_to_bc_rmse_ratio": responses["calibrated"]["command_change_rms"] / np.sqrt(responses["zero"]["bc_command_mse"]),
        "data_role": "historical_BC_labels_only", "bc_mse_reproduction": "passed",
        "label_plan_checks": "passed", "envelope": old["calibration"]}


def worker_native(job):
    weights, seed, mode, period, predictor, alpha, envelope = job
    model, args = paths._WORKER
    model.load_state_dict(weights)
    policy_seed = paths.native.native.spec.policy_seed(args.optimizer_seed, seed)
    torch.manual_seed(policy_seed)
    plan = CalibratedPlan(predictor, period, args.maximum_subgoal_delta, alpha, envelope)
    kwargs = paths.native.native.spec.rollout_arguments(args.optimizer_seed, seed, phase="train", mode="training")
    kwargs["sample"] = False
    batch, row, raw = paths.native.joint.rollout(model, args, f"fixed{period}", seed=seed, capture=False,
        lower_credit="task_option", upper_plan_decoder=plan.decode, lower_reference_builder=plan,
        lower_actor_context_builder=plan.actor_context, lower_value_context_builder=plan.value_context, **kwargs)
    if batch is not None or raw is not None or not (row["upper_sample"] and row["lower_sample"]):
        raise ValueError("Stage77 changed native actor sampling or materialized training data")
    torch.testing.assert_close(paths.native.joint.inference_weights(model), weights, atol=0, rtol=0)
    return {"seed": seed, "mode": mode, "alpha": alpha,
        **{k: row[k] for k in (*spec.METRICS, "episode_length", "decision_steps", "lower_seed")},
        "upper_calls": row["upper_inference_calls"], "lower_calls": row["lower_inference_calls"],
        "policy_seed": policy_seed, "upper_proposed_actions": np.asarray(plan.proposed_actions).tolist(),
        "network_check": "passed", "plan_ols_fits": plan.ols_fits, "plan_ridge_predictions": plan.ridge_predictions,
        "reference_evaluations": plan.calls, "actor_context_evaluations": plan.context_calls,
        "reference_residual_squared_integral": plan.reference_residual_squared_integral,
        "velocity_residual_squared_integral": plan.velocity_residual_squared_integral,
        "above_label_speed_q99_rate": plan.speed_tail_rows / row["episode_length"],
        "outside_label_axis_range_rate": plan.axis_outside_rows / row["episode_length"]}


def paired_endpoints(period, evaluation, seeds, *, protocol=spec):
    if set(evaluation) != set(protocol.MODES) or any([r["seed"] for r in rows] != seeds for rows in evaluation.values()):
        raise ValueError("Stage77 native mode or seed roster changed")
    for rows in evaluation.values():
        for row, base in zip(rows, evaluation["zero"]):
            if any(row[k] != base[k] for k in paths.PAIR_KEYS):raise ValueError("Stage77 native common-noise pairing changed")
    effects = {}
    for metric in protocol.METRICS:
        values_by_mode = {m: np.asarray([r[metric] for r in evaluation[m]], dtype=np.float64) for m in protocol.MODES}
        for a, b in protocol.CONTRAST_PAIRS:
            name, values = f"{a}_minus_{b}", values_by_mode[a] - values_by_mode[b]
            if not np.isfinite(values).all():raise ValueError("Stage77 native endpoint is nonfinite")
            effects[f"{period}/{metric}/{name}"] = float(values.mean())
    return effects


def production_check(evaluation, replays, seeds):
    paths.check_production({"R0V0": evaluation["zero"], "R1V1": evaluation["original"]}, replays, seeds)


def check_calibration(c):
    if c["alpha"] != calibration_alpha(c["responses"]["zero"]["bc_command_mse"], c["responses"]["original"]["command_change_rms"]):
        raise ValueError("Stage77 historical calibration changed")


def run(root, *, preflight, output, protocol=spec, calibrator=calibrate,
        qualify_source=support.qualify, calibration_check=check_calibration):
    prerequisite = json.loads(protocol.source_result(root, preflight=preflight).read_text())
    source_preflight = protocol.source_preflight(preflight)
    qualify_source(prerequisite, preflight=source_preflight)
    if prerequisite["root"] != root:raise ValueError("Stage77 prerequisite root changed")
    clones, predictor, initialization = support.native.load_source(root, preflight=source_preflight)
    args, roles = protocol.arguments(root, preflight=preflight), protocol.seed_roles(root, preflight=preflight)
    cost, groups, started = dict.fromkeys(protocol.budget(preflight=preflight), 0), {}, time.monotonic()
    cost.update(source_clone_loads=len(clones), forecaster_loads=1, layout_loads=1)
    bounds = support.layout_bounds(args, roles["calibration_labels"][0])
    planning = dict.fromkeys(paths.PLANNING_KEYS, 0)
    with ProcessPoolExecutor(max_workers=protocol.options(preflight=preflight)["workers"], mp_context=mp.get_context("spawn"),
            initializer=init_worker, initargs=(clones[str(protocol.PERIODS[0])].config, args)) as pool:
        for period in protocol.PERIODS:
            clone = clones[str(period)]
            snapshot, weights = copy.deepcopy(clone.state_dict()), paths.native.joint.inference_weights(clone)
            calibration = calibrator(root, period, model=clone, predictor=predictor, args=args, roles=roles,
                bounds=bounds, old=prerequisite["groups"][str(period)], cost=cost, preflight=preflight)
            print(f"calibration {root}/{period}: alpha={calibration['alpha']:.8f} frozen before native probes", flush=True)
            alphas = protocol.mode_alphas(calibration)
            evaluation = {m: list(pool.map(worker_native, [(weights, s, m, period, predictor, alphas[m], calibration["envelope"])
                for s in roles["native_evaluation"]])) for m in protocol.MODES}
            replays = {}
            if protocol.production_replay_count(preflight):
                replays = {m: list(pool.map(paths.native.worker_native, [(weights, s, arm, period, predictor)
                    for s in roles["native_evaluation"]])) for m, arm in (("R0V0", "zero_train"), ("R1V1", "joint_ppo"))}
                production_check(evaluation, replays, roles["native_evaluation"])
                cost["production_equivalence_checks"] += sum(map(len, replays.values()))
            for rows in [*evaluation.values(), *replays.values()]:
                for row in rows:
                    paths.check_row(row, period, args.horizon)
                    cost["native_episodes"] += 1
                    cost["native_steps"] += row["episode_length"]
                    cost["native_lower_calls"] += row["lower_calls"]
                    cost["native_upper_calls"] += row["upper_calls"]
                    cost["native_network_checks"] += 1
                    for k in planning:planning[k] += row[k]
            effects = paired_endpoints(period, evaluation, roles["native_evaluation"], protocol=protocol)
            cost["native_pair_checks"] += len(roles["native_evaluation"])
            support.assert_frozen(clone, snapshot)
            cost["frozen_model_checks"] += 1
            groups[str(period)] = {"calibration": calibration, "evaluation": evaluation, "production_replays": replays,
                "effects": effects, "pairing": "passed", "source_and_Adam_unchanged": "passed"}
    cell = {"status": "complete", "protocol": protocol.EXPERIMENT_PROTOCOL, "contract": protocol.contract(), "root": root,
        "preflight": preflight, "seed_roles": roles, "cost": cost, "native_planning_cost": planning, "groups": groups,
        "source_initialization": initialization, "optimizer_steps": 0, "critic_fits": 0, "forecaster_fits": 0,
        "checkpoint_writes": 0, "native_trace_writes": 0, "wall_seconds": time.monotonic() - started}
    qualify(cell, preflight=preflight, protocol=protocol, calibration_check=calibration_check)
    write_json(output, cell)
    write_json(output.parent / "completion" / "ready.json", {"protocol": protocol.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return cell


def qualify(cell, *, preflight, protocol=spec, calibration_check=check_calibration):
    if (cell["status"] != "complete" or cell["protocol"] != protocol.EXPERIMENT_PROTOCOL or cell["contract"] != protocol.contract()
            or cell["root"] not in protocol.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["cost"] != protocol.realized_budget(cell, preflight=preflight) or cell["seed_roles"] != protocol.seed_roles(cell["root"], preflight=preflight)
            or any(cell[k] for k in ("optimizer_steps", "critic_fits", "forecaster_fits", "checkpoint_writes", "native_trace_writes"))
            or set(cell["groups"]) != {str(p) for p in protocol.PERIODS}):
        raise ValueError("Stage77 frozen protocol or budget changed")
    h = protocol.arguments(cell["root"], preflight=preflight).horizon
    planning = dict.fromkeys(paths.PLANNING_KEYS, 0)
    for p, g in cell["groups"].items():
        c = g["calibration"]
        if (c["data_role"] != "historical_BC_labels_only" or c["bc_mse_reproduction"] != "passed"
                or c["label_plan_checks"] != "passed" or g["source_and_Adam_unchanged"] != "passed" or g["pairing"] != "passed"):
            raise ValueError("Stage77 historical calibration or source model changed")
        np.testing.assert_allclose(c["responses"]["zero"]["bc_command_mse"], c["saved_bc_mse"], atol=1e-7, rtol=1e-5)
        calibration_check(c)
        if g["effects"] != paired_endpoints(p, g["evaluation"], cell["seed_roles"]["native_evaluation"], protocol=protocol):
            raise ValueError("Stage77 paired native accounting changed")
        if protocol.production_replay_count(preflight):production_check(g["evaluation"], g["production_replays"], cell["seed_roles"]["native_evaluation"])
        elif g["production_replays"]:raise ValueError("Stage77 full run repeats settled production checks")
        for mode, rows in g["evaluation"].items():
            alpha = protocol.mode_alphas(c)[mode]
            if any(r["mode"] != mode or r["alpha"] != alpha for r in rows):raise ValueError("Stage77 execution scaling changed")
        for rows in [*g["evaluation"].values(), *g["production_replays"].values()]:
            for row in rows:
                paths.check_row(row, int(p), h)
                for k in planning:planning[k] += row[k]
    if planning != cell["native_planning_cost"]:raise ValueError("Stage77 native planning accounting changed")
    return cell


def aggregate(cells, *, preflight, protocol=spec, calibration_check=check_calibration):
    if len({c["root"] for c in cells}) != len(cells) or {c["root"] for c in cells} != set(protocol.roots(preflight=preflight)):
        raise ValueError("Stage77 requires every frozen root")
    rows = [qualify(c, preflight=preflight, protocol=protocol, calibration_check=calibration_check) for c in sorted(cells, key=lambda c: c["root"])]
    effects = [{k: v for g in c["groups"].values() for k, v in g["effects"].items()} for c in rows]
    x = np.asarray([[e[k] for k in protocol.ENDPOINTS] for e in effects])
    endpoints = {k: {"mean": float(x[:, i].mean())} for i, k in enumerate(protocol.ENDPOINTS)}
    if not preflight:
        idx = np.random.default_rng(np.random.SeedSequence(protocol.BOOTSTRAP_SEED)).integers(0, len(rows), (protocol.BOOTSTRAP_DRAWS, len(rows)))
        tail = .05 / (2 * len(protocol.ENDPOINTS))
        bounds = np.quantile(x[idx].mean(1), [tail, 1-tail], axis=0)
        for i, k in enumerate(protocol.ENDPOINTS):
            endpoints[k].update(ci=bounds[:, i].tolist(), effect="positive" if bounds[0, i] > 0 else "negative" if bounds[1, i] < 0 else "inconclusive")
    return {"status": "preflight_passed" if preflight else "complete", "protocol": protocol.EXPERIMENT_PROTOCOL,
        "contract": protocol.contract(), "mechanical_gate": "passed", "root_rows": rows, "endpoints": endpoints,
        "cost": {k: sum(c["cost"][k] for c in rows) for k in protocol.budget(preflight=preflight)},
        "native_planning_cost": {k: sum(c["native_planning_cost"][k] for c in rows) for k in paths.PLANNING_KEYS},
        "native_trial_prerequisite": "hold_Stage67_credit_gate_unchanged", "performance_claim": "frozen_decoder_repair_validation_only"}
