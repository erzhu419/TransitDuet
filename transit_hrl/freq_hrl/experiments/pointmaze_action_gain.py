"""Causal closed-loop upper-coordinate interventions with a frozen lower."""
from concurrent.futures import ProcessPoolExecutor
import copy
import multiprocessing as mp
import time

import numpy as np
import torch
from . import pointmaze_crossed_advice as crossed
from . import pointmaze_feasible_credit as statistics
from .pointmaze_root_response import write_json
from scripts import pointmaze_action_gain_stage110_spec as spec

source = crossed.source


def offset_action(action, variant):
    result = np.asarray(action, dtype=np.float32).copy()
    if variant in spec.DIRECTIONS:
        coordinate = int(variant[4])
        result[coordinate] += spec.EPSILON if variant.endswith("plus") else -spec.EPSILON
    return result


def episode(weights, *, seed, variant, period, predictor, envelope):
    model, args = source.native._WORKER
    model.load_state_dict(weights)
    policy_seed, lower_seed = source.scenario.spec.noise_seeds(args.optimizer_seed, seed, seed)
    torch.manual_seed(policy_seed)
    planned, learned = variant != "blind", variant == "mean" or variant in spec.DIRECTIONS
    plan = (source.native.curves.CalibratedPlan(predictor, period, args.maximum_subgoal_delta, spec.ALPHA, envelope)
        if variant == "zero" or learned else source.baseline.forecast.PlanReference("ridge_velocity", predictor, period)
        if variant == "forecast" else None)
    scale = source.native.joint.scale_for(args)
    task = source.native.joint._make_task(env_id=args.env_id, seed=seed, horizon=args.horizon, **source.native.joint._task_options(args))
    try:
        obs = task.reset()
        low, high = source.native.joint.pointmaze_goal_bounds(task.environment)
        history = source.native.joint.PointMazeRegimeFeatureBuilder(time_scale=scale)
        history.reset(obs)
        model.reset_recurrent_inference()
        with torch.no_grad():
            std = model.lower_actor.log_std.exp().clamp(1e-4, 3.).numpy()
        rewards, distances, innovations, measurements, commands, decisions = [], [], [], [], [], []
        initial, action_square = None, 0.
        for step in range(args.horizon):
            feedback = source.baseline.flat_state(history, obs)
            if initial is None:
                initial = feedback.copy()
            measurements.append(obs.task_measurement.copy())
            if (learned or variant == "zero") and step%period == 0:
                latent = (offset_action(model.act_upper(history.upper_state(obs, oracle_context=None), sample=False)["action"], variant)
                    if learned else np.zeros(4, dtype=np.float32))
                action_square += float(np.square(latent.astype(np.float64)).sum())
                plan.decode(action=latent, observation=obs, history=history, step=step, world_low=low, world_high=high)
                if learned:
                    decisions.append(step)
            state = (source.advice_state(feedback, obs,
                plan(observation=obs, history=history, subgoal=None, age=step%period, step=step, world_low=low, world_high=high),
                plan.actor_context(age=step%period, step=step, horizon=args.horizon)) if planned
                else np.r_[feedback, np.zeros(4, dtype=np.float32)])
            value_state = np.r_[state, source.baseline.clocks.time_context(age=step%period, step=step, horizon=args.horizon, clock=True)].astype(np.float32)
            torch.manual_seed(lower_seed+step)
            output = model.act_lower(state, sample=True, value_state=value_state)
            innovations.append((np.asarray(output["action"])-np.asarray(output["mean_action"]))/std)
            command = source.native.joint.squash_box_action(np.asarray(output["action"], dtype=np.float32), task.action_low, task.action_high)
            commands.append(command.copy())
            obs, reward, terminated, truncated, info = task.step(command)
            if bool(terminated or truncated) and step+1 != args.horizon:
                raise RuntimeError("Directional gain requires the registered full native horizon")
            rewards.append(float(reward))
            distances.append(float(info["tracking_distance"]))
            history.update(obs)
        torch.testing.assert_close(source.native.joint.inference_weights(model), weights, atol=0, rtol=0)
        row = {"seed": seed, "variant": variant, "policy_seed": policy_seed, "lower_seed": lower_seed,
            "episode_return": float(np.sum(rewards)), "tracking_squared_error_integral": float(np.dot(distances, distances)*scale.dt_seconds),
            "episode_length": args.horizon, "upper_calls": len(decisions), "lower_calls": args.horizon,
            "decision_steps": decisions, "upper_sample": False, "lower_sample": True, "alpha": spec.ALPHA,
            "upper_action_rms": float(np.sqrt(action_square/(4*(args.horizon//period)))) if learned else 0.,
            "network_check": "passed", "plan_renewals": args.horizon//period if planned else 0,
            **{k: 0 if plan is None else getattr(plan, attr) for k, attr in (("plan_ols_fits", "ols_fits"),
                ("plan_ridge_predictions", "ridge_predictions"), ("reference_evaluations", "calls"), ("actor_context_evaluations", "context_calls"))}}
        return row, {"initial_feedback": initial, "measurements": np.asarray(measurements),
            "innovations": np.asarray(innovations), "commands": np.asarray(commands)}
    finally:
        task.environment.close()


def worker_pair(job):
    weights, seed, period, predictor, envelope = job
    rows, base, largest = {}, None, 0.
    for variant in spec.VARIANTS:
        row, audit = episode(weights, seed=seed, variant=variant, period=period, predictor=predictor, envelope=envelope)
        if base is None:
            base = audit
        else:
            np.testing.assert_array_equal(audit["initial_feedback"], base["initial_feedback"])
            np.testing.assert_array_equal(audit["measurements"], base["measurements"])
            np.testing.assert_allclose(audit["innovations"], base["innovations"], atol=3e-5, rtol=0)
            largest = max(largest, float(np.max(np.abs(audit["innovations"]-base["innovations"]))))
        if variant == "zero":
            np.testing.assert_array_equal(audit["commands"], base["commands"])
            for key in ("episode_return", "tracking_squared_error_integral"):
                if row[key] != rows["forecast"][key]:
                    raise ValueError("Zero residual changed the forecast closed loop")
        rows[variant] = row
    return {"seed": seed, "evaluation": rows, "pair_checks": "passed", "lower_innovation_max_error": largest}


def check_row(row, *, root, period, horizon, variant):
    planned, learned = variant != "blind", variant == "mean" or variant in spec.DIRECTIONS
    expected = {"variant": variant, "episode_length": horizon, "upper_calls": horizon//period if learned else 0,
        "lower_calls": horizon, "decision_steps": list(range(0, horizon, period)) if learned else [],
        "upper_sample": False, "lower_sample": True, "alpha": spec.ALPHA, "network_check": "passed",
        "plan_renewals": horizon//period if planned else 0, "plan_ols_fits": horizon//period-1 if planned else 0,
        "plan_ridge_predictions": horizon//period-1 if planned else 0, "reference_evaluations": horizon if planned else 0,
        "actor_context_evaluations": horizon if planned else 0}
    noise = source.scenario.spec.noise_seeds(root, row["seed"], row["seed"])
    if any(row[k] != v for k, v in expected.items()) or (row["policy_seed"], row["lower_seed"]) != noise:
        raise ValueError("Directional native feedback, sampling, decoder, noise or call schedule changed")


def effects(period, rows):
    if any(set(pair["evaluation"]) != set(spec.VARIANTS) for pair in rows):
        raise ValueError("Directional intervention roster changed")
    r = {v: np.asarray([pair["evaluation"][v]["episode_return"] for pair in rows]) for v in spec.VARIANTS}
    values = {"mean_minus_forecast": r["mean"]-r["forecast"], "forecast_minus_blind": r["forecast"]-r["blind"]}
    values.update({v+"_minus_forecast": r[v]-r["forecast"] for v in spec.DIRECTIONS})
    for i in range(4):
        plus, minus = r[f"axis{i}_plus"], r[f"axis{i}_minus"]
        values[f"axis{i}_slope"] = (plus-minus)/(2*spec.EPSILON)
        values[f"axis{i}_curvature"] = (plus+minus-2*r["mean"])/spec.EPSILON**2
    return {f"{period}/{k}": float(v.mean()) for k, v in values.items()}


def run(root, *, preflight, output):
    models, predictor, record, calibrations = crossed.load_source(root)
    args, roles, o = spec.arguments(root, preflight=preflight), spec.seed_roles(root, preflight=preflight), spec.options(preflight=preflight)
    started, groups, cost = time.monotonic(), {}, dict.fromkeys(spec.budget(preflight=preflight), 0)
    with ProcessPoolExecutor(max_workers=o["workers"], mp_context=mp.get_context("spawn"), initializer=source.native.init_worker,
            initargs=(models["50"]["learned_hint"].config, args)) as pool:
        for period in spec.PERIODS:
            model = models[str(period)]["learned_hint"]
            before, panels = copy.deepcopy(model.state_dict()), {}
            weights = source.native.joint.inference_weights(model)
            for panel, seeds in roles["panels"].items():
                pairs = list(pool.map(worker_pair, [(weights, s, period, predictor, calibrations[str(period)]["envelope"]) for s in seeds]))
                panels[panel] = {"pairs": pairs, "effects": effects(period, pairs)}
                for pair in pairs:
                    cost["native_pair_groups"] += 1
                    cost["zero_forecast_command_checks"] += 1
                    for k in ("initial_feedback_pair_checks", "external_measurement_pair_checks", "lower_innovation_pair_checks"):
                        cost[k] += len(spec.VARIANTS)-1
                    for variant, row in pair["evaluation"].items():
                        check_row(row, root=root, period=period, horizon=args.horizon, variant=variant)
                        for k, v in (("native_episodes", 1), ("native_steps", row["episode_length"]), ("native_lower_calls", row["lower_calls"]),
                                ("native_upper_calls", row["upper_calls"]), ("native_network_checks", 1)):
                            cost[k] += v
                        for k in ("plan_renewals", "plan_ols_fits", "plan_ridge_predictions", "reference_evaluations", "actor_context_evaluations"):
                            cost[k] += row[k]
                print(f"{spec.EXPERIMENT_PROTOCOL} {root}/{period}/{panel}: all12 paired interventions complete", flush=True)
            source.native.curves.support.assert_frozen(model, before)
            cost["source_freeze_checks"] += 1
            groups[str(period)] = {"panels": panels, "effects": effects(period, [pair for g in panels.values() for pair in g["pairs"]]),
                "task_options": spec.task_options(root, preflight=preflight), "source_freeze": "passed"}
    cell = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "seed_roles": roles, "source_record": record, "cost": cost, "groups": groups,
        "source_loads": {"donor_checkpoints": 8, "source_cells": 2, "source_clones": 2, "forecasters": 1,
            "decoders": 2, "loaded_advice_models": 4, "executed_advice_models": 2},
        "policy_updates": 0, "critic_fits": 0, "checkpoint_writes": 0, "native_trace_writes": 0, "wall_seconds": time.monotonic()-started}
    qualify(cell, preflight=preflight)
    write_json(output, cell)
    write_json(output.parent/"completion"/"ready.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return cell


def qualify(cell, *, preflight):
    root, h = cell["root"], spec.arguments(cell["root"], preflight=preflight).horizon
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or root not in spec.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["source_record"] != spec.source.source.source_record(root) or cell["seed_roles"] != spec.seed_roles(root, preflight=preflight)
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS} or cell["cost"] != spec.budget(preflight=preflight)
            or any(cell[k] for k in ("policy_updates", "critic_fits", "checkpoint_writes", "native_trace_writes"))):
        raise ValueError("Directional gain donor, protocol, roster, budget or freeze changed")
    for p, group in cell["groups"].items():
        if group["source_freeze"] != "passed" or group["task_options"] != spec.task_options(root, preflight=preflight) or set(group["panels"]) != set(spec.PANELS):
            raise ValueError("Directional source freeze, task or A/B panel changed")
        for name, panel in group["panels"].items():
            if [pair["seed"] for pair in panel["pairs"]] != cell["seed_roles"]["panels"][name] or panel["effects"] != effects(p, panel["pairs"]):
                raise ValueError("Directional independent panel or credit identity changed")
            for pair in panel["pairs"]:
                if pair["pair_checks"] != "passed" or not 0 <= pair["lower_innovation_max_error"] <= 3e-5:
                    raise ValueError("Directional pairing changed exogenous history or lower noise")
                for variant, row in pair["evaluation"].items():
                    if row["seed"] != pair["seed"]:
                        raise ValueError("Directional scenario mapping changed")
                    check_row(row, root=root, period=int(p), horizon=h, variant=variant)
                forecast, zero = pair["evaluation"]["forecast"], pair["evaluation"]["zero"]
                if any(forecast[k] != zero[k] for k in ("episode_return", "tracking_squared_error_integral")):
                    raise ValueError("Zero action no longer reproduces the forecast control loop")
        pooled = effects(p, [pair for panel in group["panels"].values() for pair in panel["pairs"]])
        if group["effects"] != pooled or not np.isfinite(list(pooled.values())).all():
            raise ValueError("Directional pooled effect or panel weights changed")
    return cell


def aggregate(cells, *, preflight):
    result = statistics.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    panel_means = {name: {k: float(np.mean([g["panels"][name]["effects"][k] for c in cells for g in c["groups"].values() if k in g["effects"]]))
        for k in spec.ENDPOINTS} for name in spec.PANELS}
    eligible = {str(p): [] for p in spec.PERIODS}
    if not preflight:
        for p in spec.PERIODS:
            for variant in spec.DIRECTIONS:
                key, slope = f"{p}/{variant}_minus_forecast", f"{p}/{variant[:5]}_slope"
                sign = 1 if variant.endswith("plus") else -1
                ci = result["endpoints"][slope]["ci"]
                if (result["endpoints"][key]["ci"][0] > 0 and (ci[0] > 0 if sign > 0 else ci[1] < 0)
                        and all(sign*panel_means[name][slope] > 0 for name in spec.PANELS)):
                    eligible[str(p)].append(variant)
    result.update(equal_root_panel_effects=panel_means, eligible_diagnostic_directions=eligible,
        closed_loop_gain="mechanical_only" if preflight else "detected_both_periods" if all(eligible.values()) else "partial" if any(eligible.values()) else "not_supported",
        performance_claim="frozen_whole_episode_bias_direction_gain_not_a_learned_HRL_policy_or_local_Q",
        native_trial_prerequisite="Stage67_critic_HOLD_unchanged")
    return result
