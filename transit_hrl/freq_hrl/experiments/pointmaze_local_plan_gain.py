"""Measure one-option plan gain without bypassing the frozen lower policy."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
import time

import numpy as np
import torch

from . import pointmaze_upper_residual_train as base
from . import pointmaze_upper_wide_plan_train as wide
from . import pointmaze_feasible_credit as statistics
from .pointmaze_root_response import write_json
from scripts import pointmaze_local_plan_gain_stage117_spec as spec


def intervention_action(variant):
    action = np.zeros(spec.ACTION_DIM, dtype=np.float64)
    if variant in spec.DIRECTIONS:
        action[int(variant[4])] = spec.EPSILON if variant.endswith("plus") else -spec.EPSILON
    elif variant not in ("zero", "forecast"):
        raise ValueError("unknown local plan intervention")
    return action


def episode(weights, lower_state, *, query, panel, variant, period, predictor, calibration):
    model, args = base.source.native._WORKER
    model.load_state_dict(weights)
    lower = base.lower_branch(model, lower_state)
    start = query["start"]
    if start % period or not 0 < start < args.horizon - period:
        raise ValueError("local plan probe requires an interior complete option and continuation")
    prefix_seed = base.source.scenario.spec.noise_seeds(
        args.optimizer_seed, query["scenario_seed"], query["prefix_noise_seed"])[1]
    suffix_seed = base.source.scenario.spec.noise_seeds(
        args.optimizer_seed, query["scenario_seed"], query["suffix_noise_seeds"][panel])[1]
    plan = (base.source.baseline.forecast.PlanReference("ridge_velocity", predictor, period)
        if variant == "forecast" else wide.WideBernsteinPlan(
            predictor, period, args.maximum_subgoal_delta, calibration["envelope"]))
    scale = base.source.native.joint.scale_for(args)
    task = base.source.native.joint._make_task(env_id=args.env_id, seed=query["scenario_seed"],
        horizon=args.horizon, **base.source.native.joint._task_options(args))
    rewards, distances, commands, means, measurements, innovations = [], [], [], [], [], []
    query_state = None
    action_count, position_square, velocity_square, clipped_points = 0, 0., 0., 0
    try:
        observation = task.reset()
        low, high = base.source.native.joint.pointmaze_goal_bounds(task.environment)
        history = base.source.native.joint.PointMazeRegimeFeatureBuilder(time_scale=scale)
        history.reset(observation)
        for step in range(args.horizon):
            feedback = base.source.baseline.flat_state(history, observation)
            measurements.append(observation.task_measurement.copy())
            if step == start:
                query_state = np.r_[feedback, observation.achieved_goal].copy()
            if variant != "forecast" and step % period == 0:
                action = intervention_action(variant) if step == start else np.zeros(spec.ACTION_DIM)
                action_count += int(np.any(action))
                plan.decode(action=action, observation=observation, history=history, step=step,
                    world_low=low, world_high=high)
                if step == start:
                    delta = plan.points.astype(np.float64) - plan.base_points
                    position_square = float(np.square(delta).sum())
                    velocity_square = float(np.square(np.diff(delta, axis=0) / scale.dt_seconds).sum())
                    clipped_points = int(np.sum((plan.points <= low) | (plan.points >= high)))
            reference = plan(observation=observation, history=history, subgoal=None, age=step % period,
                step=step, world_low=low, world_high=high)
            velocity = plan.actor_context(age=step % period, step=step, horizon=args.horizon)
            state = base.source.advice_state(feedback, observation, reference, velocity)
            with torch.inference_mode():
                torch.manual_seed((prefix_seed if step < start else suffix_seed) + step)
                distribution = lower.distribution(torch.as_tensor(state).view(1, -1))
                raw = distribution.rsample()[0]
                means.append(distribution.mean[0].numpy().copy())
                innovations.append(((raw - distribution.mean[0]) / distribution.stddev[0]).numpy())
                command = base.source.native.joint.squash_box_action(raw.numpy(), task.action_low, task.action_high)
            commands.append(command.copy())
            observation, reward, terminated, truncated, info = task.step(command)
            if bool(terminated or truncated) and step + 1 != args.horizon:
                raise RuntimeError("local plan probe ended before the registered horizon")
            rewards.append(float(reward))
            distances.append(float(info["tracking_distance"]))
            history.update(observation)
        reward = np.asarray(rewards, dtype=np.float64)
        row = {"variant": variant, "panel": panel, "scenario_seed": query["scenario_seed"], "start": start,
            "prefix_seed": prefix_seed, "suffix_seed": suffix_seed, "episode_length": args.horizon,
            "episode_return": float(reward.sum()), "prefix_return": float(reward[:start].sum()),
            "option_return": float(reward[start:start + period].sum()),
            "tail_return": float(reward[start + period:].sum()), "suffix_return": float(reward[start:].sum()),
            "tracking_squared_error_integral": float(np.dot(distances, distances) * scale.dt_seconds),
            "upper_actor_calls": 0, "lower_calls": args.horizon, "intervention_decisions": action_count,
            "plan_renewals": args.horizon // period, "plan_fits": plan.ols_fits,
            "reference_evaluations": plan.calls, "actor_context_evaluations": plan.context_calls,
            "plan_position_delta_rms": float(np.sqrt(position_square / (2 * (period + 1)))),
            "plan_velocity_delta_rms": float(np.sqrt(velocity_square / (2 * period))),
            "plan_bound_point_fraction": clipped_points / (2 * (period + 1))}
        audit = {"query_state": query_state, "prefix_rewards": reward[:start],
            "rewards": reward, "commands": np.asarray(commands), "means": np.asarray(means),
            "measurements": np.asarray(measurements), "innovations": np.asarray(innovations)}
        return row, audit
    finally:
        task.environment.close()


def gradients(panels, key="suffix_return"):
    return {name: [(rows[f"axis{i}_plus"][key] - rows[f"axis{i}_minus"][key]) / (2 * spec.EPSILON)
        for i in range(spec.ACTION_DIM)] for name, rows in panels.items()}


def crossfit_gain(panels, selection_key):
    candidates = ("zero", *spec.DIRECTIONS)
    selected, gains = {}, []
    for fit, test in (("A", "B"), ("B", "A")):
        choice = max(candidates, key=lambda variant: panels[fit][variant][selection_key])
        selected[fit] = choice
        gains.append(panels[test][choice]["suffix_return"] - panels[test]["zero"]["suffix_return"])
    return float(np.mean(gains)), selected


def effects(period, queries):
    a, b = (np.asarray([q["gradients"][name] for q in queries], dtype=np.float64).ravel()
            for name in spec.PANELS)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    suffix, option = zip(*[(crossfit_gain(q["panels"], "suffix_return")[0],
                           crossfit_gain(q["panels"], "option_return")[0]) for q in queries])
    return {f"{period}/crossfit_suffix_gain": float(np.mean(suffix)),
        f"{period}/suffix_gradient_dot": float(np.mean(a * b)),
        f"{period}/suffix_gradient_cosine": float(a @ b / norm) if norm else 0.,
        f"{period}/suffix_minus_option_selection": float(np.mean(np.asarray(suffix) - option))}


def worker_query(job):
    weights, lower_state, query, period, predictor, calibration = job
    panels, common, largest = {}, None, 0.
    for panel in spec.PANELS:
        rows, reference = {}, None
        for variant in spec.VARIANTS:
            row, audit = episode(weights, lower_state, query=query, panel=panel, variant=variant,
                period=period, predictor=predictor, calibration=calibration)
            if common is None:
                common = audit
            else:
                for key in ("query_state", "prefix_rewards", "measurements"):
                    np.testing.assert_array_equal(audit[key], common[key])
                np.testing.assert_array_equal(audit["commands"][:query["start"]], common["commands"][:query["start"]])
            if reference is None:
                reference = audit
            else:
                error = float(np.max(np.abs(audit["innovations"] - reference["innovations"])))
                largest = max(largest, error)
                np.testing.assert_allclose(audit["innovations"], reference["innovations"], atol=3e-5, rtol=0)
                np.testing.assert_allclose(row["episode_return"] - rows["forecast"]["episode_return"],
                    row["suffix_return"] - rows["forecast"]["suffix_return"], atol=1e-9, rtol=0)
            if variant == "zero":
                for key in ("commands", "means", "rewards"):
                    np.testing.assert_array_equal(audit[key], reference[key])
            start, stop = query["start"], query["start"] + period
            row.update(query_mean_delta_rms=float(np.sqrt(np.square(audit["means"][start] - reference["means"][start]).mean())),
                option_command_delta_rms=float(np.sqrt(np.square(audit["commands"][start:stop] - reference["commands"][start:stop]).mean())),
                tail_command_delta_rms=float(np.sqrt(np.square(audit["commands"][stop:] - reference["commands"][stop:]).mean())))
            rows[variant] = row
        panels[panel] = rows
    model, _ = base.source.native._WORKER
    torch.testing.assert_close(base.source.native.joint.inference_weights(model), weights, atol=0, rtol=0)
    return {"query": query, "panels": panels, "gradients": gradients(panels),
        "option_gradients": gradients(panels, "option_return"), "pairing": "passed",
        "source_freeze": "passed", "innovation_max_error": largest}


def run(root, *, preflight, output):
    base.source.qualify(json.loads(spec.source_result(root).read_text()), preflight=False)
    models, predictor, _, calibrations = base.source.load_source(root)
    args, roles, options = spec.arguments(root, preflight=preflight), spec.seed_roles(root, preflight=preflight), spec.options(preflight=preflight)
    cost = dict.fromkeys(spec.budget(preflight=preflight), 0)
    cost.update(source_cell_loads=1, source_clone_loads=len(models), lower_checkpoint_loads=len(models))
    groups, started = {}, time.monotonic()
    with ProcessPoolExecutor(max_workers=options["workers"], mp_context=mp.get_context("spawn"),
            initializer=base.source.native.init_worker, initargs=(models["50"].config, args)) as pool:
        for period in spec.PERIODS:
            model = models[str(period)]
            before = copy.deepcopy(model.state_dict())
            lower_state = base.load_lower_state(root, period, protocol=spec)
            weights = base.source.native.joint.inference_weights(model)
            queries = list(pool.map(worker_query, [(weights, lower_state, q, period, predictor, calibrations[str(period)])
                for q in roles["queries"]]))
            for query in queries:
                cost["native_pair_groups"] += 1
                cost["prefix_pair_checks"] += len(spec.PANELS) * len(spec.VARIANTS) - 1
                cost["external_pair_checks"] += len(spec.PANELS) * len(spec.VARIANTS) - 1
                cost["innovation_pair_checks"] += len(spec.PANELS) * (len(spec.VARIANTS) - 1)
                cost["zero_forecast_checks"] += len(spec.PANELS)
                for rows in query["panels"].values():
                    for row in rows.values():
                        cost["native_episodes"] += 1
                        cost["native_steps"] += row["episode_length"]
                        cost["native_lower_calls"] += row["lower_calls"]
                        cost["suffix_credit_checks"] += 1
                        for key in ("plan_renewals", "plan_fits", "reference_evaluations", "actor_context_evaluations"):
                            cost[key] += row[key]
            base.source.native.curves.support.assert_frozen(model, before)
            cost["frozen_source_checks"] += 1
            groups[str(period)] = {"queries": queries, "effects": effects(period, queries)}
            print(f"Stage117 root{root} period{period}: {len(queries)} paired local-plan queries complete", flush=True)
    cell = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "root": root, "preflight": preflight, "seed_roles": roles, "groups": groups, "cost": cost,
        "policy_updates": 0, "critic_fits": 0, "checkpoint_writes": 0, "native_trace_writes": 0,
        "wall_seconds": time.monotonic() - started}
    qualify(cell, preflight=preflight)
    write_json(output, cell)
    write_json(output.parent / "completion" / "ready.json",
        {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return cell


def qualify(cell, *, preflight):
    root = cell["root"]
    if (root not in spec.roots(preflight=preflight) or cell["status"] != "complete"
            or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or cell["preflight"] != preflight or cell["seed_roles"] != spec.seed_roles(root, preflight=preflight)
            or cell["cost"] != spec.budget(preflight=preflight)
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}
            or any(cell[k] for k in ("policy_updates", "critic_fits", "checkpoint_writes", "native_trace_writes"))):
        raise ValueError("Stage117 frozen source, roster or budget changed")
    horizon = spec.arguments(root, preflight=preflight).horizon
    for period, group in cell["groups"].items():
        period = int(period)
        if [q["query"] for q in group["queries"]] != cell["seed_roles"]["queries"]:
            raise ValueError("Stage117 query roster changed")
        for q in group["queries"]:
            if q["pairing"] != "passed" or q["source_freeze"] != "passed" or not 0 <= q["innovation_max_error"] <= 3e-5:
                raise ValueError("Stage117 prefix, external truth or noise pairing failed")
            if set(q["panels"]) != set(spec.PANELS):
                raise ValueError("Stage117 independent noise panels changed")
            for panel, rows in q["panels"].items():
                if set(rows) != set(spec.VARIANTS):
                    raise ValueError("Stage117 intervention roster changed")
                for variant, row in rows.items():
                    expected = {"variant": variant, "panel": panel, "scenario_seed": q["query"]["scenario_seed"],
                        "start": q["query"]["start"], "episode_length": horizon, "upper_actor_calls": 0,
                        "lower_calls": horizon, "intervention_decisions": int(variant in spec.DIRECTIONS),
                        "plan_renewals": horizon // period, "plan_fits": horizon // period - 1,
                        "reference_evaluations": horizon, "actor_context_evaluations": horizon}
                    if any(row[k] != value for k, value in expected.items()):
                        raise ValueError("Stage117 one-option plan or full lower feedback changed")
                    for role, noise in (("prefix_seed", q["query"]["prefix_noise_seed"]),
                                       ("suffix_seed", q["query"]["suffix_noise_seeds"][panel])):
                        if row[role] != base.source.scenario.spec.noise_seeds(root, row["scenario_seed"], noise)[1]:
                            raise ValueError("Stage117 prefix/suffix noise roles changed")
                    np.testing.assert_allclose(row["episode_return"], row["prefix_return"] + row["suffix_return"], atol=1e-9, rtol=0)
                    np.testing.assert_allclose(row["suffix_return"], row["option_return"] + row["tail_return"], atol=1e-9, rtol=0)
                    if not np.isfinite([value for value in row.values() if isinstance(value, (float, int))]).all():
                        raise ValueError("Stage117 native result is nonfinite")
                if any(rows["forecast"][k] != rows["zero"][k] for k in (
                        "episode_return", "suffix_return", "tracking_squared_error_integral", "option_command_delta_rms", "tail_command_delta_rms")):
                    raise ValueError("Stage117 zero residual does not reproduce forecast")
            if q["gradients"] != gradients(q["panels"]) or q["option_gradients"] != gradients(q["panels"], "option_return"):
                raise ValueError("Stage117 full-suffix credit changed")
        if group["effects"] != effects(period, group["queries"]) or not np.isfinite(list(group["effects"].values())).all():
            raise ValueError("Stage117 paired root effects changed")
    return cell


def aggregate(cells, *, preflight):
    result = statistics.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    passed = {} if preflight else {str(p): all(result["endpoints"][f"{p}/{m}"]["ci"][0] > 0
        for m in ("crossfit_suffix_gain", "suffix_gradient_dot")) for p in spec.PERIODS}
    result.update(local_plan_gain_gate="mechanical_only" if preflight else
        "supported_both_periods" if all(passed.values()) else "partial" if any(passed.values()) else "not_supported",
        period_local_plan_gain_gate=passed,
        performance_claim="conditional_one_option_plan_headroom_not_a_deployable_policy_or_joint_HRL")
    return result
