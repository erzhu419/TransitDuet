"""Paired local option Q differences without retraining the strong flat base."""
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
import time

import numpy as np
import torch
from freq_hrl.rl.optional_action_residual import OptionalActionResidual
from . import pointmaze_optional_plan as source
from . import pointmaze_feasible_credit as statistics
from .pointmaze_root_response import write_json
from scripts import pointmaze_option_residual_stage111_spec as spec

# Re-export the frozen optional-plan runtime for the next branch-only stage.
native, baseline, scenario = source.native, source.baseline, source.scenario
advice_state = source.advice_state


def load_source(root):
    originals, predictor, _, calibrations = source.load_source(root)
    cell = json.loads(spec.source_result(root).read_text())
    source.qualify(cell, preflight=False)
    for period, model in originals.items():
        p, path = int(period), spec.donor_checkpoint(root, int(period))
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if ((saved["protocol"], saved["root"], saved["period"], saved["method"], saved["updates"]) !=
                (spec.source.EXPERIMENT_PROTOCOL, root, p, "blind", 8)
                or cell["groups"][period]["trained"]["blind"]["checkpoint"] != str(path)):
            raise ValueError("Local residual requires all final Stage107 blind donors")
        before = copy.deepcopy(model.state_dict())
        model.load_state_dict(saved["weights"])
        source.learning.check_training_freeze(model, before, ("lower",))
    return originals, predictor, spec.source_record(root), calibrations


def query_advice(model, actor, obs, history, *, period, predictor, envelope, args, start, bounds):
    feedback = source.baseline.flat_state(history, obs)
    states = {"blind": np.r_[feedback, np.zeros(4, dtype=np.float32)]}
    forecast = source.baseline.forecast.PlanReference("ridge_velocity", predictor, period)
    learned = source.native.curves.CalibratedPlan(predictor, period, args.maximum_subgoal_delta, 1., envelope)
    learned.decode(action=model.act_upper(history.upper_state(obs, oracle_context=None), sample=False)["action"],
        observation=obs, history=history, step=start, world_low=bounds[0], world_high=bounds[1])
    for name, plan in (("forecast", forecast), ("learned", learned)):
        states[name] = source.advice_state(feedback, obs,
            plan(observation=obs, history=history, subgoal=None, age=0, step=start, world_low=bounds[0], world_high=bounds[1]),
            plan.actor_context(age=0, step=start, horizon=args.horizon))
    with torch.inference_mode():
        base = actor.base.distribution(torch.as_tensor(states["blind"]).view(1, -1))
        for state in states.values():
            current = actor.distribution(torch.as_tensor(state).view(1, -1))
            torch.testing.assert_close(current.mean, base.mean, atol=0, rtol=0)
            torch.testing.assert_close(current.stddev, base.stddev, atol=0, rtol=0)
    return {name: float(np.linalg.norm(state[392:].astype(np.float64))) for name, state in states.items()}, {
        "branch_zero_checks": len(states), "advice_upper_calls": 1,
        "advice_ols_fits": forecast.ols_fits+learned.ols_fits,
        "advice_ridge_predictions": forecast.ridge_predictions+learned.ridge_predictions,
        "advice_reference_calls": forecast.calls+learned.calls,
        "advice_context_calls": forecast.context_calls+learned.context_calls}


def episode(model, actor, args, *, query, panel, variant, period, predictor, envelope):
    start, seed = query["start"], query["scenario_seed"]
    if start%period or not 0 < start <= args.horizon-period:
        raise ValueError("Query must be a complete interior option")
    prefix = source.scenario.spec.noise_seeds(args.optimizer_seed, seed, query["prefix_noise_seed"])[1]
    suffix = source.scenario.spec.noise_seeds(args.optimizer_seed, seed, query["suffix_noise_seeds"][panel])[1]
    task = source.native.joint._make_task(env_id=args.env_id, seed=seed, horizon=args.horizon, **source.native.joint._task_options(args))
    try:
        obs = task.reset()
        history = source.native.joint.PointMazeRegimeFeatureBuilder(time_scale=source.native.joint.scale_for(args))
        history.reset(obs)
        bounds = source.native.joint.pointmaze_goal_bounds(task.environment)
        rewards, commands, measurements, innovations = [], [], [], []
        state_at_query, advice, advice_cost = None, None, {}
        for step in range(args.horizon):
            state = np.r_[source.baseline.flat_state(history, obs), np.zeros(4, dtype=np.float32)]
            if step == start:
                state_at_query = state.copy()
                if panel == "A" and variant == "zero":
                    advice, advice_cost = query_advice(model, actor, obs, history, period=period, predictor=predictor,
                        envelope=envelope, args=args, start=start, bounds=bounds)
            # Only this option changes; all preceding/following policy functions remain flat.
            with torch.inference_mode():
                actor.readout.bias.zero_()
                if variant != "zero" and start <= step < start+period:
                    actor.readout.bias[int(variant[4])] = spec.EPSILON if variant.endswith("plus") else -spec.EPSILON
                torch.manual_seed((prefix if step < start else suffix)+step)
                dist = actor.distribution(torch.as_tensor(state).view(1, -1))
                raw = dist.rsample()[0]
                innovations.append(((raw-dist.mean[0])/dist.stddev[0]).numpy())
                command = source.native.joint.squash_box_action(raw.numpy(), task.action_low, task.action_high)
            commands.append(command.copy())
            measurements.append(obs.task_measurement.copy())
            obs, reward, terminated, truncated, _ = task.step(command)
            if bool(terminated or truncated) and step+1 != args.horizon:
                raise RuntimeError("Local Q requires the registered episode boundary")
            rewards.append(float(reward))
            history.update(obs)
        with torch.no_grad():
            actor.readout.bias.zero_()
        row = {"variant": variant, "panel": panel, "episode_length": args.horizon, "episode_return": float(np.sum(rewards)),
            "suffix_return": float(np.sum(rewards[start:])), "option_return": float(np.sum(rewards[start:start+period])),
            "prefix_seed": prefix, "suffix_seed": suffix, "residual_steps": 0 if variant == "zero" else period,
            "lower_calls": len(commands)}
        return row, {"query_state": state_at_query, "prefix_commands": np.asarray(commands[:start]),
            "prefix_rewards": np.asarray(rewards[:start]), "measurements": np.asarray(measurements),
            "innovations": np.asarray(innovations), "advice_norms": advice, "advice_cost": advice_cost}
    finally:
        task.environment.close()


def gradients(panels):
    return {name: [(rows[f"axis{i}_plus"]["suffix_return"]-rows[f"axis{i}_minus"]["suffix_return"])/(2*spec.EPSILON)
        for i in range(2)] for name, rows in panels.items()}


def worker_query(job):
    weights, query, period, predictor, envelope = job
    model, args = source.native._WORKER
    model.load_state_dict(weights)
    actor = OptionalActionResidual(model.lower_actor, feedback_dim=392, advice_dim=4)
    initial = copy.deepcopy(actor.state_dict())
    panels, common, largest, advice = {}, None, 0., None
    cost = dict.fromkeys(spec.budget(preflight=True), 0)
    for panel in spec.PANELS:
        rows, panel_base = {}, None
        for variant in spec.VARIANTS:
            row, audit = episode(model, actor, args, query=query, panel=panel, variant=variant, period=period,
                predictor=predictor, envelope=envelope)
            if common is None:
                common = audit
                advice = audit["advice_norms"]
                cost.update(audit["advice_cost"])
            else:
                for key in ("query_state", "prefix_commands", "prefix_rewards", "measurements"):
                    np.testing.assert_array_equal(audit[key], common[key])
                cost["prefix_pair_checks"] += 1
                cost["exogenous_pair_checks"] += 1
            if panel_base is None:
                panel_base = audit
            else:
                error = float(np.max(np.abs(audit["innovations"]-panel_base["innovations"])))
                largest = max(largest, error)
                np.testing.assert_allclose(audit["innovations"], panel_base["innovations"], atol=3e-5, rtol=0)
                np.testing.assert_allclose(row["episode_return"]-rows["zero"]["episode_return"],
                    row["suffix_return"]-rows["zero"]["suffix_return"], atol=1e-9, rtol=0)
                cost["innovation_pair_checks"] += 1
                cost["suffix_credit_identity_checks"] += 1
            cost["native_episodes"] += 1
            cost["native_steps"] += row["episode_length"]
            cost["native_lower_calls"] += row["lower_calls"]
            rows[variant] = row
        panels[panel] = rows
    torch.testing.assert_close(actor.state_dict(), initial, atol=0, rtol=0)
    torch.testing.assert_close(source.native.joint.inference_weights(model), weights, atol=0, rtol=0)
    cost.update(native_pair_groups=1, network_freeze_checks=1)
    return {"query": query, "panels": panels, "gradients": gradients(panels), "advice_norms": advice,
        "pairing": "passed", "source_freeze": "passed", "zero_branch": "passed", "innovation_max_error": largest, "cost": cost}


def effects(period, queries):
    a, b = (np.asarray([q["gradients"][name] for q in queries], dtype=np.float64).ravel() for name in spec.PANELS)
    norm = float(np.linalg.norm(a)*np.linalg.norm(b))
    return {f"{period}/local_credit_cosine": float(a@b/norm) if norm else 0.,
        f"{period}/local_credit_dot": float(np.mean(a*b)),
        f"{period}/local_credit_rms": float(np.sqrt(np.mean(np.r_[a, b]**2)))}


def run(root, *, preflight, output):
    models, predictor, record, calibration = load_source(root)
    args, roles, o = spec.arguments(root, preflight=preflight), spec.seed_roles(root, preflight=preflight), spec.options(preflight=preflight)
    started, groups, cost = time.monotonic(), {}, dict.fromkeys(spec.budget(preflight=preflight), 0)
    with ProcessPoolExecutor(max_workers=o["workers"], mp_context=mp.get_context("spawn"), initializer=source.native.init_worker,
            initargs=(models["50"].config, args)) as pool:
        for p in spec.PERIODS:
            model = models[str(p)]
            before = copy.deepcopy(model.state_dict())
            weights = source.native.joint.inference_weights(model)
            queries = list(pool.map(worker_query, [(weights, q, p, predictor, calibration[str(p)]["envelope"]) for q in roles["queries"]]))
            source.native.curves.support.assert_frozen(model, before)
            cost["source_freeze_checks"] += 1
            for query in queries:
                for key, value in query["cost"].items():
                    cost[key] += value
            groups[str(p)] = {"queries": queries, "effects": effects(p, queries), "task_options": spec.task_options(root, preflight=preflight)}
            print(f"{spec.EXPERIMENT_PROTOCOL} {root}/{p}: {len(queries)} local paired queries complete", flush=True)
    cell = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "seed_roles": roles, "source_record": record, "groups": groups, "cost": cost,
        "policy_updates": 0, "critic_fits": 0, "checkpoint_writes": 0, "native_trace_writes": 0,
        "source_loads": {"source_cells": 2, "donor_checkpoints": 6, "source_clones": 2, "forecasters": 1, "decoders": 2},
        "wall_seconds": time.monotonic()-started}
    qualify(cell, preflight=preflight)
    write_json(output, cell)
    write_json(output.parent/"completion"/"ready.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return cell


def qualify(cell, *, preflight):
    root, h = cell["root"], spec.arguments(cell["root"], preflight=preflight).horizon
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or root not in spec.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["source_record"] != spec.source_record(root) or cell["seed_roles"] != spec.seed_roles(root, preflight=preflight)
            or cell["cost"] != spec.budget(preflight=preflight) or set(cell["groups"]) != {str(p) for p in spec.PERIODS}
            or any(cell[k] for k in ("policy_updates", "critic_fits", "checkpoint_writes", "native_trace_writes"))):
        raise ValueError("Local option protocol, source, roster or budget changed")
    for p, group in cell["groups"].items():
        if group["task_options"] != spec.task_options(root, preflight=preflight) or [q["query"] for q in group["queries"]] != cell["seed_roles"]["queries"]:
            raise ValueError("Local option task or query roster changed")
        for query in group["queries"]:
            if (any(query[k] != "passed" for k in ("pairing", "source_freeze", "zero_branch"))
                    or not 0 <= query["innovation_max_error"] <= 3e-5 or set(query["panels"]) != set(spec.PANELS)
                    or set(query["advice_norms"]) != set(spec.ARMS) or query["advice_norms"]["blind"] != 0.):
                raise ValueError("Local option pairing, initial function or freeze failed")
            for name, rows in query["panels"].items():
                prefix = source.scenario.spec.noise_seeds(root, query["query"]["scenario_seed"], query["query"]["prefix_noise_seed"])[1]
                suffix = source.scenario.spec.noise_seeds(root, query["query"]["scenario_seed"], query["query"]["suffix_noise_seeds"][name])[1]
                if set(rows) != set(spec.VARIANTS):
                    raise ValueError("Local option variant roster changed")
                for v, row in rows.items():
                    if ((row["variant"], row["panel"], row["episode_length"], row["prefix_seed"], row["suffix_seed"], row["residual_steps"], row["lower_calls"]) !=
                            (v, name, h, prefix, suffix, 0 if v == "zero" else int(p), h)):
                        raise ValueError("Local option changed intervention duration or independent suffix noise")
            if query["gradients"] != gradients(query["panels"]):
                raise ValueError("Local option Q difference changed")
        e = effects(p, group["queries"])
        if group["effects"] != e or not np.isfinite(list(e.values())).all():
            raise ValueError("Local option credit aggregation changed")
    return cell


def aggregate(cells, *, preflight):
    result = statistics.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    passed = {} if preflight else {str(p): all(result["endpoints"][f"{p}/{m}"]["ci"][0] > 0 for m in spec.METRICS[:2]) for p in spec.PERIODS}
    result.update(local_credit_gate="mechanical_only" if preflight else "supported_both_periods" if all(passed.values()) else "partial" if any(passed.values()) else "not_supported",
        period_credit_gate=passed, performance_claim="none_local_action_bias_credit_only_no_policy_training",
        native_trial_prerequisite="Stage67_critic_HOLD_unchanged")
    return result
