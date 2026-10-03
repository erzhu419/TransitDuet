"""Learn optional plan advice without displacing instantaneous flat feedback."""

from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import replace
import json
import multiprocessing as mp
import time

import numpy as np
import torch
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, LevelTrajectoryBatch
from . import pointmaze_plan_baselines as baseline
from .pointmaze_root_response import write_json
from scripts import pointmaze_optional_plan_stage107_spec as spec

learning, native, scenario = baseline.learning, baseline.native, baseline.scenario


def expand_model(source, flat_weights, upper_weights):
    model = FrequencySeparatedActorCriticPPO(replace(source.config, lower_state_dim=396, lower_value_state_dim=398))
    weights = copy.deepcopy(flat_weights)
    weights["upper_actor"] = copy.deepcopy(upper_weights["upper_actor"])
    actor = weights["lower_actor"]["net.0.weight"]
    value = weights["lower_value"]["net.0.weight"]
    weights["lower_actor"]["net.0.weight"] = torch.cat((actor, actor.new_zeros((actor.shape[0], 4))), 1)
    weights["lower_value"]["net.0.weight"] = torch.cat((value[:, :392], value.new_zeros((value.shape[0], 4)), value[:, 392:]), 1)
    model.load_state_dict(weights)
    return model


def load_source(root):
    originals, predictor, _, calibrations = baseline.joint.fresh.load_source(root)
    cell = json.loads(spec.source_result(root).read_text())
    baseline.qualify(cell, preflight=False)
    models = {}
    for p in spec.PERIODS:
        donors = {}
        for method in ("flat_lower", "joint_conditioned"):
            path = spec.donor_checkpoint(root, p, method)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if ((saved["protocol"], saved["root"], saved["period"], saved["method"], saved["updates"]) !=
                    (spec.source.EXPERIMENT_PROTOCOL, root, p, method, spec.source.options(preflight=False)["updates"])
                    or cell["groups"][str(p)]["trained"][method]["checkpoint"] != str(path)):
                raise ValueError("Optional advice requires all registered final Stage106 donors")
            model = copy.deepcopy(originals[str(p)])
            snapshot = copy.deepcopy(model.state_dict())
            model.load_state_dict(saved["weights"])
            learning.check_training_freeze(model, snapshot, spec.source.METHODS[method])
            donors[method] = saved["weights"]
        models[str(p)] = expand_model(originals[str(p)], donors["flat_lower"], donors["joint_conditioned"])
    return models, predictor, spec.source_record(root), calibrations


def advice_kind(variant):
    return variant if variant in ("forecast_hint", "learned_hint") else "blind"


def advice_state(feedback, observation, reference, velocity):
    hint = np.r_[reference-observation.target, velocity-feedback[-2:]].astype(np.float32)
    return np.r_[feedback, hint].astype(np.float32)


def worker_episode(job):
    weights, seed, noise_seed, variant, period, predictor, alpha, envelope, collect = job
    model, args = native._WORKER
    model.load_state_dict(weights)
    policy_seed, lower_seed = scenario.spec.noise_seeds(args.optimizer_seed, seed, noise_seed)
    torch.manual_seed(policy_seed)
    kind = advice_kind(variant)
    plan = (native.curves.CalibratedPlan(predictor, period, args.maximum_subgoal_delta, alpha, envelope)
        if kind == "learned_hint" else baseline.forecast.PlanReference("ridge_velocity", predictor, period)
        if kind == "forecast_hint" else None)
    scale = native.joint.scale_for(args)
    task = native.joint._make_task(env_id=args.env_id, seed=seed, horizon=args.horizon, **native.joint._task_options(args))
    try:
        observation = task.reset()
        low, high = native.joint.pointmaze_goal_bounds(task.environment)
        history = native.joint.PointMazeRegimeFeatureBuilder(time_scale=scale)
        history.reset(observation)
        model.reset_recurrent_inference()
        data = {k: [] for k in ("state", "value_state", "action", "reward", "old_logp", "old_value")}
        rewards, distances, innovations, decisions = [], [], [], []
        for step in range(args.horizon):
            feedback = baseline.flat_state(history, observation)
            if kind == "learned_hint" and step%period == 0:
                upper_state = history.upper_state(observation, oracle_context=None)
                output = model.act_upper(upper_state, sample=True)
                with torch.inference_mode():
                    dist = model.upper_actor.distribution(torch.as_tensor(upper_state, dtype=torch.float32).view(1, -1))
                    innovations.append(((np.asarray(output["action"])-dist.mean[0].numpy())/dist.stddev[0].numpy()).tolist())
                plan.decode(action=np.asarray(output["action"], dtype=np.float32), observation=observation,
                    history=history, step=step, world_low=low, world_high=high)
                decisions.append(step)
            if plan is None:
                state = np.r_[feedback, np.zeros(4, dtype=np.float32)]
            else:
                reference = plan(observation=observation, history=history, subgoal=None, age=step%period,
                    step=step, world_low=low, world_high=high)
                velocity = plan.actor_context(age=step%period, step=step, horizon=args.horizon)
                state = advice_state(feedback, observation, reference, velocity)
            value_state = np.r_[state, baseline.clocks.time_context(age=step%period, step=step, horizon=args.horizon, clock=True)].astype(np.float32)
            torch.manual_seed(lower_seed+step)
            output = model.act_lower(state, sample=True, value_state=value_state)
            action = native.joint.squash_box_action(np.asarray(output["action"], dtype=np.float32), task.action_low, task.action_high)
            after, reward, terminated, truncated, info = task.step(action)
            if bool(terminated or truncated) and step+1 != args.horizon:
                raise RuntimeError("Optional advice episode ended before its registered horizon")
            if collect:
                for k, v in (("state", state), ("value_state", value_state), ("action", output["action"]),
                        ("reward", reward), ("old_logp", output["logp"]), ("old_value", output["value"])):
                    data[k].append(v)
            rewards.append(float(reward))
            distances.append(float(info["tracking_distance"]))
            observation = after
            history.update(observation)
        batch = None
        if collect:
            done = np.zeros(args.horizon, dtype=np.float32)
            done[-1] = 1.
            batch = LevelTrajectoryBatch(**{k: np.asarray(v, dtype=np.float32) for k, v in data.items()},
                duration=np.ones(args.horizon, dtype=np.int64), done=done)
        torch.testing.assert_close(native.joint.inference_weights(model), weights, atol=0, rtol=0)
        return batch, {"seed": seed, "noise_seed": noise_seed, "policy_seed": policy_seed, "lower_seed": lower_seed,
            "variant": variant, "episode_return": float(np.sum(rewards)), "episode_length": args.horizon,
            "tracking_squared_error_integral": float(np.dot(distances, distances)*scale.dt_seconds),
            "upper_calls": len(decisions), "lower_calls": args.horizon, "decision_steps": decisions,
            "upper_standard_noise": innovations, "network_check": "passed", "plan_renewals": 0 if plan is None else args.horizon//period,
            **{k: 0 if plan is None else getattr(plan, attr) for k, attr in
                (("plan_ols_fits", "ols_fits"), ("plan_ridge_predictions", "ridge_predictions"),
                ("reference_evaluations", "calls"), ("actor_context_evaluations", "context_calls"))}}
    finally:
        task.environment.close()


def check_row(row, period, horizon):
    kind = advice_kind(row["variant"])
    planned, learned = kind != "blind", kind == "learned_hint"
    expected = {"episode_length": horizon, "lower_calls": horizon, "upper_calls": horizon//period if learned else 0,
        "decision_steps": list(range(0, horizon, period)) if learned else [], "network_check": "passed",
        "plan_renewals": horizon//period if planned else 0,
        "plan_ols_fits": horizon//period-1 if planned else 0, "plan_ridge_predictions": horizon//period-1 if planned else 0,
        "reference_evaluations": horizon if planned else 0, "actor_context_evaluations": horizon if planned else 0}
    if any(row[k] != v for k, v in expected.items()) or len(row["upper_standard_noise"]) != expected["upper_calls"]:
        raise ValueError("Optional advice changed native planning or inference calls")


def pair_check(group, roster, *, root):
    if [r["noise_seed"] for _, r in group] != roster["noise_seeds"] or any(r["seed"] != roster["scenario_seed"] for _, r in group):
        raise ValueError("Optional advice scenario roster changed")
    for _, r in group:
        if (r["policy_seed"], r["lower_seed"]) != scenario.spec.noise_seeds(root, r["seed"], r["noise_seed"]):
            raise ValueError("Optional advice changed independent policy noise")
    np.testing.assert_array_equal(group[0][0].state[0, :392], group[1][0].state[0, :392])
    np.testing.assert_array_equal(group[0][0].state[:, 6:390], group[1][0].state[:, 6:390])


def paired_effects(period, evaluation, seeds):
    if set(evaluation) != set(spec.VARIANTS) or any([r["seed"] for r in rows] != seeds for rows in evaluation.values()):
        raise ValueError("Optional advice final variant or scenario roster changed")
    for rows in evaluation.values():
        for row, base in zip(rows, evaluation["base"]):
            if (row["policy_seed"], row["lower_seed"]) != (base["policy_seed"], base["lower_seed"]):
                raise ValueError("Optional advice lost paired independent evaluation noise")
    return {f"{period}/{a}_minus_{b}": float(np.mean([x["episode_return"]-y["episode_return"]
        for x, y in zip(evaluation[a], evaluation[b])])) for a, b in spec.CONTRAST_PAIRS}


def run(root, *, preflight, output):
    sources, predictor, initialization, calibrations = load_source(root)
    args, roles, o = spec.arguments(root, preflight=preflight), spec.seed_roles(root, preflight=preflight), spec.options(preflight=preflight)
    cost, planning, groups = dict.fromkeys(spec.budget(preflight=preflight), 0), dict.fromkeys(spec.planning_budget(preflight=preflight), 0), {}
    cost.update(source_clone_loads=len(sources), forecaster_loads=1, decoder_loads=len(sources), source_cell_loads=1,
        donor_checkpoint_loads=2*len(sources), expanded_model_initializations=len(sources))
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=o["workers"], mp_context=mp.get_context("spawn"), initializer=native.init_worker,
            initargs=(sources[str(spec.PERIODS[0])].config, args)) as pool:
        for period in spec.PERIODS:
            source = sources[str(period)]
            snapshot = copy.deepcopy(source.state_dict())
            calibration = calibrations[str(period)]

            def episodes(weights, roster, variant, collect=False):
                jobs = [(weights, s, n, variant, period, predictor, calibration["alpha"], calibration["envelope"], collect) for s, n in roster]
                pairs = list(pool.map(worker_episode, jobs))
                for _, row in pairs:
                    check_row(row, period, args.horizon)
                    for k, v in (("native_episodes", 1), ("native_steps", row["episode_length"]),
                            ("native_lower_calls", row["lower_calls"]), ("native_upper_calls", row["upper_calls"]),
                            ("native_network_checks", 1), ("credit_episodes" if collect else "evaluation_episodes", 1)):
                        cost[k] += v
                    for k in planning:
                        planning[k] += row[k]
                return pairs

            models, histories = {m: copy.deepcopy(source) for m in spec.METHODS}, {m: [] for m in spec.METHODS}
            cost["training_models_initialized"] += len(models)
            for iteration, round_roles in enumerate(roles["training_rounds"], 1):
                for method, model in models.items():
                    batches = {}
                    for name in ("A", "B"):
                        roster = round_roles["credit_"+name]+round_roles["lower_credit_"+name]
                        pairs = episodes(native.joint.inference_weights(model), [(s["scenario_seed"], n) for s in roster for n in s["noise_seeds"]], method, True)
                        batches[name] = [pairs[i:i+2] for i in range(0, len(pairs), 2)]
                        for group, registered in zip(batches[name], roster):
                            pair_check(group, registered, root=root)
                            cost["scenario_pair_checks"] += 1
                    row = learning.update_mean(model, batches, method=method, period=period, horizon=args.horizon,
                        cost=cost, allocation=spec.allocation(method, period), score_builder=baseline.primitive_scores)
                    row.update(iteration=iteration)
                    histories[method].append(row)
                    del batches, pairs
                print(f"{spec.EXPERIMENT_PROTOCOL} {root}/{period}: lower update {iteration}/{o['updates']}, upper frozen", flush=True)
            weights = {m: native.joint.inference_weights(model) for m, model in models.items()}
            weights.update(base=native.joint.inference_weights(source), forecast_blinded=weights["forecast_hint"], learned_blinded=weights["learned_hint"])
            evaluation = {v: [r for _, r in episodes(weights[v], [(s, s) for s in roles["native_evaluation"]], v)] for v in spec.VARIANTS}
            cost["native_pair_checks"] += len(roles["native_evaluation"])
            trained = {}
            for method, model in models.items():
                learning.check_training_freeze(model, snapshot, ("lower",))
                checkpoint = None if preflight else learning.final_checkpoint(model, output, root=root, period=period,
                    method=method, updates=o["updates"], protocol=spec)
                cost["checkpoint_writes"] += int(checkpoint is not None)
                trained[method] = {"history": histories[method], "evaluation_update": o["updates"], "checkpoint": checkpoint, "final_freeze_check": "passed"}
            native.curves.support.assert_frozen(source, snapshot)
            cost["frozen_model_checks"] += 1
            groups[str(period)] = {"evaluation": evaluation, "effects": paired_effects(period, evaluation, roles["native_evaluation"]),
                "trained": trained, "task_options": spec.task_options(root, preflight=preflight), "source_and_Adam_unchanged": "passed"}
    cell = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root, "preflight": preflight,
        "seed_roles": roles, "source_initialization": initialization, "cost": cost, "native_planning_cost": planning, "groups": groups,
        "optimizer_steps": 0, "critic_fits": 0, "forecaster_fits": 0, "native_trace_writes": 0, "wall_seconds": time.monotonic()-started}
    qualify(cell, preflight=preflight)
    write_json(output, cell)
    write_json(output.parent/"completion"/"ready.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return cell


def qualify(cell, *, preflight):
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or cell["root"] not in spec.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["source_initialization"] != spec.source_record(cell["root"])
            or cell["seed_roles"] != spec.seed_roles(cell["root"], preflight=preflight)
            or cell["cost"] != spec.budget(preflight=preflight) or cell["native_planning_cost"] != spec.planning_budget(preflight=preflight)
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}
            or any(cell[k] for k in ("optimizer_steps", "critic_fits", "forecaster_fits", "native_trace_writes"))):
        raise ValueError("Optional advice source, protocol, samples or planning budget changed")
    o, h = spec.options(preflight=preflight), spec.arguments(cell["root"], preflight=preflight).horizon
    paths = 4*o["credit_scenarios_per_batch"]*o["rollouts_per_scenario"]
    for p, g in cell["groups"].items():
        if (g["source_and_Adam_unchanged"] != "passed" or g["task_options"] != spec.task_options(cell["root"], preflight=preflight)
                or set(g["trained"]) != set(spec.METHODS)):
            raise ValueError("Optional advice source freeze, task or learner roster changed")
        effects = paired_effects(p, g["evaluation"], cell["seed_roles"]["native_evaluation"])
        if g["effects"] != effects or not np.isfinite(list(effects.values())).all():
            raise ValueError("Optional advice final reward contrasts changed")
        for v, rows in g["evaluation"].items():
            for r in rows:
                check_row(r, int(p), h)
                expected = scenario.spec.noise_seeds(cell["root"], r["seed"], r["seed"])
                if r["variant"] != v or (r["noise_seed"], r["policy_seed"], r["lower_seed"]) != (r["seed"], *expected):
                    raise ValueError("Optional advice final independent noise changed")
        for method, t in g["trained"].items():
            if (t["evaluation_update"] != o["updates"] or t["final_freeze_check"] != "passed" or bool(t["checkpoint"]) == preflight
                    or [r["iteration"] for r in t["history"]] != list(range(1, o["updates"]+1))):
                raise ValueError("Optional advice final-only update or checkpoint changed")
            for r in t["history"]:
                if r["freeze_check"] != "passed" or set(r["actors"]) != {"lower"}:
                    raise ValueError("Optional advice updated upper or frozen parameters")
                a = r["actors"]["lower"]
                geometry = a["geometry"]
                if (a["std_check"] != "passed" or geometry["radius_check"] != "passed" or geometry["nominal_fisher_kl"] != spec.FISHER_RADIUS
                        or a["gradient_episodes"] != paths or a["decision_calls_per_episode"] != h or a["max_abs_old_logp_difference"] > .0002
                        or r["exact_sum_kl"] != geometry["exact_kl"]["plus"] or r["exact_call_weighted_kl"] != r["exact_sum_kl"]
                        or r["nominal_call_weighted_kl"] != spec.FISHER_RADIUS):
                    raise ValueError("Optional advice credit or matched lower KL changed")
    return cell


def aggregate(cells, *, preflight):
    result = learning.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    result.update(primary_endpoints=list(spec.PRIMARY_ENDPOINTS),
        optional_plan_confirmation="mechanical_only" if preflight else (
            "supported" if all(result["endpoints"][k]["ci"][0] > 0 for k in spec.PRIMARY_ENDPOINTS) else "not_supported"),
        performance_claim="conditional_frozen_upper_optional_hint_learning_above_strong_flat_not_joint_HRL_or_frequency_superiority")
    return result
