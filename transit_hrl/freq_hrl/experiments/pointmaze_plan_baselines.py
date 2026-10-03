"""Test learned-plan value against fully funded forecast and flat learners."""

from concurrent.futures import ProcessPoolExecutor
import copy
import multiprocessing as mp
import time

import numpy as np
import torch

from freq_hrl.rl.smdp_actor_critic import LevelTrajectoryBatch, concat_level_batches
from . import pointmaze_joint_conditioned as joint
from . import pointmaze_forecast_tracking as forecast
from . import pointmaze_critic_clock as clocks
from .pointmaze_root_response import write_json
from scripts import pointmaze_plan_baselines_stage106_spec as spec

learning, native, scenario = joint.learning, joint.learning.native, joint.scenario


def policy_kind(variant):
    return "flat" if variant.startswith("flat_") else "forecast" if variant.startswith("forecast_") else "joint"


def flat_state(history, observation):
    rows = history.history.reshape(-1, 6)
    velocity = (rows[-1, :2]-rows[-2, :2])/history.time_scale.dt_seconds
    return np.r_[history.upper_state(observation, oracle_context=None), velocity].astype(np.float32)


def primitive_episode(job):
    weights, seed, noise_seed, variant, period, predictor, alpha, envelope, collect = job
    model, args = native._WORKER
    model.load_state_dict(weights)
    policy_seed, lower_seed = scenario.spec.noise_seeds(args.optimizer_seed, seed, noise_seed)
    kind = policy_kind(variant)
    if kind not in ("forecast", "flat"):
        raise ValueError("Primitive baseline cannot call a joint planner")
    scale = native.joint.scale_for(args)
    task = native.joint._make_task(env_id=args.env_id, seed=seed, horizon=args.horizon, **native.joint._task_options(args))
    try:
        observation = task.reset()
        low, high = native.joint.pointmaze_goal_bounds(task.environment)
        history = native.joint.PointMazeRegimeFeatureBuilder(time_scale=scale)
        history.reset(observation)
        model.reset_recurrent_inference()
        plan = forecast.PlanReference("ridge_velocity", predictor, period) if kind == "forecast" else None
        data = {k: [] for k in ("state", "value_state", "action", "reward", "old_logp", "old_value")}
        rewards, distances = [], []
        for step in range(args.horizon):
            if plan is None:
                state = flat_state(history, observation)
            else:
                reference = plan(observation=observation, history=history, subgoal=None, age=step%period,
                    step=step, world_low=low, world_high=high)
                state = np.r_[history.lower_state(observation, subgoal=reference),
                    plan.actor_context(age=step%period, step=step, horizon=args.horizon)].astype(np.float32)
            # Critic-only clocks are frozen and unused by MC credit, as in the joint learner.
            value_state = np.r_[state, clocks.time_context(age=step%period, step=step, horizon=args.horizon, clock=True)].astype(np.float32)
            torch.manual_seed(lower_seed+step)
            output = model.act_lower(state, sample=True, value_state=value_state)
            action = native.joint.squash_box_action(np.asarray(output["action"], dtype=np.float32), task.action_low, task.action_high)
            after, reward, terminated, truncated, info = task.step(action)
            if bool(terminated or truncated) and step+1 != args.horizon:
                raise RuntimeError("Primitive baseline must retain the full native episode")
            if collect:
                for key, value in (("state", state), ("value_state", value_state), ("action", output["action"]),
                        ("reward", reward), ("old_logp", output["logp"]), ("old_value", output["value"])):
                    data[key].append(value)
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
            "upper_calls": 0, "lower_calls": args.horizon, "upper_replay_forward_calls": 0,
            "upper_standard_noise": [], "decision_steps": [], "network_check": "passed",
            "plan_renewals": 0 if plan is None else args.horizon//period,
            "plan_ols_fits": 0 if plan is None else plan.ols_fits,
            "plan_ridge_predictions": 0 if plan is None else plan.ridge_predictions,
            "reference_evaluations": 0 if plan is None else plan.calls,
            "actor_context_evaluations": 0 if plan is None else plan.context_calls}
    finally:
        task.environment.close()


def primitive_pair_check(group, roster, *, root):
    if [r["noise_seed"] for _, r in group] != roster["noise_seeds"] or any(r["seed"] != roster["scenario_seed"] for _, r in group):
        raise ValueError("Primitive scenario roster changed")
    for _, row in group:
        if (row["policy_seed"], row["lower_seed"]) != scenario.spec.noise_seeds(root, row["seed"], row["noise_seed"]):
            raise ValueError("Primitive baseline changed independent noise")
    np.testing.assert_array_equal(group[0][0].state[0], group[1][0].state[0])
    np.testing.assert_array_equal(group[0][0].state[:, 6:390], group[1][0].state[:, 6:390])


def primitive_scores(model, batches, *, period, horizon, cost, actor_names):
    if actor_names != ("lower",):
        raise ValueError("Primitive baseline has only lower policy credit")
    gradients, states, score_costs, signal_rms = {}, [], {}, {}
    for name, groups in batches.items():
        returns = []
        for group in groups:
            values = []
            for batch, row in group:
                if (batch.size != horizon or not np.all(batch.duration == 1) or np.flatnonzero(batch.done).tolist() != [horizon-1]):
                    raise ValueError("Primitive credit changed episode boundaries")
                r = native.independent.exact_returns(batch, 1.)
                np.testing.assert_allclose(r[0], row["episode_return"], atol=.002, rtol=0)
                values.append(r)
                cost["objective_checks"] += 1
                cost["mc_calls"] += 1
            returns.append(values)
        level = concat_level_batches(b for group in groups for b, _ in group)
        signal = scenario.leave_other_out(np.asarray(returns)).reshape(-1)
        g, _, mask, c = native.reliability.episode_scores(model.lower_actor, level, {"scenario": signal},
            horizon=horizon, clip_ratio=model.config.clip_ratio)
        gradients[name], score_costs[name] = g["scenario"], c
        states.append(level.state)
        signal_rms[name] = float(np.sqrt(np.square(signal).mean()))
        for key in ("actor_score_forward_batches", "actor_score_backward_batches"):
            cost[key] += c[key]
    return {"lower": {"gradients": gradients, "states": np.concatenate(states), "sigma_mask": mask,
        "score_costs": score_costs, "signal_rms": signal_rms}}


def check_row(row, period, horizon):
    if policy_kind(row["variant"]) == "joint":
        native.curves.paths.check_row(row, period, horizon)
        return
    planned = policy_kind(row["variant"]) == "forecast"
    expected = {"episode_length": horizon, "lower_calls": horizon, "upper_calls": 0,
        "upper_replay_forward_calls": 0, "upper_standard_noise": [], "decision_steps": [],
        "network_check": "passed", "plan_renewals": horizon//period if planned else 0,
        "plan_ols_fits": horizon//period-1 if planned else 0, "plan_ridge_predictions": horizon//period-1 if planned else 0,
        "reference_evaluations": horizon if planned else 0, "actor_context_evaluations": horizon if planned else 0}
    if any(row[k] != v for k, v in expected.items()):
        raise ValueError("Primitive policy invoked upper or changed planning calls")


def paired_effects(period, evaluation, seeds):
    if any([r["seed"] for r in rows] != seeds for rows in evaluation.values()):
        raise ValueError("Baseline comparison lost paired evaluation scenarios")
    for variant, rows in evaluation.items():
        for row, base in zip(rows, evaluation["joint_base"]):
            if (row["policy_seed"], row["lower_seed"]) != (base["policy_seed"], base["lower_seed"]):
                raise ValueError("Baseline comparison changed independent policy noise")
            if policy_kind(variant) == "joint":
                if row["decision_steps"] != base["decision_steps"]:
                    raise ValueError("Joint evaluation renewal schedule changed")
                np.testing.assert_allclose(row["upper_standard_noise"], base["upper_standard_noise"], atol=2e-6, rtol=0)
    return {f"{period}/{a}_minus_{b}": float(np.mean([x["episode_return"]-y["episode_return"]
        for x, y in zip(evaluation[a], evaluation[b])])) for a, b in spec.CONTRAST_PAIRS}


def run(root, *, preflight, output):
    clones, predictor, initialization, calibrations = joint.fresh.load_source(root)
    args, roles, o = spec.arguments(root, preflight=preflight), spec.seed_roles(root, preflight=preflight), spec.options(preflight=preflight)
    cost, planning, groups = dict.fromkeys(spec.budget(preflight=preflight), 0), dict.fromkeys(spec.planning_budget(preflight=preflight), 0), {}
    cost.update(source_clone_loads=len(clones), forecaster_loads=1, decoder_loads=len(clones))
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=o["workers"], mp_context=mp.get_context("spawn"), initializer=native.init_worker,
            initargs=(clones[str(spec.PERIODS[0])].config, args)) as pool:
        for period in spec.PERIODS:
            original = clones[str(period)]
            snapshot = copy.deepcopy(original.state_dict())
            alpha, envelope = calibrations[str(period)]["alpha"], calibrations[str(period)]["envelope"]

            def episodes(weights, roster, variant, collect=False, *, pair_worker=None):
                jobs = [(weights, s, n, variant, period, predictor, alpha, envelope, collect) for s, n in roster]
                if pair_worker is not None:
                    pairs = [item for pair in pool.map(pair_worker, [jobs[i:i+2] for i in range(0, len(jobs), 2)]) for item in pair]
                else:
                    worker = scenario.worker_native if policy_kind(variant) == "joint" else primitive_episode
                    pairs = list(pool.map(worker, jobs))
                for _, r in pairs:
                    check_row(r, period, args.horizon)
                    for key, value in (("native_episodes", 1), ("native_steps", r["episode_length"]),
                            ("native_lower_calls", r["lower_calls"]), ("native_upper_calls", r["upper_calls"]),
                            ("pairing_upper_forward_calls", r["upper_calls"]), ("native_network_checks", 1),
                            ("upper_replay_forward_calls", r["upper_replay_forward_calls"]),
                            ("credit_episodes" if collect else "evaluation_episodes", 1)):
                        cost[key] += value
                    for key in planning:
                        planning[key] += r[key]
                return pairs

            models = {m: copy.deepcopy(original) for m in spec.METHODS}
            cost["training_models_initialized"] += len(models)
            histories = {m: [] for m in models}
            for iteration, round_roles in enumerate(roles["training_rounds"], 1):
                for method, model in models.items():
                    weights = native.joint.inference_weights(model)
                    if policy_kind(method) == "joint":
                        actor_batches = joint.collect_actor_credit(episodes, weights, round_roles, method, cost, root=root)
                        batches, score_builder = None, None
                        sets = actor_batches.values()
                    else:
                        batches = {}
                        for name in ("A", "B"):
                            roster = round_roles["credit_"+name]+round_roles["lower_credit_"+name]
                            pairs = episodes(weights, [(r["scenario_seed"], n) for r in roster for n in r["noise_seeds"]], method, True)
                            batches[name] = [pairs[i:i+2] for i in range(0, len(pairs), 2)]
                            for pair, registered in zip(batches[name], roster):
                                primitive_pair_check(pair, registered, root=root)
                                cost["scenario_pair_checks"] += 1
                                cost["primitive_pair_checks"] += 1
                        actor_batches, score_builder, sets = None, primitive_scores, [batches]
                    mean = float(np.mean([r["episode_return"] for bs in sets for pairs in bs.values() for group in pairs for _, r in group]))
                    row = learning.update_mean(model, batches, method=method, period=period, horizon=args.horizon,
                        cost=cost, allocation=spec.allocation(method, period), actor_batches=actor_batches, score_builder=score_builder)
                    row.update(iteration=iteration, credit_mean_reward_before_update=mean)
                    histories[method].append(row)
                    del batches, actor_batches, sets
                print(f"{spec.EXPERIMENT_PROTOCOL} {root}/{period}: update {iteration}/{o['updates']}, no evaluation selection", flush=True)
            weights = {m: native.joint.inference_weights(model) for m, model in models.items()}
            weights.update({v: native.joint.inference_weights(original) for v in spec.VARIANTS if v.endswith("_base")})
            evaluation = {v: [r for _, r in episodes(weights[v], [(s, s) for s in roles["native_evaluation"]], v)] for v in spec.VARIANTS}
            cost["native_pair_checks"] += len(roles["native_evaluation"])
            trained = {}
            for method, model in models.items():
                checkpoint = None if preflight else learning.final_checkpoint(model, output, root=root, period=period,
                    method=method, updates=o["updates"], protocol=spec)
                cost["checkpoint_writes"] += int(checkpoint is not None)
                learning.check_training_freeze(model, snapshot, spec.METHODS[method])
                trained[method] = {"history": histories[method], "evaluation_update": o["updates"], "checkpoint": checkpoint, "final_freeze_check": "passed"}
            native.curves.support.assert_frozen(original, snapshot)
            cost["frozen_model_checks"] += 1
            groups[str(period)] = {"alpha": alpha, "trained": trained, "evaluation": evaluation,
                "effects": paired_effects(period, evaluation, roles["native_evaluation"]),
                "task_options": spec.task_options(root, preflight=preflight), "source_and_Adam_unchanged": "passed"}
    cell = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "seed_roles": roles, "source_initialization": initialization, "groups": groups,
        "cost": cost, "native_planning_cost": planning, "optimizer_steps": 0, "critic_fits": 0, "forecaster_fits": 0,
        "native_trace_writes": 0, "wall_seconds": time.monotonic()-started}
    qualify(cell, preflight=preflight)
    write_json(output, cell)
    write_json(output.parent/"completion"/"ready.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return cell


def qualify(cell, *, preflight):
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or cell["root"] not in spec.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["seed_roles"] != spec.seed_roles(cell["root"], preflight=preflight)
            or cell["source_initialization"] != spec.source_record(cell["root"])
            or cell["cost"] != spec.budget(preflight=preflight) or cell["native_planning_cost"] != spec.planning_budget(preflight=preflight)
            or any(cell[k] for k in ("optimizer_steps", "critic_fits", "forecaster_fits", "native_trace_writes"))
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}):
        raise ValueError("Plan baseline protocol, source, sample or planning budget changed")
    o, h = spec.options(preflight=preflight), spec.arguments(cell["root"], preflight=preflight).horizon
    for p, g in cell["groups"].items():
        if (g["task_options"] != spec.task_options(cell["root"], preflight=preflight) or g["source_and_Adam_unchanged"] != "passed"
                or set(g["evaluation"]) != set(spec.VARIANTS) or set(g["trained"]) != set(spec.METHODS)):
            raise ValueError("Plan baseline task or policy roster changed")
        for v, rows in g["evaluation"].items():
            if [r["seed"] for r in rows] != cell["seed_roles"]["native_evaluation"]:
                raise ValueError("Fresh final evaluation roster changed")
            for r in rows:
                check_row(r, int(p), h)
                expected = scenario.spec.noise_seeds(cell["root"], r["seed"], r["seed"])
                if (r["variant"] != v or (r["noise_seed"], r["policy_seed"], r["lower_seed"]) != (r["seed"], *expected)
                        or "upper_noise_seed" in r or r["upper_replay_forward_calls"]):
                    raise ValueError("Training conditioning leaked into independent evaluation")
                if policy_kind(v) == "joint" and r["alpha"] != g["alpha"]:
                    raise ValueError("Joint decoder changed")
        effects = paired_effects(p, g["evaluation"], cell["seed_roles"]["native_evaluation"])
        if g["effects"] != effects or not np.isfinite(list(effects.values())).all():
            raise ValueError("Final paired reward effects changed")
        for method, t in g["trained"].items():
            if (t["evaluation_update"] != o["updates"] or t["final_freeze_check"] != "passed"
                    or bool(t["checkpoint"]) == preflight or [r["iteration"] for r in t["history"]] != list(range(1, o["updates"]+1))):
                raise ValueError("Final-only checkpoint or update count changed")
            for r in t["history"]:
                alloc = spec.allocation(method, int(p))
                if r["freeze_check"] != "passed" or set(r["actors"]) != set(alloc):
                    raise ValueError("Active actor or frozen parameters changed")
                exact, weighted = 0., 0.
                for a, row in r["actors"].items():
                    geometry = row["geometry"]
                    paths = 2*o["credit_scenarios_per_batch"]*o["rollouts_per_scenario"]*(1 if policy_kind(method) == "joint" else 2)
                    if (row["std_check"] != "passed" or geometry["radius_check"] != "passed"
                            or geometry["nominal_fisher_kl"] != spec.FISHER_RADIUS*alloc[a]
                            or row["gradient_episodes"] != paths or row["decision_calls_per_episode"] != (h//int(p) if a == "upper" else h)
                            or row["max_abs_old_logp_difference"] > .0002):
                        raise ValueError("Registered MC credit, radius or log-probability changed")
                    exact += geometry["exact_kl"]["plus"]
                    weighted += geometry["exact_kl"]["plus"]/(int(p) if a == "upper" else 1)
                nominal = spec.FISHER_RADIUS*sum(alloc.values())
                if (exact != r["exact_sum_kl"] or not .5*nominal <= exact <= 2*nominal
                        or not np.isclose(r["nominal_call_weighted_kl"], spec.FISHER_RADIUS, rtol=0, atol=1e-15)
                        or weighted != r["exact_call_weighted_kl"]):
                    raise ValueError("Matched nominal call-weighted KL changed")
    return cell


def aggregate(cells, *, preflight):
    result = learning.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    result.update(primary_endpoints=list(spec.PRIMARY_ENDPOINTS),
        plan_baseline_confirmation="mechanical_only" if preflight else (
            "supported" if all(result["endpoints"][k]["ci"][0] > 0 for k in spec.PRIMARY_ENDPOINTS) else "not_supported"),
        performance_claim="teacher_assisted_MC_mean_learning_plan_increment_not_from_scratch_flat_PPO_or_frequency_superiority")
    return result
