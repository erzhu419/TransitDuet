"""Train an explicit upper residual while keeping the Stage112 lower branch fixed."""
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from freq_hrl.rl.optional_action_residual import OptionalActionResidual
from freq_hrl.rl.smdp_actor_critic import LevelTrajectoryBatch, concat_level_batches
from . import pointmaze_option_residual as source
from . import pointmaze_option_residual_train as lower_training
from . import pointmaze_independent_credit as independent
from .pointmaze_root_response import write_json
from scripts import pointmaze_upper_residual_train_stage113_spec as spec


def upper_branch(model):
    return OptionalActionResidual(model.upper_actor, feedback_dim=model.config.upper_state_dim, advice_dim=0)


def lower_branch(model, state):
    actor = lower_training.branch(model)
    actor.load_state_dict(state)
    return actor


def plan_for_arm(arm, predictor, period, calibration, args):
    if arm == "forecast":
        return source.baseline.forecast.PlanReference("ridge_velocity", predictor, period)
    if arm == "learned":
        return source.native.curves.CalibratedPlan(predictor, period, args.maximum_subgoal_delta,
            1.0, calibration["envelope"])
    raise ValueError("unknown upper residual arm")


def native_episode(source_weights, lower_state, upper_state, *, seed, noise_seed, arm, period,
        predictor, calibration, args, collect, upper_sample,
        upper_factory=upper_branch, plan_factory=plan_for_arm):
    model, _ = source.native._WORKER
    model.load_state_dict(source_weights)
    lower = lower_branch(model, lower_state)
    upper = None if upper_state is None else upper_factory(model)
    if upper is not None:
        upper.load_state_dict(upper_state)
    policy_seed, lower_seed = source.scenario.spec.noise_seeds(args.optimizer_seed, seed, noise_seed)
    torch.manual_seed(policy_seed)
    plan = plan_factory(arm, predictor, period, calibration, args)
    scale = source.native.joint.scale_for(args)
    task = source.native.joint._make_task(env_id=args.env_id, seed=seed, horizon=args.horizon,
        **source.native.joint._task_options(args))
    data = {key: [] for key in ("state", "value_state", "action", "reward", "old_logp", "old_value")}
    upper_data = {key: [] for key in ("state", "action", "old_logp")}
    rewards, distances, innovations, decisions = [], [], [], []
    try:
        observation = task.reset()
        low, high = source.native.joint.pointmaze_goal_bounds(task.environment)
        history = source.native.joint.PointMazeRegimeFeatureBuilder(time_scale=scale)
        history.reset(observation)
        model.reset_recurrent_inference()
        for step in range(args.horizon):
            feedback = source.baseline.flat_state(history, observation)
            if arm == "learned" and step % period == 0:
                upper_state_now = history.upper_state(observation, oracle_context=None).astype(np.float32)
                if upper is None:
                    raise ValueError("learned upper arm requires a trainable upper branch")
                with torch.inference_mode():
                    distribution = upper.distribution(torch.as_tensor(upper_state_now).view(1, -1))
                    upper_action = distribution.rsample()[0] if upper_sample else distribution.mean[0]
                    upper_logp = distribution.log_prob(upper_action).sum()
                plan.decode(action=upper_action.detach().numpy(), observation=observation, history=history, step=step,
                    world_low=low, world_high=high)
                decisions.append(step)
                if collect:
                    upper_data["state"].append(upper_state_now)
                    upper_data["action"].append(upper_action.detach().numpy())
                    upper_data["old_logp"].append(upper_logp.item())
            reference = plan(observation=observation, history=history, subgoal=None, age=step % period,
                step=step, world_low=low, world_high=high)
            velocity = plan.actor_context(age=step % period, step=step, horizon=args.horizon)
            state = source.advice_state(feedback, observation, reference, velocity)
            value_state = np.r_[state, source.baseline.clocks.time_context(age=step % period,
                step=step, horizon=args.horizon, clock=True)].astype(np.float32)
            torch.manual_seed(lower_seed + step)
            with torch.inference_mode():
                distribution = lower.distribution(torch.as_tensor(state).view(1, -1))
                action_tensor = distribution.rsample()[0]
                logp = distribution.log_prob(action_tensor).sum()
                value = model.lower_value(torch.as_tensor(value_state).view(1, -1))[0]
            action = source.native.joint.squash_box_action(action_tensor.numpy(), task.action_low, task.action_high)
            innovations.append(((action_tensor - distribution.mean[0]) / distribution.stddev[0]).numpy())
            after, reward, terminated, truncated, info = task.step(action)
            if bool(terminated or truncated) and step + 1 != args.horizon:
                raise RuntimeError("upper residual episode ended before registered horizon")
            if collect:
                for key, value_out in (("state", state), ("value_state", value_state), ("action", action_tensor.numpy()),
                        ("reward", reward), ("old_logp", logp.item()), ("old_value", value.item())):
                    data[key].append(value_out)
            rewards.append(float(reward)); distances.append(float(info["tracking_distance"]))
            observation = after; history.update(observation)
        lower_batch = None
        upper_batch = None
        if collect:
            done = np.zeros(args.horizon, dtype=np.float32); done[-1] = 1.
            lower_batch = LevelTrajectoryBatch(**{key: np.asarray(value, dtype=np.float32) for key, value in data.items()},
                duration=np.ones(args.horizon, dtype=np.int64), done=done)
            if upper_data["state"]:
                upper_done = np.zeros(len(upper_data["state"]), dtype=np.float32); upper_done[-1] = 1.
                upper_reward = np.asarray([np.sum(rewards[start:start + period])
                    for start in range(0, args.horizon, period)], dtype=np.float32)
                upper_batch = LevelTrajectoryBatch(
                    state=np.asarray(upper_data["state"], dtype=np.float32),
                    action=np.asarray(upper_data["action"], dtype=np.float32),
                    reward=upper_reward,
                    duration=np.ones(len(upper_data["state"]), dtype=np.int64), done=upper_done,
                    old_logp=np.asarray(upper_data["old_logp"], dtype=np.float32),
                    old_value=np.zeros(len(upper_data["state"]), dtype=np.float32))
        row = {"seed": seed, "noise_seed": noise_seed, "arm": arm, "episode_return": float(np.sum(rewards)),
            "episode_length": args.horizon, "upper_calls": len(decisions), "lower_calls": args.horizon,
            "decision_steps": decisions, "upper_sample": upper_sample, "lower_sample": True,
            "network_check": "passed", "plan_renewals": args.horizon // period,
            "plan_ols_fits": plan.ols_fits, "plan_ridge_predictions": plan.ridge_predictions,
            "reference_evaluations": plan.calls, "actor_context_evaluations": plan.context_calls,
            "tracking_squared_error_integral": float(np.dot(distances, distances) * scale.dt_seconds)}
        return lower_batch, upper_batch, row, np.asarray(innovations)
    finally:
        task.environment.close()


def training_pair(job):
    source_weights, lower_state, upper_state, scenario_seed, noise_seeds, period, predictor, calibration, args = job
    outputs = [native_episode(source_weights, lower_state, upper_state, seed=scenario_seed, noise_seed=noise,
        arm="learned", period=period, predictor=predictor, calibration=calibration, args=args,
        collect=True, upper_sample=True) for noise in noise_seeds]
    np.testing.assert_array_equal(outputs[0][2]["decision_steps"], outputs[1][2]["decision_steps"])
    return {"lower_batches": [out[0] for out in outputs], "upper_batches": [out[1] for out in outputs],
        "rows": [out[2] for out in outputs], "innovations": [out[3] for out in outputs], "pairing": "passed"}


def score_upper(actor, pair_groups, *, horizon, period, cost, protocol=spec):
    """Compute separate paired MC score gradients for the registered A/B arms."""
    gradients, states, score_costs, signal_rms = {}, [], {}, {}
    decisions = horizon // period
    for name, pairs in pair_groups.items():
        returns = np.asarray([[independent.exact_returns(batch, 1.) for batch in pair["lower_batches"]]
            for pair in pairs], dtype=np.float64)
        for pair_returns, pair in zip(returns, pairs):
            for values, row in zip(pair_returns, pair["rows"]):
                np.testing.assert_allclose(values[0], row["episode_return"], atol=.002, rtol=0)
                cost["objective_checks"] += 1; cost["mc_calls"] += 1
        signal = source.scenario.leave_other_out(returns).reshape(-1)
        batches = [batch for pair in pairs for batch in pair["upper_batches"]]
        upper = concat_level_batches(batches)
        repeated = np.repeat(signal, decisions)
        scored, score_cost = lower_training.residual_actor_gradients(actor, upper, {"scenario": repeated},
            clip_ratio=.2, chunk_size=protocol.CHUNK_SIZE)
        gradients[name] = scored["scenario"]
        states.append(upper.state)
        score_costs[name] = score_cost
        signal_rms[name] = float(np.sqrt(np.square(signal).mean()))
        cost["actor_score_forward_batches"] += score_cost["actor_score_forward_batches"]
        cost["actor_score_backward_batches"] += score_cost["actor_score_backward_batches"]
    return {"gradients": gradients, "states": np.concatenate(states), "score_costs": score_costs,
        "signal_rms": signal_rms}


def evaluation_group(job):
    source_weights, lower_state, upper_state, seed, period, predictor, calibration, args = job
    rows, innovations = {}, {}
    variants = {"forecast": ("forecast", None), "learned": ("learned", upper_state),
        "learned_blinded": ("forecast", upper_state)}
    for variant, (arm, state) in variants.items():
        _, _, row, innovation = native_episode(source_weights, lower_state, state, seed=seed, noise_seed=seed,
            arm=arm, period=period, predictor=predictor, calibration=calibration, args=args,
            collect=False, upper_sample=False)
        row.update(variant=variant, state_arm=arm)
        rows[variant], innovations[variant] = row, innovation
    reference = innovations["forecast"]
    for innovation in innovations.values():
        np.testing.assert_allclose(innovation, reference, atol=3e-5, rtol=0)
    return {"seed": seed, "evaluation": rows, "pairing": "passed"}


def paired_effects(period, evaluation, seeds, protocol=spec):
    if set(evaluation) != set(protocol.ARMS) or any([row["seed"] for row in rows] != seeds for rows in evaluation.values()):
        raise ValueError("upper residual evaluation roster changed")
    return {f"{period}/{a}_minus_{b}": float(np.mean([x["episode_return"] - y["episode_return"]
        for x, y in zip(evaluation[a], evaluation[b])])) for a, b in protocol.CONTRASTS}


def check_row(row, *, period, horizon, variant=None):
    arm = row["state_arm"] if "state_arm" in row else row["arm"]
    learned = arm == "learned"
    expected_upper = horizon // period if learned else 0
    expected = {
        "episode_length": horizon,
        "lower_calls": horizon,
        "upper_calls": expected_upper,
        "decision_steps": list(range(0, horizon, period)) if learned else [],
    }
    if any(row[key] != value for key, value in expected.items()) or row["network_check"] != "passed":
        raise ValueError("upper residual native schedule changed")
    if row["plan_renewals"] != horizon // period:
        raise ValueError("upper residual plan renewal schedule changed")
    if variant is not None and row["variant"] != variant:
        raise ValueError("upper residual evaluation variant changed")


def load_lower_state(root, period, protocol=spec):
    path = protocol.lower_checkpoint(root, period)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("protocol"), payload.get("root"), payload.get("period"), payload.get("arm")) != (
            "pointmaze_option_residual_train_stage112_v1", root, period, "learned"):
        raise ValueError("Stage112 learned lower checkpoint changed")
    return payload["weights"]


def run(root, *, preflight, output, protocol=spec, score_fn=score_upper, qualify_fn=None,
        upper_factory=upper_branch, training_pair_fn=training_pair,
        evaluation_group_fn=evaluation_group):
    if qualify_fn is None:
        qualify_fn = qualify
    if root not in protocol.roots(preflight=preflight):
        raise ValueError("upper residual root changed")
    source_cell = json.loads(protocol.source_result(root).read_text())
    source.qualify(source_cell, preflight=False)
    models, predictor, _, calibrations = source.load_source(root)
    args, roles, options = protocol.arguments(root, preflight=preflight), protocol.seed_roles(root, preflight=preflight), protocol.options(preflight=preflight)
    cost, groups = dict.fromkeys(protocol.budget(preflight=preflight), 0), {}
    cost.update(source_cell_loads=1, source_clone_loads=len(models), lower_checkpoint_loads=len(models),
        upper_branch_initializations=len(models))
    with ProcessPoolExecutor(max_workers=options["workers"], mp_context=mp.get_context("spawn"),
            initializer=source.native.init_worker, initargs=(models["50"].config, args)) as pool:
        for period in protocol.PERIODS:
            model = models[str(period)]
            source_snapshot = copy.deepcopy(model.state_dict())
            source_weights = source.native.joint.inference_weights(model)
            lower_state = load_lower_state(root, period, protocol)
            upper = upper_factory(model)
            histories = []
            for iteration, round_roles in enumerate(roles["training_rounds"], 1):
                pair_groups = {}
                for name in ("credit_A", "credit_B"):
                    roster = round_roles[name]
                    jobs = [(source_weights, lower_state, copy.deepcopy(upper.state_dict()), row["scenario_seed"], row["noise_seeds"],
                        period, predictor, calibrations[str(period)], args) for row in roster]
                    pairs = list(pool.map(training_pair_fn, jobs)); pair_groups[name[-1]] = pairs
                    for pair, registered in zip(pairs, roster):
                        if ([row["noise_seed"] for row in pair["rows"]] != registered["noise_seeds"]
                                or any(row["seed"] != registered["scenario_seed"] for row in pair["rows"])
                                or pair["pairing"] != "passed"):
                            raise ValueError("upper residual training seed roster changed")
                        for row in pair["rows"]:
                            check_row(row, period=period, horizon=args.horizon)
                            cost["training_episodes"] += 1; cost["native_episodes"] += 1; cost["native_steps"] += args.horizon
                            cost["native_lower_calls"] += args.horizon; cost["native_upper_calls"] += row["upper_calls"]
                            cost["native_network_checks"] += 1; cost["planning_renewals"] += row["plan_renewals"]
                            cost["planning_fits"] += row["plan_ols_fits"]; cost["planning_predictions"] += row["plan_ridge_predictions"]
                            cost["planning_reference_calls"] += row["reference_evaluations"]; cost["planning_context_calls"] += row["actor_context_evaluations"]
                        cost["scenario_pair_checks"] += 1
                score = score_fn(upper, pair_groups, horizon=args.horizon, period=period, cost=cost, protocol=protocol)
                history = lower_training.residual_update(upper, score, cost=cost)
                history["iteration"] = iteration; histories.append(history)
            evaluation = {variant: [] for variant in protocol.ARMS}
            for seed in roles["native_evaluation"]:
                group = pool.submit(evaluation_group_fn, (source_weights, lower_state, copy.deepcopy(upper.state_dict()), seed,
                    period, predictor, calibrations[str(period)], args)).result()
                if group["pairing"] != "passed": raise ValueError("upper residual pairing failed")
                for variant, row in group["evaluation"].items():
                    check_row(row, period=period, horizon=args.horizon, variant=variant)
                    evaluation[variant].append(row); cost["evaluation_episodes"] += 1; cost["native_episodes"] += 1
                    cost["native_steps"] += args.horizon; cost["native_lower_calls"] += args.horizon
                    cost["native_upper_calls"] += row["upper_calls"]; cost["native_network_checks"] += 1
                    cost["planning_renewals"] += row["plan_renewals"]; cost["planning_fits"] += row["plan_ols_fits"]
                    cost["planning_predictions"] += row["plan_ridge_predictions"]; cost["planning_reference_calls"] += row["reference_evaluations"]
                    cost["planning_context_calls"] += row["actor_context_evaluations"]
                cost["native_pair_checks"] += 1
            source.native.curves.support.assert_frozen(model, source_snapshot)
            cost["frozen_model_checks"] += 1
            torch.testing.assert_close(upper.base.state_dict(), upper_factory(model).base.state_dict(), atol=0, rtol=0)
            if not preflight:
                path = output.parent / "final_weights" / f"period_{period}_upper.pt"; path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"protocol": protocol.EXPERIMENT_PROTOCOL, "root": root, "period": period, "weights": upper.state_dict()}, path)
            cost["checkpoint_writes"] += int(not preflight)
            groups[str(period)] = {"evaluation": evaluation, "effects": paired_effects(period, evaluation, roles["native_evaluation"], protocol),
                "history": histories, "source_and_lower_unchanged": "passed"}
    if cost != protocol.budget(preflight=preflight):
        details = {k: (cost[k], protocol.budget(preflight=preflight)[k]) for k in cost if cost[k] != protocol.budget(preflight=preflight)[k]}
        raise ValueError(f"upper residual budget mismatch: {details}")
    result = {"status": "complete", "protocol": protocol.EXPERIMENT_PROTOCOL, "contract": protocol.contract(), "root": root,
        "preflight": preflight, "seed_roles": roles, "cost": cost, "groups": groups}
    qualify_fn(result, preflight=preflight)
    write_json(output, result); write_json(Path(output).parent / "completion" / "ready.json",
        {"protocol": protocol.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return result


def qualify(cell, *, preflight):
    root = cell["root"]
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL
            or cell["contract"] != spec.contract() or root not in spec.roots(preflight=preflight)
            or cell["preflight"] != preflight or cell["seed_roles"] != spec.seed_roles(root, preflight=preflight)
            or cell["cost"] != spec.budget(preflight=preflight)
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}
            or any(cell.get(key, 0) for key in ("optimizer_steps", "critic_fits", "native_trace_writes"))):
        raise ValueError("Stage113 protocol, source, roster, budget or frozen path changed")
    horizon = spec.arguments(root, preflight=preflight).horizon
    for period, group in cell["groups"].items():
        if group["source_and_lower_unchanged"] != "passed":
            raise ValueError("Stage113 source or lower branch freeze failed")
        effects = paired_effects(int(period), group["evaluation"], cell["seed_roles"]["native_evaluation"])
        if group["effects"] != effects or not np.isfinite(list(effects.values())).all():
            raise ValueError("Stage113 paired evaluation changed")
        if set(group["evaluation"]) != set(spec.ARMS):
            raise ValueError("Stage113 evaluation roster changed")
        for variant, rows in group["evaluation"].items():
            for row in rows:
                check_row(row, period=int(period), horizon=horizon, variant=variant)
                if row["seed"] not in cell["seed_roles"]["native_evaluation"]:
                    raise ValueError("Stage113 evaluation seed changed")
    return cell


def aggregate(cells, *, preflight):
    statistics = __import__("freq_hrl.experiments.pointmaze_feasible_credit", fromlist=["aggregate"])
    result = statistics.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    passed = {} if preflight else {str(period): all(
        result["endpoints"][f"{period}/{a}_minus_{b}"]["ci"][0] > 0 for a, b in spec.CONTRASTS)
        for period in spec.PERIODS}
    result.update(upper_gain_gate="mechanical_only" if preflight else
        "supported_both_periods" if all(passed.values()) else
        "partial" if any(passed.values()) else "not_supported",
        period_upper_gain_gate=passed,
        performance_claim="forecast_anchored_upper_residual_gain_with_learned_blinded_ablation",
        lower_source="Stage112 learned lower branch frozen")
    return result
