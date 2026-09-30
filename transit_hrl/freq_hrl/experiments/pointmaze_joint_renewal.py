"""Native joint PPO training of waypoints, feedback control and plan renewal."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from freq_hrl.domains.mujoco import RelativeSubgoalAdapter
from freq_hrl.rl.smdp_actor_critic import (
    FrequencySeparatedActorCriticPPO, HierarchicalRolloutBuilder,
    PromotionRolloutBuilder, concat_hierarchical_batches,
)
from .pointmaze_goal_validation import POINTMAZE_LOWER_ACTION_COST, pointmaze_goal_bounds, squash_box_action
from .pointmaze_plan_validity_branching import _task_options
from .pointmaze_plan_value_qualification import PointMazeRegimeFeatureBuilder, _make_task
from .pointmaze_root_response import load_controller, raw_directory, scale_for, write_json
from scripts import pointmaze_joint_renewal_stage35_spec as spec


def gate_state(observation, history, subgoal, *, age, step, horizon, current_only):
    measured = history.history
    if current_only:
        measured = np.tile(observation.task_measurement, history.time_scale.history_steps)
    return np.concatenate((observation.physical, observation.target_error, measured,
                           subgoal - observation.achieved_goal,
                           [age / spec.MAX_AGE_STEPS, (horizon - step) / horizon])).astype(np.float32)


def inference_weights(model):
    names = ["upper_actor", "lower_actor", "upper_value", "lower_value"]
    if model.promotion_actor is not None:
        names += ["promotion_actor", "promotion_value"]
    return {name: {key: value.detach().cpu().clone() for key, value in getattr(model, name).state_dict().items()}
            for name in names}


def make_model(controller, method, *, root):
    torch.manual_seed(root + 35003)
    config = replace(controller.config,
                     promotion_state_dim=controller.config.upper_state_dim + 4 if method.startswith("learned_") else 0,
                     promotion_init_logit=0., promotion_learning_rate=3e-4,
                     promotion_entropy_coef=.01)
    model = FrequencySeparatedActorCriticPPO(config)
    for name in ("upper_actor", "lower_actor", "upper_value", "lower_value"):
        getattr(model, name).load_state_dict(getattr(controller, name).state_dict())
    return model


def rollout(model, args, method, *, seed, sample, capture=False, gate_sample=None, gate_seed=None,
            lower_credit="intrinsic_option", lower_sample=None, upper_sample=None, lower_seed=None,
            lower_value_context_builder=None, lower_reference_builder=None, lower_actor_context_builder=None,
            upper_plan_decoder=None):
    if lower_credit not in ("intrinsic_option", "intrinsic_episode", "task_option", "task_episode"):
        raise ValueError("unregistered lower credit")
    scale = scale_for(args)
    task = _make_task(env_id=args.env_id, seed=seed, horizon=args.horizon, **_task_options(args))
    try:
        observation = task.reset()
        low, high = pointmaze_goal_bounds(task.environment)
        adapter = RelativeSubgoalAdapter(maximum_delta=np.full(2, args.maximum_subgoal_delta, dtype=np.float32),
                                        world_low=low, world_high=high, action_cost=POINTMAZE_LOWER_ACTION_COST)
        history = PointMazeRegimeFeatureBuilder(time_scale=scale)
        history.reset(observation)
        model.reset_recurrent_inference()
        builder = HierarchicalRolloutBuilder(gamma=model.config.gamma) if sample else None
        gate_builder = PromotionRolloutBuilder(gamma=model.config.gamma) if sample else None
        decisions, gate_steps, gate_actions, gate_states, gate_probabilities = [], [], [], [], []
        trace = {key: [] for key in ("physical", "measurement", "achieved_before", "target_before",
                                     "subgoal_before", "subgoal", "achieved_after", "distance", "reward", "action")}
        if lower_value_context_builder is not None:
            trace["lower_value_context"] = []
        if lower_reference_builder is not None:
            trace["lower_reference"] = []
        if lower_actor_context_builder is not None:
            trace["lower_actor_context"] = []
        rewards, distances = [], []
        last_plan = -spec.MAX_AGE_STEPS
        subgoal = observation.achieved_goal.copy()
        upper_time = lower_time = gate_time = 0.
        started = time.perf_counter()
        for step in range(args.horizon):
            age = step - last_plan
            plan_now = step == 0
            previous_subgoal = subgoal.copy()
            if step:
                if method in ("fixed50", "fixed100"):
                    plan_now = age >= int(method[5:])
                elif age >= spec.MAX_AGE_STEPS:
                    plan_now = True
                elif age >= spec.CHECK_STEPS and step % spec.CHECK_STEPS == 0:
                    state = gate_state(observation, history, subgoal, age=age, step=step,
                                       horizon=args.horizon, current_only=method == "learned_current")
                    clock = time.perf_counter()
                    if gate_seed is not None:
                        torch.manual_seed(int(gate_seed) + step)
                    output = model.act_promotion(state, sample=sample if gate_sample is None else gate_sample)
                    gate_time += time.perf_counter() - clock
                    plan_now = bool(output["action"])
                    gate_steps.append(step)
                    gate_actions.append(int(plan_now))
                    if capture:
                        gate_states.append(state)
                        gate_probabilities.append(output["probability"])
                    if sample:
                        gate_builder.begin(state=state, action=float(plan_now), logp=output["logp"], value=output["value"])
            if plan_now:
                state = history.upper_state(observation, oracle_context=None)
                clock = time.perf_counter()
                output = model.act_upper(state, sample=sample if upper_sample is None else upper_sample)
                upper_time += time.perf_counter() - clock
                if upper_plan_decoder is None:
                    subgoal = adapter.decode(np.asarray(output["action"], dtype=np.float32), observation.achieved_goal)
                else:
                    subgoal = upper_plan_decoder(action=np.asarray(output["action"], dtype=np.float32),
                        observation=observation, history=history, step=step, world_low=low, world_high=high)
                if sample:
                    builder.begin_upper(state=state, action=output["action"], logp=output["logp"], value=output["value"])
                decisions.append(step)
                last_plan = step
            # Keep the upper anchor separate from its option-phase reference.
            reference = subgoal if lower_reference_builder is None else lower_reference_builder(
                observation=observation, history=history, subgoal=subgoal, age=step - last_plan,
                step=step, world_low=low, world_high=high)
            state = history.lower_state(observation, subgoal=reference)
            value_state, value_context = None, None
            if lower_value_context_builder is not None:
                value_context = lower_value_context_builder(age=step - last_plan, step=step, horizon=args.horizon)
                value_state = np.concatenate((state, value_context)).astype(np.float32)
            actor_context, cost_state = None, None
            if lower_actor_context_builder is not None:
                actor_context = lower_actor_context_builder(age=step - last_plan, step=step, horizon=args.horizon)
                cost_state = state if model.lower_cost_value is not None else None
                if value_state is None:
                    value_state = state
                state = np.concatenate((state, actor_context)).astype(np.float32)
            clock = time.perf_counter()
            if lower_seed is not None and (sample if lower_sample is None else lower_sample):
                torch.manual_seed(int(lower_seed) + step)
            value_kwargs = {} if value_state is None else {"value_state": value_state}
            if cost_state is not None:
                value_kwargs["cost_state"] = cost_state
            output = model.act_lower(state, sample=sample if lower_sample is None else lower_sample, **value_kwargs)
            lower_time += time.perf_counter() - clock
            action = squash_box_action(np.asarray(output["action"], dtype=np.float32), task.action_low, task.action_high)
            after, reward, terminated, truncated, info = task.step(action)
            done = bool(terminated or truncated)
            if done and step + 1 != args.horizon:
                raise RuntimeError("joint-renewal native episode ended early")
            charged = float(reward) - spec.CALL_COST * int(plan_now)
            if sample:
                lower_reward = float(reward) if lower_credit.startswith("task_") else adapter.intrinsic_reward(
                    achieved_before=observation.achieved_goal, achieved_after=after.achieved_goal,
                    subgoal=reference, action=action)
                builder.add_lower(state=state, action=output["action"], logp=output["logp"], value=output["value"],
                                  reward=float(lower_reward), upper_reward=charged, done=done, cost=0.,
                                  value_state=value_state, cost_state=cost_state)
                gate_builder.add_reward(charged, done=done)
            rewards.append(float(reward))
            distances.append(float(info["tracking_distance"]))
            if capture:
                values = (observation.physical, observation.task_measurement, observation.achieved_goal,
                          observation.target, previous_subgoal, subgoal, after.achieved_goal,
                          info["tracking_distance"], reward, action)
                for key, value in zip(trace, values):
                    trace[key].append(np.asarray(value).copy())
                if value_context is not None:
                    trace["lower_value_context"].append(np.asarray(value_context).copy())
                if lower_reference_builder is not None:
                    trace["lower_reference"].append(np.asarray(reference).copy())
                if actor_context is not None:
                    trace["lower_actor_context"].append(np.asarray(actor_context).copy())
            observation = after
            history.update(observation)
        wall = time.perf_counter() - started
        durations = np.diff([*decisions, args.horizon])
        if min(durations) < spec.CHECK_STEPS or max(durations) > spec.MAX_AGE_STEPS:
            raise RuntimeError("joint-renewal option duration outside registered range")
        batch = None
        if sample:
            builder.finish(terminal=True)
            gate_builder.finish(terminal=True)
            batch = builder.build()
            # Renewal is chosen from the next observed state, so mark these
            # option boundaries after their actual decision times are known.
            if lower_credit.endswith("_option"):
                batch.lower.done[np.asarray(decisions[1:], dtype=int) - 1] = 1.
            batch.promotion = gate_builder.build() if method.startswith("learned_") else None
        reward_sum = float(np.sum(rewards))
        row = {"seed": int(seed), "method": method, "episode_length": args.horizon,
               "episode_return": reward_sum, "tracking_squared_error_integral": float(np.dot(distances, distances) * scale.dt_seconds),
               "upper_inference_calls": len(decisions), "candidate_preview_calls": 0,
               "lower_inference_calls": args.horizon, "gate_inference_calls": len(gate_steps),
               "charged_utility": reward_sum - spec.CALL_COST * len(decisions),
               "decision_steps": decisions, "gate_steps": gate_steps, "gate_actions": gate_actions,
               "gate_sample": bool(sample if gate_sample is None else gate_sample), "gate_seed": gate_seed,
               "lower_sample": bool(sample if lower_sample is None else lower_sample),
               "upper_sample": bool(sample if upper_sample is None else upper_sample), "lower_seed": lower_seed,
               "upper_inference_seconds": upper_time, "lower_inference_seconds": lower_time,
               "gate_inference_seconds": gate_time, "episode_wall_seconds": wall}
        if sample:
            row["lower_training_credit"] = {
                "mode": lower_credit, "primitive_steps": batch.lower.size,
                "reward_sum": float(np.sum(batch.lower.reward, dtype=np.float64)),
                "task_reward_sum": float(np.sum(np.asarray(rewards, dtype=np.float32), dtype=np.float64)),
                "done_count": int(np.sum(batch.lower.done)), "option_count": len(decisions)}
        raw = {key: np.asarray(value) for key, value in trace.items()} if capture else None
        if capture:
            raw.update(decision_steps=np.asarray(decisions), gate_steps=np.asarray(gate_steps),
                       gate_actions=np.asarray(gate_actions), gate_states=np.asarray(gate_states),
                       gate_probabilities=np.asarray(gate_probabilities))
        return batch, row, raw
    finally:
        task.environment.close()


_WORKER = None


def init_worker(config, args, method, lower_credit="intrinsic_option"):
    global _WORKER
    torch.set_num_threads(1)
    _WORKER = FrequencySeparatedActorCriticPPO(config), args, method, lower_credit


def worker_rollout(job):
    weights, seed, sample, raw_path = job
    model, args, method, lower_credit = _WORKER
    model.load_state_dict(weights)
    torch.manual_seed(int(seed) + args.optimizer_seed)
    batch, row, raw = rollout(model, args, method, seed=seed, sample=sample,
                              capture=raw_path is not None, lower_credit=lower_credit)
    if raw_path is not None:
        np.savez_compressed(raw_path, **raw)
    return batch, row


def train(root, method, *, preflight, output):
    if method not in spec.METHODS:
        raise ValueError("unregistered joint-renewal method")
    args = spec.source.arguments(root, preflight=preflight)
    opt, roles = spec.options(preflight=preflight), spec.seed_roles(root, preflight=preflight)
    controller, source_cell, replay = load_controller(args, spec.source_result(root, preflight=preflight))
    model = make_model(controller, method, root=root)
    initial_weights = inference_weights(model)
    raw = raw_directory(output)
    history, costs, updates = [], [], {}
    best_rank, best_iteration = None, None
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"),
                             initializer=init_worker, initargs=(model.config, args, method)) as pool:
        def episodes(seeds, *, sample, capture=False):
            weights = inference_weights(model)
            jobs = [(weights, seed, sample, str(raw / f"episode_{seed}.npz") if capture else None) for seed in seeds]
            pairs = list(pool.map(worker_rollout, jobs))
            costs.extend({"phase": "train" if sample else "eval" if capture else "selection",
                          **{key: row[key] for key in ("upper_inference_calls", "lower_inference_calls", "gate_inference_calls")}}
                         for _, row in pairs)
            return pairs

        def select(iteration):
            nonlocal best_rank, best_iteration
            rows = [row for _, row in episodes(roles["selection"], sample=False)]
            rank = (float(np.mean([row["charged_utility"] for row in rows])),
                    -float(np.mean([row["tracking_squared_error_integral"] for row in rows])))
            history.append({"iteration": iteration, "utility": rank[0], "ise": -rank[1],
                            "return": float(np.mean([row["episode_return"] for row in rows])),
                            "calls": float(np.mean([row["upper_inference_calls"] for row in rows]))})
            if best_rank is None or rank > best_rank:
                best_rank, best_iteration = rank, iteration
                torch.save({"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "method": method,
                            "iteration": iteration, "state_dict": model.state_dict()}, raw / "selected.pt")
            print(f"root {root} {method}: selection iteration {iteration}; elapsed {time.monotonic() - started:.1f}s", flush=True)

        select(0)
        for iteration in range(1, opt["iterations"] + 1):
            offset = (iteration - 1) * opt["rollouts_per_iteration"]
            pairs = episodes(roles["training"][offset:offset + opt["rollouts_per_iteration"]], sample=True)
            metrics = model.update(concat_hierarchical_batches([batch for batch, _ in pairs]))
            for key, value in metrics.items():
                if "optimizer_steps" in key:
                    updates[key] = updates.get(key, 0) + int(value)
            if iteration % opt["selection_interval"] == 0:
                select(iteration)
        trained_weights = inference_weights(model)
        trained_changes = {name: float(np.sqrt(sum(float(torch.sum((weights[key] - initial_weights[name][key]) ** 2))
                                                 for key in weights))) for name, weights in trained_weights.items()}
        torch.save({"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "method": method,
                    "iteration": opt["iterations"], "state_dict": model.state_dict()}, raw / "final.pt")
        selected = torch.load(raw / "selected.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(selected["state_dict"])
        final_weights = inference_weights(model)
        changes = {name: float(np.sqrt(sum(float(torch.sum((weights[key] - initial_weights[name][key]) ** 2))
                                         for key in weights))) for name, weights in final_weights.items()}
        evaluation = [row for _, row in episodes(roles["evaluation"], sample=False, capture=True)]
    totals = {phase: {key: sum(row[key] for row in costs if row["phase"] == phase)
                      for key in ("upper_inference_calls", "lower_inference_calls", "gate_inference_calls")}
              for phase in ("train", "selection", "eval")}
    totals["factual_replay"] = {"upper_inference_calls": len(source_cell["factual_row"]["decision_steps"]),
                                "lower_inference_calls": args.horizon, "gate_inference_calls": 0}
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
              "root": root, "method": method, "preflight": preflight, "options": opt, "seed_roles": roles,
              "budget": spec.budget(preflight=preflight), "inference_counts": totals,
              "source_selected_iteration": source_cell["selected_checkpoint_iteration"], "source_replay": replay,
              "checkpoint": str(raw / "selected.pt"), "selected_iteration": best_iteration,
              "final_checkpoint": str(raw / "final.pt"),
              "selection_history": history, "optimizer_steps": updates,
              "trained_parameter_change_norms": trained_changes,
              "selected_parameter_change_norms": changes,
              "parameter_count": {name: sum(p.numel() for p in getattr(model, name).parameters()) for name in final_weights},
              "evaluation_rows": evaluation, "wall_seconds": time.monotonic() - started}
    write_json(output, result)
    return result


def audit_result(result, *, raw_path):
    root, method, preflight = result["root"], result["method"], result["preflight"]
    if (result["status"] != "complete" or result["protocol"] != spec.EXPERIMENT_PROTOCOL
            or result["contract"] != spec.contract() or result["options"] != spec.options(preflight=preflight)
            or result["seed_roles"] != spec.seed_roles(root, preflight=preflight)
            or result["budget"] != spec.budget(preflight=preflight)):
        raise ValueError("joint-renewal result violates the frozen protocol")
    args = spec.source.arguments(root, preflight=preflight)
    if method not in spec.METHODS:
        raise ValueError("unregistered treatment")
    for phase, field in (("train", "training_primitive_steps"), ("selection", "selection_primitive_steps"),
                         ("eval", "evaluation_primitive_steps"), ("factual_replay", "factual_replay_primitive_steps")):
        if result["inference_counts"][phase]["lower_inference_calls"] != result["budget"][field]:
            raise ValueError("primitive training/selection/evaluation accounting changed")
    levels = ("upper", "lower", "promotion") if method.startswith("learned_") else ("upper", "lower")
    for level in levels:
        if result["optimizer_steps"].get(level + "_actor_optimizer_steps", 0) < 1:
            raise ValueError("registered actor was never updated")
        if result["trained_parameter_change_norms"][level + "_actor"] <= 0:
            raise ValueError("registered actor weights did not change")
    candidates = [row["iteration"] for row in result["selection_history"]]
    opt = result["options"]
    if candidates != [0, *range(opt["selection_interval"], opt["iterations"] + 1, opt["selection_interval"])]:
        raise ValueError("checkpoint selection budget changed")
    best = max(result["selection_history"], key=lambda row: (row["utility"], -row["ise"]))
    if result["selected_iteration"] != best["iteration"]:
        raise ValueError("checkpoint selection used an unregistered objective")
    rows = result["evaluation_rows"]
    if [row["seed"] for row in rows] != result["seed_roles"]["evaluation"]:
        raise ValueError("joint-renewal evaluation roster incomplete")
    audit_trajectories(rows, args=args, method=method, raw_path=raw_path)
    return {"status": "passed", "root": root, "method": method, "episodes": len(rows),
            "candidate_preview_calls": 0, "checks": "native_metrics_causal_gate_execution_and_call_accounting"}


def audit_trajectories(rows, *, args, method, raw_path):
    """Validate executed renewal trajectories shared by the training ablations."""
    for row in rows:
        seed = row["seed"]
        with np.load(Path(raw_path) / f"episode_{seed}.npz") as archive:
            trace = {key: archive[key] for key in archive.files}
            steps = np.asarray(row["decision_steps"])
            np.testing.assert_array_equal(trace["decision_steps"], steps)
            if steps[0] != 0 or len(np.unique(steps)) != len(steps):
                raise ValueError("invalid native upper schedule")
            duration = np.diff([*steps, args.horizon])
            if np.any(duration < spec.CHECK_STEPS) or np.any(duration > spec.MAX_AGE_STEPS):
                raise ValueError("native option duration changed")
            if method.startswith("fixed"):
                np.testing.assert_array_equal(steps, np.arange(0, args.horizon, int(method[5:])))
            np.testing.assert_allclose(trace["distance"], np.linalg.norm(trace["achieved_after"] - trace["target_before"], axis=1), atol=1e-7, rtol=1e-6)
            np.testing.assert_allclose(trace["reward"], np.exp(-trace["distance"]), atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(row["episode_return"], trace["reward"].sum(), atol=1e-8, rtol=0)
            np.testing.assert_allclose(row["tracking_squared_error_integral"], np.dot(trace["distance"], trace["distance"]) * .01, atol=1e-8, rtol=0)
            np.testing.assert_allclose(row["charged_utility"], row["episode_return"] - spec.CALL_COST * len(steps), atol=1e-8, rtol=0)
            if row["upper_inference_calls"] != len(steps) or row["candidate_preview_calls"] or row["lower_inference_calls"] != args.horizon:
                raise ValueError("native inference accounting changed")
            gates = dict(zip(row["gate_steps"], row["gate_actions"]))
            np.testing.assert_array_equal(trace["gate_steps"], row["gate_steps"])
            np.testing.assert_array_equal(trace["gate_actions"], row["gate_actions"])
            if row["gate_inference_calls"] != len(gates):
                raise ValueError("native gate accounting changed")
            last = 0
            index = 0
            for step in range(1, args.horizon):
                age = step - last
                expected = age >= int(method[5:]) if method.startswith("fixed") else age >= spec.MAX_AGE_STEPS
                eligible = method.startswith("learned_") and spec.CHECK_STEPS <= age < spec.MAX_AGE_STEPS and step % spec.CHECK_STEPS == 0
                if eligible:
                    if step not in gates:
                        raise ValueError("missing eligible renewal decision")
                    expected = bool(gates[step])
                    measured = trace["measurement"][max(0, step - 63):step + 1]
                    measured = np.concatenate((np.repeat(trace["measurement"][:1], 64 - len(measured), axis=0), measured)).reshape(-1)
                    if method == "learned_current":
                        measured = np.tile(trace["measurement"][step], 64)
                    state = np.concatenate((trace["physical"][step], trace["target_before"][step] - trace["achieved_before"][step],
                                            measured, trace["subgoal_before"][step] - trace["achieved_before"][step],
                                            [age / spec.MAX_AGE_STEPS, (args.horizon - step) / args.horizon])).astype(np.float32)
                    np.testing.assert_array_equal(trace["gate_states"][index], state)
                    index += 1
                elif step in gates:
                    raise ValueError("unexpected renewal decision")
                if expected != (step in steps):
                    raise ValueError("renewal action was not executed")
                if expected:
                    last = step
                else:
                    np.testing.assert_array_equal(trace["subgoal"][step], trace["subgoal"][step - 1])


def aggregate(results):
    full = {(result["root"], result["method"]): result for result in results}
    if len(results) != len(spec.OPTIMIZER_ROOTS) * len(spec.METHODS) or set(full) != {(root, method) for root in spec.OPTIMIZER_ROOTS for method in spec.METHODS}:
        raise ValueError("full joint-renewal root/method roster incomplete")
    vectors = []
    root_rows = []
    means = {method: {} for method in spec.METHODS}
    keys = ("episode_return", "tracking_squared_error_integral", "upper_inference_calls", "charged_utility")
    for root in spec.OPTIMIZER_ROOTS:
        values = {method: {key: float(np.mean([row[key] for row in full[root, method]["evaluation_rows"]])) for key in keys} for method in spec.METHODS}
        h, f = values["learned_history"], values["fixed50"]
        vector = [h["episode_return"] - f["episode_return"], f["tracking_squared_error_integral"] - h["tracking_squared_error_integral"],
                  f["upper_inference_calls"] - h["upper_inference_calls"],
                  h["charged_utility"] - values["fixed100"]["charged_utility"],
                  h["charged_utility"] - values["learned_current"]["charged_utility"]]
        vectors.append(vector)
        root_rows.append({"root": root, "means": values, "endpoints": dict(zip(spec.ENDPOINTS, vector))})
    x = np.asarray(vectors)
    rng = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED))
    draws = x[rng.integers(0, len(x), (spec.BOOTSTRAP_DRAWS, len(x)))].mean(axis=1)
    bounds = np.quantile(draws, [.05 / (2 * len(spec.ENDPOINTS)), 1 - .05 / (2 * len(spec.ENDPOINTS))], axis=0)
    endpoints = {name: {"mean": float(x[:, i].mean()), "ci": bounds[:, i].tolist(), "supported": bool(bounds[0, i] > 0)} for i, name in enumerate(spec.ENDPOINTS)}
    for method in spec.METHODS:
        means[method] = {key: float(np.mean([row["means"][method][key] for row in root_rows])) for key in keys}
    return {"status": "stage35_development_gate_passed" if all(item["supported"] for item in endpoints.values()) else "stage35_development_gate_failed",
            "root_count": len(x), "primary_endpoints": endpoints, "means": means, "root_rows": root_rows}
