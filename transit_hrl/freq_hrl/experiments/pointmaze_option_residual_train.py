"""Train only the qualified optional action residual above a frozen flat donor."""
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
import time

import numpy as np
import torch

from freq_hrl.rl.optional_action_residual import OptionalActionResidual
from freq_hrl.rl.smdp_actor_critic import LevelTrajectoryBatch, concat_level_batches
from . import pointmaze_independent_credit as independent
from . import pointmaze_option_residual as source
from . import pointmaze_feasible_credit as statistics
from .pointmaze_root_response import write_json
from scripts import pointmaze_option_residual_train_stage112_spec as spec


def branch(model):
    return OptionalActionResidual(model.lower_actor, feedback_dim=392, advice_dim=4)


def branch_weights(model):
    return copy.deepcopy(branch(model).state_dict())


def plan_for_arm(model, arm, period, predictor, calibration, args):
    if arm == "blind":
        return None
    if arm == "forecast":
        return source.baseline.forecast.PlanReference("ridge_velocity", predictor, period)
    if arm == "learned":
        return source.native.curves.CalibratedPlan(predictor, period, args.maximum_subgoal_delta,
            calibration["alpha"], calibration["envelope"])
    raise ValueError(f"unknown residual arm: {arm}")


def native_episode(source_weights, residual_state, *, seed, noise_seed, arm, period, predictor,
        calibration, args, collect):
    model, _ = source.native._WORKER
    model.load_state_dict(source_weights)
    residual = branch(model)
    residual.load_state_dict(residual_state)
    policy_seed, lower_seed = source.scenario.spec.noise_seeds(args.optimizer_seed, seed, noise_seed)
    torch.manual_seed(policy_seed)
    plan = plan_for_arm(model, arm, period, predictor, calibration, args)
    scale = source.native.joint.scale_for(args)
    task = source.native.joint._make_task(env_id=args.env_id, seed=seed, horizon=args.horizon,
        **source.native.joint._task_options(args))
    try:
        observation = task.reset()
        low, high = source.native.joint.pointmaze_goal_bounds(task.environment)
        history = source.native.joint.PointMazeRegimeFeatureBuilder(time_scale=scale)
        history.reset(observation)
        model.reset_recurrent_inference()
        data = {key: [] for key in ("state", "value_state", "action", "reward", "old_logp", "old_value")}
        rewards, distances, innovations, decisions = [], [], [], []
        initial_feedback = None
        for step in range(args.horizon):
            feedback = source.baseline.flat_state(history, observation)
            if initial_feedback is None:
                initial_feedback = feedback.copy()
            if plan is not None and arm == "learned" and step % period == 0:
                upper = model.act_upper(history.upper_state(observation, oracle_context=None), sample=False)
                plan.decode(action=np.asarray(upper["action"], dtype=np.float32), observation=observation,
                    history=history, step=step, world_low=low, world_high=high)
                decisions.append(step)
            if plan is None:
                state = np.r_[feedback, np.zeros(4, dtype=np.float32)]
            else:
                reference = plan(observation=observation, history=history, subgoal=None, age=step % period,
                    step=step, world_low=low, world_high=high)
                velocity = plan.actor_context(age=step % period, step=step, horizon=args.horizon)
                state = source.advice_state(feedback, observation, reference, velocity)
            value_state = np.r_[state, source.baseline.clocks.time_context(age=step % period,
                step=step, horizon=args.horizon, clock=True)].astype(np.float32)
            torch.manual_seed(lower_seed + step)
            with torch.inference_mode():
                distribution = residual.distribution(torch.as_tensor(state, dtype=torch.float32).view(1, -1))
                action_tensor = distribution.rsample()[0]
                logp = distribution.log_prob(action_tensor).sum()
                value = model.lower_value(torch.as_tensor(value_state, dtype=torch.float32).view(1, -1))[0]
            action = source.native.joint.squash_box_action(action_tensor.numpy(), task.action_low, task.action_high)
            innovations.append(((action_tensor - distribution.mean[0]) / distribution.stddev[0]).numpy())
            after, reward, terminated, truncated, info = task.step(action)
            if bool(terminated or truncated) and step + 1 != args.horizon:
                raise RuntimeError("branch training episode ended before the registered horizon")
            if collect:
                for key, value_out in (("state", state), ("value_state", value_state), ("action", action_tensor.numpy()),
                        ("reward", reward), ("old_logp", logp.item()), ("old_value", value.item())):
                    data[key].append(value_out)
            rewards.append(float(reward))
            distances.append(float(info["tracking_distance"]))
            observation = after
            history.update(observation)
        batch = None
        if collect:
            done = np.zeros(args.horizon, dtype=np.float32)
            done[-1] = 1.
            batch = LevelTrajectoryBatch(**{key: np.asarray(value, dtype=np.float32) for key, value in data.items()},
                duration=np.ones(args.horizon, dtype=np.int64), done=done)
        torch.testing.assert_close(source.native.joint.inference_weights(model), source_weights, atol=0, rtol=0)
        row = {"seed": seed, "noise_seed": noise_seed, "policy_seed": policy_seed, "lower_seed": lower_seed,
            "arm": arm, "episode_return": float(np.sum(rewards)), "episode_length": args.horizon,
            "tracking_squared_error_integral": float(np.dot(distances, distances) * scale.dt_seconds),
            "upper_calls": len(decisions), "lower_calls": args.horizon, "decision_steps": decisions,
            "upper_sample": False, "lower_sample": True, "network_check": "passed",
            "plan_renewals": 0 if plan is None else args.horizon // period,
            "plan_ols_fits": 0 if plan is None else plan.ols_fits,
            "plan_ridge_predictions": 0 if plan is None else plan.ridge_predictions,
            "reference_evaluations": 0 if plan is None else plan.calls,
            "actor_context_evaluations": 0 if plan is None else plan.context_calls,
            "upper_standard_noise": [], "initial_feedback": initial_feedback.tolist()}
        return batch, row, np.asarray(innovations)
    finally:
        task.environment.close()


def training_pair(job):
    source_weights, residual_state, scenario_seed, noise_seeds, arm, period, predictor, calibration, args = job
    outputs = [native_episode(source_weights, residual_state, seed=scenario_seed, noise_seed=noise,
        arm=arm, period=period, predictor=predictor, calibration=calibration, args=args, collect=True)
        for noise in noise_seeds]
    np.testing.assert_array_equal(outputs[0][1]["initial_feedback"], outputs[1][1]["initial_feedback"])
    return {"batches": [out[0] for out in outputs], "rows": [out[1] for out in outputs],
        "innovations": [out[2] for out in outputs], "pairing": "passed"}


def evaluation_group(job):
    source_weights, residual_states, seed, period, predictor, calibration, args = job
    evaluations = {}
    innovations = {}
    for variant in spec.VARIANTS:
        if variant == "base":
            arm, state_arm, residual_state = "blind", "blind", residual_states["base"]
        elif variant in ("blind", "forecast", "learned"):
            arm = state_arm = variant
            residual_state = residual_states[variant]
        elif variant == "forecast_blinded":
            arm, state_arm, residual_state = "blind", "blind", residual_states["forecast"]
        elif variant == "learned_blinded":
            arm, state_arm, residual_state = "blind", "blind", residual_states["learned"]
        else:
            raise ValueError("evaluation variant changed")
        _, row, innovation = native_episode(source_weights, residual_state, seed=seed, noise_seed=seed,
            arm=state_arm, period=period, predictor=predictor, calibration=calibration, args=args, collect=False)
        row.update(variant=variant, state_arm=state_arm, residual_arm=("base" if variant == "base" else
            "forecast" if variant == "forecast_blinded" else "learned" if variant == "learned_blinded" else variant))
        evaluations[variant], innovations[variant] = row, innovation
    reference = innovations["base"]
    for variant in spec.VARIANTS:
        np.testing.assert_allclose(innovations[variant], reference, atol=3e-5, rtol=0)
    return {"seed": seed, "evaluation": evaluations, "pairing": "passed"}


def check_row(row, *, period, horizon, variant=None):
    arm = row["state_arm"] if variant is not None else row["arm"]
    expected_upper = horizon // period if arm == "learned" else 0
    expected_plan = horizon // period if arm in ("forecast", "learned") else 0
    expected = {"episode_length": horizon, "lower_calls": horizon, "upper_calls": expected_upper,
        "decision_steps": list(range(0, horizon, period)) if arm == "learned" else [],
        "upper_sample": False, "lower_sample": True, "network_check": "passed",
        "plan_renewals": expected_plan, "plan_ols_fits": max(0, expected_plan - 1),
        "plan_ridge_predictions": max(0, expected_plan - 1), "reference_evaluations": horizon if expected_plan else 0,
        "actor_context_evaluations": horizon if expected_plan else 0}
    if any(row[key] != value for key, value in expected.items()):
        raise ValueError("branch native call schedule changed")
    if variant is not None and row["variant"] != variant:
        raise ValueError("branch evaluation roster changed")


def score_branch(actor, batches, *, horizon, cost):
    gradients, states, score_costs, signal_rms = {}, [], {}, {}
    for name, groups in batches.items():
        returns = np.asarray([[independent.exact_returns(batch, 1.) for batch, _ in group] for group in groups], dtype=np.float64)
        for group_returns, group in zip(returns, groups):
            for values, (_, row) in zip(group_returns, group):
                np.testing.assert_allclose(values[0], row["episode_return"], atol=.002, rtol=0)
                cost["objective_checks"] += 1
                cost["mc_calls"] += 1
        signal = source.scenario.leave_other_out(returns).reshape(-1)
        lower = concat_level_batches(batch for group in groups for batch, _ in group)
        scored, score_cost = residual_actor_gradients(actor, lower, {"scenario": signal},
            clip_ratio=.2, chunk_size=spec.CHUNK_SIZE)
        gradients[name], states, score_costs[name] = scored["scenario"], states + [lower.state], score_cost
        signal_rms[name] = float(np.sqrt(np.square(signal).mean()))
        cost["actor_score_forward_batches"] += score_cost["actor_score_forward_batches"]
        cost["actor_score_backward_batches"] += score_cost["actor_score_backward_batches"]
    return {"gradients": gradients, "states": np.concatenate(states), "score_costs": score_costs,
        "signal_rms": signal_rms}


def residual_actor_gradients(actor, lower, signals, *, clip_ratio, chunk_size):
    """Compute score gradients only for the trainable residual readout."""
    params = [actor.readout.weight, actor.readout.bias]
    gradients = {key: np.zeros(sum(parameter.numel() for parameter in params), dtype=np.float64)
        for key in (*signals, "entropy")}
    state = torch.as_tensor(lower.state, dtype=torch.float32)
    action = torch.as_tensor(lower.action, dtype=torch.float32)
    old_logp = torch.as_tensor(lower.old_logp, dtype=torch.float32)
    tensor_signals = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in signals.items()}
    count, logp_error = 0, 0.
    for start in range(0, lower.size, chunk_size):
        stop = min(start + chunk_size, lower.size)
        logp, entropy = actor.log_prob_entropy(state[start:stop], action[start:stop])
        logp_error = max(logp_error, float((logp.detach() - old_logp[start:stop]).abs().max()))
        ratio = torch.exp((logp - old_logp[start:stop]).clamp(-20., 20.))
        clipped = ratio.clamp(1. - clip_ratio, 1. + clip_ratio)
        losses = {key: -torch.minimum(ratio * value[start:stop], clipped * value[start:stop]).sum() / lower.size
            for key, value in tensor_signals.items()}
        losses["entropy"] = -entropy.sum() / lower.size
        for index, (key, loss) in enumerate(losses.items()):
            values = (torch.autograd.grad(loss, params, retain_graph=index < len(losses) - 1, allow_unused=True)
                if loss.requires_grad else (None, None))
            gradients[key] += np.concatenate([np.zeros(parameter.numel()) if value is None
                else value.detach().double().numpy().reshape(-1) for parameter, value in zip(params, values)])
        count += 1
    return gradients, {"actor_score_forward_batches": count, "actor_score_backward_batches": count * len(gradients),
        "max_abs_old_logp_difference": logp_error}


def residual_update(actor, score, *, cost):
    gradient = np.stack([score["gradients"]["A"], score["gradients"]["B"]], axis=0).mean(0)
    norm = float(np.linalg.norm(gradient))
    if not np.isfinite(norm) or norm == 0.:
        raise ValueError("residual branch gradient is undefined")
    tangent = -gradient / norm
    weight_size = actor.readout.weight.numel()
    dweight = torch.as_tensor(tangent[:weight_size], dtype=actor.readout.weight.dtype).reshape_as(actor.readout.weight)
    dbias = torch.as_tensor(tangent[weight_size:], dtype=actor.readout.bias.dtype).reshape_as(actor.readout.bias)
    states = torch.as_tensor(score["states"], dtype=torch.float32)
    dmean = states @ dweight.t() + dbias
    old_distribution = actor.distribution(states)
    full_std = old_distribution.stddev.detach().clamp(1e-4, 3.)
    if full_std.shape[-1] < dmean.shape[-1]:
        raise ValueError("residual branch distribution is narrower than its trainable readout")
    fisher_std = full_std[..., -dmean.shape[-1]:]
    fisher = float(torch.square(dmean / fisher_std).sum(-1).mean().item())
    if not np.isfinite(fisher) or fisher <= 0.:
        raise ValueError("residual branch Fisher geometry is undefined")
    step = float(np.sqrt(2. * spec.FISHER_RADIUS / fisher))
    base_before = copy.deepcopy(actor.base.state_dict())
    candidate = copy.deepcopy(actor)
    with torch.no_grad():
        candidate.readout.weight.add_(dweight, alpha=step)
        candidate.readout.bias.add_(dbias, alpha=step)
        new = candidate.distribution(states)
        kl = float(torch.distributions.kl_divergence(
            torch.distributions.Normal(old_distribution.mean, full_std),
            torch.distributions.Normal(new.mean, new.stddev)).sum(-1).mean().item())
    if not .5 * spec.FISHER_RADIUS <= kl <= 2. * spec.FISHER_RADIUS:
        raise ValueError("residual branch KL radius failed")
    actor.load_state_dict(candidate.state_dict())
    torch.testing.assert_close(actor.base.state_dict(), base_before, atol=0, rtol=0)
    cost["residual_fisher_batches"] += int(np.ceil(len(score["states"]) / spec.CHUNK_SIZE))
    cost["residual_kl_checks"] += 1
    cost["residual_parameter_updates"] += 1
    cost["training_freeze_checks"] += 1
    return {"geometry": {"gradient_norm": norm, "fisher": fisher, "step": step,
        "nominal_kl": spec.FISHER_RADIUS, "exact_kl": kl, "radius_check": "passed"},
        "score_costs": score["score_costs"], "signal_rms": score["signal_rms"]}


def paired_effects(period, evaluation, seeds):
    if set(evaluation) != set(spec.VARIANTS) or any([row["seed"] for row in rows] != seeds for rows in evaluation.values()):
        raise ValueError("branch evaluation scenario roster changed")
    for rows in evaluation.values():
        for row, base in zip(rows, evaluation["base"]):
            if (row["policy_seed"], row["lower_seed"]) != (base["policy_seed"], base["lower_seed"]):
                raise ValueError("branch evaluation noise changed")
    return {f"{period}/{a}_minus_{b}": float(np.mean([x["episode_return"] - y["episode_return"]
        for x, y in zip(evaluation[a], evaluation[b])])) for a, b in spec.CONTRAST_PAIRS}


def run(root, *, preflight, output):
    models, predictor, _, calibrations = source.load_source(root)
    stage111 = json.loads(spec.source_result(root).read_text())
    source.qualify(stage111, preflight=False)
    args, roles, options = spec.arguments(root, preflight=preflight), spec.seed_roles(root, preflight=preflight), spec.options(preflight=preflight)
    started = time.monotonic()
    cost, groups = dict.fromkeys(spec.budget(preflight=preflight), 0), {}
    cost.update(source_clone_loads=len(models), forecaster_loads=1, decoder_loads=len(models), source_cell_loads=1,
        donor_checkpoint_loads=len(models), branch_initializations=len(models)*len(spec.ARMS))
    with ProcessPoolExecutor(max_workers=options["workers"], mp_context=mp.get_context("spawn"), initializer=source.native.init_worker,
            initargs=(models["50"].config, args)) as pool:
        for period in spec.PERIODS:
            model = models[str(period)]
            source_snapshot = copy.deepcopy(model.state_dict())
            source_weights = source.native.joint.inference_weights(model)
            residuals = {arm: branch(model) for arm in spec.ARMS}
            initial = branch_weights(model)
            histories = {arm: [] for arm in spec.ARMS}
            for iteration, round_roles in enumerate(roles["training_rounds"], 1):
                for arm in spec.ARMS:
                    batches = {}
                    for name in ("A", "B"):
                        roster = round_roles["credit_" + name] + round_roles["lower_credit_" + name]
                        jobs = [(source_weights, copy.deepcopy(residuals[arm].state_dict()), row["scenario_seed"], row["noise_seeds"], arm,
                            period, predictor, calibrations[str(period)], args) for row in roster]
                        pairs = list(pool.map(training_pair, jobs))
                        batches[name] = []
                        for pair, registered in zip(pairs, roster):
                            rows = pair["rows"]
                            if [r["noise_seed"] for r in rows] != registered["noise_seeds"] or any(r["seed"] != registered["scenario_seed"] for r in rows):
                                raise ValueError("branch training seed roster changed")
                            for row in rows:
                                check_row(row, period=period, horizon=args.horizon)
                                for key, value in (("training_episodes", 1), ("native_episodes", 1), ("native_steps", args.horizon),
                                    ("native_lower_calls", args.horizon), ("native_upper_calls", row["upper_calls"]),
                                    ("native_network_checks", 1)):
                                    cost[key] += value
                                for key, value in (("planning_renewals", row["plan_renewals"]), ("planning_fits", row["plan_ols_fits"]),
                                    ("planning_predictions", row["plan_ridge_predictions"]), ("planning_reference_calls", row["reference_evaluations"]),
                                    ("planning_context_calls", row["actor_context_evaluations"])):
                                    cost[key] += value
                            cost["scenario_pair_checks"] += 1
                            batches[name].append(list(zip(pair["batches"], pair["rows"])))
                    score = score_branch(residuals[arm], batches, horizon=args.horizon, cost=cost)
                    history = residual_update(residuals[arm], score, cost=cost)
                    history["iteration"] = iteration
                    histories[arm].append(history)
            eval_weights = {"base": initial, **{arm: copy.deepcopy(residuals[arm].state_dict()) for arm in spec.ARMS}}
            evaluation = {}
            for variant in spec.VARIANTS:
                evaluation[variant] = []
            for seed in roles["native_evaluation"]:
                job = (source_weights, eval_weights, seed, period, predictor, calibrations[str(period)], args)
                group = pool.submit(evaluation_group, job).result()
                if group["pairing"] != "passed":
                    raise ValueError("branch evaluation pairing failed")
                for variant, row in group["evaluation"].items():
                    check_row(row, period=period, horizon=args.horizon, variant=variant)
                    evaluation[variant].append(row)
                    for key, value in (("evaluation_episodes", 1), ("native_episodes", 1), ("native_steps", args.horizon),
                        ("native_lower_calls", args.horizon), ("native_upper_calls", row["upper_calls"]),
                        ("native_network_checks", 1)):
                        cost[key] += value
                    for key, value in (("planning_renewals", row["plan_renewals"]), ("planning_fits", row["plan_ols_fits"]),
                        ("planning_predictions", row["plan_ridge_predictions"]), ("planning_reference_calls", row["reference_evaluations"]),
                        ("planning_context_calls", row["actor_context_evaluations"])):
                        cost[key] += value
                cost["native_pair_checks"] += 1
            trained = {}
            for arm in spec.ARMS:
                checkpoint = None
                if not preflight:
                    path = output.parent / "final_weights" / f"period_{period}_{arm}.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "period": period, "arm": arm,
                        "updates": options["updates"], "weights": residuals[arm].state_dict()}, path)
                    checkpoint = str(path)
                cost["checkpoint_writes"] += int(checkpoint is not None)
                trained[arm] = {"history": histories[arm], "checkpoint": checkpoint,
                    "final_freeze_check": "passed"}
            source.native.curves.support.assert_frozen(model, source_snapshot)
            cost["frozen_model_checks"] += 1
            groups[str(period)] = {"evaluation": evaluation, "effects": paired_effects(period, evaluation, roles["native_evaluation"]),
                "trained": trained, "source_and_Adam_unchanged": "passed", "task_options": spec.task_options(root, preflight=preflight)}
            print(f"{spec.EXPERIMENT_PROTOCOL} {root}/{period}: all3 branch arms complete", flush=True)
    cell = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "seed_roles": roles, "source_record": spec.source_record(root), "cost": cost, "groups": groups,
        "optimizer_steps": 0, "critic_fits": 0, "upper_updates": 0, "native_trace_writes": 0,
        "wall_seconds": time.monotonic() - started}
    qualify(cell, preflight=preflight)
    write_json(output, cell)
    write_json(output.parent / "completion" / "ready.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return cell


def qualify(cell, *, preflight):
    root = cell["root"]
    mismatches = []
    if cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract():
        mismatches.append("protocol_or_contract")
    if root not in spec.roots(preflight=preflight) or cell["preflight"] != preflight:
        mismatches.append("root_or_mode")
    if cell["source_record"] != spec.source_record(root):
        mismatches.append("source_record")
    if cell["seed_roles"] != spec.seed_roles(root, preflight=preflight):
        mismatches.append("seed_roles")
    if cell["cost"] != spec.budget(preflight=preflight):
        expected_cost = spec.budget(preflight=preflight)
        details = [f"{key}={cell['cost'][key]}!={expected_cost[key]}" for key in expected_cost
            if cell["cost"][key] != expected_cost[key]]
        mismatches.append("cost[" + ";".join(details) + "]")
    if set(cell["groups"]) != {str(p) for p in spec.PERIODS}:
        mismatches.append("period_groups")
    if any(cell[k] for k in ("optimizer_steps", "critic_fits", "upper_updates", "native_trace_writes")):
        mismatches.append("frozen_training_paths")
    if mismatches:
        raise ValueError("Stage112 qualification mismatch: " + ",".join(mismatches))
    h = spec.arguments(root, preflight=preflight).horizon
    for p, group in cell["groups"].items():
        if group["source_and_Adam_unchanged"] != "passed" or group["task_options"] != spec.task_options(root, preflight=preflight):
            raise ValueError("Stage112 source or task freeze failed")
        effects = paired_effects(p, group["evaluation"], cell["seed_roles"]["native_evaluation"])
        if group["effects"] != effects or not np.isfinite(list(effects.values())).all():
            raise ValueError("Stage112 paired evaluation changed")
        if set(group["trained"]) != set(spec.ARMS) or set(group["evaluation"]) != set(spec.VARIANTS):
            raise ValueError("Stage112 arm or evaluation roster changed")
        for variant, rows in group["evaluation"].items():
            for row in rows:
                check_row(row, period=int(p), horizon=h, variant=variant)
                if row["seed"] not in cell["seed_roles"]["native_evaluation"]:
                    raise ValueError("Stage112 evaluation seed changed")
    return cell


def aggregate(cells, *, preflight):
    result = statistics.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    passed = {} if preflight else {str(p): all(result["endpoints"][f"{p}/{a}_minus_{b}"]["ci"][0] > 0
        for a, b in spec.CONTRAST_PAIRS[:3]) for p in spec.PERIODS}
    result.update(branch_gain_gate="mechanical_only" if preflight else "supported_both_periods" if all(passed.values())
        else "partial" if any(passed.values()) else "not_supported", period_branch_gain_gate=passed,
        performance_claim="branch_only_learned_advice_gain_with_advice_blinded_ablation",
        upper_learning_prerequisite="only_after_branch_gain_gate_and_no Stage67 critic reopening")
    return result
