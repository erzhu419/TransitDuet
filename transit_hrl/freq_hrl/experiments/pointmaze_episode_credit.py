"""Paired native task-credit training with critic targets held unchanged."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import math
import multiprocessing as mp
import time

import numpy as np
import torch
from freq_hrl.rl.smdp_actor_critic import concat_level_batches
from . import pointmaze_joint_renewal as joint
from . import pointmaze_critic_clock as clocks
from .pointmaze_actor_acceptance import first_batch_pair
from .pointmaze_critic_calibration import policy_drift
from .pointmaze_update_direction import load_pair, score_targets
from .pointmaze_root_response import raw_directory, write_json
from scripts import pointmaze_episode_credit_stage46_spec as spec


def update(model, batch, task_rewards, treatment, *, specification=spec, actor_advantage=None):
    spec = specification
    if treatment not in spec.TREATMENTS:
        raise ValueError("unregistered actor credit treatment")
    rewards = np.asarray(task_rewards, dtype=np.float64)
    if rewards.ndim != 2 or len(rewards) < 2 or rewards.size != batch.size:
        raise ValueError("actor credit requires at least two complete native episodes")
    np.testing.assert_array_equal(batch.reward, rewards.astype(np.float32).reshape(-1))
    advantage, returns = model._gae(batch.reward, batch.done, batch.duration, batch.old_value,
                                    batch.next_value, batch.terminal)
    episode_advantage = score_targets(rewards).reshape(-1).astype(np.float32)
    if actor_advantage is None:
        actor_advantage = advantage if treatment == "gae" else episode_advantage
    normalized, episode_normalized = model._normalize(advantage), model._normalize(episode_advantage)
    metrics = model._update_level(level="lower", batch=batch, actor=model.lower_actor, value_net=model.lower_value,
        actor_optimizer=model.lower_actor_optimizer, value_optimizer=model.lower_value_optimizer,
        actor_advantage=None if treatment == "gae" else actor_advantage)
    return {"actor_optimizer_steps": int(metrics["lower_actor_optimizer_steps"]),
            "value_optimizer_steps": int(metrics["lower_value_optimizer_steps"]),
            "actor_advantage_mean": float(actor_advantage.mean()), "actor_advantage_std": float(actor_advantage.std()),
            "critic_target_mean": float(returns.mean()), "critic_target_std": float(returns.std()),
            "gae_mc_normalized_mse": float(np.mean((normalized - episode_normalized) ** 2)),
            "gae_mc_normalized_dot": float(np.mean(normalized * episode_normalized)),
            "task_return_mean": float(rewards.sum(axis=1).mean()), "native_episodes": len(rewards),
            "critic_credit": "original_task_option_gae", "actor_credit": treatment}


def worker_rollout(job, *, specification=spec):
    spec = specification
    weights, seed, method, phase, mode, path = job
    model, args, _, _ = joint._WORKER
    model.load_state_dict(weights)
    torch.manual_seed(spec.policy_seed(args.optimizer_seed, seed))
    batch, row, raw = joint.rollout(model, args, "learned_history", seed=seed, capture=True,
        lower_credit="task_option", lower_value_context_builder=clocks.context_builder(method),
        **spec.rollout_arguments(args.optimizer_seed, seed, phase=phase, mode=mode))
    row.update(policy_seed=spec.policy_seed(args.optimizer_seed, seed), deployment_mode=mode)
    clocks.audit_context(None if batch is None else batch.lower, row, raw["lower_value_context"],
                         clock=method.endswith("_clock"))
    if batch is not None:
        np.testing.assert_array_equal(batch.lower.reward, raw["reward"].astype(np.float32))
    np.savez_compressed(path, **raw)
    return None if batch is None else batch.lower, row, raw["reward"] if phase == "train" else None


def train(root, *, preflight, output, specification=spec, rollout_worker=worker_rollout, update_fn=None, source_index=0):
    spec = specification
    args, opt = spec.arguments(root, preflight=preflight), spec.options(preflight=preflight)
    roles = spec.seed_roles(root, preflight=preflight)
    sources = {m: json.loads(spec.source_result(root, m, preflight=preflight).read_text()) for m in spec.METHODS}
    before = {m: load_pair(sources[m], root=root, method=m, preflight=preflight)[source_index] for m in spec.METHODS}
    reference = before[spec.METHODS[0]]
    frozen = joint.inference_weights(reference)
    for model in before.values():
        for name, weights in frozen.items():
            if name != "lower_value":
                torch.testing.assert_close(getattr(model, name).state_dict(), weights, atol=0, rtol=0)
    raw, started = raw_directory(output), time.monotonic()
    keys = ("primitive_steps", "upper_inference_calls", "lower_inference_calls", "gate_inference_calls")
    counts = {phase: dict.fromkeys(keys, 0) for phase in ("train", "eval")}
    evaluation, training, pairs, checkpoints, expected_steps, pair_details = {}, {}, {}, {}, {}, {}
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"), initializer=joint.init_worker,
                             initargs=(reference.config, args, "learned_history", "task_option")) as pool:
        def episodes(model, method, policy, phase, mode, seeds, iteration):
            directory = raw / policy.replace(":", "/") / str(iteration) / mode
            directory.mkdir(parents=True, exist_ok=True)
            weights = joint.inference_weights(model)
            outputs = list(pool.map(rollout_worker, [(weights, seed, method, phase, mode,
                           str(directory / f"episode_{seed}.npz")) for seed in seeds]))
            rows = [row for _, row, _ in outputs]
            joint.audit_trajectories(rows, args=args, method="learned_history", raw_path=directory)
            for row in rows:
                counts[phase]["primitive_steps"] += row["episode_length"]
                for key in keys[1:]:
                    counts[phase][key] += row[key]
            return outputs

        def snapshot(model, method, policy, iteration):
            for name, weights in frozen.items():
                if name not in ("lower_actor", "lower_value"):
                    torch.testing.assert_close(getattr(model, name).state_dict(), weights, atol=0, rtol=0)
            directory = raw / policy.replace(":", "/")
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"iteration_{iteration}.pt"
            torch.save({"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "policy": policy,
                        "iteration": iteration, "state_dict": model.state_dict()}, path)
            checkpoints.setdefault(policy, {})[str(iteration)] = str(path)
            evaluation.setdefault(policy, {})[str(iteration)] = {mode: [row for _, row, _ in episodes(
                model, method, policy, "eval", mode, roles["evaluation"], iteration)] for mode in spec.MODES}

        snapshot(reference, "frozen", "frozen", 0)
        for method in spec.METHODS:
            training[method], pair_details[method], first, first_critic = {}, {}, None, None
            config = before[method].config
            size = opt["rollouts_per_iteration"] * args.horizon
            expected_steps[method] = max(1, int(config.epochs)) * math.ceil(size / max(1, min(int(config.minibatch_size), size)))
            for treatment in spec.TREATMENTS:
                model, policy = copy.deepcopy(before[method]), f"{method}:{treatment}"
                history = training[method][treatment] = []
                for iteration in range(1, opt["learning_iterations"] + 1):
                    start = (iteration - 1) * opt["rollouts_per_iteration"]
                    rollouts = episodes(model, method, policy, "train", "training",
                                        roles["training"][start:start + opt["rollouts_per_iteration"]], iteration)
                    batch = concat_level_batches([b for b, _, _ in rollouts])
                    rewards = np.stack([r for _, _, r in rollouts])
                    if iteration == 1:
                        if treatment == "gae":
                            first = (batch, rewards.copy())
                        else:
                            pairs[method] = first_batch_pair(first[0], batch)
                            np.testing.assert_array_equal(first[1], rewards)
                            pairs[method]["native_task_rewards"] = "passed"
                            pair_details[method][treatment] = pairs[method]
                    old_actor = copy.deepcopy(model.lower_actor)
                    np.random.seed(spec.shuffle_seed(root, iteration))
                    metrics = (update(model, batch, rewards, treatment) if update_fn is None else
                               update_fn(model, batch, rewards, treatment, root=root, iteration=iteration))
                    if iteration == 1:
                        critic = (model.lower_value.state_dict(), model.lower_value_optimizer.state_dict())
                        if treatment == "gae":
                            first_critic = copy.deepcopy(critic)
                        else:
                            torch.testing.assert_close(critic, first_critic, atol=0, rtol=0)
                            pairs[method]["first_critic_update"] = "passed"
                    history.append({"iteration": iteration, "primitive_steps": batch.size, **metrics,
                        "policy_drift": policy_drift(model.lower_actor, old_actor, batch.state),
                        "rollout_sampling": [{k: row[k] for k in ("seed", "policy_seed", "upper_sample", "gate_sample", "lower_sample", "lower_seed", "gate_seed")}
                                             for _, row, _ in rollouts],
                        "inference_counts": {k: sum(row[k] for _, row, _ in rollouts) for k in keys[1:]}})
                    if iteration in spec.snapshots(preflight=preflight):
                        snapshot(model, method, policy, iteration)
                        print(f"trained {root}/{policy} iteration {iteration}: actor credit {treatment}", flush=True)
    budget = spec.budget(preflight=preflight)
    if counts["train"]["primitive_steps"] != budget["training_primitive_steps"] or counts["eval"]["primitive_steps"] != budget["evaluation_primitive_steps"]:
        raise ValueError("episode credit native accounting changed")
    warm = spec.warmup_iterations(preflight=preflight) + source_index
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "options": opt, "seed_roles": roles, "budget": budget, "inference_counts": counts,
        "training": training, "evaluation_rows": evaluation, "first_batch_pairs": pairs,
        "first_batch_pair_details": pair_details,
        "source_checkpoints": {m: sources[m]["snapshots"][str(warm)]["checkpoint"] for m in spec.METHODS},
        "source_checkpoint_iteration": warm,
        "checkpoints": checkpoints, "expected_steps_per_update": expected_steps, "fixed_upper_gate_networks": "passed",
        "native_trace_audits": budget["native_trace_audits"], "wall_seconds": time.monotonic() - started}
    write_json(output, result)
    return result


def check_sampling(rows, root, seeds, *, phase, mode, specification=spec):
    spec = specification
    if [r["seed"] for r in rows] != seeds:
        raise ValueError("episode credit paired seed roster changed")
    for row in rows:
        kwargs = spec.rollout_arguments(root, row["seed"], phase=phase, mode=mode)
        if (row["policy_seed"] != spec.policy_seed(root, row["seed"])
                or (phase == "eval" and row["deployment_mode"] != mode)
                or any(row[k] != v for k, v in kwargs.items() if k != "sample")):
            raise ValueError("episode credit paired sampling changed")


def aggregate(results, *, preflight, specification=spec, step_counts=None):
    spec = specification
    roots = spec.roots(preflight=preflight)
    cells = {r["root"]: r for r in results}
    if len(cells) != len(results) or set(cells) != set(roots):
        raise ValueError("episode credit root roster incomplete")
    opt, budget, root_rows = spec.options(preflight=preflight), spec.budget(preflight=preflight), []
    totals = {"actor_optimizer_steps": 0, "value_optimizer_steps": 0}
    for root in roots:
        cell, horizon = cells[root], spec.arguments(root, preflight=preflight).horizon
        if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
                or cell["preflight"] != preflight or cell["options"] != opt or cell["budget"] != budget
                or cell["seed_roles"] != spec.seed_roles(root, preflight=preflight)
                or cell["fixed_upper_gate_networks"] != "passed" or cell["native_trace_audits"] != budget["native_trace_audits"]):
            raise ValueError("episode credit result violates the frozen protocol")
        if set(cell["training"]) != set(spec.METHODS) or set(cell["first_batch_pairs"]) != set(spec.METHODS) or set(cell["evaluation_rows"]) != set(spec.POLICIES):
            raise ValueError("episode credit policy roster incomplete")
        diagnostics, train_counts = {}, dict.fromkeys(cell["inference_counts"]["train"], 0)
        for method in spec.METHODS:
            pair = cell["first_batch_pairs"][method]
            if (pair["status"] != "passed" or pair["native_task_rewards"] != "passed"
                    or pair["first_critic_update"] != "passed" or pair["transitions"] != opt["rollouts_per_iteration"] * horizon):
                raise ValueError("episode credit first batch pair changed")
            if set(cell["training"][method]) != set(spec.TREATMENTS):
                raise ValueError("episode credit treatment roster incomplete")
            diagnostics[method] = {}
            histories = cell["training"][method]
            for treatment in spec.TREATMENTS[1:]:
                for key in ("critic_target_mean", "critic_target_std", "task_return_mean", "gae_mc_normalized_mse", "gae_mc_normalized_dot"):
                    if histories["gae"][0][key] != histories[treatment][0][key]:
                        raise ValueError("episode credit paired initial targets differ")
            for treatment in spec.TREATMENTS:
                history = histories[treatment]
                if [r["iteration"] for r in history] != list(range(1, opt["learning_iterations"] + 1)):
                    raise ValueError("episode credit training roster incomplete")
                for row in history:
                    steps = cell["expected_steps_per_update"][method]
                    expected = ({"actor_optimizer_steps": steps, "value_optimizer_steps": steps}
                                if step_counts is None else step_counts(row, steps))
                    if (row["actor_credit"] != treatment or row["critic_credit"] != "original_task_option_gae"
                            or any(row[k] != expected[k] for k in totals)
                            or row["native_episodes"] != opt["rollouts_per_iteration"]):
                        raise ValueError("episode credit optimizer or target accounting changed")
                    begin = (row["iteration"] - 1) * opt["rollouts_per_iteration"]
                    seeds = cell["seed_roles"]["training"][begin:begin + opt["rollouts_per_iteration"]]
                    check_sampling(row["rollout_sampling"], root, seeds, phase="train", mode="training", specification=spec)
                    if row["primitive_steps"] != len(seeds) * horizon or row["inference_counts"]["lower_inference_calls"] != row["primitive_steps"]:
                        raise ValueError("episode credit training step accounting changed")
                    train_counts["primitive_steps"] += row["primitive_steps"]
                    for key in row["inference_counts"]:
                        train_counts[key] += row["inference_counts"][key]
                    for key in totals:
                        totals[key] += row[key]
                diagnostics[method][treatment] = {"first": {k: history[0][k] for k in (
                    "actor_advantage_mean", "actor_advantage_std", "critic_target_mean", "critic_target_std",
                    "gae_mc_normalized_mse", "gae_mc_normalized_dot", "task_return_mean")},
                    **{k: sum(r[k] for r in history) for k in totals}}
        means = {"0": {m: {} for m in spec.MODES}, **{str(i): {m: {} for m in spec.MODES} for i in spec.snapshots(preflight=preflight)}}
        eval_counts = dict.fromkeys(cell["inference_counts"]["eval"], 0)
        for policy in spec.POLICIES:
            snapshots = (0,) if policy == "frozen" else spec.snapshots(preflight=preflight)
            if set(cell["evaluation_rows"][policy]) != {str(i) for i in snapshots}:
                raise ValueError("episode credit snapshot roster incomplete")
            for iteration in snapshots:
                stages = cell["evaluation_rows"][policy][str(iteration)]
                if set(stages) != set(spec.MODES):
                    raise ValueError("episode credit deployment roster incomplete")
                for mode, rows in stages.items():
                    check_sampling(rows, root, cell["seed_roles"]["evaluation"], phase="eval", mode=mode, specification=spec)
                    for row in rows:
                        if row["episode_length"] != horizon:
                            raise ValueError("episode credit evaluation horizon changed")
                        eval_counts["primitive_steps"] += row["episode_length"]
                        for key in eval_counts:
                            if key != "primitive_steps":
                                eval_counts[key] += row[key]
                    means[str(iteration)][mode][policy] = {k: float(np.mean([r[k] for r in rows])) for k in spec.METRICS}
        if (train_counts != cell["inference_counts"]["train"] or eval_counts != cell["inference_counts"]["eval"]
                or train_counts["primitive_steps"] != budget["training_primitive_steps"] or eval_counts["primitive_steps"] != budget["evaluation_primitive_steps"]):
            raise ValueError("episode credit inference accounting changed")
        root_rows.append({"root": root, "means": means, "diagnostics": diagnostics,
                          "endpoints": spec.contrasts(means, preflight=preflight, diagnostics=cell)})
    summary = {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "root_rows": root_rows, "optimizer_steps": totals,
        "native_trace_audits": sum(r["native_trace_audits"] for r in results), "verification_primitive_steps": 0,
        "method_cost": {k: sum(r["inference_counts"][p][k] for r in results for p in ("train", "eval"))
                        for k in ("primitive_steps", "upper_inference_calls", "lower_inference_calls", "gate_inference_calls")}}
    if not preflight:
        x = np.asarray([[row["endpoints"][k] for k in spec.ENDPOINTS] for row in root_rows])
        indices = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED)).integers(0, len(x), (spec.BOOTSTRAP_DRAWS, len(x)))
        tail = .05 / (2 * spec.CI_FAMILY_SIZE)
        bounds = np.quantile(x[indices].mean(axis=1), [tail, 1 - tail], axis=0)
        summary["primary_endpoints"] = {k: {"mean": float(x[:, i].mean()), "ci": bounds[:, i].tolist(),
            "effect": "positive" if bounds[0, i] > 0 else "negative" if bounds[1, i] < 0 else "inconclusive"} for i, k in enumerate(spec.ENDPOINTS)}
        counts = np.stack([(indices == i).sum(axis=1) for i in range(len(x))], axis=1)
        np.testing.assert_allclose(np.quantile(counts @ x / len(x), [tail, 1 - tail], axis=0), bounds, atol=1e-10, rtol=0)
        summary["independent_statistics"] = {"status": "passed", "endpoints": len(spec.ENDPOINTS)}
    return summary
