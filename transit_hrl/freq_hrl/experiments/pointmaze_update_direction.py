"""Compare a real PPO displacement with training and held-out task objectives."""

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig, concat_level_batches
from . import pointmaze_joint_renewal as joint
from . import pointmaze_critic_clock as clocks
from .pointmaze_critic_calibration import policy_drift
from .pointmaze_root_response import raw_directory, write_json
from scripts import pointmaze_update_direction_stage43_spec as spec


def surrogate_terms(model, actor, batch):
    advantage, _ = model._gae(batch.reward, batch.done, batch.duration, batch.old_value,
                              batch.next_value, batch.terminal)
    with torch.no_grad():
        state = torch.as_tensor(batch.state, dtype=torch.float32, device=model.device)
        action = torch.as_tensor(batch.action, dtype=torch.float32, device=model.device)
        logp, entropy = actor.log_prob_entropy(state, action)
        ratio = torch.exp((logp - torch.as_tensor(batch.old_logp, device=model.device)).clamp(-20., 20.))
        clipped = ratio.clamp(1. - model.config.clip_ratio, 1. + model.config.clip_ratio)
        normalized = torch.as_tensor(model._normalize(advantage), device=model.device)
        reward = torch.minimum(ratio * normalized, clipped * normalized).mean()
        return {"clipped_surrogate": float(reward), "unclipped_surrogate": float((ratio * normalized).mean()),
                "entropy": float(entropy.mean()), "actor_objective": float(reward + model.config.entropy_coef * entropy.mean()),
                "clip_fraction": float(((ratio - 1.).abs() > model.config.clip_ratio).float().mean())}


def score_targets(task_rewards):
    rewards = np.asarray(task_rewards, dtype=np.float64)
    if rewards.ndim != 2 or len(rewards) < 2:
        raise ValueError("task score needs at least two complete episodes")
    returns = np.cumsum(rewards[:, ::-1], axis=1)[:, ::-1]
    baseline = (returns.sum(axis=0, keepdims=True) - returns) / (len(returns) - 1)
    return returns - baseline


def task_direction(before, after, batch, task_rewards):
    centered = score_targets(task_rewards)
    if batch.size != centered.size:
        raise ValueError("task rewards and held-out actor transitions differ")
    state = torch.as_tensor(batch.state, dtype=torch.float32, device=before.device)
    action = torch.as_tensor(batch.action, dtype=torch.float32, device=before.device)
    logp, _ = before.lower_actor.log_prob_entropy(state, action)
    objective = (logp * torch.as_tensor(centered.reshape(-1), dtype=torch.float32, device=before.device)).sum() / len(centered)
    parameters = list(before.lower_actor.parameters())
    gradients = torch.autograd.grad(objective, parameters)
    displacement = [new.detach() - old.detach() for old, new in zip(parameters, after.lower_actor.parameters())]
    dot = sum((g * delta).sum() for g, delta in zip(gradients, displacement))
    grad_norm = torch.sqrt(sum(g.square().sum() for g in gradients))
    step_norm = torch.sqrt(sum(delta.square().sum() for delta in displacement))
    return {"heldout_task_direction": float(dot.detach()), "task_gradient_norm": float(grad_norm),
            "parameter_displacement_norm": float(step_norm)}


def load_pair(result, *, root, method, preflight):
    source = spec.source
    if (result["status"] != "complete" or result["protocol"] != source.EXPERIMENT_PROTOCOL
            or (result["root"], result["method"], result["preflight"]) != (root, method, preflight)
            or result["contract"] != source.contract() or result["seed_roles"] != source.seed_roles(root, preflight=preflight)):
        raise ValueError("first-update source differs from Stage-42")
    warm = source.options(preflight=preflight)["warmup_iterations"]
    models = []
    for iteration in (warm, warm + 1):
        payload = torch.load(result["snapshots"][str(iteration)]["checkpoint"], map_location="cpu", weights_only=False)
        if (payload["protocol"], payload["root"], payload["method"], payload["iteration"]) != (
                source.EXPERIMENT_PROTOCOL, root, method, iteration):
            raise ValueError("first-update checkpoint identity changed")
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(**payload["state_dict"]["config"]))
        model.load_state_dict(payload["state_dict"])
        if (model.config.state_encoder != "mlp" or model.lower_cost_value is not None
                or model.config.lower_actor_anchor_coef or model.config.lower_projection_consistency_coef):
            raise ValueError("first-update diagnostic requires the registered plain lower PPO")
        models.append(model)
    if models[0].config != models[1].config:
        raise ValueError("first-update configuration changed")
    return models


def worker_rollout(job):
    weights, seed, phase, mode, path, method = job
    model, args, _, _ = joint._WORKER
    model.load_state_dict(weights)
    kwargs = (spec.source.rollout_arguments(args.optimizer_seed, method, seed, phase="train", mode="learning")
              if phase == "reconstruction" else spec.rollout_arguments(args.optimizer_seed, seed, mode=mode, reference=method == "frozen"))
    policy_seed = seed + args.optimizer_seed if phase == "reconstruction" else spec.policy_seed(args.optimizer_seed, seed)
    torch.manual_seed(policy_seed)
    batch, row, raw = joint.rollout(model, args, "learned_history", seed=seed, capture=True,
        lower_credit=spec.source.LOWER_CREDIT[method], lower_value_context_builder=clocks.context_builder(method), **kwargs)
    row.update(policy_seed=policy_seed, deployment_mode=mode)
    clocks.audit_context(None if batch is None else batch.lower, row, raw["lower_value_context"], clock=spec.source.VALUE_CLOCK[method])
    np.savez_compressed(path, **raw)
    return None if batch is None else batch.lower, row, raw["reward"]


def check_first_batch(before, after, batch, rows, result):
    warm = result["options"]["warmup_iterations"]
    saved = result["training"][warm]
    advantage, target = before._gae(batch.reward, batch.done, batch.duration, batch.old_value)
    observed = {"primitive_steps": batch.size, "reward_sum": float(np.sum(batch.reward, dtype=np.float64)),
                "task_reward_sum": sum(r["lower_training_credit"]["task_reward_sum"] for r in rows),
                "done_count": int(batch.done.sum()), "option_count": sum(r["lower_training_credit"]["option_count"] for r in rows),
                "old_value_mean": float(batch.old_value.mean()), "gae_target_mean": float(target.mean()),
                "advantage_mean": float(advantage.mean()), "advantage_std": float(advantage.std())}
    for key, value in observed.items():
        np.testing.assert_allclose(value, saved[key], atol=1e-10, rtol=0, err_msg=f"first-update batch {key} differs")
    if {"seed": rows[0]["seed"], **rows[0]["lower_training_credit"]} != result["first_learning_credit"]:
        raise ValueError("first-learning credit replay changed")
    sampling = [{k: row[k] for k in saved["rollout_sampling"][0]} for row in rows]
    if sampling != saved["rollout_sampling"]:
        raise ValueError("first-update sampling changed")
    drift = policy_drift(after.lower_actor, before.lower_actor, batch.state)
    for key, value in drift.items():
        np.testing.assert_allclose(value, saved["policy_drift"][key], atol=1e-10, rtol=0)
    return {"status": "passed", **observed, "policy_drift": drift}


def diagnose(root, *, preflight, output):
    opt, roles = spec.options(preflight=preflight), spec.seed_roles(root, preflight=preflight)
    args = spec.source.source.arguments(root, preflight=preflight)
    sources = {m: json.loads(spec.source_result(root, m, preflight=preflight).read_text()) for m in spec.METHODS}
    models = {m: load_pair(sources[m], root=root, method=m, preflight=preflight) for m in spec.METHODS}
    reference = models[spec.METHODS[0]][0]
    frozen = joint.inference_weights(reference)
    for before, after in models.values():
        for name, weights in frozen.items():
            if name != "lower_value":
                torch.testing.assert_close(getattr(before, name).state_dict(), weights, rtol=0, atol=0)
            if name not in ("lower_actor", "lower_value"):
                torch.testing.assert_close(getattr(after, name).state_dict(), weights, rtol=0, atol=0)
    raw, started = raw_directory(output), time.monotonic()
    counts = {phase: {k: 0 for k in ("primitive_steps", "upper_inference_calls", "lower_inference_calls", "gate_inference_calls")}
              for phase in ("reconstruction", "evaluation")}
    methods, evaluation = {}, {}
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"), initializer=joint.init_worker,
                             initargs=(reference.config, args, "learned_history", "intrinsic_option")) as pool:
        def episodes(model, method, *, phase, mode, seeds):
            directory = raw / phase / method / mode
            directory.mkdir(parents=True, exist_ok=True)
            weights = joint.inference_weights(model)
            pairs = list(pool.map(worker_rollout, [(weights, seed, phase, mode, str(directory / f"episode_{seed}.npz"), method) for seed in seeds]))
            rows = [row for _, row, _ in pairs]
            joint.audit_trajectories(rows, args=args, method="learned_history", raw_path=directory)
            for row in rows:
                counts[phase]["primitive_steps"] += row["episode_length"]
                for key in ("upper_inference_calls", "lower_inference_calls", "gate_inference_calls"):
                    counts[phase][key] += row[key]
            return pairs

        for method, (before, after) in models.items():
            pairs = episodes(before, method, phase="reconstruction", mode="learning", seeds=roles["reconstruction"])
            batch = concat_level_batches([b for b, _, _ in pairs])
            checked = check_first_batch(before, after, batch, [row for _, row, _ in pairs], sources[method])
            initial, final = surrogate_terms(before, before.lower_actor, batch), surrogate_terms(before, after.lower_actor, batch)
            methods[method] = {"reconstruction": checked, "surrogate_before": initial, "surrogate_after": final,
                "training_surrogate_gain": final["clipped_surrogate"] - initial["clipped_surrogate"],
                "actor_objective_gain": final["actor_objective"] - initial["actor_objective"],
                "source_checkpoints": [sources[method]["snapshots"][str(i)]["checkpoint"] for i in
                                       (opt["warmup_iterations"], opt["warmup_iterations"] + 1)]}
            print(f"reconstructed {root}/{method}: original first-update batch and policy drift", flush=True)
        for policy in spec.POLICIES:
            evaluation[policy] = {}
            model = reference if policy == "frozen" else models[policy][1]
            for mode in spec.MODES:
                pairs = episodes(model, policy, phase="evaluation", mode=mode, seeds=roles["evaluation"])
                evaluation[policy][mode] = [row for _, row, _ in pairs]
                if policy == "frozen" and mode == "lower_sampled":
                    heldout = concat_level_batches([b for b, _, _ in pairs])
                    rewards = np.stack([reward for _, _, reward in pairs])
                    for method, (before, after) in models.items():
                        methods[method].update(task_direction(before, after, heldout, rewards))
                    zero = task_direction(reference, reference, heldout, rewards)
                    np.testing.assert_equal(zero["heldout_task_direction"], 0.)
            print(f"evaluated {root}/{policy}: both fresh native deployment modes", flush=True)
    budget = spec.budget(preflight=preflight)
    for phase in counts:
        if counts[phase]["primitive_steps"] != budget[phase + "_primitive_steps"]:
            raise ValueError("first-update native accounting changed")
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
              "root": root, "preflight": preflight, "seed_roles": roles, "budget": budget, "inference_counts": counts,
              "optimizer_steps": 0, "methods": methods, "evaluation_rows": evaluation,
              "zero_displacement_task_direction": zero["heldout_task_direction"],
              "native_trace_audits": len(spec.METHODS) * len(roles["reconstruction"]) + len(spec.POLICIES) * len(spec.MODES) * len(roles["evaluation"]),
              "wall_seconds": time.monotonic() - started}
    write_json(output, result)
    return result


def aggregate(results, *, preflight):
    roots = spec.roots(preflight=preflight)
    cells = {r["root"]: r for r in results}
    if len(cells) != len(results) or set(cells) != set(roots):
        raise ValueError("first-update root roster incomplete")
    root_rows = []
    for root in roots:
        cell = cells[root]
        if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL
                or cell["preflight"] != preflight or cell["contract"] != spec.contract()
                or cell["seed_roles"] != spec.seed_roles(root, preflight=preflight)
                or cell["budget"] != spec.budget(preflight=preflight) or cell["optimizer_steps"] != 0):
            raise ValueError("first-update result violates the frozen protocol")
        if set(cell["methods"]) != set(spec.METHODS) or set(cell["evaluation_rows"]) != set(spec.POLICIES):
            raise ValueError("first-update policy roster incomplete")
        for policy in spec.POLICIES:
            for mode in spec.MODES:
                rows = cell["evaluation_rows"][policy][mode]
                if [r["seed"] for r in rows] != cell["seed_roles"]["evaluation"]:
                    raise ValueError("fresh paired evaluation roster changed")
                for row in rows:
                    kwargs = spec.rollout_arguments(root, row["seed"], mode=mode, reference=policy == "frozen")
                    if (row["policy_seed"] != spec.policy_seed(root, row["seed"])
                            or row["deployment_mode"] != mode or any(row[k] != v for k, v in kwargs.items() if k != "sample")):
                        raise ValueError("fresh evaluation sampling changed")
        means = {mode: {p: {k: float(np.mean([r[k] for r in cell["evaluation_rows"][p][mode]])) for k in spec.METRICS}
                        for p in spec.POLICIES} for mode in spec.MODES}
        endpoints = {}
        for method in spec.METHODS:
            for key in ("training_surrogate_gain", "heldout_task_direction"):
                endpoints[f"{method}:{key}"] = cell["methods"][method][key]
            for mode in spec.MODES:
                endpoints[f"{method}:{mode}_return"] = means[mode][method]["episode_return"] - means[mode]["frozen"]["episode_return"]
        root_rows.append({"root": root, "methods": cell["methods"], "means": means, "endpoints": endpoints,
                          "inference_counts": cell["inference_counts"], "native_trace_audits": cell["native_trace_audits"]})
    summary = {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
               "contract": spec.contract(), "root_rows": root_rows, "optimizer_steps": 0,
               "native_trace_audits": sum(r["native_trace_audits"] for r in results),
               "method_cost": {k: sum(p[k] for r in results for p in r["inference_counts"].values())
                               for k in ("primitive_steps", "upper_inference_calls", "lower_inference_calls", "gate_inference_calls")},
               "verification_primitive_steps": 0}
    if not preflight:
        x = np.asarray([[r["endpoints"][key] for key in spec.ENDPOINTS] for r in root_rows])
        rng = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED))
        indices = rng.integers(0, len(x), (spec.BOOTSTRAP_DRAWS, len(x)))
        tail = .05 / (2 * spec.CI_FAMILY_SIZE)
        bounds = np.quantile(x[indices].mean(axis=1), [tail, 1 - tail], axis=0)
        summary["primary_endpoints"] = {k: {"mean": float(x[:, i].mean()), "ci": bounds[:, i].tolist(),
            "effect": "positive" if bounds[0, i] > 0 else "negative" if bounds[1, i] < 0 else "inconclusive"} for i, k in enumerate(spec.ENDPOINTS)}
        counts = np.stack([(indices == i).sum(axis=1) for i in range(len(x))], axis=1)
        np.testing.assert_allclose(np.quantile(counts @ x / len(x), [tail, 1 - tail], axis=0), bounds, atol=1e-10, rtol=0)
        summary["independent_statistics"] = {"status": "passed", "endpoints": len(spec.ENDPOINTS)}
    return summary
