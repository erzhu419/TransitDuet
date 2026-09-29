"""Fit full-task state baselines without using the queried episode's labels."""

from functools import partial
import math

import numpy as np
import torch
from freq_hrl.rl.dual_actor_critic import ValueNet
from . import pointmaze_episode_credit as episodes
from .pointmaze_root_response import raw_directory
from scripts import pointmaze_state_baseline_stage47_spec as spec


def baseline_settings(config, state_dim):
    return {"state_dim": state_dim, "hidden_dim": config.hidden_dim, "epochs": max(1, int(config.epochs)),
        "minibatch_size": int(config.minibatch_size), "learning_rate": config.lower_learning_rate,
        "value_coef": config.value_coef, "max_grad_norm": config.max_grad_norm}


def state_credit(states, task_rewards, config, *, root, iteration):
    rewards, states = np.asarray(task_rewards, dtype=np.float64), np.asarray(states, dtype=np.float32)
    if rewards.ndim != 2 or len(rewards) < 3 or states.ndim != 3 or states.shape[:2] != rewards.shape:
        raise ValueError("state credit needs at least three complete episodes with pre-action states")
    returns = np.cumsum(rewards[:, ::-1], axis=1)[:, ::-1].copy()
    settings, predictions, folds = baseline_settings(config, states.shape[-1]), np.empty_like(returns), []
    # Only the other episodes define normalization and fit labels; query labels are diagnostics after fitting.
    for held_out in range(len(rewards)):
        fit_indices = [i for i in range(len(rewards)) if i != held_out]
        time_mean = returns[fit_indices].mean(axis=0)
        residual = returns[fit_indices] - time_mean
        target_scale = float(residual.std()) + 1e-8
        fit_states = states[fit_indices].reshape(-1, states.shape[-1])
        mean, scale = fit_states.mean(axis=0), fit_states.std(axis=0) + 1e-8
        x = torch.from_numpy((fit_states - mean) / scale)
        y = torch.from_numpy((residual.reshape(-1) / target_scale).astype(np.float32))
        seed, steps = spec.baseline_seed(root, iteration, held_out), 0
        rng = np.random.default_rng(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            value = ValueNet(states.shape[-1], config.hidden_dim)
            with torch.no_grad():
                value.net[-1].weight.zero_()
                value.net[-1].bias.zero_()
            optimizer = torch.optim.Adam(value.parameters(), lr=config.lower_learning_rate)
            size = max(1, min(settings["minibatch_size"], len(x)))
            for _ in range(settings["epochs"]):
                order = rng.permutation(len(x))
                for start in range(0, len(x), size):
                    rows = order[start:start + size]
                    loss = config.value_coef * (value(x[rows]) - y[rows]).square().mean()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(value.parameters(), config.max_grad_norm)
                    optimizer.step()
                    steps += 1
            with torch.no_grad():
                predicted = value(torch.from_numpy((states[held_out] - mean) / scale)).numpy()
                fit_mse = float((value(x) - y).square().mean())
        predictions[held_out] = time_mean + target_scale * predicted
        folds.append({"held_out": held_out, "fit_indices": fit_indices, "seed": seed,
            "optimizer_steps": steps, "fit_normalized_mse": fit_mse,
            "held_out_mse": float(np.mean((returns[held_out] - predictions[held_out]) ** 2)),
            "time_loo_mse": float(np.mean((returns[held_out] - time_mean) ** 2))})
    return (returns - predictions).reshape(-1).astype(np.float32), predictions, {"settings": settings, "folds": folds,
        "baseline_optimizer_steps": sum(f["optimizer_steps"] for f in folds)}


def gradient_dispersion(model, batch, task_rewards, state_advantage):
    count, horizon = np.asarray(task_rewards).shape
    logp, _ = model.lower_actor.log_prob_entropy(torch.as_tensor(batch.state, device=model.device),
                                                torch.as_tensor(batch.action, device=model.device))
    parameters, diagnostics = list(model.lower_actor.parameters()), {}
    for treatment, advantage in (("episode_mc", episodes.score_targets(task_rewards).reshape(-1)),
                                 ("state_mc", state_advantage)):
        weights = torch.as_tensor(model._normalize(advantage), device=model.device).reshape(count, horizon)
        scores = (logp.reshape(count, horizon) * weights).mean(dim=1)
        gradients = []
        for i in range(count):
            values = torch.autograd.grad(scores[i], parameters, retain_graph=not (treatment == "state_mc" and i == count - 1))
            gradients.append(torch.cat([g.detach().reshape(-1) for g in values]).double())
        matrix = torch.stack(gradients)
        mean = matrix.mean(dim=0)
        diagnostics[treatment] = {"trace_dispersion": float((matrix - mean).square().sum() / (count - 1)),
                                  "mean_gradient_norm": float(mean.norm())}
    return {**diagnostics, "gradient_backward_calls": 2 * count}


def update(model, batch, task_rewards, treatment, *, root, iteration, artifact_directory=None):
    extra, advantage = {}, None
    if treatment == "state_mc":
        rewards = np.asarray(task_rewards)
        advantage, baseline, extra = state_credit(batch.value_state.reshape(*rewards.shape, -1), rewards,
                                                   model.config, root=root, iteration=iteration)
        extra["gradient_dispersion"] = gradient_dispersion(model, batch, rewards, advantage)
        if artifact_directory is not None:
            artifact_directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(artifact_directory / f"iteration_{iteration}.npz", baseline=baseline,
                                actor_advantage=advantage)
    return {**episodes.update(model, batch, task_rewards, treatment, specification=spec, actor_advantage=advantage), **extra}


def worker_rollout(job):
    return episodes.worker_rollout(job, specification=spec)


def train(root, *, preflight, output):
    return episodes.train(root, preflight=preflight, output=output, specification=spec,
        rollout_worker=worker_rollout, update_fn=partial(update, artifact_directory=raw_directory(output) / "baseline"))


def aggregate(results, *, preflight):
    summary = episodes.aggregate(results, preflight=preflight, specification=spec)
    auxiliary = {"baseline_optimizer_steps": 0, "gradient_backward_calls": 0}
    method = spec.METHODS[0]
    for cell, root_row in zip(sorted(results, key=lambda c: c["root"]), summary["root_rows"]):
        opt, horizon = cell["options"], spec.arguments(cell["root"], preflight=preflight).horizon
        pairs = cell["first_batch_pair_details"][method]
        if set(pairs) != set(spec.TREATMENTS[1:]) or any(
                p["status"] != "passed" or p["native_task_rewards"] != "passed" or p["first_critic_update"] != "passed"
                or p["transitions"] != opt["rollouts_per_iteration"] * horizon for p in pairs.values()):
            raise ValueError("state baseline first-batch pairing incomplete")
        history = cell["training"][method]["state_mc"]
        settings = history[0]["settings"]
        count = opt["rollouts_per_iteration"]
        fit_size = (count - 1) * horizon
        fit_steps = settings["epochs"] * math.ceil(fit_size / max(1, min(settings["minibatch_size"], fit_size)))
        for row in history:
            if row["settings"] != settings or len(row["folds"]) != count:
                raise ValueError("state baseline fit configuration changed")
            for index, fold in enumerate(row["folds"]):
                if (fold["held_out"] != index or fold["fit_indices"] != [i for i in range(count) if i != index]
                        or fold["seed"] != spec.baseline_seed(cell["root"], row["iteration"], index)
                        or fold["optimizer_steps"] != fit_steps):
                    raise ValueError("state baseline episode-held-out fit provenance changed")
            if row["baseline_optimizer_steps"] != count * fit_steps or row["gradient_dispersion"]["gradient_backward_calls"] != 2 * count:
                raise ValueError("state baseline auxiliary cost accounting changed")
            auxiliary["baseline_optimizer_steps"] += row["baseline_optimizer_steps"]
            auxiliary["gradient_backward_calls"] += row["gradient_dispersion"]["gradient_backward_calls"]
        root_row["state_baseline"] = {"settings": settings, "first_gradient_dispersion": history[0]["gradient_dispersion"],
            "first_held_out_mse": float(np.mean([f["held_out_mse"] for f in history[0]["folds"]])),
            "first_time_loo_mse": float(np.mean([f["time_loo_mse"] for f in history[0]["folds"]])),
            "baseline_optimizer_steps": sum(r["baseline_optimizer_steps"] for r in history),
            "gradient_backward_calls": sum(r["gradient_dispersion"]["gradient_backward_calls"] for r in history)}
    summary["auxiliary_cost"] = auxiliary
    return summary
