"""Replay PPO trials transactionally under an empirical episode KL budget."""

import copy

import numpy as np
import torch
from . import pointmaze_episode_credit as episodes
from scripts import pointmaze_episode_kl_stage48_spec as spec

LOWER_STATE = ("lower_actor", "lower_value", "lower_actor_optimizer", "lower_value_optimizer")


def policy_terms(model, batch, reference, advantage, episode_count):
    with torch.no_grad():
        state = torch.as_tensor(batch.state, dtype=torch.float32, device=model.device)
        old, new = reference.distribution(state), model.lower_actor.distribution(state)
        old64 = torch.distributions.Normal(old.mean.double(), old.stddev.double())
        new64 = torch.distributions.Normal(new.mean.double(), new.stddev.double())
        conditional = torch.distributions.kl_divergence(old64, new64).sum(dim=-1).clamp_min(0.)
        kl = conditional.reshape(episode_count, -1).sum(dim=1).cpu().numpy()
        action = torch.as_tensor(batch.action, dtype=torch.float32, device=model.device)
        logp, entropy = model.lower_actor.log_prob_entropy(state, action)
        ratio = torch.exp((logp - torch.as_tensor(batch.old_logp, device=model.device)).clamp(-20., 20.))
        weight = torch.as_tensor(model._normalize(advantage), device=model.device)
        clipped = ratio.clamp(1. - model.config.clip_ratio, 1. + model.config.clip_ratio)
        surrogate = torch.minimum(ratio * weight, clipped * weight).mean()
        return {"episode_kl": kl.tolist(), "max_episode_kl": float(kl.max()), "mean_episode_kl": float(kl.mean()),
                "clipped_surrogate": float(surrogate), "actor_objective": float(surrogate + model.config.entropy_coef * entropy.mean())}


def update(model, batch, task_rewards, treatment, *, root, iteration):
    if treatment not in spec.TREATMENTS:
        raise ValueError("unregistered episode KL treatment")
    initial = {name: copy.deepcopy(getattr(model, name).state_dict()) for name in LOWER_STATE}
    reference = copy.deepcopy(model.lower_actor)
    advantage = (model._gae(batch.reward, batch.done, batch.duration, batch.old_value, batch.next_value, batch.terminal)[0]
                 if treatment == "gae" else episodes.score_targets(task_rewards).reshape(-1).astype(np.float32))
    before = policy_terms(model, batch, reference, advantage, len(task_rewards))
    trials, first_critic, metrics, accepted, selected = [], None, None, False, 0.
    for backtrack in range(spec.MAX_BACKTRACKS + 1 if treatment == "episode_kl" else 1):
        # Restart the complete lower transaction so accepted Adam moments match the actual scaled-LR path.
        for name in LOWER_STATE:
            getattr(model, name).load_state_dict(copy.deepcopy(initial[name]))
        scale = 2. ** -backtrack
        for group, saved in zip(model.lower_actor_optimizer.param_groups, initial["lower_actor_optimizer"]["param_groups"]):
            group["lr"] = saved["lr"] * scale
        np.random.seed(spec.shuffle_seed(root, iteration))
        current = episodes.update(model, batch, task_rewards, treatment, specification=spec)
        terms = policy_terms(model, batch, reference, advantage, len(task_rewards))
        trials.append({"scale": scale, **terms, **{k: current[k] for k in ("actor_optimizer_steps", "value_optimizer_steps")}})
        if backtrack == 0:
            metrics = current
            first_critic = {name: copy.deepcopy(getattr(model, name).state_dict()) for name in ("lower_value", "lower_value_optimizer")}
        accepted = treatment != "episode_kl" or terms["max_episode_kl"] <= spec.KL_BUDGET
        if accepted:
            selected = scale
            break
    if not accepted:
        for name in ("lower_actor", "lower_actor_optimizer"):
            getattr(model, name).load_state_dict(copy.deepcopy(initial[name]))
    for group, saved in zip(model.lower_actor_optimizer.param_groups, initial["lower_actor_optimizer"]["param_groups"]):
        group["lr"] = saved["lr"]
    for name, state in first_critic.items():
        getattr(model, name).load_state_dict(state)
    deployed = policy_terms(model, batch, reference, advantage, len(task_rewards))
    return {**metrics, **{k: sum(t[k] for t in trials) for k in ("actor_optimizer_steps", "value_optimizer_steps")},
        "retained_actor_steps": metrics["actor_optimizer_steps"] if accepted else 0,
        "retained_value_steps": metrics["value_optimizer_steps"], "accepted": accepted, "selected_scale": selected,
        "before_terms": before, "deployed_terms": deployed, "trials": trials, "kl_check_calls": 2 + len(trials)}


def worker_rollout(job):
    return episodes.worker_rollout(job, specification=spec)


def train(root, *, preflight, output):
    return episodes.train(root, preflight=preflight, output=output, specification=spec,
                          rollout_worker=worker_rollout, update_fn=update)


def optimizer_steps(row, steps):
    trials = row["trials"]
    bounded = row["actor_credit"] == "episode_kl"
    if not 1 <= len(trials) <= (spec.MAX_BACKTRACKS + 1 if bounded else 1):
        raise ValueError("episode KL trial roster changed")
    for index, trial in enumerate(trials):
        if (trial["scale"] != 2. ** -index or trial["actor_optimizer_steps"] != steps or trial["value_optimizer_steps"] != steps
                or len(trial["episode_kl"]) != row["native_episodes"]
                or trial["max_episode_kl"] != max(trial["episode_kl"])
                or trial["mean_episode_kl"] != float(np.mean(trial["episode_kl"]))):
            raise ValueError("episode KL trial or executed cost changed")
        if bounded and index < len(trials) - 1 and trial["max_episode_kl"] <= spec.KL_BUDGET:
            raise ValueError("episode KL did not select the first feasible trial")
    accept = not bounded or trials[-1]["max_episode_kl"] <= spec.KL_BUDGET
    if (row["accepted"] != accept or row["selected_scale"] != (trials[-1]["scale"] if accept else 0.)
            or (not accept and len(trials) != spec.MAX_BACKTRACKS + 1)
            or row["retained_actor_steps"] != (steps if accept else 0) or row["retained_value_steps"] != steps
            or row["kl_check_calls"] != len(trials) + 2
            or row["deployed_terms"] != (dict((k, trials[-1][k]) for k in row["deployed_terms"]) if accept else row["before_terms"])
            or (bounded and row["deployed_terms"]["max_episode_kl"] > spec.KL_BUDGET)):
        raise ValueError("episode KL selected policy or retained accounting changed")
    return {"actor_optimizer_steps": len(trials) * steps, "value_optimizer_steps": len(trials) * steps}


def aggregate(results, *, preflight):
    summary = episodes.aggregate(results, preflight=preflight, specification=spec, step_counts=optimizer_steps)
    retained, checks = {"actor_optimizer_steps": 0, "value_optimizer_steps": 0}, 0
    by_root = {c["root"]: c for c in results}
    for root_row in summary["root_rows"]:
        cell, method = by_root[root_row["root"]], spec.METHODS[0]
        pairs = cell["first_batch_pair_details"][method]
        if set(pairs) != set(spec.TREATMENTS[1:]) or any(p["status"] != "passed" or p["native_task_rewards"] != "passed"
                or p["first_critic_update"] != "passed" or p["transitions"] != cell["options"]["rollouts_per_iteration"] * spec.arguments(cell["root"], preflight=preflight).horizon
                for p in pairs.values()):
            raise ValueError("episode KL first-batch pair incomplete")
        diagnostic = {}
        for treatment, history in cell["training"][method].items():
            for row in history:
                retained["actor_optimizer_steps"] += row["retained_actor_steps"]
                retained["value_optimizer_steps"] += row["retained_value_steps"]
                checks += row["kl_check_calls"]
            diagnostic[treatment] = {"accepted_updates": sum(r["accepted"] for r in history),
                "trial_updates": sum(len(r["trials"]) for r in history),
                "selected_scales": [r["selected_scale"] for r in history],
                "first_full_candidate": history[0]["trials"][0], "first_deployed": history[0]["deployed_terms"],
                "max_deployed_episode_kl": max(r["deployed_terms"]["max_episode_kl"] for r in history),
                "executed_actor_steps": sum(r["actor_optimizer_steps"] for r in history),
                "retained_actor_steps": sum(r["retained_actor_steps"] for r in history)}
        if diagnostic["episode_mc"]["first_full_candidate"] != diagnostic["episode_kl"]["first_full_candidate"]:
            raise ValueError("episode KL full candidate differs from paired original MC")
        root_row["kl_diagnostics"] = diagnostic
    summary["retained_optimizer_steps"] = retained
    summary["kl_check_calls"] = checks
    return summary
