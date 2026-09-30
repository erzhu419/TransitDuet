"""Reconstruct archived on-policy batches and measure unchanged PPO updates."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from freq_hrl.domains.mujoco.pointmaze_regime import PointMazeRegimeObservation
from freq_hrl.rl.smdp_actor_critic import (FrequencySeparatedActorCriticPPO, HierarchicalRolloutBuilder,
    concat_hierarchical_batches)
from . import pointmaze_joint_renewal as joint
from . import pointmaze_matched_upper as previous
from .pointmaze_plan_value_qualification import PointMazeRegimeFeatureBuilder
from .pointmaze_goal_validation import squash_box_action
from .pointmaze_root_response import write_json
from scripts import pointmaze_update_diagnostics_stage58_spec as spec


_WORKER = None


def init_worker(config, args):
    global _WORKER
    torch.set_num_threads(1)
    _WORKER = FrequencySeparatedActorCriticPPO(config), args


def reconstruct(model, args, raw, *, seed, period):
    torch.manual_seed(spec.previous.policy_seed(args.optimizer_seed, seed))
    model.reset_recurrent_inference()
    history = PointMazeRegimeFeatureBuilder(time_scale=joint.scale_for(args))
    builder = HierarchicalRolloutBuilder(gamma=model.config.gamma)
    upper_actions, actions = [], []
    lower_seed = spec.previous.rollout_arguments(args.optimizer_seed, seed, phase="train", mode="training")["lower_seed"]
    # Preserve scalar inference and RNG order; batched inference would not reproduce rollout log probabilities.
    for step in range(args.horizon):
        sample = raw["measurement"][step]
        obs = PointMazeRegimeObservation(physical=raw["physical"][step], achieved_goal=raw["achieved_before"][step],
            target=raw["target_before"][step], task_measurement=sample, force=sample[2:4], distractor=sample[4:6])
        history.reset(obs) if step == 0 else history.update(obs)
        plan_now = step % period == 0
        if plan_now:
            state = history.upper_state(obs, oracle_context=None)
            output = model.act_upper(state, sample=True)
            upper_actions.append(output["action"])
            builder.begin_upper(state=state, action=output["action"], logp=output["logp"], value=output["value"])
        base = history.lower_state(obs, subgoal=raw["lower_reference"][step])
        state = np.concatenate((base, raw["lower_actor_context"][step])).astype(np.float32)
        value_state = np.concatenate((base, raw["lower_value_context"][step])).astype(np.float32)
        cost_state = base if model.lower_cost_value is not None else None
        torch.manual_seed(int(lower_seed) + step)
        kwargs = {} if cost_state is None else {"cost_state": cost_state}
        output = model.act_lower(state, sample=True, value_state=value_state, **kwargs)
        actions.append(squash_box_action(output["action"], np.full(2, -1.), np.full(2, 1.)))
        reward = float(raw["reward"][step])
        builder.add_lower(state=state, action=output["action"], logp=output["logp"], value=output["value"],
            reward=reward, upper_reward=reward - joint.spec.CALL_COST * int(plan_now),
            done=step + 1 == args.horizon, cost=0., value_state=value_state, cost_state=cost_state)
    np.testing.assert_array_equal(np.asarray(upper_actions), raw["upper_proposed_action"])
    np.testing.assert_array_equal(np.asarray(actions), raw["action"])
    np.testing.assert_array_equal(raw["decision_steps"], np.arange(0, args.horizon, period))
    builder.finish(terminal=True)
    batch = builder.build()
    batch.lower.done[raw["decision_steps"][1:].astype(int) - 1] = 1.
    return batch


def worker_reconstruct(job):
    weights, path, seed, period = job
    model, args = _WORKER
    model.load_state_dict(weights)
    with np.load(path) as archive:
        raw = {k: archive[k] for k in archive.files}
    batch = reconstruct(model, args, raw, seed=seed, period=period)
    torch.testing.assert_close(joint.inference_weights(model), weights, atol=0, rtol=0)
    return batch, {"seed": seed, "episode_return": float(np.sum(raw["reward"])),
        "lower_calls": args.horizon, "upper_calls": args.horizon // period, "action_check": "passed"}


def value_terms(prediction, target):
    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    variance = float(np.var(target))
    return {"mse": float(np.mean(np.square(prediction - target))),
        "explained_variance": None if variance == 0 else float(1. - np.var(target - prediction) / variance)}


def observed_update(model, batch, *, level, phase, root, period, iteration, episode_count):
    actor, value_net = getattr(model, level + "_actor"), getattr(model, level + "_value")
    level_batch = getattr(batch, level)
    state = torch.as_tensor(level_batch.state, dtype=torch.float32, device=model.device)
    value_state = torch.as_tensor(level_batch.state if level_batch.value_state is None else level_batch.value_state,
        dtype=torch.float32, device=model.device)
    action = torch.as_tensor(level_batch.action, dtype=torch.float32, device=model.device)
    old_logp = torch.as_tensor(level_batch.old_logp, dtype=torch.float32, device=model.device)
    advantage, target = model._gae(level_batch.reward, level_batch.done, level_batch.duration,
        level_batch.old_value, level_batch.next_value, level_batch.terminal)
    with torch.no_grad():
        before = actor.distribution(state)
        old = torch.distributions.Normal(before.mean.double().clone(), before.stddev.double().clone())
        before_logp = before.log_prob(action).sum(dim=-1)
        before_value = value_terms(value_net(value_state).cpu().numpy(), target)
    np.random.seed(spec.previous.shuffle_seed(root, period, iteration, phase=phase, level=level))
    metrics = model._update_level(level=level, batch=level_batch, actor=actor, value_net=value_net,
        actor_optimizer=getattr(model, level + "_actor_optimizer"), value_optimizer=getattr(model, level + "_value_optimizer"),
        actor_updates_enabled=phase == "train")
    with torch.no_grad():
        after = actor.distribution(state)
        current = torch.distributions.Normal(after.mean.double(), after.stddev.double())
        conditional = torch.distributions.kl_divergence(old, current).sum(dim=-1).clamp_min(0.)
        episode_kl = conditional.reshape(episode_count, -1).sum(dim=1)
        log_ratio = after.log_prob(action).sum(dim=-1) - old_logp
        ratio = torch.exp(log_ratio.clamp(-20., 20.))
        clipped = (ratio < 1. - model.config.clip_ratio) | (ratio > 1. + model.config.clip_ratio)
        after_value = value_terms(value_net(value_state).cpu().numpy(), target)
    return {"level": level, "batch_size": level_batch.size,
        "optimizer_steps": {k: int(v) for k, v in metrics.items() if "optimizer_steps" in k},
        "ppo_metrics": metrics, "kl_mean": float(conditional.mean()), "kl_max": float(conditional.max()),
        "episode_kl_mean": float(episode_kl.mean()), "episode_kl_max": float(episode_kl.max()),
        "clip_fraction": float(clipped.float().mean()), "ratio_mean": float(ratio.mean()),
        "before_logp_error_max": float((before_logp - old_logp).abs().max()),
        "before_logp_error_mean": float((before_logp - old_logp).abs().mean()),
        "value_before": before_value, "value_after": after_value,
        "target_mean": float(np.mean(target)), "target_std": float(np.std(target)),
        "advantage_std": float(np.std(advantage)), "std_before": float(old.stddev.mean()),
        "std_after": float(current.stddev.mean()),
        "mean_action_change_rms": float(torch.sqrt(torch.square(current.mean - old.mean).mean()))}


def replay(root, *, preflight, output):
    file = spec.source_result(root, preflight=preflight)
    c = json.loads(file.read_text())
    source_row, _, expected_steps = previous.qualify(c, preflight=preflight)
    clones, _, source = previous.load_source(root, preflight=preflight)
    args, opt, budget = spec.previous.arguments(root, preflight=preflight), spec.options(preflight=preflight), spec.budget(preflight=preflight)
    source_raw, started = file.parent.with_name(file.parent.name + "_raw"), time.monotonic()
    histories, checks = {}, {}
    costs = dict.fromkeys(budget, 0)
    costs.update(source_clone_loads=len(clones), forecaster_loads=1)
    config = clones[str(spec.PERIODS[0])].config
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"),
                             initializer=init_worker, initargs=(config, args)) as pool:
        for period in spec.PERIODS:
            p, clone = str(period), clones[str(period)]
            models = {arm: copy.deepcopy(clone) for arm in spec.TRAIN_POLICIES}
            histories[p], checks[p] = {}, {}
            for phase, key in (("warmup", "calibration"), ("train", "training")):
                histories[p][phase] = {}
                for arm, model in models.items():
                    history = []
                    for item in c[key][p][arm]["history"]:
                        iteration = item["iteration"]
                        weights = joint.inference_weights(model)
                        directory = source_raw / p / arm / phase / str(iteration) / "training"
                        jobs = [(weights, str(directory / f"episode_{row['seed']}.npz"), row["seed"], period) for row in item["rows"]]
                        pairs = list(pool.map(worker_reconstruct, jobs))
                        for (_, row), original in zip(pairs, item["rows"]):
                            if row["episode_return"] != original["episode_return"]:
                                raise ValueError("Stage58 archived reward differs from source")
                            costs["archive_episodes"] += 1
                            costs["reconstructed_lower_calls"] += row["lower_calls"]
                            costs["reconstructed_upper_calls"] += row["upper_calls"]
                        batch = concat_hierarchical_batches([b for b, _ in pairs])
                        diagnostics = [observed_update(model, batch, level=level, phase=phase, root=root,
                            period=period, iteration=iteration, episode_count=len(pairs)) for level in spec.levels(arm, phase)]
                        steps = {k: v for d in diagnostics for k, v in d["optimizer_steps"].items()}
                        if steps != item["optimizer_steps"]:
                            raise ValueError("Stage58 reproduced optimizer steps differ from source")
                        n = len(diagnostics)
                        costs["diagnostic_updates"] += n
                        costs["diagnostic_distribution_passes"] += 2 * n
                        costs["diagnostic_value_passes"] += 2 * n
                        costs["diagnostic_gae_calls"] += n
                        history.append({"iteration": iteration, "native_return_mean": float(np.mean([r["episode_return"] for _, r in pairs])),
                            "source_actions_check": "passed", "levels": diagnostics})
                    histories[p][phase][arm] = history
            for arm, model in models.items():
                final = torch.load(c["checkpoints"][p][arm], map_location="cpu", weights_only=False)
                if (final["protocol"], final["root"], final["period"], final["policy"]) != (spec.previous.EXPERIMENT_PROTOCOL, root, period, arm):
                    raise ValueError("Stage58 final reference checkpoint differs from Stage57")
                torch.testing.assert_close(model.state_dict(), final["state_dict"], atol=0, rtol=0)
                checks[p][arm] = "passed"
                costs["final_checkpoint_loads"] += 1
                print(f"reproduced {root}/period{period}/{arm}: final networks and optimizers exact", flush=True)
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "root": root, "preflight": preflight, "source_result": str(file), "source_initialization": source,
        "options": opt, "budget": budget, "cost": costs, "histories": histories, "final_state_checks": checks,
        "expected_optimizer_steps": expected_steps, "source_endpoints": source_row["endpoints"],
        "wall_seconds": time.monotonic() - started}
    qualify(result, preflight=preflight)
    write_json(output, result)
    return result


def summarize(rows):
    return {"iterations": len(rows),
        **{k + "_mean": float(np.mean([r[k] for r in rows])) for k in
           ("kl_mean", "episode_kl_mean", "clip_fraction", "before_logp_error_mean")},
        "episode_kl_max": max(r["episode_kl_max"] for r in rows),
        "before_logp_error_max": max(r["before_logp_error_max"] for r in rows),
        "value_mse_first": rows[0]["value_before"]["mse"], "value_mse_last": rows[-1]["value_after"]["mse"],
        "value_ev_first": rows[0]["value_before"]["explained_variance"], "value_ev_last": rows[-1]["value_after"]["explained_variance"],
        "target_std_first": rows[0]["target_std"], "target_std_last": rows[-1]["target_std"],
        "std_first": rows[0]["std_before"], "std_last": rows[-1]["std_after"]}


def qualify(c, *, preflight):
    if (c["status"] != "complete" or c["protocol"] != spec.EXPERIMENT_PROTOCOL or c["contract"] != spec.contract()
            or c["preflight"] != preflight or c["root"] not in spec.roots(preflight=preflight)
            or c["options"] != spec.options(preflight=preflight) or c["budget"] != spec.budget(preflight=preflight)
            or c["cost"] != c["budget"]):
        raise ValueError("Stage58 protocol or replay accounting changed")
    opt, steps, groups = c["options"], dict.fromkeys(c["expected_optimizer_steps"], 0), {}
    if set(c["histories"]) != {str(p) for p in spec.PERIODS} or set(c["final_state_checks"]) != set(c["histories"]):
        raise ValueError("Stage58 period roster changed")
    for p, phases in c["histories"].items():
        if set(phases) != {"warmup", "train"} or c["final_state_checks"][p] != dict.fromkeys(spec.TRAIN_POLICIES, "passed"):
            raise ValueError("Stage58 final-state identity or phase roster changed")
        groups[p] = {}
        for phase, arms in phases.items():
            groups[p][phase] = {}
            iterations = opt["critic_warmup_iterations"] if phase == "warmup" else opt["learning_iterations"]
            if set(arms) != set(spec.TRAIN_POLICIES):
                raise ValueError("Stage58 arm roster changed")
            for arm, history in arms.items():
                if [r["iteration"] for r in history] != list(range(1, iterations + 1)):
                    raise ValueError("Stage58 iteration roster changed")
                by_level = {level: [] for level in spec.levels(arm, phase)}
                for row in history:
                    if row["source_actions_check"] != "passed" or [r["level"] for r in row["levels"]] != list(by_level):
                        raise ValueError("Stage58 action identity or updated-level roster changed")
                    for d in row["levels"]:
                        if phase == "warmup" and (d["kl_max"] != 0 or d["std_before"] != d["std_after"]):
                            raise ValueError("Stage58 critic calibration changed actor")
                        by_level[d["level"]].append(d)
                        for key in steps:
                            steps[key] += d["optimizer_steps"].get(key + "_optimizer_steps", 0)
                groups[p][phase][arm] = {level: summarize(rows) for level, rows in by_level.items()}
    if steps != c["expected_optimizer_steps"]:
        raise ValueError("Stage58 total optimizer steps differ from Stage57")
    return {"root": c["root"], "groups": groups, "source_endpoints": c["source_endpoints"]}, steps


def aggregate(cells, *, preflight):
    if len(cells) != len(spec.roots(preflight=preflight)) or {c["root"] for c in cells} != set(spec.roots(preflight=preflight)):
        raise ValueError("Stage58 root roster incomplete")
    by_root = {c["root"]: c for c in cells}
    rows, steps, cost = [], {}, dict.fromkeys(spec.budget(preflight=preflight), 0)
    for root in spec.roots(preflight=preflight):
        c = by_root[root]
        row, actual = qualify(c, preflight=preflight)
        rows.append(row)
        for key, value in actual.items():
            steps[key] = steps.get(key, 0) + value
        for key, value in c["cost"].items():
            cost[key] += value
    return {"status": "preflight_passed" if preflight else "diagnostics_complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "root_rows": rows, "cost": cost, "replayed_optimizer_steps": steps,
        "performance_claim": "none_new_environment_and_evaluation_steps_zero"}
