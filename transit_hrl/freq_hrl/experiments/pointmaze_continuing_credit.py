"""Recalibrate continuing lower values and compare one guarded PPO update."""

from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import replace
import json
import math
import multiprocessing as mp
import time

import numpy as np
import torch

from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, concat_hierarchical_batches
from . import pointmaze_first_update as guarded
from . import pointmaze_update_diagnostics as diagnostics
from . import pointmaze_matched_upper as previous
from . import pointmaze_joint_renewal as joint
from . import pointmaze_credit_diagnostics as credit
from .pointmaze_critic_calibration import monte_carlo_returns
from .pointmaze_root_response import raw_directory, write_json
from scripts import pointmaze_episode_credit_stage63_spec as spec


_EPISODE_WORKER = None


def init_worker(config, args):
    global _EPISODE_WORKER
    diagnostics.init_worker(config, args)
    _EPISODE_WORKER = FrequencySeparatedActorCriticPPO(config)


def episode_predictions(model, lower):
    # Scalar value inference preserves the native old-value arithmetic exactly.
    with torch.no_grad():
        return np.asarray([float(model.lower_value(torch.as_tensor(state, dtype=torch.float32,
            device=model.device).unsqueeze(0)).item()) for state in lower.value_state], dtype=np.float32)


def worker_pair(job):
    option_weights, episode_weights, path, seed, period = job
    batch, row = diagnostics.worker_reconstruct((option_weights, path, seed, period))
    _EPISODE_WORKER.load_state_dict(episode_weights)
    values = episode_predictions(_EPISODE_WORKER, batch.lower)
    torch.testing.assert_close(joint.inference_weights(_EPISODE_WORKER), episode_weights, atol=0, rtol=0)
    return batch, values, {**row, "episode_value_calls": len(values), "network_checks": 2}


def episode_batch(batch, values, horizon):
    done = np.zeros_like(batch.lower.done)
    done[horizon - 1::horizon] = 1.
    return replace(batch, lower=replace(batch.lower, done=done, old_value=values))


def lower_calibration(model, batch, *, root, period, iteration):
    np.random.seed(spec.source.source.previous.shuffle_seed(root, period, iteration, phase="warmup", level="lower"))
    metrics = model._update_level(level="lower", batch=batch.lower, actor=model.lower_actor, value_net=model.lower_value,
        actor_optimizer=model.lower_actor_optimizer, value_optimizer=model.lower_value_optimizer, actor_updates_enabled=False)
    return {k: int(v) for k, v in metrics.items() if "optimizer_steps" in k}


def probe(option, episode, option_batch, continuing_batch):
    lower, other = option_batch.lower, continuing_batch.lower
    option_mc = monte_carlo_returns(lower, option.config.gamma)
    episode_mc = monte_carlo_returns(other, episode.config.gamma)
    a, _ = option._gae(lower.reward, lower.done, lower.duration, lower.old_value)
    b, _ = episode._gae(other.reward, other.done, other.duration, other.old_value)
    same_values, _ = option._gae(other.reward, other.done, other.duration, lower.old_value)
    return {"option_value_option_mc": credit.value_metrics(lower.old_value, option_mc),
        "option_value_episode_mc": credit.value_metrics(lower.old_value, episode_mc),
        "episode_value_episode_mc": credit.value_metrics(other.old_value, episode_mc),
        "recalibrated_advantage_alignment": credit.alignment(a, b),
        "same_value_boundary_alignment": credit.alignment(a, same_values),
        "option_done_count": int(lower.done.sum()), "episode_done_count": int(other.done.sum())}


def replay(root, *, preflight, output):
    file = spec.source.source.source_result(root, preflight=preflight)
    original = json.loads(file.read_text())
    reference = json.loads(spec.source_result(root, preflight=preflight).read_text())
    if any((c["status"], c["root"], c["preflight"]) != ("complete", root, preflight) for c in (original, reference)):
        raise ValueError("Stage63 requires the completed Stage57/60 sources")
    if original["contract"] != spec.source.source.previous.contract() or reference["contract"] != spec.source.contract():
        raise ValueError("Stage63 frozen source changed")
    roles = spec.seed_roles(root, preflight=preflight)
    if set(roles["calibration"]).intersection(roles["first_training_probe"]):
        raise ValueError("Stage63 critic probe overlaps calibration")
    clones, _, source = previous.load_source(root, preflight=preflight)
    args, opt, budget = spec.arguments(root, preflight=preflight), spec.options(preflight=preflight), spec.budget(preflight=preflight)
    cost, started = dict.fromkeys(budget, 0), time.monotonic()
    cost.update(source_clone_loads=len(clones), forecaster_loads=1)
    source_raw = file.parent.with_name(file.parent.name + "_raw")
    output_raw = raw_directory(output)
    comparisons, warmup_steps = {}, dict.fromkeys(("upper_actor", "upper_value", "lower_actor", "lower_value"), 0)
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"), initializer=init_worker,
            initargs=(clones[str(spec.PERIODS[0])].config, args)) as pool:
        def batches(option, episode, period, arm, phase, item):
            directory = source_raw / str(period) / arm / phase / str(item["iteration"]) / "training"
            weights, other = joint.inference_weights(option), joint.inference_weights(episode)
            pairs = list(pool.map(worker_pair, [(weights, other, str(directory / f"episode_{r['seed']}.npz"), r["seed"], period)
                                               for r in item["rows"]]))
            for (_, _, row), original_row in zip(pairs, item["rows"]):
                if row["episode_return"] != original_row["episode_return"] or row["action_check"] != "passed":
                    raise ValueError("Stage63 source action or reward identity failed")
                cost["archive_episodes"] += 1
                cost["reconstructed_lower_calls"] += row["lower_calls"]
                cost["reconstructed_upper_calls"] += row["upper_calls"]
                cost["episode_critic_scalar_calls"] += row["episode_value_calls"]
                cost["archive_network_checks"] += row["network_checks"]
            batch = concat_hierarchical_batches([b for b, _, _ in pairs])
            return batch, episode_batch(batch, np.concatenate([v for _, v, _ in pairs]), args.horizon)

        for period in spec.PERIODS:
            p, clone = str(period), clones[str(period)]
            comparisons[p] = {}
            for arm in spec.TRAIN_POLICIES:
                option, episode = copy.deepcopy(clone), copy.deepcopy(clone)
                history = original["calibration"][p][arm]["history"]
                if [r["seed"] for item in history for r in item["rows"]] != roles["calibration"]:
                    raise ValueError("Stage63 calibration seed roster changed")
                for item in history:
                    batch, continuing = batches(option, episode, period, arm, "warmup", item)
                    control_steps = previous.update(option, batch, arm=arm, phase="warmup", root=root,
                        period=period, iteration=item["iteration"])
                    if control_steps != item["optimizer_steps"]:
                        raise ValueError("Stage63 control calibration steps differ from source")
                    extra_steps = lower_calibration(episode, continuing, root=root, period=period, iteration=item["iteration"])
                    for row in (control_steps, extra_steps):
                        for key in warmup_steps:
                            warmup_steps[key] += row.get(key + "_optimizer_steps", 0)
                    cost["warmup_critic_updates"] += 3
                    cost["ppo_gae_calls"] += 3
                for model in (option, episode):
                    for level in ("upper", "lower"):
                        for kind in ("actor", "actor_optimizer"):
                            name = level + "_" + kind
                            torch.testing.assert_close(getattr(model, name).state_dict(), getattr(clone, name).state_dict(), atol=0, rtol=0)
                            cost["warmup_actor_state_checks"] += 1
                first = original["training"][p][arm]["history"][0]
                if [r["seed"] for r in first["rows"]] != roles["first_training_probe"]:
                    raise ValueError("Stage63 first training seed roster changed")
                batch, continuing = batches(option, episode, period, arm, "train", first)
                fit = probe(option, episode, batch, continuing)
                cost["probe_mc_calls"] += 2
                cost["diagnostic_gae_calls"] += 3
                control = [guarded.guarded_update(option, batch, level=level, root=root, period=period,
                    episode_count=opt["rollouts_per_iteration"], guard_type=guarded.BacktrackingKLGuard)
                    for level in spec.source.source.levels(arm, "train")]
                if control != reference["comparisons"][p][arm]["treatments"]["backtracking_kl"]:
                    raise ValueError("Stage63 option control differs from exact Stage60 update")
                # Frozen first-batch upper work is common to both credit treatments.
                for kind in ("actor", "value", "actor_optimizer", "value_optimizer"):
                    name = "upper_" + kind
                    getattr(episode, name).load_state_dict(getattr(option, name).state_dict())
                    cost["upper_state_transfers"] += 1
                other = [guarded.guarded_update(episode, continuing, level="lower", root=root, period=period,
                    episode_count=opt["rollouts_per_iteration"], guard_type=guarded.BacktrackingKLGuard)]
                for kind in ("actor", "value", "actor_optimizer", "value_optimizer"):
                    name = "upper_" + kind
                    torch.testing.assert_close(getattr(episode, name).state_dict(), getattr(option, name).state_dict(), atol=0, rtol=0)
                    cost["upper_pair_state_checks"] += 1
                n = len(control) + len(other)
                cost["diagnostic_updates"] += n
                cost["ppo_gae_calls"] += n
                cost["diagnostic_gae_calls"] += n
                cost["diagnostic_distribution_passes"] += 2 * n
                cost["diagnostic_value_passes"] += 2 * n
                checkpoints = {}
                for treatment, model in zip(spec.TREATMENTS, (option, episode)):
                    checkpoint = output_raw / p / arm / treatment / "policy.pt"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "period": period, "arm": arm,
                        "treatment": treatment, "state_dict": model.state_dict()}, checkpoint)
                    checkpoints[treatment] = str(checkpoint)
                    cost["candidate_checkpoint_writes"] += 1
                comparisons[p][arm] = {"control_reproduction": "passed", "source_actions_check": "passed",
                    "warmup_actors_and_Adam": "passed", "upper_networks_and_Adam_pair": "passed",
                    "critic_probe": fit, "updates": dict(zip(spec.TREATMENTS, (control, other))), "checkpoints": checkpoints}
                print(f"episode-credit first update {root}/period{period}/{arm}: control and shared upper exact", flush=True)
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "root": root, "preflight": preflight, "options": opt, "seed_roles": roles, "budget": budget, "cost": cost,
        "source_initialization": source, "source_result": str(file), "source_reference": str(spec.source_result(root, preflight=preflight)),
        "warmup_optimizer_steps": warmup_steps, "comparisons": comparisons, "wall_seconds": time.monotonic() - started}
    qualify(result, preflight=preflight)
    write_json(output, result)
    return result


def qualify(c, *, preflight):
    if (c["status"] != "complete" or c["protocol"] != spec.EXPERIMENT_PROTOCOL or c["contract"] != spec.contract()
            or c["root"] not in spec.roots(preflight=preflight) or c["preflight"] != preflight
            or c["options"] != spec.options(preflight=preflight) or c["seed_roles"] != spec.seed_roles(c["root"], preflight=preflight)
            or c["budget"] != spec.budget(preflight=preflight) or c["cost"] != c["budget"]
            or set(c["comparisons"]) != {str(p) for p in spec.PERIODS}):
        raise ValueError("Stage63 frozen roster, protocol or cost changed")
    cfg, opt = c["source_initialization"]["config"], c["options"]
    horizon = spec.arguments(c["root"], preflight=preflight).horizon
    expected_warm = dict.fromkeys(c["warmup_optimizer_steps"], 0)
    steps = dict(expected_warm)
    guard_cost, frozen, fit_failures, groups = {}, [], [], {}
    for p, arms in c["comparisons"].items():
        if set(arms) != set(spec.TRAIN_POLICIES):
            raise ValueError("Stage63 arm roster changed")
        sizes = {"lower": horizon * opt["rollouts_per_iteration"], "upper": horizon // int(p) * opt["rollouts_per_iteration"]}
        counts = {level: max(1, cfg["epochs"]) * math.ceil(n / cfg["minibatch_size"]) for level, n in sizes.items()}
        expected_warm["upper_value"] += len(arms) * opt["critic_warmup_iterations"] * counts["upper"]
        expected_warm["lower_value"] += 2 * len(arms) * opt["critic_warmup_iterations"] * counts["lower"]
        groups[p] = {}
        for arm, cell in arms.items():
            if (any(cell[k] != "passed" for k in ("control_reproduction", "source_actions_check", "warmup_actors_and_Adam", "upper_networks_and_Adam_pair"))
                    or set(cell["updates"]) != set(spec.TREATMENTS) or set(cell["checkpoints"]) != set(spec.TREATMENTS)):
                raise ValueError("Stage63 control, shared upper or checkpoint roster changed")
            fit = cell["critic_probe"]
            paths = opt["rollouts_per_iteration"]
            if fit["option_done_count"] != paths * horizon // int(p) or fit["episode_done_count"] != paths:
                raise ValueError("Stage63 lower termination semantics changed")
            e, baseline = fit["episode_value_episode_mc"], fit["option_value_episode_mc"]
            if e["explained_variance"] is None or e["explained_variance"] <= 0 or e["mse"] >= baseline["mse"]:
                fit_failures.append({"root": c["root"], "period": int(p), "arm": arm})
            groups[p][arm] = {"critic_probe": fit, "updates": {}}
            for treatment, updates in cell["updates"].items():
                levels = list(spec.source.source.levels(arm, "train")) if treatment == "option_credit" else ["lower"]
                if [d["level"] for d in updates] != levels:
                    raise ValueError("Stage63 shared/updated level roster changed")
                groups[p][arm]["updates"][treatment] = []
                for d in updates:
                    level, g = d["level"], d["guard"]
                    count = counts[level]
                    wanted = {level + "_actor_optimizer_steps": count, level + "_value_optimizer_steps": count,
                              level + "_cost_value_optimizer_steps": 0, level + "_advantage_optimizer_steps": 0}
                    if (d["optimizer_steps"] != wanted or d["batch_size"] != sizes[level]
                            or g["attempted_actor_steps"] != count or len(g["steps"]) != count
                            or d["kl_mean"] != g["steps"][-1]["deployed_kl"] or d["kl_mean"] > spec.KL_BUDGET):
                        raise ValueError("Stage63 optimizer attempts or conditional KL budget changed")
                    for kind in ("actor", "value"):
                        steps[level + "_" + kind] += count
                    for key in ("candidate_evaluations", "parameter_interpolation_trials", "guard_distribution_passes",
                                "state_snapshot_calls", "rollback_state_checks", "retained_actor_steps", "rejected_actor_steps"):
                        guard_cost[key] = guard_cost.get(key, 0) + g[key]
                    if g["retained_actor_steps"] == 0 or d["mean_action_change_rms"] == 0:
                        frozen.append({"root": c["root"], "period": int(p), "arm": arm, "treatment": treatment, "level": level})
                    groups[p][arm]["updates"][treatment].append({key: d[key] for key in
                        ("level", "kl_mean", "kl_max", "clip_fraction", "mean_action_change_rms", "value_before", "value_after")})
    if c["warmup_optimizer_steps"] != expected_warm:
        raise ValueError("Stage63 critic calibration optimizer accounting changed")
    steps = {k: v + expected_warm[k] for k, v in steps.items()}
    return {"root": c["root"], "groups": groups}, steps, guard_cost, frozen, fit_failures


def aggregate(cells, *, preflight):
    by_root = {c["root"]: c for c in cells}
    if len(by_root) != len(cells) or set(by_root) != set(spec.roots(preflight=preflight)):
        raise ValueError("Stage63 complete root roster required")
    rows, steps, guard_cost, frozen, fit_failures = [], {}, {}, [], []
    for root in spec.roots(preflight=preflight):
        row, actual, guards, failures, bad_fit = qualify(by_root[root], preflight=preflight)
        rows.append(row)
        for source, target in ((actual, steps), (guards, guard_cost)):
            for key, value in source.items():
                target[key] = target.get(key, 0) + value
        frozen.extend(failures)
        fit_failures.extend(bad_fit)
    return {"status": "preflight_passed" if preflight else "fixed_batch_valid", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "root_rows": rows, "cost": {k: sum(c["cost"][k] for c in cells) for k in spec.budget(preflight=preflight)},
        "executed_optimizer_steps": steps, "guard_cost": guard_cost, "frozen_actors": frozen, "episode_critic_fit_failures": fit_failures,
        "mechanical_gate": "failed" if frozen else "passed", "episode_critic_fit_gate": "failed" if fit_failures else "passed",
        "native_trial_prerequisite": "not_applicable_preflight" if preflight else "hold" if frozen or fit_failures else "passed_not_performance_evidence",
        "performance_claim": "none_archive_only"}
