"""Control actual PPO minibatch steps without changing the frozen PPO core."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import math
import multiprocessing as mp
import time

import numpy as np
import torch
from . import pointmaze_update_diagnostics as diagnostics
from . import pointmaze_matched_upper as previous
from . import pointmaze_joint_renewal as joint
from .pointmaze_root_response import write_json
from freq_hrl.rl.smdp_actor_critic import concat_hierarchical_batches
from scripts import pointmaze_first_update_stage59_spec as spec


class ConditionalKLGuard:
    """Rollback infeasible Adam steps against one fixed sampling distribution."""

    def __init__(self, actor, optimizer, state, budget):
        self.actor, self.optimizer, self.state, self.budget = actor, optimizer, state, budget
        with torch.no_grad():
            old = actor.distribution(state)
            self.reference = torch.distributions.Normal(old.mean.double().clone(), old.stddev.double().clone())
        self.steps, self.deployed_kl = [], 0.

    def before_step(self, optimizer, args, kwargs):
        self.actor_before = copy.deepcopy(self.actor.state_dict())
        self.adam_before = copy.deepcopy(optimizer.state_dict())

    def after_step(self, optimizer, args, kwargs):
        with torch.no_grad():
            new = self.actor.distribution(self.state)
            current = torch.distributions.Normal(new.mean.double(), new.stddev.double())
            kl = torch.distributions.kl_divergence(self.reference, current).sum(dim=-1).clamp_min(0.)
            candidate, maximum = float(kl.mean()), float(kl.max())
        accepted = candidate <= self.budget
        if accepted:
            self.deployed_kl = candidate
        else:
            self.actor.load_state_dict(self.actor_before)
            optimizer.load_state_dict(self.adam_before)
            torch.testing.assert_close(self.actor.state_dict(), self.actor_before, atol=0, rtol=0)
            torch.testing.assert_close(optimizer.state_dict(), self.adam_before, atol=0, rtol=0)
        self.steps.append({"step": len(self.steps) + 1, "candidate_kl": candidate,
            "candidate_max_kl": maximum, "accepted": accepted, "deployed_kl": self.deployed_kl,
            "rollback_check": None if accepted else "passed"})

    def __enter__(self):
        self.pre_hook = self.optimizer.register_step_pre_hook(self.before_step)
        self.post_hook = self.optimizer.register_step_post_hook(self.after_step)
        return self

    def __exit__(self, *exception):
        self.pre_hook.remove()
        self.post_hook.remove()

    def record(self):
        retained = sum(r["accepted"] for r in self.steps)
        return {"steps": self.steps, "attempted_actor_steps": len(self.steps), "retained_actor_steps": retained,
            "rejected_actor_steps": len(self.steps) - retained,
            "guard_distribution_passes": len(self.steps) + 1,
            "state_snapshot_calls": 2 * len(self.steps), "rollback_state_checks": 2 * (len(self.steps) - retained)}


def guarded_update(model, batch, *, level, root, period, episode_count, budget=spec.KL_BUDGET):
    state = torch.as_tensor(getattr(batch, level).state, dtype=torch.float32, device=model.device)
    with ConditionalKLGuard(getattr(model, level + "_actor"), getattr(model, level + "_actor_optimizer"), state, budget) as guard:
        row = diagnostics.observed_update(model, batch, level=level, phase="train", root=root,
            period=period, iteration=1, episode_count=episode_count)
    row["guard"] = guard.record()
    return row


def replay(root, *, preflight, output):
    file = spec.source.source_result(root, preflight=preflight)
    c = json.loads(file.read_text())
    reference = json.loads(spec.diagnostic_result(root, preflight=preflight).read_text())
    if (c["protocol"] != spec.source.previous.EXPERIMENT_PROTOCOL or c["contract"] != spec.source.previous.contract()
            or reference["protocol"] != spec.source.EXPERIMENT_PROTOCOL or reference["contract"] != spec.source.contract()
            or any((x["status"], x["root"], x["preflight"]) != ("complete", root, preflight) for x in (c, reference))):
        raise ValueError("Stage59 requires the frozen Stage57/58 sources")
    clones, _, source = previous.load_source(root, preflight=preflight)
    args, opt, budget = spec.source.previous.arguments(root, preflight=preflight), spec.options(preflight=preflight), spec.budget(preflight=preflight)
    raw, started = file.parent.with_name(file.parent.name + "_raw"), time.monotonic()
    costs = dict.fromkeys(budget, 0)
    costs.update(source_clone_loads=len(clones), forecaster_loads=1)
    warmup_steps = dict.fromkeys(("upper_actor", "upper_value", "lower_actor", "lower_value"), 0)
    config, comparisons = clones[str(spec.PERIODS[0])].config, {}
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"),
            initializer=diagnostics.init_worker, initargs=(config, args)) as pool:
        def archive_batch(model, period, arm, phase, item):
            directory = raw / str(period) / arm / phase / str(item["iteration"]) / "training"
            weights = joint.inference_weights(model)
            pairs = list(pool.map(diagnostics.worker_reconstruct, [(weights, str(directory / f"episode_{r['seed']}.npz"), r["seed"], period)
                                                                  for r in item["rows"]]))
            for (_, row), original in zip(pairs, item["rows"]):
                if row["episode_return"] != original["episode_return"]:
                    raise ValueError("Stage59 archived reward differs from source")
                costs["archive_episodes"] += 1
                costs["reconstructed_lower_calls"] += row["lower_calls"]
                costs["reconstructed_upper_calls"] += row["upper_calls"]
            return concat_hierarchical_batches([b for b, _ in pairs])

        for period in spec.PERIODS:
            p, clone = str(period), clones[str(period)]
            comparisons[p] = {}
            for arm in spec.TRAIN_POLICIES:
                model = copy.deepcopy(clone)
                for item in c["calibration"][p][arm]["history"]:
                    batch = archive_batch(model, period, arm, "warmup", item)
                    steps = previous.update(model, batch, arm=arm, phase="warmup", root=root, period=period, iteration=item["iteration"])
                    if steps != item["optimizer_steps"]:
                        raise ValueError("Stage59 critic warmup steps differ from source")
                    for key in warmup_steps:
                        warmup_steps[key] += steps.get(key + "_optimizer_steps", 0)
                    costs["warmup_critic_updates"] += 2
                batch = archive_batch(model, period, arm, "train", c["training"][p][arm]["history"][0])
                plain, bounded = copy.deepcopy(model), copy.deepcopy(model)
                rows = {t: [] for t in spec.TREATMENTS}
                for level in spec.source.levels(arm, "train"):
                    row = diagnostics.observed_update(plain, batch, level=level, phase="train", root=root,
                        period=period, iteration=1, episode_count=opt["rollouts_per_iteration"])
                    rows["plain"].append(row)
                    rows["conditional_kl"].append(guarded_update(bounded, batch, level=level, root=root, period=period,
                        episode_count=opt["rollouts_per_iteration"]))
                if rows["plain"] != reference["histories"][p]["train"][arm][0]["levels"]:
                    raise ValueError("Stage59 plain first-update diagnostics differ from exact Stage58 replay")
                for level in ("upper", "lower"):
                    for kind in ("value", "value_optimizer"):
                        name = level + "_" + kind
                        torch.testing.assert_close(getattr(plain, name).state_dict(), getattr(bounded, name).state_dict(), atol=0, rtol=0)
                if arm == "zero_train":
                    for kind in ("actor", "actor_optimizer"):
                        name = "upper_" + kind
                        torch.testing.assert_close(getattr(bounded, name).state_dict(), getattr(model, name).state_dict(), atol=0, rtol=0)
                n = sum(len(v) for v in rows.values())
                costs["diagnostic_updates"] += n
                costs["diagnostic_distribution_passes"] += 2 * n
                costs["diagnostic_value_passes"] += 2 * n
                costs["diagnostic_gae_calls"] += n
                comparisons[p][arm] = {"source_actions_check": "passed", "plain_reproduction": "passed",
                    "critic_networks_and_Adam_pair": "passed", "treatments": rows}
                print(f"paired first update {root}/period{period}/{arm}: source and critics exact", flush=True)
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "root": root, "preflight": preflight, "options": opt, "budget": budget, "cost": costs,
        "source_result": str(file), "source_diagnostics": str(spec.diagnostic_result(root, preflight=preflight)),
        "source_initialization": source, "warmup_optimizer_steps": warmup_steps, "comparisons": comparisons,
        "wall_seconds": time.monotonic() - started}
    qualify(result, preflight=preflight)
    write_json(output, result)
    return result


def qualify(c, *, preflight):
    if (c["status"] != "complete" or c["protocol"] != spec.EXPERIMENT_PROTOCOL or c["contract"] != spec.contract()
            or c["root"] not in spec.roots(preflight=preflight) or c["preflight"] != preflight
            or c["options"] != spec.options(preflight=preflight) or c["budget"] != spec.budget(preflight=preflight)
            or c["cost"] != c["budget"] or set(c["comparisons"]) != {str(p) for p in spec.PERIODS}):
        raise ValueError("Stage59 frozen source roster or cost changed")
    cfg, opt = c["source_initialization"]["config"], c["options"]
    horizon = spec.source.previous.arguments(c["root"], preflight=preflight).horizon
    steps = dict(c["warmup_optimizer_steps"])
    expected_warmup = dict.fromkeys(steps, 0)
    extra = dict.fromkeys(("guard_distribution_passes", "state_snapshot_calls", "rollback_state_checks", "retained_actor_steps", "rejected_actor_steps"), 0)
    rows, frozen = [], []
    for p, arms in c["comparisons"].items():
        if set(arms) != set(spec.TRAIN_POLICIES):
            raise ValueError("Stage59 arm roster changed")
        sizes = {"lower": horizon * opt["rollouts_per_iteration"], "upper": (horizon // int(p)) * opt["rollouts_per_iteration"]}
        for level in sizes:
            count = max(1, cfg["epochs"]) * math.ceil(sizes[level] / cfg["minibatch_size"])
            expected_warmup[level + "_value"] += len(spec.TRAIN_POLICIES) * opt["critic_warmup_iterations"] * count
        for arm, cell in arms.items():
            if any(cell[k] != "passed" for k in ("source_actions_check", "plain_reproduction", "critic_networks_and_Adam_pair")) or set(cell["treatments"]) != set(spec.TREATMENTS):
                raise ValueError("Stage59 source or paired critic identity failed")
            for treatment, updates in cell["treatments"].items():
                if [d["level"] for d in updates] != list(spec.source.levels(arm, "train")):
                    raise ValueError("Stage59 updated-level roster changed")
                for d in updates:
                    level = d["level"]
                    count = max(1, cfg["epochs"]) * math.ceil(sizes[level] / cfg["minibatch_size"])
                    expected = {level + "_actor_optimizer_steps": count, level + "_value_optimizer_steps": count,
                        level + "_cost_value_optimizer_steps": 0, level + "_advantage_optimizer_steps": 0}
                    if d["optimizer_steps"] != expected or d["batch_size"] != sizes[level]:
                        raise ValueError("Stage59 nominal optimizer attempts changed")
                    for kind in ("actor", "value"):
                        steps[level + "_" + kind] += count
                    compact = {"period": int(p), "arm": arm, "level": level, "treatment": treatment,
                        **{k: d[k] for k in ("kl_mean", "kl_max", "episode_kl_mean", "episode_kl_max", "clip_fraction",
                                             "mean_action_change_rms", "value_before", "value_after")}}
                    if treatment == "conditional_kl":
                        g, deployed = d["guard"], 0.
                        if [s["step"] for s in g["steps"]] != list(range(1, count + 1)):
                            raise ValueError("Stage59 guard attempt roster changed")
                        accepted = 0
                        for s in g["steps"]:
                            accept = s["candidate_kl"] <= spec.KL_BUDGET
                            if accept:
                                accepted += 1
                                deployed = s["candidate_kl"]
                            if s["accepted"] != accept or s["deployed_kl"] != deployed or s["rollback_check"] != (None if accept else "passed"):
                                raise ValueError("Stage59 KL acceptance or Adam rollback changed")
                        expected_guard = {"attempted_actor_steps": count, "retained_actor_steps": accepted,
                            "rejected_actor_steps": count - accepted, "guard_distribution_passes": count + 1,
                            "state_snapshot_calls": 2 * count, "rollback_state_checks": 2 * (count - accepted)}
                        if any(g[k] != v for k, v in expected_guard.items()) or d["kl_mean"] != deployed or deployed > spec.KL_BUDGET:
                            raise ValueError("Stage59 deployed KL or guard cost changed")
                        for key in extra:
                            extra[key] += g[key]
                        compact.update(expected_guard)
                        if accepted == 0 or d["mean_action_change_rms"] == 0:
                            frozen.append({"root": c["root"], "period": int(p), "arm": arm, "level": level})
                    rows.append(compact)
    if c["warmup_optimizer_steps"] != expected_warmup:
        raise ValueError("Stage59 critic warmup optimizer accounting changed")
    return {"root": c["root"], "updates": rows}, steps, extra, frozen


def aggregate(cells, *, preflight):
    if len(cells) != len(spec.roots(preflight=preflight)) or {c["root"] for c in cells} != set(spec.roots(preflight=preflight)):
        raise ValueError("Stage59 complete root roster required")
    by_root, rows, frozen = {c["root"]: c for c in cells}, [], []
    cost, steps, extra = dict.fromkeys(spec.budget(preflight=preflight), 0), {}, {}
    for root in spec.roots(preflight=preflight):
        c = by_root[root]
        row, actual, guard, failures = qualify(c, preflight=preflight)
        rows.append(row)
        frozen.extend(failures)
        for source, target in ((c["cost"], cost), (actual, steps), (guard, extra)):
            for k, v in source.items():
                target[k] = target.get(k, 0) + v
    return {"status": "preflight_passed" if preflight else "fixed_batch_valid", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "root_rows": rows, "cost": cost, "executed_optimizer_steps": steps,
        "guard_cost_and_retained_steps": extra, "frozen_actors": frozen,
        "native_trial_prerequisite": "hold_frozen_actor" if frozen else "nonzero_step_passed_not_performance_evidence",
        "performance_claim": "none_archive_only_zero_new_native_and_evaluation_steps"}
