"""Unit-consistent critic continuation and paired guarded native actor test."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import math
import multiprocessing as mp
import time

import numpy as np
import torch

from freq_hrl.rl.smdp_actor_critic import concat_hierarchical_batches
from . import pointmaze_value_targets as values
from . import pointmaze_continuing_credit as continuing
from . import pointmaze_first_update as guarded
from . import pointmaze_matched_upper as native
from . import pointmaze_native_update as evaluation
from . import pointmaze_joint_renewal as joint
from . import pointmaze_learned_plan as learned
from . import pointmaze_credit_diagnostics as credit
from .pointmaze_root_response import raw_directory, write_json
from scripts import pointmaze_normalized_update_stage65_spec as spec


def restore_fit(base, checkpoint, *, root, period, arm, treatment):
    if (checkpoint["protocol"], checkpoint["root"], checkpoint["period"], checkpoint["arm"], checkpoint["treatment"]) != (
            spec.source.EXPERIMENT_PROTOCOL, root, period, arm, treatment):
        raise ValueError("Stage65 critic source changed")
    if checkpoint["config"] != base.state_dict()["config"]:
        raise ValueError("Stage65 critic configuration changed")
    fit = values.ValueFit(copy.deepcopy(base), treatment)
    fit.location, fit.scale = checkpoint["location"], checkpoint["scale"]
    fit.initialized = treatment.endswith("normalized")
    fit.model.lower_value.load_state_dict(checkpoint["value_training_state"])
    fit.model.lower_value_optimizer.load_state_dict(checkpoint["value_optimizer_training_units"])
    torch.testing.assert_close(fit.public_state(), checkpoint["public_value_state"], atol=0, rtol=0)
    return fit


def public_model(fit):
    model = copy.deepcopy(fit.model)
    model.lower_value.load_state_dict(fit.public_state())
    return model


def actor_update(fit, lower, *, root, period):
    model, cfg = fit.model, fit.model.config
    if cfg.lower_actor_anchor_coef or cfg.lower_projection_consistency_coef or model.lower_cost_value is not None:
        raise ValueError("Stage65 uses the frozen unconstrained Stage55 actor objective")
    lower.validate(state_dim=cfg.lower_state_dim, action_dim=cfg.lower_action_dim, level="lower",
        value_state_dim=model.lower_value_state_dim)
    advantage, target = model._gae(lower.reward, lower.done, lower.duration, lower.old_value, lower.next_value, lower.terminal)
    state = torch.as_tensor(lower.state, dtype=torch.float32, device=model.device)
    action = torch.as_tensor(lower.action, dtype=torch.float32, device=model.device)
    old_logp = torch.as_tensor(lower.old_logp, dtype=torch.float32, device=model.device)
    advantage_t = torch.as_tensor(model._normalize(advantage), dtype=torch.float32, device=model.device)
    actor, optimizer = model.lower_actor, model.lower_actor_optimizer
    with torch.no_grad():
        before = actor.distribution(state)
        before_mean = before.mean.clone()
    np.random.seed(spec.deployment.shuffle_seed(root, period, 1, phase="train", level="lower"))
    indices, count = np.arange(lower.size), 0
    with guarded.BacktrackingKLGuard(actor, optimizer, state, spec.KL_BUDGET) as guard:
        for _ in range(max(1, cfg.epochs)):
            np.random.shuffle(indices)
            for start in range(0, lower.size, min(cfg.minibatch_size, lower.size)):
                idx = torch.as_tensor(indices[start:start + cfg.minibatch_size], dtype=torch.long, device=model.device)
                logp, entropy = actor.log_prob_entropy(state[idx], action[idx])
                ratio = torch.exp((logp - old_logp[idx]).clamp(-20., 20.))
                clipped = torch.clamp(ratio, 1. - cfg.clip_ratio, 1. + cfg.clip_ratio)
                policy = -torch.minimum(ratio * advantage_t[idx], clipped * advantage_t[idx]).mean()
                loss = policy - float(cfg.entropy_coef) * entropy.mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                optimizer.step()
                count += 1
    with torch.no_grad():
        after = actor.distribution(state)
        old = torch.distributions.Normal(before.mean.double(), before.stddev.double())
        current = torch.distributions.Normal(after.mean.double(), after.stddev.double())
        kl = torch.distributions.kl_divergence(old, current).sum(-1).clamp_min(0.)
        rms = float((after.mean - before_mean).square().mean().sqrt())
    return {"actor_optimizer_steps": count, "actor_forward_minibatches": count,
        "kl_mean": float(kl.mean()), "kl_max": float(kl.max()), "mean_action_change_rms": rms,
        "advantage_std": float(np.std(advantage)), "guard": guard.record()}, target


def init_worker(config, args):
    continuing.init_worker(config, args)
    native.init_worker(config, args)


def train(root, *, preflight, output):
    source = json.loads(spec.source_result(root, preflight=preflight).read_text())
    _, _, failures = values.qualify(source, preflight=preflight)
    if source["root"] != root or (failures and not preflight):
        raise ValueError("Stage65 full candidate critic prerequisite failed")
    if not preflight:
        qualification = json.loads(spec.source_qualification(preflight=False).read_text())
        if qualification["protocol"] != spec.source.EXPERIMENT_PROTOCOL or qualification["candidate_fit_gate"] != "passed":
            raise ValueError("Stage65 requires completed full Stage64 qualification")
    original = json.loads(spec.source.training_result(root, preflight=preflight).read_text())
    upper = json.loads(spec.upper_result(root, preflight=preflight).read_text())
    if (upper["protocol"], upper["root"], upper["preflight"], upper["status"], upper["contract"]) != (
            spec.source.source.EXPERIMENT_PROTOCOL, root, preflight, "complete", spec.source.source.contract()):
        raise ValueError("Stage65 common upper source changed")
    clones, predictor, initialization = native.load_source(root, preflight=preflight)
    args, opt, roles = spec.arguments(root, preflight=preflight), spec.options(preflight=preflight), spec.seed_roles(root, preflight=preflight)
    for origin in (spec.source, spec.deployment, spec.deployment.SOURCE_SPEC, spec.native_source):
        used = {s for seeds in origin.seed_roles(root, preflight=preflight).values() for s in seeds}
        if used.intersection(roles["evaluation"]):
            raise ValueError("Stage65 native evaluation overlaps previous fitting/training/evaluation")
    archive = spec.source.training_result(root, preflight=preflight).parent
    archive = archive.with_name(archive.name + "_raw")
    raw, started, budget = raw_directory(output), time.monotonic(), spec.budget(preflight=preflight)
    cost = dict.fromkeys(budget["archive"], 0)
    cost.update(source_clone_loads=len(clones), forecaster_loads=1)
    groups, evaluations = {}, {}
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"), initializer=init_worker,
            initargs=(clones[str(spec.PERIODS[0])].config, args)) as pool:
        for period in spec.PERIODS:
            p, clone = str(period), clones[str(period)]
            groups[p] = {}
            evaluations[p] = {"clone": evaluation.evaluate(pool, clone, predictor, args, arm="clone", period=period,
                seeds=roles["evaluation"], directory=raw / p / "clone")}
            for arm in spec.TRAIN_POLICIES:
                common = copy.deepcopy(clone)
                upper_cell = upper["comparisons"][p][arm]
                if upper_cell["upper_networks_and_Adam_pair"] != "passed":
                    raise ValueError("Stage65 common upper pair failed in source")
                saved = torch.load(upper_cell["checkpoints"]["option_credit"], map_location="cpu", weights_only=False)
                if (saved["protocol"], saved["root"], saved["period"], saved["arm"], saved["treatment"]) != (
                        spec.source.source.EXPERIMENT_PROTOCOL, root, period, arm, "option_credit"):
                    raise ValueError("Stage65 upper checkpoint identity changed")
                cost["upper_checkpoint_loads"] += 1
                for kind in ("actor", "value", "actor_optimizer", "value_optimizer"):
                    getattr(common, "upper_" + kind).load_state_dict(saved["state_dict"]["upper_" + kind])
                if arm == "zero_train":
                    for kind in ("actor", "actor_optimizer"):
                        torch.testing.assert_close(getattr(common, "upper_" + kind).state_dict(),
                            getattr(clone, "upper_" + kind).state_dict(), atol=0, rtol=0)
                fits, weights = {}, {}
                for t in spec.TREATMENTS:
                    payload = torch.load(source["groups"][p][arm]["treatments"][t]["checkpoint"], map_location="cpu", weights_only=False)
                    fits[t] = restore_fit(common, payload, root=root, period=period, arm=arm, treatment=t)
                    cost["critic_checkpoint_loads"] += 1
                    cost["critic_resume_checks"] += 1
                    weights[t] = joint.inference_weights(clone)
                    weights[t]["lower_value"] = fits[t].public_state()
                first = original["training"][p][arm]["history"][0]
                if [r["seed"] for r in first["rows"]] != roles["first_training"]:
                    raise ValueError("Stage65 first actor batch changed")
                path = archive / p / arm / "train" / "1" / "training"
                pairs = list(pool.map(continuing.worker_pair, [(weights["gae_raw"], weights[spec.CANDIDATE],
                    str(path / f"episode_{r['seed']}.npz"), r["seed"], period) for r in first["rows"]]))
                for (_, _, row), old in zip(pairs, first["rows"]):
                    if row["episode_return"] != old["episode_return"] or row["action_check"] != "passed":
                        raise ValueError("Stage65 archived action or reward changed")
                    for key, src in (("archive_episodes", None), ("reconstructed_lower_calls", "lower_calls"),
                            ("reconstructed_upper_calls", "upper_calls"), ("extra_critic_scalar_calls", "episode_value_calls"),
                            ("archive_network_checks", "network_checks")):
                        cost[key] += 1 if src is None else row[src]
                batch = concat_hierarchical_batches([b for b, _, _ in pairs])
                lower = {"gae_raw": continuing.episode_batch(batch, batch.lower.old_value, args.horizon).lower,
                    spec.CANDIDATE: continuing.episode_batch(batch, np.concatenate([v for _, v, _ in pairs]), args.horizon).lower}
                mc = values.monte_carlo_returns(lower["gae_raw"], clone.config.gamma)
                cost["probe_mc_calls"] += 1
                rows = {}
                evaluations[p][arm] = {"frozen_lower": evaluations[p]["clone"] if arm == "zero_train" else
                    evaluation.evaluate(pool, common, predictor, args, arm=arm, period=period, seeds=roles["evaluation"],
                        directory=raw / p / arm / "frozen_lower")}
                for t, fit in fits.items():
                    before = credit.value_metrics(lower[t].old_value, mc)
                    if before != source["groups"][p][arm]["treatments"][t]["episode_mc"]:
                        raise ValueError("Stage65 restored critic probe differs from Stage64")
                    cost["source_probe_checks"] += 1
                    update, gae_target = actor_update(fit, lower[t], root=root, period=period)
                    cost["actor_updates"] += 1
                    cost["actor_gae_calls"] += 1
                    value = fit.update(lower[t], mc if t == spec.CANDIDATE else gae_target, root=root, period=period, iteration=1, phase="train")
                    cost["critic_continuation_updates"] += 1
                    cost["MC_continuation_updates"] += int(t == spec.CANDIDATE)
                    public = public_model(fit)
                    with torch.no_grad():
                        prediction = public.lower_value(torch.as_tensor(lower[t].value_state, dtype=torch.float32, device=public.device)).cpu().numpy()
                    cost["post_update_public_value_passes"] += 1
                    for kind in ("actor", "value", "actor_optimizer", "value_optimizer"):
                        torch.testing.assert_close(getattr(fit.model, "upper_" + kind).state_dict(),
                            getattr(common, "upper_" + kind).state_dict(), atol=0, rtol=0)
                        cost["upper_frozen_state_checks"] += 1
                    checkpoint = raw / p / arm / t / "policy.pt"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "period": period, "arm": arm,
                        "treatment": t, "location": fit.location, "scale": fit.scale,
                        "training_state_dict": fit.model.state_dict(), "public_inference_weights": joint.inference_weights(public)}, checkpoint)
                    cost["candidate_checkpoint_writes"] += 1
                    rows[t] = {"actor": update, "critic": value, "frame": {"location": fit.location, "scale": fit.scale},
                        "source_probe": before, "post_update_probe": credit.value_metrics(prediction, mc), "checkpoint": str(checkpoint)}
                    evaluations[p][arm][t] = evaluation.evaluate(pool, public, predictor, args, arm=arm, period=period,
                        seeds=roles["evaluation"], directory=raw / p / arm / t / "evaluation")
                groups[p][arm] = {"source_actions_and_probe": "passed", "upper_pair_and_frozen": "passed", "updates": rows}
                print(f"normalized first update {root}/period{period}/{arm}: units, source probe and shared upper exact", flush=True)
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "options": opt, "seed_roles": roles, "budget": budget, "archive_cost": cost,
        "source_initialization": initialization, "source_critic_budget": source["budget"], "source_critic_cost": source["cost"],
        "groups": groups, "evaluation_rows": evaluations, "new_forecaster_fits": 0, "new_training_native_steps": 0,
        "wall_seconds": time.monotonic() - started}
    _, counts, _, _ = qualify(result, preflight=preflight)
    result["native_evaluation_counts"] = counts
    result["native_trace_audits"] = budget["native_trace_audits"]
    write_json(output, result)
    return result


def native_rows(rows, *, root, args, arm, period, seeds):
    if [r["seed"] for r in rows] != seeds:
        raise ValueError("Stage65 paired evaluation roster changed")
    counts = dict.fromkeys(learned.COUNT_KEYS, 0)
    for r in rows:
        calls = args.horizon // period
        wanted = {"episode_length": args.horizon, "upper_inference_calls": calls, "lower_inference_calls": args.horizon,
            "gate_inference_calls": 0, "candidate_preview_calls": 0, "plan_ols_fits": calls - 1, "audit_ols_fits": calls - 1,
            "plan_ridge_predictions": calls - 1, "audit_ridge_predictions": calls - 1,
            "reference_evaluations": args.horizon, "actor_context_evaluations": args.horizon, "upper_plan_decodes": calls,
            "bernstein_basis_evaluations": period + 1, "audit_bernstein_basis_evaluations": period + 1}
        if (r["arm"] != arm or r["period"] != period or r["phase"] != "eval" or r["deployment_mode"] != spec.MODE
                or r["policy"] != spec.deployment.execution(arm) or r["policy_seed"] != spec.deployment.policy_seed(root, r["seed"])
                or r["method"] != f"fixed{period}" or r["rollout_network_check"] != "passed" or r["lower_actor_type"] != "GaussianActor"
                or r["decision_steps"] != list(range(0, args.horizon, period)) or any(r[k] != v for k, v in wanted.items())):
            raise ValueError("Stage65 native execution or cost changed")
        if ((arm != "joint_ppo" and (r["executed_action_rms"] or r["executed_plan_delta_squared_sum"]))
                or (arm == "joint_ppo" and r["executed_action_rms"] != r["proposed_action_rms"])):
            raise ValueError("Stage65 native upper action execution changed")
        counts["primitive_steps"] += r["episode_length"]
        for key in learned.COUNT_KEYS[1:]:
            counts[key] += r[key]
    return {k: float(np.mean([r[k] for r in rows])) for k in spec.native_source.METRICS}, counts


def qualify(c, *, preflight):
    root, budget = c["root"], spec.budget(preflight=preflight)
    if (root not in spec.roots(preflight=preflight) or c["status"] != "complete" or c["protocol"] != spec.EXPERIMENT_PROTOCOL
            or c["contract"] != spec.contract() or c["preflight"] != preflight or c["options"] != spec.options(preflight=preflight)
            or c["seed_roles"] != spec.seed_roles(root, preflight=preflight) or c["budget"] != budget
            or c["archive_cost"] != budget["archive"] or c["new_forecaster_fits"] or c["new_training_native_steps"]
            or set(c["groups"]) != {str(p) for p in spec.PERIODS} or set(c["evaluation_rows"]) != set(c["groups"])):
        raise ValueError("Stage65 frozen protocol or accounting changed")
    args, opt, cfg = spec.arguments(root, preflight=preflight), c["options"], c["source_initialization"]["config"]
    steps = max(1, cfg["epochs"]) * math.ceil(args.horizon * opt["rollouts_per_iteration"] / cfg["minibatch_size"])
    counts, frozen, means, compact = dict.fromkeys(learned.COUNT_KEYS, 0), [], {}, copy.deepcopy(c["groups"])
    totals = dict.fromkeys(("actor_optimizer_steps", "actor_forward_minibatches", "value_optimizer_steps",
        "value_forward_minibatches", "MC_supervised_optimizer_steps", "guard_distribution_passes", "candidate_evaluations",
        "parameter_interpolation_trials", "state_snapshot_calls", "rollback_state_checks", "retained_actor_steps", "rejected_actor_steps"), 0)
    for period in spec.PERIODS:
        p, groups, stage = str(period), c["groups"][str(period)], c["evaluation_rows"][str(period)]
        if set(groups) != set(spec.TRAIN_POLICIES) or set(stage) != {"clone", *spec.TRAIN_POLICIES}:
            raise ValueError("Stage65 arm roster changed")
        clone, n = native_rows(stage["clone"], root=root, args=args, arm="clone", period=period, seeds=c["seed_roles"]["evaluation"])
        for k in counts:
            counts[k] += n[k]
        means[p] = {}
        for arm, cell in groups.items():
            if (cell["source_actions_and_probe"] != "passed" or cell["upper_pair_and_frozen"] != "passed"
                    or set(cell["updates"]) != set(spec.TREATMENTS) or set(stage[arm]) != {"frozen_lower", *spec.TREATMENTS}):
                raise ValueError("Stage65 pairing or treatment roster changed")
            if arm == "zero_train":
                if stage[arm]["frozen_lower"] != stage["clone"]:
                    raise ValueError("Stage65 zero frozen lower must reuse clone")
                baseline = clone
            else:
                baseline, n = native_rows(stage[arm]["frozen_lower"], root=root, args=args, arm=arm, period=period,
                    seeds=c["seed_roles"]["evaluation"])
                for k in counts:
                    counts[k] += n[k]
            means[p][arm] = {"clone": clone, "frozen_lower": baseline}
            for t, row in cell["updates"].items():
                actor, value, guard = row["actor"], row["critic"], row["actor"]["guard"]
                if (any(d[k] != steps for d, keys in ((actor, ("actor_optimizer_steps", "actor_forward_minibatches")),
                        (value, ("value_optimizer_steps", "value_forward_minibatches"))) for k in keys)
                        or guard["attempted_actor_steps"] != steps or len(guard["steps"]) != steps
                        or actor["kl_mean"] != guard["steps"][-1]["deployed_kl"] or actor["kl_mean"] > spec.KL_BUDGET):
                    raise ValueError("Stage65 matched optimizer budget or KL changed")
                if guard["retained_actor_steps"] == 0 or actor["mean_action_change_rms"] == 0:
                    frozen.append({"root": root, "period": period, "arm": arm, "treatment": t})
                for d in (actor, value):
                    for k in ("actor_optimizer_steps", "actor_forward_minibatches", "value_optimizer_steps", "value_forward_minibatches"):
                        totals[k] += d.get(k, 0)
                totals["MC_supervised_optimizer_steps"] += value["value_optimizer_steps"] if t == spec.CANDIDATE else 0
                for k in totals:
                    if k in guard:
                        totals[k] += guard[k]
                compact[p][arm]["updates"][t].pop("checkpoint")
                compact[p][arm]["updates"][t]["actor"]["guard"].pop("steps")
                m, n = native_rows(stage[arm][t], root=root, args=args, arm=arm, period=period, seeds=c["seed_roles"]["evaluation"])
                means[p][arm][t] = m
                for k in counts:
                    counts[k] += n[k]
    if counts != budget["native_evaluation"] or ("native_evaluation_counts" in c and c["native_evaluation_counts"] != counts):
        raise ValueError("Stage65 total native cost changed")
    if "native_trace_audits" in c and c["native_trace_audits"] != budget["native_trace_audits"]:
        raise ValueError("Stage65 trace audit count changed")
    return {"root": root, "groups": compact, "means": means, "endpoints": spec.contrasts(means)}, counts, totals, frozen


def aggregate(cells, *, preflight):
    by_root = {c["root"]: c for c in cells}
    if len(by_root) != len(cells) or set(by_root) != set(spec.roots(preflight=preflight)):
        raise ValueError("Stage65 complete frozen root roster required")
    rows, frozen, counts, totals = [], [], {}, {}
    for root in spec.roots(preflight=preflight):
        row, n, steps, failures = qualify(by_root[root], preflight=preflight)
        rows.append(row)
        frozen.extend(failures)
        for src, dst in ((n, counts), (steps, totals)):
            for k, v in src.items():
                dst[k] = dst.get(k, 0) + v
    summary = {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "root_rows": rows, "native_evaluation_counts": counts, "executed_optimizer_and_guard_cost": totals,
        "archive_cost": {k: sum(c["archive_cost"][k] for c in cells) for k in spec.budget(preflight=preflight)["archive"]},
        "native_trace_audits": sum(c["native_trace_audits"] for c in cells), "frozen_actors": frozen,
        "mechanical_gate": "failed" if frozen else "passed", "new_forecaster_fits": 0, "new_training_native_steps": 0}
    if not preflight:
        x = np.asarray([[r["endpoints"][key] for key in spec.ENDPOINTS] for r in rows])
        indices = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED)).integers(0, len(x), (spec.BOOTSTRAP_DRAWS, len(x)))
        tail = .05 / (2 * len(spec.ENDPOINTS))
        bounds = np.quantile(x[indices].mean(axis=1), [tail, 1 - tail], axis=0)
        endpoints = {key: {"mean": float(x[:, i].mean()), "ci": bounds[:, i].tolist(),
            "effect": "positive" if bounds[0, i] > 0 else "negative" if bounds[1, i] < 0 else "inconclusive"}
            for i, key in enumerate(spec.ENDPOINTS)}
        def positive_against(b):
            return all(endpoints[f"period{p}:{a}:mc_normalized_minus_{b}"]["effect"] == "positive"
                for p in spec.PERIODS for a in spec.TRAIN_POLICIES)
        summary.update(primary_endpoints=endpoints, repair_gain_gate="passed" if not frozen and positive_against("gae_raw") else "failed",
            training_gain_gate="passed" if not frozen and positive_against("frozen_lower") and positive_against("clone") else "failed")
    return summary
