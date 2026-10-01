"""Measure native reward response to historical, fixed-radius directions."""

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
from . import pointmaze_continuing_credit as continuing
from . import pointmaze_credit_reliability as reliability
from . import pointmaze_independent_credit as independent
from . import pointmaze_horizon_value as values
from . import pointmaze_normalized_update as normalized
from . import pointmaze_update_diagnostics as diagnostics
from . import pointmaze_matched_upper as native
from . import pointmaze_joint_renewal as joint
from . import pointmaze_upper_execution as execution
from . import pointmaze_objective_audit as previous
from .pointmaze_root_response import write_json
from scripts import pointmaze_native_direction_stage73_spec as spec


def parameter_tangents(actor, vector):
    result, offset = {}, 0
    for name, p in actor.named_parameters():
        result[name] = torch.as_tensor(vector[offset:offset + p.numel()], dtype=p.dtype, device=p.device).reshape(p.shape)
        offset += p.numel()
    if offset != len(vector):
        raise ValueError("Stage73 actor direction dimension changed")
    return result


def shifted_actor(actor, tangents, step):
    candidate = copy.deepcopy(actor)
    with torch.no_grad():
        for name, p in candidate.named_parameters():p.add_(tangents[name], alpha=step)
    return candidate


def matched_perturbations(actor, states, gradient, *, delta, chunk_size):
    norm = float(np.linalg.norm(gradient))
    if not np.isfinite(norm) or norm == 0.:
        raise ValueError("Stage73 historical direction is undefined")
    tangent = parameter_tangents(actor, -gradient / norm)
    parameters = dict(actor.named_parameters())
    total, old_means, chunks = 0., [], 0
    for start in range(0, len(states), chunk_size):
        state = torch.as_tensor(states[start:start + chunk_size], dtype=torch.float32)
        def distribution(p):
            net = {k[4:]: v for k, v in p.items() if k.startswith("net.")}
            return torch.func.functional_call(actor.net, net, (state,)), p["log_std"].exp().clamp(1e-4, 3.)
        with torch.no_grad():
            (mean, std), (dmean, dstd) = torch.func.jvp(distribution, (parameters,), (tangent,))
        total += float(((dmean.double() / std.double()) ** 2).sum())
        total += 2 * len(state) * float(((dstd.double() / std.double()) ** 2).sum())
        old_means.append(mean.double())
        chunks += 1
    fisher = total / len(states)
    if not np.isfinite(fisher) or fisher <= 0.:
        raise ValueError("Stage73 historical Fisher direction is undefined")
    step = math.sqrt(2 * delta / fisher)
    actors = {sign: shifted_actor(actor, tangent, sign_value * step) for sign, sign_value in (("plus", 1.), ("minus", -1.))}
    kl = dict.fromkeys(actors, 0.)
    for i, start in enumerate(range(0, len(states), chunk_size)):
        state = torch.as_tensor(states[start:start + chunk_size], dtype=torch.float32)
        old = torch.distributions.Normal(old_means[i], std.double())
        with torch.no_grad():
            for sign, candidate in actors.items():
                dist = candidate.distribution(state)
                new = torch.distributions.Normal(dist.mean.double(), dist.stddev.double())
                kl[sign] += float(torch.distributions.kl_divergence(old, new).sum()) / len(states)
    average = (kl["plus"] + kl["minus"]) / 2
    if not .5 * delta <= average <= 2 * delta:
        raise ValueError("Stage73 fixed Fisher radius is not local on historical states")
    return actors, {"historical_loss_gradient_norm": norm, "step": step, "fisher_quadratic": fisher,
        "nominal_fisher_kl": delta, "exact_kl": kl, "radius_check": "passed"}, {
        "fisher_jvp_batches": chunks, "exact_kl_forward_batches": 2 * chunks}


_WORKER = None


def init_worker(config, args):
    global _WORKER
    diagnostics.init_worker(config, args)
    _WORKER = FrequencySeparatedActorCriticPPO(config), args


def worker_native(job):
    weights, seed, arm, period, predictor = job
    model, args = _WORKER
    model.load_state_dict(weights)
    policy_seed = native.spec.policy_seed(args.optimizer_seed, seed)
    torch.manual_seed(policy_seed)
    plan = execution.ExecutedPlan(predictor, period, args.maximum_subgoal_delta, native.spec.execution(arm))
    kwargs = native.spec.rollout_arguments(args.optimizer_seed, seed, phase="train", mode="training")
    kwargs["sample"] = False
    batch, row, raw = joint.rollout(model, args, f"fixed{period}", seed=seed, capture=False,
        lower_credit="task_option", upper_plan_decoder=plan.decode, lower_reference_builder=plan,
        lower_actor_context_builder=plan.actor_context, lower_value_context_builder=plan.value_context, **kwargs)
    if batch is not None or raw is not None:
        raise ValueError("Stage73 native probe unexpectedly materialized training data or raw traces")
    torch.testing.assert_close(joint.inference_weights(model), weights, atol=0, rtol=0)
    return {"seed": seed, "episode_return": row["episode_return"], "episode_length": row["episode_length"],
        "upper_calls": row["upper_inference_calls"], "lower_calls": row["lower_inference_calls"],
        "decision_steps": row["decision_steps"], "policy_seed": policy_seed, "lower_seed": row["lower_seed"],
        "upper_proposed_actions": np.asarray(plan.proposed_actions).tolist(), "network_check": "passed",
        "plan_ols_fits": plan.ols_fits, "plan_ridge_predictions": plan.ridge_predictions,
        "reference_evaluations": plan.calls, "actor_context_evaluations": plan.context_calls}


def paired_effects(evaluation, seeds):
    if set(evaluation) != set(spec.VARIANTS) or any([r["seed"] for r in rows] != seeds for rows in evaluation.values()):
        raise ValueError("Stage73 native paired roster changed")
    for i in range(len(seeds)):
        base = evaluation["base"][i]
        for rows in evaluation.values():
            row = rows[i]
            for key in ("policy_seed", "lower_seed", "decision_steps", "upper_proposed_actions"):
                if row[key] != base[key]:raise ValueError("Stage73 native common-noise pairing changed")
    reward = {k: np.asarray([r["episode_return"] for r in rows]) for k, rows in evaluation.items()}
    result = {}
    for d in spec.DIRECTIONS:
        plus, minus, base = reward[d + "_plus"], reward[d + "_minus"], reward["base"]
        result[d] = {"plus_minus": float((plus - minus).mean()), "plus_base": float((plus - base).mean()),
            "minus_base": float((minus - base).mean())}
    return result


def run(root, *, preflight, output):
    source = json.loads(spec.source_result(root, preflight=preflight).read_text())
    previous.qualify(source, preflight=preflight)
    if source["root"] != root:raise ValueError("Stage73 prerequisite root changed")
    legacy = spec.source.legacy
    historical_file = legacy.values_source.training_result(root, preflight=preflight)
    historical = json.loads(historical_file.read_text())
    archive = historical_file.parent.with_name(historical_file.parent.name + "_raw")
    controls = json.loads(legacy.values_source.source_result(root, preflight=preflight).read_text())
    factored = json.loads(legacy.source.source_result(root, preflight=preflight).read_text())
    clones, predictor, initialization = native.load_source(root, preflight=preflight)
    opt, args, roles = spec.options(preflight=preflight), spec.arguments(root, preflight=preflight), spec.seed_roles(root, preflight=preflight)
    cost, groups, started = dict.fromkeys(spec.budget(preflight=preflight), 0), {}, time.monotonic()
    cost.update(source_clone_loads=len(clones), forecaster_loads=1)
    planning = dict.fromkeys(("plan_ols_fits", "plan_ridge_predictions", "reference_evaluations", "actor_context_evaluations"), 0)
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"), initializer=init_worker,
            initargs=(clones[str(spec.PERIODS[0])].config, args)) as pool:
        for period in spec.PERIODS:
            p, clone = str(period), clones[str(period)]
            snapshot = copy.deepcopy(clone.state_dict())
            groups[p] = {}
            for arm in spec.TRAIN_POLICIES:
                saved = torch.load(controls["groups"][p][arm]["treatments"]["mc_normalized"]["checkpoint"], map_location="cpu", weights_only=False)
                fits = {"control": normalized.restore_fit(clone, saved, root=root, period=period, arm=arm, treatment="mc_normalized")}
                saved = torch.load(factored["groups"][p][arm]["candidate_checkpoint"], map_location="cpu", weights_only=False)
                if (saved["protocol"], saved["root"], saved["period"], saved["arm"], saved["horizon"]) != (
                        legacy.values_source.EXPERIMENT_PROTOCOL, root, period, arm, args.horizon):
                    raise ValueError("Stage73 fixed factored critic identity changed")
                fits["factored"] = values.FactoredValueFit.restore(copy.deepcopy(clone), saved)
                fit_snapshots = {t: copy.deepcopy(f.model.state_dict()) for t, f in fits.items()}
                cost["critic_checkpoint_loads"] += len(fits)
                items = historical["calibration"][p][arm]["history"]
                if len(items) != opt["calibration_batches"] or [r["seed"] for item in items for r in item["rows"]] != roles["calibration"]:
                    raise ValueError("Stage73 historical direction fitting roster changed")
                states, gradients, frame = [], {}, None
                for item in items:
                    seeds = [r["seed"] for r in item["rows"]]
                    path = archive / p / arm / "warmup" / str(item["iteration"]) / "training"
                    pairs = list(pool.map(diagnostics.worker_reconstruct, [(joint.inference_weights(clone),
                        str(path / f"episode_{seed}.npz"), seed, period) for seed in seeds]))
                    for (_, row), old in zip(pairs, item["rows"]):
                        if row["episode_return"] != old["episode_return"] or row["action_check"] != "passed":
                            raise ValueError("Stage73 historical actions or rewards changed")
                        cost["calibration_archive_episodes"] += 1
                        cost["archive_network_checks"] += 1
                        cost["reconstructed_lower_calls"] += row["lower_calls"]
                        cost["reconstructed_upper_calls"] += row["upper_calls"]
                    batch = concat_hierarchical_batches([b for b, _ in pairs])
                    lower = continuing.episode_batch(batch, batch.lower.old_value, args.horizon).lower
                    if frame is None:
                        frame = {"location": float(lower.reward.astype(np.float64).mean()), "sample_count": lower.size,
                            "data_role": "first_historical_calibration_only"}
                        cost["historical_reward_rate_fits"] += 1
                    remaining = args.horizon - np.arange(lower.size) % args.horizon
                    signals = {"native_mc": independent.exact_returns(lower, 1.) - remaining * frame["location"]}
                    cost["mc_calls"] += 1
                    for t, fit in fits.items():
                        pred = values.control_predictions(fit, lower, clone) if t == "control" else fit.predictions(lower)
                        b = replace(lower, old_value=pred)
                        signals["gae_" + t], _ = fit.model._gae(b.reward, b.done, b.duration, b.old_value, b.next_value, b.terminal)
                        cost["calibration_value_rows"] += lower.size
                        cost["gae_calls"] += 1
                    g, arrays, _, score_cost = reliability.episode_scores(clone.lower_actor, lower, signals,
                        horizon=args.horizon, clip_ratio=clone.config.clip_ratio)
                    fold = reliability.fold_gradients(g, arrays, range(len(seeds)))
                    for d in spec.DIRECTIONS:
                        direction = g[d].mean(0) if d == "native_mc" else fold[d] + clone.config.entropy_coef * fold["entropy"]
                        gradients[d] = gradients.get(d, np.zeros_like(direction)) + direction / len(items)
                    for key in ("actor_score_forward_batches", "actor_score_backward_batches"):cost[key] += score_cost[key]
                    states.append(lower.state)
                states = np.concatenate(states)
                geometry, weights = {}, {"base": joint.inference_weights(clone)}
                for d, gradient in gradients.items():
                    actors, geometry[d], c = matched_perturbations(clone.lower_actor, states, gradient,
                        delta=spec.FISHER_RADIUS, chunk_size=spec.CHUNK_SIZE)
                    for key, v in c.items():cost[key] += v
                    cost["historical_direction_fits"] += 1
                    cost["calibration_radius_checks"] += 1
                    for sign, actor in actors.items():
                        weights[d + "_" + sign] = {**joint.inference_weights(clone), "lower_actor": copy.deepcopy(actor.state_dict())}
                        cost["actor_parameter_perturbations"] += 1
                del states
                print(f"native direction {root}/{p}/{arm}: three historical directions and fixed radii frozen", flush=True)
                evaluation = {}
                for variant in spec.VARIANTS:
                    rows = list(pool.map(worker_native, [(weights[variant], seed, arm, period, predictor) for seed in roles["native_evaluation"]]))
                    for row in rows:
                        if (row["episode_length"] != args.horizon or row["lower_calls"] != args.horizon or
                                row["upper_calls"] != args.horizon // period or row["network_check"] != "passed"):
                            raise ValueError("Stage73 native rollout cost or weights changed")
                        cost["native_episodes"] += 1
                        cost["native_steps"] += row["episode_length"]
                        cost["native_lower_calls"] += row["lower_calls"]
                        cost["native_upper_calls"] += row["upper_calls"]
                        cost["native_network_checks"] += 1
                        for key in planning:planning[key] += row[key]
                    evaluation[variant] = rows
                effects = paired_effects(evaluation, roles["native_evaluation"])
                cost["native_pair_checks"] += len(roles["native_evaluation"])
                for t, fit in fits.items():
                    independent.assert_frozen(fit.model, fit_snapshots[t])
                    cost["frozen_model_checks"] += 1
                groups[p][arm] = {"geometry": geometry, "native_baseline_frame": frame, "effects": effects,
                    "evaluation": evaluation, "model_and_Adam_unchanged": "passed", "pairing": "passed"}
                print(f"native direction {root}/{p}/{arm}: all seven paired native probes completed", flush=True)
            independent.assert_frozen(clone, snapshot)
            cost["frozen_model_checks"] += 1
    cell = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "seed_roles": roles, "cost": cost, "native_planning_cost": planning,
        "groups": groups, "source_initialization": initialization, "optimizer_steps": 0, "critic_fits": 0,
        "forecaster_fits": 0, "checkpoint_writes": 0, "native_trace_writes": 0, "wall_seconds": time.monotonic() - started}
    qualify(cell, preflight=preflight)
    write_json(output, cell)
    write_json(output.parent / "completion" / "ready.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return cell


def qualify(cell, *, preflight):
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or cell["root"] not in spec.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["cost"] != spec.budget(preflight=preflight) or cell["seed_roles"] != spec.seed_roles(cell["root"], preflight=preflight)
            or any(cell[k] for k in ("optimizer_steps", "critic_fits", "forecaster_fits", "checkpoint_writes", "native_trace_writes"))
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}):
        raise ValueError("Stage73 frozen native direction audit or budget changed")
    for arms in cell["groups"].values():
        if set(arms) != set(spec.TRAIN_POLICIES):raise ValueError("Stage73 execution roster changed")
        for g in arms.values():
            if g["model_and_Adam_unchanged"] != "passed" or g["pairing"] != "passed" or set(g["geometry"]) != set(spec.DIRECTIONS):
                raise ValueError("Stage73 source or direction roster changed")
            if g["effects"] != paired_effects(g["evaluation"], cell["seed_roles"]["native_evaluation"]):
                raise ValueError("Stage73 native paired accounting changed")
    return cell


def aggregate(cells, *, preflight):
    if len({c["root"] for c in cells}) != len(cells) or {c["root"] for c in cells} != set(spec.roots(preflight=preflight)):
        raise ValueError("Stage73 requires every frozen root")
    rows = [qualify(c, preflight=preflight) for c in sorted(cells, key=lambda c: c["root"])]
    x = np.asarray([[r["groups"][p][a]["effects"][d][contrast] for p, a, d, contrast in
        (key.split("/") for key in spec.ENDPOINTS)] for r in rows])
    endpoints = {key: {"mean": float(x[:, i].mean())} for i, key in enumerate(spec.ENDPOINTS)}
    if not preflight:
        idx = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED)).integers(0, len(rows), (spec.BOOTSTRAP_DRAWS, len(rows)))
        tail = .05 / (2 * len(spec.ENDPOINTS))
        bounds = np.quantile(x[idx].mean(1), [tail, 1 - tail], axis=0)
        for i, key in enumerate(spec.ENDPOINTS):
            endpoints[key].update(ci=bounds[:, i].tolist(), effect="positive" if bounds[0, i] > 0 else "negative" if bounds[1, i] < 0 else "inconclusive")
    return {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "mechanical_gate": "passed", "root_rows": rows, "endpoints": endpoints,
        "cost": {k: sum(c["cost"][k] for c in rows) for k in spec.budget(preflight=preflight)},
        "native_planning_cost": {k: sum(c["native_planning_cost"][k] for c in rows) for k in rows[0]["native_planning_cost"]},
        "native_trial_prerequisite": "hold_Stage67_credit_gate_unchanged", "performance_claim": "none_forward_direction_response_only"}
