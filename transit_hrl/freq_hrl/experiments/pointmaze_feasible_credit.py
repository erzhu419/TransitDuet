"""Test fresh task-return actor directions without adopting a candidate policy."""

from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import replace
import json
import multiprocessing as mp
import time

import numpy as np
import torch

from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, concat_hierarchical_batches
from . import pointmaze_calibrated_residual as curves
from . import pointmaze_bounded_residual as bounded
from . import pointmaze_credit_reliability as reliability
from . import pointmaze_independent_credit as independent
from . import pointmaze_native_direction as direction
from . import pointmaze_joint_renewal as joint
from .pointmaze_root_response import write_json
from scripts import pointmaze_feasible_credit_stage79_spec as spec

_WORKER = None


def init_worker(config, args):
    global _WORKER
    torch.set_num_threads(1)
    _WORKER = FrequencySeparatedActorCriticPPO(replace(config, gamma=1.)), args


def task_returns(batch, period, horizon, native_return):
    lower, upper = batch.lower, replace(batch.upper, reward=batch.upper.reward + joint.spec.CALL_COST)
    if (lower.size != horizon or upper.size != horizon // period or
            not np.all(lower.duration == 1) or not np.all(upper.duration == period) or
            np.flatnonzero(lower.done).tolist() != [horizon - 1] or
            np.flatnonzero(upper.done).tolist() != [upper.size - 1]):
        raise ValueError("Stage79 requires true episode boundaries and fixed option durations")
    returns = {"lower": independent.exact_returns(lower, 1.), "upper": independent.exact_returns(upper, 1.)}
    np.testing.assert_allclose(returns["upper"], returns["lower"][::period], atol=.002, rtol=0)
    np.testing.assert_allclose([returns["lower"][0], returns["upper"][0]], native_return, atol=.002, rtol=0)
    return returns


def worker_native(job):
    weights, seed, variant, period, predictor, alpha, envelope, collect = job
    _, args = _WORKER
    native = curves.paths.native.native
    policy_seed = native.spec.policy_seed(args.optimizer_seed, seed)
    lower_seed = native.spec.rollout_arguments(args.optimizer_seed, seed, phase="train", mode="training")["lower_seed"]
    return native_episode(weights, seed=seed, variant=variant, period=period, predictor=predictor,
        alpha=alpha, envelope=envelope, collect=collect, policy_seed=policy_seed, lower_seed=lower_seed)


def native_episode(weights, *, seed, variant, period, predictor, alpha, envelope, collect, policy_seed, lower_seed):
    model, args = _WORKER
    model.load_state_dict(weights)
    native = curves.paths.native.native
    torch.manual_seed(policy_seed)
    plan = curves.CalibratedPlan(predictor, period, args.maximum_subgoal_delta, alpha, envelope)
    noise = []

    def decode(**kw):
        state = kw["history"].upper_state(kw["observation"], oracle_context=None)
        with torch.inference_mode():
            dist = model.upper_actor.distribution(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
            noise.append(((np.asarray(kw["action"]) - dist.mean[0].numpy()) / dist.stddev[0].numpy()).tolist())
        return plan.decode(**kw)

    kw = native.spec.rollout_arguments(args.optimizer_seed, seed, phase="train", mode="training")
    kw.update(sample=collect, lower_seed=lower_seed)
    batch, row, raw = joint.rollout(model, args, f"fixed{period}", seed=seed, capture=False,
        lower_credit="task_episode", upper_plan_decoder=decode, lower_reference_builder=plan,
        lower_actor_context_builder=plan.actor_context, lower_value_context_builder=plan.value_context, **kw)
    if raw is not None or (batch is not None) != collect or not (row["upper_sample"] and row["lower_sample"]):
        raise ValueError("Stage79 changed stochastic sampling or materialized a native trace")
    torch.testing.assert_close(joint.inference_weights(model), weights, atol=0, rtol=0)
    result = {"seed": seed, "variant": variant, "alpha": alpha, "policy_seed": policy_seed,
        **{k: row[k] for k in ("episode_return", "tracking_squared_error_integral", "episode_length", "decision_steps", "lower_seed")},
        "upper_calls": row["upper_inference_calls"], "lower_calls": row["lower_inference_calls"],
        "upper_standard_noise": noise, "network_check": "passed",
        **{k: getattr(plan, attr) for k, attr in (("plan_ols_fits", "ols_fits"), ("plan_ridge_predictions", "ridge_predictions"),
            ("reference_evaluations", "calls"), ("actor_context_evaluations", "context_calls"))},
        "above_label_speed_q99_rate": plan.speed_tail_rows / row["episode_length"]}
    return batch, result


def credit_directions(clone, batches, *, period, horizon, cost):
    rates = {name: float(np.mean([row["episode_return"] for _, row in pairs])) / horizon for name, pairs in batches.items()}
    returns = {}
    for name, pairs in batches.items():
        returns[name] = [task_returns(b, period, horizon, row["episode_return"]) for b, row in pairs]
        cost["objective_checks"] += len(pairs)
        cost["mc_calls"] += 2 * len(pairs)
    candidates, summaries = {}, {}
    for actor_name, length, duration in (("upper", horizon // period, period), ("lower", horizon, 1)):
        actor = getattr(clone, actor_name + "_actor")
        gradients, states, masks, score_costs = {}, [], None, {}
        for name, pairs in batches.items():
            level = getattr(concat_hierarchical_batches([b for b, _ in pairs]), actor_name)
            remaining = np.tile(horizon - np.arange(length) * duration, len(pairs))
            baseline_rate = rates["B" if name == "A" else "A"]
            signal = np.concatenate([r[actor_name] for r in returns[name]]) - remaining * baseline_rate
            g, _, masks, c = reliability.episode_scores(actor, level, {"mc": signal},
                horizon=length, clip_ratio=clone.config.clip_ratio)
            gradients[name], score_costs[name] = g["mc"], c
            states.append(level.state)
            for key in ("actor_score_forward_batches", "actor_score_backward_batches"):cost[key] += c[key]
        mean_gradient = np.concatenate(list(gradients.values())).mean(0)
        actors, geometry, c = direction.matched_perturbations(actor, np.concatenate(states), mean_gradient,
            delta=spec.FISHER_RADIUS, chunk_size=spec.CHUNK_SIZE)
        for key in c:cost[key] += c[key]
        cost["actor_parameter_perturbations"] += len(actors)
        for sign, changed in actors.items():
            weights = joint.inference_weights(clone)
            weights[actor_name + "_actor"] = changed.state_dict()
            candidates[actor_name + "_" + sign] = weights
        summaries[actor_name] = {"geometry": geometry, "score_costs": score_costs,
            "credit_batch_cosine": {part: reliability.scores.cosine(gradients["A"].mean(0)[mask], gradients["B"].mean(0)[mask])
                for part, mask in (("all", np.ones(len(masks), dtype=bool)), ("mean", ~masks), ("log_std", masks))},
            "conditional_batch_noise": {k: independent.gradient_noise(v) for k, v in gradients.items()},
            "pooled_loss_gradient_norm": float(np.linalg.norm(mean_gradient))}
    return candidates, {"baseline_reward_rates": rates, "actors": summaries}


def paired_effects(period, evaluation, seeds, *, protocol=spec):
    if set(evaluation) != set(protocol.VARIANTS) or any([r["seed"] for r in rows] != seeds for rows in evaluation.values()):
        raise ValueError("Stage79 native variant or evaluation roster changed")
    for rows in evaluation.values():
        for row, base in zip(rows, evaluation["base"]):
            if any(row[k] != base[k] for k in ("policy_seed", "lower_seed", "decision_steps")):
                raise ValueError("Stage79 native seeds or decision schedule changed")
            np.testing.assert_allclose(row["upper_standard_noise"], base["upper_standard_noise"], atol=2e-6, rtol=0)
    return {f"{period}/{a}_minus_{b}": float(np.mean([x["episode_return"] - y["episode_return"]
        for x, y in zip(evaluation[a], evaluation[b])])) for a, b in protocol.CONTRAST_PAIRS}


def run(root, *, preflight, output):
    source = json.loads(spec.source_result(root).read_text())
    if (source["status"], source["protocol"], source["root"], source["preflight"], source["contract"]) != (
            "complete", spec.source.EXPERIMENT_PROTOCOL, root, False, spec.source.contract()):
        raise ValueError("Stage79 needs the completed full Stage78 source")
    clones, predictor, initialization = curves.support.native.load_source(root, preflight=False)
    args, roles = spec.arguments(root, preflight=preflight), spec.seed_roles(root, preflight=preflight)
    cost, groups, started = dict.fromkeys(spec.budget(preflight=preflight), 0), {}, time.monotonic()
    cost.update(source_clone_loads=len(clones), forecaster_loads=1, decoder_loads=len(clones))
    planning = dict.fromkeys(curves.paths.PLANNING_KEYS, 0)
    with ProcessPoolExecutor(max_workers=spec.options(preflight=preflight)["workers"], mp_context=mp.get_context("spawn"),
            initializer=init_worker, initargs=(clones[str(spec.PERIODS[0])].config, args)) as pool:
        for period in spec.PERIODS:
            clone = clones[str(period)]
            snapshot, weights = copy.deepcopy(clone.state_dict()), joint.inference_weights(clone)
            calibration = source["groups"][str(period)]["calibration"]
            bounded.check_calibration(calibration)
            alpha, envelope = calibration["alpha"], calibration["envelope"]

            def episodes(w, seeds, variant, collect=False):
                pairs = list(pool.map(worker_native, [(w, s, variant, period, predictor,
                    0. if variant == "zero" else alpha, envelope, collect) for s in seeds]))
                for _, row in pairs:
                    curves.paths.check_row(row, period, args.horizon)
                    cost["native_episodes"] += 1
                    cost["native_steps"] += row["episode_length"]
                    cost["native_lower_calls"] += row["lower_calls"]
                    cost["native_upper_calls"] += row["upper_calls"]
                    cost["pairing_upper_forward_calls"] += row["upper_calls"]
                    cost["native_network_checks"] += 1
                    cost["credit_episodes" if collect else "evaluation_episodes"] += 1
                    for key in planning:planning[key] += row[key]
                return pairs

            batches = {name: episodes(weights, roles["credit_" + name], "base", True) for name in ("A", "B")}
            candidates, credit = credit_directions(clone, batches, period=period, horizon=args.horizon, cost=cost)
            del batches
            candidates.update(base=weights, zero=weights)
            evaluation = {variant: [r for _, r in episodes(candidates[variant], roles["native_evaluation"], variant)]
                for variant in spec.VARIANTS}
            effects = paired_effects(period, evaluation, roles["native_evaluation"])
            cost["native_pair_checks"] += len(roles["native_evaluation"])
            curves.support.assert_frozen(clone, snapshot)
            cost["frozen_model_checks"] += 1
            groups[str(period)] = {"alpha": alpha, "credit": credit, "evaluation": evaluation, "effects": effects,
                "pairing": "passed", "source_and_Adam_unchanged": "passed"}
            print(f"feasible native credit {root}/{period}: alpha={alpha:.8f} fixed; upper/lower symmetric directions evaluated", flush=True)
    cell = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "seed_roles": roles, "cost": cost, "native_planning_cost": planning, "groups": groups,
        "source_initialization": initialization, "optimizer_steps": 0, "critic_fits": 0, "forecaster_fits": 0,
        "checkpoint_writes": 0, "native_trace_writes": 0, "wall_seconds": time.monotonic() - started}
    qualify(cell, preflight=preflight)
    write_json(output, cell)
    write_json(output.parent / "completion" / "ready.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return cell


def qualify(cell, *, preflight):
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or cell["root"] not in spec.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["cost"] != spec.budget(preflight=preflight) or cell["seed_roles"] != spec.seed_roles(cell["root"], preflight=preflight)
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}
            or any(cell[k] for k in ("optimizer_steps", "critic_fits", "forecaster_fits", "checkpoint_writes", "native_trace_writes"))):
        raise ValueError("Stage79 protocol, cost, roster or no-adoption contract changed")
    h = spec.arguments(cell["root"], preflight=preflight).horizon
    for p, g in cell["groups"].items():
        if g["pairing"] != "passed" or g["source_and_Adam_unchanged"] != "passed":
            raise ValueError("Stage79 pairing or source freeze failed")
        if g["effects"] != paired_effects(p, g["evaluation"], cell["seed_roles"]["native_evaluation"]):
            raise ValueError("Stage79 native effects changed")
        for variant, rows in g["evaluation"].items():
            for r in rows:
                curves.paths.check_row(r, int(p), h)
                if r["variant"] != variant or r["alpha"] != (0. if variant == "zero" else g["alpha"]):
                    raise ValueError("Stage79 decoder changed during evaluation")
        for actor in g["credit"]["actors"].values():
            if actor["geometry"]["radius_check"] != "passed":
                raise ValueError("Stage79 Fisher candidate is outside the frozen KL range")
    return cell


def aggregate(cells, *, preflight, protocol=spec, qualifier=None):
    if qualifier is None:qualifier = qualify
    if len(cells) != len(protocol.roots(preflight=preflight)) or {c["root"] for c in cells} != set(protocol.roots(preflight=preflight)):
        raise ValueError("Stage79 requires all frozen roots")
    rows = [qualifier(c, preflight=preflight) for c in sorted(cells, key=lambda c: c["root"])]
    effects = [{k: v for g in c["groups"].values() for k, v in g["effects"].items()} for c in rows]
    x = np.asarray([[e[k] for k in protocol.ENDPOINTS] for e in effects])
    endpoints = {k: {"mean": float(x[:, i].mean())} for i, k in enumerate(protocol.ENDPOINTS)}
    if not preflight:
        idx = np.random.default_rng(np.random.SeedSequence(protocol.BOOTSTRAP_SEED)).integers(0, len(rows), (protocol.BOOTSTRAP_DRAWS, len(rows)))
        tail = .05 / (2 * len(protocol.ENDPOINTS))
        ci = np.quantile(x[idx].mean(1), [tail, 1 - tail], axis=0)
        for i, k in enumerate(protocol.ENDPOINTS):
            endpoints[k].update(ci=ci[:, i].tolist(), effect="positive" if ci[0, i] > 0 else "negative" if ci[1, i] < 0 else "inconclusive")
    return {"status": "preflight_passed" if preflight else "complete", "protocol": protocol.EXPERIMENT_PROTOCOL,
        "contract": protocol.contract(), "mechanical_gate": "passed", "root_rows": rows, "endpoints": endpoints,
        "cost": {k: sum(c["cost"][k] for c in rows) for k in protocol.budget(preflight=preflight)},
        "native_trial_prerequisite": "hold_Stage67_credit_gate_unchanged", "performance_claim": "frozen_decoder_fresh_MC_direction_only"}
