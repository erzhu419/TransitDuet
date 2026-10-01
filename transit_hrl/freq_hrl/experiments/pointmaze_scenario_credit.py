"""Separate exogenous scenario variability from action-dependent task credit."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
import time

import numpy as np

from . import pointmaze_feasible_credit as native
from .pointmaze_root_response import write_json
from scripts import pointmaze_scenario_credit_stage80_spec as spec


def worker_native(job):
    weights, seed, noise_seed, variant, period, predictor, alpha, envelope, collect = job
    _, args = native._WORKER
    policy_seed, lower_seed = spec.noise_seeds(args.optimizer_seed, seed, noise_seed)
    batch, row = native.native_episode(weights, seed=seed, variant=variant, period=period,
        predictor=predictor, alpha=alpha, envelope=envelope, collect=collect,
        policy_seed=policy_seed, lower_seed=lower_seed)
    row["noise_seed"] = noise_seed
    return batch, row


def check_scenario_pair(pairs, scenario):
    if ([r["seed"] for _, r in pairs] != [scenario["scenario_seed"]] * len(pairs) or
            [r["noise_seed"] for _, r in pairs] != scenario["noise_seeds"] or
            len({r["policy_seed"] for _, r in pairs}) != len(pairs) or
            len({r["lower_seed"] for _, r in pairs}) != len(pairs)):
        raise ValueError("Stage80 scenario/noise roles are not independent replicates")
    reference = pairs[0][0].lower.state
    for batch, _ in pairs[1:]:
        np.testing.assert_array_equal(batch.lower.state[0, :4], reference[0, :4])
        np.testing.assert_array_equal(batch.lower.state[:, 6:390], reference[:, 6:390])


def leave_other_out(returns):
    # The baseline excludes this rollout's actions; shared exogenous truth is fixed.
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 3 or r.shape[1] < 2:
        raise ValueError("Stage80 needs scenario x independent replicate x time returns")
    return r - (r.sum(axis=1, keepdims=True) - r) / (r.shape[1] - 1)


def group_noise(gradients, scenarios, replicas):
    grouped = gradients.reshape(scenarios, replicas, -1).mean(1)
    return native.independent.gradient_noise(grouped)


def credit_directions(clone, batches, *, period, horizon, cost):
    rates = {name: float(np.mean([r["episode_return"] for group in groups for _, r in group])) / horizon
        for name, groups in batches.items()}
    returns = {}
    for name, groups in batches.items():
        returns[name] = [[native.task_returns(batch, period, horizon, r["episode_return"]) for batch, r in group]
            for group in groups]
        cost["objective_checks"] += sum(map(len, groups))
        cost["mc_calls"] += 2 * sum(map(len, groups))
    candidates, actors = {}, {}
    for actor_name, length, duration in (("upper", horizon // period, period), ("lower", horizon, 1)):
        actor = getattr(clone, actor_name + "_actor")
        gradients = {method: {} for method in spec.METHODS}
        states, score_costs, signal_rms = [], {}, {}
        for name, groups in batches.items():
            pairs = [pair for group in groups for pair in group]
            level = getattr(native.concat_hierarchical_batches([b for b, _ in pairs]), actor_name)
            r = np.asarray([[v[actor_name] for v in group] for group in returns[name]])
            remaining = horizon - np.arange(length) * duration
            signals = {"rate": (r - remaining * rates["B" if name == "A" else "A"]).reshape(-1),
                "scenario": leave_other_out(r).reshape(-1)}
            g, _, sigma_mask, c = native.reliability.episode_scores(actor, level, signals,
                horizon=length, clip_ratio=clone.config.clip_ratio)
            for method in spec.METHODS:gradients[method][name] = g[method]
            states.append(level.state)
            score_costs[name] = c
            signal_rms[name] = {m: float(np.sqrt(np.square(v).mean())) for m, v in signals.items()}
            for key in ("actor_score_forward_batches", "actor_score_backward_batches"):cost[key] += c[key]
        rows = {}
        for method, episodes in gradients.items():
            mean_gradient = np.concatenate(list(episodes.values())).mean(0)
            changed, geometry, c = native.direction.matched_perturbations(actor, np.concatenate(states), mean_gradient,
                delta=spec.FISHER_RADIUS, chunk_size=spec.CHUNK_SIZE)
            for key in c:cost[key] += c[key]
            cost["actor_parameter_perturbations"] += len(changed)
            for sign, candidate in changed.items():
                weights = native.joint.inference_weights(clone)
                weights[actor_name + "_actor"] = candidate.state_dict()
                candidates[f"{actor_name}_{method}_{sign}"] = weights
            rows[method] = {"geometry": geometry, "pooled_loss_gradient_norm": float(np.linalg.norm(mean_gradient)),
                "credit_batch_cosine": {part: native.reliability.scores.cosine(episodes["A"].mean(0)[mask], episodes["B"].mean(0)[mask])
                    for part, mask in (("all", np.ones(len(sigma_mask), dtype=bool)), ("mean", ~sigma_mask), ("log_std", sigma_mask))},
                "scenario_group_noise": {name: group_noise(g, len(batches[name]), len(batches[name][0])) for name, g in episodes.items()}}
        actors[actor_name] = {"methods": rows, "score_costs": score_costs, "signal_rms": signal_rms}
    return candidates, {"baseline_reward_rates": rates, "actors": actors}


def credit_effects(period, credit):
    result = {}
    for actor_name, row in credit["actors"].items():
        rate, scenario = [row["methods"][m] for m in spec.METHODS]
        result[f"{period}/{actor_name}/scenario_minus_rate_cosine"] = scenario["credit_batch_cosine"]["all"] - rate["credit_batch_cosine"]["all"]
        variance = {m: float(np.mean([v["covariance_trace"] for v in row["methods"][m]["scenario_group_noise"].values()]))
            for m in spec.METHODS}
        result[f"{period}/{actor_name}/log_rate_over_scenario_gradient_variance"] = float(np.log(variance["rate"] / variance["scenario"]))
    return result


def run(root, *, preflight, output):
    source = json.loads(spec.source_result(root).read_text())
    if (source["status"], source["protocol"], source["root"], source["preflight"], source["contract"]) != (
            "complete", spec.source.source.EXPERIMENT_PROTOCOL, root, False, spec.source.source.contract()):
        raise ValueError("Stage80 needs the completed full Stage78 decoder")
    clones, predictor, initialization = native.curves.support.native.load_source(root, preflight=False)
    args, roles = spec.arguments(root, preflight=preflight), spec.seed_roles(root, preflight=preflight)
    cost, groups, started = dict.fromkeys(spec.budget(preflight=preflight), 0), {}, time.monotonic()
    cost.update(source_clone_loads=len(clones), forecaster_loads=1, decoder_loads=len(clones))
    planning = dict.fromkeys(native.curves.paths.PLANNING_KEYS, 0)
    with ProcessPoolExecutor(max_workers=spec.options(preflight=preflight)["workers"], mp_context=mp.get_context("spawn"),
            initializer=native.init_worker, initargs=(clones[str(spec.PERIODS[0])].config, args)) as pool:
        for period in spec.PERIODS:
            clone = clones[str(period)]
            snapshot, weights = copy.deepcopy(clone.state_dict()), native.joint.inference_weights(clone)
            calibration = source["groups"][str(period)]["calibration"]
            native.bounded.check_calibration(calibration)
            alpha, envelope = calibration["alpha"], calibration["envelope"]

            def episodes(w, scenarios, variant, collect=False):
                pairs = list(pool.map(worker_native, [(w, scenario, noise, variant, period, predictor,
                    0. if variant == "zero" else alpha, envelope, collect) for scenario, noise in scenarios]))
                for _, row in pairs:
                    native.curves.paths.check_row(row, period, args.horizon)
                    cost["native_episodes"] += 1
                    cost["native_steps"] += row["episode_length"]
                    cost["native_lower_calls"] += row["lower_calls"]
                    cost["native_upper_calls"] += row["upper_calls"]
                    cost["pairing_upper_forward_calls"] += row["upper_calls"]
                    cost["native_network_checks"] += 1
                    cost["credit_episodes" if collect else "evaluation_episodes"] += 1
                    for key in planning:planning[key] += row[key]
                return pairs

            batches = {}
            for name in ("A", "B"):
                roster = roles["credit_" + name]
                pairs = episodes(weights, [(s["scenario_seed"], n) for s in roster for n in s["noise_seeds"]], "base", True)
                replicas = spec.options(preflight=preflight)["rollouts_per_scenario"]
                batches[name] = [pairs[i:i+replicas] for i in range(0, len(pairs), replicas)]
                for group, scenario in zip(batches[name], roster):
                    check_scenario_pair(group, scenario)
                    cost["scenario_pair_checks"] += 1
            candidates, credit = credit_directions(clone, batches, period=period, horizon=args.horizon, cost=cost)
            del batches
            candidates.update(base=weights, zero=weights)
            evaluation = {v: [r for _, r in episodes(candidates[v], [(s, s) for s in roles["native_evaluation"]], v)] for v in spec.VARIANTS}
            effects = native.paired_effects(period, evaluation, roles["native_evaluation"], protocol=spec)
            effects.update(credit_effects(period, credit))
            cost["native_pair_checks"] += len(roles["native_evaluation"])
            native.curves.support.assert_frozen(clone, snapshot)
            cost["frozen_model_checks"] += 1
            groups[str(period)] = {"alpha": alpha, "credit": credit, "evaluation": evaluation, "effects": effects,
                "scenario_pairing": "passed", "pairing": "passed", "source_and_Adam_unchanged": "passed"}
            print(f"scenario credit {root}/{period}: same-data rate/scenario directions evaluated, alpha={alpha:.8f} frozen", flush=True)
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
        raise ValueError("Stage80 protocol, budget, seeds or source freeze changed")
    h = spec.arguments(cell["root"], preflight=preflight).horizon
    planning = dict.fromkeys(native.curves.paths.PLANNING_KEYS, 0)
    for p, g in cell["groups"].items():
        if any(g[k] != "passed" for k in ("pairing", "scenario_pairing", "source_and_Adam_unchanged")):
            raise ValueError("Stage80 scenario independence or source freeze failed")
        expected = native.paired_effects(p, g["evaluation"], cell["seed_roles"]["native_evaluation"], protocol=spec)
        expected.update(credit_effects(p, g["credit"]))
        if g["effects"] != expected or not np.isfinite(list(expected.values())).all():
            raise ValueError("Stage80 native or credit effects changed")
        for actor in g["credit"]["actors"].values():
            for method in actor["methods"].values():
                if method["geometry"]["radius_check"] != "passed":raise ValueError("Stage80 fixed KL radius failed")
                for name, noise in method["scenario_group_noise"].items():
                    if noise["episodes"] != len(cell["seed_roles"]["credit_" + name]):
                        raise ValueError("Stage80 noise counted trajectories instead of scenario groups")
        for variant, rows in g["evaluation"].items():
            for r in rows:
                native.curves.paths.check_row(r, int(p), h)
                if r["variant"] != variant or r["alpha"] != (0. if variant == "zero" else g["alpha"]):
                    raise ValueError("Stage80 decoder or variant changed")
                for key in planning:planning[key] += r[key]
        n = sum(len(s["noise_seeds"]) for name in ("A", "B") for s in cell["seed_roles"]["credit_" + name])
        for key in planning:
            per_episode = h if key in ("reference_evaluations", "actor_context_evaluations") else h // int(p) - 1
            planning[key] += n * per_episode
    if planning != cell["native_planning_cost"]:raise ValueError("Stage80 native planning calls changed")
    return cell


def aggregate(cells, *, preflight):
    return native.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
