"""Isolate mean learning from Gaussian exploration-scale updates."""

import numpy as np
import torch

from . import pointmaze_scenario_credit as scenario
from scripts import pointmaze_actor_parts_stage81_spec as spec

native = scenario.native


def masks(sigma_mask):
    return {"full": np.ones(len(sigma_mask), dtype=bool), "mean": ~sigma_mask, "log_std": sigma_mask}


def check_parameter_part(actor, candidate, part):
    if part == "mean":torch.testing.assert_close(candidate.log_std, actor.log_std, atol=0, rtol=0)
    if part == "log_std":torch.testing.assert_close(candidate.net.state_dict(), actor.net.state_dict(), atol=0, rtol=0)


def std_summary(actor, changed):
    log_std = actor.log_std.detach().double().numpy()
    std = actor.log_std.detach().exp().clamp(1e-4, 3.).double().numpy()
    candidates = {}
    for sign, candidate in changed.items():
        sigma = candidate.log_std.detach().exp().clamp(1e-4, 3.).double().numpy()
        candidates[sign] = {"log_std": candidate.log_std.detach().double().numpy().tolist(),
            "log_std_delta": (candidate.log_std.detach().double().numpy() - log_std).tolist(),
            "std": sigma.tolist(), "std_ratio": (sigma / std).tolist()}
    return {"source_log_std": log_std.tolist(), "source_std": std.tolist(), "candidates": candidates}


def credit_directions(clone, batches, *, period, horizon, cost):
    returns = {}
    for name, groups in batches.items():
        returns[name] = [[native.task_returns(b, period, horizon, row["episode_return"]) for b, row in group] for group in groups]
        cost["objective_checks"] += sum(map(len, groups))
        cost["mc_calls"] += 2 * sum(map(len, groups))
    candidates, actors = {}, {}
    for actor_name, length in (("upper", horizon // period), ("lower", horizon)):
        actor = getattr(clone, actor_name + "_actor")
        gradients, states, score_costs, signal_rms = {}, [], {}, {}
        for name, groups in batches.items():
            pairs = [pair for group in groups for pair in group]
            level = getattr(native.concat_hierarchical_batches([b for b, _ in pairs]), actor_name)
            r = np.asarray([[v[actor_name] for v in group] for group in returns[name]])
            signal = scenario.leave_other_out(r).reshape(-1)
            g, _, sigma_mask, c = native.reliability.episode_scores(actor, level, {"scenario": signal},
                horizon=length, clip_ratio=clone.config.clip_ratio)
            gradients[name] = g["scenario"]
            states.append(level.state)
            score_costs[name] = c
            signal_rms[name] = float(np.sqrt(np.square(signal).mean()))
            for key in ("actor_score_forward_batches", "actor_score_backward_batches"):cost[key] += c[key]
        pooled = np.concatenate(list(gradients.values())).mean(0)
        rows = {}
        for part, mask in masks(sigma_mask).items():
            projected = np.where(mask, pooled, 0.)
            changed, geometry, c = native.direction.matched_perturbations(actor, np.concatenate(states), projected,
                delta=spec.FISHER_RADIUS, chunk_size=spec.CHUNK_SIZE)
            for key in c:cost[key] += c[key]
            cost["actor_parameter_perturbations"] += len(changed)
            for sign, candidate in changed.items():
                check_parameter_part(actor, candidate, part)
                cost["parameter_part_checks"] += 1
                weights = native.joint.inference_weights(clone)
                weights[actor_name + "_actor"] = candidate.state_dict()
                candidates[f"{actor_name}_{part}_{sign}"] = weights
            rows[part] = {"geometry": geometry, "parameter_part_check": "passed", "std": std_summary(actor, changed),
                "credit_batch_cosine": native.reliability.scores.cosine(gradients["A"].mean(0)[mask], gradients["B"].mean(0)[mask]),
                "scenario_group_noise": {name: scenario.group_noise(g[:, mask], len(batches[name]), len(batches[name][0]))
                    for name, g in gradients.items()}}
        actors[actor_name] = {"parts": rows, "score_costs": score_costs, "signal_rms": signal_rms,
            "raw_loss_gradient_norms": {part: float(np.linalg.norm(pooled[mask])) for part, mask in masks(sigma_mask).items()},
            "raw_log_std_loss_gradient": pooled[sigma_mask].tolist()}
    return candidates, {"actors": actors}


def qualify(cell, *, preflight):
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or cell["root"] not in spec.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["cost"] != spec.budget(preflight=preflight) or cell["seed_roles"] != spec.seed_roles(cell["root"], preflight=preflight)
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}
            or any(cell[k] for k in ("optimizer_steps", "critic_fits", "forecaster_fits", "checkpoint_writes", "native_trace_writes"))):
        raise ValueError("Stage81 protocol, budget, seeds or source freeze changed")
    h = spec.arguments(cell["root"], preflight=preflight).horizon
    planning = dict.fromkeys(native.curves.paths.PLANNING_KEYS, 0)
    for p, g in cell["groups"].items():
        if any(g[k] != "passed" for k in ("pairing", "scenario_pairing", "source_and_Adam_unchanged")):
            raise ValueError("Stage81 scenario independence or source freeze failed")
        effects = native.paired_effects(p, g["evaluation"], cell["seed_roles"]["native_evaluation"], protocol=spec)
        if g["effects"] != effects or not np.isfinite(list(effects.values())).all():
            raise ValueError("Stage81 paired native effects changed")
        if set(g["credit"]["actors"]) != {"upper", "lower"}:raise ValueError("Stage81 actor roster changed")
        for actor in g["credit"]["actors"].values():
            if set(actor["parts"]) != set(spec.PARTS):raise ValueError("Stage81 parameter subspaces changed")
            for part, row in actor["parts"].items():
                if row["geometry"]["radius_check"] != "passed" or row["parameter_part_check"] != "passed":
                    raise ValueError("Stage81 radius or untouched-parameter check failed")
                if part == "mean" and any(any(v != 0. for v in s["log_std_delta"]) for s in row["std"]["candidates"].values()):
                    raise ValueError("Stage81 mean-only candidate changed exploration scale")
                for name, noise in row["scenario_group_noise"].items():
                    if noise["episodes"] != len(cell["seed_roles"]["credit_" + name]):
                        raise ValueError("Stage81 noise counted dependent trajectories instead of scenarios")
        for variant, rows in g["evaluation"].items():
            for r in rows:
                native.curves.paths.check_row(r, int(p), h)
                if r["variant"] != variant or r["alpha"] != (0. if variant == "zero" else g["alpha"]):
                    raise ValueError("Stage81 decoder or variant changed")
                for key in planning:planning[key] += r[key]
        n = sum(len(s["noise_seeds"]) for name in ("A", "B") for s in cell["seed_roles"]["credit_" + name])
        for key in planning:
            per_episode = h if key in ("reference_evaluations", "actor_context_evaluations") else h // int(p) - 1
            planning[key] += n * per_episode
    if planning != cell["native_planning_cost"]:raise ValueError("Stage81 native planning cost changed")
    return cell


def run(root, *, preflight, output):
    return scenario.run(root, preflight=preflight, output=output, protocol=spec, credit_builder=credit_directions,
        credit_endpoints=lambda period, credit: {}, qualifier=qualify)


def aggregate(cells, *, preflight):
    return native.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
