"""Measure whether upper/lower mean-credit directions coexist in native control."""

import numpy as np
import torch

from . import pointmaze_actor_parts as parts
from scripts import pointmaze_joint_mean_stage82_spec as spec

native = parts.native


def check_joint_composition(weights, candidates, signs):
    for actor, sign in zip(("upper", "lower"), signs):
        key = actor + "_actor"
        torch.testing.assert_close(weights[key], candidates[f"{actor}_half_{sign}"][key], atol=0, rtol=0)


def credit_directions(clone, batches, *, period, horizon, cost):
    scored = parts.scenario_actor_scores(clone, batches, period=period, horizon=horizon, cost=cost)
    candidates, actors = {}, {}
    for actor_name, score in scored.items():
        actor = getattr(clone, actor_name + "_actor")
        gradients, sigma = score["gradients"], score["sigma_mask"]
        projected = np.where(sigma, 0., np.concatenate(list(gradients.values())).mean(0))
        allocations = {}
        for allocation, fraction in spec.ALLOCATIONS.items():
            changed, geometry, c = native.direction.matched_perturbations(actor, score["states"], projected,
                delta=spec.FISHER_RADIUS * fraction, chunk_size=spec.CHUNK_SIZE)
            for key in c:cost[key] += c[key]
            cost["actor_parameter_perturbations"] += len(changed)
            for sign, candidate in changed.items():
                parts.check_parameter_part(actor, candidate, "mean")
                cost["parameter_part_checks"] += 1
                weights = native.joint.inference_weights(clone)
                weights[actor_name + "_actor"] = candidate.state_dict()
                candidates[f"{actor_name}_{allocation}_{sign}"] = weights
            allocations[allocation] = {"geometry": geometry, "parameter_part_check": "passed",
                "std": parts.std_summary(actor, changed)}
        actors[actor_name] = {"allocations": allocations, "score_costs": score["score_costs"],
            "signal_rms": score["signal_rms"], "mean_loss_gradient_norm": float(np.linalg.norm(projected)),
            "credit_batch_cosine": native.reliability.scores.cosine(gradients["A"].mean(0)[~sigma], gradients["B"].mean(0)[~sigma]),
            "scenario_group_noise": {name: parts.scenario.group_noise(g[:, ~sigma], len(batches[name]), len(batches[name][0]))
                for name, g in gradients.items()}}
    joint_geometry = {}
    for variant, signs in spec.JOINT_SIGNS.items():
        weights = native.joint.inference_weights(clone)
        for actor, sign in zip(("upper", "lower"), signs):
            weights[actor + "_actor"] = candidates[f"{actor}_half_{sign}"][actor + "_actor"]
        check_joint_composition(weights, candidates, signs)
        cost["joint_composition_checks"] += 1
        candidates[variant] = weights
        joint_geometry[variant] = {"nominal_sum_kl": spec.FISHER_RADIUS,
            "exact_sum_kl": sum(actors[a]["allocations"]["half"]["geometry"]["exact_kl"][s]
                for a, s in zip(("upper", "lower"), signs)), "composition_check": "passed"}
    return candidates, {"actors": actors, "joint_geometry": joint_geometry}


def paired_effects(period, evaluation, seeds, *, protocol=spec):
    result = native.paired_effects(period, evaluation, seeds, protocol=protocol)
    reward = {v: np.asarray([r["episode_return"] for r in rows]) for v, rows in evaluation.items()}
    result[f"{period}/joint_plus_interaction"] = float(np.mean(
        reward["joint_plus"] + reward["base"] - reward["upper_half_plus"] - reward["lower_half_plus"]))
    return result


def qualify(cell, *, preflight):
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or cell["root"] not in spec.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["cost"] != spec.budget(preflight=preflight) or cell["seed_roles"] != spec.seed_roles(cell["root"], preflight=preflight)
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}
            or any(cell[k] for k in ("optimizer_steps", "critic_fits", "forecaster_fits", "checkpoint_writes", "native_trace_writes"))):
        raise ValueError("Stage82 protocol, budget, seeds or source freeze changed")
    h = spec.arguments(cell["root"], preflight=preflight).horizon
    planning = dict.fromkeys(native.curves.paths.PLANNING_KEYS, 0)
    for p, g in cell["groups"].items():
        if any(g[k] != "passed" for k in ("pairing", "scenario_pairing", "source_and_Adam_unchanged")):
            raise ValueError("Stage82 scenario independence or source freeze failed")
        expected = paired_effects(p, g["evaluation"], cell["seed_roles"]["native_evaluation"])
        if g["effects"] != expected or not np.isfinite(list(expected.values())).all():
            raise ValueError("Stage82 paired rewards or joint interaction changed")
        credit = g["credit"]
        if set(credit["actors"]) != {"upper", "lower"} or set(credit["joint_geometry"]) != set(spec.JOINT_SIGNS):
            raise ValueError("Stage82 actor or joint-sign roster changed")
        for actor in credit["actors"].values():
            if set(actor["allocations"]) != set(spec.ALLOCATIONS):raise ValueError("Stage82 allocation roster changed")
            for allocation, row in actor["allocations"].items():
                geometry = row["geometry"]
                if (geometry["radius_check"] != "passed" or row["parameter_part_check"] != "passed"
                        or geometry["nominal_fisher_kl"] != spec.FISHER_RADIUS * spec.ALLOCATIONS[allocation]
                        or any(any(v != 0. for v in s["log_std_delta"]) for s in row["std"]["candidates"].values())):
                    raise ValueError("Stage82 mean-only radius or frozen std failed")
            for name, noise in actor["scenario_group_noise"].items():
                if noise["episodes"] != len(cell["seed_roles"]["credit_" + name]):
                    raise ValueError("Stage82 dependent trajectories counted as independent scenarios")
        for v, signs in spec.JOINT_SIGNS.items():
            row = credit["joint_geometry"][v]
            exact = sum(credit["actors"][a]["allocations"]["half"]["geometry"]["exact_kl"][s]
                for a, s in zip(("upper", "lower"), signs))
            if (row["composition_check"] != "passed" or row["nominal_sum_kl"] != spec.FISHER_RADIUS
                    or row["exact_sum_kl"] != exact or not .5 * spec.FISHER_RADIUS <= exact <= 2 * spec.FISHER_RADIUS):
                raise ValueError("Stage82 joint composition or sum-of-level KL changed")
        for variant, rows in g["evaluation"].items():
            for r in rows:
                native.curves.paths.check_row(r, int(p), h)
                if r["variant"] != variant or r["alpha"] != (0. if variant == "zero" else g["alpha"]):
                    raise ValueError("Stage82 decoder or variant changed")
                for key in planning:planning[key] += r[key]
        n = sum(len(s["noise_seeds"]) for name in ("A", "B") for s in cell["seed_roles"]["credit_" + name])
        for key in planning:planning[key] += n * (h if key in ("reference_evaluations", "actor_context_evaluations") else h // int(p) - 1)
    if planning != cell["native_planning_cost"]:raise ValueError("Stage82 native planning cost changed")
    return cell


def run(root, *, preflight, output):
    return parts.scenario.run(root, preflight=preflight, output=output, protocol=spec, credit_builder=credit_directions,
        credit_endpoints=lambda period, credit: {}, qualifier=qualify, effects_builder=paired_effects)


def aggregate(cells, *, preflight):
    return native.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
