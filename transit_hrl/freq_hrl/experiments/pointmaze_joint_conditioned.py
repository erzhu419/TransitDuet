"""Separate upper credit from lower-only common-upper noise conditioning."""

import numpy as np

from . import pointmaze_upper_common_noise as common
from . import pointmaze_fresh_joint as fresh
from . import pointmaze_actor_swap as swaps
from scripts import pointmaze_joint_conditioned_stage102_spec as spec

learning, scenario = common.learning, common.scenario


def check_independent_pair(group, roster, *, root):
    scenario.check_scenario_pair(group, roster)
    for _, row in group:
        expected = scenario.spec.noise_seeds(root, row["seed"], row["noise_seed"])
        if ((row["policy_seed"], row["lower_seed"]) != expected or "upper_noise_seed" in row
                or row["upper_replay_forward_calls"] != 0):
            raise ValueError("Independent actor credit replayed upper noise")


def collect_actor_credit(episodes, weights, roles, method, cost, *, root):
    result = {}
    for actor in ("upper", "lower"):
        conditioned = actor == "lower" and method == "joint_conditioned"
        result[actor] = {}
        for name in ("A", "B"):
            roster = roles[("lower_" if actor == "lower" else "") + "credit_" + name]
            rows = episodes(weights, [(s["scenario_seed"], n) for s in roster for n in s["noise_seeds"]],
                method, True, pair_worker=common.worker_pair if conditioned else None)
            groups = [rows[i:i+2] for i in range(0, len(rows), 2)]
            for group, registered in zip(groups, roster):
                if conditioned:
                    common.check_scenario_pair(group, registered, root=root)
                else:
                    check_independent_pair(group, registered, root=root)
                cost["scenario_pair_checks"] += 1
                cost["lower_common_pair_checks" if conditioned else actor + "_independent_pair_checks"] += 1
            result[actor][name] = groups
    return result


def prepare_evaluation(weights, cost):
    result = swaps.compose_weights(weights["base"], {m: weights[m] for m in spec.METHODS}, protocol=spec)
    cost["actor_composition_checks"] += len(result)
    return result, {"actor_composition": "passed",
        "training_credit": "separate_actor_batches_independent_upper_conditioned_lower_only"}


def qualify(cell, *, preflight):
    common.budget_training.call_budget.qualify(cell, preflight=preflight, protocol=spec)
    if cell["source_initialization"] != spec.source_record(cell["root"]):
        raise ValueError("Joint conditioned learning changed the original teacher cohort")
    for p, group in cell["groups"].items():
        if (group["actor_composition"] != "passed" or
                group["training_credit"] != "separate_actor_batches_independent_upper_conditioned_lower_only"):
            raise ValueError("Joint actor-specific credit contract changed")
        for rows in group["evaluation"].values():
            for row in rows:
                expected = scenario.spec.noise_seeds(cell["root"], row["seed"], row["seed"])
                if ((row["noise_seed"], row["policy_seed"], row["lower_seed"]) != (row["seed"], *expected)
                        or "upper_noise_seed" in row or row["upper_replay_forward_calls"] != 0):
                    raise ValueError("Lower common credit noise leaked into evaluation")
        e = group["effects"]
        np.testing.assert_allclose(e[f"{p}/joint_conditioned_minus_joint_independent"],
            e[f"{p}/joint_conditioned_minus_independent_upper_conditioned_lower"] +
            e[f"{p}/independent_upper_conditioned_lower_minus_joint_independent"], atol=1e-12, rtol=0)
    return cell


def run(root, *, preflight, output):
    return learning.run(root, preflight=preflight, output=output, protocol=spec, qualifier=qualify,
        source_loader=fresh.load_source,
        actor_credit_collector=lambda episodes, weights, roles, method, cost:
            collect_actor_credit(episodes, weights, roles, method, cost, root=root),
        evaluation_weights=lambda p, w, c: prepare_evaluation(w, c))


def aggregate(cells, *, preflight):
    result = learning.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    result.update(primary_endpoints=list(spec.PRIMARY_ENDPOINTS),
        performance_claim="separate_actor_credit_joint_MC_mean_learning_not_full_actor_critic_or_frequency_superiority",
        joint_conditioning_confirmation="mechanical_only" if preflight else (
            "supported" if all(result["endpoints"][k]["ci"][0] > 0 for k in spec.PRIMARY_ENDPOINTS) else "not_supported"))
    return result
