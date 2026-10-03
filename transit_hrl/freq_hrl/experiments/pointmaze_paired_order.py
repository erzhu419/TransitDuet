"""Train both update orders with common actor-specific random rosters."""

import copy

import numpy as np

from . import pointmaze_joint_conditioned as joint
from scripts import pointmaze_paired_order_stage104_spec as spec

learning, common, swaps = joint.learning, joint.common, joint.swaps


def collect_credit(episodes, weights, roles, method, cost, *, root, actors):
    result = {}
    for actor in actors:
        conditioned = actor == "lower" and method in ("joint_conditioned", "staged_common")
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
                    joint.check_independent_pair(group, registered, root=root)
                cost["scenario_pair_checks"] += 1
                cost["lower_common_pair_checks" if conditioned else actor + "_independent_pair_checks"] += 1
            result[actor][name] = groups
    return result


def train_models(period, models, roles, episodes, cost, *, root, preflight):
    n = spec.options(preflight=preflight)["updates"]
    horizon = spec.arguments(root, preflight=preflight).horizon
    history = {m: [] for m in models}
    initial = {m: copy.deepcopy(models[m].state_dict()) for m in spec.STAGED_METHODS}
    lower_final = {}
    for step in spec.update_schedule(period, preflight=preflight):
        method = step["method"]
        model = models[method]
        weights = learning.native.joint.inference_weights(model)
        credit = collect_credit(episodes, weights, roles["training_rounds"][step["credit_iteration"]-1],
            method, cost, root=root, actors=step["allocation"])
        mean = float(np.mean([r["episode_return"] for batches in credit.values()
            for groups in batches.values() for group in groups for _, r in group]))
        row = learning.update_mean(model, None, method=method, period=period, horizon=horizon, cost=cost,
            allocation=step["allocation"], actor_batches=credit)
        del credit, weights
        boundary = "not_applicable"
        if step["phase"] != "joint" and step["credit_iteration"] == n:
            reference = initial[method] if step["phase"] == "lower" else lower_final[method]
            learning.check_training_freeze(model, reference, step["allocation"])
            cost["phase_boundary_checks"] += 1
            boundary = "passed"
            if step["phase"] == "lower":
                lower_final[method] = copy.deepcopy(model.state_dict())
        row.update(iteration=step["iteration"], credit_iteration=step["credit_iteration"], phase=step["phase"],
            phase_boundary_freeze=boundary, credit_mean_reward_before_update=mean)
        history[method].append(row)
        print(f"{spec.EXPERIMENT_PROTOCOL} {root}/{period}/{method}: {step['phase']} actor round {step['credit_iteration']}/{n} done", flush=True)
    return history


def prepare_evaluation(period, weights, cost, *, preflight):
    result = swaps.compose_weights(weights["base"], {m: weights[m] for m in spec.METHODS}, protocol=spec)
    cost["actor_composition_checks"] += len(result)
    return result, {"actor_composition": "passed", "phase_freeze": "passed",
        "training_schedule": "joint_simultaneous_vs_all_lower_then_all_upper_same_actor_rosters",
        "matched_training_budget": spec.matched_path_budget(period, preflight=preflight)}


def qualify(cell, *, preflight):
    learning.qualify(cell, preflight=preflight, protocol=spec, update_schedule=spec.update_schedule)
    if cell["source_initialization"] != spec.source_record(cell["root"]):
        raise ValueError("Paired-order source teacher or decoder changed")
    o = spec.options(preflight=preflight)
    h = spec.arguments(cell["root"], preflight=preflight).horizon
    episodes = 2*o["credit_scenarios_per_batch"]*o["rollouts_per_scenario"]
    for p, g in cell["groups"].items():
        if (g["actor_composition"] != "passed" or g["phase_freeze"] != "passed"
                or g["training_schedule"] != "joint_simultaneous_vs_all_lower_then_all_upper_same_actor_rosters"
                or g["matched_training_budget"] != spec.matched_path_budget(int(p), preflight=preflight)):
            raise ValueError("Paired-order schedule, phase freeze or method-path budget changed")
        schedule = spec.update_schedule(int(p), preflight=preflight)
        for method, t in g["trained"].items():
            steps = [s for s in schedule if s["method"] == method]
            nominal_total = 0.
            for row, step in zip(t["history"], steps):
                boundary = "passed" if step["phase"] != "joint" and step["credit_iteration"] == o["updates"] else "not_applicable"
                if (row["phase"], row["credit_iteration"], row["phase_boundary_freeze"]) != (step["phase"], step["credit_iteration"], boundary):
                    raise ValueError("Paired-order update phase or shared credit roster changed")
                a = step["allocation"]
                nominal = spec.FISHER_RADIUS*sum(v/(int(p) if k == "upper" else 1) for k, v in a.items())
                exact = sum(r["geometry"]["exact_kl"]["plus"]/(int(p) if k == "upper" else 1) for k, r in row["actors"].items())
                if (row["nominal_call_weighted_kl"] != nominal or row["exact_call_weighted_kl"] != exact
                        or not .5*nominal <= exact <= 2*nominal):
                    raise ValueError("Paired-order call-weighted KL changed")
                nominal_total += nominal
                for actor, r in row["actors"].items():
                    if r["gradient_episodes"] != episodes or r["decision_calls_per_episode"] != (h//int(p) if actor == "upper" else h):
                        raise ValueError("Paired-order actor samples changed")
            if not np.isclose(nominal_total, o["updates"]*spec.FISHER_RADIUS, atol=1e-12, rtol=0):
                raise ValueError("Paired-order cumulative nominal budget changed")
        for rows in g["evaluation"].values():
            for row in rows:
                expected = joint.scenario.spec.noise_seeds(cell["root"], row["seed"], row["seed"])
                if ((row["noise_seed"], row["policy_seed"], row["lower_seed"]) != (row["seed"], *expected)
                        or "upper_noise_seed" in row or row["upper_replay_forward_calls"]):
                    raise ValueError("Paired-order training conditioning leaked into evaluation")
    return cell


def run(root, *, preflight, output):
    return learning.run(root, preflight=preflight, output=output, protocol=spec, qualifier=qualify,
        source_loader=joint.fresh.load_source,
        training_loop=lambda p, m, r, e, c: train_models(p, m, r, e, c, root=root, preflight=preflight),
        evaluation_weights=lambda p, w, c: prepare_evaluation(p, w, c, preflight=preflight))


def aggregate(cells, *, preflight):
    result = learning.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    result.update(primary_endpoints=list(spec.PRIMARY_ENDPOINTS),
        performance_claim="same_actor_roster_MC_update_order_comparison_not_full_actor_critic_or_frequency_superiority",
        paired_order_confirmation="mechanical_only" if preflight else (
            "supported" if all(result["endpoints"][k]["ci"][0] > 0 for k in spec.PRIMARY_ENDPOINTS) else "not_supported"))
    return result
