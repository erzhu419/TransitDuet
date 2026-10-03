"""Adapt original teacher means to a registered faster regime distribution."""

from . import pointmaze_joint_conditioned as joint
from scripts import pointmaze_conditioning_shift_stage105_spec as spec

learning = joint.learning


def prepare_evaluation(period, weights, cost, *, root, preflight):
    composed = joint.swaps.compose_weights(weights["base"], {m: weights[m] for m in spec.METHODS}, protocol=spec)
    cost["actor_composition_checks"] += len(composed)
    return composed, {"actor_composition": "passed", "task_options": spec.task_options(root, preflight=preflight),
        "training_credit": "separate_actor_batches_independent_upper_conditioned_lower_only"}


def qualify(cell, *, preflight):
    joint.common.budget_training.call_budget.qualify(cell, preflight=preflight, protocol=spec)
    if cell["source_initialization"] != spec.source_record(cell["root"]):
        raise ValueError("Shifted conditioning changed the original teacher or decoder")
    for g in cell["groups"].values():
        if (g["actor_composition"] != "passed" or g["task_options"] != spec.task_options(cell["root"], preflight=preflight)
                or g["training_credit"] != "separate_actor_batches_independent_upper_conditioned_lower_only"):
            raise ValueError("Shifted task or actor-specific credit changed")
        for rows in g["evaluation"].values():
            for r in rows:
                expected = joint.scenario.spec.noise_seeds(cell["root"], r["seed"], r["seed"])
                if ((r["noise_seed"], r["policy_seed"], r["lower_seed"]) != (r["seed"], *expected)
                        or "upper_noise_seed" in r or r["upper_replay_forward_calls"]):
                    raise ValueError("Training conditioning leaked into shifted evaluation")
    return cell


def run(root, *, preflight, output):
    return learning.run(root, preflight=preflight, output=output, protocol=spec, qualifier=qualify,
        source_loader=joint.fresh.load_source,
        actor_credit_collector=lambda e, w, r, m, c: joint.collect_actor_credit(e, w, r, m, c, root=root),
        evaluation_weights=lambda p, w, c: prepare_evaluation(p, w, c, root=root, preflight=preflight))


def aggregate(cells, *, preflight):
    result = learning.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    result.update(primary_endpoints=list(spec.PRIMARY_ENDPOINTS),
        performance_claim="fast_regime_adaptation_teacher_initialized_joint_MC_mean_learning_not_cross_domain_or_frequency_superiority",
        shifted_conditioning_confirmation="mechanical_only" if preflight else (
            "supported" if all(result["endpoints"][k]["ci"][0] > 0 for k in spec.PRIMARY_ENDPOINTS) else "not_supported"))
    return result
