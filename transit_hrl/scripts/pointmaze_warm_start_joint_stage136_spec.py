"""One shared on-policy round to isolate warm-start joint PPO damage."""

import math

from scripts import pointmaze_native_selection_replication_stage135_spec as source
from scripts import pointmaze_joint_reference_stage121_spec as ppo

ROOT, PERIODS = source.ROOT, source.PERIODS
ROOTS = source.ROOTS[:2]
PROTOCOL = "pointmaze_warm_start_joint_stage136_v1"
EXPERIMENT_PROTOCOL = PROTOCOL
POLICY = "selected_upper_one_round_shared_batch_actor_critic_diagnosis"
RUNNER_SCRIPT = "scripts/run_pointmaze_warm_start_joint_stage136.py"
SOURCE_RUN = "pointmaze_native_selection_replication_stage135_frozen_20261008_r1"
WORKERS, SCENARIOS, NOISE_FOLDS, EVALUATION_EPISODES = 8, 8, 2, 32
METHODS = ("upper_only", "lower_only", "joint")
VARIANTS = ("source_forecast", "warm_start", *METHODS, "joint_blinded")
CONTRASTS = (("warm_start", "source_forecast"), ("upper_only", "warm_start"),
    ("lower_only", "warm_start"), ("joint", "warm_start"),
    ("joint", "source_forecast"), ("joint", "lower_only"),
    ("joint", "upper_only"), ("joint", "joint_blinded"))


def arguments(root):
    return source.arguments(root)


def source_result(root):
    return ROOT / "results" / SOURCE_RUN / "cells" / f"replicate_{root}" / "result.json"


def seed_roles(root):
    base = 136100000 + ROOTS.index(root) * 100000
    return {"training": [{"scenario_seed": base + i + 1,
        "noise_seeds": [base + 10001 + 2*i, base + 10002 + 2*i]}
        for i in range(SCENARIOS)],
        "evaluation": list(range(base + 90001, base + 90001 + EVALUATION_EPISODES))}


def optimizer_seed(root, period, level):
    return root + 136000000 + period * 100 + {"upper": 1, "lower": 2}[level]


def budget():
    h, periods = arguments(ROOTS[0]).horizon, len(PERIODS)
    paths, e = SCENARIOS * NOISE_FOLDS, EVALUATION_EPISODES
    episodes = periods * (3 * paths + len(VARIANTS) * e)
    upper = sum(2 * ppo.EPOCHS * math.ceil(paths * (h//p) / ppo.MINIBATCH) for p in PERIODS)
    lower = periods * 2 * ppo.EPOCHS * math.ceil(paths * h / ppo.MINIBATCH)
    return {"source_cell_loads": 1, "source_clone_loads": periods,
        "lower_checkpoint_loads": periods, "upper_checkpoint_loads": periods,
        "collection_episodes": periods * paths, "training_comparator_episodes": periods * 2 * paths,
        "evaluation_episodes": periods * len(VARIANTS) * e,
        "native_episodes": episodes, "native_steps": episodes*h, "native_lower_calls": episodes*h,
        "native_upper_calls": sum((2*paths + 4*e) * (h//p) for p in PERIODS),
        "native_donor_response_calls": 2*episodes*h,
        "planning_renewals": sum((3*paths + len(VARIANTS)*e) * (h//p) for p in PERIODS),
        "planning_fits": sum((3*paths + len(VARIANTS)*e) * (h//p-1) for p in PERIODS),
        "planning_reference_calls": episodes*h, "credit_checks": periods*paths,
        "upper_actor_optimizer_steps": upper, "upper_value_optimizer_steps": upper,
        "lower_actor_optimizer_steps": lower, "lower_value_optimizer_steps": lower,
        "ppo_update_calls": periods*4,
        "upper_diagnostic_gradient_batches": sum(4*math.ceil(SCENARIOS*(h//p)/ppo.MINIBATCH) for p in PERIODS),
        "lower_diagnostic_gradient_batches": periods*4*math.ceil(SCENARIOS*h/ppo.MINIBATCH),
        "checkpoint_writes": 0, "native_trace_writes": 0}


def contract():
    return {"source": SOURCE_RUN, "roots": list(ROOTS), "periods": list(PERIODS),
        "warm_start": "Stage135_validation_selected_upper_exact_saved_fit_no_evaluation_selection",
        "training": "one_shared_on_policy_batch_per_period_same_initial_actors_and_critics",
        "interventions": list(METHODS), "optimizer": ppo.contract()["ppo"],
        "credit": "unchanged_MC_minus_remaining_time_critic_LOO_only_reported_not_used",
        "layer_isolation": "per_level_optimizer_seeds_joint_equals_separate_updates_in_parameter_space",
        "execution": ppo.contract()["execution"], "rosters": "fresh_Stage136_scenarios_and_two_noise_folds",
        "statistics": "two_source_roots_descriptive_effects_no_confirmation_CI_or_automatic_extension",
        "limits": "warm_start_credit_diagnosis_not_joint_HRL_frequency_separation_or_promotion_confirmation",
        "artifacts": "scalar_JSON_no_checkpoint_or_trace_writes_inherited_cost_separate"}
