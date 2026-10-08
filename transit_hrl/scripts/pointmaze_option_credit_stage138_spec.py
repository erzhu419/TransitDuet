"""Native mean-action replacement baseline for sampled upper option credit."""

import math

from scripts import pointmaze_bounded_ppo_stage137_spec as source

ROOT, ROOTS, PERIODS = source.ROOT, source.ROOTS, source.PERIODS
PROTOCOL = "pointmaze_option_credit_stage138_v1"
EXPERIMENT_PROTOCOL = PROTOCOL
POLICY = "paired_single_option_mean_baseline_fixed_KL_PPO_credit"
RUNNER_SCRIPT = "scripts/run_pointmaze_option_credit_stage138.py"
SOURCE_RUN = "pointmaze_bounded_ppo_stage137_probe_20261008_r1"
WARM_SOURCE_RUN = source.source.SOURCE_RUN
WORKERS, SCENARIOS, EVALUATION_EPISODES = 16, 8, 32
PANELS, METHODS = ("A", "B"), ("critic", "option_credit")
VARIANTS = ("source_forecast", "warm_start",
    *tuple(m+"_"+sign for m in METHODS for sign in ("plus", "minus")))
FISHER_RADIUS, METRICS = source.FISHER_RADIUS, source.METRICS
CONTRASTS = (("warm_start", "source_forecast"),
    *tuple((m+"_"+s, "warm_start") for m in METHODS for s in ("plus", "minus")),
    ("option_credit_plus", "critic_plus"),
    *tuple((m+"_plus", m+"_minus") for m in METHODS))
arguments = source.arguments


def source_result(root):
    return ROOT/"results"/SOURCE_RUN/"cells"/f"replicate_{root}"/"result.json"


def warm_result(root):
    return ROOT/"results"/WARM_SOURCE_RUN/"cells"/f"replicate_{root}"/"result.json"


def seed_roles(root):
    base = 138100000 + ROOTS.index(root)*100000
    return {"training": [{"scenario_seed": base+i+1,
        "noise_seeds": {"A": base+10001+i, "B": base+20001+i}} for i in range(SCENARIOS)],
        "evaluation": list(range(base+90001, base+90001+EVALUATION_EPISODES))}


def optimizer_seed(root, period):
    return root+138000000+100*period+1


def budget():
    h, paths, e = arguments(ROOTS[0]).horizon, SCENARIOS*len(PANELS), EVALUATION_EPISODES
    episodes = [paths*(1+h//p)+len(VARIANTS)*e for p in PERIODS]
    chunks = [math.ceil(paths*(h//p)/source.source.ppo.MINIBATCH) for p in PERIODS]
    optimizer = len(METHODS)*source.source.ppo.EPOCHS*sum(chunks)
    return {"source_cell_loads": 2, "lower_checkpoint_loads": len(PERIODS), "upper_checkpoint_loads": len(PERIODS),
        "collection_episodes": len(PERIODS)*paths, "counterfactual_episodes": sum(paths*(h//p) for p in PERIODS),
        "evaluation_episodes": len(PERIODS)*len(VARIANTS)*e,
        "native_episodes": sum(episodes), "native_steps": sum(episodes)*h, "native_lower_calls": sum(episodes)*h,
        "native_upper_calls": sum((n-e)*(h//p) for n,p in zip(episodes,PERIODS)),
        "native_donor_response_calls": 2*sum(episodes)*h,
        "planning_renewals": sum(n*(h//p) for n,p in zip(episodes,PERIODS)),
        "planning_fits": sum(n*(h//p-1) for n,p in zip(episodes,PERIODS)),
        "planning_reference_calls": sum(episodes)*h,
        "credit_checks": sum(paths*(1+h//p) for p in PERIODS),
        "upper_actor_optimizer_steps": optimizer, "upper_value_optimizer_steps": optimizer,
        "lower_actor_optimizer_steps": 0, "lower_value_optimizer_steps": 0,
        "score_gradient_batches": sum(4*math.ceil(SCENARIOS*(h//p)/source.source.ppo.MINIBATCH) for p in PERIODS),
        "fisher_jvp_batches": len(METHODS)*sum(chunks), "exact_kl_forward_batches": 2*len(METHODS)*sum(chunks),
        "checkpoint_writes": 0, "native_trace_writes": 0}


def contract():
    return {"source": SOURCE_RUN, "warm_source": WARM_SOURCE_RUN,
        "credit": "sampled_full_return_minus_single_option_replaced_by_current_mean_full_return",
        "baseline": "same_pre_action_prefix_and_fixed_future_innovations_independent_of_current_action_draw",
        "training": "fresh_shared_on_policy_paths_only_counterfactual_paths_are_labels_not_PPO_samples",
        "optimizer": "shared_upper_PPO_original_MC_value_targets_same_optimizer_seed_both_credit_methods",
        "radius": FISHER_RADIUS, "upper_std": .15, "authority": .05,
        "freeze": "all_deployed_lower_critics_teacher_forecaster_and_std",
        "evaluation": "fresh_paired_native_scenes_both_signs_no_winner_or_confirmation_CI",
        "artifacts": "compact_JSON_no_checkpoints_or_traces_full_prefix_replay_cost_counted",
        "limits": "simulator_query_credit_diagnosis_not_joint_HRL_or_stochastic_mean_objective_equivalence"}
