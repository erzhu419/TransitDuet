"""Reduce upper exploration with equal mean-output update RMS."""

import math

from scripts import pointmaze_score_geometry_stage139_spec as source
from scripts import pointmaze_joint_reference_stage121_spec as ppo
from scripts import pointmaze_native_selection_replication_stage135_spec as warm_source

ROOT,ROOTS,PERIODS = source.ROOT,source.ROOTS,source.PERIODS
PROTOCOL = "pointmaze_exploration_match_stage140_v1"
EXPERIMENT_PROTOCOL = PROTOCOL
POLICY = "paired_upper_std_015_005_fresh_option_credit_equal_mean_step_Fisher"
RUNNER_SCRIPT = "scripts/run_pointmaze_exploration_match_stage140.py"
SOURCE_RUN = "pointmaze_score_geometry_stage139_probe_20261008_r1"
WARM_SOURCE_RUN = source.WARM_SOURCE_RUN
WORKERS,SCENARIOS,EVALUATION_EPISODES = 16,8,32
SCALES,STDS = ("original","reduced"),{"original":.15,"reduced":.05}
PANELS,MODES = ("A","B"),source.MODES
VARIANTS = ("source_forecast","warm_start","natural_plus","natural_minus")
CONTRASTS = (("warm_start","source_forecast"),("natural_plus","warm_start"),
    ("natural_minus","warm_start"),("natural_plus","natural_minus"),("natural_plus","source_forecast"))
FISHER_RADIUS,DAMPING,METRICS,arguments = source.FISHER_RADIUS,source.DAMPING,source.METRICS,source.arguments
MEAN_STEP_RMS = STDS["original"]*math.sqrt(2*FISHER_RADIUS/ppo.UPPER_ACTION_DIM)
warm_result = source.warm_result


def source_result(root):
    return ROOT/"results"/SOURCE_RUN/"cells"/f"replicate_{root}"/"result.json"


def radius(scale):
    return FISHER_RADIUS*(STDS["original"]/STDS[scale])**2


def seed_roles(root):
    base = 140100000+ROOTS.index(root)*100000
    return {"training":[{"scenario_seed":base+i+1,"noise_seeds":{"A":base+10001+i,"B":base+20001+i}}
        for i in range(SCENARIOS)],"evaluation":list(range(base+90001,base+90001+EVALUATION_EPISODES))}


def budget():
    h,e,sc = arguments(ROOTS[0]).horizon,EVALUATION_EPISODES,len(SCALES)
    paths = SCENARIOS*len(PANELS)
    actual_eval = 2+sc*(2+len(VARIANTS)-1)
    episodes = [sc*paths*(1+h//p)+actual_eval*e for p in PERIODS]
    chunks = [math.ceil(paths*(h//p)/ppo.MINIBATCH) for p in PERIODS]
    return {"source_cell_loads":2,"lower_checkpoint_loads":len(PERIODS),"upper_checkpoint_loads":len(PERIODS),
        "collection_episodes":len(PERIODS)*sc*paths,"counterfactual_episodes":sc*paths*sum(h//p for p in PERIODS),
        "evaluation_episodes":len(PERIODS)*actual_eval*e,
        "evaluation_alias_assignments":len(PERIODS)*(len(SCALES)*len(MODES)*len(VARIANTS)-actual_eval)*e,
        "native_episodes":sum(episodes),"native_steps":sum(episodes)*h,"native_lower_calls":sum(episodes)*h,
        "native_upper_calls":sum((n-e)*(h//p) for n,p in zip(episodes,PERIODS)),
        "native_donor_response_calls":2*sum(episodes)*h,"planning_reference_calls":sum(episodes)*h,
        "planning_renewals":sum(n*(h//p) for n,p in zip(episodes,PERIODS)),
        "planning_fits":sum(n*(h//p-1) for n,p in zip(episodes,PERIODS)),
        "credit_checks":sum(sc*paths*(1+h//p) for p in PERIODS),
        "upper_actor_optimizer_steps":0,"upper_value_optimizer_steps":0,
        "lower_actor_optimizer_steps":0,"lower_value_optimizer_steps":0,
        "score_gradient_batches":sc*(sum(chunks)+sum(4*math.ceil(paths/2*(h//p)/ppo.MINIBATCH) for p in PERIODS)),
        "mean_score_forward_batches":sc*sum(chunks),"empirical_fisher_solves":sc*len(PERIODS),
        "fisher_jvp_batches":sc*sum(chunks),"exact_kl_forward_batches":2*sc*sum(chunks),
        "policy_geometry_forward_batches":3*sc*len(PERIODS),"upper_candidate_weight_steps":2*sc*len(PERIODS),
        "checkpoint_writes":0,"native_trace_writes":0}


def contract():
    return {"source":SOURCE_RUN,"warm_source":WARM_SOURCE_RUN,"upper_stds":STDS,"mean_step_RMS":MEAN_STEP_RMS,
        "radii":{s:radius(s) for s in SCALES},"damping":DAMPING,"authority":.05,
        "training":"fresh_paired_scenarios_and_noise_folds_fresh_single_option_mean_baseline_credits_per_std",
        "optimizer":"existing_standardized_damped_Fisher_score_no_Adam_or_critic_updates",
        "freeze":"all_lower_critics_teacher_forecaster_authority_and_warm_mean_std_only_varies_as_registered",
        "evaluation":"fresh_paired_mean_and_sampled_upper_both_signs_same_innovations",
        "reuse":"one_forecast_and_one_warm_mean_execution_per_scene_shared_across_scale_mode_tables_aliases_counted",
        "artifacts":"full_native_query_cost_compact_JSON_no_checkpoint_or_trace_writes_inherited_cost_separate",
        "limits":"exploration_and_objective_diagnosis_not_equal_KL_no_evaluation_winner_CI_or_joint_HRL_claim"}
