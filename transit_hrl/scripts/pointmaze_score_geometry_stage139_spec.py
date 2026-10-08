"""Isolate optimizer geometry and sampled-versus-mean upper deployment."""

import math

from scripts import pointmaze_option_credit_stage138_spec as source
from scripts.pointmaze_native_geometry_stage126_spec import DAMPING

ROOT, ROOTS, PERIODS = source.ROOT, source.ROOTS, source.PERIODS
PROTOCOL = "pointmaze_score_geometry_stage139_v1"
EXPERIMENT_PROTOCOL = PROTOCOL
POLICY = "cached_option_credit_Adam_score_damped_Fisher_mean_or_sampled"
RUNNER_SCRIPT = "scripts/run_pointmaze_score_geometry_stage139.py"
SOURCE_RUN = "pointmaze_option_credit_stage138_probe_20261008_r1"
WARM_SOURCE_RUN = source.WARM_SOURCE_RUN
WORKERS, EVALUATION_EPISODES = 16, 32
METHODS, MODES = ("adam", "score", "natural_score"), ("mean", "sampled")
VARIANTS = ("source_forecast", "warm_start",
    *tuple(m+"_"+sign for m in METHODS for sign in ("plus", "minus")))
CONTRASTS = (("warm_start", "source_forecast"),
    *tuple((m+"_"+s,"warm_start") for m in METHODS for s in ("plus","minus")),
    ("score_plus","adam_plus"),("natural_score_plus","adam_plus"),("natural_score_plus","score_plus"),
    *tuple((m+"_plus",m+"_minus") for m in METHODS))
FISHER_RADIUS, METRICS, arguments = source.FISHER_RADIUS, source.METRICS, source.arguments


def source_result(root):
    return ROOT/"results"/SOURCE_RUN/"cells"/f"replicate_{root}"/"result.json"


warm_result = source.warm_result


def seed_roles(root):
    base = 139100000+ROOTS.index(root)*100000
    return {"replayed_training": source.seed_roles(root)["training"],
        "evaluation": list(range(base+90001,base+90001+EVALUATION_EPISODES))}


def budget():
    h,e = arguments(ROOTS[0]).horizon,EVALUATION_EPISODES
    paths = len(seed_roles(ROOTS[0])["replayed_training"])*len(source.PANELS)
    actual_variants = 1+len(MODES)*(len(VARIANTS)-1)
    episodes = len(PERIODS)*(paths+actual_variants*e)
    chunks = [math.ceil(paths*(h//p)/source.source.source.ppo.MINIBATCH) for p in PERIODS]
    return {"source_cell_loads":2,"lower_checkpoint_loads":len(PERIODS),"upper_checkpoint_loads":len(PERIODS),
        "replay_episodes":len(PERIODS)*paths,"counterfactual_episodes":0,
        "evaluation_episodes":len(PERIODS)*actual_variants*e,"evaluation_alias_assignments":len(PERIODS)*e,
        "native_episodes":episodes,"native_steps":episodes*h,"native_lower_calls":episodes*h,
        "native_upper_calls":sum((paths+(actual_variants-1)*e)*(h//p) for p in PERIODS),
        "native_donor_response_calls":2*episodes*h,"planning_reference_calls":episodes*h,
        "planning_renewals":sum((paths+actual_variants*e)*(h//p) for p in PERIODS),
        "planning_fits":sum((paths+actual_variants*e)*(h//p-1) for p in PERIODS),"credit_checks":len(PERIODS)*paths,
        "upper_actor_optimizer_steps":source.source.source.ppo.EPOCHS*sum(chunks),
        "upper_value_optimizer_steps":source.source.source.ppo.EPOCHS*sum(chunks),
        "lower_actor_optimizer_steps":0,"lower_value_optimizer_steps":0,
        "score_gradient_batches":sum(chunks)+sum(4*math.ceil(paths/2*(h//p)/source.source.source.ppo.MINIBATCH) for p in PERIODS),
        "mean_score_forward_batches":sum(chunks),"empirical_fisher_solves":len(PERIODS),
        "fisher_jvp_batches":len(METHODS)*sum(chunks),"exact_kl_forward_batches":2*len(METHODS)*sum(chunks),
        "upper_candidate_weight_steps":len(PERIODS)*len(METHODS)*2,"checkpoint_writes":0,"native_trace_writes":0}


def contract():
    return {"source":SOURCE_RUN,"warm_source":WARM_SOURCE_RUN,
        "training":"exact_Stage138_sampled_state_and_cached_single_option_credit_replay_not_new_labels",
        "geometry":"same_option_credit_Adam_displacement_or_anchor_score_or_existing_standardized_damped_Fisher_mean",
        "radius":FISHER_RADIUS,"damping":DAMPING,"upper_std":.15,"authority":.05,
        "freeze":"all_deployed_lower_critics_teacher_forecaster_std_and_authority",
        "evaluation":"fresh_paired_mean_and_sampled_upper_same_upper_lower_innovations_both_signs",
        "forecast":"one_native_forecast_execution_shared_by_both_upper_modes_cost_and_alias_counted",
        "artifacts":"compact_JSON_no_checkpoint_or_trace_writes_inherited_label_cost_not_zero",
        "limits":"optimizer_objective_diagnosis_no_evaluation_winner_confirmation_CI_or_joint_HRL_claim"}
