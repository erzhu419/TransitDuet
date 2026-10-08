"""Use cached option labels to isolate high-level update conditioning."""

import math

from scripts import pointmaze_local_option_probe_stage142_spec as source
from scripts import pointmaze_joint_reference_stage121_spec as ppo

ROOT, ROOTS, PERIODS = source.ROOT, source.ROOTS, source.PERIODS
PROTOCOL = "pointmaze_option_conditioning_stage143_v1"
EXPERIMENT_PROTOCOL = PROTOCOL
POLICY = "cached_wide_local_option_gradients_raw392_vs_causal26_fixed_mean_step"
RUNNER_SCRIPT = "scripts/run_pointmaze_option_conditioning_stage143.py"
SOURCE_RUN = "pointmaze_local_option_probe_stage142_probe_20261008_r1"
WIDE_SOURCE_RUN = source.SOURCE_RUN
WORKERS, EVALUATION_EPISODES = 16, 32
PANELS, MODES, PROBES = source.PANELS, source.MODES, source.METHODS
REPRESENTATIONS = ("raw", "compact")
STD, RADIUS, MEAN_STEP_RMS = source.STD, source.RADIUS, source.MEAN_STEP_RMS
METHODS = tuple(p+"_"+r for p in PROBES for r in REPRESENTATIONS)
VARIANTS = ("source_forecast", "warm_start", *tuple(m+"_"+s for m in METHODS for s in ("plus", "minus")))
CONTRASTS = (("warm_start", "source_forecast"),
    *tuple((m+"_"+s, "warm_start") for m in METHODS for s in ("plus", "minus")),
    *tuple((p+"_compact_plus", p+"_raw_plus") for p in PROBES),
    *tuple((m+"_plus", m+"_minus") for m in METHODS),
    *tuple((m+"_plus", "source_forecast") for m in METHODS))
arguments, warm_result = source.arguments, source.warm_result


def source_result(root, *, wide=False):
    run = WIDE_SOURCE_RUN if wide else SOURCE_RUN
    return ROOT/"results"/run/"cells"/f"replicate_{root}"/"result.json"


def seed_roles(root):
    base = 143100000+ROOTS.index(root)*100000
    return {"replayed_training": source.seed_roles(root)["replayed_training"],
        "evaluation": list(range(base+90001, base+90001+EVALUATION_EPISODES))}


def budget():
    h, e = arguments(ROOTS[0]).horizon, EVALUATION_EPISODES
    paths = len(seed_roles(ROOTS[0])["replayed_training"])*len(PANELS)
    per_period = paths+(2*len(VARIANTS)-1)*e
    chunks = sum(math.ceil(paths*(h//p)/ppo.MINIBATCH) for p in PERIODS)
    return {"source_cell_loads": 3, "lower_checkpoint_loads": len(PERIODS), "upper_checkpoint_loads": len(PERIODS),
        "replay_episodes": len(PERIODS)*paths, "collection_episodes": 0, "counterfactual_episodes": 0,
        "evaluation_episodes": len(PERIODS)*(2*len(VARIANTS)-1)*e,
        "evaluation_alias_assignments": len(PERIODS)*e, "credit_checks": len(PERIODS)*paths,
        "native_episodes": len(PERIODS)*per_period, "native_steps": len(PERIODS)*per_period*h,
        "native_lower_calls": len(PERIODS)*per_period*h,
        "native_upper_calls": sum((per_period-e)*(h//p) for p in PERIODS),
        "native_donor_response_calls": 2*len(PERIODS)*per_period*h,
        "planning_reference_calls": len(PERIODS)*per_period*h,
        "planning_renewals": sum(per_period*(h//p) for p in PERIODS),
        "planning_fits": sum(per_period*(h//p-1) for p in PERIODS),
        "empirical_fisher_solves": len(REPRESENTATIONS)*len(PERIODS),
        "fisher_jvp_batches": len(METHODS)*chunks, "exact_kl_forward_batches": 2*len(METHODS)*chunks,
        "policy_geometry_forward_batches": (1+2*len(METHODS))*len(PERIODS),
        "upper_candidate_weight_steps": 2*len(METHODS)*len(PERIODS),
        "upper_actor_optimizer_steps": 0, "upper_value_optimizer_steps": 0,
        "lower_actor_optimizer_steps": 0, "lower_value_optimizer_steps": 0,
        "checkpoint_writes": 0, "native_trace_writes": 0}


def contract():
    return {"source": SOURCE_RUN, "wide_source": WIDE_SOURCE_RUN, "warm_source": source.source.WARM_SOURCE_RUN,
        "training": "one_mean_path_replay_reconstructs_both_archived_probe_gradients_no_new_labels",
        "conditioning": "raw392_or_existing_causal26_projection_fit_before_lifting_update_to_original_actor",
        "geometry": "same_standardized_damped_mean_geometry_damping1_shared_factorization_per_representation",
        "noise_diagnostic": "A_B_masked_label_panels_share_training_design_not_held_out_scene_validation",
        "policy_std": STD, "mean_step_RMS": MEAN_STEP_RMS, "KL_radius": RADIUS,
        "freeze": "warm_actor_baseline_lower_values_teacher_forecaster_authority005",
        "evaluation": "fresh_paired_mean_and_sampled_scenes_both_signs_no_winner_no_CI",
        "artifacts": "compact_JSON_actual_replay_and_evaluation_cost_inherited_queries_separate_no_checkpoints",
        "limits": "update_subspace_diagnosis_not_frequency_filtering_or_joint_HRL_confirmation"}
