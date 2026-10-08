"""Exclude whole scenes when fitting cached native option gradients."""

import math

from scripts import pointmaze_option_conditioning_stage143_spec as source

ROOT, ROOTS, PERIODS = source.ROOT, source.ROOTS, source.PERIODS
PROTOCOL = "pointmaze_option_crossfit_stage144_v1"
EXPERIMENT_PROTOCOL = PROTOCOL
POLICY = "leave_one_scene_out_both_noise_panels_cached_option_gradients"
RUNNER_SCRIPT = "scripts/run_pointmaze_option_crossfit_stage144.py"
SOURCE_RUN = "pointmaze_option_conditioning_stage143_probe_20261008_r1"
WORKERS = 16
PROBES, PANELS, METHODS = source.PROBES, source.PANELS, source.METHODS
VARIANTS = tuple(m+"_"+s for m in METHODS for s in ("plus", "minus"))
CONTRASTS = tuple((v, "warm_start") for v in VARIANTS)+tuple(
    (p+"_compact_plus", p+"_raw_plus") for p in PROBES)
arguments = source.arguments


def seed_roles(root):
    training = source.seed_roles(root)["replayed_training"]
    return {"replayed_training": training, "held_out_scene_order": [r["scenario_seed"] for r in training]}


def source_result(root):
    return ROOT/"results"/SOURCE_RUN/"cells"/f"replicate_{root}"/"result.json"


def budget():
    h = arguments(ROOTS[0]).horizon
    scenes = len(seed_roles(ROOTS[0])["replayed_training"])
    paths = scenes*len(PANELS)
    per_period = paths*(1+len(VARIANTS))
    chunks = sum(math.ceil((paths-len(PANELS))*(h//p)/source.ppo.MINIBATCH) for p in PERIODS)*scenes
    return {"source_cell_loads": 4, "lower_checkpoint_loads": len(PERIODS), "upper_checkpoint_loads": len(PERIODS),
        "replay_episodes": paths*len(PERIODS), "collection_episodes": 0, "counterfactual_episodes": 0,
        "evaluation_episodes": paths*len(VARIANTS)*len(PERIODS), "credit_checks": paths*len(PERIODS),
        "native_episodes": per_period*len(PERIODS), "native_steps": per_period*len(PERIODS)*h,
        "native_lower_calls": per_period*len(PERIODS)*h,
        "native_upper_calls": sum(per_period*(h//p) for p in PERIODS),
        "native_donor_response_calls": 2*per_period*len(PERIODS)*h,
        "planning_reference_calls": per_period*len(PERIODS)*h,
        "planning_renewals": sum(per_period*(h//p) for p in PERIODS),
        "planning_fits": sum(per_period*(h//p-1) for p in PERIODS),
        "empirical_fisher_solves": scenes*len(PERIODS)*len(source.REPRESENTATIONS),
        "fisher_jvp_batches": len(METHODS)*chunks, "exact_kl_forward_batches": 2*len(METHODS)*chunks,
        "policy_geometry_forward_batches": (1+2*len(METHODS))*scenes*len(PERIODS),
        "held_out_geometry_forward_batches": (1+len(VARIANTS))*paths*len(PERIODS),
        "upper_candidate_weight_steps": len(VARIANTS)*scenes*len(PERIODS),
        "upper_actor_optimizer_steps": 0, "upper_value_optimizer_steps": 0,
        "lower_actor_optimizer_steps": 0, "lower_value_optimizer_steps": 0,
        "checkpoint_writes": 0, "native_trace_writes": 0}


def contract():
    return {"source": SOURCE_RUN, "query_sources": [source.SOURCE_RUN, source.WIDE_SOURCE_RUN],
        "split": "leave_one_whole_scene_out_exclude_both_A_B_noise_panels_before_fitting_and_geometry",
        "conditioning": source.contract()["conditioning"],
        "policy_std": source.STD, "mean_step_RMS": source.MEAN_STEP_RMS, "KL_radius": source.RADIUS,
        "freeze": source.contract()["freeze"],
        "evaluation": "native_mean_control_on_excluded_scene_both_noise_panels_both_signs_no_selection",
        "diagnostics": "held_out_linearized_query_gain_and_actual_return_and_output_RMS_separate",
        "artifacts": "compact_JSON_replay_and_held_out_control_cost_no_new_queries_or_checkpoints",
        "limits": "crossfit_of_archived_training_scenes_not_new_root_confirmation_or_joint_HRL"}
