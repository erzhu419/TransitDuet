"""Separate PPO update scale from its direction on the same warm-start batch."""

import math

from scripts import pointmaze_warm_start_joint_stage136_spec as source

ROOT, ROOTS, PERIODS = source.ROOT, source.ROOTS, source.PERIODS
PROTOCOL = "pointmaze_bounded_ppo_stage137_v1"
EXPERIMENT_PROTOCOL = PROTOCOL
POLICY = "same_Adam_direction_fixed_KL_raw_or_causal26_subspace"
RUNNER_SCRIPT = "scripts/run_pointmaze_bounded_ppo_stage137.py"
SOURCE_RUN = "pointmaze_warm_start_joint_stage136_probe_20261008_r1"
WORKERS, EVALUATION_EPISODES = 8, 32
METHODS = ("bounded_adam", "compact_adam")
VARIANTS = ("source_forecast", "warm_start", "raw_adam",
    *tuple(m + "_" + sign for m in METHODS for sign in ("plus", "minus")))
FISHER_RADIUS = source.source.FISHER_RADIUS
CONTRASTS = (("warm_start", "source_forecast"), ("raw_adam", "warm_start"),
    *tuple((m + "_" + sign, "warm_start") for m in METHODS for sign in ("plus", "minus")),
    ("bounded_adam_plus", "raw_adam"), ("compact_adam_plus", "bounded_adam_plus"),
    *tuple((m + "_plus", m + "_minus") for m in METHODS))
METRICS = ("episode_return", "tracking_squared_error_integral", "reference_correction_rms",
    "reference_correction_peak", "plan_delta_rms", "upper_mean_rms")
arguments = source.arguments


def source_result(root):
    return ROOT / "results" / SOURCE_RUN / "cells" / f"replicate_{root}" / "result.json"


def seed_roles(root):
    base = 137100000 + ROOTS.index(root) * 100000
    return {"replayed_training": source.seed_roles(root)["training"],
        "evaluation": list(range(base + 90001, base + 90001 + EVALUATION_EPISODES))}


def budget():
    h, periods = arguments(ROOTS[0]).horizon, len(PERIODS)
    paths = len(seed_roles(ROOTS[0])["replayed_training"]) * source.NOISE_FOLDS
    e = EVALUATION_EPISODES
    episodes = periods * (paths + len(VARIANTS) * e)
    batches = [math.ceil(paths*(h//p)/source.ppo.MINIBATCH) for p in PERIODS]
    optimizer = source.ppo.EPOCHS * sum(batches)
    return {"source_cell_loads": 2, "lower_checkpoint_loads": periods, "upper_checkpoint_loads": periods,
        "replay_episodes": periods*paths, "evaluation_episodes": periods*len(VARIANTS)*e,
        "native_episodes": episodes, "native_steps": episodes*h, "native_lower_calls": episodes*h,
        "native_upper_calls": sum((paths + (len(VARIANTS)-1)*e)*(h//p) for p in PERIODS),
        "native_donor_response_calls": 2*episodes*h,
        "planning_renewals": sum((paths + len(VARIANTS)*e)*(h//p) for p in PERIODS),
        "planning_fits": sum((paths + len(VARIANTS)*e)*(h//p-1) for p in PERIODS),
        "planning_reference_calls": episodes*h, "credit_checks": periods*paths,
        "upper_actor_optimizer_steps": optimizer, "upper_value_optimizer_steps": optimizer,
        "lower_actor_optimizer_steps": 0, "lower_value_optimizer_steps": 0,
        "fisher_jvp_batches": len(METHODS)*sum(batches), "exact_kl_forward_batches": 2*len(METHODS)*sum(batches),
        "policy_geometry_forward_batches": periods*(1+len(VARIANTS)-1),
        "checkpoint_writes": 0, "native_trace_writes": 0}


def contract():
    return {"source": SOURCE_RUN, "warm_source": source.SOURCE_RUN,
        "training": "exact_Stage136_sampled_paths_and_upper_optimizer_seed_reconstructed_not_new_training",
        "repair": "same_Adam_displacement_at_fixed_KL_and_orthogonal_projection_to_existing_causal26_row_space",
        "radius": FISHER_RADIUS, "signs": ["plus", "minus"], "variants": list(VARIANTS),
        "credit": "unchanged_MC_minus_original_critic_no_LOO_loss_or_new_exploration_setting",
        "freeze": "all_deployed_lower_and_value_weights_teacher_forecaster_std_and_authority",
        "evaluation": "fresh_Stage137_paired_scenes_no_winner_selection_no_confirmation_CI",
        "metrics": list(METRICS), "limits": "scale_and_subspace_diagnosis_not_joint_HRL_or_frequency_confirmation",
        "artifacts": "compact_JSON_no_checkpoint_or_trace_writes_inherited_chain_cost_separate"}
