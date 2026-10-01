"""Historical directions, symmetric cloned-policy probes on fresh native paths."""

import math
from scripts import pointmaze_objective_audit_stage72_spec as source

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_native_direction_stage73_v1"
POLICY = "native_direction"
RUNNER_SCRIPT = "scripts/run_pointmaze_native_direction_stage73.py"
PERIODS, TRAIN_POLICIES = source.PERIODS, source.TRAIN_POLICIES
roots, arguments = source.roots, source.arguments
DIRECTIONS = ("gae_control", "gae_factored", "native_mc")
VARIANTS = ("base", *(f"{d}_{sign}" for d in DIRECTIONS for sign in ("plus", "minus")))
FISHER_RADIUS, CHUNK_SIZE = .001, 1024
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (73, 73073)
ENDPOINTS = tuple(f"{p}/{a}/{d}/{contrast}" for p in PERIODS for a in TRAIN_POLICIES for d in DIRECTIONS
    for contrast in ("plus_minus", "plus_base", "minus_base"))


def options(*, preflight):
    old = source.source.options(preflight=preflight)
    return {"calibration_batches": old["calibration_batches"],
        "calibration_episodes_per_batch": old["calibration_episodes_per_batch"],
        "evaluation_episodes": 4 if preflight else 32, "workers": 2 if preflight else 8}


def seed_roles(root, *, preflight):
    old = source.source.seed_roles(root, preflight=preflight)
    base = 73_090_000 if preflight else 73_100_000 + roots(preflight=False).index(root) * 10000
    seeds = list(range(base + 5001, base + 5001 + options(preflight=preflight)["evaluation_episodes"]))
    if set(seeds).intersection([*old["calibration"], *(s for b in old["archive_batches"] for s in b)]):
        raise ValueError("Stage73 native probes overlap historical fitting or Stage69 probes")
    return {"calibration": old["calibration"], "native_evaluation": seeds}


def source_result(root, *, preflight):
    run = "pointmaze_objective_audit_stage72_" + ("preflight" if preflight else "full") + "_20261001_r1"
    return ROOT / "results" / run / "cells" / f"replicate_{root}" / "result.json"


def budget(*, preflight):
    opt = options(preflight=preflight)
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    cases, directions = len(PERIODS) * len(TRAIN_POLICIES), len(DIRECTIONS)
    n = opt["calibration_batches"] * opt["calibration_episodes_per_batch"]
    e = opt["evaluation_episodes"] * len(VARIANTS)
    forward = cases * n * math.ceil(h / CHUNK_SIZE)
    fisher = cases * directions * math.ceil(n * h / CHUNK_SIZE)
    return {"calibration_archive_episodes": cases * n, "archive_network_checks": cases * n,
        "reconstructed_lower_calls": cases * n * h,
        "reconstructed_upper_calls": len(TRAIN_POLICIES) * n * sum(h // p for p in PERIODS),
        "source_clone_loads": len(PERIODS), "forecaster_loads": 1, "critic_checkpoint_loads": 2 * cases,
        "historical_reward_rate_fits": cases, "historical_direction_fits": cases * directions,
        "calibration_value_rows": 2 * cases * n * h, "mc_calls": cases * opt["calibration_batches"],
        "gae_calls": 2 * cases * opt["calibration_batches"],
        "actor_score_forward_batches": forward, "actor_score_backward_batches": 5 * forward,
        "fisher_jvp_batches": fisher, "exact_kl_forward_batches": 2 * fisher,
        "calibration_radius_checks": cases * directions, "actor_parameter_perturbations": 2 * cases * directions,
        "native_episodes": cases * e, "native_steps": cases * e * h,
        "native_lower_calls": cases * e * h,
        "native_upper_calls": len(TRAIN_POLICIES) * e * sum(h // p for p in PERIODS),
        "native_network_checks": cases * e, "native_pair_checks": cases * opt["evaluation_episodes"],
        "frozen_model_checks": len(PERIODS) + 2 * cases}


def contract():
    return {"source": source.EXPERIMENT_PROTOCOL, "directions": list(DIRECTIONS), "variants": list(VARIANTS),
        "fitting": "all_Stage57_historical_warmup_only_no_Stage69_or_fresh_native_labels",
        "gae_directions": "mean_of_separately_PPO_normalized_warmup_batch_loss_gradients_plus_unchanged_entropy_coefficient",
        "native_direction": "raw_undiscounted_MC_episode_loss_gradient_first_historical_reward_rate_time_baseline_no_entropy",
        "perturbation": "negative_L2_normalized_loss_gradient_all_lower_actor_parameters_same_plus_minus_step_sqrt(2_delta/F)",
        "fisher_radius": FISHER_RADIUS, "fisher": "conditional_diagonal_Gaussian_Fisher_via_functional_JVP_including_native_std_clamp",
        "radius_check": "historical_average_of_exact_plus_minus_old_to_new_KL_between_half_and_twice_nominal_delta_no_radius_sweep",
        "native_pairing": "fresh_environment_seeds_same_stepwise_lower_noise_initial_policy_rng_and_fixed_upper_proposals_all_seven_variants",
        "execution": "same_Stage57_training_distribution_both_actors_sampled_fixed50_100_zero_normal_execution",
        "statistics": "equal_root_paired_reward_means_percentile_root_bootstrap_two_sided_Bonferroni36_all_endpoints",
        "decision": "forward_response_audit_only_no_optimizer_critic_or_forecaster_fit_no_policy_adoption_Stage67_HOLD_unchanged",
        "artifacts": "no_native_raw_trace_or_candidate_checkpoint_writes_compact_JSON_only",
        "limits": "finite_radius_response_not_infinitesimal_gradient_teacher_initialized_development_roots_not_full_training_generalization_or_frequency_claim"}
