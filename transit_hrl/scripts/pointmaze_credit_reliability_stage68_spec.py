"""Read-only episode-split reliability after the failed Stage67 credit gate."""

import math
from scripts import pointmaze_horizon_value_stage67_spec as source

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_credit_reliability_stage68_v1"
POLICY = "credit_reliability"
RUNNER_SCRIPT = "scripts/run_pointmaze_credit_reliability_stage68.py"
PERIODS, TRAIN_POLICIES, TREATMENTS = source.PERIODS, source.TRAIN_POLICIES, source.TREATMENTS
roots, arguments, seed_roles, training_result = source.roots, source.arguments, source.seed_roles, source.training_result
CHUNK_SIZE, WORKERS = 1024, 2


def source_result(root, *, preflight):
    run = "pointmaze_horizon_value_stage67_" + ("preflight" if preflight else "full") + "_20261001_r1"
    return ROOT / "results" / run / "cells" / f"replicate_{root}" / "result.json"


def budget(*, preflight):
    episodes = source.options(preflight=preflight)["rollouts_per_iteration"]
    horizon = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    cases, fits = len(PERIODS) * len(TRAIN_POLICIES), len(PERIODS) * len(TRAIN_POLICIES) * len(TREATMENTS)
    forwards = fits * episodes * math.ceil(horizon / CHUNK_SIZE)
    return {"archive_episodes": cases * episodes, "reconstructed_lower_calls": cases * episodes * horizon,
        "reconstructed_upper_calls": len(TRAIN_POLICIES) * episodes * sum(horizon // p for p in PERIODS),
        "archive_network_checks": cases * episodes, "source_clone_loads": len(PERIODS), "forecaster_loads": 1,
        "critic_checkpoint_loads": fits, "probe_value_rows": fits * episodes * horizon,
        "mc_calls": cases, "gae_calls": fits, "source_probe_checks": fits, "source_gradient_checks": fits,
        "actor_score_forward_batches": forwards, "actor_score_backward_batches": 4 * forwards,
        "balanced_partitions": cases * math.comb(episodes - 1, episodes // 2 - 1), "frozen_model_checks": fits}


def contract():
    return {"source": source.EXPERIMENT_PROTOCOL, "treatments": list(TREATMENTS),
        "batch": "same_disjoint_Stage57_first_training_archive_reproduces_Stage67_values_and_full_credit_gradients",
        "gradients": "per_episode_raw_GAE_MC_and_constant_score_then_exact_fold_center_and_scale",
        "linearity": "preupdate_ratios_inside_PPO_clip_interval_required_for_fold_reconstruction",
        "partitions": "all_unordered_balanced_episode_halves_35_full_1_preflight_no_chosen_split",
        "comparisons": "within_GAE_within_MC_disjoint_half_GAE_MC_and_both_critics_against_each_same_MC_reference",
        "parts": ["all", "mean", "log_std"], "entropy": "unchanged_recorded_separately",
        "statistics": "descriptive_cosines_and_positive_fractions_partitions_dependent_not_independent_CI_samples",
        "frozen": "all_networks_and_Adam_bitexact_no_sampling_fitting_or_optimizer_steps",
        "selection": "same_eight_roots_both_periods_both_arms_no_lambda_LR_seed_or_gate_sweep",
        "decision": "diagnosis_only_Stage67_native_HOLD_not_replaced_or_released_by_reliability",
        "limits": "finite_archive_gradient_stability_not_true_policy_gradient_or_reward_improvement"}
