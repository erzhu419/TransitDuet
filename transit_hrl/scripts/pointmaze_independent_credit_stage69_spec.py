"""Independent frozen-policy credit confirmation, not a replacement adoption gate."""

import math
from scripts import pointmaze_credit_reliability_stage68_spec as previous
from scripts import pointmaze_matched_upper_stage57_spec as native

ROOT = previous.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_independent_credit_stage69_v1"
POLICY = "independent_credit"
RUNNER_SCRIPT = "scripts/run_pointmaze_independent_credit_stage69.py"
PERIODS, TRAIN_POLICIES, TREATMENTS = previous.PERIODS, previous.TRAIN_POLICIES, previous.TREATMENTS
roots, arguments, training_result, source_result = previous.roots, previous.arguments, previous.training_result, previous.source_result


def options(*, preflight):
    return {"batches": 2 if preflight else 4, "episodes_per_batch": 2 if preflight else 8,
        "workers": 2 if preflight else 8}


def seed_roles(root, *, preflight):
    opt = options(preflight=preflight)
    base = 26_690_000 if preflight else 26_700_000 + roots(preflight=preflight).index(root) * 10000
    fresh = [list(range(base + i * 100 + 1, base + i * 100 + 1 + opt["episodes_per_batch"]))
        for i in range(opt["batches"])]
    old = previous.seed_roles(root, preflight=preflight)
    return {"anchor": old["first_training_probe"], "fresh_batches": fresh}


def native_budget(*, preflight):
    opt = options(preflight=preflight)
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    n = len(TRAIN_POLICIES) * opt["batches"] * opt["episodes_per_batch"]
    calls = n * sum(h // p for p in PERIODS)
    fits = n * sum(h // p - 1 for p in PERIODS)
    steps = len(PERIODS) * n * h
    return {"primitive_steps": steps, "upper_inference_calls": calls, "lower_inference_calls": steps,
        "gate_inference_calls": 0, "plan_ols_fits": fits, "audit_ols_fits": fits,
        "plan_ridge_predictions": fits, "audit_ridge_predictions": fits,
        "reference_evaluations": steps, "actor_context_evaluations": steps, "upper_plan_decodes": calls,
        "bernstein_basis_evaluations": n * sum(p + 1 for p in PERIODS),
        "audit_bernstein_basis_evaluations": n * sum(p + 1 for p in PERIODS)}


def budget(*, preflight):
    opt = options(preflight=preflight)
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    old = len(seed_roles(roots(preflight=preflight)[0], preflight=preflight)["anchor"])
    fresh = opt["batches"] * opt["episodes_per_batch"]
    cases = len(PERIODS) * len(TRAIN_POLICIES)
    forwards = cases * (old + fresh) * math.ceil(h / previous.CHUNK_SIZE)
    return {"source_clone_loads": len(PERIODS), "forecaster_loads": 1, "critic_checkpoint_loads": 2 * cases,
        "archive_episodes": cases * old, "reconstructed_lower_calls": cases * old * h,
        "reconstructed_upper_calls": len(TRAIN_POLICIES) * old * sum(h // p for p in PERIODS),
        "archive_network_checks": cases * old, "native_episodes": cases * fresh,
        "native_trace_audits": cases * fresh, "native_network_checks": cases * fresh,
        "probe_value_rows": 2 * cases * (old + fresh) * h, "mc_calls": cases * (opt["batches"] + 1),
        "gae_calls": 2 * cases * (opt["batches"] + 1), "td_identity_checks": 2 * cases * (opt["batches"] + 1),
        "source_probe_checks": 2 * cases, "actor_score_forward_batches": forwards,
        "actor_score_backward_batches": 5 * forwards, "frozen_model_checks": len(PERIODS) + 2 * cases}


def contract():
    return {"source": previous.source.EXPERIMENT_PROTOCOL, "treatments": list(TREATMENTS),
        "sampling": "frozen_Stage55_clones_saved_forecaster_Stage57_training_noise_no_policy_updates",
        "roster": "same_eight_development_roots_period50_100_zero_train_joint_ppo_four_independent_batches_of_eight",
        "pairing": "same_fresh_reset_action_seeds_across_cases_both_critics_on_identical_paths_within_case",
        "common_baseline": "discounted_remaining_mass_times_Stage67_first_calibration_rate_location_time_only",
        "reference": "raw_uncentered_unscaled_MC_minus_common_baseline_uniform_time_discounted_return_convention",
        "directions": "each_batch_own_PPO_center_scale_GAE_separate_entropy_adjusted_directions_vs_disjoint_MC_batches",
        "noise": "independent_raw_episode_gradient_sample_covariance_unbiased_signal_power_no_zero_clamp",
        "TD": "true_episode_done_GAE_minus_MC_advantage_equals_filtered_future_value_error_no_lambda_sweep",
        "frozen": "all_networks_and_Adam_unchanged_no_fits_optimizer_steps_or_checkpoint_writes",
        "decision": "diagnostic_only_Stage67_HOLD_unchanged_no_native_reward_adoption_trial",
        "limits": "finite_MC_reference_reused_teacher_initialized_development_roots_not_true_gradient_or_frequency_proof"}
