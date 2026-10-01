"""Fresh native Monte Carlo directions under the frozen Stage78 decoder."""

import math
from scripts import pointmaze_bounded_residual_stage78_spec as source

ROOT, PERIODS, CHUNK_SIZE = source.ROOT, source.PERIODS, source.CHUNK_SIZE
EXPERIMENT_PROTOCOL = "pointmaze_feasible_credit_stage79_v1"
POLICY = "feasible_native_credit"
RUNNER_SCRIPT = "scripts/run_pointmaze_feasible_credit_stage79.py"
roots, arguments = source.roots, source.arguments
FISHER_RADIUS = .001
VARIANTS = ("base", "zero", "upper_plus", "upper_minus", "lower_plus", "lower_minus")
CONTRAST_PAIRS = (("base", "zero"), *tuple((a + "_" + s, b)
    for a in ("upper", "lower") for s, b in (("plus", a + "_minus"), ("plus", "base"),
                                             ("minus", "base"), ("plus", "zero"))))
ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a, b in CONTRAST_PAIRS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (79, 79079)


def options(*, preflight):
    return {"workers": 2 if preflight else 8, "credit_episodes_per_batch": 2 if preflight else 16,
        "evaluation_episodes": 4 if preflight else 32}


def source_result(root):
    return ROOT / "results" / "pointmaze_bounded_residual_stage78_full_20261002_r1" / "cells" / f"replicate_{root}" / "result.json"


def seed_roles(root, *, preflight):
    base = 79_090_000 if preflight else 79_100_000 + roots(preflight=False).index(root) * 10000
    o = options(preflight=preflight)
    return {"credit_A": list(range(base + 1, base + 1 + o["credit_episodes_per_batch"])),
        "credit_B": list(range(base + 1001, base + 1001 + o["credit_episodes_per_batch"])),
        "native_evaluation": list(range(base + 5001, base + 5001 + o["evaluation_episodes"]))}


def budget(*, preflight):
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    o = options(preflight=preflight)
    n, e = 2 * o["credit_episodes_per_batch"], o["evaluation_episodes"]
    count = len(PERIODS) * (n + len(VARIANTS) * e)
    forwards = n * sum(math.ceil(h / CHUNK_SIZE) + math.ceil((h // p) / CHUNK_SIZE) for p in PERIODS)
    fisher = sum(math.ceil(n * h / CHUNK_SIZE) + math.ceil(n * (h // p) / CHUNK_SIZE) for p in PERIODS)
    return {"source_clone_loads": len(PERIODS), "forecaster_loads": 1, "decoder_loads": len(PERIODS),
        "credit_episodes": n * len(PERIODS), "evaluation_episodes": e * len(VARIANTS) * len(PERIODS),
        "native_episodes": count, "native_steps": count * h, "native_lower_calls": count * h,
        "native_upper_calls": (n + e * len(VARIANTS)) * sum(h // p for p in PERIODS),
        "pairing_upper_forward_calls": (n + e * len(VARIANTS)) * sum(h // p for p in PERIODS),
        "native_network_checks": count, "native_pair_checks": e * len(PERIODS),
        "objective_checks": n * len(PERIODS), "mc_calls": 2 * n * len(PERIODS),
        "actor_score_forward_batches": forwards, "actor_score_backward_batches": 3 * forwards,
        "fisher_jvp_batches": fisher, "exact_kl_forward_batches": 2 * fisher,
        "actor_parameter_perturbations": 4 * len(PERIODS), "frozen_model_checks": len(PERIODS)}


def contract():
    return {"source": source.EXPERIMENT_PROTOCOL, "periods": list(PERIODS), "variants": list(VARIANTS),
        "decoder": "exact_saved_full_Stage78_alpha_and_envelope_no_new_scale_search",
        "credit": "fresh_on_policy_undiscounted_native_task_episode_MC_upper_option_rewards_restore_constant_call_cost",
        "baseline": "remaining_primitive_steps_times_opposite_independent_credit_batch_mean_reward_rate",
        "direction": "negative_raw_MC_loss_gradient_all_actor_parameters_no_entropy_or_advantage_normalization",
        "perturbation": "separate_upper_lower_symmetric_Fisher_radius_.001_exact_raw_Gaussian_KL_check_no_radius_sweep",
        "pairing": "same_environment_and_stepwise_lower_noise_same_standardized_upper_noise_not_same_proposals",
        "statistics": "all18_reward_contrasts_equal_root_bootstrap_65536_Bonferroni18_tracking_and_credit_descriptive",
        "decision": "direction_validation_only_no_actor_adoption_Stage67_HOLD_unchanged",
        "artifacts": "server_memory_training_batches_compact_JSON_only_no_trace_or_candidate_checkpoint_writes",
        "limits": "teacher_initialized_fixed_period_diagnostic_not_joint_training_or_frequency_superiority_candidate_command_bound_not_certified"}
