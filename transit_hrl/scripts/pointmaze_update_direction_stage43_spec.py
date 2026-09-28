"""Read-only diagnosis of the Stage-42 first lower PPO update."""

import numpy as np
from scripts import pointmaze_critic_clock_stage42_spec as source

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_update_direction_stage43_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_update_direction_stage43.py"
POLICY = "first_update"
METHODS = source.METHODS[1:]
POLICIES = ("frozen", *METHODS)
MODES, METRICS = source.MODES, source.METRICS
SOURCE_FULL_RUN = "pointmaze_critic_clock_stage42_v1_full_20260928_r1"
SOURCE_PREFLIGHT_RUN = "pointmaze_critic_clock_stage42_v1_preflight_20260928_r1"
ENDPOINTS = tuple(f"{m}:{k}" for m in METHODS for k in
                  ("training_surrogate_gain", "heldout_task_direction", "deterministic_return", "lower_sampled_return"))
CI_FAMILY_SIZE = len(ENDPOINTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (43, 43043)
roots, options = source.roots, source.options


def source_result(root, method, *, preflight):
    run = SOURCE_PREFLIGHT_RUN if preflight else SOURCE_FULL_RUN
    return ROOT / "results" / run / "cells" / method / f"replicate_{root}" / "result.json"


def seed_roles(root, *, preflight):
    roster, opt = roots(preflight=preflight), options(preflight=preflight)
    index = roster.index(root)
    base = 10_790_000 if preflight else 10_800_000 + index * 10000
    offset = opt["warmup_iterations"] * opt["rollouts_per_iteration"]
    return {"reconstruction": source.seed_roles(root, preflight=preflight)["training"][offset:offset + opt["rollouts_per_iteration"]],
            "evaluation": list(range(base + 3001, base + 3001 + opt["evaluation_paths"]))}


def policy_seed(root, seed):
    return int(np.random.SeedSequence([43, root, seed, 43017]).generate_state(1)[0])


def rollout_arguments(root, seed, *, mode, reference):
    sampled = mode == "lower_sampled"
    return {"sample": reference and sampled, "upper_sample": False, "gate_sample": False,
            "lower_sample": sampled, "gate_seed": None,
            "lower_seed": int(np.random.SeedSequence([43, root, seed, 43019]).generate_state(1)[0]) if sampled else None}


def budget(*, preflight):
    opt = options(preflight=preflight)
    horizon = source.source.arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    reconstruction = len(METHODS) * opt["rollouts_per_iteration"] * horizon
    evaluation = len(POLICIES) * len(MODES) * opt["evaluation_paths"] * horizon
    return {"reconstruction_primitive_steps": reconstruction, "evaluation_primitive_steps": evaluation,
            "total_primitive_steps": reconstruction + evaluation, "optimizer_steps": 0}


def contract():
    return {"source_protocol": source.EXPERIMENT_PROTOCOL, "methods": list(METHODS),
            "checkpoints": "fixed_last_warmup_and_first_lower_update_no_selection",
            "training_reconstruction": "original_first_learning_batch_seeds_sampling_credit_values_and_GAE",
            "surrogate": "original_normalized_GAE_clipped_PPO_full_batch_after_minus_before",
            "task_direction": "heldout_undiscounted_full_episode_score_gradient_dot_actual_parameter_displacement",
            "score_baseline": "leave_one_episode_out_mean_return_to_go_at_each_primitive_time_no_option_cuts",
            "evaluation": "fresh_paired_paths_deterministic_upper_gate_deterministic_or_sampled_lower",
            "baseline": "shared_pre_update_actor_after_exact_actor_and_frozen_level_equality",
            "optimizer_steps": 0, "root_exclusion": "forbidden", "checkpoint_selection": "none",
            "sequential_extension": "forbidden", "primary_endpoints": list(ENDPOINTS),
            "bootstrap_seed": list(BOOTSTRAP_SEED), "bootstrap_draws": BOOTSTRAP_DRAWS,
            "interval": "two_sided_percentile_bonferroni_16_paired_equal_root_means",
            "decision": "positive_training_gain_and_negative_task_direction_local_conflict_negative_sampled_return_finite_update_conflict",
            "evidence_role": "conditional_diagnosis_existing_development_weights_with_fresh_evaluation_not_confirmation"}
