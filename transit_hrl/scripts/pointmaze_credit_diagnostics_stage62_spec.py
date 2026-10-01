"""Freeze archive-only credit and critic diagnostics after the first update."""

from scripts import pointmaze_native_update_stage61_spec as source
from scripts import pointmaze_update_diagnostics_stage58_spec as training_source

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_credit_diagnostics_stage62_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_credit_diagnostics_stage62.py"
POLICY, METHODS = "credit_diagnostics", source.METHODS
PERIODS, TRAIN_POLICIES, roots, arguments = source.PERIODS, source.TRAIN_POLICIES, source.roots, source.arguments
SPLITS = ("training_first_batch", "heldout_deterministic")
TREATMENT, VALUE_BATCH_SIZE, BOUNDARY_WINDOW = "backtracking_kl", 512, 5
SOURCE_PREFLIGHT_RUN = "pointmaze_native_update_stage61_preflight_20261001_r1"
SOURCE_FULL_RUN = "pointmaze_native_update_stage61_full_20261001_r1"


def source_result(root, *, preflight):
    run = SOURCE_PREFLIGHT_RUN if preflight else SOURCE_FULL_RUN
    return ROOT / "results" / run / "cells" / f"replicate_{root}" / "result.json"


def seed_roles(root, *, preflight):
    count = source.deployment.options(preflight=preflight)["rollouts_per_iteration"]
    return {SPLITS[0]: source.deployment.seed_roles(root, preflight=preflight)["training"][:count],
            SPLITS[1]: source.seed_roles(root, preflight=preflight)["evaluation"]}


def options(*, preflight):
    return {"value_batch_size": VALUE_BATCH_SIZE, "boundary_window": BOUNDARY_WINDOW,
            "training_paths": 2 if preflight else 8, "heldout_paths": 2 if preflight else 16}


def budget(*, preflight):
    opt = options(preflight=preflight)
    paths = opt["training_paths"] + opt["heldout_paths"]
    horizon = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    episodes = len(PERIODS) * len(TRAIN_POLICIES) * paths
    upper = len(TRAIN_POLICIES) * paths * sum(horizon // p for p in PERIODS)
    return {"archive_episodes": episodes, "checkpoint_loads": len(PERIODS) * len(TRAIN_POLICIES),
        "feature_lower_rows": episodes * horizon, "feature_upper_rows": upper,
        "lower_value_rows": episodes * horizon, "upper_value_rows": upper,
        "lower_value_forward_batches": episodes * ((horizon + VALUE_BATCH_SIZE - 1) // VALUE_BATCH_SIZE),
        "upper_value_forward_batches": len(TRAIN_POLICIES) * paths
            * sum((horizon // p + VALUE_BATCH_SIZE - 1) // VALUE_BATCH_SIZE for p in PERIODS),
        "gae_calls": 4 * episodes, "mc_return_calls": 3 * episodes,
        "native_steps": 0, "actor_inference_calls": 0, "optimizer_steps": 0,
        "new_fits": 0, "forecaster_loads": 0, "checkpoint_writes": 0}


def contract():
    return {"source_protocol": source.EXPERIMENT_PROTOCOL,
        "source_runs": [SOURCE_PREFLIGHT_RUN, SOURCE_FULL_RUN], "treatment": TREATMENT,
        "periods": list(PERIODS), "arms": list(TRAIN_POLICIES), "splits": list(SPLITS),
        "weights": "actual_Stage61_evaluated_post_first_update_backtracking_checkpoint_no_updates",
        "training": "Stage57_first_batch_used_by_the_shared_first_update_critic",
        "heldout": "Stage61_fresh_deterministic_native_paths_not_used_to_fit_critic",
        "features": "same_causal_history_upper_state_lower_reference_and_value_context_no_actor_or_forecaster_inference",
        "lower_variants": ["option_terminal", "option_trace_continuing_bootstrap", "episode_continuing"],
        "targets": "existing_SMDP_GAE_and_discounted_MC_option_or_episode_returns_upper_charged_discounted_native_returns",
        "summaries": "pooled_samples_within_root_arm_period_split_then_equal_root_descriptive_means_no_performance_gate",
        "selection": "all_roots_both_periods_both_arms_fixed_training_batch_and_all_heldout_paths_no_fit_or_tuning",
        "limits": "post_update_diagnostic_not_original_training_advantage_not_causal_credit_ablation_deterministic_policy_distribution_differs_from_stochastic_training_option_critic_not_consistent_continuing_value"}
