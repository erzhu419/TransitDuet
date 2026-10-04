"""Decision-level upper credit above the frozen Stage112 lower branch."""
from scripts import pointmaze_upper_residual_train_stage113_spec as base

ROOT, PERIODS = base.ROOT, base.PERIODS
EXPERIMENT_PROTOCOL = "pointmaze_upper_local_credit_train_stage114_v1"
POLICY = "forecast_anchored_upper_residual_local_option_credit"
RUNNER_SCRIPT = "scripts/run_pointmaze_upper_local_credit_train_stage114.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_upper_local_credit_train_stage114.py"
SOURCE_RUN, LOWER_RUN = base.SOURCE_RUN, base.LOWER_RUN
ARMS, CONTRASTS, ENDPOINTS = base.ARMS, base.CONTRASTS, base.ENDPOINTS
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, CHUNK_SIZE = 65536, (114, 114114), base.CHUNK_SIZE


def roots(*, preflight):
    return base.roots(preflight=preflight)


def arguments(root, *, preflight):
    return base.arguments(root, preflight=preflight)


def options(*, preflight):
    return base.options(preflight=preflight)


def source_result(root):
    return base.source_result(root)


def lower_checkpoint(root, period):
    return base.lower_checkpoint(root, period)


def seed_roles(root, *, preflight):
    roots(preflight=preflight).index(root)
    base_seed = 114000000 if preflight else 114100000 + roots(preflight=False).index(root) * 100000
    o = options(preflight=preflight)
    rounds = [{name: [{"scenario_seed": base_seed + 10000 * j + offset + i,
        "noise_seeds": [base_seed + 10000 * j + offset + 2001 + 2 * i,
            base_seed + 10000 * j + offset + 2002 + 2 * i]}
        for i in range(o["credit_scenarios_per_batch"])]
        for name, offset in (("credit_A", 1), ("credit_B", 1001))}
        for j in range(o["updates"])]
    return {"training_rounds": rounds,
        "native_evaluation": list(range(base_seed + 95001,
            base_seed + 95001 + o["evaluation_episodes"]))}


def budget(*, preflight):
    return base.budget(preflight=preflight)


def contract():
    return {"source": SOURCE_RUN, "lower_source": LOWER_RUN,
        "architecture": "forecast_anchored_Bernstein_residual_upper_readout390_to4_zero_initialized",
        "training": "decision_aligned_local_option_return_updates_only_lower_branch_critic_std_and_upper_base_frozen",
        "representation": "alpha1_forecast_base_plus_upper_residual_no_calibration_shrinkage",
        "credit": "per_upper_decision_undiscounted_option_return_leave_other_out_over_paired_noise",
        "arms": list(ARMS), "primary_endpoints": list(ENDPOINTS),
        "statistics": "all4_equal_root_bootstrap65536_Bonferroni4_seed114_114114_no_selection",
        "decision": "positive_CI_for_learned_minus_forecast_and_learned_minus_learned_blinded_both_periods",
        "freeze": "Stage112_learned_lower_branch_Stage111_source_upper_base_upper_std_values_and_critic",
        "artifacts": "server_final_upper_weights_only_compact_JSON_no_native_trace_pull",
        "limits": "upper_branch_only_not_full_joint_actor_critic_until_gate"}
