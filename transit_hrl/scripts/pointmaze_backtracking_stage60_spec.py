"""Freeze actor-only backtracking against the same Stage57 first-update batch."""

from scripts import pointmaze_first_update_stage59_spec as previous

ROOT, source = previous.ROOT, previous.source
EXPERIMENT_PROTOCOL = "pointmaze_backtracking_stage60_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_backtracking_stage60.py"
POLICY, METHODS = "backtracking", previous.METHODS
PERIODS, TRAIN_POLICIES = previous.PERIODS, previous.TRAIN_POLICIES
roots, options, diagnostic_result = previous.roots, previous.options, previous.diagnostic_result
TREATMENTS = ("plain", "conditional_kl", "backtracking_kl")
KL_BUDGET, BACKTRACK_FACTOR, MAX_BACKTRACKS = .02, .5, 12
NATIVE_PREREQUISITE_TREATMENT = "backtracking_kl"
IDENTITY_CHECKS = (*previous.IDENTITY_CHECKS, "rejection_only_reproduction")
REJECTION_PREFLIGHT_RUN = "pointmaze_first_update_stage59_preflight_20260930_r1"
REJECTION_FULL_RUN = "pointmaze_first_update_stage59_full_20260930_r1"


def rejection_result(root, *, preflight):
    run = REJECTION_PREFLIGHT_RUN if preflight else REJECTION_FULL_RUN
    return ROOT / "results" / run / "cells" / f"replicate_{root}" / "result.json"


def budget(*, preflight):
    result = previous.budget(preflight=preflight)
    for key in ("diagnostic_updates", "diagnostic_distribution_passes", "diagnostic_value_passes", "diagnostic_gae_calls"):
        result[key] = result[key] // len(previous.TREATMENTS) * len(TREATMENTS)
    return result


def contract():
    return {**previous.contract(), "treatments": list(TREATMENTS),
        "backtrack_factor": BACKTRACK_FACTOR, "max_backtracks": MAX_BACKTRACKS,
        "intervention": "one_Adam_proposal_per_minibatch_first_feasible_parameter_displacement_scale_restore_actor_and_Adam_between_rejected_trials",
        "Adam_semantics": "accepted_scaled_proposal_keeps_single_gradient_moment_update_and_original_LR_all_rejected_restores_original_actor_and_Adam",
        "identity": "plain_exact_Stage58_rejection_only_exact_Stage59_all_three_final_critics_and_Adam_exact",
        "decision": "nonzero_backtracking_actor_prerequisite_only_no_automatic_native_launch",
        "guard_cost": "count_all_candidate_distributions_interpolation_trials_snapshots_and_rollback_checks_not_as_extra_optimizer_steps"}
