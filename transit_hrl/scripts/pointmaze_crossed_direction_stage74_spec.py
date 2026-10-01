"""Cross frozen historical directions with native upper execution."""

from scripts import pointmaze_native_direction_stage73_spec as source

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_crossed_direction_stage74_v1"
POLICY = "crossed_direction"
RUNNER_SCRIPT = "scripts/run_pointmaze_crossed_direction_stage74.py"
PERIODS, DIRECTIONS = source.PERIODS, source.DIRECTIONS
FIT_ARMS = EXECUTION_ARMS = source.TRAIN_POLICIES
roots, arguments, options = source.roots, source.arguments, source.options
FISHER_RADIUS, CHUNK_SIZE = source.FISHER_RADIUS, source.CHUNK_SIZE
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (74, 74074)


def variant(fit, direction, sign):
    return f"{fit}:{direction}:{sign}"


VARIANTS = ("base", *(variant(f, d, s) for f in FIT_ARMS for d in DIRECTIONS for s in ("plus", "minus")))
ENDPOINTS = (
    *(f"cell/{p}/{f}/{e}/{d}/{c}" for p in PERIODS for f in FIT_ARMS for e in EXECUTION_ARMS
      for d in DIRECTIONS for c in ("plus_minus", "plus_base", "minus_base")),
    *(f"execution/{p}/{f}/{d}" for p in PERIODS for f in FIT_ARMS for d in DIRECTIONS),
    *(f"fitting/{p}/{e}/{d}" for p in PERIODS for e in EXECUTION_ARMS for d in DIRECTIONS),
    *(f"interaction/{p}/{d}" for p in PERIODS for d in DIRECTIONS))


def seed_roles(root, *, preflight):
    old = source.seed_roles(root, preflight=preflight)
    base = 74_090_000 if preflight else 74_100_000 + roots(preflight=False).index(root) * 10000
    seeds = list(range(base + 5001, base + 5001 + options(preflight=preflight)["evaluation_episodes"]))
    if set(seeds).intersection([*old["calibration"], *old["native_evaluation"]]):
        raise ValueError("Stage74 probes overlap historical fitting or Stage73 evaluation")
    return {"calibration": old["calibration"], "native_evaluation": seeds}


def source_result(root, *, preflight):
    run = "pointmaze_native_direction_stage73_" + ("preflight" if preflight else "full") + "_20261001_r1"
    return ROOT / "results" / run / "cells" / f"replicate_{root}" / "result.json"


def budget(*, preflight):
    b = source.budget(preflight=preflight)
    n = options(preflight=preflight)["evaluation_episodes"]
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    count = len(PERIODS) * len(EXECUTION_ARMS) * len(VARIANTS) * n
    b.update(native_episodes=count, native_steps=count * h, native_lower_calls=count * h,
        native_upper_calls=len(EXECUTION_ARMS) * len(VARIANTS) * n * sum(h // p for p in PERIODS),
        native_network_checks=count, cross_execution_pair_checks=len(PERIODS) * n,
        stage73_direction_reproductions=len(PERIODS) * len(FIT_ARMS) * len(DIRECTIONS))
    return b


def contract():
    return {**source.contract(), "source": source.EXPERIMENT_PROTOCOL, "variants": list(VARIANTS),
        "fitting_arms": list(FIT_ARMS), "execution_arms": list(EXECUTION_ARMS),
        "design": "two_fitting_arms_cross_two_execution_arms_same_cloned_parameters_and_step_across_execution",
        "direction_reproduction": "rebuild_all_Stage73_geometry_and_historical_reward_frames_exactly_before_native_evaluation",
        "native_pairing": "fresh_common_environment_seeds_stepwise_lower_noise_initial_policy_rng_and_upper_proposals_all13_variants_both_execution_arms",
        "execution_contrast": "normal_minus_zero_residual_of_plus_base_at_fixed_fitting_arm",
        "fitting_contrast": "joint_history_minus_zero_history_of_plus_base_at_fixed_execution_arm",
        "interaction": "difference_of_execution_contrasts_joint_history_minus_zero_history",
        "statistics": "all72_cell_signed_contrasts_12_execution_12_fitting_6_interactions_equal_root_bootstrap_Bonferroni102",
        "decision": "crossed_forward_response_diagnosis_no_direction_selection_or_policy_adoption_Stage67_HOLD_unchanged",
        "limits": "execution_comparison_holds_parameters_not_target_state_KL_fixed_teacher_initialized_development_roots_not_training_OOD_or_frequency_claim"}
