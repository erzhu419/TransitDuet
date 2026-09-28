"""Frozen lower reward-source by credit-boundary factorial."""

from scripts import pointmaze_level_deployment_stage37_spec as previous

ROOT, source, POLICY = previous.ROOT, previous.source, previous.POLICY
EXPERIMENT_PROTOCOL = "pointmaze_lower_credit_stage38_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_lower_credit_stage38.py"
OPTIMIZER_ROOTS, PREFLIGHT_ROOTS = previous.OPTIMIZER_ROOTS, previous.PREFLIGHT_ROOTS
METHODS = ("frozen", "intrinsic_option", "intrinsic_episode", "task_option", "task_episode")
LOWER_CREDIT = {m: "intrinsic_option" if m == "frozen" else m for m in METHODS}
COMPONENTS = {m: () if m == "frozen" else ("lower",) for m in METHODS}
COHORTS = previous.COHORTS
ENDPOINTS = tuple(f"{m}:frozen:return" for m in METHODS[1:]) + (
    "reward_source:return", "credit_boundary:return", "interaction:return")
CI_FAMILY_SIZE = len(ENDPOINTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (38, 38039)
SHUFFLE_SEED_NAMESPACE = 38
AGGREGATE_STATUS = "stage38_lower_credit_diagnosis_complete"


def roots(*, preflight):
    return previous.roots(preflight=preflight)


def native_method(method):
    if method not in METHODS:
        raise ValueError("unregistered Stage-38 method")
    return "learned_history"


def options(*, preflight):
    return previous.options(preflight=preflight)


def seed_roles(root, *, preflight):
    roster = roots(preflight=preflight)
    if root not in roster:
        raise ValueError("unregistered Stage-38 root")
    base = 9_390_000 if preflight else 9_400_000 + roster.index(root) * 10000
    opt = options(preflight=preflight)
    return {"training": list(range(base + 1, base + 1 + opt["iterations"] * opt["rollouts_per_iteration"])),
            "selection": list(range(base + 2001, base + 2001 + opt["selection_paths"])),
            "evaluation": list(range(base + 3001, base + 3001 + opt["evaluation_paths"]))}


def source_result(root, *, preflight):
    return previous.source_result(root, preflight=preflight)


def budget(*, preflight):
    return previous.budget(preflight=preflight)


def verification_budget(*, preflight):
    horizon = source.arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    return {"native_episodes_per_cell": 3,
            "total_primitive_steps": 3 * horizon * len(roots(preflight=preflight)) * len(METHODS)}


def contrasts(v):
    f, io, ie, to, te = (v[m]["episode_return"] for m in METHODS)
    return [io - f, ie - f, to - f, te - f,
            .5 * (to + te - io - ie), .5 * (ie + te - io - to), te - to - ie + io]


def contract():
    return {**previous.contract(), "methods": list(METHODS),
            "updated_actor_and_critic_levels": {m: list(COMPONENTS[m]) for m in METHODS},
            "lower_credit": LOWER_CREDIT,
            "lower_reward": "factorial_intrinsic_adapter_or_unscaled_native_task_reward",
            "reward_source": "intrinsic_adapter_reward_or_unscaled_native_task_reward_no_added_call_cost",
            "credit_boundary": "option_terminal_at_replanning_or_episode_terminal_only",
            "training": "on_policy_lower_actor_and_critic_only_upper_and_gate_frozen",
            "critic_initialization": "same_inherited_lower_critic_all_arms_no_reset_or_reward_rescaling",
            "minibatch_shuffle_seed": "numpy_seedsequence_38_optimizer_root_iteration",
            "primary_endpoints": list(ENDPOINTS), "all_primary_endpoints": list(ENDPOINTS),
            "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": list(BOOTSTRAP_SEED),
            "interval": "two_sided_percentile_bonferroni_7_endpoints",
            "interaction": "task_episode_minus_task_option_minus_intrinsic_episode_plus_intrinsic_option",
            "reward_source_effect": "mean_task_minus_mean_intrinsic_over_boundaries",
            "credit_boundary_effect": "mean_episode_minus_mean_option_over_reward_sources",
            "verification": "both_cohort_checkpoints_and_initial_training_credit_native_replay",
            "decision": "diagnose_lower_credit_not_algorithm_superiority",
            "evidence_role": "conditional_lower_credit_development_not_independent_confirmation"}
