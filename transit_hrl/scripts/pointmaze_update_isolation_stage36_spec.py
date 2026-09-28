"""Frozen 2x2 gate/controller update isolation on fresh development paths."""

from scripts import pointmaze_joint_renewal_stage35_spec as previous

ROOT = previous.ROOT
source = previous.source
POLICY = previous.POLICY
EXPERIMENT_PROTOCOL = "pointmaze_update_isolation_stage36_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_update_isolation_stage36.py"
OPTIMIZER_ROOTS = previous.OPTIMIZER_ROOTS
PREFLIGHT_ROOTS = previous.PREFLIGHT_ROOTS
METHODS = ("frozen", "gate_only", "controller_only", "joint", "fixed50")
COMPONENTS = {"frozen": (), "gate_only": ("promotion",),
              "controller_only": ("upper", "lower"),
              "joint": ("upper", "lower", "promotion"), "fixed50": ("upper", "lower")}
COHORTS = ("final", "selected")
ENDPOINTS = ("gate_only:frozen:return", "controller_only:frozen:return",
             "joint:frozen:return", "interaction:return", "joint:fixed50:return",
             "joint:fixed50:ise", "joint:fixed50:calls")
BOOTSTRAP_DRAWS = 65536
BOOTSTRAP_SEED = (36, 36039)
SHUFFLE_SEED_NAMESPACE = 36
CI_FAMILY_SIZE = len(ENDPOINTS)
AGGREGATE_STATUS = "stage36_component_diagnosis_complete"


def contrasts(values):
    f, g, c, j, b = (values[m] for m in METHODS)
    return [g["episode_return"] - f["episode_return"], c["episode_return"] - f["episode_return"],
            j["episode_return"] - f["episode_return"],
            j["episode_return"] - c["episode_return"] - g["episode_return"] + f["episode_return"],
            j["episode_return"] - b["episode_return"],
            b["tracking_squared_error_integral"] - j["tracking_squared_error_integral"],
            b["upper_inference_calls"] - j["upper_inference_calls"]]


def roots(*, preflight):
    return PREFLIGHT_ROOTS if preflight else OPTIMIZER_ROOTS


def native_method(method):
    if method not in METHODS:
        raise ValueError("unregistered Stage-36 method")
    return "fixed50" if method == "fixed50" else "learned_history"


def options(*, preflight):
    return {**previous.options(preflight=preflight), "evaluation_cohorts": list(COHORTS)}


def seed_roles(root, *, preflight):
    roster = roots(preflight=preflight)
    if root not in roster:
        raise ValueError("unregistered Stage-36 root")
    base = 7_390_000 if preflight else 7_400_000 + roster.index(root) * 10000
    opt = options(preflight=preflight)
    return {"training": list(range(base + 1, base + 1 + opt["iterations"] * opt["rollouts_per_iteration"])),
            "selection": list(range(base + 2001, base + 2001 + opt["selection_paths"])),
            "evaluation": list(range(base + 3001, base + 3001 + opt["evaluation_paths"]))}


def source_result(root, *, preflight):
    return previous.source_result(root, preflight=preflight)


def budget(*, preflight):
    old = previous.budget(preflight=preflight)
    return {**old, "evaluation_primitive_steps": 2 * old["evaluation_primitive_steps"],
            "total_primitive_steps": old["total_primitive_steps"] + old["evaluation_primitive_steps"]}


def contract():
    return {"source_protocol": source.EXPERIMENT_PROTOCOL, "methods": list(METHODS),
            "updated_actor_and_critic_levels": {m: list(COMPONENTS[m]) for m in METHODS},
            "initialization": "same_stage33_controller_and_stage35_gate_initialization_fresh_optimizers",
            "frozen_components": "skip_actor_and_critic_optimizer_updates_exact_weights_unchanged",
            "training": "same_native_rollouts_and_smdp_ppo_as_stage35_no_hyperparameter_change",
            "minibatch_shuffle_seed": "numpy_seedsequence_36_optimizer_root_iteration",
            "gate_input": previous.contract()["gate_input"],
            "gate_check_steps": previous.CHECK_STEPS, "max_plan_age_steps": previous.MAX_AGE_STEPS,
            "upper_call_cost_in_reward_units": previous.CALL_COST,
            "upper_and_gate_reward": previous.contract()["upper_and_gate_reward"],
            "lower_reward": previous.contract()["lower_reward"], "candidate_previews": 0,
            "lower_feedback": "every_primitive_step", "sample_training_actions": "all_levels_in_every_arm",
            "checkpoint_selection": previous.contract()["checkpoint_selection"],
            "checkpoint_candidates": previous.contract()["checkpoint_candidates"],
            "evaluation_cohorts": list(COHORTS), "primary_cohort": "final",
            "evaluation_pairing": "same_untouched_paths_both_cohorts_and_all_methods",
            "primary_endpoints": list(ENDPOINTS), "interaction": "joint_minus_controller_only_minus_gate_only_plus_frozen",
            "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": list(BOOTSTRAP_SEED),
            "familywise_alpha": .05, "interval": "two_sided_percentile_bonferroni_7_endpoints",
            "statistical_unit": "optimizer_seed_root", "root_weighting": "equal",
            "decision": "diagnose_component_effects_on_final_weights_not_algorithm_success_gate",
            "evidence_role": "conditional_component_isolation_development_not_independent_confirmation",
            "root_exclusion": "forbidden", "sequential_extension": "forbidden"}
