"""Frozen critic-only calibration and early lower PPO drift diagnosis."""

import numpy as np
from scripts import pointmaze_lower_credit_stage38_spec as previous

ROOT, source = previous.ROOT, previous.source
EXPERIMENT_PROTOCOL = "pointmaze_critic_calibration_stage39_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_critic_calibration_stage39.py"
METHODS = ("frozen", "intrinsic_delayed", "intrinsic_calibrated", "task_delayed", "task_calibrated")
MODES = ("deterministic", "lower_sampled")
LOWER_CREDIT = {m: "task_option" if m.startswith("task_") else "intrinsic_option" for m in METHODS}
ENDPOINTS = tuple(f"{m}:frozen:final_return" for m in METHODS[1:]) + (
    "intrinsic:calibration:final_return", "task:calibration:final_return",
    "intrinsic:calibration:first_update_return", "task:calibration:first_update_return")
CI_FAMILY_SIZE = len(ENDPOINTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (39, 39039)
METRICS = ("episode_return", "tracking_squared_error_integral", "upper_inference_calls", "charged_utility")


def roots(*, preflight):
    return previous.roots(preflight=preflight)


def options(*, preflight):
    return {"warmup_iterations": 2 if preflight else 16,
            "learning_iterations": 2 if preflight else 16,
            "rollouts_per_iteration": 1 if preflight else 8,
            "evaluation_paths": 2 if preflight else 16, "workers": 1 if preflight else 8}


def iterations(*, preflight):
    opt = options(preflight=preflight)
    return opt["warmup_iterations"] + opt["learning_iterations"]


def snapshots(*, preflight):
    warm = options(preflight=preflight)["warmup_iterations"]
    return (0, warm, warm + 1, warm + 2) if preflight else (0, warm, warm + 1, warm + 4, iterations(preflight=False))


def update_kind(method, iteration, *, preflight):
    if method not in METHODS or not 1 <= iteration <= iterations(preflight=preflight):
        raise ValueError("unregistered calibration method/iteration")
    if method == "frozen":
        return "none"
    if iteration <= options(preflight=preflight)["warmup_iterations"]:
        return "critic" if method.endswith("_calibrated") else "none"
    return "actor_critic"


def seed_roles(root, *, preflight):
    roster = roots(preflight=preflight)
    if root not in roster:
        raise ValueError("unregistered calibration root")
    base = 10_390_000 if preflight else 10_400_000 + roster.index(root) * 10000
    opt = options(preflight=preflight)
    count = iterations(preflight=preflight) * opt["rollouts_per_iteration"]
    return {"training": list(range(base + 1, base + 1 + count)), "probe": [base + 2001],
            "evaluation": list(range(base + 3001, base + 3001 + opt["evaluation_paths"]))}


def policy_seed(root, seed):
    return int(np.random.SeedSequence([39, root, seed, 39019]).generate_state(1)[0])


def source_result(root, *, preflight):
    return previous.source_result(root, preflight=preflight)


def budget(*, preflight):
    horizon = source.arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    opt = options(preflight=preflight)
    values = {"training_primitive_steps": iterations(preflight=preflight) * opt["rollouts_per_iteration"] * horizon,
              "evaluation_primitive_steps": len(snapshots(preflight=preflight)) * len(MODES) * opt["evaluation_paths"] * horizon,
              "probe_primitive_steps": horizon, "factual_replay_primitive_steps": horizon}
    return {**values, "total_primitive_steps": sum(values.values())}


def verification_budget(*, preflight):
    horizon = source.arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    count = len(snapshots(preflight=preflight)) * len(MODES) + 2
    return {"native_episodes_per_cell": count,
            "total_primitive_steps": count * horizon * len(roots(preflight=preflight)) * len(METHODS)}


def contrasts(means, *, preflight=False):
    final = means[str(iterations(preflight=preflight))]["deterministic"]
    first = means[str(options(preflight=preflight)["warmup_iterations"] + 1)]["deterministic"]
    values = [final[m]["episode_return"] - final["frozen"]["episode_return"] for m in METHODS[1:]]
    for stage in (final, first):
        values.extend(stage[s + "_calibrated"]["episode_return"] - stage[s + "_delayed"]["episode_return"]
                      for s in ("intrinsic", "task"))
    return values


def contract():
    return {"controller_source": "same_stage33_controller_and_stage35_initial_gate_as_stage38",
            "methods": list(METHODS), "lower_credit": LOWER_CREDIT,
            "warmup": "calibrated_critic_only_delayed_no_updates_same_frozen_actor_data",
            "learning": "equal_lower_actor_and_critic_updates_after_warmup",
            "frozen_levels": "upper_and_gate_actor_and_critic_always_frozen",
            "critic_initialization": "inherited_no_reset_no_reward_rescale",
            "ppo": "unchanged_shared_update_level_with_existing_actor_updates_enabled_argument",
            "training_sampling": "on_policy_all_levels_seed_environment_plus_optimizer_root",
            "deployment_modes": list(MODES), "deployment_upper_and_gate": "deterministic_in_both_modes",
            "lower_sampling_seed": "numpy_seedsequence_39_root_environment_seed_39019",
            "probe": "one_disjoint_initial_stochastic_episode_never_used_for_updates",
            "critic_error": "fixed_probe_discounted_monte_carlo_returns_respecting_option_cuts",
            "policy_drift": "exact_gaussian_kl_and_squashed_mean_action_change_on_fixed_probe_and_update_batch",
            "shuffle_seed": "numpy_seedsequence_39_root_global_iteration",
            "primary_endpoints": list(ENDPOINTS), "primary_cohort": "final_and_first_update_deterministic",
            "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": list(BOOTSTRAP_SEED),
            "interval": "two_sided_percentile_bonferroni_8_endpoints",
            "statistical_unit": "equal_weight_optimizer_root_means_after_path_averaging",
            "checkpoint_selection": "none_all_registered_snapshots_retained",
            "root_exclusion": "forbidden", "sequential_extension": "forbidden",
            "decision": "critic_calibration_and_early_update_diagnosis_not_algorithm_superiority",
            "evidence_role": "conditional_development_on_reused_roots_not_independent_confirmation"}
