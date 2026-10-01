"""Matched-budget reward-rate versus independent same-scenario MC baselines."""

import math
import numpy as np
from scripts import pointmaze_feasible_credit_stage79_spec as source

ROOT, PERIODS, CHUNK_SIZE = source.ROOT, source.PERIODS, source.CHUNK_SIZE
roots, arguments, source_result = source.roots, source.arguments, source.source_result
FISHER_RADIUS = source.FISHER_RADIUS
EXPERIMENT_PROTOCOL = "pointmaze_scenario_credit_stage80_v1"
POLICY = "scenario_crossfit_credit"
RUNNER_SCRIPT = "scripts/run_pointmaze_scenario_credit_stage80.py"
METHODS = ("rate", "scenario")
DIRECTIONS = tuple(f"{actor}_{method}" for actor in ("upper", "lower") for method in METHODS)
VARIANTS = ("base", "zero", *tuple(f"{d}_{s}" for d in DIRECTIONS for s in ("plus", "minus")))
CONTRAST_PAIRS = (("base", "zero"), *tuple((d + "_" + s, b) for d in DIRECTIONS
    for s, b in (("plus", d + "_minus"), ("plus", "base"), ("minus", "base"), ("plus", "zero"))),
    *((f"{a}_scenario_plus", f"{a}_rate_plus") for a in ("upper", "lower")))
CREDIT_METRICS = ("scenario_minus_rate_cosine", "log_rate_over_scenario_gradient_variance")
ENDPOINTS = (*tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a, b in CONTRAST_PAIRS),
    *tuple(f"{p}/{a}/{m}" for p in PERIODS for a in ("upper", "lower") for m in CREDIT_METRICS))
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (80, 80080)


def options(*, preflight):
    return {"workers": 2 if preflight else 8, "credit_scenarios_per_batch": 2 if preflight else 16,
        "rollouts_per_scenario": 2, "evaluation_episodes": 4 if preflight else 32}


def seed_roles(root, *, preflight):
    base = 80_090_000 if preflight else 80_100_000 + roots(preflight=False).index(root) * 10000
    n, e = options(preflight=preflight)["credit_scenarios_per_batch"], options(preflight=preflight)["evaluation_episodes"]
    return {"credit_" + name: [{"scenario_seed": base + offset + i,
        "noise_seeds": [base + offset + 2001 + 2 * i, base + offset + 2002 + 2 * i]} for i in range(n)]
        for name, offset in (("A", 1), ("B", 1001))} | {
        "native_evaluation": list(range(base + 5001, base + 5001 + e))}


def noise_seeds(root, scenario_seed, noise_seed):
    return tuple(map(int, np.random.SeedSequence([80, root, scenario_seed, noise_seed, 80017]).generate_state(2)))


def budget(*, preflight):
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    o = options(preflight=preflight)
    n, e = 2 * o["credit_scenarios_per_batch"] * o["rollouts_per_scenario"], o["evaluation_episodes"]
    count = len(PERIODS) * (n + len(VARIANTS) * e)
    forwards = n * sum(math.ceil(h / CHUNK_SIZE) + math.ceil((h // p) / CHUNK_SIZE) for p in PERIODS)
    fisher = len(METHODS) * sum(math.ceil(n * h / CHUNK_SIZE) + math.ceil(n * (h // p) / CHUNK_SIZE) for p in PERIODS)
    return {"source_clone_loads": len(PERIODS), "forecaster_loads": 1, "decoder_loads": len(PERIODS),
        "credit_episodes": n * len(PERIODS), "evaluation_episodes": e * len(VARIANTS) * len(PERIODS),
        "native_episodes": count, "native_steps": count * h, "native_lower_calls": count * h,
        "native_upper_calls": (n + e * len(VARIANTS)) * sum(h // p for p in PERIODS),
        "pairing_upper_forward_calls": (n + e * len(VARIANTS)) * sum(h // p for p in PERIODS),
        "native_network_checks": count, "native_pair_checks": e * len(PERIODS),
        "scenario_pair_checks": n // o["rollouts_per_scenario"] * len(PERIODS),
        "objective_checks": n * len(PERIODS), "mc_calls": 2 * n * len(PERIODS),
        "actor_score_forward_batches": forwards, "actor_score_backward_batches": 4 * forwards,
        "fisher_jvp_batches": fisher, "exact_kl_forward_batches": 2 * fisher,
        "actor_parameter_perturbations": 4 * len(METHODS) * len(PERIODS), "frozen_model_checks": len(PERIODS)}


def contract():
    return {"source": source.source.EXPERIMENT_PROTOCOL, "periods": list(PERIODS), "variants": list(VARIANTS),
        "decoder": source.contract()["decoder"], "credit": source.contract()["credit"],
        "baseline_control": "remaining_primitive_steps_times_opposite_disjoint_scenario_batch_mean_reward_rate",
        "baseline_candidate": "same_exogenous_scenario_time_aligned_other_independent_action_noise_rollout_MC_return",
        "sampling": "two_disjoint_batches_16_scenarios_each_two_independent_noise_replicates_exact_initial_physical_and_exogenous_history_match",
        "noise_units": "average_gradients_within_scenario_then_measure_covariance_across_independent_scenario_groups",
        "direction": source.contract()["direction"], "perturbation": source.contract()["perturbation"],
        "pairing": source.contract()["pairing"],
        "statistics": "38_reward_and8_credit_contrasts_equal_root_bootstrap65536_Bonferroni46_no_threshold_or_reward_selection",
        "decision": source.contract()["decision"], "artifacts": source.contract()["artifacts"],
        "limits": "offline_training_control_variate_uses_other_rollout_future_not_inference_input_not_a_state_value_fit_or_joint_HRL_proof"}
