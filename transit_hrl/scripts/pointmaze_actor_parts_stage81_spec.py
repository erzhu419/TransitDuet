"""Fresh fixed-radius mean versus exploration-scale native actor directions."""

import math
from scripts import pointmaze_scenario_credit_stage80_spec as source

ROOT, PERIODS, CHUNK_SIZE = source.ROOT, source.PERIODS, source.CHUNK_SIZE
roots, arguments, source_result, options = source.roots, source.arguments, source.source_result, source.options
FISHER_RADIUS = source.FISHER_RADIUS
EXPERIMENT_PROTOCOL = "pointmaze_actor_parts_stage81_v1"
POLICY = "scenario_actor_parts"
RUNNER_SCRIPT = "scripts/run_pointmaze_actor_parts_stage81.py"
PARTS = ("full", "mean", "log_std")
DIRECTIONS = tuple(f"{actor}_{part}" for actor in ("upper", "lower") for part in PARTS)
VARIANTS = ("base", "zero", *tuple(f"{d}_{s}" for d in DIRECTIONS for s in ("plus", "minus")))
CONTRAST_PAIRS = (("base", "zero"), *tuple((d + "_" + s, b) for d in DIRECTIONS
    for s, b in (("plus", d + "_minus"), ("plus", "base"), ("minus", "base"), ("plus", "zero"))),
    *tuple((f"{a}_{p}_plus", f"{a}_{q}_plus") for a in ("upper", "lower")
        for p, q in (("full", "mean"), ("full", "log_std"), ("mean", "log_std"))))
ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a, b in CONTRAST_PAIRS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (81, 81081)


def seed_roles(root, *, preflight):
    base = 81_090_000 if preflight else 81_100_000 + roots(preflight=False).index(root) * 10000
    n, e = options(preflight=preflight)["credit_scenarios_per_batch"], options(preflight=preflight)["evaluation_episodes"]
    return {"credit_" + name: [{"scenario_seed": base + offset + i,
        "noise_seeds": [base + offset + 2001 + 2 * i, base + offset + 2002 + 2 * i]} for i in range(n)]
        for name, offset in (("A", 1), ("B", 1001))} | {
        "native_evaluation": list(range(base + 5001, base + 5001 + e))}


def budget(*, preflight):
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    o = options(preflight=preflight)
    n, e = 2 * o["credit_scenarios_per_batch"] * o["rollouts_per_scenario"], o["evaluation_episodes"]
    count = len(PERIODS) * (n + len(VARIANTS) * e)
    forwards = n * sum(math.ceil(h / CHUNK_SIZE) + math.ceil((h // p) / CHUNK_SIZE) for p in PERIODS)
    fisher = len(PARTS) * sum(math.ceil(n * h / CHUNK_SIZE) + math.ceil(n * (h // p) / CHUNK_SIZE) for p in PERIODS)
    return {"source_clone_loads": len(PERIODS), "forecaster_loads": 1, "decoder_loads": len(PERIODS),
        "credit_episodes": n * len(PERIODS), "evaluation_episodes": e * len(VARIANTS) * len(PERIODS),
        "native_episodes": count, "native_steps": count * h, "native_lower_calls": count * h,
        "native_upper_calls": (n + e * len(VARIANTS)) * sum(h // p for p in PERIODS),
        "pairing_upper_forward_calls": (n + e * len(VARIANTS)) * sum(h // p for p in PERIODS),
        "native_network_checks": count, "native_pair_checks": e * len(PERIODS),
        "scenario_pair_checks": n // o["rollouts_per_scenario"] * len(PERIODS),
        "objective_checks": n * len(PERIODS), "mc_calls": 2 * n * len(PERIODS),
        "actor_score_forward_batches": forwards, "actor_score_backward_batches": 3 * forwards,
        "fisher_jvp_batches": fisher, "exact_kl_forward_batches": 2 * fisher,
        "actor_parameter_perturbations": 4 * len(PARTS) * len(PERIODS), "frozen_model_checks": len(PERIODS),
        "parameter_part_checks": 4 * len(PARTS) * len(PERIODS)}


def contract():
    return {"source": source.contract()["source"], "periods": list(PERIODS), "variants": list(VARIANTS),
        "decoder": source.contract()["decoder"], "credit": source.contract()["credit"],
        "baseline": source.contract()["baseline_candidate"], "sampling": source.contract()["sampling"],
        "noise_mapping": "Stage80_explicit_environment_and_action_noise_mapping_on_fresh_Stage81_seed_roles",
        "noise_units": source.contract()["noise_units"],
        "direction": "same_raw_scenario_MC_gradient_projected_to_full_mean_net_only_or_log_std_only_no_entropy_or_normalization",
        "perturbation": source.contract()["perturbation"], "pairing": source.contract()["pairing"],
        "parameter_freeze": "mean_only_log_std_bit_exact_frozen_log_std_only_entire_mean_net_bit_exact_frozen_other_actor_and_Adam_unchanged",
        "statistics": "all62_reward_contrasts_equal_root_bootstrap65536_Bonferroni62_credit_and_std_shift_descriptive",
        "decision": source.contract()["decision"], "artifacts": source.contract()["artifacts"],
        "limits": "radius_matched_parameter_subspace_tests_not_additive_gain_decomposition_offline_baseline_future_not_actor_input_not_joint_HRL"}
