"""Matched-noise actor training: simultaneous versus lower-then-upper."""

from scripts import pointmaze_joint_conditioned_stage102_spec as source
from scripts import pointmaze_joint_staged_stage103_spec as comparison
from scripts.pointmaze_iterative_mc_stage83_spec import training_budget

ROOT, PERIODS, CHUNK_SIZE, FISHER_RADIUS = source.ROOT, source.PERIODS, source.CHUNK_SIZE, source.FISHER_RADIUS
roots, arguments, options, source_record = source.roots, source.arguments, source.options, source.source_record
METHODS = {m: ("upper", "lower") for m in ("joint_independent", "joint_conditioned", "staged_independent", "staged_common")}
JOINT_METHODS, STAGED_METHODS = tuple(METHODS)[:2], tuple(METHODS)[2:]
COMPOSITIONS = {"base": ("source", "source"), "zero": ("source", "source"), **{m: (m, m) for m in METHODS}}
VARIANTS = tuple(COMPOSITIONS)
CONTRAST_PAIRS, ENDPOINTS, PRIMARY_ENDPOINTS = comparison.CONTRAST_PAIRS, comparison.ENDPOINTS, comparison.PRIMARY_ENDPOINTS
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (104, 104104)
EXPERIMENT_PROTOCOL = "pointmaze_paired_order_stage104_v1"
POLICY = "same_actor_rosters_joint_vs_lower_then_upper"
RUNNER_SCRIPT = "scripts/run_pointmaze_paired_order_stage104.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_paired_order_stage104.py"


def allocation(method, period):
    if method not in METHODS:
        raise ValueError("Unregistered paired-order learner")
    return source.allocation("joint_independent", period)


def seed_roles(root, *, preflight):
    roots(preflight=preflight).index(root)
    base = 104_000_000 if preflight else 104_100_000 + roots(preflight=False).index(root)*100000
    o = options(preflight=preflight)
    rounds = [{name: [{"scenario_seed": base+10000*j+offset+i,
        "noise_seeds": [base+10000*j+offset+2001+2*i, base+10000*j+offset+2002+2*i]}
        for i in range(o["credit_scenarios_per_batch"])]
        for name, offset in (("credit_A", 1), ("credit_B", 1001), ("lower_credit_A", 4001), ("lower_credit_B", 5001))}
        for j in range(o["updates"])]
    return {"training_rounds": rounds,
        "native_evaluation": list(range(base+95001, base+95001+o["evaluation_episodes"]))}


def update_schedule(period, *, preflight):
    n = options(preflight=preflight)["updates"]
    steps = []
    for phase, methods, actors, offset in (("joint", JOINT_METHODS, ("upper", "lower"), 0),
            ("lower", STAGED_METHODS, ("lower",), 0), ("upper", STAGED_METHODS, ("upper",), n)):
        for i in range(1, n+1):
            for method in methods:
                steps.append({"method": method, "iteration": offset+i, "credit_iteration": i, "phase": phase,
                    "allocation": {a: allocation(method, period)[a] for a in actors}})
    return steps


def matched_path_budget(period, *, preflight):
    o = options(preflight=preflight)
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    n = 2*o["credit_scenarios_per_batch"]*o["rollouts_per_scenario"]*o["updates"]
    a = allocation("joint_conditioned", period)
    b = {"upper_credit_paths": n, "lower_credit_paths": n, "native_training_episodes": 2*n,
        "upper_mean_updates": o["updates"], "lower_mean_updates": o["updates"],
        "upper_cumulative_nominal_KL": o["updates"]*FISHER_RADIUS*a["upper"],
        "lower_cumulative_nominal_KL": o["updates"]*FISHER_RADIUS*a["lower"],
        "common_lower_extra_upper_replay_forwards": (n//2)*(h//period)}
    return {"joint": dict(b), "staged": dict(b), "status": "matched",
        "update_operations": {"joint": o["updates"], "staged": 2*o["updates"]}}


def budget(*, preflight):
    o = options(preflight=preflight)
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    b = training_budget(o, METHODS, VARIANTS, horizon=h, preflight=preflight)
    extra = b["credit_episodes"]
    for k in ("credit_episodes", "objective_checks", "mc_calls", "scenario_pair_checks"):
        b[k] *= 2
    for k in ("native_episodes", "native_network_checks"):
        b[k] += extra
    for k in ("native_steps", "native_lower_calls"):
        b[k] += extra*h
    for k in ("native_upper_calls", "pairing_upper_forward_calls"):
        b[k] += extra//len(PERIODS)*sum(h//p for p in PERIODS)
    operations = len(PERIODS)*len(update_schedule(PERIODS[0], preflight=preflight))
    b.update(policy_updates=operations, training_freeze_checks=operations)
    pairs = len(PERIODS)*o["updates"]*2*o["credit_scenarios_per_batch"]
    return {**b, "upper_independent_pair_checks": len(METHODS)*pairs,
        "lower_independent_pair_checks": 2*pairs, "lower_common_pair_checks": 2*pairs,
        "actor_composition_checks": len(PERIODS)*len(VARIANTS),
        "phase_boundary_checks": 2*len(PERIODS)*len(STAGED_METHODS),
        "upper_replay_forward_calls": 2*o["updates"]*2*o["credit_scenarios_per_batch"]*sum(h//p for p in PERIODS)}


def contract():
    inherited = source.contract()
    return {k: inherited[k] for k in ("source", "decoder", "credit", "artifacts")} | {
        "periods": list(PERIODS), "variants": list(VARIANTS), "methods": list(METHODS),
        "training": "all_four_from_original_Stage96_U0_L0_joint_simultaneous_staged_all_lower_then_all_upper_eight_updates_per_actor_preflight_two",
        "sampling": "same_registered_scenario_noise_rosters_for_each_actor_and_update_across_all_four_learners_separate_upper_lower_pools",
        "conditioning": "only_joint_conditioned_and_staged_common_lower_credit_share_upper_innovations_independent_lower_noise",
        "freeze": "std_values_source_Adam_forecaster_decoder_fixed_staged_upper_frozen_during_lower_and_learned_lower_frozen_during_upper",
        "budget": "equal_per_actor_samples_updates_and_cumulative_nominal_call_weighted_KL_common_lower_replay_matched_no_trained_donor_reuse",
        "evaluation": "fresh_paired_original_independent_noise_six_complete_policies_only_after_all_registered_updates_no_checkpoint_selection",
        "statistics": "all18_reward_contrasts_equal_root_bootstrap65536_Bonferroni18_seed104_104104_no_stage_pooling",
        "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "decision": "both_joint_conditioned_minus_staged_common_primary_corrected_CI_lower_bounds_positive_preflight_mechanical_only",
        "limits": "same_task_teacher_initialized_MC_update_order_comparison_not_full_actor_critic_equal_realized_trajectory_KL_or_frequency_superiority"}
