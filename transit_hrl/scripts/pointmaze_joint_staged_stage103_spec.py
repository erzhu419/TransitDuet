"""Fresh paired evaluation of equal method-path joint and staged budgets."""

from scripts import pointmaze_joint_conditioned_stage102_spec as joint
from scripts import pointmaze_fresh_staged_upper_stage100_spec as upper
from scripts import pointmaze_fresh_lower_stage99_spec as lower
from scripts.pointmaze_iterative_mc_stage83_spec import training_budget

ROOT, PERIODS, CHUNK_SIZE, FISHER_RADIUS = joint.ROOT, joint.PERIODS, joint.CHUNK_SIZE, joint.FISHER_RADIUS
roots, arguments, source_record = joint.roots, joint.arguments, joint.source_record
SOURCE_PROTOCOLS = {"joint": joint, "upper": upper, "lower": lower}
SOURCE_RUNS = {"joint": "pointmaze_joint_conditioned_stage102_full_20261003_r1",
    "upper": "pointmaze_fresh_staged_upper_stage100_full_20261003_r1",
    "lower": "pointmaze_fresh_lower_stage99_full_20261003_r1"}
DONORS = {m: owner for owner, methods in (
    ("lower", ("source_upper_independent", "source_upper_common")),
    ("upper", ("staged_independent", "staged_common")),
    ("joint", ("joint_independent", "joint_conditioned"))) for m in methods}
METHODS = {}
COMPOSITIONS = {"base": ("source", "source"), "zero": ("source", "source"),
    "joint_independent": ("joint_independent", "joint_independent"),
    "joint_conditioned": ("joint_conditioned", "joint_conditioned"),
    "staged_independent": ("staged_independent", "source_upper_independent"),
    "staged_common": ("staged_common", "source_upper_common")}
VARIANTS = tuple(COMPOSITIONS)
CONTRAST_PAIRS = (("joint_conditioned", "staged_common"), ("joint_independent", "staged_independent"),
    ("joint_conditioned", "joint_independent"), ("staged_common", "staged_independent"),
    *((m, "base") for m in VARIANTS if m not in ("base", "zero")), ("base", "zero"))
ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a, b in CONTRAST_PAIRS)
PRIMARY_ENDPOINTS = tuple(f"{p}/joint_conditioned_minus_staged_common" for p in PERIODS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (103, 103103)
EXPERIMENT_PROTOCOL = "pointmaze_joint_staged_stage103_v1"
POLICY = "frozen_equal_path_joint_staged_evaluation"
RUNNER_SCRIPT = "scripts/run_pointmaze_joint_staged_stage103.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_joint_staged_stage103.py"


def options(*, preflight):
    return {**joint.options(preflight=preflight), "updates": 0}


def donor_result(root, owner):
    return ROOT / "results" / SOURCE_RUNS[owner] / "cells" / f"replicate_{root}" / "result.json"


def checkpoint_path(root, period, method):
    return donor_result(root, DONORS[method]).parent / "final_weights" / f"period_{period}_{method}.pt"


def seed_roles(root, *, preflight):
    roots(preflight=preflight).index(root)
    base = 103_000_000 if preflight else 103_100_000 + roots(preflight=False).index(root)*100000
    return {"training_rounds": [], "native_evaluation": list(range(base+95001,
        base+95001+options(preflight=preflight)["evaluation_episodes"]))}


def method_path_budgets(root, period):
    oj, ou, ol = [p.options(preflight=False) for p in (joint, upper, lower)]
    paths = lambda o: 2*o["credit_scenarios_per_batch"]*o["rollouts_per_scenario"]*o["updates"]
    aj, au, al = joint.allocation("joint_conditioned", period), upper.allocation("staged_common", period), lower.allocation("source_upper_common", period)
    h = joint.arguments(root, preflight=False).horizon
    if not (vars(joint.arguments(root, preflight=False)) == vars(upper.arguments(root, preflight=False)) == vars(lower.arguments(root, preflight=False))):
        raise ValueError("Joint/staged native training settings differ")
    common = lambda n: (n//2)*(h//period)
    jb = {"upper_credit_paths": paths(oj), "lower_credit_paths": paths(oj),
        "native_episodes": 2*paths(oj), "upper_mean_updates": oj["updates"], "lower_mean_updates": oj["updates"],
        "upper_cumulative_nominal_KL": oj["updates"]*FISHER_RADIUS*aj["upper"],
        "lower_cumulative_nominal_KL": oj["updates"]*FISHER_RADIUS*aj["lower"],
        "common_lower_extra_upper_replay_forwards": common(paths(oj))}
    sb = {"upper_credit_paths": paths(ou), "lower_credit_paths": paths(ol),
        "native_episodes": paths(ou)+paths(ol), "upper_mean_updates": ou["updates"], "lower_mean_updates": ol["updates"],
        "upper_cumulative_nominal_KL": ou["updates"]*FISHER_RADIUS*au["upper"],
        "lower_cumulative_nominal_KL": ol["updates"]*FISHER_RADIUS*al["lower"],
        "common_lower_extra_upper_replay_forwards": common(paths(ol))}
    if jb != sb:
        raise ValueError("Joint/staged method-path training budgets do not match")
    for name, protocol, method in (("joint", joint, "joint_independent"), ("upper", upper, "staged_independent"),
            ("lower", lower, "source_upper_independent")):
        if protocol.allocation(method, period) != {"joint": aj, "upper": au, "lower": al}[name]:
            raise ValueError("Independent/common controls changed their allocation")
    return {"joint": jb, "staged": sb, "status": "matched", "native_horizon": h,
        "scope": "method_paths_only_preparation_campaign_and_evaluation_costs_separate_not_final_trajectory_KL"}


def budget(*, preflight):
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    return {**training_budget(options(preflight=preflight), METHODS, VARIANTS, horizon=h, preflight=preflight),
        "checkpoint_loads": len(PERIODS)*len(DONORS), "checkpoint_freeze_checks": len(PERIODS)*len(DONORS),
        "actor_composition_checks": len(PERIODS)*len(VARIANTS), "matched_training_budget_checks": len(PERIODS),
        "upper_replay_forward_calls": 0}


def contract():
    c = joint.contract()
    return {k: c[k] for k in ("source", "decoder", "credit", "artifacts")} | {
        "periods": list(PERIODS), "variants": list(VARIANTS), "methods": {},
        "source_runs": SOURCE_RUNS, "registered_final_donors": DONORS,
        "compositions_upper_lower": {v: list(pair) for v, pair in COMPOSITIONS.items()},
        "training": "none_frozen_full_Stage102_joint_Stage99_U0_lower_and_Stage100_U0_upper_only_no_Stage101_UJ_refinement",
        "sampling": "fresh_paired_Stage103_rollouts_original_independent_upper_lower_noise_all_six_policies",
        "evaluation": "fixed_final_donors_no_checkpoint_root_period_selection_or_recalibration",
        "budget": "equal_method_path_actor_samples_updates_common_lower_replay_and_cumulative_nominal_call_weighted_KL_not_equal_historical_campaign",
        "statistics": "all18_reward_contrasts_equal_root_bootstrap65536_Bonferroni18_seed103_103103",
        "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "decision": "both_joint_conditioned_minus_staged_common_primary_corrected_CI_lower_bounds_positive_preflight_mechanical_only",
        "limits": "different_registered_training_rosters_paired_fresh_evaluation_recipe_comparison_not_isolated_update_order_causality_full_actor_critic_frequency_or_unseen_task_proof"}
