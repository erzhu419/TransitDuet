"""Fixed decision-call-weighted KL allocation with full shared MC batches."""

from scripts import pointmaze_iterative_mc_stage83_spec as source

ROOT, PERIODS, CHUNK_SIZE = source.ROOT, source.PERIODS, source.CHUNK_SIZE
roots, arguments, source_result, options = source.roots, source.arguments, source.source_result, source.options
FISHER_RADIUS = source.FISHER_RADIUS
EXPERIMENT_PROTOCOL = "pointmaze_call_weighted_stage87_v1"
POLICY = "call_weighted_mc"
RUNNER_SCRIPT = "scripts/run_pointmaze_call_weighted_stage87.py"
METHODS = {"joint_call": ("upper", "lower"), "joint_level": ("upper", "lower"), "lower_trained": ("lower",)}
VARIANTS = ("base", "zero", *METHODS)
CONTRAST_PAIRS = (("joint_call", "lower_trained"), ("joint_call", "joint_level"), ("joint_level", "lower_trained"),
    *((m,b) for m in METHODS for b in ("base", "zero")), ("base", "zero"))
ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a,b in CONTRAST_PAIRS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (87, 87087)


def allocation(method, period):
    return {"joint_call": {"upper": .5, "lower": 1.-.5/period},
        "joint_level": {"upper": .5, "lower": .5}, "lower_trained": {"lower": 1.}}[method]


def seed_roles(root, *, preflight):
    base = 87_000_000 if preflight else 87_100_000 + roots(preflight=False).index(root)*100000
    o = options(preflight=preflight)
    rounds = [{"credit_"+name: [{"scenario_seed": base+10000*j+offset+i,
        "noise_seeds": [base+10000*j+offset+2001+2*i, base+10000*j+offset+2002+2*i]}
        for i in range(o["credit_scenarios_per_batch"])] for name,offset in (("A",1),("B",1001))}
        for j in range(o["updates"])]
    return {"training_rounds": rounds, "native_evaluation": list(range(base+95001,base+95001+o["evaluation_episodes"]))}


def budget(*, preflight):
    return source.training_budget(options(preflight=preflight),METHODS,VARIANTS,
        horizon=arguments(roots(preflight=preflight)[0],preflight=preflight).horizon,preflight=preflight)


def contract():
    return {**source.contract(), "variants": list(VARIANTS),
        "methods": {str(p): {m:allocation(m,p) for m in METHODS} for p in PERIODS},
        "noise_mapping": "Stage80_explicit_mapping_fresh_disjoint_Stage87_round_and_evaluation_roles",
        "sampling": "each_method_same_exogenous_roster_8_fresh_rounds_64_episodes_all_episodes_used_by_every_active_actor_preflight2x8",
        "budget": "per_native_step_nominal_K_lower_plus_K_upper_over_period_.001_joint_call_and_lower_only_joint_level_.0005_plus_.0005_over_period",
        "allocation": "upper_.0005_fixed_joint_call_lower_.001_minus_.0005_over_period_no_radius_or_allocation_search",
        "diagnostics": "nominal_and_empirical_call_weighted_old_history_conditional_KL_per_round_actor_episode_and_decision_counts",
        "statistics": "all20_final_reward_contrasts_equal_root_bootstrap65536_Bonferroni20",
        "limits": "teacher_initialized_fixed_std_decoder_MC_mean_training_empirical_old_history_KL_not_final_trajectory_KL_or_frequency_superiority"}
