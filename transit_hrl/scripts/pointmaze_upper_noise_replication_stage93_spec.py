"""Fresh-sample upper-noise conditioning with retrained paired controls."""

from scripts import pointmaze_upper_common_noise_stage92_spec as source
from scripts.pointmaze_iterative_mc_stage83_spec import training_budget

ROOT, PERIODS, CHUNK_SIZE, FISHER_RADIUS = source.ROOT, source.PERIODS, source.CHUNK_SIZE, source.FISHER_RADIUS
roots, arguments, source_result, options = source.roots, source.arguments, source.source_result, source.options
EXPERIMENT_PROTOCOL = "pointmaze_upper_noise_replication_stage93_v1"
POLICY = "fresh_lower_upper_noise_replication"
RUNNER_SCRIPT = "scripts/run_pointmaze_upper_noise_replication_stage93.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_upper_noise_replication_stage93.py"
REFERENCE_RUN = "pointmaze_upper_common_noise_stage92_full_20261002_r1"
CHECKPOINT_METHODS = ("joint_call",)
METHODS = {m:("lower",) for m in (
    "source_upper_independent", "source_upper_common", "joint_upper_independent", "joint_upper_common")}
NOISE_MODES = {m:("common_upper_independent_lower" if m in source.METHODS else "independent_upper_and_lower") for m in METHODS}
CONTROLS = {"lower_matched":"source_upper_independent", "lower_fixed_upper":"joint_upper_independent"}
COMPOSITIONS = {v:(upper,CONTROLS.get(lower,lower)) for v,(upper,lower) in source.COMPOSITIONS.items()}
VARIANTS, CONTRAST_PAIRS, ENDPOINTS, PRIMARY_ENDPOINTS = source.VARIANTS, source.CONTRAST_PAIRS, source.ENDPOINTS, source.PRIMARY_ENDPOINTS
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (93, 93093)


def donor_result(root, method):
    if method != "joint_call":raise ValueError("Stage93 only reuses the fixed Stage88 upper donor")
    return source.donor_result(root,method)


def allocation(method, period):
    if method not in METHODS:raise ValueError("Stage93 trains only the four registered lower means")
    return source.allocation("source_upper_common",period)


def seed_roles(root, *, preflight):
    roles = source.seed_roles(root,preflight=preflight)
    shift = 4_900_000 if preflight else 5_000_000
    for r in roles["training_rounds"]:
        for b in ("A","B"):
            for s in r["credit_"+b]:
                s["scenario_seed"] += shift
                s["noise_seeds"] = [n+shift for n in s["noise_seeds"]]
    base = 93_090_000 if preflight else 93_195_000 + roots(preflight=False).index(root)*100000
    roles["native_evaluation"] = list(range(base+1,base+1+options(preflight=preflight)["evaluation_episodes"]))
    return roles


def budget(*, preflight):
    h = arguments(roots(preflight=preflight)[0],preflight=preflight).horizon
    o = options(preflight=preflight)
    n = 2*o["credit_scenarios_per_batch"]*o["rollouts_per_scenario"]
    return {**training_budget(o,METHODS,VARIANTS,horizon=h,preflight=preflight),
        "checkpoint_loads":len(PERIODS), "checkpoint_freeze_checks":len(PERIODS),
        "training_initialization_checks":len(PERIODS)*len(METHODS),
        "actor_composition_checks":len(PERIODS)*len(VARIANTS),
        "upper_replay_forward_calls":len(source.METHODS)*o["updates"]*(n//2)*sum(h//p for p in PERIODS)}


def contract():
    common = source.contract()
    return {**common, "reference_run":REFERENCE_RUN,
        "methods":{str(p):{m:allocation(m,p) for m in METHODS} for p in PERIODS},
        "training_runs":{"joint_call":common["training_runs"]["joint_call"]},
        "checkpoint_protocols":{"joint_call":common["checkpoint_protocols"]["joint_call"]},
        "checkpoint_methods":list(CHECKPOINT_METHODS),
        "compositions_upper_lower":{v:list(pair) for v,pair in COMPOSITIONS.items()},
        "training":"four_new_lower_means_from_L0_U0_or_final_Stage88_UJ_frozen_independent_and_common_upper_noise_eight_updates_preflight_two",
        "sampling":"fresh_Stage93_scenarios_and_lower_noise_all_four_methods_same_rosters_64_episodes_per_update_common_first_replica_unchanged_second_upper_innovations_replayed",
        "noise_mapping":"Stage80_mapping_new_Stage93_training_and_eval_roles_independent_controls_use_legacy_path_common_training_only_full_upper_replay_eval_legacy_path",
        "training_noise_pairing":NOISE_MODES,
        "budget":"same_per_learner_lower_KL_.00099_at50_.000995_at100_and64_episodes_8updates_no_search_independent_controls_retrained",
        "statistics":"all26_reward_contrasts_equal_root_bootstrap65536_Bonferroni26_seed93_93093_covariance_descriptive_only",
        "decision":"fresh_sample_confirmation_requires_all_four_primary_lower_CI_bounds_positive_no_cross_stage_pooling_or_period_selection_preflight_mechanical_only",
        "replication":"same_eight_frozen_teachers_and_Stage88_UJ_new_training_and_eval_samples_no_Stage90_91_92_lower_weight_reuse",
        "limits":"conditional_lower_learning_not_new_teachers_or_joint_HRL_shared_upper_innovations_not_fixed_actions_extra_replay_compute_frequency_superiority_closed_Stage67_HOLD"}
