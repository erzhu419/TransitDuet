"""Fresh-sample replication of the frozen Stage87 call-weighted MC protocol."""

from scripts import pointmaze_call_weighted_stage87_spec as source

ROOT, PERIODS, CHUNK_SIZE, FISHER_RADIUS = source.ROOT, source.PERIODS, source.CHUNK_SIZE, source.FISHER_RADIUS
roots, arguments, source_result, options, allocation, budget = source.roots, source.arguments, source.source_result, source.options, source.allocation, source.budget
METHODS, VARIANTS, CONTRAST_PAIRS, ENDPOINTS = source.METHODS, source.VARIANTS, source.CONTRAST_PAIRS, source.ENDPOINTS
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = source.BOOTSTRAP_DRAWS, source.BOOTSTRAP_SEED
EXPERIMENT_PROTOCOL = "pointmaze_call_weighted_replication_stage88_v1"
POLICY = "call_weighted_mc_replication"
RUNNER_SCRIPT = "scripts/run_pointmaze_call_weighted_replication_stage88.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_call_weighted_replication_stage88.py"
REFERENCE_RUN = "pointmaze_call_weighted_stage87_full_20261002_r1"
PRIMARY_ENDPOINTS = tuple(f"{p}/joint_call_minus_{b}" for p in PERIODS for b in ("lower_trained", "joint_level"))


def seed_roles(root, *, preflight):
    roles = source.seed_roles(root,preflight=preflight)
    for r in roles["training_rounds"]:
        for b in ("A","B"):
            for s in r["credit_"+b]:
                s["scenario_seed"] += 1_000_000
                s["noise_seeds"] = [n+1_000_000 for n in s["noise_seeds"]]
    roles["native_evaluation"] = [s+1_000_000 for s in roles["native_evaluation"]]
    return roles


def contract():
    return {**source.contract(),
        "noise_mapping": "Stage80_explicit_mapping_fresh_disjoint_Stage88_round_and_evaluation_roles",
        "replication": "same_eight_frozen_Stage78_teachers_fresh_training_and_evaluation_samples_no_Stage87_weight_reuse",
        "reference_run": REFERENCE_RUN,
        "confirmation": "all_four_primary_CI_lower_bounds_strictly_positive_in_unchanged_Bonferroni20_family_no_cross_stage_pooling"}


def confirmation(summary):
    return {"status": "mechanical_only" if summary["status"] == "preflight_passed" else
        ("confirmed" if all(summary["endpoints"][k]["ci"][0] > 0 for k in PRIMARY_ENDPOINTS) else "not_confirmed"),
        "primary_endpoints": list(PRIMARY_ENDPOINTS), "reference_run": REFERENCE_RUN,
        "population": "same_eight_frozen_teachers_fresh_sample_replication"}
