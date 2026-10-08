"""Isolate native mean-option probe radius from policy noise and step size."""

from scripts import pointmaze_mean_option_query_stage141_spec as source

ROOT, ROOTS, PERIODS = source.ROOT, source.ROOTS, source.PERIODS
PROTOCOL = "pointmaze_local_option_probe_stage142_v1"
EXPERIMENT_PROTOCOL = PROTOCOL
POLICY = "cached_wide_vs_local_mean_option_probes_fixed_policy_std_and_mean_step"
RUNNER_SCRIPT = "scripts/run_pointmaze_local_option_probe_stage142.py"
SOURCE_RUN = "pointmaze_mean_option_query_stage141_probe_20261008_r1"
WORKERS, EVALUATION_EPISODES = 16, 32
PANELS, MODES = source.PANELS, source.MODES
STD, RADIUS, MEAN_STEP_RMS = source.STD, source.RADIUS, source.MEAN_STEP_RMS
PROBE_SCALES = {"wide": 1., "local": .1}
METHODS = tuple(PROBE_SCALES)
VARIANTS = ("source_forecast", "warm_start", *tuple(m+"_"+s for m in METHODS for s in ("plus", "minus")))
CONTRASTS = (("warm_start", "source_forecast"),
    *tuple((m+"_"+s, "warm_start") for m in METHODS for s in ("plus", "minus")),
    ("local_plus", "wide_plus"), *tuple((m+"_plus", m+"_minus") for m in METHODS),
    *tuple((m+"_plus", "source_forecast") for m in METHODS))
arguments, warm_result = source.arguments, source.warm_result


def source_result(root):
    return ROOT/"results"/SOURCE_RUN/"cells"/f"replicate_{root}"/"result.json"


def seed_roles(root):
    base = 142100000+ROOTS.index(root)*100000
    return {"replayed_training": source.seed_roles(root)["replayed_training"],
        "evaluation": list(range(base+90001, base+90001+EVALUATION_EPISODES))}


def budget():
    return {**source.budget(), "score_gradient_batches": 0, "mean_score_forward_batches": 0}


def contract():
    return {"source": SOURCE_RUN, "warm_source": source.WARM_SOURCE_RUN, "policy_std": STD,
        "probe_std": {m: STD*s for m, s in PROBE_SCALES.items()}, "mean_step_RMS": MEAN_STEP_RMS,
        "KL_radius": RADIUS, "geometry": "same_standardized_damped_mean_geometry_damping1",
        "pairing": "identical_mean_training_paths_and_probe_innovations_new_local_queries_cached_wide_labels",
        "freeze": "warm_mean_lower_values_teacher_forecaster_policy_std_authority005",
        "evaluation": "new_paired_mean_and_sampled_scenes_both_signs_no_winner_or_CI",
        "artifacts": "compact_JSON_full_replay_and_new_query_cost_no_checkpoints_or_traces_inherited_cost_separate",
        "limits": "curvature_is_not_proof_of_gradient_bias_query_radius_diagnosis_not_joint_HRL_confirmation"}
