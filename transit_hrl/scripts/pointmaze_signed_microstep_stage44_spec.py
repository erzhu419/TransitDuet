"""Fixed signed microsteps along the Stage-42 first lower update."""

import numpy as np
from scripts import pointmaze_update_direction_stage43_spec as source

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_signed_microstep_stage44_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_signed_microstep_stage44.py"
POLICY = "signed_microstep"
METHODS, MODES, METRICS = source.METHODS, source.MODES, source.METRICS
MICROSTEP = 1. / 16.
SCALES = {"plus_micro": MICROSTEP, "minus_micro": -MICROSTEP, "full": 1.}
POLICIES = ("frozen", *(f"{m}:{v}" for m in METHODS for v in SCALES))
SOURCE_FULL_RUN, SOURCE_PREFLIGHT_RUN = source.SOURCE_FULL_RUN, source.SOURCE_PREFLIGHT_RUN
source_result, roots = source.source_result, source.roots
ENDPOINTS = tuple(f"{m}:{k}" for m in METHODS for k in
                  ("plus_vs_zero", "minus_vs_zero", "signed_slope", "full_vs_plus"))
CI_FAMILY_SIZE = len(ENDPOINTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (44, 44044)


def options(*, preflight):
    opt = source.options(preflight=preflight)
    return {"evaluation_paths": opt["evaluation_paths"], "workers": opt["workers"],
            "horizon": source.source.source.arguments(roots(preflight=preflight)[0], preflight=preflight).horizon}


def seed_roles(root, *, preflight):
    index = roots(preflight=preflight).index(root)
    base = 10_890_000 if preflight else 10_900_000 + index * 10000
    return {"evaluation": list(range(base + 3001, base + 3001 + options(preflight=preflight)["evaluation_paths"]))}


def policy_seed(root, seed):
    return int(np.random.SeedSequence([44, root, seed, 44017]).generate_state(1)[0])


def rollout_arguments(root, seed, *, mode):
    sampled = mode == "lower_sampled"
    return {"sample": False, "upper_sample": False, "gate_sample": False, "lower_sample": sampled,
            "gate_seed": None,
            "lower_seed": int(np.random.SeedSequence([44, root, seed, 44019]).generate_state(1)[0]) if sampled else None}


def budget(*, preflight):
    opt = options(preflight=preflight)
    traces = len(POLICIES) * len(MODES) * opt["evaluation_paths"]
    return {"evaluation_primitive_steps": traces * opt["horizon"], "total_primitive_steps": traces * opt["horizon"],
            "native_trace_audits": traces, "optimizer_steps": 0}


def contrasts(means):
    reference = means["frozen"]["episode_return"]
    values = {}
    for method in METHODS:
        plus, minus, full = (means[f"{method}:{v}"]["episode_return"] for v in SCALES)
        for key, value in zip(("plus_vs_zero", "minus_vs_zero", "signed_slope", "full_vs_plus"),
                              (plus - reference, minus - reference, (plus - minus) / (2 * MICROSTEP), full - plus)):
            values[f"{method}:{key}"] = value
    return values


def contract():
    return {"source_protocol": source.source.EXPERIMENT_PROTOCOL, "methods": list(METHODS),
            "checkpoints": "fixed_last_warmup_and_first_lower_update_no_selection",
            "lower_actor": "before_plus_alpha_times_after_minus_before_all_parameters_including_log_std",
            "scales": SCALES, "baseline_scale": 0., "other_networks": "unchanged_each_before_checkpoint",
            "baseline": "shared_pre_update_actor_after_exact_actor_and_frozen_level_equality",
            "evaluation": "fresh_paired_paths_deterministic_upper_gate_deterministic_or_sampled_lower",
            "lower_sampling_seed": "numpy_seedsequence_44_root_environment_seed_44019_plus_primitive_step",
            "primary_mode": "lower_sampled", "descriptive_mode": "deterministic",
            "primary_endpoints": list(ENDPOINTS), "signed_slope": "finite_central_difference_not_exact_derivative",
            "optimizer_steps": 0, "checkpoint_selection": "none", "root_exclusion": "forbidden",
            "scale_selection": "forbidden", "sequential_extension": "forbidden",
            "bootstrap_seed": list(BOOTSTRAP_SEED), "bootstrap_draws": BOOTSTRAP_DRAWS,
            "interval": "two_sided_percentile_bonferroni_16_paired_equal_root_means",
            "decision": "positive_micro_or_signed_slope_and_negative_full_vs_plus_supports_bounded_finite_step_attenuation",
            "evidence_role": "conditional_diagnosis_existing_development_weights_with_fresh_evaluation_not_confirmation"}
