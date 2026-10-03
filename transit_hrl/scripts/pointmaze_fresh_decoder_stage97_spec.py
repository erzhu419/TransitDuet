"""Freeze command-bounded decoders on the fresh Stage96 teacher cohort."""

import copy
import math
import numpy as np
from scripts import pointmaze_fresh_teachers_stage96_spec as source

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_fresh_decoder_stage97_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_fresh_decoder_stage97.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_fresh_decoder_stage97.py"
POLICY, METHODS, ENDPOINTS = "fresh_decoder", ("calibration",), ()
PERIODS, MODES = source.PERIODS, ("zero", "bounded")
CHUNK_SIZE = 1024
SOURCE_RUN = "pointmaze_fresh_teachers_stage96_full_20261003_r1"


def roots(*, preflight):
    return source.OPTIMIZER_ROOTS[:1] if preflight else source.OPTIMIZER_ROOTS


def options(*, preflight):
    return {"probe_episodes_per_mode_period": 4, "workers": 2 if preflight else 8}


def arguments(root, *, preflight):
    roots(preflight=preflight).index(root)
    args = copy.copy(source.arguments(root, preflight=False))
    args.preflight, args.horizon = preflight, 300 if preflight else 1200
    return args


def source_result(root):
    return ROOT / "results" / SOURCE_RUN / "cells" / f"replicate_{root}" / "result.json"


def label_archive(root, period, seed):
    file = source_result(root)
    return file.parent.with_name(file.parent.name + "_raw") / str(period) / "teacher" / "labels" / "0" / "deterministic" / f"episode_{seed}.npz"


def seed_roles(root, *, preflight):
    base = 97_090_000 if preflight else 97_100_000 + roots(preflight=False).index(root) * 10000
    return {"calibration_labels": source.seed_roles(root, preflight=False)["labels"],
        "native_probe": list(range(base + 5001, base + 5001 + options(preflight=preflight)["probe_episodes_per_mode_period"]))}


def proposal_seed(root, seed, period):
    return int(np.random.SeedSequence([97, root, seed, period, 97017]).generate_state(1)[0])


def noise_seeds(root, seed):
    return tuple(int(np.random.SeedSequence([97, root, seed, tag]).generate_state(1)[0]) for tag in (97019, 97029))


def budget(*, preflight):
    root = roots(preflight=preflight)[0]
    roles = seed_roles(root, preflight=preflight)
    h, label_h = arguments(root, preflight=preflight).horizon, source.arguments(root, preflight=False).horizon
    n, e, periods = len(roles["calibration_labels"]), len(roles["native_probe"]), len(PERIODS)
    episodes = periods * len(MODES) * e
    return {"source_clone_loads": periods, "forecaster_loads": 1, "layout_loads": 1,
        "label_archive_loads": periods * n, "label_state_rows": periods * n * label_h,
        "label_plan_checks": periods * n, "bc_mse_reproductions": periods,
        "calibration_curve_decodes": n * sum(label_h // p for p in PERIODS),
        "calibration_upper_proposals": n * sum(label_h // p for p in PERIODS),
        "constraint_response_evaluations": periods, "offline_actor_rows": 3 * periods * n * label_h,
        "offline_actor_forward_batches": 3 * periods * n * math.ceil(label_h / CHUNK_SIZE),
        "frozen_model_checks": periods, "native_episodes": episodes, "native_steps": episodes * h,
        "native_upper_calls": len(MODES) * e * sum(h // p for p in PERIODS),
        "native_lower_calls": episodes * h, "native_network_checks": episodes,
        "native_pair_checks": periods * e, "optimizer_steps": 0, "checkpoint_writes": 0, "native_trace_writes": 0}


def realized_budget(cell, *, preflight):
    b = budget(preflight=preflight)
    label_h = source.arguments(cell["root"], preflight=False).horizon
    n = len(cell["seed_roles"]["calibration_labels"])
    evaluations = sum(len(g["calibration"]["solver_trace"]) for g in cell["groups"].values())
    b.update(constraint_response_evaluations=evaluations,
        offline_actor_rows=(2 * len(PERIODS) + evaluations) * n * label_h,
        offline_actor_forward_batches=(2 * len(PERIODS) + evaluations) * n * math.ceil(label_h / CHUNK_SIZE))
    return b


def contract():
    return {"source_protocol": source.EXPERIMENT_PROTOCOL, "source_run": SOURCE_RUN,
        "source": "full_new_root_teachers_and_forecasters_only_even_in_preflight",
        "labels": "all_eight_full_BC_label_paths_per_period_reconstruct_causal_lower_states_reproduce_saved_command_MSE",
        "proposals": "fixed_root_label_period_seed_zero_upper_mean_source_Gaussian_std_no_reward_or_validation_fit",
        "constraint": "Stage78_same_first_feasible_halving_from_min(1,BC_RMSE/original_command_change_RMS)",
        "response": "actual_nonlinear_lower_command_change_RMS_leq_BC_command_RMSE",
        "curve": "same_convex_blend_world_clipped_points_position_and_dt_velocity_coherent",
        "envelope": "Stage76_same_label_velocity_q99_and_axis_support_only",
        "probes": "zero_and_bounded_independent_native_noise_four_fresh_paths_per_period_sampling_both_actors_no_training",
        "decision": "mechanical_only_no_return_threshold_no_alpha_retuning_no_root_selection_no_CI",
        "selection": "both_period_decoders_frozen_before_any_native_probe_no_best_checkpoint",
        "cost": "first_feasible_solver_variable_passes_counted_exactly_no_native_traces_or_checkpoint_writes",
        "limits": "new_decoder_prerequisite_not_unseen_teacher_staged_reward_confirmation_not_frequency_superiority"}
