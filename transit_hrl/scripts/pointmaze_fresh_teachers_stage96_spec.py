"""Fresh native initializers and BC teachers, without reward-based admission."""

from argparse import Namespace
from pathlib import Path
import math

import numpy as np
from scripts import pointmaze_budgeted_trigger_stage9_spec as native

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PROTOCOL = "pointmaze_fresh_teachers_stage96_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_fresh_teachers_stage96.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_fresh_teachers_stage96.py"
POLICY, METHODS, ENDPOINTS = "fresh_teachers", ("native_initializer", "clone"), ()
OPTIMIZER_ROOTS = (410011, 410023, 410037, 410049, 410061, 410073, 410089, 410101)
PREFLIGHT_ROOTS, PERIODS = (410001,), (50, 100)


def roots(*, preflight):
    return PREFLIGHT_ROOTS if preflight else OPTIMIZER_ROOTS


def options(*, preflight):
    return {"controller_iterations": 2 if preflight else 384,
        "controller_paths_per_iteration": 1 if preflight else 8,
        "controller_diagnostic_paths": 2 if preflight else 16,
        "critic_warmup_iterations": 2 if preflight else 16,
        "warmup_paths_per_iteration": 1 if preflight else 8,
        "fitting_paths": 2 if preflight else 32, "label_paths": 2 if preflight else 8,
        "bc_epochs": 4 if preflight else 64, "workers": 2 if preflight else 8}


def arguments(root, *, preflight):
    roots(preflight=preflight).index(root)
    values = native.cell_options(208001 if preflight else 209011, preflight=preflight)
    for key in ("methods", "train", "selection", "branch_fit", "trigger_eval"):
        values.pop(key)
    return Namespace(**values, optimizer_seed=root, preflight=preflight)


def seed_roles(root, *, preflight):
    index = roots(preflight=preflight).index(root)
    base = 96_090_000 if preflight else 96_100_000 + index * 10000
    opt = options(preflight=preflight)
    counts = {"controller_training": opt["controller_iterations"] * opt["controller_paths_per_iteration"],
        "critic_warmup": opt["critic_warmup_iterations"] * opt["warmup_paths_per_iteration"],
        "fitting": opt["fitting_paths"], "labels": opt["label_paths"],
        "controller_diagnostic": opt["controller_diagnostic_paths"]}
    return {name: list(range(base + offset + 1, base + offset + 1 + counts[name])) for name, offset in
        (("controller_training", 0), ("critic_warmup", 3500), ("fitting", 3700),
         ("labels", 4000), ("controller_diagnostic", 5000))}


def policy_seed(root, seed, phase):
    return int(np.random.SeedSequence([96, root, seed, {"controller": 1, "warmup": 2, "labels": 3}[phase]]).generate_state(1)[0])


def lower_seed(root, seed):
    return int(np.random.SeedSequence([96, root, seed, 96019]).generate_state(1)[0])


def shuffle_seed(root, iteration, phase):
    return int(np.random.SeedSequence([96, root, iteration, int(phase == "warmup"), 96023]).generate_state(1)[0])


def budget(*, preflight):
    opt, h = options(preflight=preflight), arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    controller = opt["controller_iterations"] * opt["controller_paths_per_iteration"]
    warm = opt["critic_warmup_iterations"] * opt["warmup_paths_per_iteration"]
    labels = len(PERIODS) * opt["label_paths"]
    diagnostics = 2 * opt["controller_diagnostic_paths"]
    controller_lower = 4 * math.ceil(opt["controller_paths_per_iteration"] * h / 1024) * opt["controller_iterations"]
    controller_upper = 4 * math.ceil(opt["controller_paths_per_iteration"] * (h // 50) / 1024) * opt["controller_iterations"]
    return {"controller_training_episodes": controller, "controller_diagnostic_episodes": diagnostics,
        "critic_warmup_episodes": warm, "label_episodes": labels,
        "native_episodes": controller + diagnostics + warm + labels,
        "native_steps": (controller + diagnostics + warm + labels) * h,
        "controller_upper_actor_steps": controller_upper, "controller_upper_value_steps": controller_upper,
        "controller_lower_actor_steps": controller_lower, "controller_lower_value_steps": controller_lower,
        "warmup_actor_steps": 0,
        "warmup_lower_value_steps": 4 * math.ceil(opt["warmup_paths_per_iteration"] * h / 1024) * opt["critic_warmup_iterations"],
        "bc_optimizer_steps": len(PERIODS) * opt["bc_epochs"] * math.ceil(opt["label_paths"] * h / 1024),
        "forecaster_driver_paths": opt["fitting_paths"], "forecaster_observations": opt["fitting_paths"] * (h + 1),
        "forecaster_rows": opt["fitting_paths"] * (h - 100), "ridge_solves": 1, "riccati_solves": 1,
        "checkpoint_writes": 4, "label_archive_writes": labels, "historical_artifact_loads": 0}


def contract():
    return {"cohort": "eight_predeclared_new_optimizer_roots_all_retained",
        "preflight": "separate_root410001_not_a_confirmation_teacher",
        "native_initializer": "same_Stage33_history_MLP_PPO_and_balanced_jitter50_native_task",
        "controller_training": "384_iterations_eight_fresh_paths_each_shared_PPO_update_no_selection_rollouts",
        "controller_checkpoint": "fixed_final_iteration384_one_based_no_best_checkpoint_or_reward_admission",
        "controller_diagnostics": "same_disjoint_paths_at_initial_and_final_no_gate_no_CI",
        "critic_warmup": "Stage42_task_clock_sixteen_lower_critic_only_iterations_all_actors_and_other_values_frozen",
        "teacher": "Stage55_LQR_velocity_teacher_causal_ridge_forecaster_fresh_label_states",
        "clone": "Stage55_tanh_command_MSE_BC64_final_lower_mean_only_std_and_upper_frozen",
        "periods": list(PERIODS), "native_trace_writes": "teacher_label_archives_only_for_later_decoder_calibration",
        "historical_artifacts": "none_no_old_teacher_forecaster_decoder_or_lower_checkpoint",
        "downstream": "fresh_decoder_then_fresh_joint_and_lower_donors_then_Stage94_staged_upper_rules",
        "downstream_controls": ["zero_plan", "U0", "fixed_UJ", "matched_independent_lower"],
        "downstream_gate": "unchanged_four_corrected_primary_reward_endpoints_at_both_periods",
        "limits": "teacher_provisioning_only_not_unseen_teacher_performance_confirmation_not_frequency_superiority"}
