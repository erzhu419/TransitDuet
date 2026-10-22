"""One prospective contract for the four prepared evidence items; no execution."""

from copy import deepcopy
import math

from .plan_validity_reliability import FAMILY_ALPHA, PROFILES
from .pointmaze_budgeted_trigger import build_parser


TASKS = ("pointmaze", "reacher_tracking")
PRIMARY_COMPARATORS = ("threshold", "termination_history")
CONTRACT = {
    "horizon": 1200, "dt_seconds": 0.01, "period_steps": 50,
    "check_stride_steps": 5, "deadline_steps": 25, "upper_calls": 24,
    "branch_window_steps": 50, "threshold_quantile": 0.75,
    "ridge_alpha_grid": [0.01, 0.1, 1, 10, 100, 1000, 10000],
    "controller_hidden_dim": 128, "termination_hidden_dim": 64,
    "learning_rate": 0.0003, "checkpoint_evaluation_interval": 8,
    "history_seconds": 0.64, "fast_period_seconds": 0.04, "maximum_subgoal_delta": 0.75,
    "ppo_epochs": 4, "ppo_minibatch_size": 1024,
    "termination_rounds": 2, "termination_epochs": 200, "effective_df": 8,
    "primary_iterations": 384, "primary_events_per_class": 4, "primary_pairs_per_path": 12,
    "controller_schedule": "balanced_jitter", "candidate": "history",
    "renewal_target": "keep_minus_renew_50_steps_no_later_upper_calls",
}
STATISTICS = {
    "global_alpha": 0.05, "task_comparator_family_alpha": FAMILY_ALPHA,
    "primary_comparators": list(PRIMARY_COMPARATORS), "profiles": list(PROFILES),
    "confirmation_roots_per_task": 8, "evaluation_paths_per_root": 512,
    "probe_paths_per_root": 8, "probes_per_path": 3, "conditional_future_replicas": 4,
    "risk_duration_steps": 25, "risk_rate_margin": 0.02, "tail_ise_margin": 0.0,
    "primary_deployment_episodes": 100, "latency_p99_cap_seconds": 0.01,
}


def draft_plan():
    parser = build_parser()
    driver_names = ("regime_dwell_seconds", "target_speed_modes", "force_pulse_amplitude",
        "force_pulse_duration_seconds", "force_pulse_gap_seconds", "distractor_amplitude", "distractor_dwell_seconds")
    return {"protocol_version": "plan_validity_unified_preregistration_v1",
        "status": "draft_pending_development_calibration", "scheduler_submission": False,
        "contract": deepcopy(CONTRACT), "statistics": deepcopy(STATISTICS),
        "pointmaze_env_id": parser.get_default("env_id"),
        "task_driver_options": {key: deepcopy(parser.get_default(key)) for key in driver_names},
        "seed_recipe": {"base": 1970000000,
            "phase_codes": {"preflight": 1, "cost_pilot": 2, "confirmation": 3},
            "task_codes": {"pointmaze": 1, "reacher_tracking": 2},
            "role_offsets": {"train_seeds": 1000, "selection_seeds": 2000,
                "branch_fit_seeds": 3000, "timing_selection_seeds": 4000, "trigger_eval_seeds": 5000},
            "phase_roots": {"preflight": 1, "cost_pilot": 2, "confirmation": 8}},
        "calibration": {"historical_seed_exposure_audit_complete": False,
            "pilot_reports": [], "recipe_frozen": False, "artifacts_frozen": False,
            "per_task": {task: {"cpu_cap_seconds": None, "environment_step_cap": None,
                "risk_distance": None, "termination_development_eligible": None} for task in TASKS},
            "artifact_bindings": []}}


def seed_cells(plan):
    recipe = plan["seed_recipe"]
    for phase, phase_code in recipe["phase_codes"].items():
        for task, task_code in recipe["task_codes"].items():
            for index in range(1, recipe["phase_roots"][phase] + 1):
                root = recipe["base"] + phase_code * 1000000 + task_code * 100000 + index * 10000
                paths = STATISTICS["evaluation_paths_per_root"] if phase == "confirmation" else 16
                roles = {}
                for role, offset in recipe["role_offsets"].items():
                    count = paths if role == "trigger_eval_seeds" else (
                        (3 if role == "branch_fit_seeds" else 2) if phase == "preflight" else 8)
                    roles[role] = list(range(root + offset, root + offset + count))
                held = roles["trigger_eval_seeds"]
                probes = held[:4] + held[len(held) // 2:len(held) // 2 + 4]
                yield {"phase": phase, "task": task, "optimizer_seed": root,
                       "seed_roles": roles, "probe_seeds": probes}


def validate_plan(plan):
    if plan["contract"] != CONTRACT or plan["statistics"] != STATISTICS or plan["scheduler_submission"]:
        raise ValueError("the registered algorithm/statistical contract changed; revise the protocol explicitly")
    expected_recipe = draft_plan()["seed_recipe"]
    if plan["seed_recipe"] != expected_recipe:
        raise ValueError("seed namespace/counts differ from the proposed reservation")
    seen, cells = set(), list(seed_cells(plan))
    for cell in cells:
        values = [cell["optimizer_seed"], *(seed for seeds in cell["seed_roles"].values() for seed in seeds)]
        if len(set(values)) != len(values) or seen.intersection(values):
            raise ValueError("seed roles, roots or phases overlap")
        seen.update(values)
    if len(TASKS) * len(PRIMARY_COMPARATORS) * FAMILY_ALPHA != STATISTICS["global_alpha"]:
        raise ValueError("task/comparator families exceed the global alpha allocation")
    calibration, pending = plan["calibration"], []
    if set(calibration["per_task"]) != set(TASKS):
        raise ValueError("calibration must cover both registered tasks")
    reference = draft_plan()
    if plan["task_driver_options"] != reference["task_driver_options"] or plan["pointmaze_env_id"] != reference["pointmaze_env_id"]:
        raise ValueError("physical task defaults differ from the registered implementation")
    if not calibration["historical_seed_exposure_audit_complete"]:
        pending.append("historical exposure audit of reserved/effective training seeds")
    if not calibration["pilot_reports"]:
        pending.append("measured cost/competence pilot reports")
    for task, row in calibration["per_task"].items():
        for name in ("cpu_cap_seconds", "environment_step_cap", "risk_distance"):
            value = row[name]
            if value is None:
                pending.append(f"{task}: {name}")
            elif not math.isfinite(value) or value <= 0:
                raise ValueError(f"{task}: invalid {name}")
        if row["termination_development_eligible"] is not True:
            pending.append(f"{task}: competent learned-termination development gate")
    if not calibration["recipe_frozen"]:
        pending.append("recipe/budget/risk/seed lock before confirmation-root training")
    bindings = calibration["artifact_bindings"]
    expected = {(cell["task"], cell["optimizer_seed"], method) for cell in cells
        if cell["phase"] == "confirmation" for method in ("history", *PRIMARY_COMPARATORS)}
    actual = [(row["task"], row["optimizer_seed"], row["method"]) for row in bindings]
    if len(set(actual)) != len(actual) or set(actual) - expected:
        raise ValueError("artifact bindings duplicate or leave the confirmation cohort")
    if set(actual) != expected:
        pending.append("all 48 frozen primary controller/rule bindings and budget admissions")
    if any(not row["budget_admitted"] or not row["controller_checkpoint"] or not row["source_report"]
           or not row["run_id"] for row in bindings):
        pending.append("binding evidence for budget admission and saved artifacts")
    if not calibration["artifacts_frozen"]:
        pending.append("global artifact lock before any confirmation-path access")
    alpha_i = FAMILY_ALPHA / (len(PROFILES) * STATISTICS["confirmation_roots_per_task"] * 2)
    zero_event_minimum = math.ceil(math.log(alpha_i) / math.log1p(-STATISTICS["risk_rate_margin"]))
    return {"status": "draft_not_locked" if pending else "ready_for_confirmation_plan_review",
        "pending": pending, "seed_cells": len(cells), "unique_reserved_seed_ids": len(seen),
        "confirmation_optimizer_roots": {task: [cell["optimizer_seed"] for cell in cells
            if cell["phase"] == "confirmation" and cell["task"] == task] for task in TASKS},
        "zero_harm_event_minimum_paths_at_registered_precision": zero_event_minimum,
        "performance_claim_supported": False, "scheduler_submission": False}


def registered_claims(plan, summaries):
    readiness = validate_plan(plan)
    expected = {(task, comparator) for task in TASKS for comparator in PRIMARY_COMPARATORS}
    by_key = {}
    for row in summaries:
        if len(row["tasks"]) != 1:
            raise ValueError("summaries must remain separate by task/comparator")
        key = row["tasks"][0], row["comparator_method"]
        if key not in expected or key in by_key:
            raise ValueError("summary duplicates or leaves the primary comparison family")
        if (row["family_alpha"] != FAMILY_ALPHA or row["risk_rate_margin"] != STATISTICS["risk_rate_margin"]
            or row["tail_ise_margin"] != STATISTICS["tail_ise_margin"]):
            raise ValueError("summary alpha/margins differ from the unified contract")
        roots = readiness["confirmation_optimizer_roots"][key[0]]
        if row["expected_roots"] != roots:
            raise ValueError("summary uses a different optimizer-root roster")
        panels = {panel["profile"]: panel for panel in row["panels"]}
        if len(panels) != len(row["panels"]) or set(panels) != set(PROFILES):
            raise ValueError("summary must retain all registered stress panels")
        for panel in panels.values():
            if panel["task"] != key[0]:
                raise ValueError("summary panel changes task")
            if panel["status"] == "analyzed" and [r["optimizer_seed"] for r in panel["root_risk_counts"]] != roots:
                raise ValueError("analyzed panel drops or replaces a registered root")
            if panel["status"] == "analyzed" and any(r["episodes"] != STATISTICS["evaluation_paths_per_root"]
                                                     for r in panel["root_risk_counts"]):
                raise ValueError("analyzed panel changes the registered episode count")
        by_key[key] = panels
    missing = sorted(expected - set(by_key))
    status = "unlocked_protocol" if readiness["pending"] else "incomplete" if missing else "analyzed"
    return {"status": status, "missing_task_comparators": [list(key) for key in missing],
        "C1_supported": None if status != "analyzed" else all(
            panels["in_distribution"].get("joint_reliability_supported", False) for panels in by_key.values()),
        "C2_supported": None if status != "analyzed" else all(
            panel.get("joint_reliability_supported", False) for panels in by_key.values() for panel in panels.values())}
