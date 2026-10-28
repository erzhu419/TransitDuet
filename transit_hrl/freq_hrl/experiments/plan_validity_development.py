"""Registered native preflight/pilot cells and compact qualification reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .plan_validity_preregistration import PRIMARY_COMPARATORS, seed_cells, validate_plan
from .plan_validity_reliability import decompose_probe
from .plan_validity_transfer import PlanValidityTaskSpec, jitter_schedule, run_episode, time_contract
from .pointmaze_goal_validation import _json_ready
from .pointmaze_plan_validity_branching import _task_options
from .pointmaze_timing_baselines import TimingRule


PROTOCOL = "plan_validity_native_development_v1"
PHASES = ("preflight", "cost_pilot")
METHODS = ("history", *PRIMARY_COMPARATORS)


def development_cells(plan, phase):
    validate_plan(plan)
    if phase not in PHASES:
        raise ValueError("only preflight and cost_pilot may use this development entry point")
    return [{**cell, "method": method} for cell in seed_cells(plan) if cell["phase"] == phase for method in METHODS]


def cell_directory(run_name, cell):
    if not run_name or Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("use a single campaign directory name")
    return Path("results") / run_name / cell["phase"] / cell["task"] / str(cell["optimizer_seed"]) / cell["method"]


def measurement_path(run_name, cell):
    name = f"{run_name}_{cell['phase']}_{cell['task']}_{cell['method']}_{cell['optimizer_seed']}"
    return cell_directory(run_name, cell) / "private" / f"{name}.json"


def cell_arguments(plan, run_name, cell, *, root):
    from scripts.run_plan_validity_compute_budget import build_compute_parser, prepare

    preflight = cell["phase"] == "preflight"
    contract = plan["contract"]
    values = {"env_id": plan["pointmaze_env_id"], "optimizer_seed": cell["optimizer_seed"],
        "horizon": 300 if preflight else contract["horizon"],
        "iterations": 2 if preflight else contract["primary_iterations"],
        "max_events_per_class": 1 if preflight else contract["primary_events_per_class"],
        "pairs_per_path": 2 if preflight else contract["primary_pairs_per_path"],
        "reference_hidden_dim": contract["controller_hidden_dim"],
        "termination_hidden_dim": contract["termination_hidden_dim"],
        "timing_rounds": contract["termination_rounds"], "termination_epochs": contract["termination_epochs"],
        "effective_df": contract["effective_df"], "threshold_quantile": contract["threshold_quantile"],
        "learning_rate": contract["learning_rate"],
        "checkpoint_evaluation_interval": contract["checkpoint_evaluation_interval"],
        "history_seconds": contract["history_seconds"], "fast_period_seconds": contract["fast_period_seconds"],
        "maximum_subgoal_delta": contract["maximum_subgoal_delta"],
        "upper_period_seconds": contract["period_steps"] * contract["dt_seconds"],
        "branch_window_seconds": contract["branch_window_steps"] * contract["dt_seconds"],
        "max_offset_steps": contract["deadline_steps"], "check_stride_steps": contract["check_stride_steps"],
        "ridge_alpha_grid": contract["ridge_alpha_grid"], **plan["task_driver_options"], **cell["seed_roles"]}
    command = ["--tasks", cell["task"], "--method", cell["method"], "--execute",
               "--output", str(Path(root) / measurement_path(run_name, cell))]
    for name, value in values.items():
        command.extend(["--" + name.replace("_", "-"), *map(str, value if isinstance(value, (list, tuple)) else [value])])
    args = build_compute_parser().parse_args(command)
    prepare(args)
    return args


def expected_runs(plan, run_name, phase):
    return [{"run_id": f"{measurement_path(run_name, cell).stem}:{cell['task']}:{cell['method']}:{cell['optimizer_seed']}",
             "task": cell["task"], "method": cell["method"], "optimizer_seed": cell["optimizer_seed"]}
            for cell in development_cells(plan, phase)]


def native_preflight(record):
    from scripts.run_plan_validity_reliability import load_controller

    controller, args = load_controller(record)
    scale = time_contract(args)
    spec = PlanValidityTaskSpec(record["task"], _task_options(args), args.env_id)
    rule = TimingRule(**record["selected_rule"])
    def run(seed, **kwargs):
        return run_episode(controller, task_spec=spec, seed=seed, args=args, time_scale=scale, **kwargs)

    replays = [run(seed, rule=rule)[1] for seed in args.timing_selection_seeds]
    if (not np.isclose(np.mean([r["tracking_squared_error_integral"] for r in replays]), record["validation_ise"],
                       rtol=1e-10, atol=1e-10)
        or not np.isclose(np.mean([r["episode_return"] for r in replays]), record["validation_return"], rtol=1e-10, atol=1e-10)):
        raise RuntimeError("saved controller does not reproduce development validation")
    seed = args.trigger_eval_seeds[0]
    batch, sampled = run(seed, schedule=jitter_schedule(args, seed), sample=True)
    durations = np.diff([*sampled["decision_steps"], args.horizon])
    boundaries = np.array([step - 1 for step in sampled["decision_steps"][1:]] + [args.horizon - 1])
    if (not np.array_equal(batch.upper.duration, durations)
        or not np.array_equal(np.flatnonzero(batch.lower.done), boundaries)):
        raise RuntimeError("native actual-duration credit or lower option boundaries differ")
    risk = {"distance": 1.0, "duration_steps": 25}
    factual = run(seed, rule=rule, risk_spec=risk)[1]
    step = 50
    def paired(future=None):
        local = [run(seed, rule=rule, branch_step=step, renew=value, continuation_seed=future)[1] for value in (False, True)]
        full = [run(seed, rule=rule, intervention_step=step, arm=arm,
                    continuation_seed=future, risk_spec=risk)[1] for arm in ("now", "wait_one_check")]
        return local, full, decompose_probe(*local, *full)

    local, full, effect = paired()
    check = next(row for row in factual["queries"] if row["step"] == step)
    replay = full[0] if check["stop"] else full[1]
    if (replay["decision_steps"] != factual["decision_steps"] or
        not np.isclose(replay["tracking_squared_error_integral"], factual["tracking_squared_error_integral"], rtol=1e-10, atol=1e-10)):
        raise RuntimeError("native intervention does not reproduce its factual arm")
    future = int(np.random.SeedSequence([args.optimizer_seed, seed, step, 403]).generate_state(1)[0])
    future_local, future_full, _ = paired(future)
    if any(not np.array_equal(row["prefix"], local[0]["prefix"]) for row in (*future_local, *future_full)):
        raise RuntimeError("native conditional future changed its prefix")
    return {"passed": True, "saved_model_replay": True, "actual_duration_credit": True,
        "identical_prefix_and_factual_replay": True, "conditional_future_prefix": True,
        "tracking_risk_fields": factual["reliability"], "risk_definition_role": "telemetry_fixture_not_operational_calibration",
        "paired_effect": effect, "upper_calls": factual["upper_decision_count"],
        "minimum_option_steps": factual["option_duration_steps_min"],
        "maximum_option_steps": factual["option_duration_steps_max"],
        "renew_waypoint_change": float(np.linalg.norm(local[1]["branch_subgoal"] - local[0]["old_subgoal"]))}


def summarize_phase(plan, run_name, phase, reports):
    from .plan_validity_preregistration import TASKS

    cells = development_cells(plan, phase)
    roster = expected_runs(plan, run_name, phase)
    expected = {row["run_id"]: cell for row, cell in zip(roster, cells)}
    seen, failures, profiles, records = set(), [], [], []
    for report in reports:
        if report["stage"] != phase or report["run_name"] != run_name or len(report["records"]) != 1:
            raise ValueError("development report belongs to another campaign/phase")
        record = report["records"][0]
        run_id = record["run_id"]
        if run_id in seen or run_id not in expected:
            raise ValueError("duplicate or unregistered development record")
        seen.add(run_id)
        args = cell_arguments(plan, run_name, expected[run_id], root=Path("."))
        configuration = _json_ready({key: value for key, value in vars(args).items() if key not in ("output", "dry_run")})
        if record["configuration"] != configuration:
            raise ValueError("measured arguments/seed roles differ from the registered cell")
        if record["status"] != "validation_complete" or not record["accounting_complete"]:
            failures.append(run_id)
        elif phase == "preflight" and not report["native_preflight"]["passed"]:
            failures.append(run_id)
        else:
            profiles.append(record["hardware_profile"])
        records.append(record)
    missing = sorted(set(expected) - seen)
    matched = bool(profiles) and all(profile == profiles[0] for profile in profiles)
    complete = not missing and not failures and matched
    eligibility = {task: all(row["termination_readiness"]["eligible_for_fresh_confirmation"] for row in records
        if row["task"] == task and row["method"] == "termination_history") and
        sum(row["task"] == task and row["method"] == "termination_history" for row in records) ==
        plan["seed_recipe"]["phase_roots"][phase] for task in TASKS} if complete and phase == "cost_pilot" else None
    return {"protocol_version": PROTOCOL, "stage": phase, "run_name": run_name,
        "plan_contract": plan["contract"], "expected_run_ids": sorted(expected),
        "task_driver_options": plan["task_driver_options"], "pointmaze_env_id": plan["pointmaze_env_id"],
        "seed_recipe": plan["seed_recipe"],
        "status": "complete" if complete else "incomplete_or_failed", "missing": missing, "failed": failures,
        "hardware_profiles_matched": matched, "hardware_profile": profiles[0] if matched else None,
        "preflight_passed": complete if phase == "preflight" else False,
        "termination_development_eligible": eligibility,
        "known_measured_cpu_seconds": sum(row["offline_cost"]["cpu_seconds"] for row in records),
        "performance_claim_supported": False}


def require_preflight_gate(plan, run_name, gate):
    expected = sorted(row["run_id"] for row in expected_runs(plan, run_name, "preflight"))
    if (gate["protocol_version"] != PROTOCOL or gate["stage"] != "preflight" or gate["run_name"] != run_name
        or gate["plan_contract"] != plan["contract"] or gate["expected_run_ids"] != expected
        or gate["seed_recipe"] != plan["seed_recipe"] or gate["task_driver_options"] != plan["task_driver_options"]
        or gate["pointmaze_env_id"] != plan["pointmaze_env_id"]
        or gate["status"] != "complete" or not gate["preflight_passed"] or not gate["hardware_profiles_matched"]):
        raise ValueError("a complete matching native preflight gate is required before pilot")


def load_plan(path):
    plan = json.loads(Path(path).read_text())["plan"]
    validate_plan(plan)
    return plan
