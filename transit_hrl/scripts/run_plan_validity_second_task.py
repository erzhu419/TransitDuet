"""Prepare the same plan-validity algorithm on PointMaze and Reacher tracking."""

from __future__ import annotations

import json

from freq_hrl.experiments.plan_validity_transfer import (
    PROTOCOL_VERSION, TASKS, PlanValidityTaskSpec, collect_renewal_labels,
    run_episode, time_contract, train_shared_controller,
)
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_plan_value_qualification import _compact_history
from freq_hrl.experiments.pointmaze_timing_baselines import collect_timing_pairs
from scripts.run_pointmaze_timing_baselines import (
    build_timing_parser, fit_timing_comparison, protocol as baseline_protocol,
)


def protocol(args):
    prepared = baseline_protocol(args)
    if len(args.tasks) != len(set(args.tasks)) or args.threshold_quantile != 0.75:
        raise ValueError("use distinct tasks and the unchanged q75 timing threshold")
    return {**prepared, "protocol_version": PROTOCOL_VERSION,
            "tasks": list(args.tasks),
            "primary_candidate": "stage9_recipe_causal_validity_interactions",
            "renewal_target": "keep_ise_minus_renew_ise_over_50_steps_no_later_upper_calls",
            "renewal_fit": "existing_fit_budgeted_trigger_predictors_grouped_ridge_oof_q75",
            "query_selection": "existing_six_event_classes_same_counts_and_blocked_jitter_calls",
            "controller_fit": "same_goal_conditioned_ppo_and_actual_duration_credit_per_task",
            "shared_comparison": "run_pointmaze_timing_baselines.fit_timing_comparison",
            "timing_rule": "first_crossing_offsets_0_5_10_15_20_deadline_25_one_call_per_50",
            "cross_task_hyperparameters": "one_shared_configuration_no_per_task_overrides",
            "transfer_meaning": "same_algorithm_retrained_per_plant_not_zero_shot_weights",
            "task_adaptations": {
                "pointmaze": "unchanged_maze_dynamics_observations_and_box_goal_bounds",
                "reacher_tracking": "stock_Reacher_v5_plant_frame_skip_1_ellipse_target_annulus_goals",
            },
            "statistical_unit": "optimizer_seed_root_per_task_not_query_or_evaluation_path",
            "performance_claim_supported": False, "scheduler_submission": False}


def run_task(args, task_name):
    spec = PlanValidityTaskSpec(task_name, _task_options(args), args.env_id)
    time_scale = time_contract(args)
    payload, controller = train_shared_controller(spec, args, time_scale)
    fit_rows = collect_renewal_labels(controller, task_spec=spec, args=args, time_scale=time_scale)

    def rollout(model, rule, *, seed, args, time_scale, intervention_step=None, arm=None):
        return run_episode(model, task_spec=spec, seed=seed, args=args, time_scale=time_scale,
                           rule=rule, intervention_step=intervention_step, arm=arm)[1]

    def pairs(model, rules, **kwargs):
        return collect_timing_pairs(model, rules, rollout_fn=rollout, **kwargs)

    result = fit_timing_comparison(args, controller, payload, fit_rows, time_scale,
                                  rollout_fn=rollout, pair_collector=pairs)
    result.update(task=task_name, task_spec=payload["task_spec"])
    result["controller_selection_history"] = _compact_history([
        row for row in payload["history"]
        if row["iteration"] < 0 or row.get("checkpoint_evaluation_performed")])
    result["fit_branch_summary"] = {
        "query_count": len(fit_rows),
        "mean_waypoint_change": sum(row["renew_waypoint_delta"] for row in fit_rows) / len(fit_rows),
        "mean_renewal_advantage": sum(row["renew_ise_advantage"] for row in fit_rows) / len(fit_rows),
        "scope": "fitting_bank_diagnostic_not_control_validation",
    }
    return result


def main(argv=None):
    parser = build_timing_parser()
    parser.description = __doc__
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    args = parser.parse_args(argv)
    resolved = protocol(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cells = []
    if not args.dry_run:
        for name in args.tasks:
            result = run_task(args, name)
            cache_path = args.output.with_name(f"{args.output.stem}.{name}.fit-cache.json")
            cache_path.write_text(json.dumps(_json_ready(result.pop("fit_cache"))) + "\n", encoding="utf-8")
            result["fit_cache_file"] = str(cache_path)
            cells.append(result)
    result = {"status": "prepared" if args.dry_run else "development_complete",
              "protocol": resolved, "cells": cells}
    args.output.write_text(json.dumps(_json_ready(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
