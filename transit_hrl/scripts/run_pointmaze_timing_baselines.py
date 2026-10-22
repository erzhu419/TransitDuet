"""Prepare or explicitly run a development-only strong timing-baseline suite."""

from __future__ import annotations

import json

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.experiments.pointmaze_budgeted_trigger import (
    balanced_jitter_schedule, build_parser, fit_budgeted_trigger_predictors,
)
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_plan_validity_branching import (
    _task_options, evaluate_plan_renewal_dataset,
)
from freq_hrl.experiments.pointmaze_plan_value_qualification import train_pointmaze_plan_value_cell
from freq_hrl.experiments.pointmaze_timing_baselines import (
    PROTOCOL_VERSION, TimingRule, collect_timing_pairs, fit_capacity_controls,
    fit_termination, rollout_rule, select_rule, termination_readiness, threshold_candidates,
)


def seed_roles(args):
    roles = {name: list(getattr(args, name)) for name in (
        "train_seeds", "selection_seeds", "branch_fit_seeds",
        "timing_selection_seeds", "trigger_eval_seeds",
    )}
    flattened = [seed for seeds in roles.values() for seed in seeds]
    if any(not seeds for seeds in roles.values()) or len(flattened) != len(set(flattened)):
        raise ValueError("controller/fit/timing-selection/test paths must be distinct")
    if len(roles["branch_fit_seeds"]) < 3:
        raise ValueError("at least three branch fitting paths are required")
    return roles


def protocol(args):
    roles = seed_roles(args)
    if (args.upper_period_seconds != 0.5 or args.horizon < 200 or args.horizon % 50
        or args.max_offset_steps != 25 or args.check_stride_steps != 5
        or args.branch_window_seconds != 0.5 or args.pairs_per_path < 1
        or args.pairs_per_path > args.horizon // 50 - 2
        or args.timing_rounds < 1 or args.termination_epochs < 1
        or args.termination_hidden_dim < 1 or args.effective_df <= 0):
        raise ValueError("this protocol uses the frozen 50/25/5 timing contract")
    return {
        "protocol_version": PROTOCOL_VERSION, "seed_roles": roles,
        "configuration": {key: value for key, value in vars(args).items()
                          if key not in {"output", "dry_run"}},
        "evidence_role": "development_only_not_independent_confirmation",
        "controller": "shared_frozen_balanced_jitter_trained_upper_and_lower",
        "budget": "one_call_per_50_step_bin_first_call_at_zero",
        "thresholds": "fit_quantiles_then_timing_selection_episode_ise",
        "capacity": "17_inputs_quadratic_170_slopes_plus_intercept_equal_effective_df",
        "termination": "19_64_64_1_by_default_cost_sensitive_paired_policy_iteration",
        "termination_credit": "wait_one_check_ise_minus_stop_now_ise_adaptive_suffix",
        "termination_data": "identical_pooled_behavior_mixture_for_both_views",
        "termination_execution": "deterministic_same_as_pair_continuation_no_forced_early_calls",
        "selection": "lowest_selection_ise_no_filtering_collapsed_checkpoints_or_roots",
        "test_access": "none_this_entry_point_is_development_only",
        "test_reuse": "reserve_new_roots_and_paths_before_any_performance_run",
        "scheduler_submission": False,
        "performance_claim_supported": False,
    }


def run_cell(args):
    seed_roles(args)
    time_scale = PhysicalTimeScaleContract(
        dt_seconds=0.01, upper_period_seconds=args.upper_period_seconds,
        history_seconds=args.history_seconds, fast_period_seconds=args.fast_period_seconds,
    )
    schedule = lambda seed: balanced_jitter_schedule(
        seed=seed, horizon=args.horizon, period_steps=50, max_offset_steps=25)
    payload, controller = train_pointmaze_plan_value_cell(
        method="hrl_regime_history", env_id=args.env_id,
        train_seeds=args.train_seeds, selection_seeds=args.selection_seeds,
        # Do not expose test paths through inherited controller diagnostics.
        eval_seeds=args.timing_selection_seeds, iterations=args.iterations,
        horizon=args.horizon, optimizer_seed=args.optimizer_seed,
        upper_period_seconds=args.upper_period_seconds, history_seconds=args.history_seconds,
        fast_period_seconds=args.fast_period_seconds, maximum_subgoal_delta=args.maximum_subgoal_delta,
        reference_hidden_dim=args.reference_hidden_dim, learning_rate=args.learning_rate,
        checkpoint_evaluation_interval=args.checkpoint_evaluation_interval,
        waypoint_perturbation=0.25, event_window_seconds=1.0, task_options=_task_options(args),
        diagnostic_schedules=("fixed",), training_decision_steps_fn=schedule,
        training_schedule_name="balanced_jitter",
    )
    fit_rows = evaluate_plan_renewal_dataset(
        controller, seeds=args.branch_fit_seeds, split="timing_fit",
        optimizer_seed=args.optimizer_seed, branch_window_seconds=0.5,
        max_events_per_class=args.max_events_per_class, env_id=args.env_id,
        horizon=args.horizon, time_scale=time_scale,
        maximum_subgoal_delta=args.maximum_subgoal_delta, task_options=_task_options(args),
        prefix_decision_steps_fn=schedule,
    )
    return fit_timing_comparison(args, controller, payload, fit_rows, time_scale)


def fit_timing_comparison(args, controller, payload, fit_rows, time_scale,
                          *, rollout_fn=None, pair_collector=None):
    """One development fitting/selection procedure for both physical tasks."""
    rollout_fn = rollout_rule if rollout_fn is None else rollout_fn
    pair_collector = collect_timing_pairs if pair_collector is None else pair_collector
    ledger = {"branch_label_replay_steps": sum(
        r["keep_primitive_steps_replayed"] + r["renew_primitive_steps_replayed"] for r in fit_rows),
        "timing_pair_replay_steps": 0, "timing_selection_steps": 0, "test_steps": 0}

    def evaluate(rule, seeds, ledger_key):
        rows = [rollout_fn(controller, rule, seed=seed, args=args, time_scale=time_scale)
                for seed in seeds]
        ledger[ledger_key] += len(rows) * args.horizon
        return rows

    selection_cache = {}

    def selection(rule):
        # Names identify distinct checkpoints in this run; no state files are read.
        if rule.name not in selection_cache:
            selection_cache[rule.name] = evaluate(rule, args.timing_selection_seeds,
                                                  "timing_selection_steps")
        return selection_cache[rule.name]

    candidates = threshold_candidates(fit_rows)
    clock, clock_scores = select_rule([r for r in candidates if r.kind == "clock"], selection)
    threshold, threshold_scores = select_rule([r for r in candidates if r.kind == "threshold"], selection)
    legacy = fit_budgeted_trigger_predictors(fit_rows, alpha_grid=args.ridge_alpha_grid,
                                             threshold_quantile=args.threshold_quantile)
    rules = [TimingRule("fixed", "clock", threshold=0), TimingRule("balanced_jitter", "jitter"), clock, threshold]
    rules.extend(TimingRule(f"stage9_recipe_{name}", "legacy", view=name,
                            threshold=fitted["threshold"], model=fitted)
                 for name, fitted in legacy.items())
    rules.extend(fit_capacity_controls(fit_rows, effective_df=args.effective_df,
                                      threshold_quantile=args.threshold_quantile))
    behavior = [TimingRule("initial_clock_10", "clock", threshold=0.4)]
    pooled_pairs, checkpoints = [], {"current": [], "history": []}
    for index in range(args.timing_rounds):
        pairs, steps = pair_collector(controller, behavior, seeds=args.branch_fit_seeds,
                                            args=args, time_scale=time_scale, round_index=index)
        pooled_pairs.extend(pairs)
        ledger["timing_pair_replay_steps"] += steps
        behavior = []
        for view in checkpoints:
            rule = fit_termination(pooled_pairs, view=view,
                                   seed=args.optimizer_seed + index,
                                   epochs=args.termination_epochs,
                                   hidden_dim=args.termination_hidden_dim)
            rule.name += f"_round_{index + 1}"
            checkpoints[view].append(rule)
            behavior.append(rule)
    readiness, checkpoint_scores = {}, {}
    for view, available in checkpoints.items():
        selected, scores = select_rule(available, selection)
        rules.append(selected)
        checkpoint_scores[view] = scores
        readiness[view] = termination_readiness(selection(selected), selection(clock))
    frozen_rules = [rule.to_dict() for rule in rules]
    return {
        "optimizer_seed": args.optimizer_seed, "frozen_rules": frozen_rules,
        "controller_selected_iteration": payload["selected_checkpoint_iteration"],
        "controller_capacity": payload["capacity"],
        "runtime_versions": payload["runtime_versions"],
        "time_scale": time_scale.metadata(),
        "controller_training_rollout_steps": args.iterations * len(args.train_seeds) * args.horizon,
        "controller_payload_selection_cost_requires_reconciliation": True,
        "ledger": ledger, "branch_fit_count": len(fit_rows),
        "paired_timing_fit_count": len(pooled_pairs),
        "clock_selection": clock_scores, "threshold_selection": threshold_scores,
        "termination_checkpoint_selection": checkpoint_scores,
        "termination_readiness": readiness, "test_accessed": False,
        "fit_cache": {"branch_fit_rows": fit_rows, "paired_timing_rows": pooled_pairs},
        "performance_claim_supported": False,
    }


def build_timing_parser():
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--timing-selection-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--timing-rounds", type=int, default=2)
    parser.add_argument("--pairs-per-path", type=int, default=12)
    parser.add_argument("--termination-epochs", type=int, default=200)
    parser.add_argument("--termination-hidden-dim", type=int, default=64)
    parser.add_argument("--effective-df", type=float, default=8.0)
    # This entry point prepares a protocol unless execution is explicitly requested.
    parser.set_defaults(dry_run=True)
    parser.add_argument("--execute", action="store_false", dest="dry_run")
    return parser


def main(argv=None):
    args = build_timing_parser().parse_args(argv)
    resolved = protocol(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cells = [] if args.dry_run else [run_cell(args)]
    if cells:
        cache_path = args.output.with_suffix(".fit-cache.json")
        cache_path.write_text(json.dumps(_json_ready(cells[0].pop("fit_cache"))) + "\n",
                              encoding="utf-8")
        cells[0]["fit_cache_file"] = str(cache_path)
    result = {"status": "prepared" if args.dry_run else "development_complete",
              "protocol": resolved, "cells": cells}
    args.output.write_text(json.dumps(_json_ready(result), indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
