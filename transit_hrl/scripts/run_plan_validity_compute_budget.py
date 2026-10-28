"""Prepare or measure one cold-build timing method; never submit or access test paths."""

from __future__ import annotations

import json
import numpy as np
import torch

from freq_hrl.experiments.compute_budget import (
    ComputeLedger, ControlLatency, count_work, measured_phase, single_thread_runtime,
)
from freq_hrl.experiments.plan_validity_transfer import (
    TASKS, PlanValidityTaskSpec, collect_renewal_features, collect_renewal_labels,
    run_episode, time_contract, train_shared_controller,
)
from freq_hrl.experiments.pointmaze_budgeted_trigger import fit_budgeted_trigger_predictors
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_plan_value_qualification import _compact_history
from freq_hrl.experiments.pointmaze_timing_baselines import (
    TimingRule, collect_timing_pairs, fit_capacity_controls, fit_termination,
    select_rule, termination_readiness, threshold_candidates,
)
from scripts.run_pointmaze_timing_baselines import build_timing_parser
from scripts.run_plan_validity_second_task import protocol as transfer_protocol


METHODS = ("fixed", "jitter", "best_clock", "threshold", "history", "current",
           "matched_history", "matched_current", "termination_history", "termination_current")
PROTOCOL_VERSION = "plan_validity_total_compute_budget_v1_prepared"


def prepare(args):
    return {**transfer_protocol(args), "protocol_version": PROTOCOL_VERSION, "method": args.method,
            "expected_runs": [{"run_id": f"{args.output.stem}:{task}:{args.method}:{args.optimizer_seed}",
                               "task": task, "method": args.method, "optimizer_seed": args.optimizer_seed}
                              for task in args.tasks],
            "measurement": "one_complete_method_pipeline_on_a_warm_single_thread_cpu_worker",
            "controller": "fresh_jitter_controller_per_method_trial_with_full_training_cost",
            "shared_comparison": "same_fitting_helpers_single_method_measurement_without_test_access",
            "cost_scope": "controller_setup_train_selection_diagnostics_data_fit_search_validation_artifacts",
            "matching": "all_trial_search_cpu_plus_N_validation_measured_deployment_cpu_under_equal_caps",
            "training_iteration_ladder": [192, 384, 768], "labels_per_class_ladder": [1, 2, 4],
            "termination_pairs_per_path_ladder": [4, 8, 12],
            "deployment_episode_ladder": [1, 100, 1000],
            "latency": "all_control_steps_p50_p95_p99_max_deadline_misses_excluding_simulator",
            "test_access": "none_cost_and_configuration_selection_are_development_only",
            "cache_charge": "no_free_pretraining_or_branch_supervision_no_incomplete_legacy_costs",
            "budget_overrun_rule": "retain_cost_and_failure_do_not_pad_or_truncate_to_call_it_matched"}


def run_trial(args, task_name):
    ledger, latency = ComputeLedger(), ControlLatency()
    spec = PlanValidityTaskSpec(task_name, _task_options(args), args.env_id)
    time_scale = time_contract(args)
    result = {"run_id": f"{args.output.stem}:{task_name}:{args.method}:{args.optimizer_seed}",
              "task": task_name, "method": args.method, "optimizer_seed": args.optimizer_seed,
              "status": "failed", "accounting_complete": True,
              "configuration": {k: v for k, v in vars(args).items() if k not in ("output", "dry_run")}}
    cache = {}

    def rollout(model, rule, *, seed, args, time_scale, intervention_step=None, arm=None):
        return run_episode(model, task_spec=spec, seed=seed, args=args, time_scale=time_scale,
                           rule=rule, intervention_step=intervention_step, arm=arm)[1]

    try:
        with single_thread_runtime() as profile:
            result["hardware_profile"] = profile
            with ledger.recording():
                with measured_phase("controller"):
                    payload, controller = train_shared_controller(spec, args, time_scale)

                selection_cache = {}

                def selection(rule):
                    if rule.name not in selection_cache:
                        with measured_phase("timing_selection"):
                            selection_cache[rule.name] = [
                                rollout(controller, rule, seed=seed, args=args, time_scale=time_scale)
                                for seed in args.timing_selection_seeds]
                    return selection_cache[rule.name]

                method = args.method
                if method in ("fixed", "jitter"):
                    selected = TimingRule(method, "jitter" if method == "jitter" else "clock")
                elif method == "best_clock":
                    with measured_phase("clock_search"):
                        selected, _ = select_rule([TimingRule(f"clock_{offset}", "clock", threshold=offset / 25)
                                                  for offset in range(0, 26, 5)], selection)
                elif method == "threshold":
                    with measured_phase("feature_acquisition"):
                        features = collect_renewal_features(controller, task_spec=spec, args=args, time_scale=time_scale)
                    cache["feature_rows"] = features
                    with measured_phase("threshold_search"):
                        selected, _ = select_rule([rule for rule in threshold_candidates(features)
                                                  if rule.kind == "threshold"], selection)
                elif method in ("history", "current", "matched_history", "matched_current"):
                    with measured_phase("branch_supervision"):
                        labels = collect_renewal_labels(controller, task_spec=spec, args=args, time_scale=time_scale)
                    cache["branch_labels"] = labels
                    with measured_phase("ridge_fit_and_oof"):
                        view = method.removeprefix("matched_")
                        if method.startswith("matched_"):
                            selected = fit_capacity_controls(labels, effective_df=args.effective_df,
                                threshold_quantile=args.threshold_quantile, views=(view,))[0]
                        else:
                            predictor = "causal_validity_interactions" if view == "history" else "current_compact_quadratic"
                            fitted = fit_budgeted_trigger_predictors(labels, alpha_grid=args.ridge_alpha_grid,
                                threshold_quantile=args.threshold_quantile, predictors=(predictor,))[predictor]
                            selected = TimingRule(predictor, "legacy", view=predictor,
                                                   threshold=fitted["threshold"], model=fitted)
                else:
                    # The matched-head protocol needs BOTH behaviors to construct its shared bank.
                    behavior = [TimingRule("initial_clock_10", "clock", threshold=0.4)]
                    pool, checkpoints = [], {"current": [], "history": []}
                    for index in range(args.timing_rounds):
                        with measured_phase("termination_pair_supervision"):
                            pairs, _ = collect_timing_pairs(controller, behavior, seeds=args.branch_fit_seeds,
                                args=args, time_scale=time_scale, round_index=index, rollout_fn=rollout)
                        count_work("termination_label_pairs", len(pairs))
                        pool.extend(pairs)
                        behavior = []
                        for view in checkpoints:
                            with measured_phase(f"termination_fit_{view}"):
                                rule = fit_termination(pool, view=view, seed=args.optimizer_seed + index,
                                    epochs=args.termination_epochs, hidden_dim=args.termination_hidden_dim)
                            rule.name += f"_round_{index + 1}"
                            checkpoints[view].append(rule)
                            behavior.append(rule)
                    cache["termination_pairs"] = pool
                    view = method.removeprefix("termination_")
                    with measured_phase("termination_checkpoint_search"):
                        selected, _ = select_rule(checkpoints[view], selection)
                        best_clock, _ = select_rule([TimingRule(f"clock_{offset}", "clock", threshold=offset / 25)
                                                    for offset in range(0, 26, 5)], selection)
                        result["termination_readiness"] = termination_readiness(selection(selected), selection(best_clock))

                # Measure only on selection paths; freeze before any future test access.
                with measured_phase("validation_latency"):
                    rows = [run_episode(controller, task_spec=spec, seed=seed, args=args,
                            time_scale=time_scale, rule=selected, latency_recorder=latency)[1]
                            for seed in args.timing_selection_seeds]
                result.update(status="validation_complete", selected_rule=selected.to_dict(),
                    validation_ise=float(np.mean([r["tracking_squared_error_integral"] for r in rows])),
                    validation_return=float(np.mean([r["episode_return"] for r in rows])),
                    validation_latency=latency.summary(episodes=len(rows), deadline_seconds=time_scale.dt_seconds),
                    controller_selected_iteration=payload["selected_checkpoint_iteration"])
                result["controller_selection_history"] = _compact_history([
                    row for row in payload["history"]
                    if row["iteration"] < 0 or row.get("checkpoint_evaluation_performed")])
                with measured_phase("artifact_export"):
                    checkpoint = args.output.with_name(f"{args.output.stem}.{task_name}.controller.pt")
                    torch.save(controller.state_dict(), checkpoint)
                    result["controller_checkpoint"] = str(checkpoint)
                    if cache:
                        cache_path = args.output.with_name(f"{args.output.stem}.{task_name}.fit-cache.json")
                        cache_path.write_text(json.dumps(_json_ready(cache)) + "\n", encoding="utf-8")
                        result["fit_cache_file"] = str(cache_path)
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        if "hardware_profile" not in result:
            result["accounting_complete"] = False
    result["offline_cost"] = ledger.summary()
    result["performance_claim_supported"] = False
    return result


def build_compute_parser():
    parser = build_timing_parser()
    parser.description = __doc__
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--method", choices=METHODS, required=True)
    return parser


def main(argv=None):
    args = build_compute_parser().parse_args(argv)
    resolved = prepare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous["status"] != "prepared" or previous["protocol"] != _json_ready(resolved):
            raise FileExistsError("preserve this measured/interrupted attempt; use a new --output and keep both plans")
    records = []
    if not args.dry_run:
        args.output.write_text(json.dumps({"status": "running", "protocol": _json_ready(resolved),
                                           "records": []}, indent=2) + "\n", encoding="utf-8")
        for name in args.tasks:
            records.append(run_trial(args, name))
            args.output.write_text(json.dumps(_json_ready({"status": "running", "protocol": resolved,
                "records": records}), indent=2) + "\n", encoding="utf-8")
    failed = any(row["status"] == "failed" for row in records)
    args.output.write_text(json.dumps(_json_ready({"status": "prepared" if args.dry_run else
        "failed" if failed else "validation_complete", "protocol": resolved, "records": records}),
        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
