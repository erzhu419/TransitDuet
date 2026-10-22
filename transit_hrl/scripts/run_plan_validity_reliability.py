"""Prepare, or explicitly evaluate, frozen local-to-episode reliability cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from freq_hrl.experiments.compute_budget import ComputeLedger, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_reliability import (
    FAMILY_ALPHA, PROFILES, PROTOCOL_VERSION, evaluate_profile, profile_options,
)
from freq_hrl.experiments.plan_validity_transfer import TASKS, PlanValidityTaskSpec, time_contract
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from freq_hrl.rl import GoalConditionedActorCriticPPO, GoalConditionedPPOConfig


def load_record(path, run_id):
    data = json.loads(path.read_text())
    rows = [row for row in data["records"] if row["run_id"] == run_id]
    if len(rows) != 1 or rows[0]["status"] != "validation_complete":
        raise ValueError("reliability needs one successfully frozen development run")
    return rows[0]


def load_controller(record):
    args = SimpleNamespace(**record["configuration"])
    scale = time_contract(args)
    dim = 6 + 6 * scale.history_steps
    model = GoalConditionedActorCriticPPO(GoalConditionedPPOConfig(
        upper_state_dim=dim, lower_state_dim=dim, goal_dim=2, action_dim=2,
        hidden_dim=args.reference_hidden_dim, learning_rate=args.learning_rate,
        epochs=4, minibatch_size=1024, init_log_std=-0.7))
    model.load_state_dict(torch.load(record["controller_checkpoint"], map_location="cpu", weights_only=True))
    return model, args


def prepare(args):
    if (len(args.evaluation_seeds) < 2 or len(args.evaluation_seeds) % 2
        or len(set(args.evaluation_seeds)) != len(args.evaluation_seeds)
        or not args.probe_seeds or not set(args.probe_seeds) <= set(args.evaluation_seeds)
        or len(set(args.probe_seeds)) != len(args.probe_seeds)
        or not np.isfinite(args.risk_distance) or args.risk_distance <= 0
        or not 0 <= args.risk_rate_margin < 1 or not np.isfinite(args.tail_ise_margin)
        or args.risk_duration_steps < 1 or args.probes_per_episode < 1
        or args.conditional_replicates < 2 or args.conditional_replicates % 2):
        raise ValueError("use distinct paired evaluation paths, positive risk thresholds and even future replicas")
    return {"protocol_version": PROTOCOL_VERSION, "task": args.task, "optimizer_seed": args.optimizer_seed,
        "candidate_report": str(args.candidate_report), "candidate_run_id": args.candidate_run_id,
        "comparator_report": str(args.comparator_report), "comparator_run_id": args.comparator_run_id,
        "registered_comparator_method": args.comparator_method,
        "evaluation_seeds": args.evaluation_seeds, "profiles": list(PROFILES),
        "probe_seeds": args.probe_seeds,
        "risk_spec": {"distance": args.risk_distance, "duration_steps": args.risk_duration_steps},
        "risk_rate_margin": args.risk_rate_margin, "tail_ise_margin": args.tail_ise_margin,
        "family_alpha": FAMILY_ALPHA,
        "probes_per_episode": args.probes_per_episode, "conditional_replicates": args.conditional_replicates,
        "controller_training": "none_load_frozen_artifacts_only", "scheduler_submission": False,
        "renewal_target": "unchanged_50_step_keep_minus_renew_without_later_upper_calls",
        "continuation": "now_vs_wait_one_check_then_frozen_rule_on_each_arms_own_state",
        "state_population": "paired_fixed_clock_jitter_reference_vs_deployed_with_fit_bank_scaling",
        "future_sampling": "conditional_hidden_driver_given_realized_past_and_current_latent_state",
        "inference_unit": "optimizer_root_not_overlapping_probes_or_future_replicas",
        "performance_claim_supported": False}


def run_case(args, resolved):
    candidate = load_record(args.candidate_report, args.candidate_run_id)
    comparator = load_record(args.comparator_report, args.comparator_run_id)
    if comparator["method"] != args.comparator_method:
        raise ValueError("source comparator differs from the registered comparison")
    resolved["comparator_eligible"] = (not comparator["method"].startswith("termination_") or
        comparator["termination_readiness"]["eligible_for_fresh_confirmation"])
    for row in (candidate, comparator):
        if row["task"] != args.task or row["optimizer_seed"] != args.optimizer_seed:
            raise ValueError("frozen source task/root differs from the declared case")
        exposed = set().union(*(row["configuration"][key] for key in
            ("train_seeds", "selection_seeds", "branch_fit_seeds", "timing_selection_seeds")))
        if exposed.intersection(args.evaluation_seeds):
            raise ValueError("evaluation paths overlap development paths")
    rule, comparator_rule = (TimingRule(**row["selected_rule"]) for row in (candidate, comparator))
    if rule.kind not in ("legacy", "ridge"):
        raise ValueError("local calibration requires the frozen renewal-value predictor, not a probability/logit head")
    controller, source = load_controller(candidate)
    baseline, baseline_source = load_controller(comparator)
    if (source.horizon < 300 or source.horizon != baseline_source.horizon or source.env_id != baseline_source.env_id
        or _task_options(source) != _task_options(baseline_source)
        or time_contract(source) != time_contract(baseline_source)):
        raise ValueError("candidate/comparator need the same physical task, horizon and clock contract")
    fit_rows = json.loads(Path(candidate["fit_cache_file"]).read_text())["branch_labels"]
    resolved["comparison"] = {"candidate_method": candidate["method"], "comparator_method": comparator["method"]}
    result = {}
    for name in PROFILES:
        spec = PlanValidityTaskSpec(args.task, profile_options(_task_options(source), name), source.env_id)
        try:
            with measured_phase(name):
                result[name] = evaluate_profile(controller, baseline, rule, comparator_rule,
                    task_spec=spec, args=source, seeds=args.evaluation_seeds, probe_seeds=args.probe_seeds, fit_rows=fit_rows,
                    probes_per_episode=args.probes_per_episode, risk_spec=resolved["risk_spec"],
                    conditional_replicates=args.conditional_replicates)
        except Exception as exc:
            result[name] = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
        args.output.write_text(json.dumps(_json_ready({"status": "running", "protocol": resolved,
            "profiles": result}), indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-run-id", required=True)
    parser.add_argument("--comparator-report", type=Path, required=True)
    parser.add_argument("--comparator-run-id", required=True)
    parser.add_argument("--comparator-method", required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--evaluation-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--probe-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--risk-distance", type=float, required=True)
    parser.add_argument("--risk-rate-margin", type=float, required=True)
    parser.add_argument("--tail-ise-margin", type=float, required=True)
    parser.add_argument("--risk-duration-steps", type=int, default=25)
    parser.add_argument("--probes-per-episode", type=int, default=3)
    parser.add_argument("--conditional-replicates", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    resolved = prepare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        old = json.loads(args.output.read_text())
        if old["status"] != "prepared" or old["protocol"] != resolved:
            raise FileExistsError("preserve this case; use a new output for an additional attempt")
    result = {"status": "prepared", "protocol": resolved, "profiles": {}}
    ledger = ComputeLedger()
    if args.execute:
        result["status"] = "running"
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        try:
            with single_thread_runtime() as profile, ledger.recording("reliability_evaluation"):
                result["hardware_profile"] = profile
                result["profiles"] = run_case(args, resolved)
            result["status"] = "complete" if all(row["status"] == "complete" for row in result["profiles"].values()) else "failed"
        except Exception as exc:
            result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        result["evaluation_cost"] = ledger.summary()
    args.output.write_text(json.dumps(_json_ready(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
