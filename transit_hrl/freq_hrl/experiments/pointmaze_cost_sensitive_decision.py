"""Path-held-out causal now/wait decisions with cost-weighted supervision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .pointmaze_averaged_label_diagnostic import load_cached_pairs
from .pointmaze_budgeted_trigger import build_parser
from .pointmaze_fresh_future_diagnostic import PROTOCOL_VERSION as FRESH_PROTOCOL
from .pointmaze_goal_validation import _json_ready
from .pointmaze_onecheck_advantage import advantage_matrix


PROTOCOL_VERSION = "pointmaze_cost_sensitive_stage25_v1_development"
ALPHA = 1.0
MAX_ITERATIONS = 256
CANDIDATE = "cost_sensitive"
LEARNED = (CANDIDATE, "uniform", "mse", "short_window")
CONTROLS = ("short_window", "uniform", "mse", "always_wait", "always_now")


def logistic_objective(theta, design, signs, weights):
    margin = signs * (design @ theta)
    loss = weights @ np.logaddexp(0., -margin) + .5 * ALPHA * (theta @ theta)
    gradient = -design.T @ (weights * signs * expit(-margin)) + ALPHA * theta
    return float(loss), gradient


def fit_model(design, targets, objective):
    targets = np.asarray(targets, dtype=float)
    if not np.isfinite(targets).all():
        raise ValueError("decision training targets must be finite")
    scale = float(np.mean(np.abs(targets))) or 1.0
    if objective == "mse":
        theta = design.T @ np.linalg.solve(
            design @ design.T + len(design) * ALPHA * np.eye(len(design)), targets / scale)
        loss = .5 * np.mean((design @ theta - targets / scale) ** 2) + .5 * ALPHA * (theta @ theta)
        iterations = 0
    else:
        if objective not in ("cost_sensitive", "uniform"):
            raise ValueError("unknown decision objective")
        weights = np.abs(targets) if objective == "cost_sensitive" else (targets != 0).astype(float)
        weights = weights / (float(weights.sum()) or 1.0)
        result = minimize(logistic_objective, np.zeros(design.shape[1]),
                          args=(design, np.sign(targets), weights), method="L-BFGS-B", jac=True,
                          options={"maxiter": MAX_ITERATIONS, "gtol": 1e-9, "ftol": 1e-12})
        if not result.success:
            raise RuntimeError(f"decision fit did not converge: {result.message}")
        theta, loss, iterations = result.x, result.fun, result.nit
    return {"weights": theta.tolist(), "objective": objective, "objective_value": float(loss),
            "iterations": int(iterations), "training_target_mean_abs": scale,
            "training_rows": len(design)}


def fit_fold(train):
    matrix, names = advantage_matrix(train)
    mean, scale = matrix.mean(axis=0), matrix.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    design = np.column_stack((np.ones(len(train)), (matrix - mean) / scale))
    window = np.asarray([r["window_ise_advantage"] for r in train])
    total = window + np.asarray([np.mean(r["replicate_tail_advantages"]) for r in train])
    models = {method: fit_model(design, window if method == "short_window" else total,
                               "cost_sensitive" if method == "short_window" else method)
              for method in LEARNED}
    return {"feature_names": names, "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
            "training_paths": sorted({r["seed"] for r in train}), "models": models}


def predict(fold, query):
    matrix, names = advantage_matrix(query)
    if names != fold["feature_names"]:
        raise ValueError("causal decision feature schema differs")
    design = np.column_stack((np.ones(len(query)), (matrix - fold["feature_mean"]) / fold["feature_scale"]))
    return {method: design @ model["weights"] for method, model in fold["models"].items()}


def summarize(rows):
    total = np.asarray([r["scoring_total_mean"] for r in rows])
    variance = np.asarray([r["scoring_mean_variance"] for r in rows])
    actions = {m: np.asarray([r["now_choices"][m] for r in rows], dtype=int)
               for m in (CANDIDATE, *CONTROLS)}
    comparisons = {}
    for control in CONTROLS:
        diff = actions[CANDIDATE] - actions[control]
        gain = diff * total
        comparisons[control] = {
            "mean_ise_benefit": float(gain.mean()),
            "conditional_mc_standard_error": float(np.sqrt(np.sum(diff ** 2 * variance)) / len(rows)),
            "different_decisions": int(np.count_nonzero(diff)),
            "positive_contributions": int(np.count_nonzero(gain > 0)),
            "negative_contributions": int(np.count_nonzero(gain < 0)),
        }
    return {"opportunities": len(rows), "now_counts": {m: int(a.sum()) for m, a in actions.items()},
            "mean_ise_benefit_vs_wait": {m: float(np.mean(a * total)) for m, a in actions.items()},
            "candidate_vs_control": comparisons,
            "development_gate_passed": all(c["mean_ise_benefit"] > 0 for c in comparisons.values())}


def qualify(pairs, fresh, *, root, fit_seeds, eval_seeds, fresh_replicates):
    key = lambda r: (r["seed"], r["check_step"])
    paths = sorted(set(fit_seeds))
    new = {key(r): r for r in fresh["rows"]}
    if (len(paths) < 2 or set(paths).intersection(eval_seeds)
            or {r["seed"] for r in pairs} != set(paths)
            or len({sum(r["seed"] == s for r in pairs) for s in paths}) != 1
            or len(pairs) != len(new) or len(new) != len(fresh["rows"])
            or {key(r) for r in pairs} != set(new)):
        raise ValueError("decision screen requires exact balanced whole-path folds")
    rows, folds = [], []
    for held in paths:
        train, query = [r for r in pairs if r["seed"] != held], [r for r in pairs if r["seed"] == held]
        fold = fit_fold(train)
        predictions = predict(fold, query)
        folds.append({"held_out_path": held, **fold})
        for i, pair in enumerate(query):
            samples = np.asarray(new[key(pair)]["replicate_tail_advantages"])
            if samples.shape != (fresh_replicates,) or not np.isfinite(samples).all():
                raise ValueError("decision scoring label budget differs")
            if pair["now_upper_call_count"] != pair["wait_upper_call_count"]:
                raise ValueError("decision paired upper call budget differs")
            scores = {m: float(p[i]) for m, p in predictions.items()}
            rows.append({"seed": held, "check_step": pair["check_step"], "scores": scores,
                         "scoring_total_mean": float(pair["window_ise_advantage"] + samples.mean()),
                         "scoring_mean_variance": float(samples.var(ddof=1) / len(samples)),
                         "now_choices": {**{m: v > 0 for m, v in scores.items()},
                                         "always_wait": False, "always_now": True}})
    return {"optimizer_seed": root, "branch_fit_seeds": fit_seeds, "trigger_eval_seeds": eval_seeds,
            "rows": rows, "folds": folds, "metrics": summarize(rows),
            "path_metrics": {str(s): summarize([r for r in rows if r["seed"] == s]) for s in paths},
            "decision_fits": len(folds) * len(LEARNED),
            "decision_optimizer_iterations": sum(m["iterations"] for f in folds for m in f["models"].values()),
            "additional_primitive_steps": 0, "controller_training_iterations": 0,
            "controller_policy_updates": 0, "evaluation_paths_used": 0}


def run_cell(args):
    _, pairs = load_cached_pairs(args)
    data = json.loads(args.fresh_result.read_text())
    if (data["status"] != "complete" or data["protocol"]["protocol_version"] != FRESH_PROTOCOL
            or data["protocol"]["optimizer_seed"] != args.optimizer_seed):
        raise ValueError("decision source is not the completed registered root")
    fresh = data["cells"][0]
    if fresh["branch_fit_seeds"] != args.branch_fit_seeds or fresh["trigger_eval_seeds"] != args.trigger_eval_seeds:
        raise ValueError("decision source seed roles differ")
    return qualify(pairs, fresh, root=args.optimizer_seed, fit_seeds=args.branch_fit_seeds,
                   eval_seeds=args.trigger_eval_seeds, fresh_replicates=args.fresh_future_replicates)


def main(argv=None):
    parser = build_parser()
    for key in ("source", "endpoint", "fresh"):
        parser.add_argument(f"--{key}-result", type=Path, required=True)
    for name, default in (("pairs-per-seed", 12), ("noise-pairs-per-seed", 2),
                          ("future-replicates", 8), ("fresh-future-replicates", 64)):
        parser.add_argument(f"--{name}", type=int, default=default)
    args = parser.parse_args(argv)
    output = {"status": "dry_run" if args.dry_run else "complete", "protocol": {
        "protocol_version": PROTOCOL_VERSION, "optimizer_seed": args.optimizer_seed,
        "candidate": CANDIDATE, "controls": CONTROLS, "alpha": ALPHA,
        "max_optimizer_iterations": MAX_ITERATIONS, "intercept_penalized": True,
        "features": "stage15_41_causal_features_training_fold_standardization",
        "zero_tie_action": "wait_one_check", "fold_unit": "whole_path",
        "evidence_role": "cached_causal_decision_development_only", "policy_deployment": False,
        "scoring_labels_previously_seen": True, "additional_primitive_steps": 0,
        "future_replicates": args.future_replicates, "fresh_future_replicates": args.fresh_future_replicates,
        **{f"{k}_result": str(getattr(args, f"{k}_result")) for k in ("source", "endpoint", "fresh")}},
        "cells": [] if args.dry_run else [run_cell(args)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
