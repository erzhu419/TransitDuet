"""Retrospective error decomposition of the frozen Stage-20 ridge folds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pointmaze_averaged_label_diagnostic import load_cached_pairs
from .pointmaze_budgeted_trigger import build_parser
from .pointmaze_fresh_future_diagnostic import PROTOCOL_VERSION as FRESH_PROTOCOL
from .pointmaze_goal_validation import _json_ready
from .pointmaze_paired_value_qualification import endpoints
from .pointmaze_regularized_pair_diagnostic import PROTOCOL_VERSION as RIDGE_PROTOCOL, RIDGE_ALPHA


PROTOCOL_VERSION = "pointmaze_ridge_error_stage22_v1_diagnostic"


def pair_design(pairs, fold):
    rows = endpoints(pairs)
    names = fold["feature_names"]
    if any(row["feature_names"] != names for row in rows):
        raise ValueError("frozen ridge feature schema differs")
    x = np.asarray([r["features"] for r in rows])
    remaining = x[:, names.index("remaining_fraction"), None]
    if not np.array_equal(remaining[::2], remaining[1::2]):
        raise ValueError("paired endpoints have different remaining time")
    phi = remaining * (x - fold["feature_mean"]) / fold["feature_scale"]
    return phi[1::2] - phi[::2]


def frozen_operator(train, query, fold):
    groups = np.asarray([r["seed"] for r in train])
    if (set(groups).intersection(r["seed"] for r in query)
            or sorted(set(groups)) != fold["training_paths"]
            or {r["seed"] for r in query} != {fold["held_out_path"]}
            or fold["alpha"] != RIDGE_ALPHA or fold["label"] != "averaged"):
        raise ValueError("frozen ridge fold or path isolation differs")
    d, q = pair_design(train, fold), pair_design(query, fold)
    weights = np.asarray([1. / np.sum(groups == s) for s in groups])
    weights /= weights.sum()
    operator = np.linalg.solve(d.T @ (weights[:, None] * d) + fold["alpha"] * np.eye(d.shape[1]),
                               d.T * weights)
    labels = np.asarray([np.mean(r["replicate_tail_advantages"]) for r in train])
    coefficients = np.asarray(fold["coefficients"])
    np.testing.assert_allclose(operator @ labels, coefficients, rtol=1e-10, atol=1e-12)
    return d @ coefficients, q @ coefficients, q @ operator


def decompose(prediction, influence, train_mean, train_mean_variance, query_mean, query_mean_variance):
    signal = influence @ train_mean
    propagated = influence ** 2 @ train_mean_variance
    label_error, mismatch = prediction - signal, signal - query_mean
    # Fresh train/query means are independent. Corrections cancel exactly in the sum.
    return {
        "training_label_realization_mse_estimate": label_error ** 2 - propagated,
        "signal_mismatch_mse_estimate": mismatch ** 2 - propagated - query_mean_variance,
        "interaction_estimate": 2 * label_error * mismatch + 2 * propagated,
        "total_mse_estimate": (prediction - query_mean) ** 2 - query_mean_variance,
        "zero_mse_estimate": query_mean ** 2 - query_mean_variance,
        "signal_prediction_estimate": signal,
        "propagated_fresh_mean_variance": propagated,
    }


def diagnose(pairs, ridge, fresh, *, fresh_replicates):
    key = lambda r: (r["seed"], r["check_step"])
    old = {key(r): r for r in ridge["rows"]}
    new = {key(r): r for r in fresh["rows"]}
    keys = {key(r) for r in pairs}
    if (keys != set(old) or keys != set(new) or len(keys) != len(pairs)
            or len(old) != len(ridge["rows"]) or len(new) != len(fresh["rows"])):
        raise ValueError("error decomposition needs the exact frozen opportunities")
    paths = sorted({r["seed"] for r in pairs})
    folds = [f for f in ridge["folds"] if f["label"] == "averaged"]
    if (sorted(f["held_out_path"] for f in folds) != paths
            or len({sum(r["seed"] == s for r in pairs) for s in paths}) != 1):
        raise ValueError("error decomposition needs balanced whole-path folds")
    moments = {}
    for pair in pairs:
        k = key(pair)
        a = np.asarray(new[k]["replicate_tail_advantages"])
        if a.shape != (fresh_replicates,) or not np.isfinite(a).all():
            raise ValueError("fresh label budget differs")
        if new[k]["predictions"]["ridge_averaged"] != old[k]["ridge_averaged_prediction"]:
            raise ValueError("fresh result changed the frozen prediction")
        moments[k] = (a.mean(), a.var(ddof=1) / len(a))

    rows, diagnostics = [], []
    for fold in folds:
        train = [r for r in pairs if r["seed"] != fold["held_out_path"]]
        query = [r for r in pairs if r["seed"] == fold["held_out_path"]]
        p_train, p_query, influence = frozen_operator(train, query, fold)
        np.testing.assert_allclose(p_query, [old[key(r)]["ridge_averaged_prediction"] for r in query],
                                   rtol=1e-10, atol=1e-12)
        mu_train, v_train = np.asarray([moments[key(r)] for r in train]).T
        mu_query, v_query = np.asarray([moments[key(r)] for r in query]).T
        old_samples = np.asarray([r["replicate_tail_advantages"] for r in train])
        old_mse = float(np.mean((p_train - old_samples.mean(axis=1)) ** 2))
        np.testing.assert_allclose(old_mse, fold["training_mse"], rtol=1e-10, atol=1e-12)
        diagnostics.append({"held_out_path": fold["held_out_path"],
                            "old_label_training_mse": old_mse,
                            "fresh_training_mse_estimate": float(np.mean((p_train - mu_train) ** 2 - v_train)),
                            "fresh_training_zero_mse_estimate": float(np.mean(mu_train ** 2 - v_train))})
        terms = decompose(p_query, influence, mu_train, v_train, mu_query, v_query)
        terms["propagated_old_label_variance_estimate"] = influence ** 2 @ (old_samples.var(axis=1, ddof=1) / old_samples.shape[1])
        for i, pair in enumerate(query):
            rows.append({"seed": pair["seed"], "check_step": pair["check_step"],
                         "frozen_prediction": float(p_query[i]), "fresh_mean": float(mu_query[i]),
                         **{k: float(v[i]) for k, v in terms.items()}})
    metrics = {k: float(np.mean([r[k] for r in rows])) for k in terms
               if k not in ("signal_prediction_estimate", "propagated_fresh_mean_variance")}
    metrics.update({k: float(np.mean([f[k] for f in diagnostics])) for k in diagnostics[0]
                    if k != "held_out_path"})
    np.testing.assert_allclose(metrics["total_mse_estimate"],
                               sum(metrics[k] for k in ("training_label_realization_mse_estimate",
                                   "signal_mismatch_mse_estimate", "interaction_estimate")), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(metrics["total_mse_estimate"],
                               fresh["metrics"]["conditional_mean_mse_estimate"]["ridge_averaged"],
                               rtol=1e-10, atol=1e-12)
    return {"optimizer_seed": ridge["optimizer_seed"], "branch_fit_seeds": ridge["branch_fit_seeds"],
            "trigger_eval_seeds": ridge["trigger_eval_seeds"], "rows": rows, "folds": diagnostics,
            "metrics": metrics, "influence_linear_systems": len(folds),
            "critic_parameter_updates": 0, "controller_training_iterations": 0,
            "additional_primitive_steps": 0, "evaluation_paths_used": 0}


def run_cell(args):
    _, pairs = load_cached_pairs(args)
    cached = []
    for path, protocol in ((args.prediction_result, RIDGE_PROTOCOL), (args.fresh_result, FRESH_PROTOCOL)):
        data = json.loads(path.read_text())
        if (data["status"] != "complete" or data["protocol"]["protocol_version"] != protocol
                or data["protocol"]["optimizer_seed"] != args.optimizer_seed):
            raise ValueError("error decomposition source is not the completed registered root")
        cell = data["cells"][0]
        if cell["branch_fit_seeds"] != args.branch_fit_seeds or cell["trigger_eval_seeds"] != args.trigger_eval_seeds:
            raise ValueError("error decomposition source seed roles differ")
        cached.append(cell)
    return diagnose(pairs, *cached, fresh_replicates=args.fresh_future_replicates)


def main(argv=None):
    parser = build_parser()
    for key in ("source", "endpoint", "prediction", "fresh"):
        parser.add_argument(f"--{key}-result", type=Path, required=True)
    parser.add_argument("--pairs-per-seed", type=int, default=12)
    parser.add_argument("--noise-pairs-per-seed", type=int, default=2)
    parser.add_argument("--future-replicates", type=int, default=8)
    parser.add_argument("--fresh-future-replicates", type=int, default=64)
    args = parser.parse_args(argv)
    output = {"status": "dry_run" if args.dry_run else "complete", "protocol": {
        "protocol_version": PROTOCOL_VERSION, "optimizer_seed": args.optimizer_seed,
        "evidence_role": "retrospective_frozen_ridge_error_decomposition_only",
        "fresh_labels_seen_in_stage21": True, "policy_deployment": False,
        "additional_primitive_steps": 0, "critic_parameter_updates": 0,
        "future_replicates": args.future_replicates, "fresh_future_replicates": args.fresh_future_replicates,
        **{f"{k}_result": str(getattr(args, f"{k}_result")) for k in ("source", "endpoint", "prediction", "fresh")}},
        "cells": [] if args.dry_run else [run_cell(args)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
