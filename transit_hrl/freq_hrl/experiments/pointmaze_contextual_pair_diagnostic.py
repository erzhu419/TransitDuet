"""Bounded causal-context interactions for shared paired continuation values."""

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


PROTOCOL_VERSION = "pointmaze_contextual_pair_stage23_v1_diagnostic"
METHODS = ("contextual", "random_context")
CONTEXT_PREFIXES = ("target_current_", "force_current_", "distractor_current_",
                    "target_velocity_", "force_rms_", "distractor_delta_")


def feature_groups(names):
    context = [i for i, name in enumerate(names)
               if name.startswith(CONTEXT_PREFIXES) or name == "distractor_change_norm"]
    state = [i for i, name in enumerate(names) if i not in context
             and name not in ("remaining_fraction", "within_bin_fraction")]
    return state, context


def designs(pairs, fold, *, root):
    rows = endpoints(pairs)
    names = fold["feature_names"]
    if any(r["feature_names"] != names for r in rows):
        raise ValueError("contextual endpoint schema differs")
    state, context = feature_groups(names)
    if not state or not context:
        raise ValueError("contextual value needs both state and causal context")
    x = np.asarray([r["features"] for r in rows], dtype=np.float64)
    remaining = x[:, names.index("remaining_fraction"), None]
    z = (x - fold["feature_mean"]) / fold["feature_scale"]
    c = np.tanh(z[:, context]) / np.sqrt(len(context))
    if not np.array_equal(remaining[::2], remaining[1::2]) or not np.array_equal(c[::2], c[1::2]):
        raise ValueError("paired endpoints must share time and exogenous context")
    directions = np.asarray([np.random.default_rng(np.random.SeedSequence(
        [root, p["seed"], p["check_step"], 23_023])).choice([-1., 1.], size=len(context)) for p in pairs])
    random_c = np.repeat(directions, 2, axis=0) / np.sqrt(len(context)) * np.linalg.norm(c, axis=1)[:, None]
    base = remaining * z
    output = {"linear": base[1::2] - base[::2]}
    for method, value in (("contextual", c), ("random_context", random_c)):
        interaction = (z[:, state, None] * value[:, None, :]).reshape(len(rows), -1)
        phi = np.concatenate((base, remaining * interaction), axis=1)
        output[method] = phi[1::2] - phi[::2]
    np.testing.assert_allclose(np.linalg.norm(output["contextual"], axis=1),
                               np.linalg.norm(output["random_context"], axis=1), rtol=1e-12, atol=1e-12)
    return output


def fit_design(train, query, targets, weights):
    # The dual system is 14x14 for full folds, rather than a 365x365 primal solve.
    b = np.sqrt(weights)[:, None] * train
    gram = b @ b.T
    solution = np.linalg.solve(gram + RIDGE_ALPHA * np.eye(len(train)),
                               np.column_stack((np.sqrt(weights) * targets, gram)))
    coefficients = b.T @ solution[:, 0]
    return query @ coefficients, {
        "parameter_count": train.shape[1], "training_pairs": len(train), "query_pairs": len(query),
        "effective_degrees_of_freedom": float(np.trace(solution[:, 1:])),
        "coefficient_norm": float(np.linalg.norm(coefficients)),
        "training_mse": float(np.sum(weights * (train @ coefficients - targets) ** 2)),
    }


def score(rows):
    means = np.asarray([r["repeat_mean"] for r in rows])
    correction = float(np.mean([r["mean_variance"] for r in rows]))
    raw = {method: float(np.mean((np.asarray([r["predictions"][method] for r in rows]) - means) ** 2))
           for method in ("zero", "linear", *METHODS)}
    corrected = {k: v - correction for k, v in raw.items()}
    return {"mse_to_repeat_mean": raw, "finite_repeat_correction": correction,
            "conditional_mean_mse_estimate": corrected,
            "development_gate_passed": corrected["contextual"] < min(
                corrected[k] for k in ("zero", "linear", "random_context"))}


def qualify(pairs, ridge, fresh, *, root, fresh_replicates):
    key = lambda r: (r["seed"], r["check_step"])
    old = {key(r): r for r in ridge["rows"]}
    new = {key(r): r for r in fresh["rows"]}
    keys = {key(r) for r in pairs}
    paths = sorted({r["seed"] for r in pairs})
    saved_folds = [f for f in ridge["folds"] if f["label"] == "averaged"]
    if (keys != set(old) or keys != set(new) or len(keys) != len(pairs)
            or len(old) != len(ridge["rows"]) or len(new) != len(fresh["rows"])
            or paths != sorted(ridge["branch_fit_seeds"]) or set(paths).intersection(ridge["trigger_eval_seeds"])
            or sorted(f["held_out_path"] for f in saved_folds) != paths
            or len({sum(r["seed"] == s for r in pairs) for s in paths}) != 1):
        raise ValueError("contextual screen needs exact frozen whole-path folds")
    for pair in pairs:
        np.testing.assert_allclose(np.mean(pair["replicate_tail_advantages"]), old[key(pair)]["repeat_mean"],
                                   rtol=1e-12, atol=1e-15)
    rows, folds = [], []
    for fold in saved_folds:
        held = fold["held_out_path"]
        train, query = [r for r in pairs if r["seed"] != held], [r for r in pairs if r["seed"] == held]
        if fold["training_paths"] != [s for s in paths if s != held] or fold["alpha"] != RIDGE_ALPHA:
            raise ValueError("contextual training paths or alpha differ")
        d, q = designs(train, fold, root=root), designs(query, fold, root=root)
        targets = np.asarray([np.mean(r["replicate_tail_advantages"]) for r in train])
        weights = np.full(len(train), 1. / len(train))
        np.testing.assert_allclose(np.sum(weights * (d["linear"] @ fold["coefficients"] - targets) ** 2),
                                   fold["training_mse"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(q["linear"] @ fold["coefficients"],
                                   [old[key(r)]["ridge_averaged_prediction"] for r in query], rtol=1e-10, atol=1e-12)
        predictions = {}
        for method in METHODS:
            predictions[method], diagnostics = fit_design(d[method], q[method], targets, weights)
            folds.append({"held_out_path": held, "training_paths": fold["training_paths"],
                          "method": method, **diagnostics})
        for i, pair in enumerate(query):
            f = new[key(pair)]
            samples = np.asarray(f["replicate_tail_advantages"])
            if samples.shape != (fresh_replicates,) or not np.isfinite(samples).all():
                raise ValueError("contextual scoring label budget differs")
            linear = old[key(pair)]["ridge_averaged_prediction"]
            if linear != f["predictions"]["ridge_averaged"]:
                raise ValueError("contextual screen linear baseline changed")
            rows.append({"seed": held, "check_step": pair["check_step"],
                         "repeat_mean": float(samples.mean()), "mean_variance": float(samples.var(ddof=1) / len(samples)),
                         "predictions": {"zero": 0., "linear": linear,
                                         **{m: float(p[i]) for m, p in predictions.items()}}})
    metrics = score(rows)
    np.testing.assert_allclose(metrics["conditional_mean_mse_estimate"]["linear"],
                               fresh["metrics"]["conditional_mean_mse_estimate"]["ridge_averaged"], rtol=1e-10, atol=1e-12)
    return {"optimizer_seed": root, "branch_fit_seeds": ridge["branch_fit_seeds"],
            "trigger_eval_seeds": ridge["trigger_eval_seeds"], "rows": rows, "folds": folds, "metrics": metrics,
            "path_metrics": {str(s): score([r for r in rows if r["seed"] == s]) for s in paths},
            "ridge_fits": len(folds), "additional_primitive_steps": 0,
            "controller_training_iterations": 0, "critic_optimizer_steps": 0, "evaluation_paths_used": 0}


def run_cell(args):
    _, pairs = load_cached_pairs(args)
    cached = []
    for path, protocol in ((args.prediction_result, RIDGE_PROTOCOL), (args.fresh_result, FRESH_PROTOCOL)):
        data = json.loads(path.read_text())
        if (data["status"] != "complete" or data["protocol"]["protocol_version"] != protocol
                or data["protocol"]["optimizer_seed"] != args.optimizer_seed):
            raise ValueError("contextual source is not the completed registered root")
        cell = data["cells"][0]
        if cell["branch_fit_seeds"] != args.branch_fit_seeds or cell["trigger_eval_seeds"] != args.trigger_eval_seeds:
            raise ValueError("contextual source seed roles differ")
        cached.append(cell)
    return qualify(pairs, *cached, root=args.optimizer_seed, fresh_replicates=args.fresh_future_replicates)


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
        "alpha": RIDGE_ALPHA, "context_transform": "tanh_training_z_over_sqrt_context_count",
        "random_context_seed_namespace": 23_023, "candidate": "contextual",
        "controls": ["zero", "linear", "random_context"],
        "evidence_role": "cached_contextual_pair_development_only", "policy_deployment": False,
        "scoring_labels_previously_seen": True, "additional_primitive_steps": 0,
        "future_replicates": args.future_replicates, "fresh_future_replicates": args.fresh_future_replicates,
        **{f"{k}_result": str(getattr(args, f"{k}_result")) for k in ("source", "endpoint", "prediction", "fresh")}},
        "cells": [] if args.dry_run else [run_cell(args)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
