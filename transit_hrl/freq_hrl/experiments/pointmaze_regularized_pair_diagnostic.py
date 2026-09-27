"""Low-capacity paired continuation regression on the frozen future cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pointmaze_averaged_label_diagnostic import (
    PROTOCOL_VERSION as BASELINE_PROTOCOL, LABELS, label_targets, load_cached_pairs,
)
from .pointmaze_budgeted_trigger import build_parser
from .pointmaze_goal_validation import _json_ready
from .pointmaze_paired_value_qualification import endpoints


PROTOCOL_VERSION = "pointmaze_regularized_pair_stage20_v1_diagnostic"
RIDGE_ALPHA = 1.0
METHODS = ("ridge_single_draw", "ridge_averaged", "stage19_single_draw", "stage19_averaged")


def fit_pair_ridge(train_pairs, query_pairs, *, label):
    train, query = endpoints(train_pairs), endpoints(query_pairs)
    groups = np.asarray([r["seed"] for r in train_pairs])
    if set(groups).intersection(r["seed"] for r in query_pairs):
        raise ValueError("paired ridge training and query paths overlap")
    names = train[0]["feature_names"]
    if any(r["feature_names"] != names for r in [*train, *query]):
        raise ValueError("paired ridge endpoint schemas differ")
    remaining_index = names.index("remaining_fraction")
    for rows in (train, query):
        if any(a["features"][remaining_index] != b["features"][remaining_index]
               for a, b in zip(rows[::2], rows[1::2])):
            raise ValueError("paired ridge endpoints must have equal remaining time")
    weights = np.asarray([1.0 / np.sum(groups == s) for s in groups])
    weights /= weights.sum()
    endpoint_weights = np.repeat(weights / 2.0, 2)
    x = np.asarray([r["features"] for r in train], dtype=np.float64)
    mean = np.sum(endpoint_weights[:, None] * x, axis=0)
    scale = np.sqrt(np.sum(endpoint_weights[:, None] * (x - mean) ** 2, axis=0))
    scale[scale < 1e-8] = 1.0

    def design(rows):
        states = np.asarray([r["features"] for r in rows], dtype=np.float64)
        values = states[:, remaining_index, None] * (states - mean) / scale
        return values[1::2] - values[::2]

    # A shared linear value difference has no intercept and preserves antisymmetry.
    d, q = design(train), design(query)
    target = label_targets(train_pairs, label)
    gram = d.T @ (weights[:, None] * d)
    system = gram + RIDGE_ALPHA * np.eye(d.shape[1])
    coefficients = np.linalg.solve(system, d.T @ (weights * target))
    prediction = q @ coefficients
    residual = d @ coefficients - target
    return prediction, {
        "label": label, "alpha": RIDGE_ALPHA, "objective": "path_weighted_mean_mse_plus_l2",
        "training_paths": sorted(map(int, set(groups))), "training_pairs": len(train_pairs),
        "query_pairs": len(query_pairs), "parameter_count": len(coefficients),
        "feature_names": names, "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
        "coefficients": coefficients.tolist(), "coefficient_norm": float(np.linalg.norm(coefficients)),
        "effective_degrees_of_freedom": float(np.trace(np.linalg.solve(system, gram))),
        "training_mse": float(np.sum(weights * residual ** 2)),
    }


def contrast_metrics(rows):
    truth = np.asarray([r["repeat_mean"] for r in rows])
    correction = float(np.mean([r["repeat_variance"] / r["replicates"] for r in rows]))
    raw = {"zero": float(np.mean(truth ** 2)), **{
        key: float(np.mean((np.asarray([r[f"{key}_prediction"] for r in rows]) - truth) ** 2))
        for key in METHODS
    }}
    corrected = {key: value - correction for key, value in raw.items()}
    return {"mse_to_repeat_mean": raw, "finite_repeat_correction": correction,
            "conditional_mean_mse_estimate": corrected,
            "qualification_passed": corrected["ridge_averaged"] < min(
                corrected["zero"], corrected["stage19_averaged"])}


def qualify(pairs, baseline, *, root, fit_seeds, eval_seeds):
    paths = set(fit_seeds)
    keys = {(r["seed"], r["check_step"]) for r in pairs}
    if (len(paths) < 2 or paths.intersection(eval_seeds) or {r["seed"] for r in pairs} != paths
            or len(keys) != len(pairs)):
        raise ValueError("regularized-pair diagnostic has wrong seed roles or duplicate pairs")
    if len({sum(r["seed"] == s for r in pairs) for s in paths}) != 1:
        raise ValueError("each path must have equal opportunity counts")
    old = {(r["seed"], r["check_step"]): r for r in baseline["rows"]}
    if set(old) != keys or len(old) != len(baseline["rows"]):
        raise ValueError("Stage-19 baseline opportunities differ")
    rows, folds = [], []
    for held_out in sorted(paths):
        train = [r for r in pairs if r["seed"] != held_out]
        query = [r for r in pairs if r["seed"] == held_out]
        predictions = {}
        for label in LABELS:
            predictions[label], diagnostics = fit_pair_ridge(train, query, label=label)
            folds.append({"held_out_path": held_out, **diagnostics})
        for i, pair in enumerate(query):
            previous = old[(pair["seed"], pair["check_step"])]
            repeats = np.asarray(pair["replicate_tail_advantages"])
            if (previous["replicates"] != len(repeats)
                    or not np.isclose(previous["repeat_mean"], repeats.mean(), rtol=1e-12, atol=1e-15)
                    or not np.isclose(previous["repeat_variance"], repeats.var(ddof=1), rtol=1e-12, atol=1e-15)
                    or previous["single_draw_label"] != repeats[0]):
                raise ValueError("Stage-19 baseline labels differ from future cache")
            rows.append({"seed": held_out, "check_step": pair["check_step"],
                         "repeat_mean": previous["repeat_mean"], "repeat_variance": previous["repeat_variance"],
                         "replicates": len(repeats), **{
                             f"ridge_{k}_prediction": float(v[i]) for k, v in predictions.items()},
                         **{f"stage19_{k}_prediction": previous[f"{k}_prediction"] for k in LABELS}})
    return {"optimizer_seed": root, "branch_fit_seeds": list(fit_seeds),
            "trigger_eval_seeds": list(eval_seeds), "evaluation_paths_used": 0,
            "pairs": len(rows), "rows": rows, "folds": folds, "metrics": contrast_metrics(rows),
            "path_metrics": {str(s): contrast_metrics([r for r in rows if r["seed"] == s]) for s in sorted(paths)},
            "ridge_fits": len(folds), "critic_optimizer_steps": 0, "controller_training_iterations": 0,
            "additional_primitive_steps": 0,
            "cached_future_pair_labels": sum(len(r["replicate_tail_advantages"]) for r in pairs)}


def run_cell(args):
    noise, pairs = load_cached_pairs(args)
    data = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    if (data["status"] != "complete" or data["protocol"]["protocol_version"] != BASELINE_PROTOCOL
            or data["protocol"]["optimizer_seed"] != args.optimizer_seed):
        raise ValueError("baseline is not the completed Stage-19 root")
    baseline = data["cells"][0]
    if (baseline["branch_fit_seeds"] != args.branch_fit_seeds
            or baseline["trigger_eval_seeds"] != args.trigger_eval_seeds):
        raise ValueError("Stage-19 baseline seed roles differ")
    result = qualify(pairs, baseline, root=args.optimizer_seed, fit_seeds=args.branch_fit_seeds,
                     eval_seeds=args.trigger_eval_seeds)
    result["inherited_stage18_additional_primitive_steps"] = noise["additional_primitive_steps"]
    return result


def main(argv=None):
    parser = build_parser()
    for key in ("source", "endpoint", "baseline"):
        parser.add_argument(f"--{key}-result", type=Path, required=True)
    parser.add_argument("--pairs-per-seed", type=int, default=12)
    parser.add_argument("--noise-pairs-per-seed", type=int, default=2)
    parser.add_argument("--future-replicates", type=int, default=8)
    args = parser.parse_args(argv)
    output = {"status": "dry_run" if args.dry_run else "complete",
              "protocol": {"protocol_version": PROTOCOL_VERSION, "optimizer_seed": args.optimizer_seed,
                           "alpha": RIDGE_ALPHA, "labels": LABELS, "candidate": "ridge_averaged",
                           "baseline_protocol": BASELINE_PROTOCOL,
                           **{f"{k}_result": str(getattr(args, f"{k}_result"))
                              for k in ("source", "endpoint", "baseline")},
                           "evidence_role": "cached_regularized_pair_diagnostic_only",
                           "policy_deployment": False, "additional_primitive_steps": 0},
              "cells": [] if args.dry_run else [run_cell(args)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
