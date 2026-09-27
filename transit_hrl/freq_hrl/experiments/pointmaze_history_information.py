"""Cached temporal-information probes, separate from policy qualification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from freq_hrl.domains.mujoco import PointMazeRegimeDriver
from .pointmaze_budgeted_trigger import build_parser
from .pointmaze_goal_validation import _json_ready
from .pointmaze_plan_validity_branching import _ridge_fit_predict
from .pointmaze_temporal_plan import (
    PROTOCOL_VERSION as CACHE_PROTOCOL, HORIZONS, METHODS, history_view,
)
from .pointmaze_timing_pair import WINDOWED_PROTOCOL_VERSION


PROTOCOL_VERSION = "pointmaze_history_information_stage27_v1_diagnostic"
LAGS = (1, 10, 25, 50)
DT_SECONDS = .01
RIDGE_ALPHA = 1.
OBJECTIVES = ("target_motion", "timing_response")


def load_cache(source_path, controller_path, *, root):
    source = json.loads(source_path.read_text())
    controller = json.loads(controller_path.read_text())
    for result, protocol in ((source, CACHE_PROTOCOL), (controller, WINDOWED_PROTOCOL_VERSION)):
        if (result["status"] != "complete" or result["protocol"]["protocol_version"] != protocol
                or result["protocol"]["optimizer_seed"] != root or len(result["cells"]) != 1):
            raise ValueError("history probe source is not the registered completed root")
    cell = source["cells"][0]
    if cell["controller_selected_iteration"] != controller["cells"][0]["controller_selected_iteration"]:
        raise ValueError("temporal cache and controller checkpoint differ")
    cache_path = Path(cell["raw_server_directory"]) / "temporal_pairs.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        x, curves = cache["sequences"], cache["curves"]
        names = cache["feature_names"].tolist()
        rows = [{"seed": int(s), "check_step": int(t), "role": str(role)}
                for s, t, role in zip(cache["seeds"], cache["check_steps"], cache["roles"])]
    if (x.shape != (len(rows), 64, 23) or curves.shape != (len(rows), 3)
            or names != cell["feature_names"] or not np.isfinite(x).all()
            or not np.isfinite(curves).all()):
        raise ValueError("temporal raw cache has an incomplete schema")
    roles = cell["temporal_seed_roles"]
    if set(roles["fit"]).intersection(roles["evaluation"]):
        raise ValueError("temporal fit and evaluation paths overlap")
    keys = [(r["seed"], r["check_step"]) for r in rows]
    if len(set(keys)) != len(rows):
        raise ValueError("temporal cache repeats an opportunity")
    for role, count in (("fit", cell["training_pairs"]), ("evaluation", cell["evaluation_pairs"])):
        selected = [r for r in rows if r["role"] == role]
        if len(selected) != count or set(r["seed"] for r in selected) != set(roles[role]):
            raise ValueError("temporal raw cache differs from the source roster")
        coverage = {str(seed): {str(offset): sum(r["seed"] == seed and r["check_step"] % 50 == offset
                                               for r in selected) for offset in (0, 5, 10, 15, 20)}
                    for seed in roles[role]}
        if coverage != cell["coverage"][role]:
            raise ValueError("temporal raw cache coverage differs from the source")
    expected = {(r["seed"], r["check_step"]): r["curve"] for r in cell["rows"]}
    actual = {(r["seed"], r["check_step"]): y.tolist() for r, y in zip(rows, curves)
              if r["role"] == "evaluation"}
    if actual != expected or len(rows) != cell["training_pairs"] + cell["evaluation_pairs"]:
        raise ValueError("temporal raw cache labels differ from the scored source")
    return source, controller["protocol"], x, curves, rows, names, cache_path


def causal_design(x, rows, names, *, method, root):
    target = [names.index(f"measured_{i}") for i in range(2)]
    valid = names.index("valid_observation")
    if not np.all(x[:, [-1, *(-1 - lag for lag in LAGS)], valid] == 1):
        raise ValueError("history probe requires observed, unpadded velocity lags")
    view = history_view(x, rows, method, root=root)
    current = view[:, -1].astype(np.float64)
    slopes = [(current[:, target] - view[:, -1-lag, target]) / (lag * DT_SECONDS) for lag in LAGS]
    return np.column_stack((current, *slopes))


def target_motion_labels(x, rows, names, *, controller_protocol):
    measured = [names.index(f"measured_{i}") for i in range(6)]
    horizon = controller_protocol["horizon"]
    drivers = {seed: PointMazeRegimeDriver(seed=seed, horizon=horizon, dt_seconds=DT_SECONDS,
                                         **controller_protocol["task_options"])
               for seed in sorted(set(r["seed"] for r in rows))}
    labels = []
    for sequence, row in zip(x, rows):
        driver, check = drivers[row["seed"]], row["check_step"]
        steps = np.maximum(0, np.arange(check-63, check+1))
        prefix = np.stack([np.concatenate(driver.sample(int(t))) for t in steps])
        if not np.allclose(sequence[:, measured], prefix, rtol=0, atol=1e-6):
            raise ValueError("regenerated exogenous prefix differs from the temporal cache")
        current = driver.sample(check)[0]
        labels.append(np.stack([(driver.sample(check+h)[0] - current) / (h * DT_SECONDS)
                                for h in HORIZONS]))
    return np.asarray(labels).reshape(len(rows), -1), len(drivers) * (horizon+1)


def fit_probes(x, targets, rows, names, *, root):
    train = np.array([r["role"] == "fit" for r in rows])
    query = np.array([r["role"] == "evaluation" for r in rows])
    if set(r["seed"] for r, keep in zip(rows, train) if keep).intersection(
            r["seed"] for r, keep in zip(rows, query) if keep):
        raise ValueError("history probe training and query paths overlap")
    predictions, fits = {}, {}
    for method in METHODS:
        design = causal_design(x, rows, names, method=method, root=root)
        predictions[method], fits[method] = {}, {}
        for objective in OBJECTIVES:
            y = targets[objective][train]
            heads = [_ridge_fit_predict(design[train], y[:, i], design[query], alpha=RIDGE_ALPHA)
                     for i in range(y.shape[1])]
            predictions[method][objective] = np.column_stack([p for p, _ in heads])
            fits[method][objective] = {"training_rows": int(train.sum()), "scalar_linear_solves": len(heads),
                                      "parameter_count": (design.shape[1]+1)*len(heads),
                                      "alpha": RIDGE_ALPHA, "feature_mean": heads[0][1]["feature_mean"],
                                      "feature_scale": heads[0][1]["feature_scale"],
                                      "weights": np.column_stack([d["weights"] for _, d in heads]).tolist()}
    return predictions, fits


def summarize(rows):
    metrics = {}
    for objective, width in (("target_motion", 2), ("timing_response", 1)):
        truth = np.array([r[objective] for r in rows])
        predictions = {m: np.array([r["predictions"][m][objective] for r in rows]) for m in METHODS}
        predictions["zero"] = np.zeros_like(truth)
        if objective == "target_motion":
            predictions["lag1_extrapolation"] = np.array([r["lag1_extrapolation"] for r in rows])
        mse = {m: float(np.mean((p-truth)**2)) for m, p in predictions.items()}
        metrics[objective] = {"rate_mse": mse, "rate_mse_by_horizon": {
            m: np.mean(((p-truth)**2).reshape(len(rows), 3, width), axis=(0, 2)).tolist()
            for m, p in predictions.items()}, "current_minus_history_mse": mse["current_repeat"]-mse["history"]}
    truth = np.array([r["timing_response"][-1] for r in rows]) * HORIZONS[-1] * DT_SECONDS
    choices = {m: np.array([r["predictions"][m]["timing_response"][-1] > 0 for r in rows], dtype=int)
               for m in METHODS}
    choices.update(always_wait=np.zeros(len(rows), dtype=int), always_now=np.ones(len(rows), dtype=int))
    metrics["timing_response"].update(
        now_counts={m: int(a.sum()) for m, a in choices.items()},
        history_ise_benefit_vs_control={m: float(np.mean((choices["history"]-a)*truth))
                                      for m, a in choices.items() if m != "history"})
    return {"opportunities": len(rows), **metrics}


def run_cell(args):
    source, controller_protocol, x, curves, rows, names, cache_path = load_cache(
        args.source_result, args.controller_result, root=args.optimizer_seed)
    motion, generated_points = target_motion_labels(x, rows, names, controller_protocol=controller_protocol)
    targets = {"target_motion": motion, "timing_response": curves / (np.array(HORIZONS)*DT_SECONDS)}
    predictions, fits = fit_probes(x, targets, rows, names, root=args.optimizer_seed)
    query = [i for i, r in enumerate(rows) if r["role"] == "evaluation"]
    lag1 = causal_design(x, rows, names, method="history", root=args.optimizer_seed)[:, 23:25]
    scores = [{"seed": rows[i]["seed"], "check_step": rows[i]["check_step"],
               **{key: value[i] for key, value in targets.items()}, "lag1_extrapolation": np.tile(lag1[i], 3),
               "predictions": {m: {key: values[j] for key, values in predicted.items()}
                               for m, predicted in predictions.items()}} for j, i in enumerate(query)]
    cell = source["cells"][0]
    return {"optimizer_seed": args.optimizer_seed, "temporal_seed_roles": cell["temporal_seed_roles"],
            "training_rows": cell["training_pairs"], "evaluation_rows": len(scores),
            "controller_selected_iteration": cell["controller_selected_iteration"],
            "new_environment_primitive_steps": 0, "controller_updates": 0, "optimizer_updates": 0,
            "linear_fits": sum(len(f) for f in fits.values()),
            "scalar_linear_solves": sum(d["scalar_linear_solves"] for f in fits.values() for d in f.values()),
            "generated_exogenous_tape_points": generated_points,
            "reused_source_primitive_steps": sum(cell["controller_reconstruction_primitive_steps"].values())
            + cell["factual_replay_primitive_steps"] + cell["temporal_replay_primitive_steps"],
            "raw_server_cache": str(cache_path), "fits": fits, "rows": scores, "metrics": summarize(scores),
            "path_metrics": {str(seed): summarize([r for r in scores if r["seed"] == seed])
                             for seed in cell["temporal_seed_roles"]["evaluation"]}}


def main(argv=None):
    parser = build_parser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--controller-result", type=Path, required=True)
    args = parser.parse_args(argv)
    output = {"status": "dry_run" if args.dry_run else "complete", "protocol": {
        "protocol_version": PROTOCOL_VERSION, "optimizer_seed": args.optimizer_seed,
        "source_result": str(args.source_result), "controller_result": str(args.controller_result),
        "evidence_role": "reused_temporal_cache_information_diagnostic_only", "policy_deployment": False,
        "velocity_lags_steps": LAGS, "prediction_horizons_steps": HORIZONS,
        "ridge_alpha": RIDGE_ALPHA, "ridge_loss": "sum_squared_error_plus_l2_unpenalized_intercept",
        "methods": METHODS, "dt_seconds": DT_SECONDS},
        "cells": [] if args.dry_run else [run_cell(args)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(output), indent=2, sort_keys=True)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
