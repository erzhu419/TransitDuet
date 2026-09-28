"""Qualify independent motion inference without retraining physical response."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from freq_hrl.core.causal_motion import CausalMotionForecaster
from freq_hrl.domains.mujoco import PointMazeRegimeDriver
from .pointmaze_budgeted_trigger import build_parser
from .pointmaze_goal_validation import _json_ready
from .pointmaze_history_information import DT_SECONDS, LAGS, RIDGE_ALPHA
from .pointmaze_plan_hold import validate_paths
from .pointmaze_state_response import HISTORY, STRIDE, PROTOCOL_VERSION as SOURCE_PROTOCOL, path_roles as source_roles
from .pointmaze_timing_pair import WINDOWED_PROTOCOL_VERSION


PROTOCOL_VERSION = "pointmaze_separate_motion_stage30_v1_development"
HORIZONS = (1, 10, 25, 50, 100)
METHODS = ("history", "current_repeat", "shuffled_history")


def evaluation_paths(root, *, preflight):
    base = {208001:3_299_000, 209011:3_300_000, 209061:3_301_000}[root]
    return list(range(base+101, base+(103 if preflight else 109)))


def sample_rows(paths, *, horizon, bounded=True):
    stop = horizon-max(HORIZONS)+1 if bounded else horizon
    return [{"seed":seed, "step":step} for seed in paths for step in range(HISTORY, stop, STRIDE)]


def observed_view(history, rows, *, method, root):
    x = np.asarray(history).copy()
    if method == "current_repeat":
        return np.repeat(x[:, -1:], x.shape[1], axis=1)
    if method == "shuffled_history":
        for i, row in enumerate(rows):
            rng = np.random.default_rng(np.random.SeedSequence([root, row["seed"], row["step"], 30_030]))
            x[i, :-1] = x[i, rng.permutation(x.shape[1]-1)]
    elif method != "history":
        raise ValueError("unknown motion view")
    return x


def tapes_for(paths, *, protocol):
    horizon = protocol["horizon"]
    tapes = {}
    for seed in paths:
        driver = PointMazeRegimeDriver(seed=seed, horizon=horizon, dt_seconds=DT_SECONDS,
                                      **protocol["task_options"])
        tapes[seed] = np.stack([np.concatenate(driver.sample(t)) for t in range(horizon+1)])
    return tapes


def labeled_motion(tapes, rows, *, cached_history=None):
    history = np.stack([tapes[r["seed"]][r["step"]-HISTORY+1:r["step"]+1] for r in rows])
    if cached_history is not None and not np.array_equal(history, cached_history):
        raise ValueError("regenerated motion prefix differs from the cached observations")
    durations = np.asarray(HORIZONS, dtype=np.float64)*DT_SECONDS
    rates = np.stack([(tapes[r["seed"]][r["step"]+np.asarray(HORIZONS)].astype(np.float64)
                       - tapes[r["seed"]][r["step"]])/durations[:, None] for r in rows])
    return history, rates


def load_cache(args):
    source = json.loads(args.source_result.read_text())
    controller = json.loads(args.controller_result.read_text())
    for result, protocol in ((source, SOURCE_PROTOCOL), (controller, WINDOWED_PROTOCOL_VERSION)):
        if (result["status"] != "complete" or result["protocol"]["protocol_version"] != protocol
                or result["protocol"]["optimizer_seed"] != args.optimizer_seed or len(result["cells"]) != 1):
            raise ValueError("separate-motion source is not the registered completed root")
    cell, original = source["cells"][0], controller["cells"][0]
    if cell["controller_selected_iteration"] != original["controller_selected_iteration"]:
        raise ValueError("motion cache and frozen physical-response controller differ")
    if controller["protocol"]["horizon"] != args.horizon:
        raise ValueError("motion cache horizon differs from the registered source")
    roles = source_roles(args.optimizer_seed, preflight=args.horizon==300)
    if roles != cell["state_seed_roles"]:
        raise ValueError("motion cache path roles differ from the registered source")
    path = Path(cell["raw_server_directory"])/"state_response.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {key:data[key] for key in ("train_x", "train_y", "query_x", "query_y", "query_seeds",
                  "query_steps", "history_action_teacher_mean", "current_action_teacher_mean")}
    train_rows = sample_rows(roles["fit"], horizon=args.horizon, bounded=False)
    query_rows = sample_rows(roles["evaluation"], horizon=args.horizon, bounded=False)
    if (arrays["train_x"].shape != (len(train_rows), HISTORY, 12)
            or arrays["query_x"].shape != (len(query_rows), HISTORY, 12)
            or arrays["train_y"].shape != (len(train_rows), 10)
            or arrays["query_y"].shape != (len(query_rows), 10)
            or list(zip(arrays["query_seeds"], arrays["query_steps"])) != [(r["seed"],r["step"]) for r in query_rows]):
        raise ValueError("motion cache rows differ from the source trajectory order")
    return cell, controller["protocol"], arrays, train_rows, query_rows, path


def fit_models(train_x, train_y, train_rows, query_x, query_rows, *, root):
    if set(r["seed"] for r in train_rows).intersection(r["seed"] for r in query_rows):
        raise ValueError("motion training and evaluation paths overlap")
    models, predictions = {}, {}
    for method in METHODS:
        model = CausalMotionForecaster(observed_dim=6, velocity_channels=(0,1), horizon_steps=HORIZONS,
                                      dt_seconds=DT_SECONDS, velocity_lags=LAGS, ridge_alpha=RIDGE_ALPHA)
        model.fit(observed_view(train_x, train_rows, method=method, root=root), train_y)
        predictions[method] = model.predict_rates(observed_view(query_x, query_rows, method=method, root=root))
        models[method] = model
    lag1 = (query_x[:, -1].astype(np.float64)-query_x[:, -2])/DT_SECONDS
    predictions["lag1_extrapolation"] = np.repeat(lag1[:, None], len(HORIZONS), axis=1)
    predictions["zero"] = np.zeros_like(predictions["history"])
    if not all(np.isfinite(p).all() for p in predictions.values()):
        raise RuntimeError("separate-motion predictions are non-finite")
    return models, predictions


def motion_metrics(truth, predictions):
    errors = {method:(prediction-truth)**2 for method,prediction in predictions.items()}
    target_mse = {method:np.mean(error[:, :, :2], axis=(0,2)) for method,error in errors.items()}
    one_step = {method:float(value[0]) for method,value in target_mse.items()}
    planning = {method:float(value[1:].mean()) for method,value in target_mse.items()}
    one_step_gate = one_step["history"] < one_step["current_repeat"]
    planning_gate = all(planning["history"] < planning[c] for c in (*METHODS[1:], "lag1_extrapolation", "zero"))
    return {"rows":len(truth), "target_rate_mse_by_horizon":target_mse, "one_step_target_rate_mse":one_step,
            "planning_target_rate_mse":planning, "all_channel_rate_mse_by_horizon":{
                method:np.mean(error,axis=0) for method,error in errors.items()},
            "one_step_gate_passed":one_step_gate, "planning_gate_passed":planning_gate,
            "motion_gate_passed":one_step_gate and planning_gate}


def compose_state_mean(physical_state_mean, external_rates, *, scales):
    mean = np.asarray(physical_state_mean).copy()
    mean[:, 4:] = external_rates[:, 0]*DT_SECONDS/scales[4:]
    return mean


def run_cell(args):
    cell, protocol, arrays, train_rows, old_rows, cache_path = load_cache(args)
    paths = evaluation_paths(args.optimizer_seed, preflight=args.horizon==300)
    validate_paths(args, {"fit":[], "evaluation":paths}, {"temporal_seed_roles":cell["state_seed_roles"]})
    source_paths = cell["state_seed_roles"]["fit"]+cell["state_seed_roles"]["evaluation"]
    tapes = tapes_for(source_paths+paths, protocol=protocol)
    for rows, x, y in ((train_rows, arrays["train_x"], arrays["train_y"]),
                      (old_rows, arrays["query_x"], arrays["query_y"])):
        observed = np.stack([tapes[r["seed"]][r["step"]-63:r["step"]+1] for r in rows])
        following = np.stack([tapes[r["seed"]][r["step"]+1]-tapes[r["seed"]][r["step"]] for r in rows])
        if not np.array_equal(x[:, :, 4:10], observed) or not np.array_equal(y[:, 4:10], following):
            raise ValueError("regenerated motion labels differ from the source cache")
    train_mask = np.array([r["step"]+max(HORIZONS)<=args.horizon for r in train_rows])
    old_mask = np.array([r["step"]+max(HORIZONS)<=args.horizon for r in old_rows])
    train_rows = [r for r,keep in zip(train_rows,train_mask) if keep]
    old_rows = [r for r,keep in zip(old_rows,old_mask) if keep]
    train_x, train_y = labeled_motion(tapes, train_rows, cached_history=arrays["train_x"][train_mask, :, 4:10])
    query_rows = sample_rows(paths, horizon=args.horizon)
    query_x, query_y = labeled_motion(tapes, query_rows)
    models, predictions = fit_models(train_x, train_y, train_rows, query_x, query_rows, root=args.optimizer_seed)
    old_x, old_y = labeled_motion(tapes, old_rows, cached_history=arrays["query_x"][old_mask, :, 4:10])
    bridge = {}
    target_scale = np.asarray(cell["scales"]["target_scale"])
    for method, model in models.items():
        rates = model.predict_rates(observed_view(old_x, old_rows, method=method, root=args.optimizer_seed))
        physical = arrays["history_action_teacher_mean"][old_mask]
        mean = compose_state_mean(physical, rates, scales=target_scale)
        if not np.array_equal(mean[:, :4], physical[:, :4]):
            raise RuntimeError("separate motion changed the frozen physical response")
        bridge[method] = {"one_step_target_rate_mse":float(np.mean((rates[:, 0, :2]-old_y[:, 0, :2])**2)),
                          "physical_prediction_unchanged":True}
    for method in ("history_action", "current_action"):
        mean = arrays[method+"_teacher_mean"][old_mask, 4:6]*target_scale[4:6]/DT_SECONDS
        bridge["stage29_"+method] = {"one_step_target_rate_mse":float(np.mean((mean-old_y[:, 0, :2])**2))}
    raw_dir = args.output.resolve().parent.with_name(args.output.parent.name+"_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_dir/"motion_forecasts.npz", train_x=train_x, train_y=train_y, query_x=query_x,
        query_y=query_y, query_seeds=[r["seed"] for r in query_rows], query_steps=[r["step"] for r in query_rows],
        **{method+"_prediction":prediction for method,prediction in predictions.items()})
    metrics = motion_metrics(query_y, predictions)
    path_metrics = {str(seed):motion_metrics(query_y[mask], {m:p[mask] for m,p in predictions.items()})
                    for seed in paths for mask in [np.array([r["seed"]==seed for r in query_rows])]}
    audit_indices = [next(i for i,r in enumerate(query_rows) if r["seed"]==seed) for seed in paths]
    return {"optimizer_seed":args.optimizer_seed, "fit_paths":cell["state_seed_roles"]["fit"],
            "evaluation_paths":paths, "fit_rows":len(train_rows), "evaluation_rows":len(query_rows),
            "cached_bridge_rows":len(old_rows), "fits":{m:model.fitted for m,model in models.items()},
            "linear_fits":len(models), "multi_rhs_linear_solves":len(models),
            "scalar_rhs_count":len(models)*len(HORIZONS)*6,
            "generated_exogenous_tape_points":sum(len(tape) for tape in tapes.values()),
            "new_environment_primitive_steps":0, "controller_updates":0, "physical_model_updates":0,
            "controller_reconstruction_primitive_steps":0, "metrics":metrics, "path_metrics":path_metrics,
            "cached_bridge_metrics":bridge, "development_gate_passed":metrics["motion_gate_passed"],
            "audit_rows":[{**query_rows[i], "observed_target_lags":query_x[i, [-1,*(-1-lag for lag in LAGS)], :2],
                           "truth_rates":query_y[i], "predicted_rates":{m:p[i] for m,p in predictions.items()}}
                          for i in audit_indices], "raw_source_cache":str(cache_path),
            "raw_server_directory":str(raw_dir), "raw_server_bytes":{
                p.name:p.stat().st_size for p in raw_dir.iterdir() if p.is_file()}}


def main(argv=None):
    parser = build_parser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--controller-result", type=Path, required=True)
    args = parser.parse_args(argv)
    output = {"status":"dry_run" if args.dry_run else "complete", "protocol":{
        "protocol_version":PROTOCOL_VERSION, "optimizer_seed":args.optimizer_seed,
        "source_result":str(args.source_result), "controller_result":str(args.controller_result),
        "history_steps":HISTORY, "velocity_lags_steps":LAGS, "velocity_channels":[0,1],
        "forecast_horizons_steps":HORIZONS, "planning_horizons_steps":HORIZONS[1:],
        "dt_seconds":DT_SECONDS, "ridge_alpha":RIDGE_ALPHA, "methods":METHODS,
        "evaluation_paths":evaluation_paths(args.optimizer_seed, preflight=args.horizon==300),
        "policy_deployment":False, "evidence_role":"fresh_exogenous_motion_development_only"},
        "cells":[] if args.dry_run else [run_cell(args)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(output), indent=2, sort_keys=True)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
