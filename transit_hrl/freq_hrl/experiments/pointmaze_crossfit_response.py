"""Fresh equal-call qualification of path-cross-fitted response correction."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from freq_hrl.core.plan_response import CrossFittedPlanResponseCritic
from freq_hrl.domains.mujoco import RelativeSubgoalAdapter
from . import pointmaze_forecast_response as response
from . import pointmaze_plan_hold as hold
from .pointmaze_budgeted_trigger import build_parser
from .pointmaze_goal_validation import _json_ready, pointmaze_goal_bounds
from .pointmaze_history_information import DT_SECONDS, RIDGE_ALPHA
from .pointmaze_plan_value_qualification import _make_task
from .pointmaze_plan_validity_branching import _task_options


PROTOCOL_VERSION = "pointmaze_crossfit_response_stage32_v1_development"
METHODS = tuple(f"{kind}_{method}" for kind in ("crossfit", "linear") for method in response.METHODS)
PRIMARY = "crossfit_history"


def evaluation_paths(root, *, preflight):
    base = {208001:3_319_000, 209011:3_320_000, 209061:3_321_000}[root]
    return list(range(base+101, base+(103 if preflight else 109)))


def load_training(args, *, selected_iteration):
    source = json.loads(args.response_result.read_text())
    if (source["status"] != "complete" or source["protocol"]["protocol_version"] != response.PROTOCOL_VERSION
            or source["protocol"]["optimizer_seed"] != args.optimizer_seed or len(source["cells"]) != 1):
        raise ValueError("cross-fit training source differs from the frozen Stage-31 root")
    cell = source["cells"][0]
    if cell["controller_selected_iteration"] != selected_iteration:
        raise ValueError("cross-fit training and controller checkpoints differ")
    path = Path(cell["raw_server_directory"])/"forecast_response.npz"
    with np.load(path, allow_pickle=False) as raw:
        train = {name:raw["train_"+name] for name in ("x", "curve", "seeds", "steps", "candidate")}
        designs = {method:raw[method+"_train_design"] for method in response.METHODS}
    count = cell["training_pairs"]
    if (train["x"].shape != (count, 64, 23) or train["curve"].shape != (count, 5)
            or not np.all(train["x"][:, :, -1] == 1) or np.any(train["steps"] < 64)
            or set(train["seeds"]) != set(cell["fit_paths"])):
        raise ValueError("cross-fit training cache lacks complete frozen fit paths")
    return train, designs, cell, path


def fit_response(train, query, designs, models, *, root):
    if set(train["seeds"]).intersection(r["seed"] for r in query):
        raise ValueError("cross-fit training and query paths overlap")
    predictions, fits, raw_models, query_designs = {}, {}, {}, {}
    for method in response.METHODS:
        x, q = designs[method], response.design(query, models, method=method, root=root)
        critic = CrossFittedPlanResponseCritic(durations_seconds=np.asarray(hold.HORIZONS)*DT_SECONDS,
                                              ridge_alpha=RIDGE_ALPHA)
        critic.fit(x, train["curve"], groups=train["seeds"])
        key = "crossfit_"+method
        predictions[key], predictions["linear_"+method] = critic.predict_rates(q), critic.linear.predict_rates(q)
        metadata = {k:v for k,v in critic.fitted.items() if k not in
                    ("training_basis", "kernel_column_mean", "residual_mean", "dual_weights", "out_of_fold_rates")}
        metadata["out_of_fold_settled_mse"] = float(np.mean(
            (critic.fitted["out_of_fold_rates"][:, -1]-train["curve"][:, -1]/1.5)**2))
        fits[key], fits["linear_"+method] = metadata, critic.linear.fitted
        raw_models[method], query_designs[method] = critic.fitted, q
    if not all(np.isfinite(p).all() for p in predictions.values()):
        raise RuntimeError("cross-fit response predictions are non-finite")
    return predictions, fits, raw_models, query_designs


def summarize(rows):
    truth = np.stack([r["curve"] for r in rows])
    rates = truth/(np.asarray(hold.HORIZONS)*DT_SECONDS)
    predictions = {m:np.stack([r["predicted_rates"][m] for r in rows]) for m in METHODS}
    mse = {m:np.mean((p-rates)**2, axis=0) for m,p in predictions.items()}
    mse["zero_value"] = np.mean(rates**2, axis=0)
    choices = {m:(p[:, -1] > 0).astype(int) for m,p in predictions.items()}
    choices.update(always_keep=np.zeros(len(rows), dtype=int), always_renew=np.ones(len(rows), dtype=int))
    gains = {m:float(np.mean((choices[PRIMARY]-a)*truth[:, -1])) for m,a in choices.items() if m != PRIMARY}
    return {"opportunities":len(rows), "rate_mse_by_horizon":mse,
            "settled_rate_mse":{m:float(v[-1]) for m,v in mse.items()},
            "renew_counts":{m:int(a.sum()) for m,a in choices.items()}, "settled_ise_benefit_vs_control":gains,
            "prediction_gate_passed":all(mse[PRIMARY][-1] < v[-1] for m,v in mse.items() if m != PRIMARY),
            "decision_gate_passed":all(g > 0 for g in gains.values())}


def run_cell(args):
    torch.set_num_threads(1)
    cache, source, controller, scale, checkpoint, factual = hold.load_controller(args)
    if controller.config.state_encoder != "mlp" or scale.history_steps != 64:
        raise ValueError("cross-fit proposal reconstruction requires the frozen stateless controller")
    train, designs, prior, training_cache = load_training(args, selected_iteration=source["controller_selected_iteration"])
    models, motion = response.load_motion(args, selected_iteration=source["controller_selected_iteration"])
    paths = evaluation_paths(args.optimizer_seed, preflight=args.horizon == 300)
    inherited = {"temporal_seed_roles":{**cache["temporal_seed_roles"], "response_fit":prior["fit_paths"],
                 "response_query":prior["evaluation_paths"], "motion_fit":motion["fit_paths"],
                 "motion_query":motion["evaluation_paths"],
                 "hold_query":hold.path_roles(args.optimizer_seed, preflight=args.horizon == 300)["evaluation"]}}
    hold.validate_paths(args, {"fit":[], "evaluation":paths}, inherited)
    cases = response.query_cases(args.optimizer_seed, paths, horizon=args.horizon, pairs_per_path=args.pairs_per_path)
    print(f"frozen training designs; sampling {len(cases)} fresh cross-fit response pairs", flush=True)
    query = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=hold.init_worker, initargs=(controller, args, scale)) as pool:
        for future in as_completed([pool.submit(hold.sample_case, case) for case in cases]):
            query.append(future.result())
            if len(query)%20 == 0 or len(query) == len(cases):
                print(f"cross-fit response pairs complete: {len(query)}/{len(cases)}", flush=True)
    query.sort(key=lambda r:(r["seed"], r["check_step"]))
    task = _make_task(env_id=args.env_id, seed=paths[0], horizon=args.horizon, **_task_options(args))
    try:
        low, high = pointmaze_goal_bounds(task.environment)
    finally:
        task.environment.close()
    adapter = RelativeSubgoalAdapter(maximum_delta=np.full(2, args.maximum_subgoal_delta, dtype=np.float32),
                                    world_low=low, world_high=high)
    for row in query:
        row["candidate_plan"], row["proposal_state"] = response.proposal(row, controller, adapter, history_steps=64)
    predictions, fits, raw_models, query_designs = fit_response(
        train, query, designs, models, root=args.optimizer_seed)
    for method in response.METHODS:
        np.testing.assert_allclose(fits["linear_"+method]["weights"], prior["fits"][method]["weights"], rtol=0, atol=1e-12)
    scores = [{**{k:v for k,v in row.items() if k not in ("sequence", "step_ise", "proposal_state")},
               "predicted_rates":{m:p[i] for m,p in predictions.items()}} for i,row in enumerate(query)]
    metrics = summarize(scores)
    path_metrics = {str(seed):summarize([r for r in scores if r["seed"] == seed]) for seed in paths}
    raw_dir = args.output.resolve().parent.with_name(args.output.parent.name+"_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_dir/"crossfit_response.npz", query_x=np.stack([r["sequence"] for r in query]),
        query_curve=np.stack([r["curve"] for r in query]), query_step_ise=np.stack([r["step_ise"] for r in query]),
        query_seeds=[r["seed"] for r in query], query_steps=[r["check_step"] for r in query],
        query_candidate=np.stack([r["candidate_plan"] for r in query]),
        query_proposal_states=np.stack([r["proposal_state"] for r in query]),
        **{m+"_query_design":q for m,q in query_designs.items()},
        **{m+"_prediction":p for m,p in predictions.items()},
        **{m+"_"+key:model[key] for m,model in raw_models.items() for key in
           ("kernel_column_mean", "residual_mean", "dual_weights", "out_of_fold_rates")})
    linear_fits = sum(raw_models[m]["linear_fits"] for m in response.METHODS)
    kernel_solves = len(raw_models)
    return {"optimizer_seed":args.optimizer_seed, "evaluation_paths":paths, "fit_paths":prior["fit_paths"],
            "training_pairs":len(train["curve"]), "evaluation_pairs":len(query),
            "controller_selected_iteration":source["controller_selected_iteration"], "controller_checkpoint":str(checkpoint),
            "factual_replay":factual, "factual_replay_primitive_steps":args.horizon,
            "fresh_pair_primitive_steps":sum(r["primitive_steps"] for r in query),
            "controller_reconstruction_primitive_steps":0, "controller_updates":0, "motion_updates":0, "physical_model_updates":0,
            "candidate_proposal_inference_calls":len(query), "reused_training_candidate_proposals":len(train["curve"]),
            "linear_fits":linear_fits, "kernel_solves":kernel_solves, "critic_fits":linear_fits+kernel_solves,
            "multi_rhs_linear_solves":linear_fits+kernel_solves, "scalar_rhs_count":5*(linear_fits+kernel_solves),
            "fits":fits, "rows":scores, "metrics":metrics, "path_metrics":path_metrics,
            "path_bootstrap_intervals":response.path_intervals(path_metrics, root=args.optimizer_seed),
            "development_gate_passed":metrics["prediction_gate_passed"] and metrics["decision_gate_passed"],
            "raw_training_cache":str(training_cache), "raw_server_directory":str(raw_dir),
            "raw_server_bytes":(raw_dir/"crossfit_response.npz").stat().st_size}


def main(argv=None):
    parser = build_parser()
    for name in ("source-result", "controller-result", "hold-result", "motion-result", "response-result"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--pairs-per-path", type=int, default=15)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    output = {"status":"dry_run" if args.dry_run else "complete", "protocol":{
        "protocol_version":PROTOCOL_VERSION, "optimizer_seed":args.optimizer_seed, "methods":METHODS, "primary":PRIMARY,
        **{name:str(getattr(args, name)) for name in ("source_result", "controller_result", "hold_result", "motion_result", "response_result")},
        "hold_steps":100, "settlement_steps":150, "response_horizons_steps":hold.HORIZONS,
        "forecast_horizons_steps":response.MOTION_HORIZONS[1:], "ridge_alpha":RIDGE_ALPHA,
        "kernel":"centered_rbf_on_training_standardized_design", "kernel_width_squared":"design_columns",
        "crossfit":"leave_one_training_path_out", "pairs_per_path":args.pairs_per_path, "workers":args.workers,
        "path_bootstrap_draws":response.BOOTSTRAP_DRAWS, "evaluation_paths":evaluation_paths(args.optimizer_seed, preflight=args.horizon == 300),
        "policy_deployment":False, "evidence_role":"fresh_path_crossfit_plan_response_development_only"},
        "cells":[] if args.dry_run else [run_cell(args)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(output), indent=2, sort_keys=True)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
