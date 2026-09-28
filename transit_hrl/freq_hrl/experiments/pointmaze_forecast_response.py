"""Frozen motion forecasts for fresh-path, equal-call keep/renew decisions."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from freq_hrl.core.causal_motion import CausalMotionForecaster
from freq_hrl.core.plan_response import PlanResponseCritic, plan_response_features
from freq_hrl.domains.mujoco import RelativeSubgoalAdapter
from .pointmaze_budgeted_trigger import build_parser
from .pointmaze_goal_validation import _json_ready, pointmaze_goal_bounds
from .pointmaze_history_information import causal_design, DT_SECONDS, LAGS, RIDGE_ALPHA
from . import pointmaze_plan_hold as hold
from .pointmaze_plan_validity_branching import _task_options
from .pointmaze_plan_value_qualification import _make_task
from .pointmaze_separate_motion import PROTOCOL_VERSION as MOTION_PROTOCOL, HORIZONS as MOTION_HORIZONS, observed_view


PROTOCOL_VERSION = "pointmaze_forecast_response_stage31_v1_development"
METHODS = ("history", "current_repeat", "shuffled_history", "lag1_extrapolation", "zero_forecast",
           "raw_history", "raw_current")
BOOTSTRAP_DRAWS = 4096


def evaluation_paths(root, *, preflight):
    base = {208001:3_309_000, 209011:3_310_000, 209061:3_311_000}[root]
    return list(range(base+101, base+(103 if preflight else 109)))


def query_cases(root, paths, *, horizon, pairs_per_path):
    bins = np.arange(2, horizon//50)
    bins = bins[bins*50+20+hold.SETTLEMENT_STEPS<=horizon]
    if not 1 <= pairs_per_path <= len(bins):
        raise ValueError("forecast-response opportunities exceed the full-history grid")
    cases = []
    for seed in paths:
        rng = np.random.default_rng(np.random.SeedSequence([root,seed,31_031]))
        selected = sorted(map(int,rng.choice(bins,size=pairs_per_path,replace=False)))
        cases.extend({"seed":seed,"check_step":b*50+((i+seed)%5)*5} for i,b in enumerate(selected))
    return cases


def load_training(args, *, selected_iteration):
    source = json.loads(args.hold_result.read_text())
    if (source["status"] != "complete" or source["protocol"]["protocol_version"] != hold.PROTOCOL_VERSION
            or source["protocol"]["optimizer_seed"] != args.optimizer_seed or len(source["cells"]) != 1):
        raise ValueError("forecast-response training cache is not the registered plan-hold root")
    cell = source["cells"][0]
    if (cell["controller_selected_iteration"] != selected_iteration
            or cell["plan_hold_seed_roles"] != hold.path_roles(args.optimizer_seed,preflight=args.horizon==300)):
        raise ValueError("forecast-response cache controller or path roles differ")
    raw_path = Path(cell["raw_server_directory"])/"plan_hold_pairs.npz"
    with np.load(raw_path,allow_pickle=False) as raw:
        x, curves, costs = raw["sequences"], raw["curves"], raw["step_ise"]
        keys = list(zip(raw["seeds"].tolist(),raw["check_steps"].tolist(),raw["roles"].tolist()))
        names = raw["feature_names"].tolist()
    expected = sorted((case["seed"],case["check_step"],role) for role,paths in cell["plan_hold_seed_roles"].items()
                      for case in hold.cases_for_paths(args.optimizer_seed,paths,horizon=args.horizon,
                                                      pairs_per_path=source["protocol"]["pairs_per_path"]))
    if (keys != expected or x.shape != (len(keys),64,len(names)) or names != cell["feature_names"]
            or costs.shape != (len(keys),2,150)):
        raise ValueError("plan-hold raw training cache differs from the registered roster")
    derived = np.cumsum(costs[:,1]-costs[:,0],axis=1)[:,np.asarray(hold.HORIZONS)-1]
    if not np.allclose(curves,derived,rtol=0,atol=1e-12):
        raise ValueError("plan-hold curve labels differ from the raw arm costs")
    train = [{"seed":seed,"check_step":check,"sequence":sequence,"curve":curve,"feature_names":names}
             for (seed,check,role),sequence,curve in zip(keys,x,curves) if role=="fit" and check>=64]
    if any(not np.all(row["sequence"][:,-1]==1) for row in train):
        raise ValueError("forecast-response training requires complete observed prefixes")
    return train,cell,raw_path


def load_motion(args, *, selected_iteration):
    result = json.loads(args.motion_result.read_text())
    if (result["status"] != "complete" or result["protocol"]["protocol_version"] != MOTION_PROTOCOL
            or result["protocol"]["optimizer_seed"] != args.optimizer_seed or len(result["cells"]) != 1):
        raise ValueError("forecast-response motion source differs from the frozen root")
    cell = result["cells"][0]
    original = json.loads(Path(result["protocol"]["controller_result"]).read_text())["cells"][0]
    if original["controller_selected_iteration"] != selected_iteration:
        raise ValueError("motion and plan-response controllers differ")
    if args.optimizer_seed!=208001 and not cell["development_gate_passed"]:
        raise ValueError("full plan-response qualification requires qualified motion inference")
    if tuple(result["protocol"]["forecast_horizons_steps"])!=MOTION_HORIZONS:
        raise ValueError("frozen motion horizons changed")
    models = {}
    for method in METHODS[:3]:
        model = CausalMotionForecaster(observed_dim=6,velocity_channels=(0,1),horizon_steps=MOTION_HORIZONS,
                                      dt_seconds=DT_SECONDS,velocity_lags=LAGS,ridge_alpha=RIDGE_ALPHA)
        model.fitted = {key:np.asarray(value) if key in ("feature_mean","feature_scale","weights") else value
                        for key,value in cell["fits"][method].items()}
        models[method] = model
    return models,cell


def proposal(row, controller, adapter, *, history_steps):
    names, x = row["feature_names"], row["sequence"]
    physical = [names.index(f"physical_{i}") for i in range(4)]
    error = [names.index(f"target_error_{i}") for i in range(2)]
    measured = [names.index(f"measured_{i}") for i in range(6)]
    achieved = [names.index(f"achieved_{i}") for i in range(2)]
    state = np.concatenate((x[-1,physical],x[-1,error],x[-history_steps:,measured].reshape(-1))).astype(np.float32)
    output = controller.plan_goal(state,sample=False)
    return adapter.decode(np.asarray(output["action"],dtype=np.float32),x[-1,achieved]),state


def design(rows, models, *, method, root):
    names = rows[0]["feature_names"]
    x = np.stack([r["sequence"] for r in rows])
    if method.startswith("raw_"):
        return causal_design(x,rows,names,method="history" if method=="raw_history" else "current_repeat",root=root)
    measured = [names.index(f"measured_{i}") for i in range(6)]
    external = x[:,:,measured]
    motion_rows = [{"seed":r["seed"],"step":r["check_step"]} for r in rows]
    if method in models:
        rates = models[method].predict_rates(observed_view(external,motion_rows,method=method,root=root))[:,1:,:2]
    elif method == "lag1_extrapolation":
        rates = np.repeat(((external[:,-1,:2].astype(float)-external[:,-2,:2])/DT_SECONDS)[:,None],4,axis=1)
    elif method == "zero_forecast":
        rates = np.zeros((len(rows),4,2))
    else:
        raise ValueError("unknown forecast-response method")
    target = external[:,-1,:2].astype(float)
    future = target[:,None]+rates*(np.asarray(MOTION_HORIZONS[1:])*DT_SECONDS)[None,:,None]
    achieved = x[:,-1,[names.index(f"achieved_{i}") for i in range(2)]].astype(float)
    old = achieved+x[:,-1,[names.index(f"waypoint_error_{i}") for i in range(2)]]
    velocity = x[:,-1,[names.index(f"physical_{i}") for i in (2,3)]]
    return plan_response_features(current_state=x[:,-1],position=achieved,velocity=velocity,
        retained_plan=old,candidate_plan=np.stack([r["candidate_plan"] for r in rows]),
        observed_target=target,forecast_targets=future)


def fit_response(train, query, models, *, root):
    if set(r["seed"] for r in train).intersection(r["seed"] for r in query):
        raise ValueError("plan-response fit and query paths overlap")
    predictions, fits, designs = {}, {}, {}
    y = np.stack([r["curve"] for r in train])
    for method in METHODS:
        x, q = (design(rows,models,method=method,root=root) for rows in (train,query))
        critic = PlanResponseCritic(durations_seconds=np.asarray(hold.HORIZONS)*DT_SECONDS,ridge_alpha=RIDGE_ALPHA)
        critic.fit(x,y)
        predictions[method],fits[method],designs[method] = critic.predict_rates(q),critic.fitted,(x,q)
    if not all(np.isfinite(p).all() for p in predictions.values()):
        raise RuntimeError("forecast-response predictions are non-finite")
    return predictions,fits,designs


def summarize(rows):
    truth = np.stack([r["curve"] for r in rows])
    predictions = {m:np.stack([r["predicted_rates"][m] for r in rows]) for m in METHODS}
    mse = {m:np.mean((p-truth/(np.asarray(hold.HORIZONS)*DT_SECONDS))**2,axis=0) for m,p in predictions.items()}
    mse["zero_value"] = np.mean((truth/(np.asarray(hold.HORIZONS)*DT_SECONDS))**2,axis=0)
    choices = {m:(p[:,-1]>0).astype(int) for m,p in predictions.items()}
    choices.update(always_keep=np.zeros(len(rows),dtype=int),always_renew=np.ones(len(rows),dtype=int))
    gains = {m:float(np.mean((choices["history"]-a)*truth[:,-1])) for m,a in choices.items() if m!="history"}
    return {"opportunities":len(rows),"rate_mse_by_horizon":mse,
            "settled_rate_mse":{m:float(v[-1]) for m,v in mse.items()},
            "renew_counts":{m:int(a.sum()) for m,a in choices.items()},"settled_ise_benefit_vs_control":gains,
            "prediction_gate_passed":all(mse["history"][-1]<v[-1] for m,v in mse.items() if m!="history"),
            "decision_gate_passed":all(g>0 for g in gains.values())}


def path_intervals(path_metrics, *, root):
    controls = path_metrics[next(iter(path_metrics))]["settled_ise_benefit_vs_control"]
    values = np.array([[m["settled_ise_benefit_vs_control"][c] for c in controls] for m in path_metrics.values()])
    rng = np.random.default_rng(np.random.SeedSequence([root,31_039]))
    draws = rng.integers(0,len(values),size=(BOOTSTRAP_DRAWS,len(values)))
    endpoints = np.percentile(values[draws].mean(axis=1),[2.5,97.5],axis=0)
    return {control:{"mean":float(values[:,i].mean()),"ci95":endpoints[:,i].tolist(),
                     "paths":len(values),"bootstrap_draws":BOOTSTRAP_DRAWS} for i,control in enumerate(controls)}


def run_cell(args):
    torch.set_num_threads(1)
    cache,source,controller,scale,checkpoint,factual = hold.load_controller(args)
    if controller.config.state_encoder!="mlp" or scale.history_steps!=64:
        raise ValueError("cached proposal reconstruction requires the registered stateless controller")
    train,hold_cell,train_cache = load_training(args,selected_iteration=source["controller_selected_iteration"])
    models,motion_cell = load_motion(args,selected_iteration=source["controller_selected_iteration"])
    paths = evaluation_paths(args.optimizer_seed,preflight=args.horizon==300)
    inherited = {"temporal_seed_roles":{**cache["temporal_seed_roles"],
        **{f"hold_{k}":v for k,v in hold_cell["plan_hold_seed_roles"].items()},
        "motion_fit":motion_cell["fit_paths"],"motion_eval":motion_cell["evaluation_paths"]}}
    hold.validate_paths(args,{"fit":[],"evaluation":paths},inherited)
    cases = query_cases(args.optimizer_seed,paths,horizon=args.horizon,pairs_per_path=args.pairs_per_path)
    print(f"cached controller and motion frozen; sampling {len(cases)} fresh equal-call pairs",flush=True)
    query = []
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context("spawn"),
                             initializer=hold.init_worker,initargs=(controller,args,scale)) as pool:
        for future in as_completed([pool.submit(hold.sample_case,c) for c in cases]):
            query.append(future.result())
            if len(query)%20==0 or len(query)==len(cases):
                print(f"forecast-response pairs complete: {len(query)}/{len(cases)}",flush=True)
    query.sort(key=lambda r:(r["seed"],r["check_step"]))
    task = _make_task(env_id=args.env_id,seed=paths[0],horizon=args.horizon,**_task_options(args))
    try:
        low,high = pointmaze_goal_bounds(task.environment)
    finally:
        task.environment.close()
    adapter = RelativeSubgoalAdapter(maximum_delta=np.full(2,args.maximum_subgoal_delta,dtype=np.float32),
                                    world_low=low,world_high=high)
    for row in [*train,*query]:
        row["candidate_plan"],row["proposal_state"] = proposal(row,controller,adapter,history_steps=scale.history_steps)
    predictions,fits,designs = fit_response(train,query,models,root=args.optimizer_seed)
    scores = [{**{k:v for k,v in r.items() if k not in ("sequence","step_ise","proposal_state")},
               "predicted_rates":{m:p[i] for m,p in predictions.items()}} for i,r in enumerate(query)]
    metrics = summarize(scores)
    path_metrics = {str(seed):summarize([r for r in scores if r["seed"]==seed]) for seed in paths}
    raw_dir = args.output.resolve().parent.with_name(args.output.parent.name+"_raw")
    raw_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(raw_dir/"forecast_response.npz",train_x=np.stack([r["sequence"] for r in train]),
        query_x=np.stack([r["sequence"] for r in query]),train_curve=np.stack([r["curve"] for r in train]),
        query_curve=np.stack([r["curve"] for r in query]),query_step_ise=np.stack([r["step_ise"] for r in query]),
        train_seeds=[r["seed"] for r in train],train_steps=[r["check_step"] for r in train],
        query_seeds=[r["seed"] for r in query],query_steps=[r["check_step"] for r in query],
        train_candidate=np.stack([r["candidate_plan"] for r in train]),query_candidate=np.stack([r["candidate_plan"] for r in query]),
        query_proposal_states=np.stack([r["proposal_state"] for r in query]),
        **{m+"_"+k:v for m,(x,q) in designs.items() for k,v in (("train_design",x),("query_design",q),("prediction",predictions[m]))})
    return {"optimizer_seed":args.optimizer_seed,"evaluation_paths":paths,"fit_paths":hold_cell["plan_hold_seed_roles"]["fit"],
        "training_pairs":len(train),"evaluation_pairs":len(query),"excluded_incomplete_fit_prefixes":hold_cell["training_pairs"]-len(train),
        "controller_selected_iteration":source["controller_selected_iteration"],"controller_checkpoint":str(checkpoint),
        "factual_replay":factual,"factual_replay_primitive_steps":args.horizon,
        "fresh_pair_primitive_steps":sum(r["primitive_steps"] for r in query),"controller_reconstruction_primitive_steps":0,
        "controller_updates":0,"motion_updates":0,"physical_model_updates":0,
        "candidate_proposal_inference_calls":len(train)+len(query),"candidate_proposals_shared_across_methods":True,
        "critic_fits":len(fits),"multi_rhs_linear_solves":len(fits),"scalar_rhs_count":len(fits)*len(hold.HORIZONS),
        "fits":fits,"rows":scores,"metrics":metrics,"path_metrics":path_metrics,
        "path_bootstrap_intervals":path_intervals(path_metrics,root=args.optimizer_seed),
        "development_gate_passed":metrics["prediction_gate_passed"] and metrics["decision_gate_passed"],
        "raw_training_cache":str(train_cache),"raw_server_directory":str(raw_dir),
        "raw_server_bytes":(raw_dir/"forecast_response.npz").stat().st_size}


def main(argv=None):
    parser = build_parser()
    for name in ("source-result","controller-result","hold-result","motion-result"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--pairs-per-path",type=int,default=15)
    parser.add_argument("--workers",type=int,default=16)
    args = parser.parse_args(argv)
    output = {"status":"dry_run" if args.dry_run else "complete","protocol":{
        "protocol_version":PROTOCOL_VERSION,"optimizer_seed":args.optimizer_seed,"methods":METHODS,
        **{name:str(getattr(args,name)) for name in ("source_result","controller_result","hold_result","motion_result")},
        "hold_steps":hold.HOLD_STEPS,"settlement_steps":hold.SETTLEMENT_STEPS,"response_horizons_steps":hold.HORIZONS,
        "forecast_horizons_steps":MOTION_HORIZONS[1:],"ridge_alpha":RIDGE_ALPHA,
        "pairs_per_path":args.pairs_per_path,"workers":args.workers,"path_bootstrap_draws":BOOTSTRAP_DRAWS,
        "evaluation_paths":evaluation_paths(args.optimizer_seed,preflight=args.horizon==300),
        "policy_deployment":False,"evidence_role":"fresh_path_frozen_forecast_plan_response_development_only"},
        "cells":[] if args.dry_run else [run_cell(args)]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(output),indent=2,sort_keys=True)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
