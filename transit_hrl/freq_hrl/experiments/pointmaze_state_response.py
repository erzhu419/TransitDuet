"""Qualify action-conditioned predictive state before learning plan value."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from freq_hrl.domains.mujoco import RelativeSubgoalAdapter
from freq_hrl.rl.predictive_state import ActionConditionedStatePredictor, gaussian_prediction_loss
from .pointmaze_budgeted_trigger import balanced_jitter_schedule, build_parser
from .pointmaze_goal_validation import POINTMAZE_LOWER_ACTION_COST, _json_ready, pointmaze_goal_bounds, squash_box_action
from .pointmaze_plan_hold import load_controller, validate_paths, path_roles as hold_roles
from .pointmaze_plan_validity_branching import _task_options
from .pointmaze_plan_value_qualification import PointMazeRegimeFeatureBuilder, _make_task


PROTOCOL_VERSION = "pointmaze_state_response_stage29_v1_development"
METHODS = ("history_action", "current_action", "history_blind", "current_blind")
HISTORY, STRIDE, PERTURBATION = 64, 5, .25
ROLLOUT_HORIZONS = (1, 5, 10)


def path_roles(root, *, preflight):
    base = {208001:3_289_000, 209011:3_290_000, 209061:3_291_000}[root]
    return {"fit":list(range(base+1, base+(3 if preflight else 17))),
            "evaluation":list(range(base+101, base+(103 if preflight else 109)))}


def checks_for_path(seed, *, preflight):
    return [100*(i+1)+((i+seed)%5)*5 for i in range(1 if preflight else 10)]


def frame(observation, previous_action):
    return np.concatenate((observation.physical, observation.task_measurement, previous_action)).astype(np.float32)


def collect(controller, *, seed, role, args, scale, check=None, axis=None, sign=None):
    task = _make_task(env_id=args.env_id, seed=seed, horizon=args.horizon, **_task_options(args))
    try:
        observation = task.reset()
        low, high = pointmaze_goal_bounds(task.environment)
        adapter = RelativeSubgoalAdapter(maximum_delta=np.full(2, args.maximum_subgoal_delta, dtype=np.float32),
                                         world_low=low, world_high=high, action_cost=POINTMAZE_LOWER_ACTION_COST)
        history = PointMazeRegimeFeatureBuilder(time_scale=scale, task_dim=6)
        history.reset(observation)
        controller.reset_recurrent_inference()
        subgoal, previous = observation.achieved_goal.copy(), np.zeros(2, dtype=np.float32)
        schedule = balanced_jitter_schedule(seed=seed, horizon=args.horizon, period_steps=50, max_offset_steps=25)
        rng = np.random.default_rng(np.random.SeedSequence([args.optimizer_seed, seed, 29_029]))
        frames, actions, baseline_actions, calls = [], [], [], []
        stop = args.horizon if check is None else check+1
        for step in range(stop):
            frames.append(frame(observation, previous))
            if step in schedule:
                output = controller.plan_goal(history.upper_state(observation, oracle_context=None), sample=False)
                subgoal = adapter.decode(np.asarray(output["action"], dtype=np.float32), observation.achieved_goal)
                calls.append(step)
            output = controller.act_conditioned(history.lower_state(observation, subgoal=subgoal), sample=False)
            baseline = squash_box_action(np.asarray(output["action"], dtype=np.float32), task.action_low, task.action_high)
            requested = baseline.copy()
            if role == "fit":
                requested += rng.uniform(-PERTURBATION, PERTURBATION, size=2)
            if step == check:
                requested[axis] += sign*PERTURBATION
            requested = np.clip(requested, task.action_low, task.action_high).astype(np.float32)
            observation, _, terminated, truncated, _ = task.step(requested)
            if (terminated or truncated) and step+1 != args.horizon:
                raise RuntimeError("state-response trajectory ended early")
            actions.append(requested)
            baseline_actions.append(baseline)
            previous = requested.copy()
            history.update(observation)
        frames.append(frame(observation, previous))
        return {"seed":seed, "role":role, "frames":np.stack(frames), "actions":np.stack(actions),
                "baseline_actions":np.stack(baseline_actions), "calls":calls, "primitive_steps":stop}
    finally:
        task.environment.close()


def init_worker(controller, args, scale):
    global _WORKER
    torch.set_num_threads(1)
    _WORKER = controller, args, scale


def collect_job(job):
    controller, args, scale = _WORKER
    return collect(controller, args=args, scale=scale, **job)


def combine_effect(case, positive, negative, reference):
    check, axis = case["check_step"], case["axis"]
    prefix = reference["frames"][:check+1]
    if any(not np.array_equal(arm["frames"][:check+1], prefix) for arm in (positive, negative)):
        raise RuntimeError("actuator intervention does not share the factual prefix")
    expected_calls = [s for s in reference["calls"] if s <= check]
    if any(arm["calls"] != expected_calls for arm in (positive, negative)):
        raise RuntimeError("actuator intervention changed upper planning")
    if not np.array_equal(positive["frames"][-1,4:10], negative["frames"][-1,4:10]):
        raise RuntimeError("actuator intervention changed the exogenous stream")
    baseline = reference["actions"][check]
    for sign, arm in ((1,positive),(-1,negative)):
        if not np.array_equal(arm["baseline_actions"][check], baseline):
            raise RuntimeError("actuator intervention changed the factual lower action")
        expected = baseline.copy()
        expected[axis] += sign*PERTURBATION
        if not np.array_equal(arm["actions"][check], np.clip(expected,-1.,1.).astype(np.float32)):
            raise RuntimeError("actuator intervention amplitude differs")
    return {**case, "sequence":prefix[check-HISTORY+1:check+1], "baseline_action":baseline,
            "actions":np.stack((positive["actions"][-1], negative["actions"][-1])),
            "deltas":np.stack((positive["frames"][-1,:10],negative["frames"][-1,:10]))-prefix[-1,:10],
            "upper_call_steps":expected_calls, "primitive_steps":2*(check+1)}


def transition_samples(paths):
    rows, sequences, actions, labels = [], [], [], []
    for path in paths:
        for step in range(HISTORY, len(path["actions"]), STRIDE):
            rows.append({"seed":path["seed"], "step":step})
            sequences.append(path["frames"][step-HISTORY+1:step+1])
            actions.append(path["actions"][step])
            labels.append(path["frames"][step+1,:10]-path["frames"][step,:10])
    return {"rows":rows, "x":np.stack(sequences), "a":np.stack(actions), "y":np.stack(labels)}


def scales_for(train):
    mean, scale = train["x"].mean(axis=(0,1)), train["x"].std(axis=(0,1))
    return {"feature_mean":mean, "feature_scale":np.where(scale>1e-6,scale,1.),
            "target_scale":np.maximum(np.sqrt(np.mean(train["y"].astype(np.float64)**2,axis=0)),1e-6)}


def view(x, action, *, method, scales):
    sequence = ((x-scales["feature_mean"])/scales["feature_scale"]).astype(np.float32)
    requested = np.asarray(action, dtype=np.float32).copy()
    if method.startswith("current"):
        sequence = np.repeat(sequence[:,-1:],HISTORY,axis=1)
    if method.endswith("blind"):
        sequence[:,:,-2:] = 0.
        requested[:] = 0.
    return sequence, requested


def predict(model, x, action, *, method, scales):
    sequence, requested = view(x,action,method=method,scales=scales)
    means, logvars = [], []
    with torch.inference_mode():
        for start in range(0,len(x),128):
            mu, lv = model(torch.from_numpy(sequence[start:start+128]),torch.from_numpy(requested[start:start+128]))
            means.append(mu.numpy().copy())
            logvars.append(lv.numpy().copy())
    return np.concatenate(means), np.concatenate(logvars)


def forecast(model, histories, action_tapes, *, method, scales):
    sequence = histories.copy()
    predictions = []
    for step in range(action_tapes.shape[1]):
        mu, _ = predict(model,sequence,action_tapes[:,step],method=method,scales=scales)
        following = sequence[:,-1,:10]+mu*scales["target_scale"]
        if step+1 in ROLLOUT_HORIZONS:
            predictions.append(following.copy())
        sequence = np.concatenate((sequence[:,1:],np.concatenate((following,action_tapes[:,step]),axis=1)[:,None]),axis=1)
    return np.stack(predictions,axis=1)


def fit_method(method, train, query, effects, rollouts, *, root, epochs, scales):
    if set(r["seed"] for r in train["rows"]).intersection(r["seed"] for r in query["rows"]):
        raise ValueError("predictive state training and evaluation paths overlap")
    torch.set_num_threads(1)
    seed = int(np.random.SeedSequence([root,29_030]).generate_state(1)[0])
    torch.manual_seed(seed)
    model = ActionConditionedStatePredictor(physical_dim=4,external_dim=6,action_dim=2)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    x, a = view(train["x"],train["a"],method=method,scales=scales)
    x, a = torch.from_numpy(x), torch.from_numpy(a)
    target = torch.tensor(train["y"]/scales["target_scale"],dtype=torch.float32)
    rng, updates = np.random.default_rng(seed), 0
    for epoch in range(epochs):
        for start in np.array_split(rng.permutation(len(x)),(len(x)+127)//128):
            optimizer.zero_grad()
            mu, lv = model(x[start],a[start])
            loss = gaussian_prediction_loss(mu,lv,target[start])
            if not torch.isfinite(loss):
                raise RuntimeError("state prediction likelihood is non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            updates += 1
        if (epoch+1)%16 == 0 or epoch+1 == epochs:
            print(f"state fit {method}: {epoch+1}/{epochs}",flush=True)
    model.eval()
    teacher = predict(model,query["x"],query["a"],method=method,scales=scales)
    effect = predict(model,np.repeat(effects["x"],2,axis=0),effects["a"].reshape(-1,2),method=method,scales=scales)
    rollout = forecast(model,rollouts["x"],rollouts["a"],method=method,scales=scales)
    if not all(np.isfinite(v).all() for v in (*teacher,*effect,rollout)):
        raise RuntimeError("state predictions are non-finite")
    return {"method":method, "teacher":teacher, "effect":effect, "rollout":rollout,
            "weights":model.state_dict(), "fit": {"epochs":epochs, "optimizer_steps":updates,
            "training_rows":len(x), "initialization_seed":seed,
            "parameter_count":sum(p.numel() for p in model.parameters())}}


def teacher_metrics(truth, mean, logvar, scale):
    error = (mean-truth/scale).astype(np.float64)
    z = error*np.exp(-.5*logvar)
    nll = .5*(error**2*np.exp(-logvar)+logvar+np.log(2*np.pi))
    return {"rows":len(truth), "physical_normalized_mse":float(np.mean(error[:,:4]**2)),
            "target_rate_mse":float(np.mean((error[:,4:6]*scale[4:6]/.01)**2)),
            "normalized_gaussian_nll":float(nll.mean()), "mse_by_coordinate":np.mean(error**2,axis=0),
            "coverage_95_by_coordinate":np.mean(np.abs(z)<=1.96,axis=0),
            "innovation_over_3sigma_by_coordinate":np.mean(np.abs(z)>3,axis=0)}


def run_cell(args):
    if args.workers<1 or args.model_epochs<1:
        raise ValueError("state-response workers and epochs must be positive")
    torch.set_num_threads(1)
    cache, source, controller, scale, checkpoint, factual = load_controller(args)
    roles = path_roles(args.optimizer_seed,preflight=args.horizon==300)
    inherited = {**cache, "temporal_seed_roles":{**cache["temporal_seed_roles"],
                 "stage28":[s for paths in hold_roles(args.optimizer_seed,preflight=args.horizon==300).values() for s in paths]}}
    validate_paths(args,roles,inherited)
    print("cached controller matched; collecting excited fit and unperturbed evaluation trajectories",flush=True)
    jobs = [{"seed":s,"role":role} for role,seeds in roles.items() for s in seeds]
    trajectories, effects = [], []
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context("spawn"),
                             initializer=init_worker,initargs=(controller,args,scale)) as pool:
        for future in as_completed([pool.submit(collect_job,job) for job in jobs]):
            trajectories.append(future.result())
        references = {p["seed"]:p for p in trajectories if p["role"]=="evaluation"}
        cases = [{"seed":s,"check_step":check,"axis":axis} for s in roles["evaluation"]
                 for check in checks_for_path(s,preflight=args.horizon==300) for axis in (0,1)]
        futures = {pool.submit(collect_job,{"seed":c["seed"],"role":"evaluation",
                   "check":c["check_step"],"axis":c["axis"],"sign":sign}):(i,sign)
                   for i,c in enumerate(cases) for sign in (1,-1)}
        arms = {}
        for future in as_completed(futures):
            i,sign = futures[future]
            arms[(i,sign)] = future.result()
        effects = [combine_effect(c,arms[(i,1)],arms[(i,-1)],references[c["seed"]]) for i,c in enumerate(cases)]
    trajectories.sort(key=lambda p:p["seed"])
    train, query = (transition_samples([p for p in trajectories if p["role"]==role]) for role in ("fit","evaluation"))
    scales = scales_for(train)
    interventions = {"x":np.stack([e["sequence"] for e in effects]), "a":np.stack([e["actions"] for e in effects]),
                     "y":np.stack([e["deltas"] for e in effects])}
    rollout_rows = [{"seed":s,"check_step":c} for s in roles["evaluation"]
                    for c in checks_for_path(s,preflight=args.horizon==300)]
    rollouts = {"x":np.stack([references[r["seed"]]["frames"][r["check_step"]-63:r["check_step"]+1] for r in rollout_rows]),
                "a":np.stack([references[r["seed"]]["actions"][r["check_step"]:r["check_step"]+10] for r in rollout_rows]),
                "y":np.stack([references[r["seed"]]["frames"][r["check_step"]+np.array(ROLLOUT_HORIZONS),:10] for r in rollout_rows])}
    print(f"state data ready: {len(train['rows'])} fit/{len(query['rows'])} evaluation transitions, {len(effects)} effect pairs",flush=True)
    with ProcessPoolExecutor(max_workers=min(args.workers,4),mp_context=mp.get_context("spawn")) as pool:
        fits = [f.result() for f in [pool.submit(fit_method,m,train,query,interventions,rollouts,
                root=args.optimizer_seed,epochs=args.model_epochs,scales=scales) for m in METHODS]]
    by_method = {f["method"]:f for f in fits}
    metrics = {m:teacher_metrics(query["y"],*by_method[m]["teacher"],scales["target_scale"]) for m in METHODS}
    true_effect = (interventions["y"][:,0,:4]-interventions["y"][:,1,:4])/scales["target_scale"][:4]
    effect_predictions = {m:(by_method[m]["effect"][0].reshape(-1,2,10)[:,0,:4]
                            -by_method[m]["effect"][0].reshape(-1,2,10)[:,1,:4]) for m in METHODS}
    effect_mse = {m:float(np.mean((v-true_effect)**2)) for m,v in effect_predictions.items()}
    effect_mse["zero"] = float(np.mean(true_effect**2))
    action_gate = all(metrics[p+"_action"]["physical_normalized_mse"]<metrics[p+"_blind"]["physical_normalized_mse"]
                      and effect_mse[p+"_action"]<min(effect_mse[p+"_blind"],effect_mse["zero"]) for p in ("history","current"))
    history_gate = metrics["history_action"]["target_rate_mse"]<metrics["current_action"]["target_rate_mse"]
    raw_dir = args.output.resolve().parent.with_name(args.output.parent.name+"_raw")
    raw_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(raw_dir/"state_response.npz",train_x=train["x"],train_a=train["a"],train_y=train["y"],
        query_x=query["x"],query_a=query["a"],query_y=query["y"],query_seeds=[r["seed"] for r in query["rows"]],
        query_steps=[r["step"] for r in query["rows"]], effect_x=interventions["x"],effect_a=interventions["a"],effect_y=interventions["y"],
        rollout_x=rollouts["x"],rollout_a=rollouts["a"],rollout_y=rollouts["y"],
        **{m+"_"+key:value for m,f in by_method.items() for key,value in
           (("teacher_mean",f["teacher"][0]),("teacher_logvar",f["teacher"][1]),
            ("effect_mean",f["effect"][0]),("effect_logvar",f["effect"][1]),("rollout",f["rollout"]))})
    torch.save({"models":{m:f["weights"] for m,f in by_method.items()}, "scales":scales},raw_dir/"state_models.pt")
    score_rows = [{**{k:e[k] for k in ("seed","check_step","axis","baseline_action","actions","upper_call_steps","primitive_steps")},
                   "true_normalized_effect":true_effect[i],
                   "predicted_normalized_effect":{m:p[i] for m,p in effect_predictions.items()}} for i,e in enumerate(effects)]
    path_metrics = {str(s):{m:teacher_metrics(query["y"][mask],by_method[m]["teacher"][0][mask],
                   by_method[m]["teacher"][1][mask],scales["target_scale"]) for m in METHODS}
                   for s in roles["evaluation"] for mask in [np.array([r["seed"]==s for r in query["rows"]])]}
    audit_indices = [next(i for i,r in enumerate(query["rows"]) if r["seed"]==s) for s in roles["evaluation"]]
    audit_rows = [{**query["rows"][i],"delta":query["y"][i],"predictions":{
                  m:{"mean":by_method[m]["teacher"][0][i],"logvar":by_method[m]["teacher"][1][i]} for m in METHODS}}
                  for i in audit_indices]
    return {"optimizer_seed":args.optimizer_seed,"state_seed_roles":roles,
            "controller_selected_iteration":source["controller_selected_iteration"],"controller_checkpoint":str(checkpoint),
            "factual_replay":factual,"controller_updates":0,"controller_reconstruction_primitive_steps":0,
            "trajectory_primitive_steps":sum(p["primitive_steps"] for p in trajectories),
            "intervention_primitive_steps":sum(e["primitive_steps"] for e in effects),"factual_replay_primitive_steps":args.horizon,
            "fit_transitions":len(train["rows"]),"evaluation_transitions":len(query["rows"]),"effect_pairs":len(effects),
            "fits":{m:f["fit"] for m,f in by_method.items()},"scales":scales,"teacher_metrics":metrics,
            "effect_normalized_mse":effect_mse,"effect_rows":score_rows,"audit_rows":audit_rows,"path_metrics":path_metrics,
            "known_action_rollout_normalized_mse_by_horizon":{
                m:np.mean(((f["rollout"]-rollouts["y"])/scales["target_scale"])**2,axis=(0,2)) for m,f in by_method.items()},
            "action_response_gate_passed":action_gate,"history_prediction_gate_passed":history_gate,
            "development_gate_passed":action_gate and history_gate,"raw_server_directory":str(raw_dir),
            "raw_server_bytes":{p.name:p.stat().st_size for p in raw_dir.iterdir() if p.is_file()}}


def main(argv=None):
    parser = build_parser()
    parser.add_argument("--source-result",type=Path,required=True)
    parser.add_argument("--controller-result",type=Path,required=True)
    parser.add_argument("--model-epochs",type=int,default=64)
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args(argv)
    output = {"status":"dry_run" if args.dry_run else "complete","protocol":{
        "protocol_version":PROTOCOL_VERSION,"optimizer_seed":args.optimizer_seed,"methods":METHODS,
        "source_result":str(args.source_result),"controller_result":str(args.controller_result),
        "history_steps":HISTORY,"sample_stride_steps":STRIDE,"perturbation":PERTURBATION,
        "gru_width":64,"latent_dim":16,"model_epochs":args.model_epochs,"workers":args.workers,
        "seed_roles":path_roles(args.optimizer_seed,preflight=args.horizon==300),
        "known_action_rollout_horizons":ROLLOUT_HORIZONS,"policy_deployment":False,
        "evidence_role":"fresh_path_action_conditioned_predictive_state_development_only"},
        "cells":[] if args.dry_run else [run_cell(args)]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(output),indent=2,sort_keys=True)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
