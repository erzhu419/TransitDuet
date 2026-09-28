#!/usr/bin/env python3
"""Recompute Stage-33 causal prefixes, fits, decisions and costs on the server."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("result", type=Path)
RESULT_PATH = parser.parse_args().result

import json
import numpy as np
import torch
from freq_hrl.core.causal_motion import CausalMotionForecaster
from freq_hrl.domains.mujoco import RelativeSubgoalAdapter
from freq_hrl.experiments import pointmaze_root_response as run
from freq_hrl.experiments import pointmaze_forecast_response as response
from freq_hrl.experiments import pointmaze_plan_hold as hold
from freq_hrl.experiments import pointmaze_separate_motion as motion
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from scripts import pointmaze_root_response_stage33_spec as spec

torch.set_num_threads(1)
result_path = Path(RESULT_PATH)
result = json.loads(result_path.read_text())
root = result["protocol"]["optimizer_seed"]
preflight = root in spec.PREFLIGHT_ROOTS
args = spec.arguments(root, preflight=preflight)
cell = result["cells"][0]
raw = Path(cell["raw_server_directory"])
controller, trained, factual = run.load_controller(args, Path(cell["controller_result"]))
assert [row["iteration"] for row in trained["history"]] == [-1, *range(args.iterations)]
assert trained["budget"] == spec.budget(root, preflight=preflight)
roles = spec.seed_roles(root, preflight=preflight)
assert cell["seed_roles"] == roles
motion_arrays = dict(np.load(raw/"motion.npz", allow_pickle=False))
motion_fits = json.loads((raw/"motion_fits.json").read_text())
motion_rows = {role:motion.sample_rows(roles[role], horizon=args.horizon) for role in ("motion_fit","motion_eval")}
tapes = motion.tapes_for(roles["motion_fit"]+roles["motion_eval"]+roles["response_fit"]+roles["response_eval"],
                        protocol={"horizon":args.horizon,"task_options":_task_options(args)})
for role, prefix in (("motion_fit","train"),("motion_eval","query")):
    x,y = motion.labeled_motion(tapes,motion_rows[role])
    np.testing.assert_array_equal(x,motion_arrays[prefix+"_x"])
    np.testing.assert_array_equal(y,motion_arrays[prefix+"_y"])
    assert list(zip(motion_arrays[prefix+"_seeds"],motion_arrays[prefix+"_steps"])) == [
        (r["seed"],r["step"]) for r in motion_rows[role]]

def ridge(x,y,q,saved):
    mean,scale = x.mean(0),x.std(0)
    scale = np.where(scale>1e-8,scale,1.)
    z = np.column_stack((np.ones(len(x)),(x-mean)/scale))
    penalty = np.eye(z.shape[1]); penalty[0,0]=0
    weights = np.linalg.solve(z.T@z+penalty,z.T@y)
    for name,v in (("feature_mean",mean),("feature_scale",scale),("weights",weights)):
        np.testing.assert_allclose(v,saved[name],rtol=1e-11,atol=1e-11)
    return np.column_stack((np.ones(len(q)),(q-mean)/scale))@weights

models={}
for method in motion.METHODS:
    model=CausalMotionForecaster(observed_dim=6,velocity_channels=(0,1),horizon_steps=motion.HORIZONS,dt_seconds=.01)
    x,q=(model.features(motion.observed_view(motion_arrays[prefix+"_x"],motion_rows[role],method=method,root=root))
         for prefix,role in (("train","motion_fit"),("query","motion_eval")))
    predicted=ridge(x,motion_arrays["train_y"].reshape(len(x),-1),q,motion_fits[method]).reshape(len(q),5,6)
    np.testing.assert_allclose(predicted,motion_arrays[method+"_prediction"],rtol=1e-11,atol=1e-11)
    model.fitted={k:np.asarray(v) if k in ("feature_mean","feature_scale","weights") else v for k,v in motion_fits[method].items()}
    models[method]=model
arrays = dict(np.load(raw/"response.npz",allow_pickle=False))
names = arrays["feature_names"].tolist()
fits = json.loads((raw/"response_fits.json").read_text())
adapter=RelativeSubgoalAdapter(maximum_delta=np.full(2,args.maximum_subgoal_delta,dtype=np.float32),
                              world_low=np.asarray(trained["world_low"]),world_high=np.asarray(trained["world_high"]))
rows={}
fit_cases,query_cases,excluded=spec.response_cases(root,preflight=preflight)
for prefix,cases in (("train",fit_cases),("query",query_cases)):
    assert list(zip(arrays[prefix+"_seed"],arrays[prefix+"_check_step"])) == sorted(
        (c["seed"],c["check_step"]) for c in cases)
    curves=np.cumsum(arrays[prefix+"_step_ise"][:,1]-arrays[prefix+"_step_ise"][:,0],axis=1)[:,np.array(hold.HORIZONS)-1]
    np.testing.assert_allclose(curves,arrays[prefix+"_curve"],rtol=0,atol=1e-12)
    rows[prefix]=[]
    for i,(seed,check) in enumerate(zip(arrays[prefix+"_seed"],arrays[prefix+"_check_step"])):
        row={"seed":int(seed),"check_step":int(check),"sequence":arrays[prefix+"_sequence"][i],
             "feature_names":names,"curve":curves[i],"candidate_plan":arrays[prefix+"_candidate_plan"][i]}
        measured=[names.index("measured_"+str(j)) for j in range(6)]
        np.testing.assert_array_equal(row["sequence"][:,measured],tapes[int(seed)][int(check)-63:int(check)+1])
        np.testing.assert_array_equal(row["sequence"][:,-1],np.ones(64))
        proposal,state=response.proposal(row,controller,adapter,history_steps=64)
        np.testing.assert_array_equal(proposal,row["candidate_plan"])
        np.testing.assert_array_equal(state,arrays[prefix+"_proposal_state"][i])
        rows[prefix].append(row)
for method in response.METHODS:
    x,q=(response.design(rows[prefix],models,method=method,root=root) for prefix in ("train","query"))
    np.testing.assert_array_equal(x,arrays[method+"_train_design"])
    np.testing.assert_array_equal(q,arrays[method+"_query_design"])
    pred=ridge(x,arrays["train_curve"]/(np.array(hold.HORIZONS)*.01),q,fits[method])
    np.testing.assert_allclose(pred,arrays[method+"_prediction"],rtol=1e-11,atol=1e-11)
    for i,row in enumerate(cell["rows"]):
        np.testing.assert_allclose(pred[i],row["predicted_rates"][method],rtol=1e-11,atol=1e-11)
for row in cell["rows"]:
    schedules=hold.pair_schedules(row["seed"],row["check_step"],args.horizon)
    assert row["upper_call_steps"] == {name:list(s) for name,s in zip(("renew","keep"),schedules)}
    assert row["upper_calls_at_horizons"]["renew"][-1] == row["upper_calls_at_horizons"]["keep"][-1]
scores=response._json_ready(response.summarize(cell["rows"]))
assert scores == cell["metrics"]
budget=spec.budget(root,preflight=preflight)
assert cell["budget"] == budget
assert cell["excluded_incomplete_fit_prefixes"] == excluded
audit={"status":"pass","optimizer_seed":root,"controller_selected_iteration":trained["selected_checkpoint_iteration"],
       "factual_replay":factual,"checked_fit_pairs":len(fit_cases),"checked_query_pairs":len(query_cases),
       "method_budget":budget,"verification_primitive_steps":args.horizon,
       "verification_proposal_calls":len(fit_cases)+len(query_cases),
       "verification_linear_solves":10,"verification_scalar_rhs":125,
       "scientific_prediction_gate":scores["prediction_gate_passed"],"scientific_decision_gate":scores["decision_gate_passed"]}
print(json.dumps(audit,sort_keys=True))
