from concurrent.futures import Future
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.core.causal_motion import CausalMotionForecaster
from freq_hrl.core.plan_response import PlanResponseCritic,plan_response_features
from freq_hrl.domains.mujoco import RelativeSubgoalAdapter
from freq_hrl.experiments import pointmaze_forecast_response as response
from freq_hrl.experiments import pointmaze_plan_hold as hold
from freq_hrl.experiments import pointmaze_state_response as state
from freq_hrl.experiments import pointmaze_separate_motion as motion
from freq_hrl.experiments import pointmaze_temporal_plan as temporal
from freq_hrl.experiments.pointmaze_plan_validity_branching import _ridge_fit_predict
from scripts import pointmaze_forecast_response_spec as spec
from scripts.submit_pointmaze_deployed_pair_diagnostic_scheduleurm import task_specification
from test_pointmaze_plan_hold import arguments
from test_pointmaze_history_information import NAMES,measured_sequences
from test_pointmaze_temporal_plan import FakeController,FakeTask


def motion_models():
    rng = np.random.default_rng(31)
    x,y = rng.normal(size=(16,64,6)),rng.normal(size=(16,5,6))
    return {m:CausalMotionForecaster(observed_dim=6,velocity_channels=(0,1),
             horizon_steps=motion.HORIZONS,dt_seconds=.01).fit(x,y) for m in response.METHODS[:3]}


class ImmediatePool:
    def __init__(self, *, initializer, initargs, **kwargs):
        initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self,*args):
        pass

    def submit(self,function,*args):
        future = Future()
        future.set_result(function(*args))
        return future


def cache_fixture(directory):
    args = arguments()
    args.workers,args.pairs_per_path = 1,1
    args.output = directory/"cell"/"result.json"
    args.hold_result,args.motion_result = directory/"hold.json",directory/"motion.json"
    roles = hold.path_roles(208001,preflight=True)
    keys = sorted((c["seed"],c["check_step"],role) for role,paths in roles.items()
                  for c in hold.cases_for_paths(208001,paths,horizon=300,pairs_per_path=2))
    x = measured_sequences([{"seed":s,"check_step":t} for s,t,_ in keys])
    rng = np.random.default_rng(31)
    costs = rng.uniform(.001,.003,size=(len(keys),2,150))
    curves = np.cumsum(costs[:,1]-costs[:,0],axis=1)[:,np.array(hold.HORIZONS)-1]
    raw = directory/"hold_raw"
    raw.mkdir()
    arrays = {"sequences":x,"curves":curves,"step_ise":costs,"seeds":[s for s,_,_ in keys],
              "check_steps":[t for _,t,_ in keys],"roles":[r for _,_,r in keys],"feature_names":NAMES}
    np.savez_compressed(raw/"plan_hold_pairs.npz",**arrays)
    args.hold_result.write_text(json.dumps({"status":"complete","protocol":{
        "protocol_version":hold.PROTOCOL_VERSION,"optimizer_seed":208001,"pairs_per_path":2},
        "cells":[{"controller_selected_iteration":2,"plan_hold_seed_roles":roles,
                  "feature_names":NAMES,"raw_server_directory":str(raw),"training_pairs":4}]}))
    original = directory/"original.json"
    original.write_text(json.dumps({"cells":[{"controller_selected_iteration":2}]}))
    models = motion_models()
    source = {"status":"complete","protocol":{"protocol_version":motion.PROTOCOL_VERSION,
              "optimizer_seed":208001,"forecast_horizons_steps":motion.HORIZONS,"controller_result":str(original)},
              "cells":[{"development_gate_passed":False,"fits":{m:model.fitted for m,model in models.items()},
                        "fit_paths":state.path_roles(208001,preflight=True)["fit"],
                        "evaluation_paths":motion.evaluation_paths(208001,preflight=True)}]}
    args.motion_result.write_text(json.dumps(response._json_ready(source)))
    return args,arrays,models


class ForecastResponseTest(unittest.TestCase):
    def test_generic_plan_geometry_and_translation_invariance(self):
        kwargs = {"current_state":np.array([[10.,20.,30.]]),"position":np.array([[0.,0.]]),
                  "velocity":np.array([[1.,2.]]),"retained_plan":np.array([[1.,0.]]),
                  "candidate_plan":np.array([[0.,1.]]),"observed_target":np.array([[2.,0.]]),
                  "forecast_targets":np.array([[[3.,0.],[2.,1.]]])}
        x = plan_response_features(**kwargs)
        np.testing.assert_equal(x[0],np.array([10,20,30,0,1,-1,1,1,5,1,0,0,1,3,1,5,-1,1,2]))
        shifted = deepcopy(kwargs)
        for key in ("position","retained_plan","candidate_plan","observed_target","forecast_targets"):
            shifted[key] += np.array([100.,-7.])
        np.testing.assert_equal(plan_response_features(**shifted),x)

    def test_generic_critic_matches_existing_ridge_and_rate_units(self):
        rng = np.random.default_rng(31)
        x,q,y = rng.normal(size=(20,7)),rng.normal(size=(3,7)),rng.normal(size=(20,5))
        durations = np.array(hold.HORIZONS)*.01
        critic = PlanResponseCritic(durations_seconds=durations).fit(x,y)
        predicted = critic.predict_rates(q)
        for i,duration in enumerate(durations):
            expected,_ = _ridge_fit_predict(x,y[:,i]/duration,q,alpha=1.)
            np.testing.assert_allclose(predicted[:,i],expected,rtol=1e-12,atol=1e-12)
        self.assertEqual(critic.fitted["parameter_count"],40)

    def test_causal_candidate_state_matches_actual_renew_upper_input(self):
        class RecordingController(FakeController):
            def __init__(self):
                self.inputs = []
            def plan_goal(self,state,sample):
                self.inputs.append(state.copy())
                return super().plan_goal(state,sample)
        args,controller = arguments(),RecordingController()
        scale = PhysicalTimeScaleContract(dt_seconds=.01,upper_period_seconds=.5,history_seconds=.64,fast_period_seconds=.04)
        bounds = (np.full(2,-2.),np.full(2,2.))
        schedule = hold.pair_schedules(3309101,105,300)[0]
        with patch.object(hold,"_make_task",return_value=FakeTask()),patch.object(hold,"pointmaze_goal_bounds",return_value=bounds):
            collected = hold.rollout_window(controller,seed=3309101,check=105,schedule=schedule,args=args,time_scale=scale)
        actual_input = controller.inputs[-1]
        adapter = RelativeSubgoalAdapter(maximum_delta=np.full(2,.75),world_low=bounds[0],world_high=bounds[1])
        row = {"sequence":collected["sequence"],"feature_names":collected["feature_names"]}
        candidate,encoded = response.proposal(row,controller,adapter,history_steps=64)
        np.testing.assert_equal(encoded,actual_input)
        achieved = row["sequence"][-1,4:6]
        np.testing.assert_allclose(candidate,achieved+.75*np.tanh([.2,-.1]),atol=1e-7)

    def test_forecast_modes_match_capacity_and_query_labels_do_not_change_fits(self):
        models = motion_models()
        rows = [{"seed":i+1,"check_step":100} for i in range(12)]
        sequences = measured_sequences(rows)
        rng = np.random.default_rng(31)
        examples = [{**r,"sequence":x,"feature_names":NAMES,"candidate_plan":rng.normal(size=2),
                     "curve":rng.normal(size=5)} for r,x in zip(rows,sequences)]
        before = {m:deepcopy(model.fitted) for m,model in models.items()}
        pred,fits,_ = response.fit_response(examples[:8],examples[8:],models,root=31)
        query = deepcopy(examples[8:])
        for row in query:
            row["curve"] *= 1000
        changed,other,_ = response.fit_response(examples[:8],query,models,root=31)
        for method in response.METHODS:
            np.testing.assert_equal(fits[method]["weights"],other[method]["weights"])
            np.testing.assert_equal(pred[method],changed[method])
            self.assertEqual(fits[method]["parameter_count"],160 if method.startswith("raw_") else 250)
        for method,model in models.items():
            np.testing.assert_equal(model.fitted["weights"],before[method]["weights"])
        with self.assertRaisesRegex(ValueError,"paths overlap"):
            response.fit_response(examples[:8],examples[:1],models,root=31)

    def test_cache_filter_and_raw_cost_labels_are_verified(self):
        with TemporaryDirectory() as temporary:
            args,arrays,_ = cache_fixture(Path(temporary))
            train,cell,path = response.load_training(args,selected_iteration=2)
            self.assertEqual(len(train),2)
            self.assertTrue(all(r["check_step"]>=64 for r in train))
            models,_ = response.load_motion(args,selected_iteration=2)
            self.assertEqual(set(models),set(response.METHODS[:3]))
            arrays["curves"][0,0] += 1
            np.savez_compressed(path,**arrays)
            with self.assertRaisesRegex(ValueError,"curve labels"):
                response.load_training(args,selected_iteration=2)

    def test_fresh_roster_full_prefixes_equal_calls_and_accounting(self):
        all_paths = set()
        for root in (208001,209011,209061):
            args = arguments(root)
            preflight = root==208001
            paths = response.evaluation_paths(root,preflight=preflight)
            inherited = {"temporal_seed_roles":{str(i):seeds for i,seeds in enumerate(
                [s for module in (hold,state,temporal) for s in module.path_roles(root,preflight=preflight).values()]
                +[motion.evaluation_paths(root,preflight=preflight)])}}
            hold.validate_paths(args,{"fit":[],"evaluation":paths},inherited)
            self.assertFalse(all_paths.intersection(paths))
            all_paths.update(paths)
            cases = response.query_cases(root,paths,horizon=args.horizon,pairs_per_path=1 if preflight else 15)
            self.assertEqual(len(cases),2 if preflight else 120)
            for seed in paths:
                checks = [c["check_step"] for c in cases if c["seed"]==seed]
                self.assertTrue(all(c>=64 and c+150<=args.horizon for c in checks))
                self.assertEqual(len(checks),len(set(checks)))
                if not preflight:
                    self.assertEqual([sum(c%50==o for c in checks) for o in (0,5,10,15,20)],[3]*5)
                for check in checks:
                    renew,keep = hold.pair_schedules(seed,check,args.horizon)
                    self.assertEqual(renew[:-1],keep[:-1])
                    self.assertEqual((renew[-1],keep[-1]),(check,check+100))

    def test_hand_computed_decision_gates_and_path_bootstrap(self):
        rows = [{"curve":np.array(hold.HORIZONS)*.01*v,"predicted_rates":{
                 m:np.full(5,v if m=="history" else 0.) for m in response.METHODS}} for v in (1.,-1.)]
        metrics = response.summarize(rows)
        self.assertTrue(metrics["prediction_gate_passed"])
        self.assertTrue(metrics["decision_gate_passed"])
        self.assertEqual(metrics["settled_ise_benefit_vs_control"]["always_keep"],.75)
        intervals = response.path_intervals({"a":metrics,"b":metrics},root=31)
        self.assertEqual(intervals["always_keep"]["ci95"],[.75,.75])
        rows[0]["predicted_rates"]["current_repeat"][-1] = 1.
        self.assertFalse(response.summarize(rows)["decision_gate_passed"])

    def test_run_cell_reuses_weights_and_retains_lower_feedback(self):
        with TemporaryDirectory() as temporary:
            args,_,_ = cache_fixture(Path(temporary))
            controller = FakeController()
            controller.config = SimpleNamespace(state_encoder="mlp")
            scale = PhysicalTimeScaleContract(dt_seconds=.01,upper_period_seconds=.5,history_seconds=.64,fast_period_seconds=.04)
            cache = {"temporal_seed_roles":temporal.path_roles(208001,preflight=True)}
            with patch.object(hold,"load_controller",return_value=(cache,{"controller_selected_iteration":2},controller,
                              scale,Path(temporary)/"controller.pt",{"absolute_errors":{"episode_return":0.}})), \
                    patch.object(response,"ProcessPoolExecutor",ImmediatePool), \
                    patch.object(hold,"_make_task",side_effect=lambda **kwargs:FakeTask()), \
                    patch.object(response,"_make_task",side_effect=lambda **kwargs:FakeTask()), \
                    patch.object(hold,"pointmaze_goal_bounds",return_value=(np.full(2,-2.),np.full(2,2.))), \
                    patch.object(response,"pointmaze_goal_bounds",return_value=(np.full(2,-2.),np.full(2,2.))):
                cell = response.run_cell(args)
            self.assertEqual((cell["training_pairs"],cell["evaluation_pairs"]),(2,2))
            self.assertEqual((cell["critic_fits"],cell["multi_rhs_linear_solves"],cell["scalar_rhs_count"]),(7,7,35))
            self.assertEqual(cell["fresh_pair_primitive_steps"],1030)
            self.assertEqual(cell["candidate_proposal_inference_calls"],4)
            for key in ("controller_updates","motion_updates","physical_model_updates","controller_reconstruction_primitive_steps"):
                self.assertEqual(cell[key],0)
            for row in cell["rows"]:
                self.assertEqual(row["upper_calls_at_horizons"]["renew"][-1],row["upper_calls_at_horizons"]["keep"][-1])

    def test_scheduler_dynamic_pool_and_no_raw_staging(self):
        for preflight in (True,False):
            task = task_specification("unit_forecast_response",spec.roots(preflight=preflight)[0],
                                      preflight=preflight,protocol_spec=spec)
            self.assertEqual((task["cpu"],task["ram_mb"]),(2,3072) if preflight else (17,24576))
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"],[f"node00{i}" for i in range(1,7)])
            self.assertFalse(any("_raw" in path for path in task["stage_input_paths"]))


if __name__ == "__main__":
    unittest.main()
