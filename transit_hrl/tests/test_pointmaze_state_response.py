from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.experiments import pointmaze_state_response as response
from freq_hrl.experiments import pointmaze_temporal_plan as temporal
from freq_hrl.experiments import pointmaze_plan_hold as hold
from freq_hrl.rl.predictive_state import ActionConditionedStatePredictor, gaussian_prediction_loss
from scripts import pointmaze_state_response_spec as spec
from scripts.submit_pointmaze_deployed_pair_diagnostic_scheduleurm import task_specification
from test_pointmaze_plan_hold import arguments
from test_pointmaze_temporal_plan import FakeController, FakeTask


class KinematicModel(torch.nn.Module):
    def forward(self, history, action):
        mean = history.new_zeros((len(history),10))
        mean[:,:2] = action*.01
        mean[:,2:4] = action-history[:,-1,2:4]
        return mean, torch.zeros_like(mean)


class StateResponseTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_generic_predictor_shapes_and_current_action_cannot_change_external_prediction(self):
        torch.manual_seed(29)
        model = ActionConditionedStatePredictor(physical_dim=4,external_dim=6,action_dim=2)
        history = torch.randn(3,64,12)
        mu, lv = model(history,torch.zeros(3,2))
        other_mu, other_lv = model(history,torch.ones(3,2))
        self.assertEqual(tuple(mu.shape),(3,10))
        self.assertEqual(tuple(lv.shape),(3,10))
        torch.testing.assert_close(mu[:,4:],other_mu[:,4:],rtol=0,atol=0)
        torch.testing.assert_close(lv[:,4:],other_lv[:,4:],rtol=0,atol=0)
        self.assertFalse(torch.equal(mu[:,:4],other_mu[:,:4]))
        self.assertTrue(torch.all((lv>=-8)&(lv<=4)))
        self.assertGreater(gaussian_prediction_loss(mu,torch.full_like(lv,2.),mu).item(),
                           gaussian_prediction_loss(mu,torch.zeros_like(lv),mu).item())

    def test_fresh_paths_exact_costs_and_existing_seed_exclusion(self):
        all_new = set()
        for root in (208001,209011,209061):
            args = arguments(root)
            preflight = root==208001
            roles = response.path_roles(root,preflight=preflight)
            all_paths = roles["fit"]+roles["evaluation"]
            self.assertFalse(all_new.intersection(all_paths))
            all_new.update(all_paths)
            inherited = {"temporal_seed_roles":{**temporal.path_roles(root,preflight=preflight),
                         "stage28":[s for paths in hold.path_roles(root,preflight=preflight).values() for s in paths]}}
            hold.validate_paths(args,roles,inherited)
            checks = {s:response.checks_for_path(s,preflight=preflight) for s in roles["evaluation"]}
            costs = len(all_paths)*args.horizon+sum(4*(c+1) for values in checks.values() for c in values)+args.horizon
            self.assertEqual(costs,2368 if preflight else 209520)
            sample_count = len(range(64,args.horizon,5))
            self.assertEqual((len(roles["fit"])*sample_count,len(roles["evaluation"])*sample_count),
                             (96,96) if preflight else (3648,1824))
            for values in checks.values():
                self.assertTrue(all(c>=64 and c+10<=args.horizon for c in values))
                if not preflight:
                    self.assertEqual([sum(c%50==o for c in values) for o in (0,5,10,15,20)],[2]*5)

    def collect(self, *, role="evaluation", check=None, axis=None, sign=None, shift=0.):
        task = FakeTask(shift)
        args = arguments()
        scale = PhysicalTimeScaleContract(dt_seconds=.01,upper_period_seconds=.5,
                                          history_seconds=.64,fast_period_seconds=.04)
        with patch.object(response,"_make_task",return_value=task), \
                patch.object(response,"pointmaze_goal_bounds",return_value=(np.full(2,-2.),np.full(2,2.))):
            trajectory = response.collect(FakeController(),seed=3289101,role=role,args=args,scale=scale,
                                           check=check,axis=axis,sign=sign)
        self.assertTrue(task.closed)
        return trajectory

    def test_actual_actions_align_with_transition_labels_and_excitation_keeps_lower_feedback(self):
        reference, excited = self.collect(), self.collect(role="fit")
        np.testing.assert_equal(reference["frames"][1:,-2:],reference["actions"])
        self.assertEqual(reference["frames"].shape,(301,12))
        self.assertEqual(reference["calls"],excited["calls"])
        self.assertFalse(np.array_equal(reference["actions"],excited["actions"]))
        self.assertTrue(np.all(np.abs(excited["actions"]-excited["baseline_actions"])<=.250001))
        self.assertFalse(np.array_equal(excited["baseline_actions"],reference["baseline_actions"]))
        samples = response.transition_samples([reference])
        self.assertEqual(samples["x"].shape,(48,64,12))
        for i,row in enumerate(samples["rows"]):
            t = row["step"]
            np.testing.assert_equal(samples["x"][i],reference["frames"][t-63:t+1])
            np.testing.assert_equal(samples["a"][i],reference["actions"][t])
            np.testing.assert_equal(samples["y"][i],reference["frames"][t+1,:10]-reference["frames"][t,:10])

    def test_signed_action_interventions_preserve_prefix_and_exogenous_truth(self):
        reference = self.collect()
        plus, minus = (self.collect(check=105,axis=0,sign=sign) for sign in (1,-1))
        effect = response.combine_effect({"seed":3289101,"check_step":105,"axis":0},plus,minus,reference)
        self.assertEqual(effect["primitive_steps"],212)
        self.assertEqual(effect["sequence"].shape,(64,12))
        np.testing.assert_allclose(effect["actions"][0]-effect["actions"][1],[.5,0.],atol=1e-7)
        np.testing.assert_allclose((effect["deltas"][0]-effect["deltas"][1])[:4],[.005,0.,.5,0.],atol=1e-6)
        np.testing.assert_equal(effect["deltas"][0,4:],effect["deltas"][1,4:])
        changed = deepcopy(minus)
        changed["frames"][100,0] += 1
        with self.assertRaisesRegex(RuntimeError,"factual prefix"):
            response.combine_effect({"check_step":105,"axis":0},plus,changed,reference)
        changed = deepcopy(minus)
        changed["frames"][-1,4] += 1
        with self.assertRaisesRegex(RuntimeError,"exogenous stream"):
            response.combine_effect({"check_step":105,"axis":0},plus,changed,reference)

    def test_factorial_views_and_known_action_rollout_use_no_future_observations(self):
        rng = np.random.default_rng(29)
        x, a = rng.normal(size=(2,64,12)).astype(np.float32), np.ones((2,2),dtype=np.float32)
        scales = {"feature_mean":np.zeros(12),"feature_scale":np.ones(12),"target_scale":np.ones(10)}
        current, known = response.view(x,a,method="current_action",scales=scales)
        np.testing.assert_equal(current,np.repeat(x[:,-1:],64,axis=1))
        np.testing.assert_equal(known,a)
        blind, unknown = response.view(x,a,method="history_blind",scales=scales)
        np.testing.assert_equal(blind[:,:,:10],x[:,:,:10])
        np.testing.assert_equal(blind[:,:,-2:],0)
        np.testing.assert_equal(unknown,0)
        tape = np.repeat(a[:,None],10,axis=1)
        predicted = response.forecast(KinematicModel(),x,tape,method="history_action",scales=scales)
        for index,h in enumerate((1,5,10)):
            np.testing.assert_allclose(predicted[:,index,:2],x[:,-1,:2]+h*.01,atol=1e-6)
            np.testing.assert_allclose(predicted[:,index,2:4],a,atol=1e-6)
            np.testing.assert_equal(predicted[:,index,4:],x[:,-1,4:10])

    def test_query_labels_do_not_change_weights_and_all_methods_have_equal_capacity(self):
        rng = np.random.default_rng(29)
        train = {"x":rng.normal(size=(8,64,12)).astype(np.float32),"a":rng.normal(size=(8,2)).astype(np.float32),
                 "y":rng.normal(size=(8,10)).astype(np.float32),"rows":[{"seed":1}]*8}
        query = {"x":train["x"][:2].copy(),"a":train["a"][:2].copy(),"y":train["y"][:2].copy(),"rows":[{"seed":2}]*2}
        effects = {"x":train["x"][:1].copy(),"a":train["a"][:2].reshape(1,2,2),"y":train["y"][:2].reshape(1,2,10)}
        rollouts = {"x":train["x"][:1].copy(),"a":rng.normal(size=(1,10,2)).astype(np.float32),"y":np.zeros((1,3,10))}
        scales = response.scales_for(train)
        results = [response.fit_method(m,train,query,effects,rollouts,root=29,epochs=1,scales=scales) for m in response.METHODS]
        self.assertEqual(len({r["fit"]["parameter_count"] for r in results}),1)
        changed = deepcopy(query)
        changed["y"] += 1000
        other = response.fit_method("history_action",train,changed,effects,rollouts,root=29,epochs=1,scales=scales)
        for key in results[0]["weights"]:
            torch.testing.assert_close(results[0]["weights"][key],other["weights"][key],rtol=0,atol=0)
        np.testing.assert_equal(results[0]["teacher"][0],other["teacher"][0])
        for index in (2,3):
            np.testing.assert_allclose(results[index]["effect"][0][0],results[index]["effect"][0][1],rtol=0,atol=1e-7)
        with self.assertRaisesRegex(ValueError,"paths overlap"):
            response.fit_method("history_action",train,train,effects,rollouts,root=29,epochs=1,scales=scales)

    def test_likelihood_mse_and_innovation_units(self):
        truth = np.ones((2,10))
        metrics = response.teacher_metrics(truth,np.zeros_like(truth),np.zeros_like(truth),np.ones(10))
        self.assertEqual(metrics["physical_normalized_mse"],1.)
        self.assertEqual(metrics["target_rate_mse"],10000.)
        self.assertAlmostEqual(metrics["normalized_gaussian_nll"],.5*(1+np.log(2*np.pi)))
        np.testing.assert_equal(metrics["coverage_95_by_coordinate"],1.)
        np.testing.assert_equal(metrics["innovation_over_3sigma_by_coordinate"],0.)

    def test_scheduler_uses_five_cpu_dynamic_pool_and_no_raw_input_staging(self):
        for preflight in (True,False):
            task = task_specification("unit_state",spec.roots(preflight=preflight)[0],preflight=preflight,protocol_spec=spec)
            self.assertEqual((task["cpu"],task["ram_mb"]),(2,3072) if preflight else (5,8192))
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"],[f"node00{i}" for i in range(1,7)])
            self.assertFalse(any("_raw" in p for p in task["stage_input_paths"]))


if __name__ == "__main__":
    unittest.main()
