import copy
import itertools
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_upper_common_noise as experiment
from scripts import pointmaze_upper_common_noise_stage92_spec as spec
from scripts.submit_pointmaze_call_weighted_stage87_scheduleurm import task_specification,qualification_task
import test_pointmaze_call_weighted as call_fixture
import test_pointmaze_feasible_credit as feasible_fixture
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_upper_paths import predictor


class UpperCommonNoiseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def donors(self, source, period):
        weights,training,payloads = {},{},{}
        protocols = {"joint_call":spec.source.teacher_source,"lower_matched":spec.source.source,"lower_fixed_upper":spec.source}
        for method,delta in (("joint_call",.01),("lower_matched",.02),("lower_fixed_upper",.03)):
            model = copy.deepcopy(source)
            with torch.no_grad():
                next(model.lower_actor.net.parameters()).add_(delta)
                if method != "lower_matched":next(model.upper_actor.net.parameters()).add_(.01)
            weights[method] = experiment.learning.native.joint.inference_weights(model)
            path = str(spec.donor_result(310011,method).parent/"final_weights"/f"period_{period}_{method}.pt")
            training[method] = {"groups":{str(period):{"trained":{method:{"checkpoint":path,"evaluation_update":8,
                "final_freeze_check":"passed","history":[{"actors":{"lower":{"scenario_covariance_trace":{"A":1.,"B":2.}}}}]}}}}}
            payloads[path] = {"protocol":protocols[method].EXPERIMENT_PROTOCOL,"root":310011,"period":period,
                "method":method,"updates":8,"weights":weights[method]}
        return weights,training,payloads

    def collect(self,model,args,roles,period,method):
        native=experiment.learning.native
        native.init_worker(model.config,args)
        weights=native.joint.inference_weights(model)
        envelope={"velocity_speed_q99":1.,"axis_min":[-1.,-1.],"axis_max":[1.,1.]}
        with patch.object(native.joint,"_make_task",side_effect=lambda **kw:DenseTask()),patch.object(
                native.joint,"pointmaze_goal_bounds",return_value=(-2*np.ones(2),2*np.ones(2))):
            return {b:[experiment.worker_pair([(weights,s["scenario_seed"],n,method,period,predictor(),.02,envelope,True)
                for n in s["noise_seeds"]]) for s in roles["credit_"+b]] for b in ("A","B")}

    def test_conditional_other_lower_baseline_has_zero_score_contribution(self):
        p=.3
        moments={}
        for common in (True,False):
            samples=[]
            for u1,u2,a1,a2 in itertools.product((-1.,1.),(-1.,1.),(0.,1.),(0.,1.)):
                if common and u1!=u2:continue
                weight=(.5 if common else .25)*(p if a1 else 1-p)*(p if a2 else 1-p)
                r1,r2=9*u1+a1*(2+u1),9*u2+a2*(2+u2)
                g=.5*(r1-r2)*((a1-p)-(a2-p))
                samples.append((weight,g,.5*(r2*(a1-p)+r1*(a2-p))))
            mean=sum(w*g for w,g,_ in samples)
            self.assertAlmostEqual(sum(w for w,_,_ in samples),1.)
            self.assertAlmostEqual(mean,2*p*(1-p))
            self.assertAlmostEqual(sum(w*b for w,_,b in samples),0.)
            moments[common]=sum(w*(g-mean)**2 for w,g,_ in samples)
        self.assertLess(moments[True],moments[False])

    def test_real_fixture_pair_shares_upper_innovations_not_lower_actions_and_eval_is_normal(self):
        model=feasible_fixture.FeasibleCreditTest().source()
        args,roles=spec.arguments(310011,preflight=True),spec.seed_roles(310011,preflight=True)
        legacy=call_fixture.CallWeightedTest().collect(model,args,roles["training_rounds"][0],50,"source_upper_common")
        batches=self.collect(model,args,roles["training_rounds"][0],50,"source_upper_common")
        pair,roster=batches["A"][0],roles["training_rounds"][0]["credit_A"][0]
        for level in ("lower","upper"):
            for key in ("state","action","reward","old_logp","old_value"):
                np.testing.assert_array_equal(getattr(getattr(pair[0][0],level),key),getattr(getattr(legacy["A"][0][0][0],level),key))
        with torch.no_grad():
            noises=[]
            for batch,_ in (pair[1],legacy["A"][0][1]):
                dist=model.lower_actor.distribution(torch.as_tensor(batch.lower.state,dtype=torch.float32))
                noises.append((batch.lower.action-dist.mean.numpy())/dist.stddev.numpy())
            np.testing.assert_allclose(*noises,atol=2e-6,rtol=0)
        with patch.object(experiment.learning.native,"_WORKER",None):
            experiment.check_scenario_pair(pair,roster,root=310011)
        self.assertEqual(pair[0][1]["policy_seed"],pair[1][1]["policy_seed"])
        self.assertNotEqual(pair[0][1]["lower_seed"],pair[1][1]["lower_seed"])
        self.assertFalse(np.array_equal(pair[0][0].lower.action,pair[1][0].lower.action))
        bad=copy.deepcopy(pair);bad[1][1]["lower_seed"]=bad[0][1]["lower_seed"]
        with self.assertRaises(ValueError):experiment.check_scenario_pair(bad,roster,root=310011)
        bad=copy.deepcopy(pair);bad[1][1]["upper_standard_noise"][0][0]+=.01
        with self.assertRaises(AssertionError):experiment.check_scenario_pair(bad,roster,root=310011)
        native=experiment.learning.native
        native.init_worker(model.config,args)
        envelope={"velocity_speed_q99":1.,"axis_min":[-1.,-1.],"axis_max":[1.,1.]}
        seed=roles["native_evaluation"][0]
        with patch.object(native.joint,"_make_task",side_effect=lambda **kw:DenseTask()),patch.object(
                native.joint,"pointmaze_goal_bounds",return_value=(-2*np.ones(2),2*np.ones(2))):
            batch,row=experiment.scenario.worker_native((native.joint.inference_weights(model),seed,seed,"base",50,predictor(),.02,envelope,False))
        self.assertIsNone(batch)
        self.assertNotIn("upper_noise_seed",row)
        self.assertEqual(row["upper_replay_forward_calls"],0)
        self.assertEqual((row["policy_seed"],row["lower_seed"]),experiment.scenario.spec.noise_seeds(310011,seed,seed))

    def test_registered_donors_use_hybrid_reference_for_LC_and_only_upper_is_installed(self):
        source=feasible_fixture.FeasibleCreditTest().source()
        before=copy.deepcopy(source.state_dict())
        weights,training,payloads=self.donors(source,50)
        models={m:copy.deepcopy(source) for m in spec.METHODS}
        cost=dict.fromkeys(spec.budget(preflight=True),0)
        with patch.object(experiment.torch,"load",side_effect=lambda p,**kw:payloads[str(p)]) as load:
            donors,metadata=experiment.prepare_training(training,310011,50,models,cost)
        self.assertEqual(load.call_count,3)
        self.assertEqual((cost["checkpoint_loads"],cost["checkpoint_freeze_checks"],cost["training_initialization_checks"]),(3,3,2))
        experiment.learning.native.curves.support.assert_frozen(models["source_upper_common"],before)
        experiment.learning.native.curves.support.assert_frozen(models["joint_upper_common"],{**before,"upper_actor":weights["joint_call"]["upper_actor"]})
        experiment.learning.native.curves.support.assert_frozen(source,before)
        self.assertEqual(metadata["training_noise_pairing"],"common_upper_independent_lower")
        self.assertEqual(set(metadata["baseline_initial_covariance"]),{"lower_matched","lower_fixed_upper"})
        bad=copy.deepcopy(payloads)
        for payload in bad.values():
            if payload["method"]=="lower_fixed_upper":payload["weights"]["upper_actor"]=before["upper_actor"]
        with patch.object(experiment.torch,"load",side_effect=lambda p,**kw:bad[str(p)]):
            with self.assertRaises(AssertionError):
                experiment.prepare_training(training,310011,50,{m:copy.deepcopy(source) for m in spec.METHODS},cost)

    def test_both_lower_means_use_all_samples_and_matched_KL_with_fixed_upper(self):
        source=feasible_fixture.FeasibleCreditTest().source()
        args,roles=spec.arguments(310011,preflight=True),spec.seed_roles(310011,preflight=True)
        cost=dict.fromkeys(spec.budget(preflight=True),0)
        for period in spec.PERIODS:
            models={m:copy.deepcopy(source) for m in spec.METHODS}
            _,training,payloads=self.donors(source,period)
            with patch.object(experiment.torch,"load",side_effect=lambda p,**kw:payloads[str(p)]):
                experiment.prepare_training(training,310011,period,models,cost)
            snapshots={m:copy.deepcopy(model.state_dict()) for m,model in models.items()}
            for r in roles["training_rounds"]:
                for method,model in models.items():
                    batches=self.collect(model,args,r,period,method)
                    for b in ("A","B"):
                        for pairs,s in zip(batches[b],r["credit_"+b]):experiment.check_scenario_pair(pairs,s,root=310011)
                    row=experiment.learning.update_mean(model,batches,method=method,period=period,horizon=args.horizon,
                        cost=cost,allocation=spec.allocation(method,period))
                    nominal,_=experiment.budget_training.call_budget.check_update(row,method=method,period=period,
                        horizon=args.horizon,preflight=True,protocol=spec)
                    self.assertEqual(set(row["actors"]),{"lower"})
                    self.assertEqual(row["actors"]["lower"]["gradient_episodes"],8)
                    self.assertAlmostEqual(nominal,.001-.0005/period,places=12)
                    experiment.learning.check_training_freeze(model,snapshots[method],("lower",))
            for method,model in models.items():
                self.assertFalse(torch.equal(model.lower_actor.net.state_dict()["0.weight"],source.lower_actor.net.state_dict()["0.weight"]))
        for k in ("actor_mean_parameter_updates","policy_updates","training_freeze_checks","actor_score_forward_batches",
                "actor_score_backward_batches","fisher_jvp_batches","exact_kl_forward_batches","checkpoint_loads",
                "checkpoint_freeze_checks","training_initialization_checks"):
            self.assertEqual(cost[k],spec.budget(preflight=True)[k],k)

    def test_all_twelve_exact_compositions_and_source_upper_controls_survive(self):
        source=feasible_fixture.FeasibleCreditTest().source()
        original=experiment.learning.native.joint.inference_weights(source)
        donors,_,_=self.donors(source,50)
        new_source=copy.deepcopy(donors["lower_matched"])
        new_joint=copy.deepcopy(donors["lower_fixed_upper"])
        weights={"base":original,"source_upper_common":new_source,"joint_upper_common":new_joint}
        cost=dict.fromkeys(spec.budget(preflight=True),0)
        composed,metadata=experiment.prepare_evaluation(donors,50,weights,cost)
        all_donors={"source":original,**donors,"source_upper_common":new_source,"joint_upper_common":new_joint}
        for variant,(upper,lower) in spec.COMPOSITIONS.items():
            torch.testing.assert_close(composed[variant],{**original,"upper_actor":all_donors[upper]["upper_actor"],
                "lower_actor":all_donors[lower]["lower_actor"]},atol=0,rtol=0)
        self.assertEqual(cost["actor_composition_checks"],12)
        self.assertEqual(metadata["final_upper_freeze"],"passed")

    def test_fixed_noise_mapping_fresh_eval_budget_and_dynamic_scheduler(self):
        b=spec.budget(preflight=False)
        self.assertEqual((b["native_episodes"]*8,b["native_steps"]*8),(22528,27033600))
        self.assertEqual((b["actor_mean_parameter_updates"]*8,b["checkpoint_writes"]*8,b["checkpoint_loads"]*8,
            b["training_initialization_checks"]*8,b["actor_composition_checks"]*8),(256,32,48,32,192))
        self.assertEqual(b["upper_replay_forward_calls"]*8,147456)
        all_eval=[]
        for preflight in (True,False):
            for root in spec.roots(preflight=preflight):
                roles=spec.seed_roles(root,preflight=preflight)
                self.assertEqual(roles["training_rounds"],spec.source.seed_roles(root,preflight=preflight)["training_rounds"])
                self.assertFalse(set(roles["native_evaluation"])&set(call_fixture.seeds(spec.source.teacher_source.seed_roles(root,preflight=False))))
                for old in (spec.source,spec.source.source):
                    self.assertFalse(set(roles["native_evaluation"])&set(old.seed_roles(root,preflight=preflight)["native_evaluation"]))
                all_eval.extend(roles["native_evaluation"])
                task=task_specification("unit_stage92",root,preflight=preflight,protocol_spec=spec)
                self.assertIn(spec.RUNNER_SCRIPT,task["cmd"])
                self.assertEqual((task["cpu"],task["ram_mb"]),(3,3072) if preflight else (9,8192))
                self.assertEqual(task["allowed_nodes"],[f"node{i:03}" for i in range(1,7)])
                self.assertFalse(task.get("require_node"))
                self.assertTrue(task["result_dir"].endswith("/completion"))
            q=qualification_task("unit_stage92",preflight=preflight,protocol_spec=spec)
            self.assertIn(spec.ANALYZER_SCRIPT,q["cmd"])
            self.assertIsNone(q["result_dir"])
        self.assertEqual(len(all_eval),len(set(all_eval)))
        for method in spec.CHECKPOINT_METHODS:
            protocol={"joint_call":spec.source.teacher_source,"lower_matched":spec.source.source,"lower_fixed_upper":spec.source}[method]
            training={"status":"complete","protocol":protocol.EXPERIMENT_PROTOCOL,"root":310011,"preflight":False,
                "contract":protocol.contract(),"seed_roles":protocol.seed_roles(310011,preflight=False)}
            experiment.check_training(training,310011,method)
            training["preflight"]=True
            with self.assertRaises(ValueError):experiment.check_training(training,310011,method)

    def test_all26_CIs_share_one_family_and_global_benefit_requires_all_four_primaries(self):
        cells=[{"root":r,"groups":{"both":{"effects":dict.fromkeys(spec.ENDPOINTS,2.)}},"cost":spec.budget(preflight=False)}
            for r in spec.roots(preflight=False)]
        with patch.object(experiment,"qualify",side_effect=lambda c,**kw:c),patch.object(spec,"BOOTSTRAP_DRAWS",128), \
                patch.object(experiment.learning.native.np,"quantile",wraps=np.quantile) as quantile:
            result=experiment.aggregate(cells,preflight=False)
        self.assertEqual(quantile.call_args.args[1],[.05/52,1-.05/52])
        self.assertEqual(result["conditioning_confirmation"],"supported")
        self.assertEqual(result["primary_endpoints"],list(spec.PRIMARY_ENDPOINTS))
        self.assertEqual(len(spec.PRIMARY_ENDPOINTS),4)
        for c in cells:c["groups"]["both"]["effects"][spec.PRIMARY_ENDPOINTS[-1]]=-1.
        with patch.object(experiment,"qualify",side_effect=lambda c,**kw:c),patch.object(spec,"BOOTSTRAP_DRAWS",128):
            self.assertEqual(experiment.aggregate(cells,preflight=False)["conditioning_confirmation"],"not_supported")
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1],preflight=False)


if __name__ == "__main__":
    unittest.main()
