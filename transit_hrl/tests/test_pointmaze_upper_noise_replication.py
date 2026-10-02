import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_upper_noise_replication as experiment
from scripts import pointmaze_upper_noise_replication_stage93_spec as spec
from scripts.submit_pointmaze_call_weighted_stage87_scheduleurm import task_specification, qualification_task
import test_pointmaze_call_weighted as call_fixture
import test_pointmaze_feasible_credit as feasible_fixture
from test_pointmaze_joint_renewal import DenseTask
import test_pointmaze_upper_common_noise as common_fixture
from test_pointmaze_upper_paths import predictor


class UpperNoiseReplicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def collect(self, model, args, roles, period, method):
        native = experiment.learning.native
        native.init_worker(model.config,args)
        weights = native.joint.inference_weights(model)
        envelope = {"velocity_speed_q99":1.,"axis_min":[-1.,-1.],"axis_max":[1.,1.]}
        with patch.object(native.joint,"_make_task",side_effect=lambda **kw:DenseTask()),patch.object(
                native.joint,"pointmaze_goal_bounds",return_value=(-2*np.ones(2),2*np.ones(2))):
            return {b:[experiment.worker_pair([(weights,s["scenario_seed"],n,method,period,predictor(),.02,envelope,True)
                for n in s["noise_seeds"]]) for s in roles["credit_"+b]] for b in ("A","B")}

    def test_all_training_and_evaluation_seeds_are_fresh_and_disjoint(self):
        new,old = [],[]
        for preflight in (True,False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root,preflight=preflight)
                new.extend(call_fixture.seeds(roles))
                for prior in (spec.source,spec.source.source,spec.source.source.source,spec.source.source.teacher_source):
                    old.extend(call_fixture.seeds(prior.seed_roles(root,preflight=preflight)))
                self.assertEqual(len(roles["training_rounds"]),2 if preflight else 8)
        self.assertEqual(len(new),len(set(new)))
        self.assertFalse(set(new)&set(old))
        self.assertEqual(spec.ENDPOINTS,spec.source.ENDPOINTS)
        self.assertEqual(spec.PRIMARY_ENDPOINTS,spec.source.PRIMARY_ENDPOINTS)
        self.assertEqual(spec.BOOTSTRAP_SEED,(93,93093))

    def test_budget_counts_four_retrained_lowers_but_only_two_replay_streams(self):
        b = spec.budget(preflight=False)
        self.assertEqual((b["native_episodes"]*8,b["native_steps"]*8),(38912,46694400))
        self.assertEqual((b["actor_mean_parameter_updates"]*8,b["checkpoint_writes"]*8,b["checkpoint_loads"]*8,
            b["training_initialization_checks"]*8,b["actor_composition_checks"]*8),(512,64,16,64,192))
        self.assertEqual(b["upper_replay_forward_calls"]*8,147456)
        b = spec.budget(preflight=True)
        self.assertEqual((b["native_episodes"],b["native_steps"],b["actor_mean_parameter_updates"],
            b["upper_replay_forward_calls"],b["checkpoint_writes"]),(224,67200,16,144,0))
        for preflight in (True,False):
            t = task_specification("unit_stage93",310011,preflight=preflight,protocol_spec=spec)
            self.assertIn(spec.RUNNER_SCRIPT,t["cmd"])
            self.assertEqual((t["cpu"],t["ram_mb"]),(3,3072) if preflight else (9,8192))
            self.assertEqual(t["allowed_nodes"],[f"node{i:03}" for i in range(1,7)])
            self.assertFalse(t.get("require_node"))
            self.assertTrue(t["result_dir"].endswith("/completion"))
            q = qualification_task("unit_stage93",preflight=preflight,protocol_spec=spec)
            self.assertIn(spec.ANALYZER_SCRIPT,q["cmd"])
            self.assertIsNone(q["result_dir"])

    def test_independent_dispatch_is_bit_exact_and_common_dispatch_preserves_first_rollout(self):
        source = feasible_fixture.FeasibleCreditTest().source()
        args,roles = spec.arguments(310011,preflight=True),spec.seed_roles(310011,preflight=True)
        for method in ("source_upper_independent","source_upper_common"):
            legacy = call_fixture.CallWeightedTest().collect(source,args,roles["training_rounds"][0],50,method)
            batches = self.collect(source,args,roles["training_rounds"][0],50,method)
            for b in ("A","B"):
                for pairs,reference,roster in zip(batches[b],legacy[b],roles["training_rounds"][0]["credit_"+b]):
                    experiment.check_scenario_pair(pairs,roster,root=310011)
                    for i in (range(2) if method=="source_upper_independent" else (0,)):
                        for level in ("lower","upper"):
                            for key in ("state","action","reward","old_logp","old_value"):
                                np.testing.assert_array_equal(getattr(getattr(pairs[i][0],level),key),getattr(getattr(reference[i][0],level),key))
                    if method=="source_upper_independent":
                        self.assertTrue(all("upper_noise_seed" not in row and row["upper_replay_forward_calls"]==0 for _,row in pairs))
                        bad = copy.deepcopy(pairs);bad[1][1]["upper_replay_forward_calls"]=1
                        with self.assertRaises(ValueError):experiment.check_scenario_pair(bad,roster,root=310011)
                    else:
                        np.testing.assert_allclose(pairs[0][1]["upper_standard_noise"],pairs[1][1]["upper_standard_noise"],atol=2e-6,rtol=0)

    def test_four_models_keep_L0_and_install_only_the_registered_upper(self):
        source = feasible_fixture.FeasibleCreditTest().source()
        before = copy.deepcopy(source.state_dict())
        cost = dict.fromkeys(spec.budget(preflight=True),0)
        for period in spec.PERIODS:
            weights,training,payloads = common_fixture.UpperCommonNoiseTest().donors(source,period)
            models = {m:copy.deepcopy(source) for m in spec.METHODS}
            with patch.object(experiment.torch,"load",side_effect=lambda p,**kw:payloads[str(p)]) as load:
                donors,metadata = experiment.prepare_training(training["joint_call"],310011,period,models,cost)
            self.assertEqual(load.call_count,1)
            self.assertEqual(set(metadata["reused_checkpoints"]),{"joint_call"})
            self.assertEqual(metadata["training_noise_pairing"],spec.NOISE_MODES)
            for i,(method,model) in enumerate(models.items()):
                expected = {**before,"upper_actor":weights["joint_call"]["upper_actor"]} if method.startswith("joint_") else before
                experiment.learning.native.curves.support.assert_frozen(model,expected)
                with torch.no_grad():next(model.lower_actor.net.parameters()).add_(.001*(i+1))
            trained = {m:experiment.learning.native.joint.inference_weights(model) for m,model in models.items()}
            original = experiment.learning.native.joint.inference_weights(source)
            composed,meta = experiment.prepare_evaluation(donors,period,{**trained,"base":original},cost)
            all_weights = {"source":original,**donors,**trained}
            for v,(upper,lower) in spec.COMPOSITIONS.items():
                torch.testing.assert_close(composed[v],{**original,"upper_actor":all_weights[upper]["upper_actor"],
                    "lower_actor":all_weights[lower]["lower_actor"]},atol=0,rtol=0)
            self.assertEqual(meta["final_upper_freeze"],"passed")
        for k in ("checkpoint_loads","checkpoint_freeze_checks","training_initialization_checks","actor_composition_checks"):
            self.assertEqual(cost[k],spec.budget(preflight=True)[k])
        experiment.learning.native.curves.support.assert_frozen(source,before)
        training["joint_call"]["groups"]["100"]["trained"]["joint_call"]["evaluation_update"]=7
        with self.assertRaises(ValueError):experiment.prepare_training(training["joint_call"],310011,100,models,cost)

    def test_all_four_lower_gradients_use_all_samples_and_identical_per_learner_KL(self):
        source = feasible_fixture.FeasibleCreditTest().source()
        args,roles = spec.arguments(310011,preflight=True),spec.seed_roles(310011,preflight=True)
        _,training,payloads = common_fixture.UpperCommonNoiseTest().donors(source,50)
        models = {m:copy.deepcopy(source) for m in spec.METHODS}
        cost = dict.fromkeys(spec.budget(preflight=True),0)
        with patch.object(experiment.torch,"load",side_effect=lambda p,**kw:payloads[str(p)]):
            experiment.prepare_training(training["joint_call"],310011,50,models,cost)
        for method,model in models.items():
            before = copy.deepcopy(model.state_dict())
            batches = self.collect(model,args,roles["training_rounds"][0],50,method)
            for b in ("A","B"):
                for pair,roster in zip(batches[b],roles["training_rounds"][0]["credit_"+b]):
                    experiment.check_scenario_pair(pair,roster,root=310011)
            row = experiment.learning.update_mean(model,batches,method=method,period=50,horizon=args.horizon,
                cost=cost,allocation=spec.allocation(method,50))
            nominal,_ = experiment.common.budget_training.call_budget.check_update(row,method=method,period=50,
                horizon=args.horizon,preflight=True,protocol=spec)
            self.assertAlmostEqual(nominal,.00099,places=12)
            self.assertEqual(row["actors"]["lower"]["gradient_episodes"],8)
            experiment.learning.check_training_freeze(model,before,("lower",))
            self.assertFalse(torch.equal(model.lower_actor.net.state_dict()["0.weight"],before["lower_actor"]["net.0.weight"]))

    def test_qualification_rejects_common_training_noise_in_evaluation(self):
        root = 310011
        seed = spec.seed_roles(root,preflight=True)["native_evaluation"][0]
        policy,lower = experiment.scenario.spec.noise_seeds(root,seed,seed)
        cell = {"root":root,"groups":{}}
        for p in spec.PERIODS:
            cell["groups"][str(p)] = {
                "reused_checkpoints":{"joint_call":str(spec.donor_result(root,"joint_call").parent/"final_weights"/f"period_{p}_joint_call.pt")},
                **dict.fromkeys(("checkpoint_freeze","training_initialization","actor_composition","final_upper_freeze"),"passed"),
                "training_noise_pairing":spec.NOISE_MODES,
                "trained":{m:{"history":[{},{}]} for m in spec.METHODS},
                "evaluation":{"base":[{"seed":seed,"noise_seed":seed,"policy_seed":policy,"lower_seed":lower,"upper_replay_forward_calls":0}]}}
        with patch.object(experiment.learning,"qualify"),patch.object(experiment.common.budget_training.call_budget,"check_update",
                side_effect=lambda row,**kw:(.001*spec.allocation(kw["method"],kw["period"])["lower"],0.)):
            experiment.qualify(cell,preflight=True)
            cell["groups"]["50"]["evaluation"]["base"][0]["upper_noise_seed"]=seed
            with self.assertRaises(ValueError):experiment.qualify(cell,preflight=True)

    def test_all26_CIs_and_all_four_primary_decision_are_unchanged(self):
        cells = [{"root":r,"groups":{"both":{"effects":dict.fromkeys(spec.ENDPOINTS,2.)}},"cost":spec.budget(preflight=False)}
            for r in spec.roots(preflight=False)]
        with patch.object(experiment,"qualify",side_effect=lambda c,**kw:c),patch.object(spec,"BOOTSTRAP_DRAWS",128), \
                patch.object(experiment.learning.native.np,"quantile",wraps=np.quantile) as quantile:
            result = experiment.aggregate(cells,preflight=False)
        self.assertEqual(quantile.call_args.args[1],[.05/52,1-.05/52])
        self.assertEqual(result["conditioning_confirmation"],"supported")
        self.assertEqual(result["primary_endpoints"],list(spec.PRIMARY_ENDPOINTS))
        for c in cells:c["groups"]["both"]["effects"][spec.PRIMARY_ENDPOINTS[-1]]=0.
        with patch.object(experiment,"qualify",side_effect=lambda c,**kw:c),patch.object(spec,"BOOTSTRAP_DRAWS",128):
            self.assertEqual(experiment.aggregate(cells,preflight=False)["conditioning_confirmation"],"not_supported")
            self.assertEqual(experiment.aggregate(cells[:1],preflight=True)["conditioning_confirmation"],"mechanical_only")
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1],preflight=False)


if __name__ == "__main__":
    unittest.main()
