import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_staged_upper as experiment
from scripts import pointmaze_staged_upper_stage94_spec as spec
from scripts.submit_pointmaze_call_weighted_stage87_scheduleurm import task_specification, qualification_task
import test_pointmaze_call_weighted as call_fixture
import test_pointmaze_feasible_credit as feasible_fixture


class StagedUpperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def donors(self, source, period):
        weights,training,payloads = {},{},{}
        for i,method in enumerate(spec.CHECKPOINT_METHODS,1):
            model = copy.deepcopy(source)
            with torch.no_grad():
                next(model.lower_actor.net.parameters()).add_(.01*i)
                if method == "joint_call":next(model.upper_actor.net.parameters()).add_(.04)
            weights[method] = experiment.learning.native.joint.inference_weights(model)
            path = str(spec.donor_result(310011,method).parent/"final_weights"/f"period_{period}_{method}.pt")
            training[method] = {"groups":{str(period):{"trained":{method:{"checkpoint":path,
                "evaluation_update":8,"final_freeze_check":"passed"}}}}}
            protocol = (experiment.previous.common.budget_training.swap_spec.source if method == "joint_call" else spec.source)
            payloads[path] = {"protocol":protocol.EXPERIMENT_PROTOCOL,"root":310011,"period":period,
                "method":method,"updates":8,"weights":weights[method]}
        return weights,training,payloads

    def test_fresh_seed_namespace_and_additional_budget_dynamic_scheduler(self):
        new,old = [],[]
        for preflight in (True,False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root,preflight=preflight)
                new.extend(call_fixture.seeds(roles))
                for prior in (spec.source,spec.source.source,spec.source.source.source,spec.source.source.source.teacher_source):
                    old.extend(call_fixture.seeds(prior.seed_roles(root,preflight=preflight)))
            task = task_specification("unit_stage94",310011,preflight=preflight,protocol_spec=spec)
            self.assertIn(spec.RUNNER_SCRIPT,task["cmd"])
            self.assertEqual((task["cpu"],task["ram_mb"]),(3,3072) if preflight else (9,8192))
            self.assertEqual(task["allowed_nodes"],[f"node{i:03}" for i in range(1,7)])
            self.assertFalse(task.get("require_node"))
            self.assertTrue(task["result_dir"].endswith("/completion"))
            q = qualification_task("unit_stage94",preflight=preflight,protocol_spec=spec)
            self.assertIn(spec.ANALYZER_SCRIPT,q["cmd"])
            self.assertIsNone(q["result_dir"])
        self.assertEqual(len(new),len(set(new)))
        self.assertFalse(set(new)&set(old))
        b = spec.budget(preflight=False)
        self.assertEqual((b["native_episodes"]*8,b["native_steps"]*8),(22016,26419200))
        self.assertEqual((b["actor_mean_parameter_updates"]*8,b["checkpoint_writes"]*8,b["checkpoint_loads"]*8,
            b["training_initialization_checks"]*8,b["actor_composition_checks"]*8),(256,32,48,32,176))
        b = spec.budget(preflight=True)
        self.assertEqual((b["native_episodes"],b["native_steps"],b["actor_mean_parameter_updates"],
            b["checkpoint_writes"],b["upper_replay_forward_calls"]),(152,45600,8,0,0))
        for p in spec.PERIODS:
            combined = .001*spec.source.allocation("source_upper_common",p)["lower"]+.0005/p
            self.assertAlmostEqual(8*combined,.008,places=12)

    def test_only_registered_lower_is_installed_and_all_compositions_are_exact(self):
        source = feasible_fixture.FeasibleCreditTest().source()
        before = copy.deepcopy(source.state_dict())
        cost = dict.fromkeys(spec.budget(preflight=True),0)
        for period in spec.PERIODS:
            weights,training,payloads = self.donors(source,period)
            models = {m:copy.deepcopy(source) for m in spec.METHODS}
            with patch.object(experiment.torch,"load",side_effect=lambda p,**kw:payloads[str(p)]) as load:
                donors,metadata = experiment.prepare_training(training,310011,period,models,cost)
            self.assertEqual(load.call_count,3)
            self.assertEqual(metadata["lower_for_method"],spec.LOWER_FOR_METHOD)
            for i,(method,model) in enumerate(models.items(),1):
                expected = {**before,"lower_actor":weights[spec.LOWER_FOR_METHOD[method]]["lower_actor"]}
                experiment.learning.native.curves.support.assert_frozen(model,expected)
                with torch.no_grad():next(model.upper_actor.net.parameters()).add_(.001*i)
            trained = {m:experiment.learning.native.joint.inference_weights(model) for m,model in models.items()}
            original = experiment.learning.native.joint.inference_weights(source)
            composed,meta = experiment.prepare_evaluation(donors,period,{**trained,"base":original},cost)
            all_weights = {"source":original,**donors,**trained}
            for v,(upper,lower) in spec.COMPOSITIONS.items():
                torch.testing.assert_close(composed[v],{**original,"upper_actor":all_weights[upper]["upper_actor"],
                    "lower_actor":all_weights[lower]["lower_actor"]},atol=0,rtol=0)
            self.assertEqual(meta["final_lower_freeze"],"passed")
            bad = copy.deepcopy(trained)
            bad["staged_common"]["lower_actor"]["net.0.weight"] += .01
            with self.assertRaises(AssertionError):experiment.prepare_evaluation(donors,period,{**bad,"base":original},cost)
        for k in ("checkpoint_loads","checkpoint_freeze_checks","training_initialization_checks","actor_composition_checks"):
            self.assertEqual(cost[k],spec.budget(preflight=True)[k])
        experiment.learning.native.curves.support.assert_frozen(source,before)

    def test_wrong_final_update_or_upper_mutation_in_lower_donor_is_rejected(self):
        source = feasible_fixture.FeasibleCreditTest().source()
        _,training,payloads = self.donors(source,50)
        cost = dict.fromkeys(spec.budget(preflight=True),0)
        path = next(p for p,v in payloads.items() if v["method"]=="source_upper_common")
        bad = copy.deepcopy(payloads)
        bad[path]["weights"]["upper_actor"]["net.0.weight"] += .01
        with patch.object(experiment.torch,"load",side_effect=lambda p,**kw:bad[str(p)]):
            with self.assertRaises(AssertionError):
                experiment.prepare_training(training,310011,50,{m:copy.deepcopy(source) for m in spec.METHODS},cost)
        training["source_upper_independent"]["groups"]["50"]["trained"]["source_upper_independent"]["evaluation_update"]=7
        with self.assertRaises(ValueError):
            experiment.prepare_training(training,310011,50,{m:copy.deepcopy(source) for m in spec.METHODS},cost)

    def test_upper_updates_use_independent_pairs_and_keep_learned_lowers_frozen(self):
        source = feasible_fixture.FeasibleCreditTest().source()
        args,roles = spec.arguments(310011,preflight=True),spec.seed_roles(310011,preflight=True)
        cost = dict.fromkeys(spec.budget(preflight=True),0)
        for period in spec.PERIODS:
            models = {m:copy.deepcopy(source) for m in spec.METHODS}
            _,training,payloads = self.donors(source,period)
            with patch.object(experiment.torch,"load",side_effect=lambda p,**kw:payloads[str(p)]):
                experiment.prepare_training(training,310011,period,models,cost)
            before = {m:copy.deepcopy(model.state_dict()) for m,model in models.items()}
            for r in roles["training_rounds"]:
                for method,model in models.items():
                    batches = call_fixture.CallWeightedTest().collect(model,args,r,period,method)
                    for b in ("A","B"):
                        for pair,roster in zip(batches[b],r["credit_"+b]):
                            experiment.check_scenario_pair(pair,roster,root=310011)
                            bad = copy.deepcopy(pair);bad[1][1]["upper_replay_forward_calls"]=1
                            with self.assertRaises(ValueError):experiment.check_scenario_pair(bad,roster,root=310011)
                    row = experiment.learning.update_mean(model,batches,method=method,period=period,horizon=args.horizon,
                        cost=cost,allocation=spec.allocation(method,period))
                    nominal,_ = experiment.call_budget.check_update(row,method=method,period=period,
                        horizon=args.horizon,preflight=True,protocol=spec)
                    self.assertAlmostEqual(nominal,.0005/period,places=12)
                    self.assertEqual(row["actors"]["upper"]["gradient_episodes"],8)
                    experiment.learning.check_training_freeze(model,before[method],("upper",))
            for method,model in models.items():
                self.assertFalse(torch.equal(model.upper_actor.net.state_dict()["0.weight"],before[method]["upper_actor"]["net.0.weight"]))
        for k in ("actor_mean_parameter_updates","policy_updates","training_freeze_checks","actor_score_forward_batches",
                "actor_score_backward_batches","fisher_jvp_batches","exact_kl_forward_batches","checkpoint_loads",
                "checkpoint_freeze_checks","training_initialization_checks"):
            self.assertEqual(cost[k],spec.budget(preflight=True)[k],k)

    def test_donor_record_requires_full_Stage93_contract_and_budget(self):
        root = 310011
        training = {"status":"complete","protocol":spec.source.EXPERIMENT_PROTOCOL,"root":root,"preflight":False,
            "contract":spec.source.contract(),"seed_roles":spec.source.seed_roles(root,preflight=False),"cost":spec.source.budget(preflight=False)}
        for method in spec.LOWER_FOR_METHOD.values():experiment.check_training(training,root,method)
        training["cost"]["native_episodes"] -= 1
        with self.assertRaises(ValueError):experiment.check_training(training,root,"source_upper_common")

    def test_qualification_rejects_replayed_upper_noise_in_evaluation(self):
        root = 310011
        seed = spec.seed_roles(root,preflight=True)["native_evaluation"][0]
        policy,lower = experiment.scenario.spec.noise_seeds(root,seed,seed)
        cell = {"root":root,"groups":{str(p):{
            "reused_checkpoints":{m:str(spec.donor_result(root,m).parent/"final_weights"/f"period_{p}_{m}.pt") for m in spec.CHECKPOINT_METHODS},
            **dict.fromkeys(("checkpoint_freeze","training_initialization","actor_composition","final_lower_freeze"),"passed"),
            "lower_for_method":spec.LOWER_FOR_METHOD,"training_noise_pairing":"independent_upper_and_lower",
            "trained":{m:{"history":[{},{}]} for m in spec.METHODS},
            "evaluation":{"base":[{"seed":seed,"noise_seed":seed,"policy_seed":policy,"lower_seed":lower,"upper_replay_forward_calls":0}]}}
            for p in spec.PERIODS}}
        with patch.object(experiment.learning,"qualify"),patch.object(experiment.call_budget,"check_update",
                side_effect=lambda row,**kw:(.0005/kw["period"],0.)):
            experiment.qualify(cell,preflight=True)
            cell["groups"]["50"]["evaluation"]["base"][0]["upper_noise_seed"]=seed
            with self.assertRaises(ValueError):experiment.qualify(cell,preflight=True)

    def test_all28_CIs_and_all_four_primary_gate(self):
        cells = [{"root":r,"groups":{"both":{"effects":dict.fromkeys(spec.ENDPOINTS,2.)}},"cost":spec.budget(preflight=False)}
            for r in spec.roots(preflight=False)]
        with patch.object(experiment,"qualify",side_effect=lambda c,**kw:c),patch.object(spec,"BOOTSTRAP_DRAWS",128), \
                patch.object(experiment.learning.native.np,"quantile",wraps=np.quantile) as quantile:
            result = experiment.aggregate(cells,preflight=False)
        self.assertEqual(len(result["endpoints"]),28)
        self.assertEqual(quantile.call_args.args[1],[.05/56,1-.05/56])
        self.assertEqual(result["staged_confirmation"],"supported")
        self.assertEqual(result["primary_endpoints"],list(spec.PRIMARY_ENDPOINTS))
        for c in cells:c["groups"]["both"]["effects"][spec.PRIMARY_ENDPOINTS[-1]]=0.
        with patch.object(experiment,"qualify",side_effect=lambda c,**kw:c),patch.object(spec,"BOOTSTRAP_DRAWS",128):
            self.assertEqual(experiment.aggregate(cells,preflight=False)["staged_confirmation"],"not_supported")
            self.assertEqual(experiment.aggregate(cells[:1],preflight=True)["staged_confirmation"],"mechanical_only")
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1],preflight=False)


if __name__ == "__main__":
    unittest.main()
