import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_fixed_upper_lower as experiment
from scripts import pointmaze_fixed_upper_lower_stage91_spec as spec
from scripts import pointmaze_call_weighted_actor_swap_stage89_spec as stage89
from scripts.submit_pointmaze_call_weighted_stage87_scheduleurm import task_specification, qualification_task
import test_pointmaze_call_weighted as call_fixture
import test_pointmaze_feasible_credit as feasible_fixture


class FixedUpperLowerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def donors(self, source, period):
        weights,training,payloads = {},{},{}
        for method,delta in (("joint_call",.01),("lower_matched",.02)):
            model = copy.deepcopy(source)
            with torch.no_grad():
                for actor in (("upper","lower") if method == "joint_call" else ("lower",)):
                    next(getattr(model,actor+"_actor").net.parameters()).add_(delta)
            weights[method] = experiment.learning.native.joint.inference_weights(model)
            path = str(spec.donor_result(310011,method).parent/"final_weights"/f"period_{period}_{method}.pt")
            protocol = spec.teacher_source if method == "joint_call" else spec.source
            training[method] = {"groups":{str(period):{"trained":{method:{"checkpoint":path,
                "evaluation_update":8,"final_freeze_check":"passed"}}}}}
            payloads[path] = {"protocol":protocol.EXPERIMENT_PROTOCOL,"root":310011,"period":period,
                "method":method,"updates":8,"weights":weights[method]}
        return weights,training,payloads

    def test_installs_only_registered_upper_preserving_source_lower_std_values_and_Adam(self):
        source = feasible_fixture.FeasibleCreditTest().source()
        before = copy.deepcopy(source.state_dict())
        weights,training,payloads = self.donors(source,50)
        model = copy.deepcopy(source)
        cost = dict.fromkeys(spec.budget(preflight=True),0)
        with patch.object(experiment.torch,"load",side_effect=lambda p,**kw:payloads[str(p)]) as load:
            donors,metadata = experiment.prepare_training(training,310011,50,{"lower_fixed_upper":model},cost)
        self.assertEqual(load.call_count,2)
        self.assertEqual((cost["checkpoint_loads"],cost["checkpoint_freeze_checks"],cost["training_initialization_checks"]),(2,2,1))
        experiment.learning.native.curves.support.assert_frozen(model,{**before,"upper_actor":weights["joint_call"]["upper_actor"]})
        experiment.learning.native.curves.support.assert_frozen(source,before)
        torch.testing.assert_close(donors,weights,atol=0,rtol=0)
        self.assertEqual(metadata["training_initialization"]["upper_checkpoint"],metadata["reused_checkpoints"]["joint_call"])
        bad = copy.deepcopy(training);bad["lower_matched"]["groups"]["50"]["trained"]["lower_matched"]["evaluation_update"]=7
        with patch.object(experiment.torch,"load",side_effect=lambda p,**kw:payloads[str(p)]):
            with self.assertRaises(ValueError):
                experiment.prepare_training(bad,310011,50,{"lower_fixed_upper":copy.deepcopy(source)},cost)
        bad_payloads = copy.deepcopy(payloads)
        for payload in bad_payloads.values():payload["weights"]["upper_actor"]["log_std"].add_(.01)
        with patch.object(experiment.torch,"load",side_effect=lambda p,**kw:bad_payloads[str(p)]):
            with self.assertRaises(AssertionError):
                experiment.prepare_training(training,310011,50,{"lower_fixed_upper":copy.deepcopy(source)},cost)

    def test_lower_updates_all_samples_with_matched_budget_and_hybrid_freeze_reference(self):
        source = feasible_fixture.FeasibleCreditTest().source()
        source_before = copy.deepcopy(source.state_dict())
        args,roles = spec.arguments(310011,preflight=True),spec.seed_roles(310011,preflight=True)
        cost = dict.fromkeys(spec.budget(preflight=True),0)
        collector = call_fixture.CallWeightedTest()
        for period in spec.PERIODS:
            model = copy.deepcopy(source)
            _,training,payloads = self.donors(source,period)
            with patch.object(experiment.torch,"load",side_effect=lambda p,**kw:payloads[str(p)]):
                experiment.prepare_training(training,310011,period,{"lower_fixed_upper":model},cost)
            initialized = copy.deepcopy(model.state_dict())
            for round_roles in roles["training_rounds"]:
                batches = collector.collect(model,args,round_roles,period,"lower_fixed_upper")
                row = experiment.learning.update_mean(model,batches,method="lower_fixed_upper",period=period,
                    horizon=args.horizon,cost=cost,allocation=spec.allocation("lower_fixed_upper",period))
                nominal,_ = experiment.previous.call_budget.check_update(row,method="lower_fixed_upper",period=period,
                    horizon=args.horizon,preflight=True,protocol=spec)
                self.assertEqual(set(row["actors"]),{"lower"})
                self.assertEqual(row["actors"]["lower"]["gradient_episodes"],8)
                self.assertAlmostEqual(nominal,.001-.0005/period,places=12)
                experiment.learning.check_training_freeze(model,initialized,("lower",))
            self.assertFalse(torch.equal(model.lower_actor.net.state_dict()["0.weight"],source.lower_actor.net.state_dict()["0.weight"]))
            with self.assertRaises(AssertionError):experiment.learning.check_training_freeze(model,source_before,("lower",))
            for part in ("upper","std","value"):
                bad = copy.deepcopy(model)
                with torch.no_grad():
                    if part == "upper":next(bad.upper_actor.net.parameters()).add_(.01)
                    if part == "std":bad.lower_actor.log_std.add_(.01)
                    if part == "value":next(bad.lower_value.parameters()).add_(.01)
                with self.assertRaises(AssertionError):experiment.learning.check_training_freeze(bad,initialized,("lower",))
        for k in ("actor_mean_parameter_updates","policy_updates","training_freeze_checks","actor_score_forward_batches",
                "actor_score_backward_batches","fisher_jvp_batches","exact_kl_forward_batches","checkpoint_loads",
                "checkpoint_freeze_checks","training_initialization_checks"):
            self.assertEqual(cost[k],spec.budget(preflight=True)[k],k)
        experiment.learning.native.curves.support.assert_frozen(source,source_before)

    def test_evaluation_keeps_original_upper_controls_and_all_eight_exact_compositions(self):
        source = feasible_fixture.FeasibleCreditTest().source()
        original = experiment.learning.native.joint.inference_weights(source)
        donors,_,_ = self.donors(source,50)
        fixed = copy.deepcopy(donors["joint_call"])
        fixed["lower_actor"]["net.0.weight"].add_(.03)
        weights = {"base":original,"zero":original,"lower_fixed_upper":fixed}
        cost = dict.fromkeys(spec.budget(preflight=True),0)
        composed,metadata = experiment.prepare_evaluation(donors,50,weights,cost)
        all_donors = {"source":original,**donors,"lower_fixed_upper":fixed}
        for variant,(upper,lower) in spec.COMPOSITIONS.items():
            torch.testing.assert_close(composed[variant],{**original,"upper_actor":all_donors[upper]["upper_actor"],
                "lower_actor":all_donors[lower]["lower_actor"]},atol=0,rtol=0)
        self.assertEqual(cost["actor_composition_checks"],8)
        self.assertEqual(metadata["final_learned_upper_frozen"],"passed")
        weights["lower_fixed_upper"]["upper_actor"]["net.0.weight"].add_(.01)
        with self.assertRaises(AssertionError):experiment.prepare_evaluation(donors,50,weights,cost)

    def test_donor_training_rosters_and_lower_comparison_identities(self):
        for method in spec.CHECKPOINT_METHODS:
            protocol = spec.teacher_source if method == "joint_call" else spec.source
            training = {"status":"complete","protocol":protocol.EXPERIMENT_PROTOCOL,"root":310011,
                "preflight":False,"contract":protocol.contract(),"seed_roles":protocol.seed_roles(310011,preflight=False)}
            experiment.check_training(training,310011,method)
            bad = copy.deepcopy(training);bad["seed_roles"]["training_rounds"][0]["credit_A"][0]["noise_seeds"][0]+=1
            with self.assertRaises(ValueError):experiment.check_training(bad,310011,method)
            bad = copy.deepcopy(training);bad["preflight"]=True
            with self.assertRaises(ValueError):experiment.check_training(bad,310011,method)
        values = {v:float(i) for i,v in enumerate(spec.VARIANTS)}
        effects = {f"50/{a}_minus_{b}":values[a]-values[b] for a,b in spec.CONTRAST_PAIRS}
        experiment.check_decomposition(50,effects)
        effects["50/joint_upper_fixed_lower_minus_joint_call"]+=.01
        with self.assertRaises(AssertionError):experiment.check_decomposition(50,effects)

    def test_same_budget_and_training_fresh_eval_unpinned_completion_only_scheduler(self):
        b = spec.budget(preflight=False)
        self.assertEqual((b["native_episodes"]*8,b["native_steps"]*8),(12288,14745600))
        self.assertEqual((b["actor_mean_parameter_updates"]*8,b["checkpoint_writes"]*8,b["checkpoint_loads"]*8,
            b["training_initialization_checks"]*8,b["actor_composition_checks"]*8),(128,16,32,16,128))
        self.assertEqual(spec.METHODS,{"lower_fixed_upper":("lower",)})
        all_eval = []
        for preflight in (True,False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root,preflight=preflight)
                self.assertEqual(roles["training_rounds"],spec.source.seed_roles(root,preflight=preflight)["training_rounds"])
                self.assertFalse(set(roles["native_evaluation"])&set(call_fixture.seeds(spec.teacher_source.seed_roles(root,preflight=False))))
                for old in (stage89,spec.source):
                    self.assertFalse(set(roles["native_evaluation"])&set(old.seed_roles(root,preflight=preflight)["native_evaluation"]))
                all_eval.extend(roles["native_evaluation"])
                task = task_specification("unit_stage91",root,preflight=preflight,protocol_spec=spec)
                self.assertIn(spec.RUNNER_SCRIPT,task["cmd"])
                self.assertEqual((task["cpu"],task["ram_mb"]),(3,3072) if preflight else (9,8192))
                self.assertEqual(task["allowed_nodes"],[f"node{i:03}" for i in range(1,7)])
                self.assertFalse(task.get("require_node"))
                self.assertTrue(task["result_dir"].endswith("/completion"))
            q = qualification_task("unit_stage91",preflight=preflight,protocol_spec=spec)
            self.assertIn(spec.ANALYZER_SCRIPT,q["cmd"])
            self.assertIsNone(q["result_dir"])
            self.assertEqual(len(q["wait_for_files"]),len(spec.roots(preflight=preflight)))
        self.assertEqual(len(all_eval),len(set(all_eval)))

    def test_all26_endpoints_corrected_as_one_equal_root_family_and_missing_root_rejected(self):
        cells = [{"root":r,"groups":{"both":{"effects":dict.fromkeys(spec.ENDPOINTS,2.)}},
            "cost":spec.budget(preflight=False)} for r in spec.roots(preflight=False)]
        with patch.object(experiment,"qualify",side_effect=lambda c,**kw:c),patch.object(spec,"BOOTSTRAP_DRAWS",128), \
                patch.object(experiment.learning.native.np,"quantile",wraps=np.quantile) as quantile:
            result = experiment.aggregate(cells,preflight=False)
        self.assertEqual(quantile.call_args.args[1],[.05/52,1-.05/52])
        self.assertEqual(set(result["endpoints"]),set(spec.ENDPOINTS))
        self.assertEqual(result["primary_endpoints"],list(spec.PRIMARY_ENDPOINTS))
        self.assertIn("not_isolated_movement_effect",result["performance_claim"])
        self.assertIn("Stage67_critic_route_HOLD_unchanged",result["native_trial_prerequisite"])
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1],preflight=False)


if __name__ == "__main__":
    unittest.main()
