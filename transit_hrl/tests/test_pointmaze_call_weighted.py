import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_call_weighted as experiment
from scripts import pointmaze_call_weighted_stage87_spec as spec
from scripts import pointmaze_aligned_order_stage86_spec as previous
from scripts.submit_pointmaze_call_weighted_stage87_scheduleurm import task_specification, qualification_task
import test_pointmaze_feasible_credit as feasible_fixture
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_upper_paths import predictor


def seeds(roles):
    rounds = roles.get("training_rounds",roles.get("training_chunks",[]))
    return [s["scenario_seed"] for r in rounds for b in ("A","B") for s in r["credit_"+b]] + [
        n for r in rounds for b in ("A","B") for s in r["credit_"+b] for n in s["noise_seeds"]] + roles["native_evaluation"]


class CallWeightedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def collect(self,model,args,roles,period,method):
        native = experiment.learning.native
        native.init_worker(model.config,args)
        weights = native.joint.inference_weights(model)
        envelope = {"velocity_speed_q99":1.,"axis_min":[-1.,-1.],"axis_max":[1.,1.]}
        with patch.object(native.joint,"_make_task",side_effect=lambda **kw:DenseTask()), patch.object(
                native.joint,"pointmaze_goal_bounds",return_value=(-2*np.ones(2),2*np.ones(2))):
            batches = {}
            for b in ("A","B"):
                pairs = [experiment.learning.parts.scenario.worker_native((weights,s["scenario_seed"],n,method,period,
                    predictor(),.02,envelope,True)) for s in roles["credit_"+b] for n in s["noise_seeds"]]
                batches[b] = [pairs[i:i+2] for i in range(0,len(pairs),2)]
        return batches

    def test_shared_full_gradients_call_budget_and_frozen_parameters_over_fresh_rounds(self):
        original = feasible_fixture.FeasibleCreditTest().source()
        before = copy.deepcopy(original.state_dict())
        args,roles = spec.arguments(310011,preflight=True),spec.seed_roles(310011,preflight=True)
        cost = dict.fromkeys(spec.budget(preflight=True),0)
        for period in spec.PERIODS:
            models = {m:copy.deepcopy(original) for m in spec.METHODS}
            for round_roles in roles["training_rounds"]:
                for method,model in models.items():
                    prior = copy.deepcopy(model.state_dict())
                    batches = self.collect(model,args,round_roles,period,method)
                    row = experiment.learning.update_mean(model,batches,method=method,period=period,horizon=args.horizon,
                        cost=cost,allocation=spec.allocation(method,period))
                    nominal,exact = experiment.check_update(row,method=method,period=period,horizon=args.horizon,preflight=True)
                    expected = .001 if method != "joint_level" else .0005+.0005/period
                    self.assertAlmostEqual(nominal,expected,places=12)
                    self.assertTrue(.5*nominal <= exact <= 2*nominal)
                    self.assertEqual({r["gradient_episodes"] for r in row["actors"].values()},{8})
                    experiment.learning.check_training_freeze(model,prior,spec.METHODS[method])
                    for a,r in row["actors"].items():
                        self.assertLess(r["max_abs_old_logp_difference"],1e-4)
                        self.assertTrue(any(not torch.equal(v,model.state_dict()[a+"_actor"][k]) for k,v in prior[a+"_actor"].items()))
                    bad = copy.deepcopy(row)
                    bad["exact_call_weighted_kl"] += .001
                    with self.assertRaises(ValueError):experiment.check_update(bad,method=method,period=period,horizon=args.horizon,preflight=True)
                    bad = copy.deepcopy(row)
                    bad["actors"]["lower"]["gradient_episodes"] //= 2
                    with self.assertRaises(ValueError):experiment.check_update(bad,method=method,period=period,horizon=args.horizon,preflight=True)
            for method,model in models.items():experiment.learning.check_training_freeze(model,before,spec.METHODS[method])
        for k in ("objective_checks","mc_calls","actor_score_forward_batches","actor_score_backward_batches","fisher_jvp_batches",
                "exact_kl_forward_batches","actor_mean_parameter_updates","policy_updates","training_freeze_checks"):
            self.assertEqual(cost[k],spec.budget(preflight=True)[k],k)
        experiment.learning.native.curves.support.assert_frozen(original,before)

    def test_chain_rule_call_counts_and_no_extra_budget_tuning(self):
        for p in spec.PERIODS:
            h = spec.arguments(310011,preflight=False).horizon
            allocation = spec.allocation("joint_call",p)
            per_path = .001*(h*allocation["lower"]+(h//p)*allocation["upper"])
            self.assertAlmostEqual(per_path/h,.001,places=12)
            self.assertEqual(allocation["upper"],.5)
            self.assertEqual(spec.options(preflight=False)["updates"],8)
            self.assertEqual(spec.allocation("joint_level",p),spec.source.allocation("joint_trained",p))

    def test_final_checkpoint_uses_new_protocol_and_reloads_exactly(self):
        model = feasible_fixture.FeasibleCreditTest().source()
        with tempfile.TemporaryDirectory() as tmp:
            path = experiment.learning.final_checkpoint(model,Path(tmp)/"result.json",root=310011,period=50,
                method="joint_call",updates=8,protocol=spec)
            saved = torch.load(path,map_location="cpu",weights_only=False)
            self.assertEqual(saved["protocol"],spec.EXPERIMENT_PROTOCOL)
            self.assertEqual((saved["method"],saved["updates"]),("joint_call",8))
            replica = copy.deepcopy(model)
            replica.load_state_dict(saved["weights"])
            torch.testing.assert_close(saved["weights"],experiment.learning.native.joint.inference_weights(replica),atol=0,rtol=0)
            self.assertEqual(len(list(Path(tmp).rglob("*.pt"))),1)

    def test_rosters_budget_and_scheduler_completion_only_dynamic_placement(self):
        budget = spec.budget(preflight=False)
        self.assertEqual((budget["native_episodes"]*8,budget["native_steps"]*8),(27136,32563200))
        self.assertEqual((budget["actor_mean_parameter_updates"]*8,budget["checkpoint_writes"]*8),(640,48))
        all_seeds = []
        for preflight in (True,False):
            for root in spec.roots(preflight=preflight):
                current = seeds(spec.seed_roles(root,preflight=preflight))
                self.assertEqual(len(current),len(set(current)))
                self.assertFalse(set(current)&set(seeds(spec.source.seed_roles(root,preflight=preflight))))
                self.assertFalse(set(current)&set(seeds(previous.seed_roles(root,preflight=preflight))))
                all_seeds.extend(current)
                task = task_specification("unit_stage87",root,preflight=preflight)
                self.assertEqual((task["cpu"],task["ram_mb"]),(3,3072) if preflight else (9,8192))
                self.assertEqual(task["allowed_nodes"],[f"node{i:03}" for i in range(1,7)])
                self.assertFalse(task.get("require_node"))
                self.assertTrue(task["result_dir"].endswith("/completion"))
            q = qualification_task("unit_stage87",preflight=preflight)
            self.assertIsNone(q["result_dir"])
            self.assertEqual(len(q["wait_for_files"]),len(spec.roots(preflight=preflight)))
        self.assertEqual(len(all_seeds),len(set(all_seeds)))

    def test_all20_endpoints_in_one_corrected_family_and_missing_roots_rejected(self):
        cells = [{"root":r,"groups":{"both":{"effects":dict.fromkeys(spec.ENDPOINTS,2.)}},"cost":spec.budget(preflight=False)}
            for r in spec.roots(preflight=False)]
        with patch.object(experiment,"qualify",side_effect=lambda c,**kw:c),patch.object(spec,"BOOTSTRAP_DRAWS",128), \
                patch.object(experiment.learning.native.np,"quantile",wraps=np.quantile) as quantile:
            result = experiment.aggregate(cells,preflight=False)
        self.assertEqual(quantile.call_args.args[1],[.05/40,1-.05/40])
        self.assertEqual(set(result["endpoints"]),set(spec.ENDPOINTS))
        self.assertIn("Stage67_critic_route_HOLD_unchanged",result["native_trial_prerequisite"])
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1],preflight=False)


if __name__ == "__main__":
    unittest.main()
