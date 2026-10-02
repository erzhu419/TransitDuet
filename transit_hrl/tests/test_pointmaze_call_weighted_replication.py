import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_call_weighted as experiment
from scripts import pointmaze_call_weighted_replication_stage88_spec as spec
from scripts.submit_pointmaze_call_weighted_stage87_scheduleurm import task_specification, qualification_task
import test_pointmaze_call_weighted as fixture


class CallWeightedReplicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_only_sampling_namespace_changes_the_frozen_training_and_statistics(self):
        seeds = []
        for preflight in (True,False):
            self.assertEqual(spec.options(preflight=preflight),spec.source.options(preflight=preflight))
            self.assertEqual(spec.budget(preflight=preflight),spec.source.budget(preflight=preflight))
            for root in spec.roots(preflight=preflight):
                before = copy.deepcopy(spec.source.seed_roles(root,preflight=preflight))
                current = fixture.seeds(spec.seed_roles(root,preflight=preflight))
                previous = fixture.seeds(before)
                self.assertEqual(current,[s+1_000_000 for s in previous])
                self.assertFalse(set(current)&set(previous))
                self.assertEqual(before,spec.source.seed_roles(root,preflight=preflight))
                seeds.extend(current)
            for period in spec.PERIODS:
                for method in spec.METHODS:self.assertEqual(spec.allocation(method,period),spec.source.allocation(method,period))
        self.assertEqual(len(seeds),len(set(seeds)))
        self.assertEqual((spec.ENDPOINTS,spec.BOOTSTRAP_DRAWS,spec.BOOTSTRAP_SEED),
            (spec.source.ENDPOINTS,spec.source.BOOTSTRAP_DRAWS,spec.source.BOOTSTRAP_SEED))
        self.assertEqual(spec.contract()["replication"],"same_eight_frozen_Stage78_teachers_fresh_training_and_evaluation_samples_no_Stage87_weight_reuse")

    def test_shared_update_and_new_qualification_use_the_exact_existing_optimizer(self):
        model = fixture.feasible_fixture.FeasibleCreditTest().source()
        before = copy.deepcopy(model.state_dict())
        args = spec.arguments(310011,preflight=True)
        roles = spec.seed_roles(310011,preflight=True)["training_rounds"][0]
        cost = dict.fromkeys(spec.budget(preflight=True),0)
        batches = fixture.CallWeightedTest().collect(model,args,roles,50,"joint_call")
        row = experiment.learning.update_mean(model,batches,method="joint_call",period=50,horizon=args.horizon,
            cost=cost,allocation=spec.allocation("joint_call",50))
        nominal,_ = experiment.check_update(row,method="joint_call",period=50,horizon=args.horizon,preflight=True,protocol=spec)
        self.assertEqual(nominal,.001)
        experiment.learning.check_training_freeze(model,before,spec.METHODS["joint_call"])
        with patch.object(experiment.learning,"qualify",side_effect=lambda c,**kw:c) as shared:
            cell={"root":310011,"groups":{"50":{"trained":{"joint_call":{"history":[row,row]}}}}}
            self.assertIs(experiment.qualify(cell,preflight=True,protocol=spec),cell)
            self.assertIs(shared.call_args.kwargs["protocol"],spec)

    def test_independent_family_rejects_missing_roots_and_applies_four_primary_gate(self):
        cells=[{"root":r,"groups":{"both":{"effects":dict.fromkeys(spec.ENDPOINTS,2.)}},"cost":spec.budget(preflight=False)}
            for r in spec.roots(preflight=False)]
        with patch.object(experiment,"qualify",side_effect=lambda c,**kw:c),patch.object(spec,"BOOTSTRAP_DRAWS",128), \
                patch.object(experiment.learning.native.np,"quantile",wraps=np.quantile) as quantile:
            result=experiment.aggregate(cells,preflight=False,protocol=spec)
        self.assertEqual(quantile.call_args.args[1],[.05/40,1-.05/40])
        self.assertEqual(spec.confirmation(result)["status"],"confirmed")
        for k in spec.PRIMARY_ENDPOINTS:
            changed=copy.deepcopy(result)
            changed["endpoints"][k]["ci"]=[0.,3.]
            self.assertEqual(spec.confirmation(changed)["status"],"not_confirmed")
        self.assertEqual(spec.confirmation({"status":"preflight_passed"})["status"],"mechanical_only")
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1],preflight=False,protocol=spec)

    def test_scheduler_uses_new_entrypoints_and_unpinned_completion_only_placement(self):
        for preflight in (True,False):
            t=task_specification("unit_stage88",310011,preflight=preflight,protocol_spec=spec)
            q=qualification_task("unit_stage88",preflight=preflight,protocol_spec=spec)
            self.assertEqual(t["project"],spec.EXPERIMENT_PROTOCOL)
            self.assertIn(spec.RUNNER_SCRIPT,t["cmd"])
            self.assertIn(spec.ANALYZER_SCRIPT,q["cmd"])
            self.assertEqual((t["cpu"],t["ram_mb"]),(3,3072) if preflight else (9,8192))
            self.assertEqual(t["allowed_nodes"],[f"node{i:03}" for i in range(1,7)])
            self.assertFalse(t.get("require_node"))
            self.assertTrue(t["result_dir"].endswith("/completion"))
            self.assertIsNone(q["result_dir"])
            self.assertEqual(len(q["wait_for_files"]),len(spec.roots(preflight=preflight)))


if __name__ == "__main__":
    unittest.main()
