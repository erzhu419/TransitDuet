import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_staged_upper as experiment
from scripts import pointmaze_staged_upper_confirmation_stage95_spec as spec
from scripts.submit_pointmaze_call_weighted_stage87_scheduleurm import task_specification, qualification_task
import test_pointmaze_call_weighted as call_fixture
import test_pointmaze_feasible_credit as feasible_fixture
import test_pointmaze_staged_upper as staged_fixture


class StagedUpperConfirmationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_new_roles_are_disjoint_without_mutating_reference_rosters(self):
        new,old = [],[]
        for preflight in (True,False):
            for root in spec.roots(preflight=preflight):
                before = spec.reference.seed_roles(root,preflight=preflight)
                roles = spec.seed_roles(root,preflight=preflight)
                self.assertEqual(before,spec.reference.seed_roles(root,preflight=preflight))
                self.assertEqual(roles,spec.seed_roles(root,preflight=preflight))
                new.extend(call_fixture.seeds(roles))
                for prior in (spec.reference,spec.source,spec.source.source,spec.source.source.source,spec.source.source.source.teacher_source):
                    old.extend(call_fixture.seeds(prior.seed_roles(root,preflight=preflight)))
        self.assertEqual(len(new),len(set(new)))
        self.assertFalse(set(new)&set(old))

    def test_donors_budget_compositions_and_all_decision_endpoints_remain_fixed(self):
        self.assertIs(spec.source,spec.reference.source)
        self.assertEqual(spec.ENDPOINTS,spec.reference.ENDPOINTS)
        self.assertEqual(spec.PRIMARY_ENDPOINTS,spec.reference.PRIMARY_ENDPOINTS)
        self.assertEqual(spec.COMPOSITIONS,spec.reference.COMPOSITIONS)
        self.assertEqual(spec.BOOTSTRAP_SEED,(95,95095))
        for preflight in (True,False):
            self.assertEqual(spec.budget(preflight=preflight),spec.reference.budget(preflight=preflight))
            self.assertEqual(spec.options(preflight=preflight),spec.reference.options(preflight=preflight))
            task = task_specification("unit_stage95",310011,preflight=preflight,protocol_spec=spec)
            self.assertIn(spec.RUNNER_SCRIPT,task["cmd"])
            self.assertEqual((task["cpu"],task["ram_mb"]),(3,3072) if preflight else (9,8192))
            self.assertEqual(task["allowed_nodes"],[f"node{i:03}" for i in range(1,7)])
            self.assertFalse(task.get("require_node"))
            self.assertTrue(task["result_dir"].endswith("/completion"))
            q = qualification_task("unit_stage95",preflight=preflight,protocol_spec=spec)
            self.assertIn(spec.ANALYZER_SCRIPT,q["cmd"])
            self.assertIsNone(q["result_dir"])
        for m in spec.CHECKPOINT_METHODS:
            self.assertEqual(spec.donor_result(310011,m),spec.reference.donor_result(310011,m))
        for p in spec.PERIODS:
            for m in spec.METHODS:self.assertEqual(spec.allocation(m,p),{"upper":.5})
        for k,v in spec.reference.contract().items():
            if k not in ("sampling","statistics","limits"):self.assertEqual(spec.contract()[k],v)

    def test_confirmation_installs_Stage93_lowers_but_no_learned_upper(self):
        source = feasible_fixture.FeasibleCreditTest().source()
        before = copy.deepcopy(source.state_dict())
        weights,training,payloads = staged_fixture.StagedUpperTest().donors(source,50)
        models = {m:copy.deepcopy(source) for m in spec.METHODS}
        cost = dict.fromkeys(spec.budget(preflight=True),0)
        with patch.object(experiment.torch,"load",side_effect=lambda p,**kw:payloads[str(p)]) as load:
            donors,_ = experiment.prepare_training(training,310011,50,models,cost,protocol=spec)
        self.assertEqual(load.call_count,3)
        for m,model in models.items():
            experiment.learning.native.curves.support.assert_frozen(model,
                {**before,"lower_actor":weights[spec.LOWER_FOR_METHOD[m]]["lower_actor"]})
        trained = {m:experiment.learning.native.joint.inference_weights(model) for m,model in models.items()}
        original = experiment.learning.native.joint.inference_weights(source)
        composed,_ = experiment.prepare_evaluation(donors,50,{"base":original,**trained},cost,protocol=spec)
        self.assertEqual(set(composed),set(spec.VARIANTS))
        experiment.learning.native.curves.support.assert_frozen(source,before)
        record = {"status":"complete","protocol":spec.source.EXPERIMENT_PROTOCOL,"root":310011,"preflight":False,
            "contract":spec.source.contract(),"seed_roles":spec.source.seed_roles(310011,preflight=False),
            "cost":spec.source.budget(preflight=False)}
        experiment.check_training(record,310011,"source_upper_common",protocol=spec)
        record["protocol"] = spec.reference.EXPERIMENT_PROTOCOL
        with self.assertRaises(ValueError):experiment.check_training(record,310011,"source_upper_common",protocol=spec)

    def test_run_forwards_confirmation_protocol_to_initialization_evaluation_and_qualification(self):
        with patch.object(Path,"read_text",return_value=json.dumps({"donor":True})), \
                patch.object(experiment,"check_training") as check, \
                patch.object(experiment,"prepare_training",return_value=({"registered":"donors"},{"initialized":True})) as initialize, \
                patch.object(experiment,"prepare_evaluation",return_value=({},{"composed":True})) as compose, \
                patch.object(experiment,"qualify") as qualify,patch.object(experiment.learning,"run") as run:
            experiment.run(310011,preflight=True,output=Path("unused.json"),protocol=spec)
            kw = run.call_args.kwargs
            self.assertIs(kw["protocol"],spec)
            self.assertEqual(kw["initialize_models"](50,{},{}),{"initialized":True})
            kw["evaluation_weights"](50,{},{});kw["qualifier"]({},preflight=True)
        self.assertEqual(check.call_count,3)
        self.assertIs(check.call_args.kwargs["protocol"],spec)
        self.assertIs(initialize.call_args.kwargs["protocol"],spec)
        self.assertIs(compose.call_args.kwargs["protocol"],spec)
        self.assertIs(qualify.call_args.kwargs["protocol"],spec)

    def test_confirmation_keeps_all28_CIs_and_all_four_primary_stop_rule(self):
        cells = [{"root":r,"groups":{"both":{"effects":dict.fromkeys(spec.ENDPOINTS,2.)}},"cost":spec.budget(preflight=False)}
            for r in spec.roots(preflight=False)]
        with patch.object(experiment,"qualify",side_effect=lambda c,**kw:c) as qualify, \
                patch.object(spec,"BOOTSTRAP_DRAWS",128), \
                patch.object(experiment.learning.native.np,"quantile",wraps=np.quantile) as quantile:
            result = experiment.aggregate(cells,preflight=False,protocol=spec)
        self.assertIs(qualify.call_args.kwargs["protocol"],spec)
        self.assertEqual(result["protocol"],spec.EXPERIMENT_PROTOCOL)
        self.assertEqual(result["contract"],spec.contract())
        self.assertEqual(quantile.call_args.args[1],[.05/56,1-.05/56])
        self.assertEqual(result["staged_confirmation"],"supported")
        for c in cells:c["groups"]["both"]["effects"][spec.PRIMARY_ENDPOINTS[-1]]=0.
        with patch.object(experiment,"qualify",side_effect=lambda c,**kw:c),patch.object(spec,"BOOTSTRAP_DRAWS",128):
            self.assertEqual(experiment.aggregate(cells,preflight=False,protocol=spec)["staged_confirmation"],"not_supported")
            self.assertEqual(experiment.aggregate(cells[:1],preflight=True,protocol=spec)["staged_confirmation"],"mechanical_only")
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1],preflight=False,protocol=spec)


if __name__ == "__main__":
    unittest.main()
