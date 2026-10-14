import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_decision_relevance as learning
from scripts import run_native_transit_decision_relevance_stage225 as spec
from scripts.submit_native_transit_decision_relevance_stage225_scheduleurm import task_specification
from tests import test_native_transit_best_action_crossfit_stage220 as old
from tests import test_native_transit_best_action_stage218 as cohort
from tests.test_native_transit_refit_binding_stage224 import decision_data


def calibration():
    return {mode: {v: {"alpha": 0. if mode == "risk" else .5} for v in learning.binding.WAIT_VARIANTS}
        for mode in ("risk", "oof", "final", "joint")}


def zero_data(rows, panels, indices, bundle=None):
    n = len(rows)
    base = {"originals": np.asarray([r["original_choice"] for r in rows]), "roots": np.asarray([r["root"] for r in rows]),
        "costs": np.zeros((n, 6)), "offset": np.zeros((n, 6)),
        **{v: np.zeros((n, 6)) for v in learning.binding.WAIT_VARIANTS}}
    return learning.truth(rows, panels, indices, base)


class DecisionRelevanceTest(unittest.TestCase):
    def test_causal_masks_identity_ties_and_empty_contenders(self):
        data = decision_data()
        before = learning.masks(data)
        self.assertEqual({k: int(v.sum()) for k, v in before.items()}, {"all": 10, "contender": 2, "front": 2, "excluded": 8})
        changed = copy.deepcopy(data)
        changed["values"] += 100.
        changed["state_conditioned_pair"] += 100.
        self.assertTrue(all(np.array_equal(v, learning.masks(changed)[k]) for k, v in before.items()))
        data["costs"][:, 1] = 0.
        report = learning.summarized(learning.evaluate(data, data, calibration()))
        self.assertEqual(report["strata"]["contender"]["candidate_cases"], 0)
        self.assertIsNone(report["strata"]["front"]["heads"]["glm"]["physical_mass_loss_min"])
        self.assertEqual(report["decisions"]["glm"]["changed_fraction"], 0.)

    def test_excluded_actions_can_reverse_global_loss_and_gradient(self):
        data = decision_data()
        r = learning.summarized(learning.evaluate(data, data, calibration()))
        v = 'state_conditioned_pair'
        self.assertGreater(r["strata"]["all"]["gradients"][v]["zero"]["mean"], 0.)
        self.assertLess(r["strata"]["contender"]["gradients"][v]["zero"]["mean"], 0.)
        for name, sign in (("all", 1), ("contender", -1)):
            h = r["strata"][name]["heads"]
            self.assertGreater(sign*(h[f'raw_{v}']["physical_mass_loss_min"]-h['glm']["physical_mass_loss_min"]), 0.)
        self.assertEqual(r['strata']['contender']['heads'][f'raw_{v}']['false_safe_count'], 0)
        self.assertEqual(r['strata']['excluded']['heads'][f'raw_{v}']['false_safe_count'], 8)
        g = r['strata']
        self.assertAlmostEqual(g['all']['gradients'][v]['zero']['contribution_all'],
            g['contender']['gradients'][v]['zero']['contribution_all']+g['excluded']['gradients'][v]['zero']['contribution_all'])

    def test_switch_harm_distinguishes_stable_and_future_flips(self):
        data, other = decision_data(), decision_data()
        other['values'][:, 1, 3] = 1.
        cal = calibration()
        before = copy.deepcopy(cal)
        first = learning.evaluate(data, other, cal)
        second = learning.evaluate(other, data, cal)
        g = 'risk_state_conditioned_pair'
        self.assertEqual(first['switches'][g]['changed_count'], 2)
        self.assertEqual(first['switches'][g]['wait_worse_current_only_count'], 2)
        self.assertEqual(second['switches'][g]['wait_worse_other_only_count'], 2)
        self.assertEqual(first['strata']['contender']['heads'][g]['stable_false_unsafe_count'], 0)
        self.assertEqual(first['strata']['contender']['flipped_wait_sign_count'], 2)
        self.assertEqual(cal, before)

    def test_aggregate_uses_candidate_counts_not_subset_mean_of_means(self):
        data, few = decision_data(), decision_data()
        few = {k: v[:1] for k, v in few.items()}
        few['costs'][0, 2] = -2.
        a, b = learning.evaluate(data, data, calibration()), learning.evaluate(few, few, calibration())
        merged = learning.summarized(learning.combine([a, b]))
        total = a['strata']['contender']['heads']['glm']['loss_sum_min']+b['strata']['contender']['heads']['glm']['loss_sum_min']
        self.assertEqual(merged['strata']['contender']['candidate_cases'], 4)
        self.assertAlmostEqual(merged['strata']['contender']['heads']['glm']['physical_mass_loss_min'], total/4)
        self.assertEqual(merged['state_cases'], 3)

    def test_complete_splits_controls_and_missing_candidate_rejection(self):
        rows, registered = cohort.fixture()
        views, panels = spec.cohort.source.matched_views(rows, registered), old.panels_for(rows)
        results = {}
        with patch.object(learning.binding, 'final_data', side_effect=zero_data):
            for key in spec.KEYS:
                view, fold, partition, direction = key
                train, held = spec.cohort.split_rows(views[view], fold)
                a = spec.PARTITIONS[partition][direction]
                oof = zero_data(learning.binding.precision.validation_rows(train), panels, a)
                results[key] = {'protocol': spec.EXPERIMENT_PROTOCOL, 'contract': spec.contract(), 'key': list(key),
                    'cell': spec.analyze_cell(train, held, panels, {}, oof, calibration(), partition, direction)}
        report = spec.summarize(results, views)
        self.assertTrue(all(x['scopes']['held_other']['state_cases']==480 and x['scopes']['oof_other']['state_cases']==960 for x in report['views'].values()))
        prior = {'views': {v: {'scopes': {s: {'variants': x['decisions'],
            'wait_heads': {g: {'physical_mass_loss_min': h['physical_mass_loss_min']} for g,h in x['strata']['all']['heads'].items()}}
            for s,x in d['scopes'].items() if s in spec.previous.SCOPES}} for v,d in report['views'].items()}}
        self.assertEqual(spec.reproduce_controls(report, prior), {'comparisons':1280,'max_abs_difference':0.})
        prior['views']['mixed_a']['scopes']['held_other']['wait_heads']['glm']['physical_mass_loss_min'] += 1.
        with self.assertRaisesRegex(ValueError,'failed reproduction'): spec.reproduce_controls(report,prior)
        results[spec.KEYS[0]]['cell']['scopes']['train_same']['strata']['front']['candidate_cases'] = 61
        with self.assertRaisesRegex(ValueError,'candidate partition'): spec.summarize(results, views)

    def test_protocol_roundtrip_and_unbound_report_only_sync(self):
        self.assertEqual(spec.contract(), json.loads(json.dumps(spec.contract())))
        for key in (spec.KEYS[0], (None,None,None,None)):
            task = task_specification('unit_relevance',*key)
            self.assertEqual((task['cpu'],task['ram_mb']), (1,3072))
            self.assertIsNone(task['require_node'])
            self.assertEqual(task['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
            self.assertNotIn('_raw',task['result_dir'])
        self.assertEqual(spec.contract()['budget']['wait_residual_updates'],0)


if __name__ == '__main__':
    unittest.main()
