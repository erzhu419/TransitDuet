import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_cost_gate as learning
from scripts import run_native_transit_cost_gate_stage232 as spec
from scripts.submit_native_transit_cost_gate_stage232_scheduleurm import task_specification
from tests import test_native_transit_factorial_stage231 as old

joint = learning.joint


def data_for(rows):
    data = old.data_for(rows)
    z = np.ones((len(rows),6))
    z[:,1] = -2.
    z[np.arange(len(rows)),data['originals']] = 0.
    data['raw_logits'] = z
    data['raw_choices'] = joint.binding.policy.selected(data['heads'][...,0],z,data['originals'])
    return data


def source_cell(train,held,panels,model,partition,direction):
    a,b = spec.PARTITIONS[partition][direction],spec.PARTITIONS[partition][1-direction]
    output = {'scopes':{}}
    for scope in spec.SCOPES:
        rows = train if scope.startswith('train') else held
        data = data_for(rows)
        prediction = spec.previous.learning.predict(data,model)
        values = joint.physical_values(rows,panels,a if scope.endswith('same') else b)
        raw = []
        for i,r in enumerate(rows):
            j = int(data['raw_choices'][i])
            raw.append({'decisions':{'raw':{'selected':j,'cost':float(values[i,j,0]),'wait':float(values[i,j,1]),
                'changed':j!=r['original_choice'],'unsafe':bool(values[i,j,1]>0)}}})
        output['scopes'][scope] = list(reversed(spec.previous.learning.records(rows,values,prediction,raw)))
    return output


class CostGateTest(unittest.TestCase):
    def test_cost_menu_interaction_and_unrebased_raw_logits(self):
        rows = [{'root':397,'original_choice':0}]
        data = data_for(rows)
        data['raw_logits'][0,2] = -1.
        prediction = learning.predict(data,old.model_fixture())
        for kind in joint.ANCHORS:
            self.assertEqual({a:int(c[0]) for a,c in prediction['choices'][kind]['rank8'].items()},
                {'physical_base':1,'physical_cost':1,'raw_base':1,'raw_cost':2})
        self.assertEqual(int(prediction['menus']['original']['rank8']['added'][0]),1)
        self.assertEqual(int(prediction['menus']['original']['rank8']['removed'][0]),0)
        ds = {a:{'cost':-.05,'wait':-.02} for a in learning.ARMS}
        ds['raw_cost'] = {'cost':-.2,'wait':-.04}
        effects = learning.effects(ds)
        np.testing.assert_allclose(effects['interaction'],[-.15,-.02])
        np.testing.assert_allclose(effects['total'],effects['cost_on_physical']+effects['gate_on_base']+effects['interaction'])

    def test_zero_ties_raw_hold_and_unused_wait_residual(self):
        rng = np.random.default_rng(17)
        n = 120
        originals = rng.integers(0,6,n)
        heads = joint.relative(rng.integers(-3,4,(n,6,2)).astype(float),originals)
        z = rng.integers(-1,2,(n,6)).astype(float)
        z[np.arange(n),originals] = 0.
        data = {'x':np.zeros((n,6,210)),'heads':heads,'originals':originals,'roots':np.full(n,397),
            'raw_logits':z,'raw_choices':joint.binding.policy.selected(heads[...,0],z,originals)}
        model = old.model_fixture()
        p = learning.predict(data,model)
        for kind in joint.ANCHORS:
            np.testing.assert_array_equal(p['choices'][kind]['zero']['raw_cost'],data['raw_choices'])
            for mode in learning.lowrank.MODES:
                np.testing.assert_array_equal(p['choices'][kind][mode]['raw_base'],data['raw_choices'])
        self.assertTrue(all(np.array_equal(c,data['raw_choices']) for c in p['choices']['raw']['selected'].values()))
        for path in model['paths'].values():
            for level in learning.lowrank.LEVELS: path['weights'][level][1] = [1000.]*198
        after = learning.predict(data,model)
        for kind in joint.ANCHORS:
            for mode in learning.lowrank.MODES:
                for arm in learning.ARMS: np.testing.assert_array_equal(p['choices'][kind][mode][arm],after['choices'][kind][mode][arm])
        model['selection']['original']['wait'] = {'mse_level':'16','joint_level':'16'}
        self.assertEqual(learning.predict(data,model)['levels'],p['levels'])

    def test_frozen_control_scores_and_mismatch_rejection(self):
        rows = old.fixtures.native_fixture.bank_fixture()[0]['training_data'][:6]
        panels,model = old.fixtures.old.panels_for(rows),old.model_fixture()
        src = list(reversed(source_cell(rows,rows,panels,model,'first_last',0)['scopes']['train_same']))
        prediction = learning.predict(data_for(rows),model)
        values = joint.physical_values(rows,panels,(0,1,2,3))
        self.assertEqual(learning.reproduce(prediction,src,values),{'choice_comparisons':270,'changed_choices':0,
            'score_comparisons':540,'max_abs_difference':0.})
        changed = copy.deepcopy(src)
        changed[0]['gates'] = {}
        changed[0]['factorial']['original']['rank8']['decisions']['cost']['selected'] = 5
        with self.assertRaisesRegex(ValueError,'controls or raw-gate identity'): learning.reproduce(prediction,changed,values)
        changed = copy.deepcopy(src)
        changed[0]['raw']['wait'] += 1.
        with self.assertRaisesRegex(ValueError,'controls or raw-gate identity'): learning.reproduce(prediction,changed,values)

    def test_future_blind_zero_fit_analysis_and_structured_join(self):
        rows = old.fixtures.native_fixture.bank_fixture()[0]['training_data'][:12]
        train,held = rows[:6],rows[6:]
        panels,model = old.fixtures.old.panels_for(rows),old.model_fixture()
        changed = copy.deepcopy(panels)
        for r in rows:
            for j in range(6):
                if j!=r['original_choice']: changed[joint.binding.learned.state_key(r)]['values'][4:,j] += 100.
        sources = [source_cell(train,held,p,model,'first_last',0) for p in (panels,changed)]
        with (patch.object(joint,'causal_data',side_effect=lambda rs,ref:data_for(rs)),
                patch.object(joint,'fit_reference',side_effect=AssertionError('No refit')),
                patch.object(learning.lowrank,'fit_path',side_effect=AssertionError('No SVD fit'))):
            before,after = [spec.analyze_cell(train,held,p,model,{},s,'first_last',0) for p,s in zip((panels,changed),sources)]
        self.assertEqual(before['levels'],after['levels'])
        self.assertEqual(before['reproduction'],after['reproduction'])
        self.assertNotEqual(before['scopes']['held_other'],after['scopes']['held_other'])
        for scope in spec.SCOPES:
            for b,a in zip(before['scopes'][scope],after['scopes'][scope]):
                for kind in joint.ANCHORS:
                    for mode in learning.lowrank.MODES:
                        for arm in learning.ARMS:
                            self.assertEqual(b['gates'][kind][mode]['decisions'][arm]['selected'],a['gates'][kind][mode]['decisions'][arm]['selected'])

    def test_full_report_coverage_counts_and_inherited_selection(self):
        rows,registered = old.fixtures.cohort.fixture()
        views = spec.training.cohort.source.matched_views(rows,registered)
        panels,model = old.fixtures.old.panels_for(rows),old.model_fixture()
        results = {}
        with patch.object(joint,'causal_data',side_effect=lambda rs,ref:data_for(rs)):
            for key in spec.KEYS:
                train,held = spec.training.cohort.split_rows(views[key[0]],key[1])
                src = source_cell(train,held,panels,model,key[2],key[3])
                cell = spec.analyze_cell(train,held,panels,model,{},src,key[2],key[3])
                results[key] = {'protocol':spec.EXPERIMENT_PROTOCOL,'contract':spec.contract(),'key':list(key),'cell':cell}
        report = spec.summarize(results,views)
        self.assertEqual(report['reproduction'],{'choice_comparisons':345600,'changed_choices':0,
            'score_comparisons':691200,'max_abs_difference':0.})
        self.assertEqual(report['network_updates'],0)
        self.assertEqual(report['focus']['key'],list(spec.FOCUS_KEY))
        self.assertTrue(all(v['scopes']['held_other']['prediction_cases']==480 for v in report['views'].values()))
        results[spec.KEYS[0]]['cell']['levels']['original']['mse'] = '3'
        with self.assertRaisesRegex(ValueError,'rank selection changed'): spec.summarize(results,views)

    def test_metric_algebra_contract_and_scheduler_scope(self):
        self.assertEqual(spec.contract(),json.loads(json.dumps(spec.contract())))
        rows = old.fixtures.native_fixture.bank_fixture()[0]['training_data'][:6]
        panels,model = old.fixtures.old.panels_for(rows),old.model_fixture()
        src = list(reversed(source_cell(rows,rows,panels,model,'first_last',0)['scopes']['train_same']))
        data = learning.records(rows,joint.physical_values(rows,panels,(0,1,2,3)),learning.predict(data_for(rows),model),src)
        metrics = spec.metrics(data)
        self.assertTrue(all(e['cost']==0. and e['wait_min']==0. for e in metrics['gates']['raw']['selected']['effects'].values()))
        for kind in joint.ANCHORS:
            for mode in learning.lowrank.MODES:
                e = metrics['gates'][kind][mode]['effects']
                for name in ('cost','wait_min'):
                    self.assertAlmostEqual(e['cost_on_raw'][name]-e['cost_on_physical'][name],e['interaction'][name])
                    self.assertAlmostEqual(e['total'][name],e['cost_on_physical'][name]+e['gate_on_base'][name]+e['interaction'][name])
        for key in (spec.KEYS[0],(None,None,None,None)):
            task = task_specification('unit_cost_gate',*key)
            self.assertEqual((task['cpu'],task['ram_mb']),(1,3072))
            self.assertIsNone(task['require_node'])
            self.assertEqual(task['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
            self.assertNotIn('_raw',task['result_dir'])


if __name__=='__main__': unittest.main()
