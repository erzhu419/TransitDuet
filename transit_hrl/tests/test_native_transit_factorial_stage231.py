import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_lowrank_factorial as learning
from scripts import run_native_transit_factorial_stage231 as spec
from scripts.submit_native_transit_factorial_stage231_scheduleurm import task_specification
from tests import test_native_transit_joint_advantage_stage226 as fixtures

joint = learning.joint


def model_fixture():
    weight = np.zeros((2,198))
    weight[:,0] = -.2
    return {'paths':{a:{'anchor':a,'scales':[1.,1.],'weights':{l:(np.zeros_like(weight) if l=='zero' else weight).tolist()
        for l in learning.lowrank.LEVELS}} for a in joint.ANCHORS},
        'selection':{a:{'joint':{'mse_level':'8','joint_level':'8' if a=='original' else None}} for a in joint.ANCHORS}}


def data_for(rows):
    n = len(rows)
    originals = np.asarray([r['original_choice'] for r in rows])
    x = np.zeros((n,6,210))
    x[:,2,0] = 1.
    heads = np.broadcast_to(np.asarray([[0.,0.],[-.2,-.1],[-.1,.1],[.5,.5],[.5,.5],[.5,.5]]),(n,6,2)).copy()
    return {'x':x,'heads':joint.relative(heads,originals),'originals':originals,'roots':np.asarray([r['root'] for r in rows]),
        'raw_choices':np.ones(n,dtype=int)}


def source_cell(train,held,panels,model,partition,direction):
    a,b = spec.PARTITIONS[partition][direction],spec.PARTITIONS[partition][1-direction]
    output = {'scopes':{}}
    for scope in spec.SCOPES:
        rows = train if scope.startswith('train') else held
        p = learning.predict(data_for(rows),model)
        values = joint.physical_values(rows,panels,a if scope.endswith('same') else b)
        saved = []
        for i,r in enumerate(rows):
            def decision(j): return {'selected':int(j),'cost':float(values[i,j,0]),'wait':float(values[i,j,1]),
                'changed':bool(j!=r['original_choice']),'unsafe':bool(values[i,j,1]>0)}
            decisions = {'raw':decision(p['raw_choices'][i])}
            for kind in joint.ANCHORS:
                for mode in learning.lowrank.MODES[:5]:
                    for arm in ('base','wait','joint'):
                        name = f'lowrank_{kind}_joint_zero' if arm=='base' else f'lowrank_{kind}_{arm}_{mode}'
                        decisions[name] = decision(p['choices'][kind][mode][arm][i])
            saved.append({**{k:r[k] for k in ('root','scenario','scene_seed','probe_index','original_choice')},'decisions':decisions})
        output['scopes'][scope] = list(reversed(saved))
    return output


class FactorialTest(unittest.TestCase):
    def test_identical_rank_predictor_toggles_and_genuine_interaction(self):
        rows = [{'root':397,'original_choice':0}]
        p = learning.predict(data_for(rows),model_fixture())
        self.assertEqual({a:int(v[0]) for a,v in p['choices']['original']['rank8'].items()},
            {'base':1,'cost':1,'wait':1,'joint':2})
        self.assertEqual(int(p['menus']['original']['rank8']['added'][0]),1)
        self.assertEqual(int(p['menus']['original']['rank8']['removed'][0]),0)
        d = {'base':{'cost':-.05,'wait':-.02},'cost':{'cost':-.05,'wait':-.02},
            'wait':{'cost':-.05,'wait':-.02},'joint':{'cost':-.2,'wait':-.04}}
        effects = learning.effects(d)
        np.testing.assert_allclose(effects['interaction'],[-.15,-.02])
        np.testing.assert_allclose(effects['joint'],effects['cost']+effects['wait']+effects['interaction'])

    def test_zero_anchor_first_ties_raw_hold_and_joint_rank_binding(self):
        rows = [{'root':397,'original_choice':0}]
        data,model = data_for(rows),model_fixture()
        p = learning.predict(data,model)
        self.assertEqual(set(int(v[0]) for v in p['choices']['original']['zero'].values()),{1})
        self.assertEqual(set(int(v[0]) for v in p['choices']['raw']['selected'].values()),{1})
        self.assertEqual(p['levels']['original']['mse'],'8')
        model['selection']['original']['wait'] = {'mse_level':'1','joint_level':'1'}
        self.assertEqual(learning.predict(data,model)['levels'],p['levels'])
        data['heads'][:] = 0.
        ties = learning.predict(data,model)
        self.assertEqual(set(int(v[0]) for v in ties['choices']['original']['zero'].values()),{0})
        self.assertEqual(set(int(v[0]) for v in ties['choices']['raw']['zero'].values()),{1})

    def test_frozen_choices_and_physical_scores_reject_mismatched_artifacts(self):
        rows = fixtures.native_fixture.bank_fixture()[0]['training_data'][:6]
        panels,model = fixtures.old.panels_for(rows),model_fixture()
        src = source_cell(rows,rows,panels,model,'first_last',0)
        old = list(reversed(src['scopes']['train_same']))
        p = learning.predict(data_for(rows),model)
        self.assertEqual(learning.reproduce_choices(p,old),{'comparisons':186,'changed_choices':0})
        values = joint.physical_values(rows,panels,(0,1,2,3))
        self.assertEqual(learning.reproduce_scores(values,p,old),{'comparisons':372,'max_abs_difference':0.})
        broken = copy.deepcopy(old)
        broken[0]['decisions']['lowrank_original_wait_rank8']['selected'] = 5
        with self.assertRaisesRegex(ValueError,'choices changed'): learning.reproduce_choices(p,broken)
        broken = copy.deepcopy(old)
        broken[0]['decisions']['raw']['wait'] += 1.
        with self.assertRaisesRegex(ValueError,'panels or future indices'): learning.reproduce_scores(values,p,broken)

    def test_label_blind_analysis_zero_fits_structured_join_and_future_scores(self):
        rows = fixtures.native_fixture.bank_fixture()[0]['training_data'][:12]
        train,held = rows[:6],rows[6:]
        panels,model = fixtures.old.panels_for(rows),model_fixture()
        src = source_cell(train,held,panels,model,'first_last',0)
        changed = copy.deepcopy(panels)
        for r in rows:
            p = changed[joint.binding.learned.state_key(r)]['values']
            for j in range(6):
                if j!=r['original_choice']: p[4:,j] += 100.
        updated = source_cell(train,held,changed,model,'first_last',0)
        with (patch.object(joint,'causal_data',side_effect=lambda rs,ref:data_for(rs)),
                patch.object(joint,'fit_reference',side_effect=AssertionError('No refit')),
                patch.object(learning.lowrank,'fit_path',side_effect=AssertionError('No SVD fit'))):
            first = spec.analyze_cell(train,held,panels,model,{},src,'first_last',0)
            second = spec.analyze_cell(train,held,changed,model,{},updated,'first_last',0)
        self.assertEqual(first['levels'],second['levels'])
        self.assertEqual(first['choice_reproduction'],second['choice_reproduction'])
        for scope in spec.SCOPES:
            for before,after in zip(first['scopes'][scope],second['scopes'][scope]):
                for a in joint.ANCHORS:
                    for mode in learning.lowrank.MODES:
                        self.assertEqual({k:v['selected'] for k,v in before['factorial'][a][mode]['decisions'].items()},
                            {k:v['selected'] for k,v in after['factorial'][a][mode]['decisions'].items()})
        self.assertNotEqual(first['scopes']['held_other'],second['scopes']['held_other'])
        self.assertEqual(first['score_reproduction']['max_abs_difference'],0.)

    def test_complete_report_real_counts_focus_and_selection_mismatch_rejection(self):
        rows,registered = fixtures.cohort.fixture()
        views,panels = spec.training.cohort.source.matched_views(rows,registered),fixtures.old.panels_for(rows)
        results,model = {},model_fixture()
        with patch.object(joint,'causal_data',side_effect=lambda rs,ref:data_for(rs)):
            for key in spec.KEYS:
                view,fold,partition,direction = key
                train,held = spec.training.cohort.split_rows(views[view],fold)
                src = source_cell(train,held,panels,model,partition,direction)
                cell = spec.analyze_cell(train,held,panels,model,{},src,partition,direction)
                results[key] = {'protocol':spec.EXPERIMENT_PROTOCOL,'contract':spec.contract(),'key':list(key),'cell':cell}
        report = spec.summarize(results,views)
        self.assertEqual(report['choice_reproduction'],{'comparisons':119040,'changed_choices':0})
        self.assertEqual(report['score_reproduction'],{'comparisons':476160,'max_abs_difference':0.})
        self.assertEqual(report['network_updates'],0)
        self.assertEqual(report['focus']['key'],list(spec.FOCUS_KEY))
        self.assertTrue(all(x['scopes']['held_other']['prediction_cases']==480 for x in report['views'].values()))
        results[spec.KEYS[0]]['cell']['levels']['original']['mse'] = '3'
        with self.assertRaisesRegex(ValueError,'rank selection changed'): spec.summarize(results,views)

    def test_additive_metric_algebra_empty_hold_effects_and_scheduler(self):
        self.assertEqual(spec.contract(),json.loads(json.dumps(spec.contract())))
        rows = fixtures.native_fixture.bank_fixture()[0]['training_data'][:6]
        panels,model = fixtures.old.panels_for(rows),model_fixture()
        src = list(reversed(source_cell(rows,rows,panels,model,'first_last',0)['scopes']['train_same']))
        p = learning.predict(data_for(rows),model)
        recorded = learning.records(rows,joint.physical_values(rows,panels,(0,1,2,3)),p,src)
        summary = spec.metrics(recorded)
        self.assertEqual(summary['factorial']['raw']['selected']['nonzero_cases'],0)
        self.assertTrue(all(e['cost']==0. and e['wait_min']==0. for e in summary['factorial']['raw']['selected']['effects'].values()))
        for a in joint.ANCHORS:
            for mode in learning.lowrank.MODES:
                e = summary['factorial'][a][mode]['effects']
                for m in ('cost','wait_min'):
                    self.assertAlmostEqual(e['joint'][m],e['cost'][m]+e['wait'][m]+e['interaction'][m])
        for key in (spec.KEYS[0],(None,None,None,None)):
            task = task_specification('unit_factorial',*key)
            self.assertEqual((task['cpu'],task['ram_mb']),(1,3072))
            self.assertIsNone(task['require_node'])
            self.assertEqual(task['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
            self.assertNotIn('_raw',task['result_dir'])


if __name__=='__main__': unittest.main()
