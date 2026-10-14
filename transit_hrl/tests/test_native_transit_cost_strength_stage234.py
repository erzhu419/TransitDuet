import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_cost_strength as learning
from scripts import run_native_transit_cost_strength_stage234 as spec
from scripts.submit_native_transit_cost_strength_stage234_scheduleurm import task_specification
from tests import test_native_transit_cost_rank_stage233 as old

joint = learning.joint


def source_cell(train,held,panels,model,partition,direction):
    source = old.source_cell(train,held,panels,model,partition,direction)
    with patch.object(learning.base,'collect',side_effect=old.fake_collect),patch.object(joint,'causal_data',side_effect=lambda rs,ref:old.old.data_for(rs)):
        return spec.previous.fit_cell(train,held,panels,model,{},source,{},partition,direction)


def fake_collect(rows,panels,indices,reused):
    unit,ledger,repro = old.fake_collect(rows,panels,indices,reused)
    selected = {}
    for a,s in unit.items():
        curves = {k:copy.deepcopy(s['curves'][c['level']]) for k,c in learning.CANDIDATES.items()}
        selected[a] = {'mse_level':'8@0.25','mean_level':None,'safe_level':None,'curves':curves,'source_scales':s['source_scales']}
    return selected,ledger,repro,unit


class CostStrengthTest(unittest.TestCase):
    def test_interpolation_endpoints_zero_aliases_and_changed_action(self):
        rows = [{'root':397,'original_choice':0}]
        data = old.old.data_for(rows)
        data['raw_logits'][0,2] = -1.
        model = old.old.old.model_fixture()
        for a in joint.ANCHORS:
            path,anchor = model['paths'][a],joint.anchors(data,a)
            baseline = joint.relative(data['heads'],anchor)[...,0]
            for level in learning.lowrank.LEVELS:
                full = learning.lowrank.increments(data,path,level,'joint')[...,0]
                np.testing.assert_array_equal(learning.base.costs(data,path,level,1.),full)
                np.testing.assert_array_equal(learning.base.costs(data,path,level,0.),baseline)
                chosen = joint.binding.policy.selected(learning.base.costs(data,path,level,0.),data['raw_logits'],anchor)
                np.testing.assert_array_equal(chosen,data['raw_choices'])
            weak = learning.base.costs(data,path,'8',.25)
            np.testing.assert_allclose(weak,baseline+.25*(learning.base.costs(data,path,'8')-baseline))
            self.assertEqual(int(joint.binding.policy.selected(weak,data['raw_logits'],anchor)[0]),1)
            self.assertEqual(int(joint.binding.policy.selected(learning.base.costs(data,path,'8'),data['raw_logits'],anchor)[0]),2)
            for alpha in learning.ALPHAS:
                np.testing.assert_array_equal(learning.base.costs(data,path,'zero',alpha),baseline)
        self.assertEqual(len(learning.CANDIDATES),17)

    def test_lower_rank_weaker_strength_ties_and_risk_constraint(self):
        ledger = [{'held_group':[r,0],'held_states':6,'fits':{'raw':{'scales':[1.,1.]}}} for r in (397,401)]
        strongest = {'1@0.25','1@0.5','3@0.25'}
        curves = [{k:{'error_sum':.1 if k=='8@0.25' else 1.,'error_cases':30,'changed_count':0 if k=='zero' else 1,
            'by_root':{str(r):{'cases':6,'cost_sum':0. if k=='zero' else (-1.8 if k in strongest else -.6),
                'wait_sum_min':0. if k=='zero' else (-1.8 if k in strongest else -.6),
                'unsafe_delta_count':1 if k.startswith('1@') and r==401 else 0}}} for k in learning.CANDIDATES} for r in (397,401)]
        s = learning.base.select(curves,ledger,'raw',levels=tuple(learning.CANDIDATES))
        self.assertEqual((s['mse_level'],s['mean_level'],s['safe_level']),('8@0.25','1@0.25','3@0.25'))
        unit = [{l:c[k] for l,k in learning.UNIT_KEYS.items()} for c in curves]
        self.assertEqual(set(learning.base.select(unit,ledger,'raw')['curves']),set(learning.lowrank.LEVELS))
        for curve in curves:
            for k in learning.CANDIDATES:
                if k!='zero':
                    for x in curve[k]['by_root'].values(): x['cost_sum'] = .6
        held = learning.base.select(curves,ledger,'raw',levels=tuple(learning.CANDIDATES))
        self.assertIsNone(held['mean_level'])
        self.assertIsNone(held['safe_level'])

    def test_clean_crossfit_unit_strength_reproduction_and_future_blindness(self):
        bank = old.old.old.fixtures.native_fixture.bank_fixture()[0]['training_data']
        gs = sorted({(r['root'],r['scene_seed']) for r in bank})[:2]
        rows = [r for r in bank if (r['root'],r['scene_seed']) in gs]
        fold = list(joint.binding.policy.prefix_folds(rows))[0]
        train,held = [rows[i] for i in fold[1]],[rows[i] for i in fold[2]]
        panels = old.old.old.fixtures.old.panels_for(rows)
        with (patch.object(joint.binding.learned.neural,'STEPS',3),
                patch.object(joint.binding.policy,'prefix_folds',return_value=[fold])):
            ds = [joint.causal_data(held,joint.fit_reference(train,panels,a)[0]) for a in ((0,1),(2,3))]
            reused = {'originals':np.concatenate([d['originals'] for d in ds]),'roots':np.concatenate([d['roots'] for d in ds]),
                'costs':np.concatenate([d['heads'][...,0] for d in ds]),'state_conditioned_pair':np.concatenate([d['raw_logits'] for d in ds])}
            prior,old_ledger,old_repro = learning.base.collect(rows,panels,(0,1,2,3),reused)
            first,ledger,repro,unit = learning.collect(rows,panels,(0,1,2,3),reused)
            changed = copy.deepcopy(panels)
            for r in rows: changed[joint.binding.learned.state_key(r)]['values'][4:,(r['original_choice']+1)%6] += 100.
            second,other,_,_ = learning.collect(rows,changed,(0,1,2,3),reused)
            altered = copy.deepcopy(panels)
            for r in held: altered[joint.binding.learned.state_key(r)]['values'][:4,(r['original_choice']+1)%6] += 100.
            third,last,_,_ = learning.collect(rows,altered,(0,1,2,3),reused)
        self.assertEqual(unit,prior)
        self.assertEqual(ledger,old_ledger)
        self.assertEqual(repro,old_repro)
        self.assertEqual(first,second)
        self.assertEqual(ledger,other)
        self.assertEqual(ledger,last)
        self.assertNotEqual(first['raw']['curves'],third['raw']['curves'])

    def test_outer_held_labels_blind_frozen_paths_and_structured_join(self):
        rows,registered = old.old.old.fixtures.cohort.fixture()
        views = spec.training.cohort.source.matched_views(rows,registered)
        train,held = spec.training.cohort.split_rows(views['mixed_b'],0)
        panels,model = old.old.old.fixtures.old.panels_for(rows),old.old.old.model_fixture()
        changed = copy.deepcopy(panels)
        for r in train: changed[joint.binding.learned.state_key(r)]['values'][4:,(r['original_choice']+1)%6] += 100.
        for r in held: changed[joint.binding.learned.state_key(r)]['values'][:,(r['original_choice']+1)%6] += 100.
        sources = [source_cell(train,held,p,model,'first_last',0) for p in (panels,changed)]
        captured = []
        def collect(rs,ps,indices,reused):
            captured.append(joint.physical_values(rs,ps,indices).tolist())
            return fake_collect(rs,ps,indices,reused)
        unchanged_model = copy.deepcopy(model)
        with patch.object(learning,'collect',side_effect=collect),patch.object(joint,'causal_data',side_effect=lambda rs,ref:old.old.data_for(rs)):
            before,after = [spec.fit_cell(train,held,p,model,{},s,{},'first_last',0) for p,s in zip((panels,changed),sources)]
        self.assertEqual(captured[0],captured[1])
        self.assertEqual(before['selection'],after['selection'])
        self.assertEqual(model,unchanged_model)
        self.assertNotEqual(before['scopes']['held_other'],after['scopes']['held_other'])
        for scope in spec.SCOPES:
            for b,a in zip(before['scopes'][scope],after['scopes'][scope]):
                self.assertEqual({n:d['selected'] for n,d in b['decisions'].items()},{n:d['selected'] for n,d in a['decisions'].items()})

    def test_complete_report_counts_and_unit_strength_mismatch_rejection(self):
        rows,registered = old.old.old.fixtures.cohort.fixture()
        views = spec.training.cohort.source.matched_views(rows,registered)
        panels,model = old.old.old.fixtures.old.panels_for(rows),old.old.old.model_fixture()
        results = {}
        with patch.object(learning,'collect',side_effect=fake_collect),patch.object(joint,'causal_data',side_effect=lambda rs,ref:old.old.data_for(rs)):
            for key in spec.KEYS:
                train,held = spec.training.cohort.split_rows(views[key[0]],key[1])
                source = source_cell(train,held,panels,model,key[2],key[3])
                cell = spec.fit_cell(train,held,panels,model,{},source,{},key[2],key[3])
                results[key] = {'protocol':spec.EXPERIMENT_PROTOCOL,'contract':spec.contract(),'key':list(key),'cell':cell}
            bad_source = copy.deepcopy(source)
            bad_source['selection']['raw']['mean_level'] = '8'
            with self.assertRaisesRegex(ValueError,'Unit-strength inner fits or selections'):
                spec.fit_cell(train,held,panels,model,{},bad_source,{},key[2],key[3])
        report = spec.summarize(results,views)
        self.assertEqual(report['control_reproduction'],{'choice_comparisons':161280,'changed_choices':0,
            'score_comparisons':322560,'max_abs_difference':0.})
        self.assertEqual(report['unit_strength_reproduction'],{'inner_fits':640,'anchor_selections':64,'differences':0})
        self.assertEqual(report['inner_candidate_evaluations'],21760)
        self.assertTrue(all(v['scopes']['held_other']['prediction_cases']==480 for v in report['views'].values()))
        results[spec.KEYS[0]]['cell']['selection']['raw']['safe_level'] = '8@0.25'
        with self.assertRaisesRegex(ValueError,'inner constraints'): spec.summarize(results,views)

    def test_raw_hold_variants_and_contract_scheduler_budget(self):
        rows = [{'root':397,'original_choice':0}]
        model = old.old.old.model_fixture()
        selection = {a:{'mse_level':'zero','mean_level':None,'safe_level':None} for a in joint.ANCHORS}
        unit = {a:{'mse_level':'zero','mean_level':None,'safe_level':None} for a in joint.ANCHORS}
        p = learning.predict(old.old.data_for(rows),model,selection,unit)
        for kind in joint.ANCHORS:
            for mode in learning.NEW_MODES: np.testing.assert_array_equal(p['choices'][f'cost_strength_{kind}_{mode}'],p['choices']['raw'])
        self.assertEqual(spec.contract(),json.loads(json.dumps(spec.contract())))
        self.assertEqual(spec.contract()['budget']['residual_SVDs'],1280)
        self.assertEqual(spec.contract()['budget']['new_native_episodes'],0)
        with patch.object(spec.previous,'contract',wraps=spec.previous.contract) as parent:
            spec.contract()
            self.assertEqual(parent.call_count,1)
        for key in (spec.KEYS[0],(None,None,None,None)):
            task = task_specification('unit_cost_strength',*key)
            self.assertEqual((task['cpu'],task['ram_mb']),(1,3072))
            self.assertIsNone(task['require_node'])
            self.assertEqual(task['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
            self.assertNotIn('_raw',task['result_dir'])


if __name__=='__main__': unittest.main()
