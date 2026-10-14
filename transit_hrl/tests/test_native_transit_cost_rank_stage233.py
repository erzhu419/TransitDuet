import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_cost_rank_selection as learning
from scripts import run_native_transit_cost_rank_stage233 as spec
from scripts.submit_native_transit_cost_rank_stage233_scheduleurm import task_specification
from tests import test_native_transit_cost_gate_stage232 as old

joint = learning.joint


def source_cell(train,held,panels,model,partition,direction):
    a,b = spec.PARTITIONS[partition][direction],spec.PARTITIONS[partition][1-direction]
    predictions = {name:learning.base.predict(old.data_for(rows),model) for name,rows in (('train',train),('held',held))}
    output = {'scopes':{}}
    for scope in spec.SCOPES:
        rows = train if scope.startswith('train') else held
        data = predictions['train' if scope.startswith('train') else 'held']
        values = joint.physical_values(rows,panels,a if scope.endswith('same') else b)
        src = []
        for i,r in enumerate(rows):
            j = int(data['raw_choices'][i])
            src.append({'raw':{'selected':j,'cost':float(values[i,j,0]),'wait':float(values[i,j,1]),
                'changed':j!=r['original_choice'],'unsafe':bool(values[i,j,1]>0)}})
        output['scopes'][scope] = list(reversed(learning.base.records(rows,values,data,src)))
    return output


def fake_collect(rows,panels,indices,reused):
    groups = {(r['root'],r['scene_seed']) for r in rows}
    ledger,curves = [],[]
    for group,_,held in joint.binding.policy.prefix_folds(rows):
        gs = sorted(map(list,groups-{group}))
        for direction in (0,1):
            a,b = list(indices[2*direction:2*direction+2]),list(indices[2*(1-direction):2*(1-direction)+2])
            fit = {'training_states':54,'training_groups':gs,'steps':0,'solver':'training_pair_TSVD',
                'paired_rows':270,'feature_count':198,'scales':[1.,1.],
                'arms':{'joint':{'loss_ledger':[{'level':l} for l in learning.lowrank.LEVELS]}}}
            ledger.append({'held_group':list(group),'direction':direction,'training_draws':a,'validation_draws':b,'held_states':len(held),
                'reference':{'draws':a,'training_states':54,'raw_steps':600,'raw_parameters':3393,'training_groups':gs,'wait_training_groups':gs},
                'fits':{k:{**copy.deepcopy(fit),'anchor':k} for k in joint.ANCHORS}})
            curves.append({l:{'error_sum':.1 if l=='8' else 1.,'error_cases':30,'changed_count':0 if l=='zero' else 1,
                'by_root':{str(group[0]):{'cases':6,'cost_sum':0. if l=='zero' else .06,
                    'wait_sum_min':0. if l=='zero' else .06,'unsafe_delta_count':0}}} for l in learning.lowrank.LEVELS})
    return {k:learning.select(curves,ledger,k) for k in joint.ANCHORS},ledger,{'max_abs_difference':0.,'changed_choices':0}


class CostRankTest(unittest.TestCase):
    def test_raw_gate_action_scoring_and_wait_worsening_constraint_input(self):
        rows = [{'root':397,'original_choice':0}]
        data = old.data_for(rows)
        data['raw_logits'][0,2] = -1.
        model = old.old.model_fixture()
        values = np.zeros((1,6,2))
        values[0,1],values[0,2] = [-.01,-.01],[-.05,.02]
        for kind in joint.ANCHORS:
            s = learning.stats(data,values,model['paths'][kind],'8')
            self.assertEqual(s['changed_count'],1)
            np.testing.assert_allclose([s['by_root']['397']['cost_sum'],s['by_root']['397']['wait_sum_min']],[-.04,.03])
            self.assertEqual(s['by_root']['397']['unsafe_delta_count'],1)
            zero = learning.stats(data,values,model['paths'][kind],'zero')
            self.assertEqual(zero['changed_count'],0)
            self.assertEqual(zero['by_root']['397']['unsafe_delta_count'],0)

    def test_shared_solver_cost_column_unchanged_and_wait_residual_unused(self):
        rows,data,values = old.old.fixtures.toy_data()
        data['raw_logits'] = np.zeros((2,6))
        data['raw_choices'] = joint.binding.policy.selected(data['heads'][...,0],data['raw_logits'],data['originals'])
        for kind in joint.ANCHORS:
            original,_,_ = learning.lowrank.fit_path(rows,data,values,kind)
            fitted,fit,curves = learning.fit_path(rows,data,values,kind,validation=(data,values))
            altered = values.copy()
            altered[...,1] *= 7.
            other,_,_ = learning.fit_path(rows,data,altered,kind)
            for level in learning.lowrank.LEVELS:
                np.testing.assert_array_equal(fitted['weights'][level],original['weights'][level])
                before = learning.lowrank.increments(data,fitted,level,'joint')[...,0]
                after = learning.lowrank.increments(data,other,level,'joint')[...,0]
                np.testing.assert_array_equal(before,after)
                self.assertEqual(curves[level],learning.stats(data,values,fitted,level))
            self.assertEqual(fit['steps'],0)

    def test_mean_vs_safe_selection_cost_mse_ties_and_raw_hold(self):
        ledger = [{'held_group':[r,0],'held_states':6,'fits':{'original':{'scales':[1.,1.]}}} for r in (397,401)]
        costs = {'zero':0.,'1':-.1,'3':-.2,'8':-.3,'16':-.3}
        curves = [{l:{'error_sum':.1 if l=='16' else 1.,'error_cases':30,'changed_count':0 if l=='zero' else 1,
            'by_root':{str(r):{'cases':6,'cost_sum':6*costs[l],'wait_sum_min':6*costs[l],
                'unsafe_delta_count':1 if l in ('8','16') and r==401 else 0}}} for l in learning.lowrank.LEVELS} for r in (397,401)]
        s = learning.select(curves,ledger,'original')
        self.assertEqual((s['mse_level'],s['mean_level'],s['safe_level']),('16','8','3'))
        self.assertEqual(s['curves']['8']['by_root']['401']['mean_delta_unsafe'],1/6)
        self.assertLess(s['curves']['3']['joint_score'],0.)
        for curve in curves:
            for l in learning.lowrank.LEVELS[1:]:
                for piece in curve[l]['by_root'].values(): piece['cost_sum'] = .6
        s = learning.select(curves,ledger,'original')
        self.assertIsNone(s['mean_level'])
        self.assertIsNone(s['safe_level'])
        curves[0]['zero']['changed_count'] = 1
        with self.assertRaisesRegex(ValueError,'zero residual'): learning.select(curves,ledger,'original')

    def test_clean_inner_references_exclude_prefix_and_future_labels(self):
        bank = old.old.fixtures.native_fixture.bank_fixture()[0]['training_data']
        groups = sorted({(r['root'],r['scene_seed']) for r in bank})[:2]
        rows = [r for r in bank if (r['root'],r['scene_seed']) in groups]
        fold = list(joint.binding.policy.prefix_folds(rows))[0]
        train,held = [rows[i] for i in fold[1]],[rows[i] for i in fold[2]]
        panels = old.old.fixtures.old.panels_for(rows)
        with (patch.object(joint.binding.learned.neural,'STEPS',3),
                patch.object(joint.binding.policy,'prefix_folds',return_value=[fold])):
            ds = [joint.causal_data(held,joint.fit_reference(train,panels,a)[0]) for a in ((0,1),(2,3))]
            reused = {'originals':np.concatenate([d['originals'] for d in ds]),'roots':np.concatenate([d['roots'] for d in ds]),
                'costs':np.concatenate([d['heads'][...,0] for d in ds]),'state_conditioned_pair':np.concatenate([d['raw_logits'] for d in ds])}
            first,ledger,reproduction = learning.collect(rows,panels,(0,1,2,3),reused)
            changed = copy.deepcopy(panels)
            for r in rows: changed[joint.binding.learned.state_key(r)]['values'][4:,(r['original_choice']+1)%6] += 100.
            second,other,_ = learning.collect(rows,changed,(0,1,2,3),reused)
            altered = copy.deepcopy(panels)
            for r in held: altered[joint.binding.learned.state_key(r)]['values'][:4,(r['original_choice']+1)%6] += 100.
            third,last,_ = learning.collect(rows,altered,(0,1,2,3),reused)
        self.assertEqual(first,second)
        self.assertEqual(ledger,other)
        self.assertEqual(ledger,last)
        self.assertNotEqual(first['original']['curves'],third['original']['curves'])
        self.assertEqual(reproduction,{'max_abs_difference':0.,'changed_choices':0})
        for f in ledger:
            self.assertNotIn(list(fold[0]),f['reference']['training_groups'])
            self.assertFalse(set(f['training_draws']) & set(f['validation_draws']))

    def test_outer_held_labels_cannot_change_selection_or_frozen_paths(self):
        rows,registered = old.old.fixtures.cohort.fixture()
        views = spec.training.cohort.source.matched_views(rows,registered)
        train,held = spec.training.cohort.split_rows(views['mixed_a'],0)
        panels,model = old.old.fixtures.old.panels_for(rows),old.old.model_fixture()
        changed = copy.deepcopy(panels)
        for r in train: changed[joint.binding.learned.state_key(r)]['values'][4:,(r['original_choice']+1)%6] += 100.
        for r in held: changed[joint.binding.learned.state_key(r)]['values'][:,(r['original_choice']+1)%6] += 100.
        sources = [source_cell(train,held,p,model,'first_last',0) for p in (panels,changed)]
        captured = []
        def collect(rs,ps,indices,reused):
            captured.append(joint.physical_values(rs,ps,indices).tolist())
            return fake_collect(rs,ps,indices,reused)
        before_model = copy.deepcopy(model)
        with patch.object(learning,'collect',side_effect=collect),patch.object(joint,'causal_data',side_effect=lambda rs,ref:old.data_for(rs)):
            before,after = [spec.fit_cell(train,held,p,model,{},s,{},'first_last',0) for p,s in zip((panels,changed),sources)]
        self.assertEqual(captured[0],captured[1])
        self.assertEqual(before['selection'],after['selection'])
        self.assertEqual(model,before_model)
        self.assertNotEqual(before['scopes']['held_other'],after['scopes']['held_other'])
        for scope in spec.SCOPES:
            for b,a in zip(before['scopes'][scope],after['scopes'][scope]):
                self.assertEqual({n:d['selected'] for n,d in b['decisions'].items()},{n:d['selected'] for n,d in a['decisions'].items()})

    def test_full_report_counts_constraints_and_leakage_rejection(self):
        rows,registered = old.old.fixtures.cohort.fixture()
        views = spec.training.cohort.source.matched_views(rows,registered)
        panels,model = old.old.fixtures.old.panels_for(rows),old.old.model_fixture()
        results = {}
        with patch.object(learning,'collect',side_effect=fake_collect),patch.object(joint,'causal_data',side_effect=lambda rs,ref:old.data_for(rs)):
            for key in spec.KEYS:
                train,held = spec.training.cohort.split_rows(views[key[0]],key[1])
                source = source_cell(train,held,panels,model,key[2],key[3])
                cell = spec.fit_cell(train,held,panels,model,{},source,{},key[2],key[3])
                results[key] = {'protocol':spec.EXPERIMENT_PROTOCOL,'contract':spec.contract(),'key':list(key),'cell':cell}
        report = spec.summarize(results,views)
        self.assertEqual(report['control_reproduction'],{'choice_comparisons':115200,'changed_choices':0,
            'score_comparisons':230400,'max_abs_difference':0.})
        self.assertEqual(report['network_updates'],384000)
        self.assertTrue(all(v['scopes']['held_other']['prediction_cases']==480 for v in report['views'].values()))
        self.assertEqual(report['focus']['key'],list(spec.FOCUS_KEY))
        result = copy.deepcopy(results[spec.KEYS[0]])
        f = result['cell']['inner_fits'][0]
        f['reference']['training_groups'].append(f['held_group'])
        with self.assertRaisesRegex(ValueError,'leaked prefix/future'): spec.qualify(result,spec.KEYS[0],views)
        result = copy.deepcopy(results[spec.KEYS[0]])
        s = result['cell']['selection']['original']
        s['safe_level'] = '8'
        with self.assertRaisesRegex(ValueError,'inner constraints'): spec.qualify(result,spec.KEYS[0],views)

    def test_contract_and_scheduler_only_linux_cpus_aggregate_pull(self):
        self.assertEqual(spec.contract(),json.loads(json.dumps(spec.contract())))
        self.assertEqual(spec.contract()['budget']['new_native_episodes'],0)
        for key in (spec.KEYS[0],(None,None,None,None)):
            task = task_specification('unit_cost_rank',*key)
            self.assertEqual((task['cpu'],task['ram_mb']),(1,3072))
            self.assertIsNone(task['require_node'])
            self.assertEqual(task['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
            self.assertNotIn('_raw',task['result_dir'])

    def test_contract_parent_is_read_once_and_archived_protocol_unchanged(self):
        archived = json.loads((spec.ROOT/'results'/spec.SOURCE_RUN/'preregistration.json').read_text())
        self.assertEqual(spec.previous.contract(),archived['contract'])
        with patch.object(spec.previous,'contract',wraps=spec.previous.contract) as parent:
            first = spec.contract()
            self.assertEqual(parent.call_count,1)
        first['budget']['fit_tasks'] = 0
        self.assertEqual(spec.contract()['budget']['fit_tasks'],32)


if __name__=='__main__': unittest.main()
