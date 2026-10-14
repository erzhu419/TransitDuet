import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_lowrank_joint as learning
from scripts import run_native_transit_lowrank_joint_stage230 as spec
from scripts.submit_native_transit_lowrank_joint_stage230_scheduleurm import task_specification
from tests import test_native_transit_joint_advantage_stage226 as fixtures

joint = learning.joint


def zero_prediction(rows):
    p = fixtures.zero_prediction(rows)
    n = len(rows)
    p['choices'].update({v:np.asarray([r['original_choice'] for r in rows]) for v in learning.VARIANTS})
    for name in learning.VARIANTS:
        if name.startswith('lowrank_') and not name.endswith('selected'):
            p['advantages'][name] = {'delta':np.zeros((n,6,2)),'scales':[1.,1.],'anchor':np.asarray([r['original_choice'] for r in rows])}
    return p


class LowrankJointTest(unittest.TestCase):
    def test_shared_wait_frozen_cost_exact_anchor_and_projection_solution(self):
        rows,data,values = fixtures.toy_data()
        for kind in joint.ANCHORS:
            model,fit,curves = learning.fit_path(rows,data,values,kind,validation=(data,values))
            anchor = joint.anchors(data,kind)
            baseline = joint.relative(data['heads'],anchor)
            mask = np.arange(6)[None,:]!=anchor[:,None]
            x = joint.relative(data['x'][...,:198],anchor)[mask]
            y = joint.relative(values-data['heads'],anchor)[mask]/model['scales']
            u,s,vt = np.linalg.svd(x,full_matrices=False)
            for k,level in zip(learning.RANKS,learning.LEVELS):
                rank = min(k,fit['numerical_rank'])
                expected = vt[:rank].T@((u[:,:rank].T@y)/s[:rank,None])
                np.testing.assert_allclose(np.asarray(model['weights'][level]).T,expected,atol=1e-12)
                w,j = learning.increments(data,model,level,'wait'),learning.increments(data,model,level,'joint')
                np.testing.assert_array_equal(w[...,0],baseline[...,0])
                np.testing.assert_array_equal(w[...,1],j[...,1])
                np.testing.assert_array_equal(w[np.arange(2),anchor],np.zeros((2,2)))
                for arm in learning.ARMS:
                    entry = next(e for e in fit['arms'][arm]['loss_ledger'] if e['level']==level)
                    stats = curves[arm][level]
                    self.assertAlmostEqual(entry['training_mse'],stats['error_sum']/stats['error_cases'])
                    self.assertAlmostEqual(sum(stats['error_sum_by_head']),stats['error_sum'])
                    self.assertEqual(entry['cost_df'],0 if arm=='wait' else rank)
                    self.assertEqual(entry['wait_df'],rank)
            self.assertEqual(fit['steps'],0)
            self.assertEqual(fit['arms']['wait']['parameter_count'],198)
            self.assertEqual(fit['arms']['joint']['parameter_count'],396)

    def test_cost_target_does_not_change_wait_fit_given_shared_reference(self):
        rows,data,values = fixtures.toy_data()
        first,_,_ = learning.fit_path(rows,data,values,'raw')
        changed = values.copy()
        changed[...,0] *= 7.
        second,_,_ = learning.fit_path(rows,data,changed,'raw')
        for level in learning.LEVELS:
            before,after = learning.increments(data,first,level,'wait'),learning.increments(data,second,level,'wait')
            np.testing.assert_array_equal(before,after)
        self.assertFalse(np.allclose(learning.increments(data,first,'8','joint')[...,0],learning.increments(data,second,'8','joint')[...,0]))

    def test_training_rank_handles_feature_dependencies_and_lower_rank_ties(self):
        rows,data,values = fixtures.toy_data()
        rng = np.random.default_rng(230)
        data['x'][...,:198] = rng.normal(size=(2,6,2))@rng.normal(size=(2,198))
        _,fit,_ = learning.fit_path(rows,data,values,'original')
        self.assertEqual(fit['numerical_rank'],2)
        for arm in learning.ARMS:
            self.assertEqual([e['retained_rank'] for e in fit['arms'][arm]['loss_ledger']],[0,1,2,2,2])
        ledger = [{'held_group':[397,0],'held_states':6,'fits':{'raw':{'scales':[1.,1.]}}}]
        curves = [{l:{'error_sum':1.,'error_cases':10,'changed_count':1,'by_root':{'397':{'cases':6,'cost_sum':0. if l=='zero' else -.6,
            'wait_sum_min':-.6}}} for l in learning.LEVELS}]
        selected = learning.ridge.select(curves,ledger,'raw',levels=learning.LEVELS)
        self.assertEqual(selected['mse_level'],'zero')
        self.assertEqual(selected['joint_level'],'1')
        for c in curves:
            for l in learning.LEVELS: c[l]['by_root']['397']['cost_sum'] = .6
        self.assertIsNone(learning.ridge.select(curves,ledger,'raw',levels=learning.LEVELS)['joint_level'])

    def test_clean_prefix_future_selection_and_per_head_validation(self):
        bank = fixtures.native_fixture.bank_fixture()[0]['training_data']
        groups = sorted({(r['root'],r['scene_seed']) for r in bank})[:2]
        rows = [r for r in bank if (r['root'],r['scene_seed']) in groups]
        fold = list(joint.binding.policy.prefix_folds(rows))[0]
        train,held = [rows[i] for i in fold[1]],[rows[i] for i in fold[2]]
        panels = fixtures.old.panels_for(rows)
        with patch.object(joint.binding.learned.neural,'STEPS',3),patch.object(joint.binding.policy,'prefix_folds',return_value=[fold]):
            ds = [joint.causal_data(held,joint.fit_reference(train,panels,a)[0]) for a in ((0,1),(2,3))]
            reused = {'originals':np.concatenate([d['originals'] for d in ds]),'roots':np.concatenate([d['roots'] for d in ds]),
                'costs':np.concatenate([d['heads'][...,0] for d in ds]),'state_conditioned_pair':np.concatenate([d['raw_logits'] for d in ds])}
            first,ledger,reproduction = learning.collect(rows,panels,(0,1,2,3),reused)
            changed = copy.deepcopy(panels)
            for r in rows: changed[joint.binding.learned.state_key(r)]['values'][4:,(r['original_choice']+1)%6] += 100.
            second,other_ledger,_ = learning.collect(rows,changed,(0,1,2,3),reused)
            altered = copy.deepcopy(panels)
            for r in held: altered[joint.binding.learned.state_key(r)]['values'][:4,(r['original_choice']+1)%6] += 100.
            third,val_ledger,_ = learning.collect(rows,altered,(0,1,2,3),reused)
        self.assertEqual(first,second)
        self.assertEqual(ledger,other_ledger)
        self.assertEqual(ledger,val_ledger)
        self.assertNotEqual(first['raw']['wait']['curves'],third['raw']['wait']['curves'])
        self.assertEqual(reproduction,{'max_abs_difference':0.,'changed_choices':0})
        for f in ledger:
            self.assertNotIn(list(fold[0]),f['reference']['training_groups'])
            self.assertFalse(set(f['training_draws']) & set(f['validation_draws']))
        for kind in joint.ANCHORS:
            for l in learning.LEVELS:
                w,j = first[kind]['wait']['curves'][l],first[kind]['joint']['curves'][l]
                self.assertEqual(w['wait_mse'],j['wait_mse'])
                self.assertAlmostEqual(w['mse'],(w['cost_mse']+w['wait_mse'])/2)

    def test_causal_inference_all29_controls_and_raw_hold(self):
        rows = fixtures.native_fixture.bank_fixture()[0]['training_data'][:6]
        panels = fixtures.old.panels_for(rows)
        with patch.object(joint.binding.precision.base.base,'STEPS',3),patch.object(joint.binding.learned.neural,'STEPS',3),patch.object(joint,'CHECKPOINTS',(0,1,3)):
            bundle,_ = joint.binding.precision.fit_bundle(rows,panels,(0,1,2,3))
            bundle['best'] = {'center':bundle['ridge']['center'],'scale':bundle['ridge']['scale'],'coefficients':[0.]*98}
            selected = {a:{'mse_step':1,'joint_step':None} for a in joint.ANCHORS}
            nonlinear,_ = joint.fit_final(rows,panels,(0,1,2,3),bundle,selected)
            linear,_ = joint.fit_final(rows,panels,(0,1,2,3),bundle,selected,path_fitter=learning.ridge.linear.fit_path)
            ridge,_ = joint.fit_final(rows,panels,(0,1,2,3),bundle,{a:{'mse_level':'zero','joint_level':None} for a in joint.ANCHORS},path_fitter=learning.ridge.fit_path)
            selected = {a:{arm:{'mse_level':'3','joint_level':None} for arm in learning.ARMS} for a in joint.ANCHORS}
            lowrank,_ = joint.fit_final(rows,panels,(0,1,2,3),bundle,selected,path_fitter=learning.fit_path)
            cal = {m:{v:{'alpha':1.} for v in joint.binding.WAIT_VARIANTS} for m in ('risk','oof','final','joint')}
            causal = [{k:v for k,v in r.items() if k!='labels'} for r in rows]
            before = learning.ridge.predict(causal,ridge,linear,nonlinear,bundle,cal)
            after = learning.predict(causal,lowrank,ridge,linear,nonlinear,bundle,cal)
        self.assertEqual(set(after['choices']),set(learning.VARIANTS))
        for n in learning.ridge.VARIANTS: np.testing.assert_array_equal(before['choices'][n],after['choices'][n])
        for a in joint.ANCHORS:
            for arm in learning.ARMS:
                np.testing.assert_array_equal(after['choices'][f'lowrank_{a}_{arm}_selected'],before['choices']['raw'])
                np.testing.assert_array_equal(after['choices'][f'lowrank_{a}_{arm}_zero'],before['choices'][f'ridge_{a}_zero'])

    def test_outer_held_labels_do_not_enter_fit_or_selection(self):
        rows,registered = fixtures.cohort.fixture()
        train,held = spec.training.cohort.split_rows(spec.training.cohort.source.matched_views(rows,registered)['mixed_a'],0)
        panels,changed = fixtures.old.panels_for(rows),fixtures.old.panels_for(rows)
        for r in train: changed[joint.binding.learned.state_key(r)]['values'][4:,(r['original_choice']+1)%6] += 100.
        for r in held: changed[joint.binding.learned.state_key(r)]['values'][:,(r['original_choice']+1)%6] += 100.
        captured = []
        selection = {a:{arm:{'mse_level':'zero','joint_level':None} for arm in learning.ARMS} for a in joint.ANCHORS}
        def collect(rs,ps,indices,reused):
            captured.append(('selection',joint.physical_values(rs,ps,indices).tolist()))
            return copy.deepcopy(selection),[],{}
        def final(rs,ps,indices,bundle,selected,**kwargs):
            captured.append(('final',joint.physical_values(rs,ps,indices).tolist(),copy.deepcopy(selected)))
            return {'paths':{},'selection':selected},{}
        with patch.object(learning,'collect',side_effect=collect),patch.object(joint,'fit_final',side_effect=final),patch.object(
                learning,'predict',side_effect=lambda rs,*args:zero_prediction(rs)):
            first = spec.fit_cell(train,held,panels,{}, {},{}, {},{}, {},'first_last',0)[0]
            second = spec.fit_cell(train,held,changed,{}, {},{}, {},{}, {},'first_last',0)[0]
        self.assertEqual(captured[:2],captured[2:])
        self.assertEqual(first['selection'],second['selection'])
        self.assertNotEqual(first['scopes']['held_other'],second['scopes']['held_other'])

    def test_complete_report_shared_arm_checks_and_reference_leak_rejection(self):
        rows,registered = fixtures.cohort.fixture()
        views,panels = spec.training.cohort.source.matched_views(rows,registered),fixtures.old.panels_for(rows)
        groups = lambda rs:sorted(map(list,{(r['root'],r['scene_seed']) for r in rs}))
        def fitted(gs,n,a): return {'anchor':a,'training_groups':gs,'training_states':n,'steps':0,'scales':[1.,1.],
            'solver':'training_pair_TSVD','paired_rows':n*5,'feature_count':198,'numerical_rank':150,'arms':{arm:{'parameter_count':198 if arm=='wait' else 396,
                'loss_ledger':[{'level':l,'retained_rank':k,'wait_df':k,'cost_df':0 if arm=='wait' else k} for k,l in zip(learning.RANKS,learning.LEVELS)]} for arm in learning.ARMS}}
        results = {}
        for key in spec.KEYS:
            v,fold,p,d = key
            train,held = spec.training.cohort.split_rows(views[v],fold)
            a,b = spec.PARTITIONS[p][d],spec.PARTITIONS[p][1-d]
            inner = []
            for g in groups(train):
                gs = [x for x in groups(train) if x!=g]
                for i in (0,1): inner.append({'held_group':g,'direction':i,'training_draws':list(a[2*i:2*i+2]),
                    'validation_draws':list(a[2*(1-i):2*(1-i)+2]),'held_states':6,'reference':{'draws':list(a[2*i:2*i+2]),
                        'training_states':54,'training_groups':gs,'wait_training_groups':gs,'raw_steps':600,'raw_parameters':3393},
                    'fits':{k:fitted(gs,54,k) for k in joint.ANCHORS}})
            c = {'partition':p,'direction':d,'training_draws':list(a),'evaluation_draws':list(b),'training_groups':groups(train),
                'held_groups':groups(held),'inner_fits':inner,'final_fits':{k:fitted(groups(train),60,k) for k in joint.ANCHORS},
                'selection':{k:{arm:{'mse_level':'zero','joint_level':None,'curves':{l:{} for l in learning.LEVELS}} for arm in learning.ARMS} for k in joint.ANCHORS},
                'inner_reference_reproduction':{'max_abs_difference':0.,'changed_choices':0},'scopes':{s:spec.records(train if s.startswith('train') else held,
                    panels,a if s.endswith('same') else b,b if s.endswith('same') else a,zero_prediction(train if s.startswith('train') else held)) for s in spec.SCOPES}}
            results[key] = {'protocol':spec.EXPERIMENT_PROTOCOL,'contract':spec.contract(),'key':list(key),'cell':c}
        report = spec.summarize(results,views)
        self.assertEqual(report['matched_arm_reproduction'],{'comparisons':320,'max_abs_difference':0.})
        self.assertEqual((report['network_updates'],report['positive_rank_solutions']),(384000,5376))
        changed = copy.deepcopy(results)
        changed[spec.KEYS[0]]['cell']['scopes']['held_other'][0]['lowrank_head_errors']['lowrank_original_wait_rank3'][1] += 1.
        with self.assertRaisesRegex(ValueError,'Matched wait predictors'): spec.summarize(changed,views)
        first = results[spec.KEYS[0]]['cell']['inner_fits'][0]
        first['reference']['training_groups'].append(first['held_group'])
        with self.assertRaisesRegex(ValueError,'leaked prefix/future'): spec.summarize(results,views)

    def test_protocol_roundtrip_frozen29_controls_and_unbound_scheduler(self):
        self.assertEqual(spec.contract(),json.loads(json.dumps(spec.contract())))
        metric = {'mean_cost':0.,'mean_wait_min':0.,'changed_fraction':0.,'unsafe_fraction':0.,'switch_fraction_to_raw':0.}
        def report(): return {'views':{v:{'scopes':{s:{'variants':{n:metric.copy() for n in learning.ridge.VARIANTS},
            'wait_heads':{n:{'physical_mass_loss_min':0.} for n in joint.CONTROLS},'joint_prediction_errors':{n:{'normalized_mse':0.,'cases':480}
                for n in learning.ridge.VARIANTS if n not in joint.CONTROLS}} for s in spec.SCOPES}} for v in spec.VIEWS}}
        a,b = report(),report()
        self.assertEqual(spec.previous.previous.reproduce_controls(a,b,names=learning.ridge.VARIANTS)['max_abs_difference'],0.)
        b['views']['mixed_a']['scopes']['held_other']['variants']['ridge_raw_selected']['mean_cost'] = 1.
        with self.assertRaisesRegex(ValueError,'Frozen source'): spec.previous.previous.reproduce_controls(a,b,names=learning.ridge.VARIANTS)
        for key in (spec.KEYS[0],(None,None,None,None)):
            task = task_specification('unit_lowrank_joint',*key)
            self.assertEqual((task['cpu'],task['ram_mb']),(1,3072))
            self.assertIsNone(task['require_node'])
            self.assertEqual(task['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
            self.assertNotIn('_raw',task['result_dir'])


if __name__=='__main__': unittest.main()
