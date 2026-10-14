import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_linear_joint as learning
from scripts import run_native_transit_linear_joint_stage228 as spec
from scripts.submit_native_transit_linear_joint_stage228_scheduleurm import task_specification
from tests import test_native_transit_joint_advantage_stage226 as nonlinear
from tests import test_native_transit_joint_selection_stage227 as repaired

joint = learning.joint


def zero_prediction(rows):
    p = nonlinear.zero_prediction(rows)
    p['choices'].update({v:np.asarray([r['original_choice'] for r in rows]) for v in learning.VARIANTS})
    return p


class LinearJointTest(unittest.TestCase):
    def test_real_tiny_fit_zero_start_exact_anchor_and_396_parameters(self):
        rows,data,values = nonlinear.toy_data()
        with patch.object(joint,'CHECKPOINTS',(0,1,3)):
            for kind in joint.ANCHORS:
                model,fit,curve = learning.fit_path(rows,data,values,kind,validation=(data,values))
                self.assertEqual((fit['steps'],fit['parameter_count']),(3,396))
                self.assertGreater(fit['parameter_update_l2'],0.)
                self.assertLess(fit['loss_ledger'][-1]['training_mse'],fit['loss_ledger'][0]['training_mse'])
                self.assertEqual(curve['3']['error_cases'],20)
                anchor = joint.anchors(data,kind)
                np.testing.assert_array_equal(learning.increments(data,model,0),joint.relative(data['heads'],anchor))
                np.testing.assert_array_equal(learning.increments(data,model,3)[np.arange(2),anchor],np.zeros((2,2)))
                self.assertEqual(model['scales'],fit['scales'])

    def test_discarded_linear_context_and_bias_are_algebraically_null(self):
        _,data,_ = nonlinear.toy_data()
        rng = np.random.default_rng(228)
        data['x'][...,198:] = rng.normal(size=(2,1,12))
        weights,bias = rng.normal(size=(210,2)),rng.normal(size=2)
        anchor = data['raw_choices']
        full = joint.relative(data['heads']+data['x']@weights+bias,anchor)
        model = {'anchor':'raw','scales':[1.,1.],'weights':{'0':weights[:198].T.tolist()}}
        np.testing.assert_allclose(learning.increments(data,model,0),full,rtol=1e-12,atol=1e-12)

    def test_new_collector_uses_corrected_physical_sign_not_fold_normalized_sign(self):
        cell = repaired.source_cell()
        with patch.object(joint,'collect',return_value=(cell['selection'],cell['inner_fits'],{'max_abs_difference':0.,'changed_choices':0})) as collect:
            selection,_,_ = learning.collect([],{},(0,1,2,3),{})
        self.assertIs(collect.call_args.kwargs['path_fitter'],learning.fit_path)
        self.assertTrue(all(x['joint_step'] is None for x in selection.values()))
        self.assertGreater(selection['raw']['curves']['0']['joint_score'],0.)

    def test_clean_prefix_future_selection_and_outer_future_isolation(self):
        bank = nonlinear.native_fixture.bank_fixture()[0]['training_data']
        groups = sorted({(r['root'],r['scene_seed']) for r in bank})[:2]
        rows = [r for r in bank if (r['root'],r['scene_seed']) in groups]
        fold = list(joint.binding.policy.prefix_folds(rows))[0]
        train,held = [rows[i] for i in fold[1]],[rows[i] for i in fold[2]]
        panels = nonlinear.old.panels_for(rows)
        with patch.object(joint,'CHECKPOINTS',(0,1,3)),patch.object(joint.binding.learned.neural,'STEPS',3),patch.object(
                joint.binding.policy,'prefix_folds',return_value=[fold]):
            ds = [joint.causal_data(held,joint.fit_reference(train,panels,a)[0]) for a in ((0,1),(2,3))]
            reused = {'originals':np.concatenate([d['originals'] for d in ds]),'roots':np.concatenate([d['roots'] for d in ds]),
                'costs':np.concatenate([d['heads'][...,0] for d in ds]),'state_conditioned_pair':np.concatenate([d['raw_logits'] for d in ds])}
            first,ledger,reproduction = learning.collect(rows,panels,(0,1,2,3),reused)
            changed = copy.deepcopy(panels)
            for r in rows: changed[joint.binding.learned.state_key(r)]['values'][4:,(r['original_choice']+1)%6] += 100.
            second,other_ledger,_ = learning.collect(rows,changed,(0,1,2,3),reused)
        self.assertEqual(first,second)
        self.assertEqual(ledger,other_ledger)
        self.assertEqual(reproduction,{'max_abs_difference':0.,'changed_choices':0})
        for f in ledger:
            self.assertNotIn(list(fold[0]),f['reference']['training_groups'])
            self.assertFalse(set(f['training_draws']) & set(f['validation_draws']))
            self.assertTrue(all(x['parameter_count']==396 for x in f['fits'].values()))

    def test_causal_inference_frozen_controls_raw_hold_and_anchor_first_ties(self):
        rows = nonlinear.native_fixture.bank_fixture()[0]['training_data'][:6]
        panels = nonlinear.old.panels_for(rows)
        with patch.object(joint.binding.precision.base.base,'STEPS',3),patch.object(joint.binding.learned.neural,'STEPS',3),patch.object(joint,'CHECKPOINTS',(0,1,3)):
            bundle,_ = joint.binding.precision.fit_bundle(rows,panels,(0,1,2,3))
            bundle['best'] = {'center':bundle['ridge']['center'],'scale':bundle['ridge']['scale'],'coefficients':[0.]*98}
            selected = {a:{'mse_step':1,'joint_step':None} for a in joint.ANCHORS}
            nn,_ = joint.fit_final(rows,panels,(0,1,2,3),bundle,selected)
            linear,fits = joint.fit_final(rows,panels,(0,1,2,3),bundle,selected,path_fitter=learning.fit_path)
            cal = {m:{v:{'alpha':1.} for v in joint.binding.WAIT_VARIANTS} for m in ('risk','oof','final','joint')}
            causal = [{k:v for k,v in r.items() if k!='labels'} for r in rows]
            before = joint.predict(causal,nn,bundle,cal)
            after = learning.predict(causal,linear,nn,bundle,cal)
        self.assertEqual(set(after['choices']),set(learning.VARIANTS))
        for name in joint.VARIANTS: np.testing.assert_array_equal(before['choices'][name],after['choices'][name])
        for a in joint.ANCHORS:
            np.testing.assert_array_equal(after['choices'][f'linear_{a}_selected'],before['choices']['raw'])
            self.assertEqual(fits[a]['parameter_count'],396)
        _,data,_ = nonlinear.toy_data()
        data['heads'][:] = 0.
        path = {'anchor':'raw','scales':[1.,1.],'weights':{'0':np.zeros((2,198)).tolist()}}
        delta = learning.increments(data,path,0)
        np.testing.assert_array_equal(joint.binding.policy.selected(delta[...,0],delta[...,1],data['raw_choices']),data['raw_choices'])

    def test_outer_held_truth_does_not_enter_linear_selection_or_final_fit(self):
        rows,registered = nonlinear.cohort.fixture()
        train,held = spec.training.cohort.split_rows(spec.training.cohort.source.matched_views(rows,registered)['mixed_a'],0)
        panels,changed = nonlinear.old.panels_for(rows),nonlinear.old.panels_for(rows)
        for r in train: changed[joint.binding.learned.state_key(r)]['values'][4:,(r['original_choice']+1)%6] += 100.
        for r in held: changed[joint.binding.learned.state_key(r)]['values'][:,(r['original_choice']+1)%6] += 100.
        captured = []
        selected = {a:{'mse_step':0,'joint_step':None} for a in joint.ANCHORS}
        def collect(rs,ps,indices,reused):
            captured.append(('selection',joint.physical_values(rs,ps,indices).tolist()))
            return copy.deepcopy(selected),[],{}
        def final(rs,ps,indices,bundle,selection,**kwargs):
            captured.append(('final',joint.physical_values(rs,ps,indices).tolist(),copy.deepcopy(selection)))
            return {'paths':{},'selection':selection},{}
        with patch.object(learning,'collect',side_effect=collect),patch.object(joint,'fit_final',side_effect=final),patch.object(
                learning,'predict',side_effect=lambda rs,*args:zero_prediction(rs)):
            first = spec.fit_cell(train,held,panels,{}, {},{}, {},'first_last',0)[0]
            second = spec.fit_cell(train,held,changed,{}, {},{}, {},'first_last',0)[0]
        self.assertEqual(captured[:2],captured[2:])
        self.assertEqual(first['selection'],second['selection'])
        self.assertNotEqual(first['scopes']['held_other'],second['scopes']['held_other'])

    def test_complete_report_budget_reference_and_coverage_rejection(self):
        rows,registered = nonlinear.cohort.fixture()
        views,panels = spec.training.cohort.source.matched_views(rows,registered),nonlinear.old.panels_for(rows)
        groups = lambda rs:sorted(map(list,{(r['root'],r['scene_seed']) for r in rs}))
        results = {}
        def fitted(gs,n,a): return {'anchor':a,'training_groups':gs,'training_states':n,'parameter_count':396,'steps':600,'scales':[1.,1.],
            'loss_ledger':[{'step':s,'training_mse':0.} for s in joint.CHECKPOINTS]}
        for key in spec.KEYS:
            view,fold,partition,direction = key
            train,held = spec.training.cohort.split_rows(views[view],fold)
            a,b = spec.PARTITIONS[partition][direction],spec.PARTITIONS[partition][1-direction]
            inner = []
            for group in groups(train):
                gs = [g for g in groups(train) if g!=group]
                for d in (0,1): inner.append({'held_group':group,'direction':d,'training_draws':list(a[2*d:2*d+2]),
                    'validation_draws':list(a[2*(1-d):2*(1-d)+2]),'held_states':6,'reference':{'draws':list(a[2*d:2*d+2]),
                        'training_states':54,'training_groups':gs,'wait_training_groups':gs,'raw_steps':600,'raw_parameters':3393},
                    'fits':{k:fitted(gs,54,k) for k in joint.ANCHORS}})
            cell = {'partition':partition,'direction':direction,'training_draws':list(a),'evaluation_draws':list(b),
                'training_groups':groups(train),'held_groups':groups(held),'inner_fits':inner,'final_fits':{k:fitted(groups(train),60,k) for k in joint.ANCHORS},
                'selection':{k:{'mse_step':0,'joint_step':None,'curves':{str(s):{} for s in joint.CHECKPOINTS}} for k in joint.ANCHORS},
                'inner_reference_reproduction':{'max_abs_difference':0.,'changed_choices':0},
                'scopes':{s:spec.training.records(train if s.startswith('train') else held,panels,a if s.endswith('same') else b,
                    b if s.endswith('same') else a,zero_prediction(train if s.startswith('train') else held)) for s in spec.SCOPES}}
            results[key] = {'protocol':spec.EXPERIMENT_PROTOCOL,'contract':spec.contract(),'key':list(key),'cell':cell}
        report = spec.summarize(results,views)
        self.assertEqual(report['network_updates'],1190400)
        self.assertTrue(all(d['scopes']['held_other']['prediction_cases']==480 for d in report['views'].values()))
        broken = copy.deepcopy(results)
        first = broken[spec.KEYS[0]]['cell']['inner_fits'][0]
        first['reference']['training_groups'].append(first['held_group'])
        with self.assertRaisesRegex(ValueError,'leaked prefix/future'): spec.summarize(broken,views)
        results[spec.KEYS[0]]['cell']['scopes']['held_other'].pop()
        with self.assertRaisesRegex(ValueError,'Dropped a linear state'): spec.summarize(results,views)

    def test_frozen_controller_reproduction_null_prediction_and_scheduler(self):
        self.assertEqual(spec.contract(),json.loads(json.dumps(spec.contract())))
        metric = {'mean_cost':0.,'mean_wait_min':0.,'changed_fraction':0.,'unsafe_fraction':0.,'switch_fraction_to_raw':0.}
        def report(): return {'views':{v:{'scopes':{s:{'variants':{n:metric.copy() for n in joint.VARIANTS},
            'wait_heads':{n:{'physical_mass_loss_min':0.} for n in joint.CONTROLS},
            'joint_prediction_errors':{n:{'normalized_mse':None if n.endswith('selected') else 0.,'cases':0 if n.endswith('selected') else 480}
                for n in joint.VARIANTS if n not in joint.CONTROLS}} for s in spec.SCOPES}} for v in spec.VIEWS}}
        a,b = report(),report()
        self.assertEqual(spec.reproduce_controls(a,b)['max_abs_difference'],0.)
        b['views']['mixed_a']['scopes']['held_other']['variants']['pair_raw_selected']['mean_cost'] = 1.
        with self.assertRaisesRegex(ValueError,'Frozen source'): spec.reproduce_controls(a,b)
        for key in (spec.KEYS[0],(None,None,None,None)):
            task = task_specification('unit_linear_joint',*key)
            self.assertEqual((task['cpu'],task['ram_mb']),(1,3072))
            self.assertIsNone(task['require_node'])
            self.assertEqual(task['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
            self.assertNotIn('_raw',task['result_dir'])


if __name__=='__main__': unittest.main()
