import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_ridge_joint as learning
from scripts import run_native_transit_ridge_joint_stage229 as spec
from scripts.submit_native_transit_ridge_joint_stage229_scheduleurm import task_specification
from tests import test_native_transit_joint_advantage_stage226 as fixtures

joint = learning.joint


def zero_prediction(rows):
    p = fixtures.zero_prediction(rows)
    p['choices'].update({n:np.asarray([r['original_choice'] for r in rows]) for n in learning.VARIANTS})
    return p


class RidgeJointTest(unittest.TestCase):
    def test_svd_matches_normal_equations_and_training_physical_errors(self):
        rows,data,values = fixtures.toy_data()
        for kind in joint.ANCHORS:
            model,fit,curve = learning.fit_path(rows,data,values,kind,validation=(data,values))
            anchor = joint.anchors(data,kind)
            mask = np.arange(6)[None,:]!=anchor[:,None]
            x = joint.relative(data['x'][...,:198],anchor)[mask]
            y = joint.relative(values-data['heads'],anchor)[mask]/model['scales']
            self.assertEqual((fit['steps'],fit['parameter_count']),(0,396))
            for entry in fit['loss_ledger'][1:]:
                expected = np.linalg.solve(x.T@x/len(x)+entry['ridge']*np.eye(198),x.T@y/len(x))
                np.testing.assert_allclose(np.asarray(model['weights'][entry['level']]).T,expected,rtol=1e-10,atol=1e-11)
                self.assertAlmostEqual(entry['training_mse'],curve[entry['level']]['error_sum']/curve[entry['level']]['error_cases'])
                self.assertAlmostEqual(entry['training_mse'],(entry['cost_mse']+entry['wait_mse'])/2)
            norms = [x['coefficient_l2'] for x in fit['loss_ledger'][1:]]
            dfs = [x['effective_df_per_head'] for x in fit['loss_ledger'][1:]]
            self.assertTrue(all(a<=b for a,b in zip(norms,norms[1:])))
            self.assertTrue(all(a<=b for a,b in zip(dfs,dfs[1:])))
            np.testing.assert_array_equal(learning.linear.increments(data,model,'zero'),joint.relative(data['heads'],anchor))
            np.testing.assert_array_equal(learning.linear.increments(data,model,learning.LEVELS[-1])[np.arange(2),anchor],np.zeros((2,2)))

    def test_rank_deficient_engineered_features_and_uniform_feature_scale(self):
        rows,data,values = fixtures.toy_data()
        rng = np.random.default_rng(229)
        data['x'][...,:198] = rng.normal(size=(2,6,2))@rng.normal(size=(2,198))
        model,fit,_ = learning.fit_path(rows,data,values,'raw')
        self.assertEqual(fit['spectrum']['numerical_rank'],2)
        self.assertLessEqual(fit['spectrum']['effective_rank'],2.+1e-12)
        self.assertLessEqual(fit['spectrum']['energy95_rank'],2)
        self.assertTrue(all(x['effective_df_per_head']<=2.+1e-12 for x in fit['loss_ledger']))
        scaled = copy.deepcopy(data)
        scaled['x'][...,:198] *= 3.
        other,_,_ = learning.fit_path(rows,scaled,values,'raw')
        for level in learning.LEVELS:
            np.testing.assert_allclose(learning.linear.increments(data,model,level),learning.linear.increments(scaled,other,level),atol=1e-11)

    def test_physical_means_sign_reversal_and_stronger_ties(self):
        ledger = [{'held_group':[397,i],'held_states':6,'fits':{'raw':{'scales':[scale,1.]}}} for i,scale in enumerate((1.,100.))]
        def curve(cost,scale): return {l:{'error_sum':0. if l=='zero' else 1.,'error_cases':10,'changed_count':1,
            'by_root':{'397':{'cases':6,'cost_sum':cost*6,'wait_sum_min':-.6,'scaled_cost_sum':cost*6/scale,'scaled_wait_sum':-.6}}} for l in learning.LEVELS}
        curves = [curve(-1.,1.),curve(2.,100.)]
        selected = learning.select(curves,ledger,'raw')
        self.assertEqual(selected['mse_level'],'zero')
        self.assertIsNone(selected['joint_level'])
        self.assertGreater(selected['curves']['zero']['joint_score'],0.)
        for c in curves:
            for l in learning.LEVELS[1:3]: c[l]['by_root']['397']['cost_sum'] = -.6
        self.assertEqual(learning.select(curves,ledger,'raw')['joint_level'],learning.LEVELS[1])

    def test_clean_inner_reference_rank_and_selection_future_isolation(self):
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
        self.assertNotEqual(first['raw']['curves'],third['raw']['curves'])
        self.assertEqual(reproduction,{'max_abs_difference':0.,'changed_choices':0})
        for f in ledger:
            self.assertNotIn(list(fold[0]),f['reference']['training_groups'])
            self.assertFalse(set(f['training_draws']) & set(f['validation_draws']))
            self.assertTrue(all(x['steps']==0 for x in f['fits'].values()))

    def test_causal_predictions_frozen19_controls_and_raw_hold(self):
        rows = fixtures.native_fixture.bank_fixture()[0]['training_data'][:6]
        panels = fixtures.old.panels_for(rows)
        with patch.object(joint.binding.precision.base.base,'STEPS',3),patch.object(joint.binding.learned.neural,'STEPS',3),patch.object(joint,'CHECKPOINTS',(0,1,3)):
            bundle,_ = joint.binding.precision.fit_bundle(rows,panels,(0,1,2,3))
            bundle['best'] = {'center':bundle['ridge']['center'],'scale':bundle['ridge']['scale'],'coefficients':[0.]*98}
            selection = {a:{'mse_step':1,'joint_step':None} for a in joint.ANCHORS}
            nonlinear,_ = joint.fit_final(rows,panels,(0,1,2,3),bundle,selection)
            linear,_ = joint.fit_final(rows,panels,(0,1,2,3),bundle,selection,path_fitter=learning.linear.fit_path)
            ridge,_ = joint.fit_final(rows,panels,(0,1,2,3),bundle,{a:{'mse_level':'zero','joint_level':None} for a in joint.ANCHORS},path_fitter=learning.fit_path)
            cal = {m:{v:{'alpha':1.} for v in joint.binding.WAIT_VARIANTS} for m in ('risk','oof','final','joint')}
            causal = [{k:v for k,v in r.items() if k!='labels'} for r in rows]
            before = learning.linear.predict(causal,linear,nonlinear,bundle,cal)
            after = learning.predict(causal,ridge,linear,nonlinear,bundle,cal)
        self.assertEqual(set(after['choices']),set(learning.VARIANTS))
        for n in learning.linear.VARIANTS: np.testing.assert_array_equal(before['choices'][n],after['choices'][n])
        for a in joint.ANCHORS:
            np.testing.assert_array_equal(after['choices'][f'ridge_{a}_selected'],before['choices']['raw'])
            np.testing.assert_array_equal(after['choices'][f'ridge_{a}_zero'],before['choices'][f'linear_{a}_zero'])
        recorded = spec.records(rows,panels,(0,1,2,3),(4,5,6,7),after)
        for r in recorded:
            for n,p in r['ridge_head_errors'].items(): self.assertAlmostEqual(sum(p)/2,r['joint_prediction_errors'][n])

    def test_outer_held_labels_do_not_enter_fit_selection_or_spectrum(self):
        rows,registered = fixtures.cohort.fixture()
        train,held = spec.training.cohort.split_rows(spec.training.cohort.source.matched_views(rows,registered)['mixed_a'],0)
        panels,changed = fixtures.old.panels_for(rows),fixtures.old.panels_for(rows)
        for r in train: changed[joint.binding.learned.state_key(r)]['values'][4:,(r['original_choice']+1)%6] += 100.
        for r in held: changed[joint.binding.learned.state_key(r)]['values'][:,(r['original_choice']+1)%6] += 100.
        captured = []
        selection = {a:{'mse_level':'zero','joint_level':None} for a in joint.ANCHORS}
        def collect(rs,ps,indices,reused):
            captured.append(('selection',joint.physical_values(rs,ps,indices).tolist()))
            return copy.deepcopy(selection),[],{}
        def final(rs,ps,indices,bundle,selected,**kwargs):
            captured.append(('final',joint.physical_values(rs,ps,indices).tolist(),copy.deepcopy(selected)))
            return {'paths':{},'selection':selected},{}
        with patch.object(learning,'collect',side_effect=collect),patch.object(joint,'fit_final',side_effect=final),patch.object(
                learning,'predict',side_effect=lambda rs,*args:zero_prediction(rs)):
            first = spec.fit_cell(train,held,panels,{}, {},{}, {},{},'first_last',0)[0]
            second = spec.fit_cell(train,held,changed,{}, {},{}, {},{},'first_last',0)[0]
        self.assertEqual(captured[:2],captured[2:])
        self.assertEqual(first['selection'],second['selection'])
        self.assertNotEqual(first['scopes']['held_other'],second['scopes']['held_other'])

    def test_complete_report_and_prefix_budget_rejection(self):
        rows,registered = fixtures.cohort.fixture()
        views,panels = spec.training.cohort.source.matched_views(rows,registered),fixtures.old.panels_for(rows)
        groups = lambda rs:sorted(map(list,{(r['root'],r['scene_seed']) for r in rs}))
        def fitted(gs,n,a): return {'anchor':a,'training_groups':gs,'training_states':n,'parameter_count':396,'steps':0,'scales':[1.,1.],
            'solver':'training_pair_SVD_ridge','loss_ledger':[{'level':l,'training_mse':0.} for l in learning.LEVELS],
            'spectrum':{'paired_rows':n*5,'feature_count':198}}
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
                'selection':{k:{'mse_level':'zero','joint_level':None,'curves':{l:{} for l in learning.LEVELS}} for k in joint.ANCHORS},
                'inner_reference_reproduction':{'max_abs_difference':0.,'changed_choices':0},'scopes':{s:spec.records(train if s.startswith('train') else held,
                    panels,a if s.endswith('same') else b,b if s.endswith('same') else a,zero_prediction(train if s.startswith('train') else held)) for s in spec.SCOPES}}
            results[key] = {'protocol':spec.EXPERIMENT_PROTOCOL,'contract':spec.contract(),'key':list(key),'cell':c}
        report = spec.summarize(results,views)
        self.assertEqual((report['network_updates'],report['positive_ridge_solutions']),(384000,8064))
        self.assertTrue(all(v['scopes']['held_other']['prediction_cases']==480 for v in report['views'].values()))
        first = results[spec.KEYS[0]]['cell']['inner_fits'][0]
        first['reference']['training_groups'].append(first['held_group'])
        with self.assertRaisesRegex(ValueError,'leaked prefix/future'): spec.summarize(results,views)

    def test_roundtrip_all19_controls_and_unbound_scheduler(self):
        self.assertEqual(spec.contract(),json.loads(json.dumps(spec.contract())))
        metric = {'mean_cost':0.,'mean_wait_min':0.,'changed_fraction':0.,'unsafe_fraction':0.,'switch_fraction_to_raw':0.}
        def report(): return {'views':{v:{'scopes':{s:{'variants':{n:metric.copy() for n in learning.linear.VARIANTS},
            'wait_heads':{n:{'physical_mass_loss_min':0.} for n in joint.CONTROLS},'joint_prediction_errors':{n:{'normalized_mse':0.,'cases':480}
                for n in learning.linear.VARIANTS if n not in joint.CONTROLS}} for s in spec.SCOPES}} for v in spec.VIEWS}}
        a,b = report(),report()
        self.assertEqual(spec.previous.reproduce_controls(a,b,names=learning.linear.VARIANTS)['max_abs_difference'],0.)
        b['views']['mixed_a']['scopes']['held_other']['variants']['linear_raw_selected']['mean_cost'] = 1.
        with self.assertRaisesRegex(ValueError,'Frozen source'): spec.previous.reproduce_controls(a,b,names=learning.linear.VARIANTS)
        for key in (spec.KEYS[0],(None,None,None,None)):
            task = task_specification('unit_ridge_joint',*key)
            self.assertEqual((task['cpu'],task['ram_mb']),(1,3072))
            self.assertIsNone(task['require_node'])
            self.assertEqual(task['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
            self.assertNotIn('_raw',task['result_dir'])


if __name__=='__main__': unittest.main()
