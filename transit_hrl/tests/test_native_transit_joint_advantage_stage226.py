import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_joint_advantage as learning
from scripts import run_native_transit_joint_advantage_stage226 as spec
from scripts.submit_native_transit_joint_advantage_stage226_scheduleurm import task_specification
from tests import test_native_transit_best_action_crossfit_stage220 as old
from tests import test_native_transit_best_action_stage218 as cohort
from tests import test_native_transit_coverage_value as native_fixture


def toy_data():
    rng = np.random.default_rng(226)
    originals, raw = np.array([0,3]), np.array([1,2])
    heads = rng.normal(size=(2,6,2))
    heads[np.arange(2),originals] = 0.
    values = heads+.4*rng.normal(size=(2,6,2))
    values[np.arange(2),originals] = 0.
    data = {"x":rng.normal(size=(2,6,210)),"heads":heads,"originals":originals,
        "raw_choices":raw,"roots":np.array([397,401])}
    rows = [{"root":397+i*4,"scene_seed":i,"original_choice":int(originals[i])} for i in range(2)]
    return rows,data,values


def zero_prediction(rows):
    n = len(rows)
    return {"costs":{k:np.zeros((n,6)) for k in ("ridge","best")},
        "wait_logits":{k:np.zeros((n,6)) for k in learning.CONTROLS},
        "choices":{k:np.asarray([r["original_choice"] for r in rows]) for k in learning.VARIANTS},"advantages":{}}


class JointAdvantageTest(unittest.TestCase):
    def test_two_physical_heads_train_and_anchor_is_exact(self):
        rows,data,values = toy_data()
        with patch.object(learning,"CHECKPOINTS",(0,1,3)):
            for kind in learning.ANCHORS:
                model,fit,curve = learning.fit_path(rows,data,values,kind,validation=(data,values))
                anchor = learning.anchors(data,kind)
                self.assertEqual((fit['steps'],fit['parameter_count']),(3,3410))
                self.assertGreater(fit['parameter_update_l2'],0.)
                self.assertTrue(np.all(np.asarray(fit['scales'])>0))
                delta = learning.increments(data,model,3)
                self.assertTrue(np.array_equal(delta[np.arange(2),anchor],np.zeros((2,2))))
                self.assertEqual(curve['3']['error_cases'],20)
                self.assertTrue(np.all(np.isfinite(delta)))

    def test_anchor_first_ties_and_both_predicted_increments(self):
        _,data,_ = toy_data()
        w = {'0.weight':np.zeros((16,210)).tolist(),'0.bias':[0.]*16,'2.weight':np.zeros((2,16)).tolist(),'2.bias':[0.,0.]}
        model = {'anchor':'raw','scales':[1.,1.],'weights':{'0':w}}
        data['heads'][:] = 0.
        self.assertTrue(np.array_equal(learning.choose(data,model,0),data['raw_choices']))
        data['heads'][:,5] = [-1.,1.]
        self.assertTrue(np.array_equal(learning.choose(data,model,0),data['raw_choices']))
        data['heads'][:,5,1] = -1.
        self.assertTrue(np.array_equal(learning.choose(data,model,0),[5,5]))

    def test_joint_checkpoint_uses_worst_root_and_raw_hold(self):
        def curve(root):
            return {str(s): {'error_sum':float(4-s),'error_cases':10,'changed_count':1,
                'by_root':{str(root):{'cases':1,'cost_sum':-.1,'wait_sum_min':-.1,
                    'scaled_cost_sum':-.1 if s==1 else -.5,'scaled_wait_sum':-.1 if s==1 else .2}}} for s in (0,1,3)}
        curves = [curve(397),curve(401)]
        with patch.object(learning,'CHECKPOINTS',(0,1,3)):
            selected = learning.select_checkpoints(curves)
            self.assertEqual(selected['mse_step'],3)
            self.assertEqual(selected['joint_step'],1)
            curves[1]['1']['by_root']['401']['scaled_wait_sum'] = .1
            self.assertIsNone(learning.select_checkpoints(curves)['joint_step'])

    def test_clean_inner_reference_and_future_isolation(self):
        bank = native_fixture.bank_fixture()[0]['training_data']
        groups = sorted({(r['root'],r['scene_seed']) for r in bank})[:2]
        rows = [r for r in bank if (r['root'],r['scene_seed']) in groups]
        fold = list(learning.binding.policy.prefix_folds(rows))[0]
        train,held = [rows[i] for i in fold[1]],[rows[i] for i in fold[2]]
        panels = old.panels_for(rows)
        with patch.object(learning,'CHECKPOINTS',(0,1,3)),patch.object(learning.binding.learned.neural,'STEPS',3),patch.object(
                learning.binding.policy,'prefix_folds',return_value=[fold]):
            predictions = []
            for a in ((0,1),(2,3)):
                ref,_ = learning.fit_reference(train,panels,a)
                predictions.append(learning.causal_data(held,ref))
            reused = {'originals':np.concatenate([x['originals'] for x in predictions]),'roots':np.concatenate([x['roots'] for x in predictions]),
                'costs':np.concatenate([x['heads'][...,0] for x in predictions]),'state_conditioned_pair':np.concatenate([x['raw_logits'] for x in predictions])}
            first,ledger,reproduction = learning.collect(rows,panels,(0,1,2,3),reused)
            other = copy.deepcopy(panels)
            for r in rows: other[learning.binding.learned.state_key(r)]['values'][4:,(r['original_choice']+1)%6] += 100.
            second,other_ledger,_ = learning.collect(rows,other,(0,1,2,3),reused)
            self.assertEqual(first,second)
            self.assertEqual(ledger,other_ledger)
            altered = copy.deepcopy(panels)
            for r in held: altered[learning.binding.learned.state_key(r)]['values'][:4,(r['original_choice']+1)%6] += 100.
            changed,changed_ledger,_ = learning.collect(rows,altered,(0,1,2,3),reused)
        self.assertEqual(ledger,changed_ledger)
        self.assertNotEqual(first['raw']['curves'],changed['raw']['curves'])
        self.assertEqual(reproduction['changed_choices'],0)
        self.assertLessEqual(reproduction['max_abs_difference'],1e-12)
        for inner in ledger:
            self.assertNotIn(list(fold[0]),inner['reference']['training_groups'])
            self.assertFalse(set(inner['training_draws']) & set(inner['validation_draws']))

    def test_final_prediction_causal_controls_and_raw_hold(self):
        rows = native_fixture.bank_fixture()[0]['training_data'][:6]
        panels = old.panels_for(rows)
        with patch.object(learning.binding.precision.base.base,'STEPS',3),patch.object(learning.binding.learned.neural,'STEPS',3),patch.object(learning,'CHECKPOINTS',(0,1,3)):
            bundle,_ = learning.binding.precision.fit_bundle(rows,panels,(0,1,2,3))
            bundle['best'] = {'center':bundle['ridge']['center'],'scale':bundle['ridge']['scale'],'coefficients':[0.]*98}
            selection = {a:{'mse_step':1,'joint_step':None} for a in learning.ANCHORS}
            model,fits = learning.fit_final(rows,panels,(0,1,2,3),bundle,selection)
            cal = {m:{v:{'alpha':1.} for v in learning.binding.WAIT_VARIANTS} for m in ('risk','oof','final','joint')}
            causal = [{k:v for k,v in r.items() if k!='labels'} for r in rows]
            prediction = learning.predict(causal,model,bundle,cal)
        self.assertEqual(set(prediction['choices']),set(learning.VARIANTS))
        self.assertEqual(set(prediction['wait_logits']),set(learning.CONTROLS))
        for a in learning.ANCHORS:
            self.assertTrue(np.array_equal(prediction['choices'][f'pair_{a}_selected'],prediction['choices']['raw']))
            self.assertEqual(fits[a]['steps'],3)

    def test_outer_held_and_other_future_labels_do_not_enter_selection_or_final_fit(self):
        rows,registered = cohort.fixture()
        train,held = spec.cohort.split_rows(spec.cohort.source.matched_views(rows,registered)['mixed_a'],0)
        panels,changed = old.panels_for(rows),old.panels_for(rows)
        for r in train: changed[learning.binding.learned.state_key(r)]['values'][4:,(r['original_choice']+1)%6] += 100.
        for r in held: changed[learning.binding.learned.state_key(r)]['values'][:,(r['original_choice']+1)%6] += 100.
        captured = []
        selection = {a:{'mse_step':50,'joint_step':None} for a in learning.ANCHORS}
        def collect(rs,ps,indices,reused):
            captured.append(('selection',learning.physical_values(rs,ps,indices).tolist()))
            return copy.deepcopy(selection),[],{}
        def final(rs,ps,indices,bundle,selected):
            captured.append(('final',learning.physical_values(rs,ps,indices).tolist(),copy.deepcopy(selected)))
            return {},{}
        with patch.object(learning,'collect',side_effect=collect),patch.object(learning,'fit_final',side_effect=final),patch.object(
                learning,'predict',side_effect=lambda rs,*args:zero_prediction(rs)):
            first = spec.fit_cell(train,held,panels,{}, {},{},'first_last',0)[0]
            second = spec.fit_cell(train,held,changed,{}, {},{},'first_last',0)[0]
        self.assertEqual(captured[:2],captured[2:])
        self.assertEqual(first['selection'],second['selection'])
        self.assertNotEqual(first['scopes']['held_other'],second['scopes']['held_other'])

    def test_complete_report_and_reference_leak_rejection(self):
        rows,registered = cohort.fixture()
        views,panels = spec.cohort.source.matched_views(rows,registered),old.panels_for(rows)
        results = {}
        groups = lambda rs: sorted(map(list,{(r['root'],r['scene_seed']) for r in rs}))
        for key in spec.KEYS:
            view,fold,partition,direction = key
            train,held = spec.cohort.split_rows(views[view],fold)
            a,b = spec.PARTITIONS[partition][direction],spec.PARTITIONS[partition][1-direction]
            def fitted(gs,n,kind): return {'anchor':kind,'training_groups':gs,'training_states':n,'parameter_count':3410,'steps':600}
            inner = []
            for group in groups(train):
                gs = [g for g in groups(train) if g!=group]
                for d in (0,1):
                    inner.append({'held_group':group,'direction':d,'training_draws':list(a[2*d:2*d+2]),'validation_draws':list(a[2*(1-d):2*(1-d)+2]),
                        'held_states':6,'reference':{'draws':list(a[2*d:2*d+2]),'training_states':54,'training_groups':gs,'wait_training_groups':gs,
                            'raw_steps':600,'raw_parameters':3393},'fits':{k:fitted(gs,54,k) for k in learning.ANCHORS}})
            cell = {'partition':partition,'direction':direction,'training_draws':list(a),'evaluation_draws':list(b),'training_groups':groups(train),'held_groups':groups(held),
                'inner_fits':inner,'final_fits':{k:fitted(groups(train),60,k) for k in learning.ANCHORS},
                'selection':{k:{'mse_step':0,'joint_step':None,'curves':{str(s):{} for s in learning.CHECKPOINTS}} for k in learning.ANCHORS},
                'inner_reference_reproduction':{'max_abs_difference':0.,'changed_choices':0},'scopes':{s:spec.records(train if s.startswith('train') else held,
                    panels,a if s.endswith('same') else b,b if s.endswith('same') else a,zero_prediction(train if s.startswith('train') else held)) for s in spec.SCOPES}}
            results[key] = {'protocol':spec.EXPERIMENT_PROTOCOL,'contract':spec.contract(),'key':list(key),'cell':cell}
        report = spec.summarize(results,views)
        self.assertEqual(report['network_updates'],1190400)
        self.assertTrue(all(d['scopes']['held_other']['prediction_cases']==480 for d in report['views'].values()))
        first = results[spec.KEYS[0]]['cell']['inner_fits'][0]
        first['reference']['training_groups'].append(first['held_group'])
        with self.assertRaisesRegex(ValueError,'leaked prefix/future'): spec.summarize(results,views)

    def test_control_reproduction_json_budget_and_scheduler(self):
        self.assertEqual(spec.contract(),json.loads(json.dumps(spec.contract())))
        metric = {'mean_cost':0.,'mean_wait_min':0.,'changed_fraction':0.,'unsafe_fraction':0.}
        mapping = {'glm':'glm','raw':'raw_state_conditioned_pair','c2v4':'oof_state_conditioned_pair'}
        a = {'views':{v:{'scopes':{s:{'variants':{g:metric.copy() for g in mapping},'wait_heads':{g:{'physical_mass_loss_min':0.} for g in mapping}}
            for s in spec.SCOPES}} for v in spec.VIEWS}}
        b = {'views':{v:{'scopes':{s:{'decisions':{g:metric.copy() for g in mapping.values()},'strata':{'all':{'heads':{g:{'physical_mass_loss_min':0.} for g in mapping.values()}}}}
            for s in spec.SCOPES}} for v in spec.VIEWS}}
        self.assertEqual(spec.reproduce_controls(a,b),{'comparisons':240,'max_abs_difference':0.})
        b['views']['mixed_a']['scopes']['held_other']['decisions']['glm']['mean_cost'] = 1.
        with self.assertRaisesRegex(ValueError,'failed reproduction'): spec.reproduce_controls(a,b)
        for key in (spec.KEYS[0],(None,None,None,None)):
            task = task_specification('unit_joint_advantage',*key)
            self.assertEqual((task['cpu'],task['ram_mb']),(1,3072))
            self.assertIsNone(task['require_node'])
            self.assertEqual(task['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
            self.assertNotIn('_raw',task['result_dir'])


if __name__ == '__main__': unittest.main()
