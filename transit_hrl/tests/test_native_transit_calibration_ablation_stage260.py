import copy
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_calibration_ablation as learning
from scripts import run_native_transit_calibration_ablation_stage260 as spec
from scripts import submit_native_transit_calibration_ablation_stage260_scheduleurm as submit
from tests import test_native_transit_prefix_influence_stage259 as fixtures


class CalibrationAblationTest(unittest.TestCase):
    def numeric_fixture(self):
        rng = np.random.default_rng(7)
        n = 12
        x = np.zeros((n,6,198)); x[..., :2] = rng.normal(size=(n,6,2))
        data = {'x':x,'originals':np.zeros(n,int),'raw_logits':np.zeros((n,6))}
        z = x[..., :2]-x[:, :1, :2]
        basis = np.eye(198)[:, :2]
        levels = learning.source.strength.lowrank.LEVELS[1:]
        geometry = {level:(basis,basis.T) for level in levels}
        prior = {'scale':1.,'weights':{level:np.zeros((2,198)).tolist() for level in ('zero',*levels)}}
        cap = {'fits':[{'level':level,'objective':'pair_gls','radius':.8} for level in levels]}
        calibration = {'target':z@np.array([[.5,.2],[-.3,.4]]).T+.02*rng.normal(size=(n,6,2)),
            'precision':np.broadcast_to(np.array([.7,1.3]),(n,15,2)).copy(),
            'balance':{'mass_max_abs_difference':0.},'penalties':dict(zip(levels,(None,0.,.2,3.)))}
        return data,calibration,prior,cap,geometry

    def test_fixed_fit_slices_without_recalibration_or_renormalization(self):
        data,c,prior,cap,geometry = self.numeric_fixture()
        original = copy.deepcopy(c)
        indices = list(range(6,12))
        with patch.object(learning.source.base,'noise_targets',side_effect=AssertionError('Recalibrated reliability')), \
                patch.object(learning.shrink.pair,'pair_precision',side_effect=AssertionError('Recalibrated precision')), \
                patch.object(learning.shrink.balanced,'balanced_precision',side_effect=AssertionError('Renormalized subset')), \
                patch.object(learning.shrink,'penalty_rule',side_effect=AssertionError('Recalibrated lambda')):
            models,fit = learning.fit_fixed(data,c,indices,prior,cap,geometry)
            c['target'][:6] += 1e6
            again,_ = learning.fit_fixed(data,c,indices,prior,cap,geometry)
        self.assertEqual(models,again)
        np.testing.assert_array_equal(c['precision'],original['precision'])
        self.assertEqual([f['penalty'] for f in fit['fits']],list(c['penalties'].values()))
        self.assertTrue(all(f['norm'] <= f['old_radius']+1e-10 for f in fit['fits']))

    def test_cached_full_control_checks_optimality_without_solving(self):
        data,c,prior,cap,geometry = self.numeric_fixture()
        models,_ = learning.fit_fixed(data,c,list(range(12)),prior,cap,geometry)
        with patch.object(learning.source.source.optimizer,'quadratic_fit',side_effect=AssertionError('Full refit')):
            checks = learning.check_cached_full(data,c,prior,cap,geometry,models)
            self.assertEqual(checks['full_KKT_checks'],8)
            self.assertLess(checks['max_kkt'],1e-8)
            bad = copy.deepcopy(models)
            level = next(iter(geometry))
            bad['free_old']['weights'][level][0][0] += .1
            with self.assertRaises(ValueError): learning.check_cached_full(data,c,prior,cap,geometry,bad)

    def test_full_calibration_uses_full_target_and_stored_penalties(self):
        data,_,prior,_,geometry = self.numeric_fixture()
        residual = np.ones((12,6,2)); reliability = np.full((12,1,2),.4)
        precision = np.ones((12,15,2))
        ledger = {'fits':[{'level':level,'penalty':None if i==0 else float(i)} for i,level in enumerate(geometry)]}
        with patch.object(learning.source.base,'noise_targets',return_value=(residual,None,reliability,None)), \
                patch.object(learning.shrink.pair,'pair_precision',return_value=(precision,{})), \
                patch.object(learning.shrink,'penalty_rule',side_effect=AssertionError('Changed full penalty')):
            c = learning.full_calibration([{}]*12,data,np.zeros((12,16,6,4)),prior,ledger)
        np.testing.assert_array_equal(c['target'],np.full((12,6,2),.4))
        self.assertEqual(c['penalties'],{f['level']:f['penalty'] for f in ledger['fits']})

    def test_transition_partition_and_nonadditive_action_counts(self):
        x = learning.action_transitions([0,0,0,0,0],[0,1,0,1,1],[0,1,1,0,2])
        self.assertEqual(x,dict.fromkeys(('unchanged','fit_only','calibration_only','cancelled','both_distinct'),1))
        self.assertEqual(learning.check_telescope({'fit_population':{'cost':2.},'recalibration':{'cost':-1.},'total':{'cost':1.}},('cost',)),0.)
        with self.assertRaises(ValueError):learning.check_telescope({'fit_population':{'cost':2.},'recalibration':{'cost':-1.},'total':{'cost':2.}},('cost',))

    def evaluation_fixture(self):
        helper = fixtures.PrefixInfluenceTest()
        old, read = helper.freeze_fixture()
        v = np.zeros((32,6,4))
        for a in range(6):v[:,a] = (a,a/4,3*a/4,2*a)
        panels = {tuple(st):{'values':v} for c in old for st in c['states']}
        def cache(run,key,file):return old if run==spec.SOURCE_RUN else read(run,key,file)
        with patch.object(spec.previous,'read_cell',side_effect=cache):
            before = spec.previous.evaluate_cell(spec.SOURCE_RUN,spec.PREFLIGHT,panels)
        actions=[]
        for c in old:
            actions.append({**{k:c[k] for k in ('cell','bank','removed','variant','category','old_key','old_key_new_gate','scope','states','raw')},
                'choices':{'full':c['full'],'fixed':[3]*len(c['full']),'refit':c['inner']},
                'function_gaps':{name:c['function_gaps'] for name in learning.PAIRS}})
        return actions,panels,before

    def test_three_endpoint_evaluation_reproduces_controls_and_telescopes(self):
        actions,panels,before = self.evaluation_fixture()
        with patch.object(spec.previous,'evaluate_cell',return_value=before):
            pieces,error,telescope = spec.evaluate_cell(actions,panels,spec.PREFLIGHT)
            self.assertEqual((error,telescope),(0.,0.))
            s = spec.summarize(pieces)
            self.assertEqual(s['bank_cell_exposures'],2)
            self.assertEqual(s['comparisons']['fit_population']['held']['effect']['delta_cost'],2.)
            self.assertEqual(s['comparisons']['recalibration']['held']['effect']['delta_cost'],-1.)
            self.assertEqual(s['comparisons']['total']['held']['effect']['delta_cost'],1.)
            self.assertEqual(s['action_transitions']['held']['both_distinct'],1200)
            actions[0]['choices']['refit']=[1]*60
            with self.assertRaisesRegex(ValueError,'reproduction|counts'):spec.evaluate_cell(actions,panels,spec.PREFLIGHT)

    def test_models_and_actions_saved_before_evaluation(self):
        actions,_,_ = self.evaluation_fixture()
        train = [{'root':r,'scene_seed':s,'probe_index':i} for r in (397,401) for s in range(5) for i in range(6)]
        held = [{**r,'scene_seed':r['scene_seed']+10} for r in train]
        models = [{'bank':bank,'held_group':[root,seed],'training_draws':list(range(bank*16,(bank+1)*16)),
            'models':{'free_old':{'choice':2}}} for bank in (0,1) for root in (397,401) for seed in range(5)]
        fit = {'fits':[{'optimality':0.,'difference_optimality':0.,'auxiliary_free_fit':{'optimality':0.,'difference_optimality':0.,'norm':1.},'norm':1.,'old_radius':1.}]*4,
            'subspace_max_abs_difference':0.}
        info = {'protocol':spec.source.EXPERIMENT_PROTOCOL,'contract':spec.source.contract(),'coverage_qualified':True,
            'pieces':[{'bank':b,'arm':'free_old_expanded8','filter':'pooled_safe','old_key':'old'} for b in (0,1)]}
        old = [{**a,'inner':a['choices']['refit'],'full':a['choices']['full']} for a in actions]
        def read(run,key,file):
            if run==spec.previous.SOURCE_RUN:return {'aggregate.json':info,'models.json':models}[file]
            if run==spec.SOURCE_RUN:return old
            if file=='fits.json':return {'full':{}}
            if file=='aggregate.json':return {'banks':[{'bank':b,'full_fit':{}} for b in (0,1)]}
            return {'reference':{},'full':{'all_mse':{}},'0':{'free_old':{'choice':1}},'1':{'free_old':{'choice':1}}}
        events=[]
        def fit_fixed(data,c,indices,*args):
            self.assertEqual(len(indices),54)
            self.assertIs(c,calibration)
            return {'free_old':{'choice':3}},fit
        def candidates(data,model):
            return {'old':{'chosen':np.full(60,model['choice']),'cost':np.full((60,6),model['choice'],float),'components':np.full((60,6,2),model['choice']/2)}}
        def evaluate(*args):
            self.assertEqual(events,['models.json','calibrations.json','actions.json'])
            return [],0.,0.
        calibration={}
        with patch.object(spec.previous,'read_cell',side_effect=read), \
                patch.object(spec.source.cached.training.cohort,'split_rows',return_value=(train,held)), \
                patch.object(spec.source.learning,'MODELS',('free_old',)), \
                patch.object(spec.source.learning.base,'causal_data',side_effect=lambda rows,ref:{'raw_choices':np.zeros(60,int)}), \
                patch.object(spec.source.learning,'deployment_geometry',return_value={}), \
                patch.object(spec.source.sampling,'read_panels',return_value={}), \
                patch.object(spec.source.learning.base,'training_draws',side_effect=lambda rows,panels,ds:np.zeros((60,len(ds),6,4))), \
                patch.object(spec.learning,'full_calibration',return_value=calibration), \
                patch.object(spec.learning,'check_cached_full',return_value={'full_KKT_checks':8,'max_kkt':0.,'subspace_max_abs_difference':0.,'profiled_head_max_abs_difference':0.}), \
                patch.object(spec.learning,'fit_fixed',side_effect=fit_fixed) as fitting, \
                patch.object(spec.source.learning.base,'candidates',side_effect=candidates), \
                patch.object(spec.diagnostic,'function_gap',return_value={}), \
                patch.object(spec,'write_json',side_effect=lambda p,value:events.append(p.name)), \
                patch.object(spec,'evaluate_cell',side_effect=evaluate):
            spec.fit_cell('unit_calibration',spec.PREFLIGHT,{spec.PREFLIGHT[0]:train})
        self.assertEqual(fitting.call_count,20)

    def test_report_preserves_parent_groups_and_budget(self):
        actions,panels,before = self.evaluation_fixture()
        with patch.object(spec.previous,'evaluate_cell',return_value=before):
            pieces,_,_ = spec.evaluate_cell(actions,panels,spec.PREFLIGHT)
        records,parents = [],[]
        for key in spec.KEYS:
            rows=[{**p,'cell':list(key)} for p in pieces]
            parents.append({'key':list(key),'pieces':[{**p,'cell':list(key)} for p in before['pieces']],
                'reproduction_max_abs_difference':0.})
            records.append({'protocol':spec.EXPERIMENT_PROTOCOL,'contract':spec.contract(),'coverage_qualified':True,'key':list(key),
                'pieces':rows,'fits':{'prefix54_fits':20,'inner_solves':160,'full_KKT_checks':16,
                    'max_kkt':0.,'full_profiled_head_max_abs_difference':0.,'subspace_max_abs_difference':0.,'radius_max_excess':0.},
                'reproduction_max_abs_difference':0.,'telescope_max_abs_difference':0.})
        expected=spec.previous.report(parents)
        lookup={tuple(r['key']):r for r in records}
        with patch.object(spec.previous,'read_cell',side_effect=lambda run,key,file:lookup[key]), \
                patch.object(spec.Path,'read_text',return_value=json.dumps(expected)),patch.object(spec,'write_json') as write:
            spec.report('unit_calibration')
            result=write.call_args.args[1]
            self.assertEqual(result['fits']['inner_solves'],5120)
            self.assertEqual(result['reproduction']['group_max_abs_difference'],0.)
            free=result['groups']['mixed_pooled']['pooled']['free_old_expanded8_pooled_safe']['removed']
            self.assertEqual(free['bank_cell_exposures'],32)
            self.assertEqual(free['action_transitions']['held']['both_distinct'],19200)
            expected['groups']['mixed_pooled']['pooled']['free_old_expanded8_pooled_safe']['removed']['regions']['held']['effect']['delta_cost']+=1
        with patch.object(spec.previous,'read_cell',side_effect=lambda run,key,file:lookup[key]), \
                patch.object(spec.Path,'read_text',return_value=json.dumps(expected)),patch.object(spec,'write_json'):
            with self.assertRaisesRegex(ValueError,'group reproduction'):spec.report('unit_calibration')

    def test_scheduler_budget_and_preflight_gate(self):
        self.assertEqual(spec.contract()['budget']['inner_solves'],32*2*10*8)
        for key in (spec.PREFLIGHT,None):
            t=submit.task_specification('unit_calibration',key)
            self.assertEqual(t['cpu'],1);self.assertIsNone(t['require_node'])
            self.assertEqual(t['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
            self.assertFalse(any('/results/' in p for p in t['stage_input_paths']))
        records=[{**submit.task_specification('unit_calibration',spec.PREFLIGHT),'id':'t0','status':'done','exit_code':0}]
        with patch.object(sys,'argv',['submit','--run-name','unit_calibration','--phase','remaining']), \
                patch.object(submit,'inventory',return_value=records), \
                patch.object(submit.Path,'read_text',return_value=json.dumps({'contract':spec.contract()})), \
                patch.dict(sys.modules,{'scheduler':SimpleNamespace(load_state=lambda:{'tasks':records})}),patch.object(submit,'execute_bulk') as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]),31)
            bulk.reset_mock();records[0]['exit_code']=1
            with self.assertRaises(SystemExit):submit.main()
            bulk.assert_not_called()


if __name__ == '__main__':unittest.main()
