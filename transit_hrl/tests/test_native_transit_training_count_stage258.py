import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_training_count_response as diagnostic
from scripts import run_native_transit_training_count_stage258 as spec
from scripts import submit_native_transit_training_count_stage258_scheduleurm as submit


def metrics(cost):
    return dict(zip(diagnostic.FIELDS,(cost,2*cost,cost/4,3*cost/4,float(cost>0),float(cost!=0))))


def stats(cost):
    return {'all':{'cases':2,**metrics(cost)},'by_root':{'397':{'cases':2,**metrics(cost)}}}


class TrainingCountTest(unittest.TestCase):
    def test_nested_training_and_disjoint_evaluation(self):
        for bank in (0,1):
            full,ev = spec.sampling.learning.draw_assignment(bank,'expanded')
            blocks = [spec.sampling.learning.draw_assignment(bank,'expanded',h)[0] for h in (0,1)]
            self.assertEqual((len(full),len(ev)),(16,16))
            self.assertEqual(blocks[0]+blocks[1],full)
            self.assertFalse(set(full)&set(ev))
            self.assertTrue(all(set(b)<set(full) for b in blocks))

    def test_absolute_gaps_detect_mean_cancellation(self):
        pieces = []
        for sign in (-1,1):
            paths = dict(zip(diagnostic.PATHS,(stats(sign),stats(2*sign),stats(0))))
            pieces.append({'paths':paths,'effects':diagnostic.contrasts(paths),
                'evaluation_halves':[paths,paths],'training_halves':[stats(sign-.1),stats(sign+.1)],
                'action_disagreement':{'inner8_full':.2,'inner16_full':.3,'inner16_inner8':.1},'full_held_anchor':metrics(0)})
        x = diagnostic.summarize(pieces,4)
        self.assertEqual(x['conditional']['contrasts']['count16_minus8']['delta_cost'],0.)
        self.assertEqual(x['conditional']['absolute_full_response_gap']['inner8']['delta_cost'],1.)
        self.assertEqual(x['conditional']['absolute_full_response_gap']['inner16']['delta_cost'],2.)
        self.assertEqual(x['bank_cell_exposures'],2)
        self.assertIsNone(diagnostic.summarize([],4)['conditional'])

    def evaluate_fixture(self):
        arm,mode = 'free_old_expanded8','pooled_safe'
        states = [[397,7,8],[397,7,9]]
        old8,new16,full,banks = [],[],[],[]
        for b in (0,1):
            ds,_ = spec.sampling.learning.draw_assignment(b,'expanded')
            new16.append({'bank':b,'held_group':[397,7],'training_draws':ds,'states':states,'raw_choices':[0,0],
                'choices':{'free_old':{'old':[3,3],'zero':[0,0]}}})
            for h in (0,1):
                dh,_ = spec.sampling.learning.draw_assignment(b,'expanded',h)
                old8.append({'bank':b,'half':h,'held_group':[397,7],'training_draws':dh,'states':states,'raw_choices':[0,0],
                    'choices':{'free_old':{'old':[1+h,1+h],'zero':[0,0]}}})
            full.append({'bank':b,'scope':'train60','states':states,'choices':{'raw':[0,0],f'prefix_refit_{arm}_{mode}':[4,4]}})
            root={'cases':4,**metrics(2),'scaled_cost':2.,'scaled_wait':4.}
            banks.append({'bank':b,'selection':{'prefix_refit':{arm:{mode:'old'}},'fixed_bundle':{arm:{mode:None}}},
                'summaries':{arm:{'old':{'pooled':{'score':4.,'by_root':{'397':root}},'halves':[]}}},
                'matrix':{scope:{'all':{'variants':{f'prefix_refit_{arm}_{mode}':metrics(cost)}}} for scope,cost in (('train60',4),('held60',-1))}})
        v = np.zeros((32,6,4))
        for a,cost in enumerate((0,1,3,2,4,0)):v[:,a] = (cost,cost/4,3*cost/4,2*cost)
        panels={tuple(st):{'values':v} for st in states}
        with patch.object(spec.learning,'MODELS',('free_old',)),patch.object(spec.learning,'SELECTORS',('expanded8',)),patch.object(spec,'FILTERS',(mode,)), \
                patch.object(spec.learning,'fit_inner',side_effect=AssertionError('Cached evaluation refit')), \
                patch.object(spec.learning,'select',side_effect=AssertionError('Reselection')), \
                patch.object(spec.learning.base,'causal_data',side_effect=AssertionError('Cached evaluation inference')):
            pieces,maximum = spec.evaluate_cell({'banks':banks},old8,full,new16,panels)
            old8[0]['states'] = list(reversed(states))
            with self.assertRaisesRegex(ValueError,'state order'):spec.evaluate_cell({'banks':banks},old8,full,new16,panels)
        return pieces,maximum

    def test_cached_halves_and_frozen_key_evaluation(self):
        pieces,maximum = self.evaluate_fixture()
        self.assertEqual(maximum,0.)
        self.assertEqual(len(pieces),2)
        p=pieces[0]
        self.assertEqual([p['paths'][x]['all']['delta_cost'] for x in diagnostic.PATHS],[2.,2.,4.])
        self.assertEqual([x['all']['delta_cost'] for x in p['training_halves']],[1.,3.])
        s=diagnostic.summarize([p],2)['conditional']
        self.assertEqual([h['delta_cost'] for h in s['training_half_count_effects']],[1.,-1.])
        self.assertEqual(p['action_disagreement']['inner16_inner8'],1.)

    def test_only_inner16_fits_and_freezes_before_evaluation(self):
        rows=[{'root':397,'scene_seed':s,'probe_index':9} for s in (7,8,8)]
        fit={'fits':[{'optimality':0.,'difference_optimality':0.,'auxiliary_free_fit':{'optimality':0.,'norm':1.},'norm':1.,'old_radius':1.}]*4,
            'subspace_max_abs_difference':0.}
        writes,draws=[],[]
        info={'protocol':spec.bundle.EXPERIMENT_PROTOCOL,'contract':spec.bundle.contract(),'coverage_qualified':True}
        def read(p):return json.dumps({'aggregate.json':info,'oof_choices.json':[],'actions.json':[],
            'models.json':{'full':{'all_mse':{'past':True}},'reference':{}},'fits.json':{'full':{}}}[p.name])
        def fitted(rs,data,values,prior,ledger,geometry):
            self.assertTrue(all(r['scene_seed']!=7 for r in rs))
            self.assertEqual(values.shape[1],16)
            self.assertEqual(prior,{'past':True})
            self.assertEqual(geometry,'full_geometry')
            return {m:{} for m in spec.learning.MODELS},fit
        def frozen(*args):
            self.assertEqual(writes,['models.json','actions.json'])
            return [],0.
        def training(rs,panels,ds):
            draws.append(ds)
            return np.zeros((len(rs),len(ds),6,4))
        with patch.object(spec.Path,'read_text',autospec=True,side_effect=read), \
                patch.object(spec.cached.training.cohort,'split_rows',return_value=(rows,[])), \
                patch.object(spec.learning.joint.binding.policy,'prefix_folds',return_value=[((397,7),[1,2],[0])]), \
                patch.object(spec.learning.base,'causal_data',side_effect=lambda rs,ref:{'raw_choices':np.zeros(len(rs),dtype=int)}), \
                patch.object(spec.learning,'deployment_geometry',return_value='full_geometry'), \
                patch.object(spec.sampling,'read_panels',return_value={}), \
                patch.object(spec.learning.base,'training_draws',side_effect=training), \
                patch.object(spec.learning,'fit_inner',side_effect=fitted) as fitting, \
                patch.object(spec.learning.base,'candidates',side_effect=lambda d,m:{'zero':{'chosen':d['raw_choices']}}), \
                patch.object(spec,'write_json',side_effect=lambda p,value:writes.append(p.name)), \
                patch.object(spec,'evaluate_cell',side_effect=frozen):
            spec.fit_cell('unit_count',spec.PREFLIGHT,{spec.PREFLIGHT[0]:rows})
        self.assertEqual(fitting.call_count,2)
        self.assertEqual(draws,[list(range(16)),list(range(16,32))])

    def test_report_preserves_categories_and_reproduces_stage257(self):
        pieces,_=self.evaluate_fixture()
        run='unit_count'
        expected={'groups':{}}
        arm='free_old_expanded8_pooled_safe'
        for group in ('mixed_pooled',*spec.VIEWS):
            expected['groups'][group]={}
            for label in ('pooled','bank0','bank1'):
                n=(16 if group=='mixed_pooled' else 8)*(2 if label=='pooled' else 1)
                expected['groups'][group][label]={arm:{cat:{'bank_cell_exposures':n if cat in ('all','removed','removed_point_harm') else 0,
                    'conditional':{'path_means':{'inner_opposite':metrics(2),'full_seen':metrics(4),'full_held':metrics(-1)},'inner_full_action_disagreement':1.}}
                    for cat in spec.CATEGORIES}}
        with patch.object(spec,'ARMS',('free_old_expanded8',)),patch.object(spec,'FILTERS',('pooled_safe',)):
            payloads={str(spec.raw_directory(spec.result_path(run,*k))/'aggregate.json'):{'protocol':spec.EXPERIMENT_PROTOCOL,'contract':spec.contract(),
                'key':list(k),'coverage_qualified':True,'pieces':pieces,'reproduction_max_abs_difference':0.,
                'fits':{'inner16_fits':20,'inner_solves':160,'subspace_max_abs_difference':0.,'max_kkt':0.,'radius_max_excess':0.}} for k in spec.KEYS}
            def read(p):return json.dumps(expected if p.name=='result.json' else payloads[str(p)])
            with patch.object(spec.Path,'read_text',autospec=True,side_effect=read),patch.object(spec,'write_json') as write:
                spec.report(run)
                r=write.call_args.args[1]
                self.assertEqual(r['fits']['inner_solves'],5120)
                self.assertEqual(r['reproduction']['category_max_abs_difference'],0.)
                self.assertEqual(r['groups']['mixed_pooled']['pooled'][arm]['removed']['bank_cell_exposures'],32)
                expected['groups']['mixed_pooled']['pooled'][arm]['removed']['conditional']['path_means']['inner_opposite']['delta_cost']+=1
                with self.assertRaisesRegex(ValueError,'Stage257 reproduction'):spec.report(run)

    def test_scheduler_budget_and_successful_preflight_gate(self):
        self.assertEqual(spec.contract(),json.loads(json.dumps(spec.contract())))
        self.assertEqual(spec.contract()['budget']['inner_solves'],32*2*10*4*2)
        self.assertEqual(spec.contract()['budget']['inner8_refits'],0)
        for key in (spec.PREFLIGHT,None):
            task=submit.task_specification('unit_count',key)
            self.assertEqual(task['cpu'],1)
            self.assertIsNone(task['require_node'])
            self.assertEqual(task['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
            self.assertFalse(any('/results/' in p for p in task['stage_input_paths']))
        records=[{**submit.task_specification('unit_count',spec.PREFLIGHT),'id':'t0','status':'done','exit_code':0}]
        with patch.object(sys,'argv',['submit','--run-name','unit_count','--phase','remaining']),patch.object(submit,'inventory',return_value=records), \
                patch.object(submit.Path,'read_text',return_value=json.dumps({'contract':spec.contract()})), \
                patch.dict(sys.modules,{'scheduler':SimpleNamespace(load_state=lambda:{'tasks':records})}),patch.object(submit,'execute_bulk') as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]),31)
            bulk.reset_mock();records[0]['exit_code']=1
            with self.assertRaises(SystemExit):submit.main()
            bulk.assert_not_called()


if __name__=='__main__':unittest.main()
