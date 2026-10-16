import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_prefix_influence as diagnostic
from scripts import run_native_transit_prefix_influence_stage259 as spec
from scripts import submit_native_transit_prefix_influence_stage259_scheduleurm as submit


class PrefixInfluenceTest(unittest.TestCase):
    def test_regions_partition_retained_and_cross_root(self):
        states = [[r, s, i] for r in (397, 401) for s in range(5) for i in range(6)]
        masks = diagnostic.regions(states, 'train60', [397, 2])
        self.assertEqual({k:int(v.sum()) for k,v in masks.items()},
            {'excluded':6,'retained_same_root':24,'retained_other_root':30,'retained':54})
        self.assertTrue(np.all(masks['excluded'] ^ masks['retained']))
        self.assertFalse(np.any(masks['retained_same_root'] & masks['retained_other_root']))
        held = diagnostic.regions(states, 'held60', [397, 2])
        self.assertEqual([int(held[k].sum()) for k in held], [30,30,60])
        single = diagnostic.regions(states[:30], 'train60', [397, 2])
        self.assertFalse(single['retained_other_root'].any())

    def test_response_uses_common_futures_and_keeps_absolute_influence(self):
        v = np.zeros((2,16,6,4))
        v[0,:,1] = (4,1,3,2)
        v[1,:,1] = (-4,-1,-3,-2)
        s = diagnostic.outcomes(v, [1,1], [0,0], [0,0])
        self.assertEqual(s['effect']['delta_cost'],0.)
        self.assertEqual(s['absolute_state_effect']['delta_cost'],4.)
        self.assertEqual(s['disagreements'],2)
        piece = {'cell':['mixed_a',0,'even_odd',0],'bank':0,'regions':{'retained':{
            'whole':s,'first8':s,'last8':s,'function_gap':{'feasible_action_cases':2,
                'squared_sums':{'cost':8.,'service':2.,'risk':2.}}}}}
        r = diagnostic.summarize([piece])['regions']['retained']
        self.assertEqual(r['effect']['delta_cost'],0.)
        self.assertEqual(r['absolute_state_effect']['delta_cost'],4.)
        self.assertEqual(r['absolute_prefix_mean_effect']['delta_cost'],0.)
        self.assertEqual(r['feasible_prediction_rms']['cost'],2.)
        self.assertEqual(diagnostic.summarize([])['bank_cell_exposures'],0)

    def test_prediction_gap_uses_only_feasible_nonidentity_actions(self):
        d = {'raw_logits':np.array([[0.,0.,1.,1.,1.,1.]]),'originals':np.array([0])}
        fp = {'cost':np.zeros((1,6)),'components':np.zeros((1,6,2))}
        p = {'cost':np.array([[99.,3.,99.,99.,99.,99.]]),'components':np.full((1,6,2),99.)}
        p['components'][0,1] = [1.,2.]
        gap = diagnostic.function_gap(d,p,fp,np.array([True]))
        self.assertEqual(gap,{'feasible_action_cases':1,'squared_sums':{'cost':9.,'service':1.,'risk':4.}})

    def fixture(self):
        train = [{'root':r,'scene_seed':s,'probe_index':i} for r in (397,401) for s in range(5) for i in range(6)]
        held = [{**r,'scene_seed':r['scene_seed']+10} for r in train]
        states = lambda rs:[[r['root'],r['scene_seed'],r['probe_index']] for r in rs]
        variant = 'free_old_expanded8_pooled_safe'
        pieces, models, excluded, actions = [],[],[],[]
        for bank in (0,1):
            pieces.append({'bank':bank,'arm':'free_old_expanded8','filter':'pooled_safe','old_key':'old','category':'removed',
                'old_key_new_gate':{'strict_gain_only':False},
                'paths':{'inner16':{'all':dict(zip(diagnostic.FIELDS,(2.,4.,.5,1.5,1.,1.)))},
                    'full_seen':{'all':dict(zip(diagnostic.FIELDS,(1.,2.,.25,.75,1.,1.)))}},
                'action_disagreement':{'inner16_full':1.}})
            pieces[-1]['evaluation_halves'] = [pieces[-1]['paths']]*2
            ds,_ = spec.source.sampling.learning.draw_assignment(bank,'expanded')
            for root in (397,401):
                for seed in range(5):
                    group = [root,seed]
                    models.append({'bank':bank,'held_group':group,'training_draws':ds,'models':{'free_old':{'choice':2}}})
                    excluded.append({'bank':bank,'held_group':group,'training_draws':ds,
                        'states':[s for s in states(train) if s[:2]==group], 'choices':{'free_old':{'old':[2]*6,'zero':[0]*6}}})
            for scope,rows in (('train60',train),('held60',held)):
                actions.append({'bank':bank,'scope':scope,'states':states(rows),
                    'choices':{'raw':[0]*60,f'prefix_refit_{variant}':[1]*60}})
        info = {'protocol':spec.source.EXPERIMENT_PROTOCOL,'contract':spec.source.contract(),'coverage_qualified':True,'pieces':pieces}
        anchor = {'banks':[{'bank':b,'matrix':{s:{'all':{'variants':{f'prefix_refit_{variant}':pieces[b]['paths']['full_seen']['all']}}}
            for s in ('train60','held60')}} for b in (0,1)]}
        def read(run,key,file):
            if run == spec.SOURCE_RUN:return {'aggregate.json':info,'models.json':models,'actions.json':excluded}[file]
            if run == spec.source.CACHE_RUN:return {'actions.json':actions,'aggregate.json':anchor}[file]
            return {'models.json':{'reference':{},'0':{'free_old':{'choice':1}},'1':{'free_old':{'choice':1}}}}[file]
        return train,held,read

    def freeze_fixture(self):
        train,held,read = self.fixture()
        def candidates(data,model):
            n = len(data['originals'])
            return {k:{'chosen':np.full(n,0 if k=='zero' else model['choice']),
                'cost':np.full((n,6),model['choice'],dtype=float),'components':np.full((n,6,2),model['choice']/2)}
                for k in ('zero','old')}
        with patch.object(spec,'read_cell',side_effect=read),patch.object(spec.source.cached.training.cohort,'split_rows',return_value=(train,held)), \
                patch.object(spec.source.learning,'MODELS',('free_old',)), \
                patch.object(spec.source.learning.base,'causal_data',side_effect=lambda rs,ref:{'raw_choices':np.zeros(len(rs),int),
                    'raw_logits':np.zeros((len(rs),6)),'originals':np.zeros(len(rs),int)}), \
                patch.object(spec.source.learning.base,'candidates',side_effect=candidates), \
                patch.object(spec.source.learning,'fit_inner',side_effect=AssertionError('Unexpected fit')), \
                patch.object(spec.source.learning,'select',side_effect=AssertionError('Unexpected selection')),patch.object(spec,'write_json') as write:
            spec.freeze_cell('unit_influence',spec.source.PREFLIGHT,{spec.source.PREFLIGHT[0]:train})
            frozen = write.call_args.args[1]
            self.assertEqual(write.call_args.args[0].name,'actions.json')
        return frozen,read

    def test_saved_models_inference_and_cached_action_reproduction(self):
        frozen,_ = self.freeze_fixture()
        self.assertEqual(len(frozen),40)
        self.assertEqual(sum(c['scope']=='held60' for c in frozen),20)
        self.assertEqual(len(frozen[0]['inner']),60)

    def test_evaluation_reproduces_excluded_and_full_anchors(self):
        frozen,read = self.freeze_fixture()
        v = np.zeros((32,6,4))
        for a in range(6):v[:,a] = (a,a/4,3*a/4,2*a)
        panels = {tuple(st):{'values':v} for c in frozen for st in c['states']}
        def cache(run,key,file):return frozen if run=='unit_influence' else read(run,key,file)
        with patch.object(spec,'read_cell',side_effect=cache):
            r = spec.evaluate_cell('unit_influence',spec.source.PREFLIGHT,panels)
            self.assertEqual(r['reproduction_max_abs_difference'],0.)
            self.assertEqual(len(r['pieces']),20)
            summary = diagnostic.summarize(r['pieces'])
            self.assertEqual(summary['bank_cell_exposures'],2)
            self.assertEqual(summary['regions']['excluded']['state_model_exposures'],120)
            self.assertEqual(summary['regions']['retained_other_root']['action_disagreement_rate'],1.)
            self.assertEqual(summary['regions']['held']['effect']['delta_cost'],1.)
            frozen[0]['inner'] = [1]*60
            with self.assertRaisesRegex(ValueError,'reproduction'):spec.evaluate_cell('unit_influence',spec.source.PREFLIGHT,panels)
        with patch.object(spec,'PRIMARY',('free_old_expanded8_pooled_safe',)):
            report = spec.report([r])
        removed = report['groups']['mixed_pooled']['pooled']['free_old_expanded8_pooled_safe']['removed']
        self.assertEqual(removed['bank_cell_exposures'],2)
        self.assertEqual(removed['regions']['held']['state_model_exposures'],1200)

    def test_all_actions_frozen_before_panel_read(self):
        events=[]
        def panel(*args):
            self.assertEqual(events,['freeze']*32)
            events.append('panels');return {}
        with patch('sys.argv',['runner','--run-name','unit_influence']), \
                patch.object(spec.source.cached.training.cohort.source.source,'load_data',return_value=({'training_data':[]},None,[],None)), \
                patch.object(spec.source.cached.training.cohort.source,'matched_views',return_value={}), \
                patch.object(spec,'freeze_cell',side_effect=lambda *args:events.append('freeze')), \
                patch.object(spec.source.sampling,'read_panels',side_effect=panel), \
                patch.object(spec,'evaluate_cell',return_value={}),patch.object(spec,'report',return_value={}),patch.object(spec,'write_json'):
            spec.main()
        self.assertEqual(events[-1],'panels')

    def test_scheduler_single_unpinned_task_and_completed_parent(self):
        task = submit.task_specification('unit_influence')
        self.assertEqual(task['cpu'],1)
        self.assertIsNone(task['require_node'])
        self.assertEqual(task['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
        self.assertFalse(any('/results/' in p for p in task['stage_input_paths']))
        parent = {'protocol':spec.source.EXPERIMENT_PROTOCOL,'contract':spec.source.contract(),'coverage_qualified':True}
        receipt = {'scheduler':{'status_counts':{'done':33},'exit_code_counts':{'0':33}}}
        with patch('sys.argv',['submit','--run-name','unit_influence']),patch.object(submit,'inventory',return_value=[]), \
                patch.object(submit.Path,'exists',return_value=False), \
                patch.object(submit.Path,'read_text',autospec=True,side_effect=lambda p:json.dumps(receipt if p.name=='completion_receipt.json' else parent)), \
                patch.object(submit.subprocess,'check_output',return_value='revision'),patch.object(submit,'write_json'),patch.object(submit,'execute_bulk') as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]),1)
            bulk.reset_mock();receipt['scheduler']['exit_code_counts']={'1':33}
            with self.assertRaises(SystemExit):submit.main()
            bulk.assert_not_called()


if __name__ == '__main__':unittest.main()
