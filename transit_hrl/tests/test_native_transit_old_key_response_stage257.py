import copy
import json
import sys
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_old_key_response as learning
from scripts import run_native_transit_old_key_response_stage257 as spec
from scripts import submit_native_transit_old_key_response_stage257_scheduleurm as submit


def metrics(cost):
    return dict(zip(learning.FIELDS, (cost, 2*cost, cost/4, 3*cost/4, float(cost>0), 1.)))


class OldKeyResponseTest(unittest.TestCase):
    def test_common_draw_pairing_root_weights_and_unsafe_mean_indicator(self):
        values = np.zeros((3,16,6,4))
        values[:,:,1] = [1.,.25,.75,2.]
        shared = np.arange(16)[None,:,None]
        values[...,:3] += shared[...,None]*np.array([1.,.25,.75])
        x = learning.combine([learning.action_stats(values[:2],[1,0],[0,0],[397,401]),
            learning.action_stats(values[2:],[1],[0],[397])])
        self.assertAlmostEqual(x['all']['delta_cost'],2/3)
        self.assertAlmostEqual(x['all']['delta_wait_min'],4/3)
        self.assertEqual(x['by_root']['397']['cases'],2)
        self.assertEqual(x['by_root']['397']['delta_cost'],1.)
        self.assertEqual(x['by_root']['401']['changed_rate'],0.)

    def test_path_decomposition_shared_root_violation_and_zero_action_response(self):
        def stats(cost,changed=1.):
            return {'all':{'cases':2,**metrics(cost),'changed_rate':changed},'by_root':{'397':{'cases':2,**metrics(cost)}}}
        paths = dict(zip(learning.PATHS,(stats(1),stats(2),stats(3),stats(-4))))
        halves = [{p:copy.deepcopy(paths[p]) for p in learning.PATHS[1:]} for _ in (0,1)]
        d = learning.diagnose(paths,halves)
        self.assertEqual([d['steps'][s]['delta_cost'] for s in learning.STEPS],[1.,1.,-7.])
        self.assertEqual(d['decomposition_max_abs_difference'],0.)
        self.assertTrue(d['flags']['same_validation_violation_in_both_opposite_halves'])
        self.assertTrue(d['flags']['seen_wait_harm_held_wait_benefit_both_halves'])
        paths['validation'] = stats(0,0.)
        paths['inner_opposite'] = stats(0,0.)
        self.assertTrue(learning.diagnose(paths,halves)['flags']['inner_no_action_full_seen_changes'])
        self.assertIsNone(learning.summarize([],5)['conditional'])

    def cached_cell(self):
        key, arm, mode = spec.KEYS[0], 'free_old_expanded8', 'pooled_safe'
        states = {scope:[[r,s,9] for r in (397,401)] for scope,s in (('train60',1),('held60',2))}
        actions = [{'bank':b,'scope':scope,'states':sts,'choices':{'raw':[0,0],f'prefix_refit_{arm}_{mode}':[2,2] if scope=='train60' else [1,1]}}
            for b in (0,1) for scope,sts in states.items()]
        cached = [{'bank':b,'half':h,'states':states['train60'],'raw_choices':[0,0],
            'choices':{'free_old':{'old':[1,1],'zero':[0,0]}}} for b in (0,1) for h in (0,1)]
        by_root = {str(r):{'cases':2,**metrics(1.),'scaled_cost':1.,'scaled_wait':2.} for r in (397,401)}
        summary = {'pooled':{'score':2.,'by_root':by_root},'halves':[]}
        panels = {}
        for scope,sts in states.items():
            for st in sts:
                v = np.zeros((32,6,4)); v[:,1] = [1.,.25,.75,2.] if scope=='train60' else [-3.,-.75,-2.25,-6.]
                v[:,2] = [2.,.5,1.5,4.]
                panels[tuple(st)] = {'values':v}
        with patch.object(spec.source.learning,'MODELS',('free_old',)),patch.object(spec.source.learning,'SELECTORS',('expanded8',)),patch.object(spec,'FILTERS',(mode,)):
            info = {'protocol':spec.source.EXPERIMENT_PROTOCOL,'contract':spec.source.contract(),'coverage_qualified':True,
                'banks':[{'bank':b,'selection':{'prefix_refit':{arm:{mode:'old'}},'fixed_bundle':{arm:{mode:None}}},
                    'summaries':{arm:{'old':summary}},'matrix':{scope:{'all':{'variants':{f'prefix_refit_{arm}_{mode}':metrics(2. if scope=='train60' else -3.)}}}
                        for scope in states}} for b in (0,1)]}
            def read(path):return json.dumps({'aggregate.json':info,'actions.json':actions,'oof_choices.json':cached}[path.name])
            with patch.object(spec.Path,'read_text',autospec=True,side_effect=read), \
                    patch.object(spec.source.learning,'fit_inner',side_effect=AssertionError('New fit')), \
                    patch.object(spec.source.learning.base,'causal_data',side_effect=AssertionError('New inference')), \
                    patch.object(spec.source.learning,'select',side_effect=AssertionError('Reselection')):
                record = spec.analyze_cell(key,panels)
                actions[0]['choices']['raw'] = [1,0]
                with self.assertRaisesRegex(ValueError,'reference actions differ'):spec.analyze_cell(key,panels)
        return record

    def test_cached_actions_reproduce_validation_and_full_without_fitting(self):
        record = self.cached_cell()
        self.assertEqual(record['reproduction_max_abs_difference'],0.)
        self.assertEqual(len(record['pieces']),2)
        p = record['pieces'][0]
        self.assertEqual([p['paths'][name]['all']['delta_cost'] for name in learning.PATHS],[1.,1.,2.,-3.])
        self.assertTrue(p['diagnosis']['flags']['seen_wait_harm_held_wait_benefit_both_halves'])

    def test_category_bank_view_reproduction_and_empty_strata(self):
        record = self.cached_cell()
        records = [{**record,'key':[view,0]} for view in spec.VIEWS]
        expected = {'groups':{}}
        arm = 'free_old_expanded8_pooled_safe'
        for view in ('mixed_pooled',*spec.VIEWS):
            expected['groups'][view] = {}
            for scope,cost in (('train60',2.),('held60',-3.)):
                expected['groups'][view][scope] = {}
                for label in ('pooled','bank0','bank1'):
                    n = (2 if view=='mixed_pooled' else 1)*(2 if label=='pooled' else 1)
                    expected['groups'][view][scope][label] = {arm:{'bank_cell_exposures':n,'before':metrics(cost),
                        'categories':{c:{'bank_cell_exposures':n if c=='removed' else 0,
                            'before_conditional_mean':metrics(cost) if c=='removed' else None} for c in spec.previous.learning.CATEGORIES}}}
        with patch.object(spec,'ARMS',('free_old_expanded8',)),patch.object(spec,'FILTERS',('pooled_safe',)):
            groups, maximum = spec.summarize(records,expected)
            x = groups['mixed_pooled']['pooled'][arm]
            self.assertEqual(x['removed']['bank_cell_exposures'],4)
            self.assertEqual(x['removed']['full_denominator_contributions']['full_seen']['delta_cost'],2.)
            self.assertIsNone(x['added']['conditional'])
            self.assertEqual(maximum,0.)
            expected['groups']['mixed_pooled']['held60']['pooled'][arm]['categories']['removed']['bank_cell_exposures'] -= 1
            with self.assertRaisesRegex(ValueError,'category denominator'):spec.summarize(records,expected)

    def test_scheduler_completion_gate_not_performance_gate(self):
        task = submit.task_specification('unit_old_key')
        self.assertEqual(task['cpu'],1)
        self.assertIsNone(task['require_node'])
        self.assertEqual(task['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
        self.assertFalse(any('/results/' in p for p in task['stage_input_paths']))
        self.assertEqual(spec.contract(),json.loads(json.dumps(spec.contract())))
        report = {'protocol':spec.previous.EXPERIMENT_PROTOCOL,'contract':spec.previous.contract(),'coverage_qualified':True}
        receipt = {'scheduler':{'status_counts':{'done':1},'exit_code_counts':{'0':1}}}
        def read(p):return json.dumps(receipt if p.name=='completion_receipt.json' else report)
        with patch.object(sys,'argv',['submit','--run-name','unit_old_key']),patch.object(submit,'inventory',return_value=[]), \
                patch.object(submit.Path,'exists',return_value=False),patch.object(submit.Path,'read_text',autospec=True,side_effect=read), \
                patch.object(submit.subprocess,'check_output',return_value='revision'),patch.object(submit,'write_json'),patch.object(submit,'execute_bulk') as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]),1)
            bulk.reset_mock();receipt['scheduler']['exit_code_counts']={'1':1}
            with self.assertRaises(SystemExit):submit.main()
            bulk.assert_not_called()


if __name__=='__main__':unittest.main()
