import copy
import json
import sys
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_selector_bridge as learning
from scripts import run_native_transit_selector_bridge_stage254 as spec
from scripts import submit_native_transit_selector_bridge_stage254_scheduleurm as submit


class SelectorBridgeTest(unittest.TestCase):
    def test_paired_decomposition_and_reference_strata(self):
        values = np.zeros((2, 16, 6, 4))
        values[:, :, 1] = [-2., -.5, -1.5, -4.]
        values[:, :, 2] = [3., 1., 2., 5.]
        validation = values[:, :2].copy()
        validation[:, :, 1] -= 1.
        inner_gate = np.ones((2, 6), bool)
        full_gate = inner_gate.copy(); full_gate[0, 1] = False
        b = learning.compare(validation, values, [1, 1], [2, 1], [0, 0], [2, 0], inner_gate, full_gate)
        x = learning.combine([b])
        means = x['all']['means']
        self.assertEqual(means['optimism']['cost'], 1.)
        self.assertEqual(means['raw_reference']['cost'], 1.5)
        self.assertEqual(means['selected_action_transfer']['cost'], 2.5)
        self.assertEqual(means['correction_transfer']['cost'], 1.)
        self.assertEqual(x['all']['rates']['inner_choice_infeasible_under_full'], .5)
        self.assertEqual(x['raw_same']['cases'], 1)
        self.assertEqual(x['raw_different']['cases'], 1)
        repeated = learning.combine([b, b, b])
        self.assertEqual(repeated['all']['cases'], 6)
        for stratum in learning.STRATA:
            self.assertEqual(x[stratum]['means'], repeated[stratum]['means'])
            self.assertEqual(x[stratum]['rates'], repeated[stratum]['rates'])

    def test_empty_strata_do_not_impute_effects(self):
        v = np.zeros((2, 16, 6, 4)); gate = np.ones((2, 6), bool)
        x = learning.combine([learning.compare(v[:, :2], v, [0, 0], [0, 0], [0, 0], [0, 0], gate, gate)])
        self.assertEqual(x['raw_different']['cases'], 0)
        self.assertIsNone(x['raw_different']['means']['optimism']['cost'])
        self.assertEqual(x['all']['means']['selected_action_transfer']['cost'], 0.)

    def test_cached_cell_reproduces_full_and_validation_means_without_fitting(self):
        key = spec.KEYS[0]
        rows = [{'root':397,'scene_seed':1,'probe_index':p,'scenario':'low_noise'} for p in (8, 9)]
        sts = [list(spec.source.information.old_future.learning.state_key(r)) for r in rows]
        variants = [f'{a}_{f}' for a in spec.ARMS for f in spec.FILTERS]
        selection = {a:{f:'test' for f in spec.FILTERS} for a in spec.ARMS}
        measured = {'delta_cost':-.001,'delta_risk':0.,'delta_service':-.001,'delta_wait_min':-.002}
        info = {'protocol':spec.source.EXPERIMENT_PROTOCOL,'contract':spec.source.contract(),'coverage_qualified':True,
            'banks':[{'bank':b,'selection':selection,'matrix':{'train60':{'all':{'variants':{v:measured for v in variants}}}},
                'summaries':{a:{'test':{'pooled':{'by_root':{'397':{'delta_cost':-.001,'delta_wait_min':-.002}}}}} for a in spec.ARMS}}
                for b in (0, 1)]}
        actions = [{'bank':b,'scope':'train60','states':sts,'choices':{'raw':[0,0],**{v:[1,1] for v in variants}}} for b in (0,1)]
        cached = [{'bank':b,'held_group':[397,1],'half':h,'states':sts,'raw_choices':[0,0],
            'choices':{m:{'test':[1,1],'zero':[0,0]} for m in spec.source.learning.MODELS}} for b in (0,1) for h in (0,1)]
        values = np.zeros((32,6,4)); values[:,1] = [-.001,0.,-.001,-.002]
        panels = {tuple(st):{'values':values} for st in sts}
        def read(p):
            payload = {'aggregate.json':info,'actions.json':actions,'oof_choices.json':cached,
                'models.json':{'reference':{},'references':{'0':{},'1':{}}},
                'result.json':{'cell':{'trials':[{'removed_group':[397,1],'half':h} for h in (0,1)]}}}
            return json.dumps(payload[p.name])
        def data(rs, reference):
            return {'raw_choices':np.zeros(len(rs),dtype=int),'raw_logits':np.zeros((len(rs),6))}
        with patch.object(spec.Path,'read_text',autospec=True,side_effect=read), \
                patch.object(spec.source.training.cohort,'split_rows',return_value=(rows,[])), \
                patch.object(spec.source.learning.joint.binding.policy,'prefix_folds',return_value=[((397,1),[],[0,1])]), \
                patch.object(spec.source.learning.base,'causal_data',side_effect=data), \
                patch.object(spec.source.learning,'select',side_effect=AssertionError('Reselection')), \
                patch.object(spec.source.learning,'rebuild_inner',side_effect=AssertionError('Refitting')):
            actual = spec.analyze_cell(key,{key[0]:rows},panels,panels)
            self.assertLess(actual['reproduction_max_abs_difference'],1e-12)
            self.assertEqual(actual['action_exposures'],96)
            info['banks'][0]['matrix']['train60']['all']['variants']['free_old_old_A2_pooled_safe'] = {**measured,'delta_cost':1.}
            with self.assertRaisesRegex(ValueError,'Changed cached full'):spec.analyze_cell(key,{key[0]:rows},panels,panels)

    def test_summary_keeps_bank_view_and_prefix_units(self):
        v = np.zeros((2,16,6,4)); gate = np.ones((2,6),bool)
        b = learning.compare(v[:,:2],v,[0,0],[0,0],[0,0],[0,0],gate,gate)
        pieces = {f'{a}_{f}':b for a in spec.ARMS for f in spec.FILTERS}
        records = [{'key':[view,0], 'blocks':[{'bank':bank,'prefix':[397,1],'scenario':'low_noise','pieces':pieces} for bank in (0,1)]}
            for view in spec.VIEWS]
        r = spec.summarize(records)
        arm = 'free_old_expanded8_pooled_safe'
        self.assertEqual(r['groups']['mixed_pooled']['pooled'][arm]['all']['cases'],8)
        self.assertEqual(r['groups']['mixed_pooled']['bank0'][arm]['all']['cases'],4)
        self.assertEqual(r['mixed_by_prefix']['397/1'][arm]['cases'],8)
        self.assertEqual(spec.contract(),json.loads(json.dumps(spec.contract())))

    def test_scheduler_requires_complete_confirmation_not_successful_claim(self):
        task = submit.task_specification('unit_bridge')
        self.assertEqual(task['cpu'],1)
        self.assertIsNone(task['require_node'])
        self.assertEqual(task['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
        self.assertNotIn('_raw',task['result_dir'])
        self.assertFalse(any('/results/' in p for p in task['stage_input_paths']))
        report = {'protocol':spec.confirmation.EXPERIMENT_PROTOCOL,'contract':spec.confirmation.contract(),
            'coverage_qualified':True,'confirmation':{'joint_pass':False}}
        receipt = {'scheduler':{'status_counts':{'done':2829},'exit_code_counts':{'0':2829}}}
        def read(p):return json.dumps(receipt if p.name=='completion_receipt.json' else report)
        with patch.object(sys,'argv',['submit','--run-name','unit_bridge']), patch.object(submit,'inventory',return_value=[]), \
                patch.object(submit.Path,'exists',return_value=False),patch.object(submit.Path,'read_text',autospec=True,side_effect=read), \
                patch.object(submit.subprocess,'check_output',return_value='test_revision'),patch.object(submit,'write_json'), \
                patch.object(submit,'execute_bulk') as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]),1)
            bulk.reset_mock();receipt['scheduler']['exit_code_counts']={'0':2828,'1':1}
            with self.assertRaises(SystemExit):submit.main()
            bulk.assert_not_called()


if __name__=='__main__':unittest.main()
