import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_joint_selection as selection
from scripts import run_native_transit_joint_selection_stage227 as spec
from scripts.submit_native_transit_joint_selection_stage227_scheduleurm import task_specification
from tests import test_native_transit_coverage_value as native_fixture
from tests import test_native_transit_best_action_crossfit_stage220 as old

learning = selection.learning


def source_cell():
    curves = {str(s):{'mse':float(s+1),'joint_score':-.1,'by_root':{'397':{'cases':12,
        'mean_delta_cost':.5,'mean_delta_wait_min':-.1,'mean_scaled_cost':-.49,'mean_scaled_wait':-.1}}} for s in learning.CHECKPOINTS}
    return {'selection':{a:{'mse_step':0,'joint_step':0,'curves':copy.deepcopy(curves)} for a in learning.ANCHORS},
        'inner_fits':[{'held_group':[397,i],'held_states':6,'fits':{a:{'scales':[scale,1.]} for a in learning.ANCHORS}}
            for i,scale in enumerate((1.,100.))]}


class JointSelectionTest(unittest.TestCase):
    def test_actual_sign_reversal_pattern_is_rejected_without_mutating_source(self):
        cell = source_cell()
        before = copy.deepcopy(cell)
        repaired = selection.repair_selection(cell,'raw')
        self.assertIsNone(repaired['joint_step'])
        self.assertEqual(repaired['mse_step'],0)
        self.assertEqual(repaired['source_scales']['397'],[50.5,1.])
        self.assertGreater(repaired['curves']['0']['joint_score'],0.)
        self.assertEqual(cell,before)

    def test_physical_selection_keeps_earlier_ties_and_strict_root_improvement(self):
        cell = source_cell()
        for step in (50,100):
            stats = cell['selection']['raw']['curves'][str(step)]['by_root']['397']
            stats['mean_delta_cost'],stats['mean_delta_wait_min'] = -.5,-.1
        repaired = selection.repair_selection(cell,'raw')
        self.assertEqual(repaired['joint_step'],50)
        for step in (50,100): cell['selection']['raw']['curves'][str(step)]['by_root']['397']['mean_delta_wait_min'] = 0.
        self.assertIsNone(selection.repair_selection(cell,'raw')['joint_step'])

    def test_saved_paths_reselected_without_fit_and_other_choices_unchanged(self):
        rows = native_fixture.bank_fixture()[0]['training_data'][:6]
        panels = old.panels_for(rows)
        with patch.object(learning.binding.precision.base.base,'STEPS',3),patch.object(learning.binding.learned.neural,'STEPS',3),patch.object(learning,'CHECKPOINTS',(0,1,3)):
            bundle,_ = learning.binding.precision.fit_bundle(rows,panels,(0,1,2,3))
            bundle['best'] = {'center':bundle['ridge']['center'],'scale':bundle['ridge']['scale'],'coefficients':[0.]*98}
            cell = source_cell()
            model,_ = learning.fit_final(rows,panels,(0,1,2,3),bundle,cell['selection'])
            cal = {m:{v:{'alpha':1.} for v in learning.binding.WAIT_VARIANTS} for m in ('risk','oof','final','joint')}
            before = learning.predict(rows,model,bundle,cal)
            with patch.object(learning,'fit_path',side_effect=AssertionError('No repeat training')):
                repaired = spec.reselect_cell(rows,rows,panels,cell,model,bundle,cal,'first_last',0)
        decisions = repaired['scopes']['train_same']
        for v in learning.VARIANTS:
            if v.endswith('selected'):
                self.assertEqual([r['decisions'][v]['selected'] for r in decisions],before['choices']['raw'].tolist())
            else:
                self.assertEqual([r['decisions'][v]['selected'] for r in decisions],before['choices'][v].tolist())
        self.assertEqual(model['selection'],cell['selection'])

    def test_controls_budget_roundtrip_and_unbound_scheduler(self):
        self.assertEqual(spec.contract(),json.loads(json.dumps(spec.contract())))
        metric = {'mean_cost':0.,'mean_wait_min':0.,'changed_fraction':0.,'unsafe_fraction':0.}
        def report():
            return {'views':{v:{'scopes':{s:{'variants':{g:metric.copy() for g in learning.VARIANTS},
                'wait_heads':{g:{'physical_mass_loss_min':0.} for g in learning.CONTROLS},
                'joint_prediction_errors':{g:{'normalized_mse':0.} for g in learning.VARIANTS if g.startswith('pair_')}} for s in spec.SCOPES}} for v in spec.VIEWS}}
        a,b = report(),report()
        self.assertEqual(spec.reproduce_controls(a,b),{'comparisons':720,'max_abs_difference':0.})
        b['views']['mixed_a']['scopes']['held_other']['variants']['pair_raw_last']['mean_cost'] = 1.
        with self.assertRaisesRegex(ValueError,'unchanged source controller'): spec.reproduce_controls(a,b)
        self.assertEqual(spec.contract()['budget']['network_updates'],0)
        for key in (spec.KEYS[0],(None,None,None,None)):
            task = task_specification('unit_physical_selection',*key)
            self.assertEqual((task['cpu'],task['ram_mb']),(1,3072))
            self.assertIsNone(task['require_node'])
            self.assertEqual(task['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
            self.assertNotIn('_raw',task['result_dir'])


if __name__=='__main__': unittest.main()
