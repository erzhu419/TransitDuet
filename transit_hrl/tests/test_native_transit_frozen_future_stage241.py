import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_frozen_future as learning
from scripts import run_native_transit_frozen_future_stage241 as spec
from scripts.submit_native_transit_frozen_future_stage241_scheduleurm import task_specification, require_completed
from tests import test_native_transit_refit_consistency_stage240 as fixtures


class FrozenFutureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.RefitConsistencyTest.setUpClass()
        train, _, panels, _, refs, refit, source, models = fixtures.RefitConsistencyTest.args
        cls.args = train, panels, refit, refs['references'], source, models

    def test_cached_actions_reproduce_without_training_or_selection(self):
        with patch.object(learning.joint, 'fit_reference', side_effect=AssertionError('Raw training')), \
                patch.object(learning.previous, 'fit_paths', side_effect=AssertionError('Residual training')), \
                patch.object(learning.previous, 'choose', side_effect=AssertionError('Reselection')):
            sealed = learning.seal_cell(*self.args)
        self.assertEqual(sealed, json.loads(json.dumps(sealed)))
        self.assertEqual(sealed['reproduction']['records'], 120)
        self.assertLessEqual(sealed['reproduction']['max_abs_difference'], 1e-12)
        self.assertEqual(len(sealed['blocks']), 6)
        for block in sealed['blocks']:
            self.assertEqual(set(block['choices']), set(learning.VARIANTS))
            self.assertEqual(block['old_oof2']['variants']['raw']['delta_cost'], 0.)

    def test_new_labels_cannot_change_frozen_actions_or_old_outcomes(self):
        sealed = learning.seal_cell(*self.args)
        before = copy.deepcopy(sealed)
        panels = {k: x['values'].copy() for k, x in self.args[1].items()}
        first = learning.evaluate_cell(sealed, panels)
        block = sealed['blocks'][0]
        state, chosen = next((i, js[i]) for js in block['choices'].values()
            for i in range(len(js)) if js[i] != block['choices']['raw'][i])
        panels[learning.state_key(block['states'][state])][:, chosen, 0] += 100.
        with patch.object(learning.previous, 'predict', side_effect=AssertionError('New inference')):
            second = learning.evaluate_cell(sealed, panels)
        self.assertEqual(sealed, before)
        self.assertEqual(first['blocks'][0]['scopes']['old_oof2'], second['blocks'][0]['scopes']['old_oof2'])
        self.assertNotEqual(first['blocks'][0]['scopes']['fresh8'], second['blocks'][0]['scopes']['fresh8'])
        self.assertEqual(second, json.loads(json.dumps(second)))

    def test_paired_units_risk_definitions_and_half_averaging(self):
        choices = {v: [0] for v in learning.VARIANTS}
        choices['joint_raw_half_safe'] = [1]
        block = {'states': [{'root': 397, 'scene_seed': 1, 'probe_index': 9}], 'choices': choices,
            'feasible': [[True, True, False, False, False, False]]}
        values = np.zeros((8, 1, 6, 4))
        values[:4, 0, 1, 0], values[4:, 0, 1, 0] = -4., 2.
        values[:4, 0, 1, 3], values[4:, 0, 1, 3] = -3., 1.
        full = learning.outcomes(block, values)['variants']['joint_raw_half_safe']
        self.assertEqual(full['delta_cost'], -1.)
        self.assertEqual(full['delta_wait_min'], -1.)
        self.assertEqual(full['delta_unsafe'], 0.)
        self.assertEqual(full['delta_draw_wait_harm_rate'], .5)
        self.assertEqual(full['gate_cost_regret'], 0.)
        halves = [learning.outcomes(block, values[ds])['variants']['joint_raw_half_safe'] for ds in (slice(0, 4), slice(4, 8))]
        self.assertEqual(np.mean([p['delta_cost'] for p in halves]), full['delta_cost'])
        self.assertNotEqual(np.mean([p['delta_unsafe'] for p in halves]), full['delta_unsafe'])
        with self.assertRaisesRegex(ValueError, 'dimensions'): learning.outcomes(block, values[:, :, :, :2])

    def test_seed_coverage_disjointness_and_closed_cohort(self):
        keys = spec.cells()
        self.assertEqual(len(keys), 1920)
        self.assertEqual(len(spec.cells(preflight=True)), 10)
        self.assertEqual({k[:4] for k in keys}, set(spec.coverage.states()))
        seeds = {spec.continuation_seed(*k) for k in keys}
        old = {spec.coverage.continuation_seed(r, s, i, p, d) for r in spec.ROOTS for s in spec.SCENARIOS
            for i in range(10) for p in spec.coverage.PROBES for d in range(8)}
        self.assertEqual(len(seeds), len(keys))
        self.assertFalse(seeds & old)
        c = spec.contract()
        self.assertEqual(c, json.loads(json.dumps(c)))
        self.assertEqual(c['budget']['new_prefix_scenes'], 0)
        self.assertEqual(c['budget']['new_raw_reference_fits'], 0)
        self.assertEqual(c['budget']['new_residual_fits'], 0)

    def test_physical_draw_qualification(self):
        root, scenario, replica, probe, draw = spec.cells(preflight=True)[0]
        labels = [{'candidate': i, **dict.fromkeys(spec.coverage.expected.LABEL_FIELDS, 0.)} for i in range(6)]
        labels[1].update(actual_cost_delta=-.2, nonfleet_cost_delta=-.2, physical_return_delta=20., wait_delta_min=-1.)
        result = {'protocol': spec.EXPERIMENT_PROTOCOL, 'contract': spec.contract(), 'software_qualified': True,
            'root': root, 'scenario': scenario, 'replica': replica, 'probe_index': probe, 'draw': draw,
            'scene_seed': spec.coverage.noise.source.scene_seed(root, scenario, replica),
            'continuation_seed': spec.continuation_seed(root, scenario, replica, probe, draw),
            'original_source_reproductions': 1, 'exact_resampled_identities': 1, 'prefix_state_rng_matches': 7,
            'frozen_weight_qualified_episodes': 8, 'window_ledger_qualified_episodes': 8, 'continuation_qualified': True,
            'native_clock_ticks': 8*61380, 'training_updates': 0, 'labels': labels, 'existing_labels': copy.deepcopy(labels),
            'original_choice': 0}
        spec.qualify_draw(json.loads(json.dumps(result)))
        result['labels'][1]['physical_return_delta'] = -20.
        with self.assertRaisesRegex(ValueError, 'physical target'): spec.qualify_draw(result)

    def test_report_aggregation_prefixes_and_unbound_scheduler(self):
        sealed = learning.seal_cell(*self.args)
        cell = learning.evaluate_cell(sealed, {k: x['values'] for k, x in self.args[1].items()})
        report = learning.aggregate([cell, cell])
        self.assertEqual(report['fresh8']['all']['matched_blocks'], 12)
        self.assertEqual(report, json.loads(json.dumps(report)))
        effects = learning.prefix_effects(cell)
        self.assertEqual(effects, json.loads(json.dumps(effects)))
        for operation, key in (('seal', None), ('report', None), ('collect', spec.cells()[0])):
            task = task_specification('unit_frozen_future', operation=operation, key=key)
            self.assertEqual(task['cpu'], 1)
            self.assertIsNone(task['require_node'])
            self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
            self.assertNotIn('_raw', task['result_dir'])
            self.assertFalse(any('/results/' in p for p in task['stage_input_paths']))
        existing = [{'id': 't1', 'signature': 'sealed', 'status': 'done'}]
        require_completed(existing, [{'signature': 'sealed'}], {'t1': {'status': 'done', 'exit_code': 0}})
        with self.assertRaises(SystemExit):
            require_completed(existing, [{'signature': 'sealed'}], {'t1': {'status': 'done', 'exit_code': 1}})


if __name__ == '__main__': unittest.main()
