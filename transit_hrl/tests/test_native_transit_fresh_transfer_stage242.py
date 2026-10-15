import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_fresh_transfer as learning
from scripts import run_native_transit_fresh_transfer_stage242 as spec
from scripts.submit_native_transit_fresh_transfer_stage242_scheduleurm import task_specification
from tests import test_native_transit_refit_consistency_stage240 as fixtures


class FreshTransferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.RefitConsistencyTest.setUpClass()
        cls.args = fixtures.RefitConsistencyTest.args
        cls.consistency, cls.models = learning.previous.fit_cell(*cls.args)
        train, held, _, reference, refs, refit, source, cached = cls.args
        cls.freeze_args = train, held, reference, refs, refit, source, cached, cls.models

    def test_no_fitting_or_reselection_and_all_cached_endpoints_match(self):
        with patch.object(learning.joint, 'fit_reference', side_effect=AssertionError('Raw refit')), \
                patch.object(learning.previous, 'fit_bundle', side_effect=AssertionError('Residual refit')), \
                patch.object(learning.previous.previous, 'choose', side_effect=AssertionError('Reselection')):
            frozen = learning.freeze(*self.freeze_args)
            result = learning.evaluate(frozen, self.args[2], self.args[2], self.args[6], self.consistency)
        self.assertEqual(result, json.loads(json.dumps(result)))
        self.assertEqual(result['reproduction']['records'], 3780)
        self.assertLessEqual(result['reproduction']['max_abs_difference'], 1e-12)
        self.assertEqual(len(result['blocks']), 6)
        actions = learning.action_cache(frozen)
        self.assertEqual(actions, json.loads(json.dumps(actions)))
        self.assertEqual(set(actions[0]['choices']), set(learning.ENDPOINTS))

    def test_original_future_sealed_actions_and_poisoned_future_invariance(self):
        train, _, old, _, refs, refit, source, cached = self.args
        sealed = spec.source.learning.seal_cell(train, old, refit, refs['references'], source, cached)
        frozen = learning.freeze(*self.freeze_args)
        self.assertEqual(learning.match_sealed_actions(frozen, sealed), 756)
        before = learning.action_cache(frozen)
        first = learning.evaluate(frozen, old, old, source, self.consistency)
        fresh = copy.deepcopy(old)
        for x in fresh.values(): x['values'][:, 1:, 0] += 100.
        with patch.object(learning.previous.previous, 'predict', side_effect=AssertionError('Future-dependent inference')):
            second = learning.evaluate(frozen, old, fresh, source, self.consistency)
        self.assertEqual(before, learning.action_cache(frozen))
        self.assertEqual(first['reproduction'], second['reproduction'])
        self.assertNotEqual(first['blocks'][0]['endpoints']['54_2_cached']['held60']['fresh8'],
            second['blocks'][0]['endpoints']['54_2_cached']['held60']['fresh8'])
        sealed['blocks'][0]['choices']['raw'][0] = 99
        with self.assertRaisesRegex(ValueError, 'sealed OOF action'): learning.match_sealed_actions(frozen, sealed)

    def test_own_common_fleet_nonfleet_and_draw_risk_units(self):
        draws = np.zeros((8, 1, 6, 4))
        draws[:, 0, 1] = [-1., .3, -1.3, 1.]
        draws[:, 0, 2] = [-2., 1., -3., -1.]
        data = {'originals': np.array([0]), 'roots': np.array([397]), 'raw_choices': np.array([0]),
            'raw_logits': np.array([[0., -1., -1., 1., 1., 1.]])}
        predictions = {v: {'chosen': np.array([2]), 'cost': draws.mean(axis=0)[..., 0]} for v in learning.VARIANTS}
        datasets = {e: {'probe6': data} for e in learning.ENDPOINTS}
        datasets[learning.ENDPOINTS[-1]] = {'probe6': {**data, 'raw_choices': np.array([1])}}
        block = {'data': datasets, 'predictions': {e: {'probe6': predictions} for e in learning.ENDPOINTS}}
        stats = learning.metrics(block, learning.ENDPOINTS[0], 'probe6', draws)['all']['variants']['raw']
        self.assertEqual(stats['delta_cost_own'], -2.)
        self.assertEqual(stats['delta_fleet_cost_own'], 1.)
        self.assertEqual(stats['delta_nonfleet_cost_own'], -3.)
        self.assertAlmostEqual(stats['delta_fleet_cost_common'], .7)
        self.assertAlmostEqual(stats['delta_nonfleet_cost_common'], -1.7)
        self.assertEqual(stats['delta_draw_wait_harm_rate_common'], -1.)
        self.assertEqual(stats['delta_draw_wait_harm_rate_own'], 0.)

    def test_compressed_cache_axes_and_state_keys_roundtrip(self):
        old, fresh = self.args[2], copy.deepcopy(self.args[2])
        for x in fresh.values(): x['values'][:, 1:, 3] -= 1.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'panels.npz'
            learning.write_panel_cache(path, old, fresh)
            a, b = learning.read_panel_cache(path)
            self.assertEqual(set(a), set(old))
            for k in old:
                np.testing.assert_array_equal(a[k]['values'], old[k]['values'])
                np.testing.assert_array_equal(b[k]['values'], fresh[k]['values'])
            bad = {k: {'values': x['values'][..., :2]} for k, x in fresh.items()}
            with self.assertRaisesRegex(ValueError, 'axes'): learning.write_panel_cache(path, old, bad)

    def test_averaging_and_no_new_episode_or_training_budget(self):
        frozen = learning.freeze(*self.freeze_args)
        cell = learning.evaluate(frozen, self.args[2], self.args[2], self.args[6], self.consistency)
        report = learning.aggregate([cell, cell])
        self.assertEqual(report['54_2_cached']['probe6']['fresh8']['all']['matched_blocks'], 12)
        self.assertEqual(report, json.loads(json.dumps(report)))
        c = spec.contract()
        self.assertEqual(c, json.loads(json.dumps(c)))
        for f in ('new_raw_reference_fits', 'new_residual_fits', 'new_native_episodes', 'training_updates'):
            self.assertEqual(c['budget'][f], 0)
        for operation, key in (('prepare', None), ('report', None), ('evaluate', spec.KEYS[0])):
            task = task_specification('unit_fresh_transfer', key, operation)
            self.assertEqual((task['cpu'], task['ram_mb']), (1, 3072))
            self.assertIsNone(task['require_node'])
            self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
            self.assertNotIn('_raw', task['result_dir'])
            self.assertFalse(any('/results/' in p for p in task['stage_input_paths']))


class CompactReportTest(unittest.TestCase):
    def test_merge_weights_source_only_blocks_not_equal_cell_counts(self):
        def matrix(blocks, value):
            p = {'matched_blocks': blocks, 'mean_cases': 6., 'variants': {v: {'delta_cost_own': value} for v in learning.VARIANTS}}
            return {e: {s: {label: {'all': p, 'by_root': {'397': p}} for label in learning.LABELS}
                for s in learning.SCOPES} for e in learning.ENDPOINTS}
        result = learning.merge_matrices([matrix(10, 1.), matrix(20, 4.)])
        p = result['54_2_cached']['probe6']['fresh8']['by_root']['397']
        self.assertEqual(p['matched_blocks'], 30)
        self.assertEqual(p['variants']['raw']['delta_cost_own'], 3.)

    def test_focal_matrix_retains_all_registered_fresh_endpoint_candidates(self):
        piece = {'matched_blocks': 20, 'variants': {v: {'delta_cost_own': i} for i, v in enumerate(learning.VARIANTS)}}
        matrix = {e: {s: {label: {'all': piece, 'by_root': {'397': piece}} for label in learning.LABELS}
            for s in learning.SCOPES} for e in learning.ENDPOINTS}
        report = {'views': {'root_397': matrix}, 'mixed_pooled': matrix, 'regression_case': matrix,
            'coverage_qualified': True}
        result = spec.compact_report(report)
        self.assertEqual(result, json.loads(json.dumps(result)))
        focal = result['mixed_pooled']['54_2_cached']['probe6']['fresh8']['all']['variants']
        self.assertEqual(set(focal), set(spec.FOCAL_VARIANTS))
        full = result['all_registered_fresh8']['60_4_native']['held60']['all']['variants']
        self.assertEqual(set(full), set(learning.VARIANTS))
        self.assertEqual(len(report['mixed_pooled']['54_2_cached']['probe6']['fresh8']['all']['variants']), 21)


if __name__ == '__main__': unittest.main()
