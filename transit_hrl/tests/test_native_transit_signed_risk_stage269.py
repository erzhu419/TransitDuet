import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_signed_risk as learning
from scripts import run_native_transit_signed_risk_stage269 as spec
from scripts import submit_native_transit_signed_risk_stage269_scheduleurm as submit


class SignedRiskTest(unittest.TestCase):
    def margins(self):
        y = np.zeros((4, 32))
        y[0, [0, 1, 16, 17, 18]] = [1., 3., 2., 4., 6.]
        y[0, [2, 19]] = [-2., -3.]
        y[1, 0] = -4.
        y[2, 22] = 8.
        y[3, [1, 20]] = [2e-13, -3e-13]
        return y

    def test_signed_products_reconstruct_direct_means_including_near_zero(self):
        y = self.margins();a = learning.audit(y)
        np.testing.assert_allclose(a['direct'], y.reshape(4, 2, 16).mean(axis=2), rtol=0, atol=0)
        self.assertLess(max(a['errors'].values()), 1e-12)
        self.assertEqual(a['counts'][0].tolist(), [[2, 1], [3, 1]])
        self.assertEqual(a['amplitude'][0].tolist(), [[2., 2.], [4., 3.]])
        self.assertNotEqual(float(a['gap_parts'][3, -1]), 0.)

    def test_zero_support_does_not_impute_conditional_amplitude(self):
        a = learning.audit(self.margins())
        self.assertTrue(np.isnan(a['amplitude'][1, 1, 1]))
        self.assertEqual(a['products'][1, 1, 1], 0.)
        self.assertTrue(a['one'][1, 1]);self.assertFalse(a['both'][1, 1])
        self.assertEqual(a['gap_parts'][1, :2].tolist(), [0., 0.])
        self.assertAlmostEqual(a['gap_parts'][1, 2], .25)
        self.assertAlmostEqual(a['gap_parts'][2, 2], .5)

    def test_symmetric_gap_split_is_exact_with_probability_amplitude_changes(self):
        a = learning.audit(self.margins())
        self.assertAlmostEqual(a['gap_parts'][0, 0], 3/16)
        self.assertAlmostEqual(a['gap_parts'][0, 1], 4/16)
        self.assertEqual(a['gap_parts'][0, 2], 0.)
        np.testing.assert_allclose(a['gap_parts'].sum(axis=1), a['gap'], atol=1e-15)
        reverse = learning.audit(self.margins().reshape(4, 2, 16)[:, ::-1].reshape(4, 32))
        np.testing.assert_allclose(reverse['gap_parts'], -a['gap_parts'], atol=1e-15)

    def test_scatter_and_signed_cross_reconstruct_paired_mean_noise(self):
        y = self.margins();a = learning.audit(y)
        expected = y.reshape(4, 2, 16).var(axis=2, ddof=1).sum(axis=1)/16
        np.testing.assert_allclose(a['variance'], expected, atol=1e-15)
        s = learning.summarize(a, [True]*4)
        self.assertAlmostEqual(sum(s['scatter_variance'].values())+s['scatter_signed_cross_sum'], expected.sum())
        self.assertAlmostEqual(sum(s['gap_parts_squared_sum'].values())+s['gap_signed_cross_sum'], s['observed_gap_squared_sum'])
        self.assertGreater(s['saved_added_signed_noise_cross_sum'], 0.)
        self.assertAlmostEqual(np.sum(a['sign_covariance'][0]), expected[0])

    def test_summary_retains_support_units_and_empty_group_nulls(self):
        a = learning.audit(self.margins());s = learning.summarize(a, [True]*4)
        self.assertEqual(s['unique_state_actions'], 4);self.assertEqual(s['action_future_keys'], 128)
        self.assertEqual(s['signs']['added_fleet']['both_banks_supported_actions'], 1)
        self.assertEqual(s['signs']['added_fleet']['one_bank_supported_actions'], 1)
        self.assertEqual(s['signs']['saved_fleet']['one_bank_supported_actions'], 1)
        self.assertEqual(s['actions_with_no_event_in_either_bank'], 1)
        self.assertEqual(s['actions_with_both_signs_observed'], 1)
        empty = learning.summarize(a, [False]*4)
        self.assertIsNone(empty['plugin_noise_RMS'])
        self.assertIsNone(empty['signs']['added_fleet']['both_supported_amplitude_gap_RMS'])
        json.dumps([s, empty], allow_nan=False)

    def test_collect_deduplicates_contexts_and_uses_union_of_eligible_actions(self):
        y = np.zeros((1, 32, 6));y[0, 0, 1] = 2.
        masks = {'all_eligible': np.ones(1, bool), **{v: np.ones(1, bool) for v in spec.PRIMARY}}
        c = {'key': ['mixed_a', 0, 'even_odd', 0], 'states': [[397, 11, 9]], 'originals': np.zeros(1, int),
            'margins': y, 'feasible': np.array([[True, True, False, False, False, False]]), 'masks': masks}
        d = {**c, 'key': ['mixed_b', 0, 'even_odd', 0], 'feasible': np.array([[True, False, True, False, False, False]])}
        keys, a, membership, unions = spec.collect([c, c, d])
        self.assertEqual(len(keys), 5);self.assertEqual(len(unions['all_eligible']), 2)
        self.assertEqual(len(membership['mixed_a', 'all_eligible']), 1)
        self.assertNotIn((397, 11, 9, 0), keys)
        s = learning.summarize(a, [k in unions['all_eligible'] for k in keys])
        self.assertEqual(s['action_future_keys'], 64)
        corrupt = y.copy();corrupt[0, 0, 1] += 1.
        with self.assertRaisesRegex(ValueError, 'target changed'):spec.collect([c, {**c, 'margins': corrupt}])

    def test_cell_reproduces_parent_events_with_correct_physical_head_and_anchor(self):
        rows = [{'root': 397, 'scene_seed': 11, 'probe_index': 9}]
        values = np.zeros((1, 32, 6, 4));values[0, 0, 1, 1] = 3.;values[..., 2] = 17.
        data = {'originals': np.array([0]), 'raw_logits': np.array([[-1., -1., 1., 1., 1., 1.]])}
        events = np.zeros((1, 32), bool);events[0, 0] = True
        p = {'_states': [[397, 11, 9]], '_events': events.tolist(), '_retained': [False],
            'category': 'removed', 'active': True, 'variant': spec.PRIMARY[0]}
        with patch.object(spec.source.source.source.source, 'read_cell', return_value={'reference': {}}), \
                patch.object(spec.cache.cached.training.cohort, 'split_rows', return_value=(rows, [])), \
                patch.object(spec.cache.learning.base, 'causal_data', return_value=data), \
                patch.object(spec.cache.learning.base, 'training_draws', return_value=values):
            c = spec.analyze_cell(['mixed_a', 0, 'even_odd', 0], {'mixed_a': []}, {}, {'pieces': [p]})
            self.assertEqual(c['margins'][0, 0, 1], 3.)
            self.assertEqual(c['masks'][spec.PRIMARY[0]].tolist(), [True])
            p['_events'] = np.zeros_like(events).tolist()
            with self.assertRaisesRegex(ValueError, 'events/states changed'):spec.analyze_cell(['mixed_a', 0, 'even_odd', 0], {'mixed_a': []}, {}, {'pieces': [p]})

    def test_main_reads_training_cache_once_without_fits_or_selection(self):
        ledger = [{'key': list(k), 'pieces': []} for k in spec.KEYS]
        with patch('sys.argv', ['runner', '--run-name', 'unit_signed']), \
                patch.object(spec.cache.cached.training.cohort.source.source, 'load_data', return_value=({'training_data': []}, None, [], None)), \
                patch.object(spec.cache.cached.training.cohort.source, 'matched_views', return_value={}), \
                patch.object(spec.cache.sampling, 'read_panels', return_value={}) as panels, \
                patch.object(spec.Path, 'read_text', return_value=json.dumps(ledger)), \
                patch.object(spec, 'analyze_cell', return_value={}) as analyze, \
                patch.object(spec, 'report', return_value=({}, [], {})), patch.object(spec, 'write_json'), patch.object(spec.np, 'savez_compressed'), \
                patch.object(spec.source.learning.coupled.calibration, 'fit_fixed', side_effect=AssertionError('Refit')), \
                patch.object(spec.cache.learning, 'select', side_effect=AssertionError('Selection')):
            spec.main()
        panels.assert_called_once_with(spec.cache.cached.previous.DATA_RUN);self.assertEqual(analyze.call_count, 32)

    def test_scheduler_unpinned_single_cpu_and_completed_parent_gate(self):
        task = submit.task_specification('unit_signed')
        self.assertEqual(task['cpu'], 1);self.assertIsNone(task['require_node'])
        self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
        parent = {'protocol': spec.source.EXPERIMENT_PROTOCOL, 'contract': spec.source.contract(), 'coverage_qualified': True}
        receipt = {'scheduler': {'status_counts': {'done': 1}, 'exit_code_counts': {'0': 1}}}
        with patch('sys.argv', ['submit', '--run-name', 'unit_signed']), patch.object(submit, 'inventory', return_value=[]), \
                patch.object(submit.Path, 'exists', return_value=False), \
                patch.object(submit.Path, 'read_text', autospec=True, side_effect=lambda p: json.dumps(receipt if p.name == 'completion_receipt.json' else parent)), \
                patch.object(submit.subprocess, 'check_output', return_value='revision'), patch.object(submit, 'write_json'), patch.object(submit, 'execute_bulk') as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]), 1)
            bulk.reset_mock();receipt['scheduler']['exit_code_counts'] = {'1': 1}
            with self.assertRaises(SystemExit):submit.main()
            bulk.assert_not_called()


if __name__ == '__main__': unittest.main()
