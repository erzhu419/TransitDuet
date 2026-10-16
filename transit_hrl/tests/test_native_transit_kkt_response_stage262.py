import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_kkt_response as learning
from scripts import run_native_transit_kkt_response_stage262 as spec
from scripts import submit_native_transit_kkt_response_stage262_scheduleurm as submit
from tests import test_native_transit_calibration_ablation_stage260 as fixtures


class KKTResponseTest(unittest.TestCase):
    def test_interior_response_has_only_data_forcing(self):
        h = np.diag([2., 4.])
        full, delta = np.array([1., 2.]), np.array([.2, -.3])
        c = learning.split_response(h, full, full+delta, -h@delta, 0., 0.)
        np.testing.assert_allclose(c['parts']['data'], delta)
        np.testing.assert_allclose(c['parts']['dual'], 0.)
        self.assertLess(c['diagnostics']['influence_equation_residual'], 1e-14)

    def test_multiplier_can_cancel_data_response(self):
        # Both optima lie on the same radius; changed dual is not extra fit data.
        full, prefix = np.array([1., 0.]), np.array([0., 1.])
        h, mf, mp = np.diag([2., 3.]), 1., 4.
        force = -(h+mp*np.eye(2))@(prefix-full)-(mp-mf)*full
        c = learning.split_response(h, full, prefix, force, mf, mp)
        np.testing.assert_allclose(c['parts']['data'], [-.5, 1.])
        np.testing.assert_allclose(c['parts']['dual'], [-.5, 0.])
        # At this query the two nonzero component predictions cancel exactly.
        s = learning.region_stats(np.array([[[2., 2.]]]), np.array([[True]]), np.array([[True]]), c, 1.)
        self.assertEqual(s['squared_response']['total'], 0.)
        self.assertAlmostEqual(s['squared_response']['cross_data_dual'], -2.)
        self.assertEqual(s['squared_response']['data'], 1.)
        self.assertEqual(s['squared_response']['dual'], 1.)

    def test_singular_response_keeps_null_displacement(self):
        c = learning.split_response(np.diag([2., 0.]), np.zeros(2), np.array([1., 3.]), np.array([-2., 0.]), 0., 0.)
        np.testing.assert_allclose(c['parts']['data'], [1., 0.])
        np.testing.assert_allclose(c['parts']['null'], [0., 3.])
        self.assertEqual(c['diagnostics']['supported_rank'], 1)
        np.testing.assert_allclose(sum(c['parts'].values()), c['delta'])

    def test_stationarity_error_is_retained_and_large_error_rejected(self):
        c = learning.split_response(np.eye(2), np.zeros(2), np.ones(2), np.array([-1.+1e-10, -1.]), 0., 0.)
        self.assertGreater(np.linalg.norm(c['parts']['residual']), 0.)
        np.testing.assert_allclose(c['parts']['residual'], [1e-10, 0.], atol=1e-15)
        with self.assertRaises(ValueError):
            learning.split_response(np.eye(2), np.zeros(2), np.ones(2), np.zeros(2), 0., 0.)

    def test_cached_active_cap_decomposition_does_not_fit(self):
        data, c, prior, cap, geometry = fixtures.CalibrationAblationTest().numeric_fixture()
        c['target'] *= 5
        indices = np.arange(6, 12)
        full, _ = learning.source.source.fit_fixed(data, c, np.arange(12), prior, cap, geometry)
        fixed, _ = learning.source.source.fit_fixed(data, c, indices, prior, cap, geometry)
        x = learning.source.source.source.joint.relative(data['x'], data['originals'])
        changed = 0
        with patch.object(learning.source.source.source.source.optimizer, 'quadratic_fit', side_effect=AssertionError('Refit')):
            for level, (basis, coordinates) in geometry.items():
                calibration = {'target': c['target'], 'precision': c['precision'], 'penalty': c['penalties'][level]}
                initial = np.asarray(prior['weights'][level])@coordinates.T
                for model in ('free_old', 'shrunk'):
                    f = learning.full_context(x@basis, calibration, data['raw_logits'] <= 0, initial,
                        np.asarray(full[model]['weights'][level])@coordinates.T, model, .8)
                    p = learning.prefix_context(x@basis, calibration, data['raw_logits'] <= 0, initial,
                        np.asarray(fixed[model]['weights'][level])@coordinates.T, indices, model, .8, f)
                    np.testing.assert_allclose(sum(p['parts'].values()), p['delta'], atol=1e-14)
                    self.assertLess(p['diagnostics']['influence_equation_residual'], 1e-8)
                    changed += abs(p['diagnostics']['dual_change']) > 1e-10
        self.assertGreater(changed, 0)

    def test_summary_preserves_inactive_denominators_and_signed_terms(self):
        c = learning.split_response(np.eye(2), np.ones(2), np.array([2., 0.]), np.array([-1., 1.]), 0., 0.)
        s = learning.region_stats(np.ones((1, 6, 2)), np.array([[True]]), np.ones((1, 6), bool), c, 2.)
        s.update(state_cases=1, disagreements=1)
        a = {'cell': ['mixed_a', 0, 'even_odd', 0], 'bank': 0, 'active': True, 'kkt': c['diagnostics'], 'regions': {'held': s}}
        zero = {**s, 'squared_response': dict.fromkeys(learning.FIELDS, 0.),
            'bands': {b: dict.fromkeys(learning.FIELDS, 0.) for b in ('low', 'rest')}, 'disagreements': 0}
        out = spec.summarize([a, {**a, 'active': False, 'regions': {'held': zero}}])
        self.assertEqual((out['prefix_model_exposures'], out['active_prefix_models']), (2, 1))
        self.assertEqual(out['regions']['held']['feasible_action_exposures'], 12)
        self.assertEqual(out['regions']['held']['action_disagreements'], 1)
        self.assertEqual(set(out['regions']['held']['squared_response']), set(learning.FIELDS))

    def test_main_reads_no_evaluation_panels_or_fits(self):
        with patch('sys.argv', ['runner', '--run-name', 'unit_kkt']), \
                patch.object(spec.cache.cached.training.cohort.source.source, 'load_data', return_value=({'training_data': []}, None, [], None)), \
                patch.object(spec.cache.cached.training.cohort.source, 'matched_views', return_value={}), \
                patch.object(spec, 'analyze_cell', return_value={}) as analyze, patch.object(spec, 'report', return_value={}), \
                patch.object(spec.Path, 'read_text', return_value='{}'), patch.object(spec, 'write_json'), \
                patch.object(spec.cache.sampling, 'read_panels', side_effect=AssertionError('Evaluation panels')), \
                patch.object(learning.source.source, 'fit_fixed', side_effect=AssertionError('Refit')), \
                patch.object(spec.cache.learning, 'select', side_effect=AssertionError('Reselection')):
            spec.main()
        self.assertEqual(analyze.call_count, 32)
        self.assertTrue(all(spec.contract()['budget'][f] == 0 for f in ('new_critic_fits', 'new_raw_reference_fits', 'new_native_episodes', 'actor_updates')))

    def test_single_unpinned_scheduler_task_and_parent_gate(self):
        task = submit.task_specification('unit_kkt')
        self.assertEqual(task['cpu'], 1)
        self.assertIsNone(task['require_node'])
        self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
        parent = {'protocol': spec.source.EXPERIMENT_PROTOCOL, 'contract': spec.source.contract(), 'coverage_qualified': True}
        receipt = {'scheduler': {'status_counts': {'done': 1}, 'exit_code_counts': {'0': 1}}}
        with patch('sys.argv', ['submit', '--run-name', 'unit_kkt']), patch.object(submit, 'inventory', return_value=[]), \
                patch.object(submit.Path, 'exists', return_value=False), \
                patch.object(submit.Path, 'read_text', autospec=True, side_effect=lambda p: json.dumps(receipt if p.name == 'completion_receipt.json' else parent)), \
                patch.object(submit.subprocess, 'check_output', return_value='revision'), patch.object(submit, 'write_json'), patch.object(submit, 'execute_bulk') as bulk:
            submit.main()
            self.assertEqual(len(bulk.call_args.args[0]), 1)
            bulk.reset_mock()
            receipt['scheduler']['exit_code_counts'] = {'1': 1}
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()


if __name__ == '__main__': unittest.main()
