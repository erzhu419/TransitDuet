import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_future_forcing as learning
from scripts import run_native_transit_future_forcing_stage263 as spec
from scripts import submit_native_transit_future_forcing_stage263_scheduleurm as submit
from tests import test_native_transit_calibration_ablation_stage260 as fixtures


class FutureForcingTest(unittest.TestCase):
    def fixture(self):
        data, c, prior, cap, geometry = fixtures.CalibrationAblationTest().numeric_fixture()
        c['target'] *= 5
        rng = np.random.default_rng(263)
        c['reliability'] = rng.uniform(.2, 1., (12, 1, 2))
        other = copy.deepcopy(c)
        other['precision'] *= rng.uniform(.5, 2., (12, 15, 2))
        other['reliability'][:] = 0.
        other['target'][:] = 0.
        residual = c['target']/c['reliability']
        residuals = {0: residual, 1: residual+rng.normal(size=residual.shape)}
        full, _ = learning.source.source.source.fit_fixed(data, c, np.arange(12), prior, cap, geometry)
        prefix, _ = learning.source.source.source.fit_fixed(data, c, np.arange(6, 12), prior, cap, geometry)
        x = learning.source.source.source.source.joint.relative(data['x'], data['originals'])
        return data, {0: c, 1: other}, prior, geometry, residuals, full, prefix, x

    def test_raw_target_uses_paired_service_risk_and_original_anchor(self):
        rng = np.random.default_rng(1)
        draws = rng.normal(size=(2, 4, 6, 4))
        data = {'originals': np.array([2, 4]), 'components': rng.normal(size=(2, 6, 2))}
        expected = np.zeros((2, 6, 2))
        for n in range(2):
            for a in range(6):
                for h, index in enumerate((2, 1)):
                    expected[n, a, h] = (np.mean(draws[n, :, a, index]-draws[n, :, data['originals'][n], index])
                        -data['components'][n, a, h]+data['components'][n, data['originals'][n], h])/3.
        np.testing.assert_allclose(learning.residual_target(draws, data, 3.), expected, atol=1e-15)

    def test_active_cap_fixed_reliability_and_factorial_identities_without_fitting(self):
        data, cs, prior, geometry, residuals, full, prefix, x = self.fixture()
        changed = 0
        with patch.object(learning.source.source.source.source.source.optimizer, 'quadratic_fit', side_effect=AssertionError('Refit')):
            for level, (basis, coordinates) in geometry.items():
                for model in ('free_old', 'shrunk'):
                    c = learning.context(x@basis, cs, residuals, data['raw_logits'] <= 0,
                        np.asarray(prior['weights'][level])@coordinates.T,
                        np.asarray(full[model]['weights'][level])@coordinates.T,
                        np.asarray(prefix[model]['weights'][level])@coordinates.T,
                        np.arange(6, 12), model, .8, 0, level)
                    v = c['responses']
                    np.testing.assert_allclose(v['anchor'], sum(v[n] for n in learning.PARTS), atol=1e-12)
                    np.testing.assert_allclose(v['joint_swap'], v['target_swap']+v['precision_swap']+v['interaction'], atol=1e-12)
                    np.testing.assert_allclose(v['target_swap'], -2*v['target_contrast'], atol=1e-12)
                    self.assertLess(c['diagnostics']['response_reproduction_error'], 1e-12)
                    changed += np.linalg.norm(v['target_swap']) > 1e-8
        self.assertGreater(changed, 0)  # Other bank's zero reliability did not erase raw labels.

    def test_identical_future_and_precision_give_zero_swaps(self):
        data, cs, prior, geometry, residuals, full, prefix, x = self.fixture()
        cs[1] = copy.deepcopy(cs[0]); residuals[1] = residuals[0].copy()
        level = next(iter(geometry));basis, coordinates = geometry[level]
        c = learning.context(x@basis, cs, residuals, data['raw_logits'] <= 0,
            np.asarray(prior['weights'][level])@coordinates.T,
            np.asarray(full['shrunk']['weights'][level])@coordinates.T,
            np.asarray(prefix['shrunk']['weights'][level])@coordinates.T,
            np.arange(6, 12), 'shrunk', .8, 0, level)
        for name in ('target_contrast', 'target_swap', 'precision_swap', 'interaction', 'joint_swap'):
            np.testing.assert_allclose(c['responses'][name], 0., atol=1e-15)
        self.assertAlmostEqual(c['diagnostics']['force_alignment']['cosine'], 1.)

    def test_cross_terms_preserve_cancellation_and_undefined_alignment(self):
        v = {n: np.zeros(2) for n in learning.NAMES}
        v['design_prior'][:] = 1.;v['target_mean'][:] = -1.
        s, error = learning.squares(v)
        self.assertEqual(s['anchor'], 0.)
        self.assertEqual(s['cross_design_prior_target_mean'], -4.)
        self.assertEqual(error, 0.)
        self.assertIsNone(learning.alignment(np.zeros(2), np.ones(2))['cosine'])

    def test_region_masks_and_inactive_summary_preserve_denominators(self):
        c = {'responses': {n: np.ones(2) for n in learning.NAMES}, 'low_basis': np.array([[1.], [0.]])}
        c['responses']['design_prior'][:] = 0.;c['responses']['target_contrast'][:] = 0.
        s = learning.region_stats(np.ones((2, 6, 2)), np.array([[True], [False]]), np.ones((2, 6), bool), c, 2.)
        s.update(state_cases=1, disagreements=1)
        a = {'cell': ['mixed_a', 0, 'even_odd', 0], 'bank': 0, 'active': True,
            'forcing': {'force_alignment': learning.alignment(np.ones(2), np.ones(2)),
                'target_force_alignment': learning.alignment(np.zeros(2), np.zeros(2))}, 'regions': {'held': s}}
        zero = {**s, 'bands': {b: dict.fromkeys(learning.FIELDS, 0.) for b in ('full', 'low', 'rest')}, 'disagreements': 0}
        out = spec.summarize([a, {**a, 'active': False, 'regions': {'held': zero}}])
        self.assertEqual((out['prefix_model_exposures'], out['active_prefix_models']), (2, 1))
        self.assertEqual(out['regions']['held']['feasible_action_exposures'], 12)
        self.assertAlmostEqual(out['regions']['held']['bands']['full']['RMS']['anchor'], np.sqrt(8.))
        self.assertEqual(out['target_force_alignment']['defined_prefix_models'], 0)

    def test_main_reuses_one_label_archive_without_fits_or_selection(self):
        with patch('sys.argv', ['runner', '--run-name', 'unit_forcing']), \
                patch.object(spec.cache.cached.training.cohort.source.source, 'load_data', return_value=({'training_data': []}, None, [], None)), \
                patch.object(spec.cache.cached.training.cohort.source, 'matched_views', return_value={}), \
                patch.object(spec.cache.sampling, 'read_panels', return_value={}) as labels, \
                patch.object(spec, 'analyze_cell', return_value={}) as analyze, patch.object(spec, 'report', return_value={}), \
                patch.object(spec.Path, 'read_text', return_value='{}'), patch.object(spec, 'write_json'), \
                patch.object(learning.source.source.source, 'fit_fixed', side_effect=AssertionError('Refit')), \
                patch.object(spec.cache.learning, 'select', side_effect=AssertionError('Selection')):
            spec.main()
        labels.assert_called_once_with(spec.cache.cached.previous.DATA_RUN)
        self.assertEqual(analyze.call_count, 32)

    def test_scheduler_single_unpinned_task_and_completed_parent(self):
        task = submit.task_specification('unit_forcing')
        self.assertEqual(task['cpu'], 1);self.assertIsNone(task['require_node'])
        self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
        parent = {'protocol': spec.source.EXPERIMENT_PROTOCOL, 'contract': spec.source.contract(), 'coverage_qualified': True}
        receipt = {'scheduler': {'status_counts': {'done': 1}, 'exit_code_counts': {'0': 1}}}
        with patch('sys.argv', ['submit', '--run-name', 'unit_forcing']), patch.object(submit, 'inventory', return_value=[]), \
                patch.object(submit.Path, 'exists', return_value=False), \
                patch.object(submit.Path, 'read_text', autospec=True, side_effect=lambda p: json.dumps(receipt if p.name == 'completion_receipt.json' else parent)), \
                patch.object(submit.subprocess, 'check_output', return_value='revision'), patch.object(submit, 'write_json'), patch.object(submit, 'execute_bulk') as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]), 1)
            bulk.reset_mock();receipt['scheduler']['exit_code_counts'] = {'1': 1}
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()


if __name__ == '__main__': unittest.main()
