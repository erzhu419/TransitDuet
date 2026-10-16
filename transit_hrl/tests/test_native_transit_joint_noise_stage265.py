import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_joint_noise as learning
from scripts import run_native_transit_joint_noise_stage265 as spec
from scripts import submit_native_transit_joint_noise_stage265_scheduleurm as submit
from tests import test_native_transit_future_forcing_stage263 as fixtures


class JointNoiseTest(unittest.TestCase):
    def test_per_state_scores_reproduce_head_gradient_for_each_draw(self):
        rng = np.random.default_rng(265)
        z = rng.normal(size=(12, 6, 3));targets = rng.normal(size=(12, 4, 6, 2))
        precision = rng.uniform(.2, 2., (12, 15, 2));eligible = np.ones((12, 6), bool)
        eligible[:3, 5] = False
        for model, penalty in (('free_old', 0.), ('shrunk', None), ('shrunk', 0.), ('shrunk', .7)):
            scores = learning.state_scores(z, targets, precision, eligible, model, penalty)
            gradient = learning.previous.label_gradient(z, precision, eligible, model, penalty)
            for draw in range(4):
                for head in (0, 1):
                    selected = np.zeros_like(targets[:, draw]);selected[..., head] = targets[:, draw, :, head]
                    np.testing.assert_allclose(scores[:, draw, head].sum(axis=0), gradient(selected), atol=1e-13)

    def test_common_action_noise_cancels_before_covariance(self):
        rng = np.random.default_rng(2)
        z = rng.normal(size=(3, 6, 2))
        targets = np.broadcast_to(rng.normal(size=(3, 4, 1, 2)), (3, 4, 6, 2))
        scores = learning.state_scores(z, targets, np.ones((3, 15, 2)), np.ones((3, 6), bool), 'free_old', 0.)
        np.testing.assert_allclose(scores, 0.)
        for c in learning.noise_covariances(scores, np.array([True, False, True])).values():
            np.testing.assert_allclose(c, 0.)

    def test_statewise_two_mean_variance_retains_head_anticorrelation(self):
        scores = np.zeros((2, 4, 2, 1))
        scores[:, :, 0, 0] = [1., -1., 11., 9.]
        scores[:, :, 1, 0] = -scores[:, :, 0, 0]
        c = learning.noise_covariances(scores, np.array([False, True]))
        self.assertAlmostEqual(c['service'][0, 0], 4.)
        self.assertAlmostEqual(c['risk'][0, 0], 4.)
        self.assertAlmostEqual(c['cross_sr'][0, 0], -8.)
        self.assertAlmostEqual(c['joint'][0, 0], 0.)
        self.assertAlmostEqual(c['excluded_service'][0, 0], 2.)
        self.assertLess(learning.covariance_error(c), 1e-14)
        # Summing state scores by aligned draw before covariance would give 8, not 4.
        scores[:, :, 1] = 0.
        self.assertAlmostEqual(learning.noise_covariances(scores, np.array([False, True]))['joint'][0, 0], 4.)

    def test_cached_context_matches_Stage264_without_new_parameter_fit(self):
        data, cs, prior, geometry, residuals, full, prefix, x = fixtures.FutureForcingTest().fixture()
        rng = np.random.default_rng(265);paired, halves = {}, {}
        for bank in (0, 1):
            noise = rng.normal(size=(12, 16, 6, 2));noise -= noise.mean(axis=1, keepdims=True)
            paired[bank] = (residuals[bank][:, None]+noise)*prior['scale']
            for half in (0, 1): halves[bank, half] = paired[bank][:, half*8:(half+1)*8].mean(axis=1)/prior['scale']
        with patch.object(learning.previous.calibration.source.source.optimizer, 'quadratic_fit', side_effect=AssertionError('Refit')):
            for level, (basis, coordinates) in geometry.items():
                for model in ('free_old', 'shrunk'):
                    args = (data['raw_logits']<=0, np.asarray(prior['weights'][level])@coordinates.T,
                        np.asarray(full[model]['weights'][level])@coordinates.T,
                        np.asarray(prefix[model]['weights'][level])@coordinates.T,
                        np.arange(6, 12), model, .8, 0, level)
                    c = learning.context(x@basis, cs[0], paired, *args, prior['scale'])
                    old = learning.previous.context(x@basis, cs[0], residuals, halves, *args)
                    for name, before in learning.OLD_NAMES.items():
                        np.testing.assert_allclose(c['responses'][name], old['responses'][before], atol=1e-11)
                    self.assertLess(c['diagnostics']['covariance_identity_residual'], 1e-12)
                    self.assertLess(c['diagnostics']['spectral_propagation_residual'], 1e-12)
                    self.assertGreaterEqual(np.linalg.eigvalsh(c['parameter_covariance']['joint']).min(), -1e-10)

    def query_fixture(self):
        c = np.array([[1., .5], [.5, 1.]])
        cov = {n: np.zeros((2, 2)) for n in learning.COVARIANCES}
        responses = {n: np.zeros(2) for n in learning.POSITIVE}
        for n in ('joint', 'service', 'excluded_joint', 'excluded_service'):
            cov[n] = c.copy();responses[n] = np.ones(2)
        context = {'parameter_covariance': cov, 'responses': responses, 'low_basis': np.array([[1.], [0.]])}
        return learning.region_stats(np.ones((1, 1, 2)), np.array([[True]]), np.array([[True]]), context, 1.)

    def test_query_noise_preserves_offdiagonal_mode_covariance(self):
        s = self.query_fixture()
        self.assertEqual(s['bands']['full']['noise_squared_sum']['joint'], 3.)
        self.assertEqual(s['bands']['low']['noise_squared_sum']['joint'], 1.)
        self.assertEqual(s['bands']['rest']['noise_squared_sum']['joint'], 1.)
        self.assertEqual(s['low_rest_noise_cross']['joint'], 1.)
        self.assertEqual(s['bands']['full']['observed_squared_sum']['joint'], 4.)

    def test_summary_keeps_inactive_denominator_and_zero_variance_gap(self):
        s = self.query_fixture();s.update(state_cases=1, disagreements=0)
        noise = {'force_noise_energy': 2., 'parameter_noise_energy': 2., 'low_parameter_noise_energy': 1.,
            'unsupported_force_noise_energy': 0., 'low_dimension_fraction': .5,
            'dominant_covariance_fraction': .75, 'dominant_KKT_mode_fraction': .5}
        a = {'cell': ['mixed_a', 0, 'even_odd', 0], 'bank': 0, 'active': True, 'noise': noise, 'regions': {'held': s}}
        zero = {**s, 'bands': {b: {'noise_squared_sum': dict.fromkeys(learning.COVARIANCES, 0.),
            'observed_squared_sum': dict.fromkeys((*learning.POSITIVE, *learning.CROSSES, 'cross_population_joint'), 0.)} for b in ('full', 'low', 'rest')},
            'low_rest_noise_cross': dict.fromkeys(learning.POSITIVE, 0.)}
        out = spec.summarize([a, {**a, 'active': False, 'regions': {'held': zero}}])
        band = out['regions']['held']['bands']['full']
        self.assertEqual(out['regions']['held']['feasible_action_exposures'], 2)
        self.assertAlmostEqual(band['noise_RMS']['joint'], np.sqrt(1.5))
        self.assertAlmostEqual(band['observed_over_noise_RMS']['joint'], np.sqrt(4/3))
        gap = {**zero, 'bands': {b: {**zero['bands'][b], 'observed_squared_sum': s['bands'][b]['observed_squared_sum']} for b in ('full', 'low', 'rest')}}
        b = spec.summarize([{**a, 'regions': {'held': gap}}])['regions']['held']['bands']['full']
        self.assertIsNone(b['observed_over_noise_RMS']['joint'])
        self.assertEqual(b['zero_noise_nonzero_gap_prefix_models']['joint'], 1)

    def test_main_reads_only_cached_archive_once_no_fits_selection(self):
        with patch('sys.argv', ['runner', '--run-name', 'unit_joint_noise']), \
                patch.object(spec.cache.cached.training.cohort.source.source, 'load_data', return_value=({'training_data': []}, None, [], None)), \
                patch.object(spec.cache.cached.training.cohort.source, 'matched_views', return_value={}), \
                patch.object(spec.cache.sampling, 'read_panels', return_value={}) as labels, \
                patch.object(spec, 'analyze_cell', return_value={}) as analyze, patch.object(spec, 'report', return_value={}), \
                patch.object(spec.Path, 'read_text', return_value='{}'), patch.object(spec, 'write_json'), \
                patch.object(learning.previous.calibration, 'fit_fixed', side_effect=AssertionError('Refit')), \
                patch.object(spec.cache.learning, 'select', side_effect=AssertionError('Selection')):
            spec.main()
        labels.assert_called_once_with(spec.cache.cached.previous.DATA_RUN)
        self.assertEqual(analyze.call_count, 32)

    def test_scheduler_single_unpinned_task_requires_completed_parent(self):
        task = submit.task_specification('unit_joint_noise')
        self.assertEqual(task['cpu'], 1);self.assertIsNone(task['require_node'])
        self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
        self.assertFalse(any('/results/' in p for p in task['stage_input_paths']))
        parent = {'protocol': spec.source.EXPERIMENT_PROTOCOL, 'contract': spec.source.contract(), 'coverage_qualified': True}
        receipt = {'scheduler': {'status_counts': {'done': 1}, 'exit_code_counts': {'0': 1}}}
        with patch('sys.argv', ['submit', '--run-name', 'unit_joint_noise']), patch.object(submit, 'inventory', return_value=[]), \
                patch.object(submit.Path, 'exists', return_value=False), \
                patch.object(submit.Path, 'read_text', autospec=True, side_effect=lambda p: json.dumps(receipt if p.name=='completion_receipt.json' else parent)), \
                patch.object(submit.subprocess, 'check_output', return_value='revision'), patch.object(submit, 'write_json'), patch.object(submit, 'execute_bulk') as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]), 1)
            bulk.reset_mock();receipt['scheduler']['exit_code_counts'] = {'1': 1}
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()


if __name__ == '__main__': unittest.main()
