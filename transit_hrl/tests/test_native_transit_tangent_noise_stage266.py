import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_tangent_noise as learning
from scripts import run_native_transit_tangent_noise_stage266 as spec
from scripts import submit_native_transit_tangent_noise_stage266_scheduleurm as submit
from tests import test_native_transit_future_forcing_stage263 as fixtures


class TangentNoiseTest(unittest.TestCase):
    def test_active_response_matches_saddle_solve_and_multiplier_sign(self):
        h = np.array([[2., .4], [.4, .7]]);u = np.array([.6, .8]);mu = .2
        op = learning.operators(h, u, mu, 1.)
        saddle = np.block([[h+mu*np.eye(2), u[:, None]], [u[None, :], np.zeros((1, 1))]])
        solution = np.linalg.solve(saddle, np.r_[-np.eye(2), np.zeros((1, 2))])
        np.testing.assert_allclose(-op['tangent'], solution[:2], atol=1e-14)
        np.testing.assert_allclose(op['dual_response'], solution[2], atol=1e-14)
        np.testing.assert_allclose(u@op['tangent'], 0., atol=1e-14)
        np.testing.assert_allclose(op['projection']@op['inverse'], op['tangent'], atol=1e-14)

    def test_interior_response_is_unchanged_with_numerical_support(self):
        h = np.diag([2., 0.]);u = np.array([.1, 0.])
        op = learning.operators(h, u, 0., 1.)
        self.assertEqual(op['diagnostics']['constraint_state'], 'interior')
        np.testing.assert_array_equal(op['tangent'], op['inverse'])
        np.testing.assert_allclose(op['inverse'], np.diag([.5, 0.]))
        np.testing.assert_array_equal(op['dual_response'], np.zeros(2))

    def test_boundary_without_strict_complementarity_is_not_smooth_interior(self):
        with self.assertRaisesRegex(ValueError, 'smooth'):
            learning.operators(np.eye(2), np.array([1., 0.]), 0., 1.)

    def test_projection_can_increase_query_variance_and_cross_is_signed(self):
        op = learning.operators(.8*np.eye(2), np.array([1., 0.]), .2, 1.)
        c = np.array([[1., -.99], [-.99, 1.]]);x = np.ones(2)
        b, t = op['inverse'], op['tangent'];r = b-t
        fixed, tangent, radial = (float(x@m@c@m.T@x) for m in (b, t, r))
        cross = float(x@(t@c@r.T+r@c@t.T)@x)
        self.assertAlmostEqual(fixed, .02);self.assertAlmostEqual(tangent, 1.)
        self.assertGreater(tangent, fixed);self.assertLess(cross, 0.)
        self.assertAlmostEqual(fixed, tangent+radial+cross)

    def cached_contexts(self):
        data, cs, prior, geometry, residuals, full, prefix, x = fixtures.FutureForcingTest().fixture()
        rng = np.random.default_rng(266);paired = {}
        for bank in (0, 1):
            noise = rng.normal(size=(12, 16, 6, 2));noise -= noise.mean(axis=1, keepdims=True)
            paired[bank] = (residuals[bank][:, None]+noise)*prior['scale']
        with patch.object(learning.previous.previous.calibration.source.source.optimizer, 'quadratic_fit', side_effect=AssertionError('Refit')):
            for level, (basis, coordinates) in geometry.items():
                for model in ('free_old', 'shrunk'):
                    c = learning.context(x@basis, cs[0], paired, data['raw_logits']<=0,
                        np.asarray(prior['weights'][level])@coordinates.T,
                        np.asarray(full[model]['weights'][level])@coordinates.T,
                        np.asarray(prefix[model]['weights'][level])@coordinates.T,
                        np.arange(6, 12), model, .8, 0, level, prior['scale'])
                    yield c, x@basis, data

    def test_cached_context_covariances_and_responses_recombine_without_fitting(self):
        active = 0
        for c, _, _ in self.cached_contexts():
            active += c['diagnostics']['constraint_state']=='strictly_active'
            for n in learning.previous.COVARIANCES:
                np.testing.assert_allclose(c['fixed']['parameter_covariance'][n],
                    c['tangent']['parameter_covariance'][n]+c['radial']['parameter_covariance'][n]+c['cross_covariance'][n], atol=1e-11)
            for n in learning.previous.POSITIVE:
                np.testing.assert_allclose(c['fixed']['responses'][n], c['tangent']['responses'][n]+c['radial']['responses'][n], atol=1e-12)
            self.assertLess(c['diagnostics']['covariance_residual'], 1e-10)
        self.assertGreater(active, 0)

    def test_query_summary_keeps_population_head_cross_terms_and_inactive_denominator(self):
        c, z, data = next(self.cached_contexts())
        s = learning.region_stats(z, np.ones((12, 1), bool), data['raw_logits']<=0, c, 2.)
        for op in spec.OPS:
            s['operators'][op].update(state_cases=12, disagreements=0)
            for band in ('full', 'low', 'rest'):
                v = s['operators'][op]['bands'][band]['noise_squared_sum']
                self.assertAlmostEqual(v['joint'], v['service']+v['risk']+v['cross_sr'])
                self.assertAlmostEqual(v['joint'], v['excluded_joint']+v['retained_joint'])
        p = {'active': True, 'constraint': c['diagnostics'], 'regions': {'held': s}}
        zero = {**s, 'operators': {op: {**s['operators'][op], 'bands': {b: {
            'noise_squared_sum': dict.fromkeys(s['operators'][op]['bands'][b]['noise_squared_sum'], 0.),
            'observed_squared_sum': dict.fromkeys(s['operators'][op]['bands'][b]['observed_squared_sum'], 0.)} for b in ('full', 'low', 'rest')},
            'low_rest_noise_cross': dict.fromkeys(spec.HEADS, 0.)} for op in spec.OPS},
            'cross': {b: dict.fromkeys(learning.previous.COVARIANCES, 0.) for b in ('full', 'low', 'rest')}}
        out = spec.summarize([p, {**p, 'active': False, 'regions': {'held': zero}}])
        self.assertEqual(out['inactive_keys'], 1)
        self.assertEqual(out['regions']['held']['feasible_action_exposures'], 2*s['operators']['fixed']['feasible_action_cases'])
        for h in spec.HEADS:
            a = out['regions']['held']
            fixed, tangent, radial = (a['operators'][op]['bands']['full']['noise_squared_sum'][h] for op in spec.OPS)
            self.assertAlmostEqual(fixed, tangent+radial+a['cross']['full'][h])
            self.assertAlmostEqual(a['operators']['fixed']['bands']['full']['noise_RMS'][h],
                np.sqrt(s['operators']['fixed']['bands']['full']['noise_squared_sum'][h]/a['feasible_action_exposures']))

    def test_main_reads_cached_labels_once_without_fits_or_selection(self):
        with patch('sys.argv', ['runner', '--run-name', 'unit_tangent']), \
                patch.object(spec.cache.cached.training.cohort.source.source, 'load_data', return_value=({'training_data': []}, None, [], None)), \
                patch.object(spec.cache.cached.training.cohort.source, 'matched_views', return_value={}), \
                patch.object(spec.cache.sampling, 'read_panels', return_value={}) as labels, \
                patch.object(spec, 'analyze_cell', return_value={}) as analyze, patch.object(spec, 'report', return_value={}), \
                patch.object(spec.Path, 'read_text', return_value='{}'), patch.object(spec, 'write_json'), \
                patch.object(learning.previous.previous.calibration, 'fit_fixed', side_effect=AssertionError('Refit')), \
                patch.object(spec.cache.learning, 'select', side_effect=AssertionError('Selection')):
            spec.main()
        labels.assert_called_once_with(spec.cache.cached.previous.DATA_RUN)
        self.assertEqual(analyze.call_count, 32)

    def test_submission_is_unpinned_one_cpu_and_requires_done_parent(self):
        task = submit.task_specification('unit_tangent')
        self.assertEqual(task['cpu'], 1);self.assertIsNone(task['require_node'])
        self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
        self.assertFalse(any('/results/' in p for p in task['stage_input_paths']))
        parent = {'protocol': spec.source.EXPERIMENT_PROTOCOL, 'contract': spec.source.contract(), 'coverage_qualified': True}
        receipt = {'scheduler': {'status_counts': {'done': 1}, 'exit_code_counts': {'0': 1}}}
        with patch('sys.argv', ['submit', '--run-name', 'unit_tangent']), patch.object(submit, 'inventory', return_value=[]), \
                patch.object(submit.Path, 'exists', return_value=False), \
                patch.object(submit.Path, 'read_text', autospec=True, side_effect=lambda p: json.dumps(receipt if p.name=='completion_receipt.json' else parent)), \
                patch.object(submit.subprocess, 'check_output', return_value='revision'), patch.object(submit, 'write_json'), patch.object(submit, 'execute_bulk') as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]), 1)
            bulk.reset_mock();receipt['scheduler']['exit_code_counts'] = {'1': 1}
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()


if __name__ == '__main__': unittest.main()
