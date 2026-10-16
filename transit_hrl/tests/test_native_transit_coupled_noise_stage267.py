import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_coupled_noise as learning
from scripts import run_native_transit_coupled_noise_stage267 as spec
from scripts import submit_native_transit_coupled_noise_stage267_scheduleurm as submit
from tests import test_native_transit_future_forcing_stage263 as fixtures


class CoupledNoiseTest(unittest.TestCase):
    def test_coupled_scores_equal_two_independent_kkt_saddle_responses(self):
        hf, hp = np.array([[2., .4], [.4, .7]]), np.array([[.9, -.2], [-.2, 1.3]])
        uf, up = np.array([.6, .8]), np.array([.8, .6]);mf, mp = .2, .4
        tf = learning.previous.operators(hf, uf, mf, 1.)['tangent']
        tp = learning.previous.operators(hp, up, mp, 1.)['tangent']
        rng = np.random.default_rng(267);sf = rng.normal(size=(3, 4, 2, 2));sp = rng.normal(size=sf.shape)
        terms = learning.response_scores(sf, sp, tf, tp)
        solve = lambda h, u, mu, g: np.linalg.solve(np.block([[h+mu*np.eye(2), u[:, None]], [u[None, :], np.zeros((1, 1))]]), np.r_[-g, 0.])[:2]
        for n in range(3):
            for d in range(4):
                for head in (0, 1):
                    actual = solve(hp, up, mp, sp[n, d, head])-solve(hf, uf, mf, sf[n, d, head])
                    np.testing.assert_allclose(terms['coupled'][n, d, head], actual, atol=1e-14)
        np.testing.assert_allclose(terms['coupled'], terms['one_ended']+terms['operator_change'], atol=1e-14)

    def test_identical_endpoints_cancel_shared_noise_not_marginal_variances(self):
        scores = np.random.default_rng(1).normal(size=(4, 8, 2, 3));retained = np.ones(4, bool)
        terms = learning.response_scores(scores, scores, np.eye(3), np.eye(3))
        cov = {n: learning.noise.noise_covariances(v, retained) for n, v in terms.items()}
        cross = learning.cross_covariances(terms['prefix'], terms['minus_full'], retained)
        for head in learning.noise.COVARIANCES:
            np.testing.assert_allclose(cov['coupled'][head], 0.)
            np.testing.assert_allclose(cross[head], -cov['prefix'][head]-cov['minus_full'][head], atol=1e-14)
        self.assertGreater(np.trace(cov['prefix']['joint']+cov['minus_full']['joint']), 0.)

    def test_same_operator_reduces_to_one_ended_with_correct_state_normalization(self):
        rng = np.random.default_rng(2);sf = rng.normal(size=(6, 8, 2, 2));sp = sf.copy()
        sp[:2] = 0.;sp[2:] *= 6/4
        terms = learning.response_scores(sf, sp, np.diag([2., .5]), np.diag([2., .5]))
        np.testing.assert_array_equal(terms['operator_change'], np.zeros_like(sf))
        np.testing.assert_allclose(terms['coupled'], terms['one_ended'], atol=1e-14)
        self.assertGreater(np.linalg.norm(terms['coupled'][:2]), 0.)

    def test_excluded_endpoint_cross_zero_and_all_head_population_identities_hold(self):
        rng = np.random.default_rng(3);sf = rng.normal(size=(6, 8, 2, 3));sp = rng.normal(size=sf.shape)
        retained = np.arange(6)>=2;sp[~retained] = 0.
        terms = learning.response_scores(sf, sp, np.diag([1., 2., 3.]), np.diag([2., 1., .5]))
        cross = learning.cross_covariances(terms['prefix'], terms['minus_full'], retained)
        for h in ('joint', 'service', 'risk', 'cross_sr'):
            np.testing.assert_array_equal(cross['excluded_'+h], np.zeros((3, 3)))
        cov = {t: learning.noise.noise_covariances(s, retained) for t, s in terms.items()}
        for t in terms:self.assertLess(learning.noise.covariance_error(cov[t]), 1e-12)
        for h in learning.noise.COVARIANCES:
            np.testing.assert_allclose(cov['coupled'][h], cov['prefix'][h]+cov['minus_full'][h]+cross[h], atol=1e-12)
        shifted = sf.copy();shifted[:, :4] += 20.;shifted[:, 4:] -= 10.
        shifted_terms = learning.response_scores(shifted, sp, np.eye(3), np.eye(3))
        plain = learning.response_scores(sf, sp, np.eye(3), np.eye(3))
        np.testing.assert_allclose(learning.cross_covariances(shifted_terms['prefix'], shifted_terms['minus_full'], retained)['joint'],
            learning.cross_covariances(plain['prefix'], plain['minus_full'], retained)['joint'], atol=1e-12)

    def cached_contexts(self):
        data, cs, prior, geometry, residuals, full, prefix, x = fixtures.FutureForcingTest().fixture()
        rng = np.random.default_rng(267);paired = {}
        for bank in (0, 1):
            v = rng.normal(size=(12, 16, 6, 2));v -= v.mean(axis=1, keepdims=True)
            paired[bank] = (residuals[bank][:, None]+v)*prior['scale']
        with patch.object(learning.calibration.source.source.optimizer, 'quadratic_fit', side_effect=AssertionError('Refit')):
            for level, (basis, coordinates) in geometry.items():
                for model in ('free_old', 'shrunk'):
                    args = (x@basis, cs[0], paired, data['raw_logits']<=0,
                        np.asarray(prior['weights'][level])@coordinates.T,
                        np.asarray(full[model]['weights'][level])@coordinates.T,
                        np.asarray(prefix[model]['weights'][level])@coordinates.T,
                        np.arange(6, 12), model, .8, 0, level, prior['scale'])
                    yield learning.context(*args), learning.previous.context(*args), x@basis, data, args

    def test_cached_one_ended_matches_Stage266_direct_factory_and_query_decompositions(self):
        for c, before, z, data, _ in self.cached_contexts():
            for h in learning.noise.COVARIANCES:
                np.testing.assert_allclose(c['terms']['one_ended']['parameter_covariance'][h], before['tangent']['parameter_covariance'][h], atol=1e-10)
            for h in learning.noise.POSITIVE:
                np.testing.assert_allclose(c['terms']['one_ended']['responses'][h], before['tangent']['responses'][h], atol=1e-11)
            self.assertLess(c['diagnostics']['direct_gradient_residual'], 1e-10)
            s = learning.region_stats(z, np.ones((12, 1), bool), data['raw_logits']<=0, c, 2.)
            self.assertLess(s['decomposition_residual'], 1e-10)
            for t in learning.TERMS:s['terms'][t].update(state_cases=12, disagreements=0)
            p = {'active': True, 'coupled': c['diagnostics'], 'regions': {'held': s}}
            zero = copy.deepcopy(s)
            for t in learning.TERMS:
                for b in zero['terms'][t]['bands'].values():
                    b['noise_squared_sum'] = dict.fromkeys(b['noise_squared_sum'], 0.)
                    b['observed_squared_sum'] = dict.fromkeys(b['observed_squared_sum'], 0.)
                zero['terms'][t]['low_rest_noise_cross'] = dict.fromkeys(learning.noise.POSITIVE, 0.)
            for b in zero['cross'].values():
                for v in b.values():
                    for f in v:v[f] = dict.fromkeys(v[f], 0.)
            out = spec.summarize([p, {**p, 'active': False, 'regions': {'held': zero}}])
            self.assertEqual(out['inactive_keys'], 1)
            n = out['regions']['held']['feasible_action_exposures']
            self.assertEqual(n, 2*s['terms']['coupled']['feasible_action_cases'])
            expected = s['terms']['coupled']['bands']['full']['noise_squared_sum']['joint']
            self.assertAlmostEqual(out['regions']['held']['terms']['coupled']['bands']['full']['noise_RMS']['joint'], np.sqrt(expected/n))

    def test_bank_orientation_changes_response_sign_not_covariance(self):
        c, _, _, _, args = next(self.cached_contexts())
        other = list(args);other[10] = 1
        with patch.object(learning.calibration.source.source.optimizer, 'quadratic_fit', side_effect=AssertionError('Refit')):
            flipped = learning.context(*other)
        for t in learning.TERMS:
            for h in learning.noise.POSITIVE:
                np.testing.assert_allclose(flipped['terms'][t]['responses'][h], -c['terms'][t]['responses'][h], atol=1e-12)
            for h in learning.noise.COVARIANCES:
                np.testing.assert_array_equal(flipped['terms'][t]['parameter_covariance'][h], c['terms'][t]['parameter_covariance'][h])

    def test_main_reads_label_archive_and_parent_ledger_once_without_fits_or_selection(self):
        ledger = [{'key': list(k), 'pieces': []} for k in spec.KEYS]
        def read(p):return json.dumps(ledger) if p.name=='cells.json' else '{}'
        with patch('sys.argv', ['runner', '--run-name', 'unit_coupled']), \
                patch.object(spec.cache.cached.training.cohort.source.source, 'load_data', return_value=({'training_data': []}, None, [], None)), \
                patch.object(spec.cache.cached.training.cohort.source, 'matched_views', return_value={}), \
                patch.object(spec.cache.sampling, 'read_panels', return_value={}) as labels, \
                patch.object(spec, 'analyze_cell', return_value={}) as analyze, patch.object(spec, 'report', return_value={}), patch.object(spec, 'compact_report', return_value={}), \
                patch.object(spec.Path, 'read_text', autospec=True, side_effect=read) as reads, patch.object(spec, 'write_json'), \
                patch.object(learning.calibration, 'fit_fixed', side_effect=AssertionError('Refit')), \
                patch.object(spec.cache.learning, 'select', side_effect=AssertionError('Selection')):
            spec.main()
        labels.assert_called_once_with(spec.cache.cached.previous.DATA_RUN)
        self.assertEqual(analyze.call_count, 32)
        self.assertEqual(sum(c.args[0].name=='cells.json' for c in reads.call_args_list), 1)

    def test_scalar_control_detects_changed_shared_covariance(self):
        a = {'noise': {'service': 1., 'risk': 2., 'cross_sr': -1.5}, 'undefined': None}
        self.assertEqual(spec.scalar_difference(a, copy.deepcopy(a)), 0.)
        b = copy.deepcopy(a);b['noise']['cross_sr'] = 0.
        self.assertGreater(spec.scalar_difference(a, b), .1)

    def test_compact_report_retains_full_query_and_signed_cross_without_mutation(self):
        terms = {t: {'bands': {b: {'noise_RMS': {'joint': 1.}} for b in ('full', 'low', 'rest')},
            'low_rest_noise_cross': {'joint': -2.}} for t in learning.TERMS}
        s = {'regions': {'held': {'terms': terms, 'cross': {b: {'endpoints': {'noise_squared_sum': {'joint': -3.}}} for b in ('full', 'low', 'rest')}}}}
        p = {'groups': {'mixed_pooled': {'pooled': {'v': {'all': s}}}}, 'focus_mixed_removed': {'v': s}, 'focus_contexts': [{'summary': s}]}
        before = copy.deepcopy(p);c = spec.compact_report(p)
        self.assertEqual(p, before)
        actual = c['groups']['mixed_pooled']['pooled']['v']['all']['regions']['held']
        self.assertEqual(set(actual['terms']['coupled']['bands']), {'full'})
        self.assertEqual(actual['terms']['coupled']['low_rest_noise_cross']['joint'], -2.)
        self.assertEqual(actual['cross']['full']['endpoints']['noise_squared_sum']['joint'], -3.)

    def test_scheduler_is_single_unpinned_cpu_and_requires_done_parent(self):
        task = submit.task_specification('unit_coupled')
        self.assertEqual(task['cpu'], 1);self.assertIsNone(task['require_node'])
        self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
        self.assertFalse(any('/results/' in p for p in task['stage_input_paths']))
        parent = {'protocol': spec.source.EXPERIMENT_PROTOCOL, 'contract': spec.source.contract(), 'coverage_qualified': True}
        receipt = {'scheduler': {'status_counts': {'done': 1}, 'exit_code_counts': {'0': 1}}}
        with patch('sys.argv', ['submit', '--run-name', 'unit_coupled']), patch.object(submit, 'inventory', return_value=[]), \
                patch.object(submit.Path, 'exists', return_value=False), \
                patch.object(submit.Path, 'read_text', autospec=True, side_effect=lambda p: json.dumps(receipt if p.name=='completion_receipt.json' else parent)), \
                patch.object(submit.subprocess, 'check_output', return_value='revision'), patch.object(submit, 'write_json'), patch.object(submit, 'execute_bulk') as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]), 1)
            bulk.reset_mock();receipt['scheduler']['exit_code_counts'] = {'1': 1}
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()


if __name__ == '__main__': unittest.main()
