import copy
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import check_grad, minimize

from freq_hrl.domains.transit import native_head_shrinkage as learning
from scripts import run_native_transit_head_shrinkage_stage251 as spec
from scripts import submit_native_transit_head_shrinkage_stage251_scheduleurm as submit
from scripts import analyze_native_transit_head_shrinkage_stage251 as analysis
from tests import test_native_transit_balanced_critic_stage247 as fixtures


class HeadShrinkageTest(unittest.TestCase):
    def test_noise_normalization_and_service_risk_covariance(self):
        z = np.broadcast_to(np.arange(6)[None, :, None], (2, 6, 1))
        draws = np.zeros((2, 4, 6, 4))
        values = np.array([1., 2.])[:, None, None]*np.array([-1.5, -.5, .5, 1.5])[None, :, None]*np.arange(6)[None, None, :]
        draws[..., 2] = values
        precision, eligible, reliability = np.ones((2, 15, 2)), np.ones((2, 6), bool), np.ones((2, 1, 2))
        noise = learning.difference_noise(z, draws, precision, eligible, reliability, 1.)
        self.assertAlmostEqual(noise['noise_energy'], 25/48)
        scaled = learning.difference_noise(z, draws, precision, eligible, reliability, 10.)
        self.assertAlmostEqual(scaled['noise_energy'], noise['noise_energy']/100)
        common = draws.copy()
        common[..., 2] += np.arange(4)[None, :, None]*10
        self.assertAlmostEqual(learning.difference_noise(z, common, precision, eligible, reliability, 1.)['noise_energy'], noise['noise_energy'])
        draws[..., 1] = values
        # Identical head innovations cancel in the difference score, not add variance.
        self.assertLess(learning.difference_noise(z, draws, precision, eligible, reliability, 1.)['noise_energy'], 1e-20)
        self.assertGreater(noise['weighted_noise_energy'], 0.)

    def test_penalty_rule_freezes_noise_only_and_has_zero_noise_free_endpoint(self):
        self.assertEqual(learning.penalty_rule(1., {'noise_energy': 2., 'weighted_noise_energy': .5}), (0., None))
        self.assertEqual(learning.penalty_rule(3., {'noise_energy': 1., 'weighted_noise_energy': .5}), (2., .25))
        self.assertEqual(learning.penalty_rule(3., {'noise_energy': 0., 'weighted_noise_energy': 0.}), (3., 0.))

    def test_penalized_profile_matches_joint_solve_and_free_endpoint(self):
        rng = np.random.default_rng(251)
        z, target = rng.normal(size=(4, 6, 3)), rng.normal(size=(4, 6, 2))
        precision, eligible = np.exp(rng.normal(size=(4, 15, 2))), np.ones((4, 6), bool)
        prior, point = rng.normal(size=3), rng.normal(size=3)
        for penalty in (0., .3, 100.):
            loss, hessian, difference, optimality = learning.profiled_shrink_loss(z, target, precision, eligible, prior, penalty)
            self.assertLess(check_grad(lambda t: loss(t)[0], lambda t: loss(t)[1], point), 1e-6)
            total, fit = learning.optimizer.quadratic_fit(loss, hessian, np.zeros(3), radius=.1)
            heads = np.stack(((total+difference(total))/2, (total-difference(total))/2))
            dz, weights, _, _ = learning.geometry(z, precision, eligible)
            dt = target[:, learning.pair.HI]-target[:, learning.pair.LO]
            def joint(t):
                h = t.reshape(2, 3)
                return float(np.sum(weights*(dz@h.T-dt)**2)/len(z)+.5*penalty*np.sum((h[0]-h[1]-prior)**2))
            ref = minimize(joint, np.stack((prior/2, -prior/2)).ravel(), method='SLSQP',
                constraints={'type': 'ineq', 'fun': lambda t: .01-np.sum((t[:3]+t[3:])**2)},
                options={'ftol': 1e-12, 'maxiter': 300})
            self.assertTrue(ref.success, ref.message)
            self.assertAlmostEqual(joint(heads), ref.fun, places=8)
            self.assertLessEqual(max(fit['optimality'], optimality(total)), 1e-8)
            for col in range(3):
                step = np.eye(3)[col]*1e-5
                np.testing.assert_allclose((loss(point+step)[1]-loss(point-step)[1])/2e-5, hessian(point)[:, col], atol=1e-8)
            if penalty == 0:
                old = learning.previous.profiled_pair_loss(z, target, precision, eligible, prior)
                np.testing.assert_allclose(difference(point), old[2](point), atol=1e-12)
                self.assertAlmostEqual(loss(point)[0], old[0](point)[0], places=12)
                np.testing.assert_allclose(loss(point)[1], old[0](point)[1], atol=1e-12)

    def test_training_only_fit_preserves_cap_and_shrinks_new_difference(self):
        f = fixtures.BalancedCriticTest()
        f.setUp()
        before = copy.deepcopy(f.cached['all_mse'])
        with patch.object(learning.joint, 'fit_reference', side_effect=AssertionError('Raw retraining')):
            models, fit = learning.fit_paths(f.rows, f.data, f.draws, f.cached['all_mse'], f.fit)
        learning.qualify_fit(fit)
        self.assertEqual(f.cached['all_mse'], before)
        for item in fit['fits']:
            self.assertLessEqual(item['difference_change_rms'], item['free_difference_change_rms']+1e-8)
            self.assertEqual(item['radius'], item['old_radius'])
            self.assertGreaterEqual(item['data_loss_after']+1e-10, item['free_data_loss_after'])
        unused = np.concatenate((f.draws, np.full_like(f.draws, 999.)), axis=1)
        self.assertEqual((models, fit), learning.fit_paths(f.rows, f.data, unused[:, :4], f.cached['all_mse'], f.fit))
        for p in learning.base.candidates(f.data, models['shrunk']).values():
            np.testing.assert_allclose(p['cost'], p['components'].sum(axis=-1), atol=1e-12)

    def test_compact_errors_covariance_diagnostics_and_bank_accounting(self):
        f = fixtures.BalancedCriticTest()
        f.setUp()
        models, fit = learning.fit_paths(f.rows, f.data, f.draws, f.cached['all_mse'], f.fit)
        models.update({a: f.cached['all_mse'] for a in ('old', 'frozen_old', 'free_old')})
        paths = {a: learning.base.candidates(f.data, m) for a, m in models.items()}
        selection = {a: {mode: None for mode in learning.FILTERS} for a in models}
        pred = learning.predict(paths, selection)
        self.assertEqual(set(pred), set(learning.VARIANTS))
        diagnostic = spec.information.learning.information_error(f.draws, f.draws, f.data, pred)
        diagnostic['ranks'] = {a: {} for a in learning.ARMS}
        matrix = learning.balanced.metrics(f.data, f.draws, pred)
        records = [{'banks': [{'bank': b, 'matrix': {'held60': matrix}, 'diagnostics': {'held60': diagnostic},
            'selection': selection, 'full_fit': fit} for b in (0, 1)]}]
        result = spec.compact(records, 'held60', None)
        self.assertEqual(result['model_bank_exposures'], 2)
        self.assertEqual(spec.compact(records, 'held60', 0)['model_bank_exposures'], 1)
        self.assertEqual(result['contrasts']['shrunk-free_old']['delta_cost'], 0.)
        self.assertEqual(result['full_fits']['16']['fits'], 2)
        e = result['models']['shrunk_pooled_safe']['evaluation']
        self.assertAlmostEqual(result['models']['shrunk_pooled_safe']['difference_error']['nonanchor_mse'],
            2*(e['service']['nonanchor_mse']+e['fleet']['nonanchor_mse'])-e['cost']['nonanchor_mse'])

    def test_scheduler_contract_and_expansion_require_successful_preflight(self):
        c = spec.contract()
        self.assertEqual(c, json.loads(json.dumps(c)))
        self.assertEqual(c['budget']['new_raw_reference_fits'], 0)
        self.assertEqual(c['budget']['new_native_episodes'], 0)
        for key in (spec.PREFLIGHT, None):
            t = submit.task_specification('unit_head_shrinkage', key)
            self.assertEqual(t['cpu'], 1)
            self.assertIsNone(t['require_node'])
            self.assertEqual(t['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
            self.assertNotIn('_raw', t['result_dir'])
            self.assertFalse(any('/results/' in p for p in t['stage_input_paths']))
        records = [{**submit.task_specification('unit_head_shrinkage', spec.PREFLIGHT), 'id': 't0', 'status': 'done', 'exit_code': 0}]
        with patch.object(sys, 'argv', ['submit', '--run-name', 'unit_head_shrinkage', '--phase', 'remaining']), \
                patch.object(submit, 'inventory', return_value=records), \
                patch.object(submit.Path, 'read_text', return_value=json.dumps({'contract': c})), \
                patch.dict(sys.modules, {'scheduler': SimpleNamespace(load_state=lambda: {'tasks': records})}), \
                patch.object(submit, 'execute_bulk') as bulk:
            submit.main()
            tasks = bulk.call_args.args[0]
            self.assertEqual(len(tasks), 31)
            bulk.reset_mock()
            records[0]['exit_code'] = 1
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()
            records[0]['exit_code'] = 0
            records.append({**tasks[0], 'id': 't1', 'status': 'queued'})
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()

    def test_cached_selection_swap_decomposition_and_analysis_task(self):
        paths = {a: {'zero': a+'0', '1': a+'1'} for a in ('free_old', 'shrunk')}
        predictions = analysis.selection_swap(paths, {'free_old': {'pooled_safe': '1'}, 'shrunk': {'pooled_safe': None}})
        self.assertEqual(predictions['shrunk_on_free_old'], 'shrunk1')
        self.assertEqual(predictions['free_old_on_shrunk'], 'free_old0')
        fields = ('delta_cost', 'delta_wait_min', 'delta_risk', 'delta_service')
        swaps = {name: dict.fromkeys(fields, value) for name, value in
            (('free_old_on_free_old', 2.), ('shrunk_on_free_old', 5.), ('free_old_on_shrunk', 4.), ('shrunk_on_shrunk', 6.))}
        d = analysis.decompose(swaps)
        self.assertEqual(d['model_at_free_selected_key']['delta_cost'], 3.)
        self.assertEqual(d['key_change_on_shrunk_model']['delta_cost'], 1.)
        self.assertEqual(d['total']['delta_cost'], 4.)
        task = submit.task_specification('unit_head_shrinkage', operation='analysis')
        self.assertIn('/analysis', task['result_dir'])
        self.assertIn('analyze_native_transit_head_shrinkage_stage251.py', task['cmd'])


if __name__ == '__main__': unittest.main()
