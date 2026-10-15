import copy
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import check_grad, minimize

from freq_hrl.domains.transit import native_head_budget as learning
from scripts import run_native_transit_head_budget_stage250 as spec
from scripts import submit_native_transit_head_budget_stage250_scheduleurm as submit
from tests import test_native_transit_balanced_critic_stage247 as fixtures


class HeadBudgetTest(unittest.TestCase):
    def test_profile_matches_joint_constrained_solve_and_envelope_gradient(self):
        rng = np.random.default_rng(250)
        z, target = rng.normal(size=(5, 6, 3)), rng.normal(size=(5, 6, 2))
        precision = np.exp(rng.normal(size=(5, 15, 2)))
        eligible = np.array([[True, True, True, False, False, False]]*5)
        initial_difference = rng.normal(size=3)
        loss, hessian, difference, optimality = learning.profiled_pair_loss(z, target, precision, eligible, initial_difference)
        point = rng.normal(size=3)
        self.assertLess(check_grad(lambda t: loss(t)[0], lambda t: loss(t)[1], point), 1e-6)
        for col in range(3):
            step = np.eye(3)[col]*1e-5
            np.testing.assert_allclose((loss(point+step)[1]-loss(point-step)[1])/2e-5, hessian(point)[:, col], atol=1e-8)
        for radius in (0., .05, 3.):
            total, fit = learning.optimizer.quadratic_fit(loss, hessian, np.zeros(3), radius=radius)
            theta = np.stack(((total+difference(total))/2, (total-difference(total))/2))
            lo, hi = learning.previous.previous.LO, learning.previous.previous.HI
            dz, dt = z[:, hi]-z[:, lo], target[:, hi]-target[:, lo]
            mask = eligible[:, lo] & eligible[:, hi]
            w = (mask/mask.sum(axis=1, keepdims=True))[..., None]*precision
            def joint(t): return float(np.sum(w*(dz@t.reshape(2, 3).T-dt)**2)/len(z))
            constraints = ({'type': 'eq', 'fun': lambda t: t[:3]+t[3:]} if radius == 0 else
                {'type': 'ineq', 'fun': lambda t: radius**2-np.sum((t[:3]+t[3:])**2)})
            ref = minimize(joint, np.zeros(6), method='SLSQP', constraints=constraints,
                options={'ftol': 1e-12, 'maxiter': 200})
            self.assertTrue(ref.success, ref.message)
            self.assertAlmostEqual(joint(theta), ref.fun, places=9)
            self.assertLessEqual(fit['norm'], radius+1e-10)
            self.assertLessEqual(optimality(total), 1e-8)
            np.testing.assert_allclose(joint(theta), loss(total)[0], atol=1e-12)

    def test_unsupported_directions_and_ineligible_labels_keep_cached_difference(self):
        z = np.zeros((2, 6, 2))
        z[:, :, 0] = np.arange(6)
        target = np.zeros((2, 6, 2))
        gate = np.array([[True, True, False, False, False, False]]*2)
        precision = np.ones((2, 15, 2))
        loss, _, difference, _ = learning.profiled_pair_loss(z, target, precision, gate, np.array([1., 7.]))
        self.assertEqual(difference(np.zeros(2))[1], 7.)
        other = target.copy()
        other[:, 2:] = 999
        poisoned, _, _, _ = learning.profiled_pair_loss(z, other, precision, gate, np.array([1., 7.]))
        self.assertEqual(loss(np.ones(2))[0], poisoned(np.ones(2))[0])
        empty, _, d, _ = learning.profiled_pair_loss(z, target, precision, np.zeros_like(gate), np.array([1., 7.]))
        self.assertEqual(empty(np.zeros(2))[0], 0.)
        np.testing.assert_array_equal(d(np.zeros(2)), [1., 7.])

    def test_factorial_uses_training_only_and_preserves_frozen_head_and_basis(self):
        fixture = fixtures.BalancedCriticTest()
        fixture.setUp()
        source = copy.deepcopy(fixture.cached['all_mse'])
        before = copy.deepcopy(source)
        with patch.object(learning.joint, 'fit_reference', side_effect=AssertionError('Raw retraining')):
            models, ledger = learning.fit_paths(fixture.rows, fixture.data, fixture.draws, source, fixture.fit)
        learning.qualify_fit(ledger)
        self.assertEqual(source, before)
        self.assertEqual(len(ledger['fits']), 12)
        for level in learning.strength.lowrank.LEVELS:
            weights = np.asarray(models['frozen_train']['weights'][level])
            cached = np.asarray(source['weights'][level])
            np.testing.assert_allclose(weights[0]-weights[1], cached[0]-cached[1], atol=1e-12)
        for f in ledger['fits']:
            k = f['retained_rank']
            self.assertEqual(f['total_cost_function_df'], k)
            self.assertEqual(f['component_function_df'], (2 if f['arm'].startswith('free') else 1)*k)
            self.assertGreaterEqual(f['training_radius'], f['old_radius'])
        unused = np.concatenate((fixture.draws, np.full_like(fixture.draws, 999.)), axis=1)
        self.assertEqual((models, ledger), learning.fit_paths(fixture.rows, fixture.data, unused[:, :4], source, fixture.fit))
        legacy, _ = learning.previous.fit_paths(fixture.rows, fixture.data, fixture.draws, fixture.cached, fixture.fit)
        for f in ledger['fits']:
            if f['arm'] == 'frozen_train' and f['radius'] == f['old_radius']:
                np.testing.assert_allclose(models['frozen_train']['weights'][f['level']], legacy['state_gls']['weights'][f['level']], atol=1e-12)

    def test_predictions_keep_controls_and_head_cost_accounting(self):
        fixture = fixtures.BalancedCriticTest()
        fixture.setUp()
        models, _ = learning.fit_paths(fixture.rows, fixture.data, fixture.draws, fixture.cached['all_mse'], fixture.fit)
        models.update(old=fixture.cached['all_mse'], frozen_old=fixture.cached['all_mse'])
        paths = {a: learning.base.candidates(fixture.data, m) for a, m in models.items()}
        selected = {a: {f: 'zero' for f in learning.FILTERS} for a in models}
        pred = learning.predict(paths, selected)
        self.assertEqual(set(pred), set(learning.VARIANTS))
        for ps in paths.values():
            for key, p in ps.items():
                np.testing.assert_allclose(p['cost'], p['components'].sum(axis=-1), atol=1e-12)
        for p in pred.values(): np.testing.assert_array_equal(p['chosen'], fixture.data['raw_choices'])

    def test_scheduler_and_contract_have_zero_replay_and_no_node_binding(self):
        c = spec.contract()
        self.assertEqual(c, json.loads(json.dumps(c)))
        self.assertEqual(c['budget']['new_raw_reference_fits'], 0)
        self.assertEqual(c['budget']['new_native_episodes'], 0)
        for key in (spec.PREFLIGHT, None):
            t = submit.task_specification('unit_head_budget', key)
            self.assertEqual(t['cpu'], 1)
            self.assertIsNone(t['require_node'])
            self.assertEqual(t['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
            self.assertNotIn('_raw', t['result_dir'])
            self.assertFalse(any('/results/' in p for p in t['stage_input_paths']))
            self.assertIn('run_native_transit_head_budget_stage250.py', t['cmd'])

    def test_compact_report_preserves_banks_and_factorial_interaction(self):
        fixture = fixtures.BalancedCriticTest()
        fixture.setUp()
        models, fit = learning.fit_paths(fixture.rows, fixture.data, fixture.draws, fixture.cached['all_mse'], fixture.fit)
        models.update(old=fixture.cached['all_mse'], frozen_old=fixture.cached['all_mse'])
        paths = {a: learning.base.candidates(fixture.data, m) for a, m in models.items()}
        selection = {a: {f: None for f in learning.FILTERS} for a in models}
        pred = learning.predict(paths, selection)
        matrix = learning.previous.metrics(fixture.data, fixture.draws, pred)
        diagnostic = spec.previous.learning.information_error(fixture.draws, fixture.draws, fixture.data, pred)
        diagnostic['ranks'] = {a: {} for a in learning.ARMS}
        banks = []
        for bank in (0, 1):
            m = copy.deepcopy(matrix)
            m['all']['variants']['free_train_pooled_safe']['delta_cost'] = 2+bank
            banks.append({'bank': bank, 'matrix': {'held60': m}, 'diagnostics': {'held60': diagnostic},
                'selection': selection, 'full_fit': fit})
        records = [{'banks': banks}]
        pooled = spec.compact(records, 'held60', None)
        self.assertEqual(pooled['contrasts']['free_train-free_old-frozen_train+frozen_old']['delta_cost'], 2.5)
        self.assertEqual(spec.compact(records, 'held60', 0)['model_bank_exposures'], 1)
        self.assertEqual(pooled['error_contrasts']['free_train-free_old-frozen_train+frozen_old']['fleet']['nonanchor_mse'], 0.)

    def test_expansion_requires_successful_preflight_and_cannot_resubmit(self):
        task = submit.task_specification('unit_head_budget', spec.PREFLIGHT)
        records = [{**task, 'id': 't0', 'status': 'done', 'exit_code': 0}]
        with patch.object(sys, 'argv', ['submit', '--run-name', 'unit_head_budget', '--phase', 'remaining']), \
                patch.object(submit, 'inventory', return_value=records), \
                patch.object(submit.Path, 'read_text', return_value=json.dumps({'contract': spec.contract()})), \
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


if __name__ == '__main__': unittest.main()
