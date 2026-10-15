import copy
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_deployment_bundle as learning
from scripts import run_native_transit_deployment_bundle_stage255 as spec
from scripts import submit_native_transit_deployment_bundle_stage255_scheduleurm as submit
from tests import test_native_transit_balanced_critic_stage247 as fixtures


class DeploymentBundleTest(unittest.TestCase):
    def setUp(self):
        f = fixtures.BalancedCriticTest()
        f.setUp()
        self.f, self.geometry = f, learning.deployment_geometry(f.data)

    def test_full_geometry_preserves_prior_and_full_rms_not_inner_rms(self):
        f = self.f
        x = learning.joint.relative(f.data['x'][..., :198], f.data['originals'])
        mask = np.arange(6)[None, :] != f.data['originals'][:, None]
        for level, (basis, coordinates) in self.geometry.items():
            z = x[mask]@basis
            np.testing.assert_allclose(z.T@z/len(z), np.eye(z.shape[1]), atol=1e-12)
            weights = np.asarray(f.cached['all_mse']['weights'][level])
            np.testing.assert_allclose(basis@(weights@coordinates.T).T, weights.T, atol=1e-12)
            total = (weights@coordinates.T).sum(axis=0)
            self.assertAlmostEqual(np.mean((z@total)**2), total@total, places=12)

    def test_same_design_reproduces_existing_solver_and_keeps_prior_immutable(self):
        f = self.f
        before = copy.deepcopy(f.cached)
        with patch.object(learning.joint, 'fit_reference', side_effect=AssertionError('Raw retraining')):
            models, fit = learning.fit_inner(f.rows, f.data, f.draws, f.cached['all_mse'], f.fit, self.geometry)
        old, _ = learning.source.rebuild_inner(f.rows, f.data, f.draws, f.cached['all_mse'], f.fit)
        self.assertEqual(f.cached, before)
        self.assertLessEqual(fit['subspace_max_abs_difference'], 1e-12)
        for model in learning.MODELS:
            for level in learning.strength.lowrank.LEVELS:
                np.testing.assert_allclose(models[model]['weights'][level], old[model]['weights'][level], atol=1e-12)

    def test_prefix_exclusion_retains_full_basis_prior_and_radius(self):
        f = self.f
        group = f.rows[0]['root'], f.rows[0]['scene_seed']
        ix = [i for i, r in enumerate(f.rows) if (r['root'], r['scene_seed']) != group]
        data = {k: np.asarray(v)[ix] for k, v in f.data.items()}
        models, fit = learning.fit_inner([f.rows[i] for i in ix], data, f.draws[ix], f.cached['all_mse'], f.fit, self.geometry)
        radii = {x['level']: x['radius'] for x in f.fit['fits'] if x['objective'] == 'pair_gls'}
        for x in fit['fits']:
            self.assertEqual(x['radius'], radii[x['level']])
            self.assertLessEqual(x['norm'], x['radius']+1e-10)
        self.assertLessEqual(fit['subspace_max_abs_difference'], 1e-12)
        self.assertEqual(set(models), set(learning.MODELS))

    def test_validation_excludes_fit_eval_and_old_A2(self):
        self.assertNotIn('old_A2', learning.SELECTORS)
        for bank in (0, 1):
            for half in (0, 1):
                ds, ev = spec.source.information.learning.draw_assignment(bank, 'expanded', half)
                small, large = [learning.source.validation_draws(bank, half, s) for s in learning.SELECTORS]
                self.assertEqual(small, large[:2])
                self.assertEqual((len(ds), len(small), len(large), len(ev)), (8, 2, 8, 16))
                self.assertFalse(set(ds) & set(large))
                self.assertFalse(set(ev) & set(large))

    def test_predict_changes_only_key_not_full_models_and_bundles_identical_if_keys_equal(self):
        f = self.f
        models, _ = learning.fit_inner(f.rows, f.data, f.draws, f.cached['all_mse'], f.fit, self.geometry)
        before = copy.deepcopy(models)
        paths = {m: learning.base.candidates(f.data, p) for m, p in models.items()}
        candidate = next(c for c in learning.strength.CANDIDATES if c != 'zero')
        selections = {b: {a: {mode: candidate
            for mode in learning.FILTERS} for a in learning.ARMS} for b in learning.BUNDLES}
        pred = learning.predict(paths, selections)
        self.assertEqual(set(pred), set(learning.VARIANTS))
        self.assertEqual(models, before)
        for a in learning.ARMS:
            for mode in learning.FILTERS:
                np.testing.assert_array_equal(pred[f'prefix_refit_{a}_{mode}']['chosen'], pred[f'fixed_bundle_{a}_{mode}']['chosen'])

    def test_scheduler_budget_and_preflight_expansion_gate(self):
        c = spec.contract()
        self.assertEqual(c, json.loads(json.dumps(c)))
        self.assertEqual(c['budget']['inner_solves'], 32*2*20*4*2)
        self.assertEqual(c['budget']['new_full_critic_fits'], 0)
        for key in (spec.PREFLIGHT, None):
            task = submit.task_specification('unit_bundle', key)
            self.assertEqual(task['cpu'], 1)
            self.assertIsNone(task['require_node'])
            self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
            self.assertFalse(any('/results/' in p for p in task['stage_input_paths']))
        records = [{**submit.task_specification('unit_bundle', spec.PREFLIGHT), 'id': 't0', 'status': 'done', 'exit_code': 0}]
        with patch.object(sys, 'argv', ['submit', '--run-name', 'unit_bundle', '--phase', 'remaining']), \
                patch.object(submit, 'inventory', return_value=records), \
                patch.object(submit.Path, 'read_text', return_value=json.dumps({'contract': c})), \
                patch.dict(sys.modules, {'scheduler': SimpleNamespace(load_state=lambda: {'tasks': records})}), \
                patch.object(submit, 'execute_bulk') as bulk:
            submit.main()
            self.assertEqual(len(bulk.call_args.args[0]), 31)
            bulk.reset_mock()
            records[0]['exit_code'] = 1
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()


if __name__ == '__main__': unittest.main()
