import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import check_grad

from freq_hrl.domains.transit import native_cost_objectives as learning
from scripts import run_native_transit_cost_objectives_stage238 as spec
from scripts.submit_native_transit_cost_objectives_stage238_scheduleurm import task_specification
from tests.test_native_transit_refit_matrix_stage237 import small_fixture


class CostObjectiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(learning.joint.binding.learned.neural, 'STEPS', 3):
            args = small_fixture()
            cell, cached = learning.previous.fit_matrix(*args)
        train, held, panels, reference, paths, source = args[:6]
        cell['selection'] = source['selection']
        cls.args = train, held, panels, reference, paths, cell, cached

    def test_gradients_and_ineligible_costs_do_not_change_losses(self):
        rng = np.random.default_rng(238)
        z = rng.normal(size=(4, 6, 3))
        offset, truth = rng.normal(size=(2, 4, 6))
        gate = np.array([[True, True, True, False, False, False]]*4)
        anchor, theta = np.zeros(4, dtype=int), rng.normal(size=3)
        losses = learning.losses(z, offset, truth, gate, anchor)
        for loss, hessian in losses.values():
            self.assertLess(check_grad(lambda t: loss(t)[0], lambda t: loss(t)[1], theta), 1e-6)
            for column in range(3):
                step = np.eye(3)[column]*1e-5
                np.testing.assert_allclose((loss(theta+step)[1]-loss(theta-step)[1])/2e-5, hessian(theta)[:, column], atol=1e-7)
        altered = truth.copy()
        altered[:, 3:] += 100.
        other = learning.losses(z, offset, altered, gate, anchor)
        for name in losses:
            self.assertEqual(losses[name][0](theta)[0], other[name][0](theta)[0])
            np.testing.assert_array_equal(losses[name][0](theta)[1], other[name][0](theta)[1])

    def test_separable_ranking_is_bounded_and_zero_radius_is_exact(self):
        z = np.zeros((1, 6, 1))
        z[0, 1, 0] = 1.
        gate = np.array([[True, True, False, False, False, False]])
        target = np.array([[0., -2., 0., 0., 0., 0.]])
        rank, hessian = learning.losses(z, np.zeros((1, 6)), target, gate, np.zeros(1, dtype=int))['rank_regret']
        theta, ledger = learning.constrained_fit(rank, hessian, np.array([1.]))
        self.assertAlmostEqual(theta[0], -1., places=4)
        self.assertLess(ledger['loss_after'], ledger['loss_before'])
        zero, ledger = learning.constrained_fit(rank, hessian, np.zeros(1))
        np.testing.assert_array_equal(zero, np.zeros(1))
        self.assertEqual(ledger['iterations'], 0)

    def test_quadratic_ball_matches_analytic_solution_without_barrier_error(self):
        matrix = np.diag([1., 2., 4.])
        desired = np.array([2., 1., -1.])
        def loss(t):
            d = t-desired
            return float(d@matrix@d/2), matrix@d
        theta, fit = learning.quadratic_fit(loss, lambda t: matrix, np.array([1., 0., 0.]))
        self.assertAlmostEqual(np.linalg.norm(theta), 1., places=10)
        self.assertLess(fit['loss_after'], fit['loss_before'])
        self.assertLessEqual(fit['optimality'], 1e-8)
        at_optimum, fit = learning.quadratic_fit(loss, lambda t: matrix, desired)
        np.testing.assert_array_equal(at_optimum, desired)
        self.assertEqual(fit['loss_after'], 0.)

    def test_cached_reference_no_refit_and_all_mse_reproduces(self):
        with patch.object(learning.joint, 'fit_reference', side_effect=AssertionError('Raw reference retraining')):
            cell, cache = learning.compare_cell(*self.args)
        r = cell['reproduction']
        self.assertEqual(r['changed_choices'], 0)
        self.assertEqual(r['choice_comparisons'], 5166)
        self.assertLessEqual(r['metric_max_abs_difference'], 1e-12)
        self.assertLessEqual(r['subspace_prediction_max_abs_difference'], 1e-12)
        self.assertEqual(len(cache), 12)
        for trial in cell['trials']:
            for fits in trial['fits'].values():
                for fit in fits:
                    self.assertLessEqual(fit['norm'], fit['radius']+1e-12)
                    self.assertLessEqual(fit['loss_after'], fit['loss_before']+1e-8)
                    self.assertEqual(fit['parameter_count'], fit['retained_rank'])

    def test_future_and_held_truth_do_not_change_fitted_models(self):
        first, cache = learning.compare_cell(*self.args)
        train, held, panels, reference, paths, source, cached = self.args
        changed = copy.deepcopy(panels)
        for r in train+held:
            values = changed[learning.joint.binding.learned.state_key(r)]['values']
            if r in held: values[:, 1:, 0] += 100.
            else: values[4:, 1:, 0] += 100.
        # Old MSE evaluation controls must change with truth, not the fits.
        altered = copy.deepcopy(source)
        data = {n: learning.joint.causal_data(rs, reference) for n, rs in (('train', train), ('held', held))}
        for index, trial in enumerate(altered['trials']):
            group = tuple(trial['removed_group']) if trial['removed_group'] else None
            ix = np.array([i for i, r in enumerate(train) if (r['root'], r['scene_seed']) != group])
            baseline = paths if index == 0 else cached['paths'][str(index)]['fixed']
            for scope in learning.previous.SCOPES:
                rows = [train[i] for i in ix] if scope == 'fit_training' else (train if scope.startswith('train') else held)
                d = learning.previous.sliced(data['train'], ix) if scope == 'fit_training' else data['train' if scope.startswith('train') else 'held']
                ds = trial['training_draws'] if scope == 'fit_training' else ([0, 1, 2, 3] if scope.endswith('same') else [4, 5, 6, 7])
                v = learning.joint.physical_values(rows, changed, ds)
                trial['arms']['fixed']['scopes'][scope] = learning.previous.metrics(d, d, v, learning.previous.predict(d, baseline, altered['selection']))
        second, other = learning.compare_cell(train, held, changed, reference, paths, altered, cached)
        self.assertEqual(cache, other)
        self.assertNotEqual(first['trials'][0]['scopes']['held_other'], second['trials'][0]['scopes']['held_other'])

    def test_same_rank_models_stay_in_cached_subspace(self):
        train, _, panels, reference, paths = self.args[:5]
        d = learning.joint.causal_data(train, reference)
        v = learning.joint.physical_values(train, panels, (0, 1, 2, 3))
        models, ledger, _ = learning.fit_objectives(d, v, paths['raw'], {'1', '3', '8', '16'})
        mask = np.arange(6)[None, :] != d['raw_choices'][:, None]
        x = learning.joint.relative(d['x'][..., :198], d['raw_choices'])[mask]
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        for f in ledger:
            w = np.array(models[f['objective']]['weights'][f['level']][0])
            k = f['retained_rank']
            np.testing.assert_allclose(w, vt[:k].T@(vt[:k]@w), atol=1e-12)

    def test_protocol_scheduler_and_equal_block_average(self):
        c = spec.contract()
        self.assertEqual(c, json.loads(json.dumps(c)))
        self.assertEqual(c['budget']['new_raw_reference_fits'], 0)
        self.assertEqual(c['budget']['raw_wait_updates'], 0)
        pieces = [{'cases': n, 'variants': {v: {'cost_mse': x} for v in learning.VARIANTS}} for n, x in ((24, 1.), (30, -1.))]
        self.assertEqual(learning.averaged(pieces)['variants']['raw']['cost_mse'], 0.)
        for key in (spec.KEYS[0], (None, None, None, None)):
            t = task_specification('unit_cost_objectives', *key)
            self.assertEqual((t['cpu'], t['ram_mb']), (1, 3072))
            self.assertIsNone(t['require_node'])
            self.assertEqual(t['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
            self.assertNotIn('_raw', t['result_dir'])

    def test_full_report_and_dropped_objective_rejected(self):
        results = {}
        for key in spec.KEYS:
            a = spec.PARTITIONS[key[2]][key[3]]
            groups = [(397, i) for i in range(10)]
            selection = {k: {f: '1@1' for f in learning.previous.diagnostic.FIELDS.values()} for k in learning.joint.ANCHORS}
            trials, lookup = [], {}
            for g in (None, *groups):
                for h in (None, 0, 1):
                    n, ds = (54 if g else 60), list(a if h is None else a[h*2:h*2+2])
                    def stats(scope):
                        count = n if scope == 'fit_training' else 60
                        p = {'cases': count, 'variants': {v: {'cost_mse': float(n)} for v in learning.VARIANTS}}
                        return {'all': p, 'by_root': {'397': p}}
                    fit = [{'objective': o, 'level': '1', 'success': True, 'parameter_count': 1, 'retained_rank': 1,
                        'iterations': 1, 'optimality': 0., 'loss_before': 2., 'loss_after': 1., 'residual_rms_physical': 1., 'residual_rms_limit_physical': 1.} for o in learning.OBJECTIVES[1:]]
                    lookup[g, h] = len(trials)
                    trials.append({'removed_group': list(g) if g else None, 'half': h, 'training_states': n, 'training_draws': ds,
                        'fits': {k: copy.deepcopy(fit) for k in learning.joint.ANCHORS}, 'scopes': {s: stats(s) for s in learning.previous.SCOPES}})
            blocks = [{'removed_group': list(g), 'half': h, 'trials': {f'{n}/{d}': lookup[q, p]
                for n, q in ((54, g), (60, None)) for d, p in ((2, h), (4, None))}} for g in groups for h in (0, 1)]
            results[key] = {'protocol': spec.EXPERIMENT_PROTOCOL, 'contract': spec.contract(), 'key': list(key), 'cell': {
                'selection': selection, 'trials': trials, 'blocks': blocks, 'reproduction': {'choice_comparisons': 68040,
                    'changed_choices': 0, 'metric_max_abs_difference': 0., 'subspace_prediction_max_abs_difference': 0.}}}
        report = spec.summarize(results)
        self.assertEqual(report['reproduction']['choice_comparisons'], 2177280)
        self.assertEqual(report['mixed_pooled']['54/2']['train_same']['all']['matched_blocks'], 320)
        self.assertEqual(report['objective_solves'], 4224)
        first = results[spec.KEYS[0]]['cell']
        first['trials'][0]['fits']['raw'].pop()
        with self.assertRaisesRegex(ValueError, 'Dropped selected-capacity'): spec.summarize(results)


if __name__ == '__main__': unittest.main()
