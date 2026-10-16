import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_local_risk as learning
from scripts import run_native_transit_local_risk_stage272 as spec
from scripts import submit_native_transit_local_risk_stage272_scheduleurm as submit


class LocalRiskTest(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(272)
        z = np.concatenate((np.ones((12, 6, 1)), rng.normal(size=(12, 6, 8))), axis=-1)
        z[:, 0, 1:] = 0.
        priors = rng.normal(size=(12, 6, 3))*.01;priors[:, 0] = 0.
        y = rng.choice([0., .4, -.5], size=(12, 16, 6), p=[.8, .1, .1]);y[:, :, 0] = 0.
        mask = np.ones((12, 6), bool);mask[:, 0] = False;mask[0, 4:] = False
        return z, y, priors, mask

    def test_interaction_depends_on_current_eligible_context_not_ineligible_action(self):
        z, _, _, eligible = self.fixture()
        w = learning.base.state_weights(eligible)
        expected = z[..., 1:]*np.tanh(np.sum(w[..., None]*z[..., 1:], axis=1)[:, None])
        np.testing.assert_allclose(learning.features(z, eligible, 'interaction'), expected)
        changed = z.copy();changed[0, 4:, 1:] = 1e9
        np.testing.assert_array_equal(learning.features(z, eligible, 'interaction')[eligible], learning.features(changed, eligible, 'interaction')[eligible])
        np.testing.assert_array_equal(learning.features(z, eligible, 'linear'), z[..., 1:])

    def test_train_only_normalization_and_matched_parameter_count(self):
        z, y, priors, eligible = self.fixture();ids = np.arange(6)
        one = learning.fit(z[ids], y[ids], priors[ids], eligible[ids], .01)
        changed = z.copy();changed[6:] += 100.
        two = learning.fit(changed[ids], y[ids], priors[ids], eligible[ids], .01)
        for kind in learning.MAPS:
            a = one['maps'][kind];b = two['maps'][kind]
            for field in ('weights', 'center', 'scale'):np.testing.assert_array_equal(a[field], b[field])
            transformed = learning.transform(learning.features(z[ids], eligible[ids], kind), a)
            w = learning.base.state_weights(eligible[ids]);w /= w.sum()
            np.testing.assert_allclose(np.sum(w[..., None]*transformed[..., 1:], axis=(0, 1)), 0., atol=1e-12)
            np.testing.assert_allclose(np.sum(w[..., None]*transformed[..., 1:]**2, axis=(0, 1)), 1., atol=1e-12)
            self.assertEqual(a['weights'].shape, (9, 3))
        self.assertEqual(one['parameters_per_prior_per_map'], 9)
        self.assertEqual(one['head_fits'], 6);self.assertEqual(one['ridge_solver_calls'], 2)

    def test_shared_solve_is_identical_to_independent_prior_fits(self):
        z, y, priors, eligible = self.fixture();m = learning.fit(z, y, priors, eligible, .01)
        for kind in learning.MAPS:
            a = m['maps'][kind];design = learning.transform(learning.features(z, eligible, kind), a)
            for j in range(3):
                beta, _ = learning.base.ridge(design[eligible], (y.mean(axis=1)-priors[..., j])[eligible, None], learning.base.state_weights(eligible)[eligible], .01)
                np.testing.assert_allclose(a['weights'][:, j], beta[:, 0], atol=1e-12)

    def test_pointwise_bounds_hold_on_arbitrarily_large_queries_and_zero_radius(self):
        z, y, priors, eligible = self.fixture();m = learning.fit(z, y, priors, eligible, .01)
        cached = {'cold_mse': np.zeros((12, 6))}
        for j, p in enumerate(learning.PRIORS):
            for suffix in ('residual', 'bounded'):cached[p+'_'+suffix] = priors[..., j]+20.
        originals = np.zeros(12, int)
        for radius in (.002, 0.):
            costs, clips = learning.predict(m, z*1e6, priors, cached, originals, eligible, radius)
            for j, p in enumerate(learning.PRIORS):
                for suffix in learning.CLIPPED:
                    key = p+'_'+suffix
                    self.assertLessEqual(np.max(np.abs(costs[key]-priors[..., j])), radius+1e-12)
                    np.testing.assert_array_equal(costs[key][:, 0], 0.)
                    learning.metrics(costs[key], priors[..., j], y, eligible, radius, clips[key])
            if radius == 0:
                for j, p in enumerate(learning.PRIORS):np.testing.assert_array_equal(costs[p+'_interaction_clip'], priors[..., j])

    def test_loss_alignment_and_clipping_denominators(self):
        z, y, priors, eligible = self.fixture();prior = priors[..., 0]
        delta = np.clip((y.mean(axis=1)-prior)*.5, -.01, .01)
        cost = prior+delta;clips = np.abs(delta) >= .01
        s = learning.metrics(cost, prior, y, eligible, .01, clips)
        r = learning.metrics(prior, prior, y, eligible, .01)
        self.assertAlmostEqual(s['squared_error_sum']-r['squared_error_sum'], s['delta_squared_sum']-2*s['alignment_sum'])
        report = learning.summarize([s, s], True)
        self.assertAlmostEqual(report['MSE'], s['squared_error_sum']/12)
        self.assertAlmostEqual(report['clipped_action_fraction'], np.count_nonzero(eligible&clips)/eligible.sum())
        json.dumps(report, allow_nan=False)

    def test_constant_training_coordinate_stays_finite(self):
        z, y, priors, eligible = self.fixture();z[..., 1] = 0.
        m = learning.fit(z, y, priors, eligible, .01)
        self.assertTrue(np.isfinite(m['maps']['interaction']['weights']).all())
        self.assertEqual(m['maps']['interaction']['scale'][0], 1.)

    def test_complete_cell_reuses_priors_and_freezes_before_opposite_labels(self):
        rng = np.random.default_rng(272)
        rows = [{'root': 397, 'scene_seed': i//6, 'probe_index': i%6} for i in range(60)]
        x = rng.normal(size=(60, 6, 198));x[:, 0] = 0.;basis = np.eye(198)[:, :8]
        data = {'x': x, 'originals': np.zeros(60, int), 'raw_logits': -np.ones((60, 6))}
        eligible = np.ones((60, 6), bool);eligible[:, 0] = False
        values = np.zeros((60, 32, 6, 4));values[..., 1] = rng.choice([0., .4, -.5], size=(60, 32, 6), p=[.8, .1, .1]);values[:, :, 0, 1] = 0.
        folds = [(None, list(range(60)), [('train', list(range(60))), ('held', list(range(60)))])]
        folds += [(list(g), ix, [('retained', ix), ('excluded', val)]) for g, ix, val in spec.cache.learning.joint.binding.policy.prefix_folds(rows)]
        old, queries, cached, errors = [], [], {}, []
        for bank in (0, 1):
            for group, indices, scopes in folds:
                for penalty in spec.PENALTIES:
                    index = len(old)
                    old.append({'bank': bank, 'removed': group, 'penalty': penalty, 'fit_indices': indices,
                        'training_draws': list(range(bank*16, (bank+1)*16)), 'priors': {p: {'radius': .01} for p in learning.PRIORS}})
                    for scope, ids in scopes:
                        yy = values[ids]
                        q = {'index': index, 'scope': scope, 'label': 'held' if scope == 'held' else 'train', 'ids': ids, 'eligible': eligible[ids].tolist()}
                        queries.append(q);costs = {'cold_mse': x[ids, :, 1]*.001}
                        for j, p in enumerate(learning.PRIORS):
                            prior = x[ids, :, 12]*.001
                            costs[p+'_prior'] = prior.copy();costs[p+'_residual'] = prior+x[ids, :, j+2]*.002
                            costs[p+'_bounded'] = prior+np.clip(x[ids, :, j+2]*.002, -.01, .01)
                            costs[p+'_outside_span'] = prior.copy()
                        cached.update({f'{index}_{scope}_{a}': c for a, c in costs.items()})
                        for which, b in (('same', bank), ('opposite', 1-bank)):
                            errors.append({'bank': bank, 'penalty': penalty, 'scope': scope+'_'+which,
                                'metrics': {a: learning.base.metrics(c, None, yy[:, b*16:(b+1)*16, :, 1], eligible[ids], np.array([.4, .5])) for a, c in costs.items()}})
        class Archive(dict):
            def __enter__(self):return self
            def __exit__(self, *args):return False
            @property
            def files(self):return list(self)
        files = {'models.json': old, 'queries.json': queries, 'errors.json': errors};writes = []
        def draws(rs, panels, ds):
            if len(rs) not in (54, 60):self.assertIn('predictions', writes)
            ids = [r['scene_seed']*6+r['probe_index'] for r in rs]
            return values[ids][:, ds]
        with patch.object(spec.Path, 'read_text', autospec=True, side_effect=lambda p: json.dumps(files[p.name])), \
                patch.object(spec.np, 'load', side_effect=[Archive(basis=basis), Archive(cached)]), \
                patch.object(spec.source, 'read_cell', return_value={'reference': {}}), patch.object(spec.Path, 'mkdir'), \
                patch.object(spec, 'write_json'), patch.object(spec.np, 'savez_compressed', side_effect=lambda *a, **k: writes.append('predictions')), \
                patch.object(spec.cache.cached.training.cohort, 'split_rows', return_value=(rows, rows)), \
                patch.object(spec.cache.learning.base, 'causal_data', return_value=data), patch.object(spec.cache.learning.base, 'training_draws', side_effect=draws), \
                patch.object(spec.source.learning, 'fit', side_effect=AssertionError('Cached residual refit')), patch.object(spec.cache.learning, 'fit_inner', side_effect=AssertionError('Prior refit')), \
                patch.object(spec.cache.learning, 'select', side_effect=AssertionError('Selection')):
            result = spec.fit_cell(spec.Path('/tmp/unit_local'), ('mixed_a', 0, 'even_odd', 0), {'mixed_a': []}, {})
        self.assertEqual(result['head_fits'], 396);self.assertEqual(result['ridge_solver_calls'], 132)
        self.assertLess(result['prediction_max_abs_error'], 1e-12);self.assertLess(result['metric_max_abs_error'], 1e-10)
        self.assertLess(result['max_checked_bound_excess'], 1e-12)

    def test_scheduler_single_unpinned_cpu_and_completed_parent_gate(self):
        task = submit.task_specification('unit_local')
        self.assertEqual(task['cpu'], 1);self.assertIsNone(task['require_node'])
        self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
        parent = {'protocol': spec.source.EXPERIMENT_PROTOCOL, 'contract': spec.source.contract(), 'coverage_qualified': True}
        receipt = {'scheduler': {'status_counts': {'done': 1}, 'exit_code_counts': {'0': 1}}}
        with patch('sys.argv', ['submit', '--run-name', 'unit_local']), patch.object(submit, 'inventory', return_value=[]), patch.object(submit.Path, 'exists', return_value=False), \
                patch.object(submit.Path, 'read_text', autospec=True, side_effect=lambda p: json.dumps(receipt if p.name == 'completion_receipt.json' else parent)), \
                patch.object(submit.subprocess, 'check_output', return_value='revision'), patch.object(submit, 'write_json'), patch.object(submit, 'execute_bulk') as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]), 1)
            bulk.reset_mock();receipt['scheduler']['exit_code_counts'] = {'1': 1}
            with self.assertRaises(SystemExit):submit.main()
            bulk.assert_not_called()


if __name__ == '__main__': unittest.main()
