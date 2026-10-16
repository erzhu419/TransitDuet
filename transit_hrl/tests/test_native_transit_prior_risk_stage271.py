import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_prior_risk as learning
from scripts import run_native_transit_prior_risk_stage271 as spec
from scripts import submit_native_transit_prior_risk_stage271_scheduleurm as submit


class PriorRiskTest(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(271)
        z = np.concatenate((np.ones((12, 6, 1)), rng.normal(size=(12, 6, 3))), axis=-1)
        prior = rng.normal(size=(12, 6))*.1;prior[:, 0] = 0.
        y = rng.choice([0., .4, -.5], size=(12, 16, 6), p=[.8, .1, .1]);y[:, :, 0] = 0.
        eligible = np.ones((12, 6), bool);eligible[:, 0] = False
        return z, y, prior, eligible

    def test_zero_radius_is_exact_prior_without_fictitious_finite_kkt(self):
        z, y, prior, eligible = self.fixture()
        model = learning.fit(z, y, prior, eligible, .01, 0.)
        np.testing.assert_array_equal(model['bounded'], 0.)
        self.assertIsNone(model['checks']['kkt']);self.assertIsNone(model['checks']['multiplier'])
        cold, _ = learning.base.ridge(z[eligible], y.mean(axis=1)[eligible, None], learning.base.state_weights(eligible)[eligible], .01)
        pred, _ = learning.predict(z, prior, cold, model, learning.project(z, prior, eligible), np.zeros(12, int))
        np.testing.assert_array_equal(pred['bounded'], prior)

    def test_large_radius_reuses_unconstrained_fit(self):
        z, y, prior, eligible = self.fixture();m = learning.fit(z, y, prior, eligible, .01, 100.)
        np.testing.assert_array_equal(m['unbounded'], m['bounded'])
        self.assertEqual(m['checks']['constraint_linear_solves'], 0)

    def test_active_rms_ball_satisfies_kkt_and_reduces_training_objective(self):
        z, y, prior, eligible = self.fixture();m = learning.fit(z, y, prior, eligible, .01, .002)
        self.assertTrue(m['checks']['active']);self.assertGreater(m['checks']['multiplier'], 0.)
        w = learning.base.state_weights(eligible);delta = (z@m['bounded'])[..., 0]
        self.assertAlmostEqual(float(np.sqrt(np.sum(w*delta**2)/w.sum())), .002)
        self.assertLess(m['checks']['kkt'], 1e-9)
        target = y.mean(axis=1)-prior
        value = np.sum(w*(delta-target)**2)/w.sum()+.01*np.sum(m['bounded'][1:]**2)
        self.assertLess(value, np.sum(w*target**2)/w.sum())

    def test_span_plus_shrink_exactly_recovers_prior_vs_cold_response(self):
        z, y, prior, eligible = self.fixture()
        w = learning.base.state_weights(eligible)[eligible]
        cold, _ = learning.base.ridge(z[eligible], y.mean(axis=1)[eligible, None], w, .01)
        m = learning.fit(z, y, prior, eligible, .01, .01)
        pred, p = learning.predict(z, prior, cold, m, learning.project(z, prior, eligible), np.zeros(12, int))
        np.testing.assert_allclose(p['gap'], p['outside']+p['shrink'], atol=1e-12)
        self.assertLess(p['identity_error'], 1e-12)
        self.assertTrue(np.all(pred['residual'][:, 0] == 0))
        stats = learning.response_stats(p, prior, pred['bounded'], eligible, .01)
        self.assertAlmostEqual(stats['gap_squared_sum'], stats['outside_squared_sum']+stats['shrink_squared_sum']+stats['signed_cross_sum'])

    def test_linear_prior_has_no_outside_span_component(self):
        z, y, _, eligible = self.fixture();prior = (z@np.arange(4)[:, None]*.01)[..., 0];prior[:, 0] = 0.
        cold, _ = learning.base.ridge(z[eligible], y.mean(axis=1)[eligible, None], learning.base.state_weights(eligible)[eligible], .01)
        m = learning.fit(z, y, prior, eligible, .01, .01)
        _, p = learning.predict(z, prior, cold, m, learning.project(z, prior, eligible), np.zeros(12, int))
        np.testing.assert_allclose(p['outside'], 0., atol=1e-12)

    def test_excluded_future_labels_cannot_change_residual_or_projection(self):
        z, y, prior, eligible = self.fixture();indices = np.arange(6)
        a = learning.fit(z[indices], y[indices], prior[indices], eligible[indices], .01, .01)
        changed = y.copy();changed[6:] += 100.
        b = learning.fit(z[indices], changed[indices], prior[indices], eligible[indices], .01, .01)
        np.testing.assert_array_equal(a['bounded'], b['bounded'])
        np.testing.assert_array_equal(a['unbounded'], b['unbounded'])

    def test_full_cached_cell_freezes_before_evaluation_without_cold_or_prior_refitting(self):
        rng = np.random.default_rng(271)
        rows = [{'root': 397, 'scene_seed': i//6, 'probe_index': i%6} for i in range(60)]
        basis = np.eye(198)[:, :8];x = rng.normal(size=(60, 6, 198));x[:, 0] = 0.
        z = np.concatenate((np.ones((60, 6, 1)), x@basis), axis=-1)
        raw = x[..., 12]*.01
        components = np.stack((raw*2, raw), axis=-1)
        d = {'x': x, 'components': components, 'originals': np.zeros(60, int), 'raw_logits': -np.ones((60, 6))}
        eligible = np.ones((60, 6), bool);eligible[:, 0] = False
        values = np.zeros((60, 32, 6, 4));values[..., 1] = rng.choice([0., .4, -.5], size=(60, 32, 6), p=[.8, .1, .1]);values[:, :, 0, 1] = 0.
        masks = [(None, list(range(60)), [('train', list(range(60))), ('held', list(range(60)))])]
        masks += [(list(g), ix, [('retained', ix), ('excluded', val)]) for g, ix, val in spec.cache.learning.joint.binding.policy.prefix_folds(rows)]
        full = {str(b): {p: {'scale': 1., 'weights': {'8': np.zeros((2, 198)).tolist()}} for p in ('free_old', 'shrunk')} for b in (0, 1)}
        prefix = [{'bank': b, 'held_group': g, 'models': full[str(b)], 'training_draws': list(range(b*16, (b+1)*16)), 'fit': {'fits': [{'level': '8', 'old_radius': .01}]}} for b in (0, 1) for g, _, _ in masks if g is not None]
        old, queries, errors, arrays = [], [], [], {'basis': basis}
        for bank in (0, 1):
            ds = list(range(bank*16, (bank+1)*16));y = values[:, ds][..., 1]
            for group, indices, qs in masks:
                for penalty in spec.source.learning.PENALTIES:
                    ix = np.asarray(indices);beta, _ = learning.base.ridge(z[ix][eligible[ix]], y[ix].mean(axis=1)[eligible[ix], None], learning.base.state_weights(eligible[ix])[eligible[ix]], penalty)
                    index = len(old);old.append({'bank': bank, 'removed': group, 'penalty': penalty, 'fit_indices': indices, 'training_draws': ds, 'direct_mse': beta.tolist(), 'amplitudes': [.4, .5]})
                    for scope, ids in qs:
                        chosen = np.asarray(ids);cold = (z[chosen]@beta)[..., 0];cold[:, 0] = 0.
                        queries.append({'index': index, 'scope': scope, 'label': 'held' if scope == 'held' else 'train', 'ids': ids, 'eligible': eligible[chosen].tolist()})
                        arrays[f'{index}_{scope}_direct_mse_cost'] = cold;arrays[f'{index}_{scope}_raw_reference_cost'] = raw[chosen]
                        for which, b in (('same', bank), ('opposite', 1-bank)):
                            yy = values[chosen][:, b*16:(b+1)*16, :, 1]
                            errors.append({'bank': bank, 'removed': group, 'penalty': penalty, 'scope': scope+'_'+which,
                                'metrics': {name: learning.base.metrics(cost, None, yy, eligible[chosen], np.array([.4, .5])) for name, cost in (('direct_mse', cold), ('raw_reference', raw[chosen]))}})
        class Archive(dict):
            @property
            def files(self):return list(self)
            def __enter__(self):return self
            def __exit__(self, *args):return False
        writes = []
        def draw(rs, panels, ds):
            if len(rs) != 60:self.assertIn('predictions', writes)
            ids = [r['scene_seed']*6+r['probe_index'] for r in rs]
            return values[ids][:, ds]
        def read(run, key, name):
            if run == spec.PREFIX_RUN:return prefix
            if run == spec.cache.cached.SOURCE_RUN:return full
            return {'reference': {}}
        files = {'models.json': old, 'queries.json': queries, 'errors.json': errors}
        with patch.object(spec, 'read_cell', side_effect=read), patch.object(spec.Path, 'read_text', autospec=True, side_effect=lambda p: json.dumps(files[p.name])), \
                patch.object(spec.np, 'load', return_value=Archive(arrays)), patch.object(spec.Path, 'mkdir'), \
                patch.object(spec, 'write_json', side_effect=lambda *a: writes.append('json')), patch.object(spec.np, 'savez_compressed', side_effect=lambda *a, **k: writes.append('predictions')), \
                patch.object(spec.cache.cached.training.cohort, 'split_rows', return_value=(rows, rows)), patch.object(spec.cache.learning.base, 'causal_data', return_value=d), \
                patch.object(spec.cache.learning.base, 'training_draws', side_effect=draw), \
                patch.object(learning.base, 'fit', side_effect=AssertionError('Cold refit')), patch.object(spec.cache.learning, 'fit_inner', side_effect=AssertionError('Prior refit')), \
                patch.object(spec.cache.learning, 'select', side_effect=AssertionError('Selection')):
            result = spec.cached_cell(spec.Path('/tmp/unit_prior'), ('mixed_a', 0, 'even_odd', 0), {'mixed_a': []}, {})
        self.assertEqual(result['fits'], 198);self.assertEqual(result['projection_solves'], 66)
        self.assertLess(result['max_metric_error'], 1e-10);self.assertLess(result['max_prediction_error'], 1e-10)

    def test_scheduler_unpinned_cpu_and_completed_parent_gate(self):
        task = submit.task_specification('unit_prior')
        self.assertEqual(task['cpu'], 1);self.assertIsNone(task['require_node'])
        self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
        parent = {'protocol': spec.source.EXPERIMENT_PROTOCOL, 'contract': spec.source.contract(), 'coverage_qualified': True}
        receipt = {'scheduler': {'status_counts': {'done': 1}, 'exit_code_counts': {'0': 1}}}
        with patch('sys.argv', ['submit', '--run-name', 'unit_prior']), patch.object(submit, 'inventory', return_value=[]), patch.object(submit.Path, 'exists', return_value=False), \
                patch.object(submit.Path, 'read_text', autospec=True, side_effect=lambda p: json.dumps(receipt if p.name == 'completion_receipt.json' else parent)), \
                patch.object(submit.subprocess, 'check_output', return_value='revision'), patch.object(submit, 'write_json'), patch.object(submit, 'execute_bulk') as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]), 1)
            bulk.reset_mock();receipt['scheduler']['exit_code_counts'] = {'1': 1}
            with self.assertRaises(SystemExit):submit.main()
            bulk.assert_not_called()


if __name__ == '__main__': unittest.main()
