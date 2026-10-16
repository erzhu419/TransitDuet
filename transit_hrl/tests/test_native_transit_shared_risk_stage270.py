import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import approx_fprime

from freq_hrl.domains.transit import native_shared_risk as learning
from scripts import run_native_transit_shared_risk_stage270 as spec
from scripts import submit_native_transit_shared_risk_stage270_scheduleurm as submit


class SharedRiskTest(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(270)
        z = np.concatenate((np.ones((12, 6, 1)), rng.normal(size=(12, 6, 3))), axis=-1)
        y = rng.choice([0., .4, -.5], size=(12, 16, 6), p=[.85, .08, .07])
        y[:, :, 0] = 0.
        mask = np.ones((12, 6), bool);mask[:, 0] = False;mask[0, 4:] = False
        return z, y, mask

    def test_state_weights_do_not_count_actions_as_independent_states(self):
        _, _, mask = self.fixture()
        np.testing.assert_allclose(learning.state_weights(mask).sum(axis=1), np.ones(12))
        self.assertAlmostEqual(learning.state_weights(mask)[0, 1], 1/3)

    def test_empty_amplitude_is_unidentified_not_imputed(self):
        z, y, mask = self.fixture();y[y < 0] = 0.
        with self.assertRaisesRegex(ValueError, 'unidentified'):learning.fit(z, y, mask, .01)

    def test_training_amplitudes_and_counts_are_signed_and_physical(self):
        _, y, mask = self.fixture()
        np.testing.assert_allclose(learning.amplitudes(y, learning.state_weights(mask)), [.4, .5])
        counts = learning.event_targets(y)
        np.testing.assert_allclose(counts.sum(axis=-1), 1.)
        np.testing.assert_allclose(counts[..., 1], (y > 1e-12).mean(axis=1))

    def test_event_gradient_and_uniform_contrast_curvature(self):
        rng = np.random.default_rng(270);z = rng.normal(size=(17, 4));z[:, 0] = 1.
        target = rng.dirichlet([12, 1, 2], size=17);w = rng.uniform(.2, 1., size=17)
        b = rng.normal(size=8)*.1
        fun = lambda v: learning.event_loss(v, z, target, w, .01)[0]
        np.testing.assert_allclose(learning.event_loss(b, z, target, w, .01)[1], approx_fprime(b, fun, 1e-7), atol=2e-7)
        z = np.ones((1, 1));g = lambda v: learning.event_loss(v, z, np.ones((1, 3))/3, np.ones(1), 0.)[1]
        h = np.column_stack([(g(np.eye(2)[i]*1e-6)-g(-np.eye(2)[i]*1e-6))/2e-6 for i in range(2)])
        np.testing.assert_allclose(h, np.eye(2), atol=1e-9)

    def test_fits_match_parameter_counts_and_converge(self):
        z, y, mask = self.fixture();m = learning.fit(z, y, mask, .01)
        counts = m['checks']['parameter_counts']
        self.assertEqual(counts['signed_mse'], counts['signed_hurdle'])
        self.assertEqual(counts['signed_mse_simplex'], counts['signed_hurdle'])
        self.assertEqual(counts['signed_hurdle'], 10)
        self.assertLess(m['checks']['event_gradient'], 1e-6)
        self.assertLess(m['checks']['normal_error'], 1e-9)

    def test_projection_is_probability_simplex_and_not_a_new_fit(self):
        x = np.array([[-.2, .4, .8], [1.1, -.05, -.05], [.2, .3, .5]])
        p = learning.simplex(x)
        np.testing.assert_allclose(p.sum(axis=-1), 1.)
        self.assertTrue(np.all(p >= 0))
        np.testing.assert_allclose(p[2], x[2])
        np.testing.assert_allclose(p[0], [0., .3, .7])

    def test_predictions_preserve_reference_and_original_anchor(self):
        z, y, mask = self.fixture();m = learning.fit(z, y, mask, .01)
        raw = np.arange(72).reshape(12, 6).astype(float);raw[:, 0] = 0.
        copy = raw.copy();cost, p = learning.predict(m, z, np.zeros(12, int), raw)
        np.testing.assert_array_equal(raw, copy);np.testing.assert_array_equal(cost['raw_reference'], raw)
        np.testing.assert_allclose(cost['signed_mse'], cost['direct_mse'], rtol=0, atol=1e-12)
        for c in cost.values():np.testing.assert_array_equal(c[:, 0], 0.)
        np.testing.assert_allclose(p['signed_hurdle'].sum(axis=-1), 1.)
        self.assertTrue(np.all(p['signed_hurdle'] >= 0))

    def test_excluded_and_opposite_labels_cannot_change_fit(self):
        z, y, mask = self.fixture();indices = np.arange(6, 12)
        one = learning.fit(z[indices], y[indices], mask[indices], .01)
        changed = y.copy();changed[:6] += 123.
        two = learning.fit(z[indices], changed[indices], mask[indices], .01)
        for key in ('direct_mse', 'signed_mse', 'signed_hurdle', 'amplitudes'):
            np.testing.assert_array_equal(one[key], two[key])

    def test_metrics_have_correct_state_denominator_and_amplitude_oracle(self):
        z, y, mask = self.fixture();m = learning.fit(z, y, mask, .01)
        raw = np.zeros((12, 6));cost, p = learning.predict(m, z, np.zeros(12, int), raw)
        row = learning.metrics(cost['signed_hurdle'], p['signed_hurdle'], y, mask, m['amplitudes'])
        self.assertEqual(row['state_exposures'], 12);self.assertEqual(row['action_exposures'], int(mask.sum()))
        self.assertAlmostEqual(row['amplitude_oracle_squared_error_sum'], 0.)
        summary = learning.summarize([row, row])
        self.assertAlmostEqual(summary['MSE'], row['state_balanced_squared_error_sum']/12)
        self.assertEqual(summary['outside_simplex_actions'], 0)
        json.dumps(summary, allow_nan=False)

    def test_main_freezes_before_evaluation_and_never_selects_or_fits_reference(self):
        events = [];model = {'checks': {'normal_error': 0., 'event_gradient': 0., 'cost_identity_error': 0.}}
        def freeze(*args):events.append('freeze');return [model], [], {}, {}
        def evaluate(*args):events.append('evaluate');return []
        with patch('sys.argv', ['runner', '--run-name', 'unit_shared']), \
                patch.object(spec.cache.cached.training.cohort.source.source, 'load_data', return_value=({'training_data': []}, None, [], None)), \
                patch.object(spec.cache.cached.training.cohort.source, 'matched_views', return_value={}), \
                patch.object(spec.cache.sampling, 'read_panels', return_value={}) as panels, \
                patch.object(spec, 'freeze_cell', side_effect=freeze), patch.object(spec, 'evaluate', side_effect=evaluate), \
                patch.object(spec, 'report', return_value={}), patch.object(spec, 'write_json'), \
                patch.object(spec.cache.learning, 'select', side_effect=AssertionError('Selection')):
            spec.main()
        self.assertEqual(events, ['freeze', 'evaluate']*32)
        panels.assert_called_once_with(spec.cache.cached.previous.DATA_RUN)

    def test_full_cell_smoke_has_66_cached_predictions_and_correct_prefix_subsets(self):
        rng = np.random.default_rng(270)
        rows = [{'root': 397, 'scene_seed': 10+i//6, 'probe_index': i%6} for i in range(60)]
        data = {'x': rng.normal(size=(60, 6, 198)), 'components': rng.normal(size=(60, 6, 2)),
            'originals': np.zeros(60, int), 'raw_logits': -np.ones((60, 6))}
        original = data['components'].copy()
        values = np.zeros((60, 16, 6, 4));values[..., 1] = rng.choice([0., .4, -.5], size=(60, 16, 6), p=[.8, .1, .1])
        values[:, :, 0, 1] = 0.
        with patch.object(spec.source.source.source.source.source, 'read_cell', return_value={'reference': {}}), \
                patch.object(spec.cache.cached.training.cohort, 'split_rows', return_value=(rows, rows)), \
                patch.object(spec.cache.learning.base, 'causal_data', return_value=data), \
                patch.object(spec.cache.learning, 'deployment_geometry', return_value={'8': (np.eye(198)[:, :8], None)}), \
                patch.object(spec.cache.learning.base, 'training_draws', return_value=values), \
                patch.object(spec.Path, 'mkdir'), patch.object(spec, 'write_json') as save, patch.object(spec.np, 'savez_compressed') as arrays:
            models, predictions, _, _ = spec.freeze_cell(spec.Path('/tmp/unit_shared'), ('mixed_a', 0, 'even_odd', 0), {'mixed_a': []}, {})
        self.assertEqual(len(models), 66);self.assertEqual(len(predictions), 132)
        self.assertEqual(sum(m['removed'] is None for m in models), 6)
        self.assertTrue(all(len(m['fit_indices']) == (60 if m['removed'] is None else 54) for m in models))
        self.assertTrue(all(m['checks']['parameter_counts']['signed_hurdle'] == 20 for m in models))
        self.assertEqual(save.call_count, 2);arrays.assert_called_once()
        np.testing.assert_array_equal(data['components'], original)

    def test_scheduler_unpinned_single_cpu_and_parent_gate(self):
        task = submit.task_specification('unit_shared')
        self.assertEqual(task['cpu'], 1);self.assertIsNone(task['require_node'])
        self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
        parent = {'protocol': spec.source.EXPERIMENT_PROTOCOL, 'contract': spec.source.contract(), 'coverage_qualified': True}
        receipt = {'scheduler': {'status_counts': {'done': 1}, 'exit_code_counts': {'0': 1}}}
        with patch('sys.argv', ['submit', '--run-name', 'unit_shared']), patch.object(submit, 'inventory', return_value=[]), \
                patch.object(submit.Path, 'exists', return_value=False), \
                patch.object(submit.Path, 'read_text', autospec=True, side_effect=lambda p: json.dumps(receipt if p.name == 'completion_receipt.json' else parent)), \
                patch.object(submit.subprocess, 'check_output', return_value='revision'), patch.object(submit, 'write_json'), patch.object(submit, 'execute_bulk') as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]), 1)
            bulk.reset_mock();receipt['scheduler']['exit_code_counts'] = {'1': 1}
            with self.assertRaises(SystemExit):submit.main()
            bulk.assert_not_called()


if __name__ == '__main__': unittest.main()
