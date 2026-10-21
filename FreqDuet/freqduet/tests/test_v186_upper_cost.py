import ast
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v186_upper_cost as stage


class ReadonlyTests(unittest.TestCase):
    def vector(self, lo, truth, target=True):
        q = dict(point_s=100., query_time_s=1000.)
        p = dict(finite=True, reference_finite=True, bounds=[lo, 110.], reference_bounds=[80., 110.],
            delta_s=80.-lo, target=target)
        r = dict(label=dict(status='resolved', full_eta_s=truth, available_time_s=1000.+truth,
            target_ready_time_s=1000.+truth), residual_s=dict(running=100.-truth, service=0., residence=0.))
        return dict(zip(stage.METRICS, stage.diagnostics(q, p, r)))

    def test_exact_signed_cost_and_tail_saving(self):
        for lo, truth, cost, saving in ((70., 75., 10., 50.), (70., 120., 10., 0.),
                (82., 81., -2., -10.), (80., 75., 0., 0.)):
            row = self.vector(lo, truth)
            self.assertEqual(row['width_cost_s'], cost)
            self.assertEqual(row['lower_saving_s'], saving)
            self.assertEqual(row['score_delta_s'], cost-saving)
        self.assertEqual(self.vector(70., 120.)['above'], 1.)
        with self.assertRaises(AssertionError):
            self.vector(70., 90., target=False)

    def test_original_episode_weights_and_partition_mass(self):
        v = np.ones(len(stage.METRICS))
        base = {'a': np.array([9., 9.]), 'b': np.array([1., 0.])}
        groups = {'a': {('all', 'all'): np.r_[9., 9., 9*v], ('state', 'x'): np.r_[9., 9., 9*v]},
            'b': {('all', 'all'): np.r_[1., 0., 0*v], ('state', 'y'): np.r_[1., 0., 0*v]}}
        rows = stage.weighted_profiles(base, groups)
        global_row = next(r['values'] for r in rows if r['axis'] == 'all')
        self.assertEqual(global_row[:3], [1., .5, 1.])
        np.testing.assert_array_equal(global_row[3:], v)
        self.assertEqual(next(r['values'][0] for r in rows if r['value'] == 'x'), .5)

    def test_missing_state_is_not_zero_effect(self):
        matrix = np.zeros((4, 3+len(stage.METRICS)))
        weights = stage.profiles.bootstrap_weights(2, 2)
        result = stage.readout(matrix, weights)
        self.assertEqual(result['conditional']['above'], dict(mean=None, ci95=None))
        self.assertEqual(result['finite_cells'], 0)

    def test_crossed_weights_and_variation(self):
        rng = np.random.RandomState(20260803)
        a, d = rng.randint(0, 2, (5000, 2)), rng.randint(0, 3, (5000, 3))
        matrix = np.array([[0., 1., 4.], [2., 5., 7.]])
        actual = stage.profiles.bootstrap_weights(2, 3)@matrix.ravel()
        expected = matrix[a[:, :, None], d[:, None, :]].mean(axis=(1, 2))
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        day = stage.upper.variation(np.tile(np.arange(3.), (2, 1)))
        self.assertEqual(day['fractions'], dict(actor=0., day=1., interaction=0.))

    def test_no_collection_fitting_or_training_entry(self):
        tree = ast.parse(Path(stage.__file__).read_text())
        calls = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
        self.assertTrue(calls.isdisjoint({'collect', 'collect_path', 'attach_labels', 'fit_key', 'shared_fit',
            'save_queries', 'train', 'envelope'}))


if __name__ == '__main__':
    unittest.main()
