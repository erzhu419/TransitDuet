import ast
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v187_running_increment as stage


class RunningIncrementTests(unittest.TestCase):
    def test_fixed_state_centering_removes_composition(self):
        groups = []
        for offset in (0., 20.):
            groups.append(np.mean([stage.moment([r+offset, s], np.full(len(stage.TARGETS), y+offset))
                for r, s, y in product_bits()], axis=0))
        within = stage.correlations(stage.centered(np.array(groups)).sum(axis=0))
        np.testing.assert_allclose(within[:, 0], 0., atol=1e-12)
        marginal = stage.correlations(stage.centered(np.sum(groups, axis=0)))
        self.assertTrue((marginal[:, 0] > .9).all())

    def test_partial_service_removes_shared_service_association(self):
        records = [stage.moment([r+3*s, s], np.full(len(stage.TARGETS), s)) for r, s, _ in product_bits()]
        result = stage.correlations(stage.centered(np.mean(records, axis=0)))
        self.assertTrue(np.isnan(result[:, 2]).all())
        records = [stage.moment([r+3*s, s], np.full(len(stage.TARGETS), r+s)) for r, s, _ in product_bits()]
        result = stage.correlations(stage.centered(np.mean(records, axis=0)))
        np.testing.assert_allclose(result[:, 2], 1., atol=1e-12)

    def test_cell_centering_separates_shared_day_offset(self):
        matrix = np.array([[np.mean([stage.moment([r+offset, s], np.full(len(stage.TARGETS), y+offset))
            for r, s, y in product_bits()], axis=0)] for offset in (0., 20.)])
        pooled = stage.correlations(stage.centered(matrix.mean(axis=0)).sum(axis=0))
        local = stage.correlations(stage.centered(matrix).sum(axis=1).mean(axis=0))
        self.assertTrue((pooled[:, 0] > .9).all())
        np.testing.assert_allclose(local[:, 0], 0., atol=1e-12)

    def test_prefix_strict_completion_window_first_query_and_support(self):
        atoms = {}
        for i in range(9):
            q = dict(denominator_eligible=True, prediction_coarse={}, query_time_s=float(i),
                clocks=dict(query_ready_order=i), episode=[i, i, 2, True], point_s=100.,
                observable=dict(position=dict(remaining_stops=3)))
            r = dict(label=dict(status='resolved', full_eta_s=110., available_time_s=100.+i,
                target_ready_time_s=100.+i), residual_s=dict(running=-10., service=0., residence=0.))
            stage.previous.add_first(atoms, q, r)
            stage.previous.add_first(atoms, q, r)
        prefix = stage.previous.Prefix(atoms)
        self.assertIsNone(prefix.query(108., 3)['global']['mean'])
        history = prefix.query(109., 3)['global']
        self.assertEqual(history['n'], 9)
        self.assertEqual(history['latest'], 108.)
        np.testing.assert_array_equal(-history['mean'][:2], [10., 0.])
        self.assertEqual(prefix.query(3709., 3)['global']['n'], 0)

    def test_batched_moments_and_missing_variation(self):
        rows = np.array([stage.moment([i, i*i], np.full(len(stage.TARGETS), i+1.)) for i in range(5)])
        batch = stage.correlations(stage.centered(np.array([rows.mean(axis=0), rows.sum(axis=0)])))
        for i, values in enumerate((rows.mean(axis=0), rows.sum(axis=0))):
            np.testing.assert_allclose(batch[i], stage.correlations(stage.centered(values)))
        empty = stage.correlations(np.zeros(stage.CSIZE))
        self.assertTrue(np.isnan(empty).all())
        self.assertIsNone(stage.estimate(np.nan, np.array([np.nan]))['mean'])

    def test_no_confirmation_fitting_replay_or_training_calls(self):
        tree = ast.parse(Path(stage.__file__).read_text())
        calls = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
        self.assertTrue(calls.isdisjoint({'collect', 'collect_path', 'attach_labels', 'fit_key', 'shared_fit',
            'save_queries', 'train', 'envelope', 'temporal_fit'}))
        self.assertEqual(stage.DAYS, tuple(range(2951001, 2951065)))
        self.assertEqual(stage.PARTITIONS['distance_stage'], ('remaining_stops', 'stage'))


def product_bits():
    return [(r, s, y) for r in (-1., 1.) for s in (-1., 1.) for y in (-1., 1.)]


if __name__ == '__main__':
    unittest.main()
