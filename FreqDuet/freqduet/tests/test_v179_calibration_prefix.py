from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v179_calibration_prefix as audit


class CalibrationPrefixTests(unittest.TestCase):
    def atoms(self, n, remaining=4, start=1000):
        return {i: ((start+i-1., i), start+i, remaining, np.array([float(i), 1., -1.])) for i in range(n)}

    def test_strict_label_clock_and_window(self):
        history = audit.Prefix(self.atoms(11))
        value = history.query(1009., 4)['global']
        self.assertEqual(value['n'], 9)
        self.assertEqual(value['latest'], 1008.)
        self.assertEqual(value['mean'][0], 4.)
        value = history.query(4600., 4)['global']
        self.assertEqual(value['n'], 11)
        self.assertEqual(history.query(4611., 4)['global']['n'], 0)

    def test_same_remaining_does_not_relax_support(self):
        atoms = self.atoms(9)
        atoms.update({20+k: v for k, v in self.atoms(8, remaining=5).items()})
        history = audit.Prefix(atoms)
        self.assertEqual(history.query(1011., 4)['same_remaining']['n'], 9)
        value = history.query(1011., 5)['same_remaining']
        self.assertEqual(value['n'], 8)
        self.assertIsNone(value['mean'])

    def test_first_prediction_per_physical_episode(self):
        atoms = {}
        def add(time, point, order):
            truth = 20.-time
            q = dict(denominator_eligible=True, prediction_coarse={}, point_s=point, query_time_s=time,
                episode=[1, 2, 3, True], clocks=dict(query_ready_order=order),
                observable=dict(position=dict(remaining_stops=4)))
            r = dict(label=dict(status='resolved', available_time_s=20., target_ready_time_s=20., full_eta_s=truth),
                residual_s=dict(running=point-truth, service=0., residence=0.))
            audit.add_first(atoms, q, r)
        add(12., 35., 2)
        add(10., 30., 1)
        add(13., 99., 3)
        self.assertEqual(len(atoms), 1)
        self.assertEqual(next(iter(atoms.values()))[3][0], 20.)

    def test_within_distance_removes_between_distance_correlation(self):
        groups = []
        for offset in (0., 10.):
            values = [audit.moment(np.array([x+offset, 0., 0.]), np.array([y+offset, 0., 0.]), y)
                for x, y in ((0., 0.), (1., 0.), (0., 1.), (1., 1.))]
            groups.append(np.mean(values, axis=0))
        correlations = audit.within_correlations(np.array(groups))
        self.assertAlmostEqual(correlations[0, 0], 0.)
        self.assertAlmostEqual(correlations[0, 1], 0.)

    def test_repeated_within_distance_signal(self):
        values = [audit.moment(np.array([x, 0., 0.]), np.array([x, 0., 0.]), float(x > 1.)) for x in range(4)]
        correlations = audit.within_correlations(np.array([np.mean(values, axis=0)]))
        self.assertAlmostEqual(correlations[0, 0], 1.)

    def test_missing_prefix_is_not_a_zero_error_signal(self):
        result = audit.signal_summary([dict(actor=1, day=1, signals=[])], [1], [1], 'same_remaining')
        self.assertEqual(result['support_mass'], 0.)
        self.assertIsNone(result['correlations'])


if __name__ == '__main__':
    unittest.main()
