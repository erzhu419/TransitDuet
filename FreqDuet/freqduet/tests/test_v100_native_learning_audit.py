import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_freqduet_v100_native_learning_audit import actor_gap, errors, soft_trip_returns


class NativeAuditTests(unittest.TestCase):
    def test_entropy_optimum_respects_mask(self):
        energy = np.asarray([[0., 1., 2., 3., 4., 5., -100.]])
        mask = np.asarray([[True] * 6 + [False]])
        p = np.exp(-energy)
        p[~mask] = 0
        p /= p.sum(axis=1, keepdims=True)
        result = actor_gap(p, energy, mask, 1.)
        self.assertAlmostEqual(result['kl_actor_to_boltzmann'], 0)
        self.assertAlmostEqual(result['total_variation'], 0)

    def test_interleaved_trip_return_excludes_current_entropy(self):
        rows = [dict(trip=1, state=[0], next_state=[1], done=False, reward=2., cost=3.),
                dict(trip=2, state=[9], next_state=[0], done=True, reward=11., cost=13.),
                dict(trip=1, state=[1], next_state=[0], done=True, reward=5., cost=7.)]
        reward, cost, trips = soft_trip_returns(rows, [-9., -8., -2.], .5, .1)
        np.testing.assert_allclose(reward, [4.6, 11., 5.])
        np.testing.assert_allclose(cost, [6.5, 13., 7.])
        self.assertEqual(trips, 2)

    def test_broken_local_replay_chain_is_not_a_valid_mc_target(self):
        rows = [dict(trip=1, state=[0], next_state=[1], done=False, reward=0., cost=0.),
                dict(trip=1, state=[2], next_state=[0], done=True, reward=0., cost=0.)]
        with self.assertRaises(AssertionError):
            soft_trip_returns(rows, [0., 0.], .99, 1.)

    def test_signed_calibration_error(self):
        result = errors([1., 3.], [2., 6.])
        self.assertEqual(result['bias'], -2.)
        self.assertEqual(result['mae'], 2.)
        self.assertAlmostEqual(result['rmse'], np.sqrt(5.))


if __name__ == '__main__':
    unittest.main()
