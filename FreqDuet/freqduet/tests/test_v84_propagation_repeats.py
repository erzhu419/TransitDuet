from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_freqduet_v84_propagation_repeats import REFERENCES, ITEMS, future_seed, describe, action_noise, pilot


class PropagationRepeatsTest(unittest.TestCase):
    def test_full_grid_keeps_every_prefix_future_action_and_mode(self):
        self.assertEqual(len(REFERENCES), 24 * 8)
        self.assertEqual(len(ITEMS), 24 * 8 * 2 * 7)
        self.assertEqual(len(set(future_seed(c, f) for c, f in REFERENCES)), len(REFERENCES))

    def test_future_mean_uncertainty_uses_replicates_not_actions(self):
        result = describe([-1.] * 8)
        self.assertEqual(result["n"], 8)
        self.assertEqual(result["ci95"], [-1., -1.])
        self.assertEqual(result["negative"], 8)
        self.assertEqual(describe([-2., 2.] * 4)["mean"], 0.)

    def test_paired_noise_cancels_common_future_variation(self):
        values = np.arange(8.)[:, None] * 100. + np.arange(7.)[None, :]
        result = action_noise(values, 3)
        np.testing.assert_array_equal(result["mean_deltas"], np.arange(7.) - 3.)
        np.testing.assert_array_equal(result["paired_sd"], np.zeros(7))
        self.assertTrue(all(x > 0 for x in result["independent_sd"]))

    def test_forked_event_uniforms_preserve_prefix_and_match_by_event(self):
        left, right = [pilot.event.EventUniforms(77) for _ in range(2)]
        self.assertEqual(left.draw(1, 2, 0), right.draw(1, 2, 0))
        for sampler in (left, right):
            sampler.seed = future_seed(0, 0) + 70000000
        right.draw(5, 4, 1)
        self.assertEqual(left.draw(1, 2, 0), right.draw(1, 2, 0))
        other = pilot.event.EventUniforms(future_seed(0, 1) + 70000000)
        self.assertNotEqual(left.draw(2, 3, 1)[1], other.draw(2, 3, 1)[1])


if __name__ == "__main__":
    unittest.main()
