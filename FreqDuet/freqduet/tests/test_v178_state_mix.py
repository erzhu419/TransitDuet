from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from decompose_freqduet_v178_state_mix import decomposition


class StateMixTest(unittest.TestCase):
    def test_pure_composition(self):
        mass = np.array([[.2, .8], [.8, .2]])
        result = decomposition(mass, mass*np.array([.1, .4]))
        self.assertAlmostEqual(result['signed_fractions']['composition'], 1.)
        self.assertAlmostEqual(result['signed_fractions']['within_state'], 0.)

    def test_pure_within_and_absent_category(self):
        mass = np.array([[.5, .5, 0.], [.5, .5, 0.]])
        lower = np.array([[.01, .02, 0.], [.08, .12, 0.]])
        result = decomposition(mass, lower)
        self.assertAlmostEqual(result['signed_fractions']['within_state'], 1.)
        self.assertAlmostEqual(result['signed_fractions']['composition'], 0.)


if __name__ == '__main__':
    unittest.main()
