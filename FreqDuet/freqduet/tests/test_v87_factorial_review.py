from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_freqduet_v87_factorial_review import attribution


class FactorialReviewTest(unittest.TestCase):
    def test_forward_changes_add_without_counting_interaction_twice(self):
        result = attribution(dict(early=5., late=10., early_latemean=6., late_earlymean=8.))
        self.assertEqual(result["late_minus_early"], 5.)
        self.assertEqual(result["mean_preference"], 1.5)
        self.assertEqual(result["state_response"], 3.5)
        self.assertEqual(result["forward_interaction"], 1.)

    def test_identical_laws_have_no_component_effect(self):
        self.assertEqual(attribution(dict(early=8., late=8., early_latemean=8., late_earlymean=8.)),
            dict(late_minus_early=0., mean_preference=0., state_response=0., forward_interaction=0.))


if __name__ == "__main__":
    unittest.main()
