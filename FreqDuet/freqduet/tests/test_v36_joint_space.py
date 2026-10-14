from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_freqduet_v36_joint_space import PRIMARY, SYSTEM_OUTCOMES, joint_space


class JointSpaceTest(unittest.TestCase):
    def frame(self, triples):
        return pd.DataFrame([{"action_s": 5 * i, **{m: 0.0 for m in SYSTEM_OUTCOMES},
                              **dict(zip(PRIMARY, triple))} for i, triple in enumerate(triples)])

    def test_cannot_buy_journey_with_worse_cv_or_readiness(self):
        frame = self.frame([(10, .2, 100), (9, .21, 80), (9.5, .19, 90), (8, .19, 110)])
        result, chosen = joint_space(frame, frame.iloc[0])
        self.assertEqual(chosen.action_s, 10)
        self.assertEqual(result["has_all_three_strict_improvement"], 1)
        self.assertEqual(result["selected_delta_restricted_total_journey_horizon_min"], -.5)
        self.assertEqual(result["pareto_improving_actions"], 1)

    def test_reference_remains_when_only_tradeoffs_exist(self):
        frame = self.frame([(10, .2, 100), (9, .3, 90), (11, .1, 80)])
        result, chosen = joint_space(frame, frame.iloc[0])
        self.assertEqual(chosen.action_s, 0)
        self.assertEqual(result["has_pareto_improvement"], 0)

    def test_equal_outcomes_are_not_strict_improvements(self):
        frame = self.frame([(10, .2, 100), (10, .2, 100), (10, .19, 100)])
        result, _ = joint_space(frame, frame.iloc[0])
        self.assertEqual(result["has_pareto_improvement"], 1)
        self.assertEqual(result["has_journey_improvement"], 0)
        self.assertEqual(result["has_all_three_strict_improvement"], 0)

    def test_nonfinite_endpoint_is_rejected(self):
        frame = self.frame([(10, .2, 100), (np.nan, .1, 90)])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            joint_space(frame, frame.iloc[0])


if __name__ == "__main__":
    unittest.main()
