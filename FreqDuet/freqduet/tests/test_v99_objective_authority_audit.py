import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v99_objective_authority_audit as audit


class ObjectiveAuditTests(unittest.TestCase):
    def test_fixed_budget_is_only_two_consumed_days(self):
        self.assertEqual(len(audit.ITEMS), 40)
        self.assertEqual(audit.DAYS, (1961001, 1961032))
        self.assertEqual(len(set(audit.ITEMS)), 40)

    def test_zero_mean_does_not_imply_zero_upper_authority(self):
        result = audit.amplitude([350., 370.], reference=360.)
        self.assertEqual(result, dict(mean=0., rms=10., max_abs=10.))

    def test_shadow_reward_masks_and_zero_tie(self):
        p = np.asarray([[1., 0, 0, 0, 0, 0, 0]])
        r = np.asarray([[0., 2., 1., 0, 0, 0, 0]])
        mask = np.asarray([[True, False, True, True, True, True, True]])
        result = audit.shadow_summary(p, r, np.zeros_like(r), [0], mask)
        self.assertEqual(result["greedy_local_holding_s"], 10.)
        self.assertEqual(result["expected_local_net_vs_zero"], 0.)
        tied = audit.shadow_summary(p, np.zeros_like(r), np.zeros_like(r), [0], mask)
        self.assertEqual(tied["greedy_local_holding_s"], 0.)

    def test_curves_use_existing_reward_and_observed_load_apis(self):
        class Regularity:
            def evaluate(self, context, action):
                return SimpleNamespace(reward_adjustment=context * action)
        class Load:
            def evaluate(self, raw, action, **kwargs):
                return float(raw[1]) * action, 0., 0.
        runner = SimpleNamespace(lower_state_dim=2, lower_departure_regularity=Regularity(),
            lower_load_holding_penalty=Load(), env=SimpleNamespace(_base_state_dim=1, lower_context_features=("load",)))
        r, c = audit.reward_curves(runner, np.asarray([0., .5, 99.]), 2.)
        np.testing.assert_array_equal(r, 2. * audit.pilot.step.ACTIONS)
        np.testing.assert_array_equal(c, .5 * audit.pilot.step.ACTIONS)


if __name__ == "__main__":
    unittest.main()
