import unittest

import numpy as np

from upper.credit_assignment import UpperCreditAssignment


class UpperCreditAssignmentTest(unittest.TestCase):
    def test_legacy_modes_reproduce_repeated_centered_assignment(self):
        assigner = UpperCreditAssignment()
        system = assigner.system_rewards(-1.2, 3)
        credit = assigner.gap_credits({1: 1.0, 2: 2.0, 3: 3.0}, [1, 2, 3])

        self.assertTrue(np.allclose(system, [-1.2, -1.2, -1.2]))
        self.assertAlmostEqual(sum(credit.values()), 0.0)
        self.assertGreater(credit[1], credit[2])
        self.assertGreater(credit[2], credit[3])

    def test_terminal_and_uniform_system_reward_preserve_single_episode_reward(self):
        terminal = UpperCreditAssignment(system_reward_mode="terminal")
        uniform = UpperCreditAssignment(system_reward_mode="uniform")

        self.assertTrue(np.allclose(
            terminal.system_rewards(-2.0, 4), [0.0, 0.0, 0.0, -2.0]))
        self.assertTrue(np.allclose(
            uniform.system_rewards(-2.0, 4), [-0.5, -0.5, -0.5, -0.5]))
        self.assertAlmostEqual(terminal.system_rewards(-2.0, 4).sum(), -2.0)
        self.assertAlmostEqual(uniform.system_rewards(-2.0, 4).sum(), -2.0)

    def test_system_reward_weight_scales_without_changing_ownership(self):
        assigner = UpperCreditAssignment(
            system_reward_mode="uniform", system_reward_weight=0.25)
        self.assertTrue(np.allclose(
            assigner.system_rewards(-2.0, 4), [-0.125] * 4))

    def test_reliability_penalty_is_budgeted_once(self):
        uniform = UpperCreditAssignment(
            reliability_reward_mode="uniform",
            reliability_reward_weight=5.0,
        )
        terminal = UpperCreditAssignment(
            reliability_reward_mode="terminal",
            reliability_reward_weight=5.0,
        )

        uniform_rewards = uniform.reliability_rewards(0.1, 0.2, 3)
        terminal_rewards = terminal.reliability_rewards(0.1, 0.2, 3)
        self.assertAlmostEqual(uniform_rewards.sum(), -1.5)
        self.assertTrue(np.allclose(terminal_rewards, [0.0, 0.0, -1.5]))

    def test_absolute_gap_credit_is_local_nonpositive_cost(self):
        assigner = UpperCreditAssignment(
            gap_credit_mode="absolute",
            gap_credit_weight=0.5,
            gap_credit_clip=2.0,
        )
        credit = assigner.gap_credits({10: 0.4, 20: 4.0}, [10, 20, 30])

        self.assertAlmostEqual(credit[10], -0.2)
        self.assertAlmostEqual(credit[20], -1.0)
        self.assertLessEqual(credit[30], 0.0)

    def test_invalid_modes_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "system_reward_mode"):
            UpperCreditAssignment(system_reward_mode="mystery")
        with self.assertRaisesRegex(ValueError, "gap_credit_mode"):
            UpperCreditAssignment(gap_credit_mode="mystery")
        with self.assertRaisesRegex(ValueError, "reliability_reward_mode"):
            UpperCreditAssignment(reliability_reward_mode="mystery")


if __name__ == "__main__":
    unittest.main()
