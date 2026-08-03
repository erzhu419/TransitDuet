import unittest

import numpy as np

from upper.plan_execution import UpperPlanExecutionContract


class UpperPlanExecutionContractTest(unittest.TestCase):
    def test_policy_command_replay_keeps_pre_ema_action(self):
        contract = UpperPlanExecutionContract(
            replay_action_source="policy_command")
        command = np.asarray([-40.0, 20.0], dtype=np.float32)
        executed = np.asarray([-10.0, 5.0], dtype=np.float32)
        np.testing.assert_array_equal(
            contract.replay_action(command, executed), command)

    def test_legacy_replay_keeps_executed_action(self):
        contract = UpperPlanExecutionContract(replay_action_source="executed")
        command = np.asarray([-40.0, 20.0], dtype=np.float32)
        executed = np.asarray([-10.0, 5.0], dtype=np.float32)
        np.testing.assert_array_equal(
            contract.replay_action(command, executed), executed)

    def test_replay_rejects_misaligned_action_shapes(self):
        contract = UpperPlanExecutionContract(
            replay_action_source="policy_command")
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            contract.replay_action([-40.0, 20.0], [-10.0])

    def test_plan_context_is_normalized_and_marks_presence(self):
        contract = UpperPlanExecutionContract(include_plan_context=True)
        context = contract.plan_context(
            {"action": np.asarray([-60.0, 20.0]), "origin": 100.0},
            decision_time_s=550.0,
            action_low=[-60.0, -60.0],
            action_high=[20.0, 20.0],
            replan_interval_s=900.0,
        )
        np.testing.assert_allclose(context, [-1.0, 1.0, 0.5, 1.0])
        self.assertEqual(
            contract.plan_context(
                None, 0.0, [-60.0, -60.0], [20.0, 20.0], 900.0
            ).shape,
            (4,),
        )

    def test_plan_context_rejects_wrong_active_action_shape(self):
        contract = UpperPlanExecutionContract(include_plan_context=True)
        with self.assertRaisesRegex(ValueError, "does not match"):
            contract.plan_context(
                {"action": np.asarray([-20.0]), "origin": 0.0},
                decision_time_s=10.0,
                action_low=[-60.0, -60.0],
                action_high=[20.0, 20.0],
                replan_interval_s=900.0,
            )

    def test_duration_is_decision_interval_aware(self):
        contract = UpperPlanExecutionContract(
            duration_discount=True,
            duration_base_s=900.0,
            duration_min_steps=0.25,
            duration_max_steps=4.0,
        )
        self.assertAlmostEqual(contract.duration_steps(450.0), 0.5)
        self.assertAlmostEqual(contract.duration_steps(1800.0), 2.0)
        self.assertAlmostEqual(contract.duration_steps(0.0), 0.25)


if __name__ == "__main__":
    unittest.main()
