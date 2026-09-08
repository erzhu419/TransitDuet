import unittest

import numpy as np

from runner_v3 import TransitDuetV2Runner
from scripts.audit_freqduet_prefix_counterfactual import (
    BranchResult,
    assert_prefix_equal,
    first_difference,
    label_row,
    perturb_direction_first_knot,
)


class FakePlanner:
    coefficient_parameterization = "full"
    basis_per_direction = 4
    shared_directions = False


def branch(label, offset=None):
    action = np.asarray([1.0, 2.0], dtype=np.float32)
    return BranchResult(
        label=label,
        candidate_offset_s=offset,
        target_identity={"decision_index": 3, "time_s": 10.0},
        context={"upper_state_000": 0.25},
        actor_action=action.copy(),
        executed_action=action.copy(),
        changed_index=None,
        terminal_dispatch=True,
        upper_prefix=[{"action": action.copy()}],
        lower_prefix=[{"previous_action_s": 5.0, "action": action.copy()}],
        rng_state={"numpy": ("x", np.asarray([1, 2], dtype=np.uint32))},
        policy_digest="policy",
        episode_row={
            "service_cost_restricted": 1.25,
            "wall_env_s": 10.0,
        },
    )


class PrefixCounterfactualTest(unittest.TestCase):
    def test_offline_hook_is_behavior_neutral_when_absent(self):
        runner = TransitDuetV2Runner.__new__(TransitDuetV2Runner)
        runner.upper_action_dim = 2
        runner.upper_action_low = np.asarray([-10.0, -10.0], dtype=np.float32)
        runner.upper_action_high = np.asarray([10.0, 10.0], dtype=np.float32)
        action = np.asarray([1.0, -1.0], dtype=np.float32)
        observed, terminal = runner._apply_offline_upper_action_intervention(
            action, True, np.asarray([0.5]), object(), 12.0)
        np.testing.assert_array_equal(observed, action)
        self.assertTrue(terminal)

    def test_offline_hook_applies_post_selector_action(self):
        runner = TransitDuetV2Runner.__new__(TransitDuetV2Runner)
        runner.upper_action_dim = 2
        runner.upper_action_low = np.asarray([-10.0, -10.0], dtype=np.float32)
        runner.upper_action_high = np.asarray([10.0, 10.0], dtype=np.float32)
        runner._offline_upper_action_intervention = lambda **_: {
            "action_vec": [3.0, 4.0],
            "write_terminal_dispatch": False,
        }
        observed, terminal = runner._apply_offline_upper_action_intervention(
            np.asarray([1.0, -1.0]), True, np.asarray([0.5]), object(), 12.0)
        np.testing.assert_array_equal(observed, np.asarray([3.0, 4.0]))
        self.assertFalse(terminal)

    def test_offline_hook_rejects_invalid_actions(self):
        runner = TransitDuetV2Runner.__new__(TransitDuetV2Runner)
        runner.upper_action_dim = 2
        runner.upper_action_low = np.asarray([-10.0, -10.0], dtype=np.float32)
        runner.upper_action_high = np.asarray([10.0, 10.0], dtype=np.float32)
        for candidate, message in (
            ([1.0], "wrong action size"),
            ([1.0, np.nan], "non-finite"),
            ([1.0, 11.0], "out-of-range"),
        ):
            with self.subTest(candidate=candidate):
                runner._offline_upper_action_intervention = (
                    lambda candidate=candidate, **_: {"action_vec": candidate})
                with self.assertRaisesRegex(ValueError, message):
                    runner._apply_offline_upper_action_intervention(
                        np.asarray([0.0, 0.0]), True, np.asarray([0.5]),
                        object(), 12.0)

    def test_first_knot_perturbation_is_direction_specific(self):
        action = np.zeros(8, dtype=np.float32)
        low = np.full(8, -120.0, dtype=np.float32)
        high = np.full(8, 120.0, dtype=np.float32)
        up, up_index = perturb_direction_first_knot(
            action, 20.0, True, FakePlanner(), low, high)
        down, down_index = perturb_direction_first_knot(
            action, -20.0, False, FakePlanner(), low, high)
        self.assertEqual(up_index, 0)
        self.assertEqual(down_index, 4)
        self.assertEqual(float(up[0]), 20.0)
        self.assertEqual(float(down[4]), -20.0)
        self.assertEqual(int(np.count_nonzero(up)), 1)
        self.assertEqual(int(np.count_nonzero(down)), 1)

    def test_prefix_contract_reports_first_exact_difference(self):
        reference = branch("reference")
        candidate = branch("candidate", 20.0)
        assert_prefix_equal(reference, candidate)
        candidate.lower_prefix[0]["previous_action_s"] = 0.0
        with self.assertRaisesRegex(RuntimeError, "lower action prefix mismatch"):
            assert_prefix_equal(reference, candidate)
        self.assertIn(
            "root.lower",
            first_difference(
                {"lower": np.asarray([1.0, 2.0])},
                {"lower": np.asarray([1.0, 3.0])},
            ),
        )

    def test_label_uses_standard_episode_cost_delta(self):
        reference = branch("reference")
        candidate = branch("candidate", 20.0)
        candidate.changed_index = 0
        candidate.executed_action[0] = 3.0
        candidate.episode_row["service_cost_restricted"] = 1.0
        row = label_row(candidate, reference)
        self.assertEqual(row["candidate_method"], "actor_firstknot_p20")
        self.assertEqual(row["candidate_changed_index"], 0)
        self.assertAlmostEqual(
            row["episode_service_cost_restricted_delta_vs_actor"], -0.25)


if __name__ == "__main__":
    unittest.main()
