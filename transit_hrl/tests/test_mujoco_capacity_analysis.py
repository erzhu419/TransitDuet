import unittest

from freq_hrl.experiments.mujoco.capacity_analysis import (
    capacity_gate_decision,
)


def environment_rows(scale, *, passes, drift, reward):
    return [
        {
            "upper_action_scale": scale,
            "environment": environment,
            "environment_gate_pass": passes,
            "relative_drift_reduction": drift,
            "episode_return_difference": reward,
        }
        for environment in ("A", "B", "C")
    ]


class MujocoCapacityAnalysisTest(unittest.TestCase):
    def test_gate_selects_one_global_scale_by_registered_tiebreak(self):
        rows = [
            *environment_rows(0.35, passes=False, drift=0.20, reward=3.0),
            *environment_rows(0.60, passes=True, drift=0.15, reward=4.0),
            *environment_rows(0.80, passes=True, drift=0.25, reward=1.0),
            *environment_rows(1.00, passes=False, drift=0.30, reward=8.0),
        ]
        decision = capacity_gate_decision(
            rows,
            environments=("A", "B", "C"),
        )
        self.assertEqual(decision["status"], "global_capacity_selected")
        self.assertEqual(decision["selected_upper_action_scale"], 0.80)

    def test_gate_rejects_environment_specific_winners(self):
        rows = [
            *environment_rows(0.35, passes=False, drift=0.20, reward=3.0),
            *environment_rows(0.60, passes=False, drift=0.15, reward=4.0),
            *environment_rows(0.80, passes=False, drift=0.25, reward=1.0),
            *environment_rows(1.00, passes=False, drift=0.30, reward=8.0),
        ]
        decision = capacity_gate_decision(
            rows,
            environments=("A", "B", "C"),
        )
        self.assertEqual(decision["status"], "no_global_scale_passed")
        self.assertIsNone(decision["selected_upper_action_scale"])


if __name__ == "__main__":
    unittest.main()
