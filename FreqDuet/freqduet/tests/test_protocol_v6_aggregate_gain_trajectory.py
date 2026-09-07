import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.audit_protocol_v6_aggregate_gain_trajectory import (
    METRICS,
    audit_v22_optimizer_trajectories,
)


CONFIG = "test_v22_candidate"
SPECS = [(CONFIG, "relative", "projected_violation_v1", 0.5, True)]


class ProtocolV6AggregateGainTrajectoryTest(unittest.TestCase):
    def _write_run(self, root: Path, seed: int, episodes=(0, 1, 2, 3)):
        run = root / "shard" / f"{CONFIG}_seed{seed}"
        run.mkdir(parents=True)
        values = [0.01, 0.005, 0.0001, 0.0001]
        frame = pd.DataFrame({
            "ep": list(episodes),
            "lower_regularity_policy_dual_update_mode": (
                "projected_violation_v1"),
            "lower_regularity_passenger_dual_update_mode": (
                "projected_violation_v1"),
            "lower_regularity_policy_constraint_cost_mode": (
                "hf_relative_gain_shortfall_v3"),
            "lower_regularity_policy_augmented_lagrangian_rho": 0.5,
            "lower_regularity_passenger_augmented_lagrangian_rho": 0.5,
        })
        for index, column in enumerate(METRICS.values()):
            frame[column] = [float(index + ep) for ep in range(4)]
        frame[METRICS["regularity_lambda"]] = values
        frame[METRICS["passenger_lambda"]] = [0.01, 0.02, 0.01, 0.02]
        frame[METRICS["regularity_scaled_gap"]] = [-0.2, -0.1, 0.1, -0.1]
        frame.to_csv(run / "diagnostics.csv", index=False)

    def test_reports_floor_hits_and_direction_changes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root, 11)
            self._write_run(root, 13)
            result = audit_v22_optimizer_trajectories(
                root,
                specs=SPECS,
                train_seeds=[11, 13],
                expected_episodes=4,
                window=2,
            )

        self.assertEqual(result["run_count"], 2)
        candidate = result["candidates"][0]
        regularity = candidate["across_seed_metrics"]["regularity_lambda"]
        passenger = candidate["across_seed_metrics"]["passenger_lambda"]
        self.assertEqual(regularity["final_floor_seed_fraction"], 1.0)
        self.assertEqual(regularity["mean_floor_hit_fraction"], 0.5)
        self.assertEqual(passenger["mean_direction_changes"], 2.0)
        gap = candidate["runs"][0]["metrics"]["regularity_scaled_gap"]
        self.assertEqual(gap["positive_fraction"], 0.25)
        self.assertEqual(gap["sign_changes"], 2)

    def test_rejects_missing_episode(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root, 11, episodes=(0, 1, 2, 4))
            with self.assertRaisesRegex(ValueError, "episodes differ"):
                audit_v22_optimizer_trajectories(
                    root,
                    specs=SPECS,
                    train_seeds=[11],
                    expected_episodes=4,
                )


if __name__ == "__main__":
    unittest.main()
