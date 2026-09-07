import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.aggregate_lower_replay_joint_projection import (
    aggregate_joint_projection_audit,
)


CONFIG = "projection_candidate"
SPECS = [(CONFIG, "aggregate", "projected_violation_v1", 0.5, True)]


class AggregateJointReplayProjectionTest(unittest.TestCase):
    def _write(self, root: Path, seed: int, *, passenger=0.08):
        run = root / f"{CONFIG}_seed{seed}"
        run.mkdir(parents=True)
        result = {
            "schema": "freqduet-replay-joint-kl-projection-v1",
            "checkpoint": str(
                Path("/remote") / f"{CONFIG}_seed{seed}"
                / "checkpoints" / "training_latest.pt"),
            "checkpoint_episode": 39,
            "constraint_cost_mode": "hf_aggregate_gain_shortfall_v4",
            "valid_transitions": 100,
            "action_bins_s": [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0],
            "regularity_cost_limit": 0.05,
            "passenger_cost_limit": 0.08,
            "per_state_deterministic_joint_feasible_fraction": 0.7,
            "learned_policy": {
                "expected_action_mean_s": 10.0,
                "expected_regularity_cost_mean": 0.06,
                "expected_passenger_cost_mean": 0.12,
                "entropy_mean": 0.5,
            },
            "projection": {
                "joint_feasible": True,
                "converged": True,
                "iterations": 7,
                "multipliers": [2.0, 3.0],
            },
            "projected_policy": {
                "expected_action_mean_s": 7.0,
                "expected_regularity_cost_mean": 0.05,
                "expected_passenger_cost_mean": passenger,
                "entropy_mean": 0.7,
                "kl_from_learned_mean": 0.4,
                "probability_l1_shift_mean": 0.3,
                "argmax_changed_fraction": 0.2,
                "expected_action_change_mean_s": -3.0,
            },
        }
        (run / "result.json").write_text(json.dumps(result))

    def test_requires_and_aggregates_exact_inventory(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, 11)
            self._write(root, 13)
            result = aggregate_joint_projection_audit(
                root, specs=SPECS, train_seeds=[11, 13])

        self.assertEqual(result["run_count"], 2)
        self.assertTrue(result["batch_projection_supported"])
        self.assertFalse(result["per_state_projection_supported"])
        metrics = result["candidates"][0]["metrics"]
        self.assertEqual(metrics["projection_action_change_mean_s"]["mean"], -3.0)

    def test_rejects_missing_seed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, 11)
            with self.assertRaisesRegex(ValueError, "missing projection"):
                aggregate_joint_projection_audit(
                    root, specs=SPECS, train_seeds=[11, 13])

    def test_rejects_projection_budget_violation(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, 11, passenger=0.081)
            with self.assertRaisesRegex(ValueError, "misses passenger"):
                aggregate_joint_projection_audit(
                    root, specs=SPECS, train_seeds=[11])


if __name__ == "__main__":
    unittest.main()
