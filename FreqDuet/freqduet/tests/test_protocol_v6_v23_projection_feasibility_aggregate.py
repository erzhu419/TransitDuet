import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.aggregate_protocol_v6_v23_projection_feasibility import (
    aggregate_v23_projection_feasibility,
)


CONFIG = "aggregate_candidate"
SPECS = [(CONFIG, "aggregate", "projected_violation_v1", 0.0, True)]


class V23ProjectionFeasibilityAggregateTest(unittest.TestCase):
    def _write(self, root: Path, seed: int, *, passes=True):
        output = root / f"{CONFIG}_seed{seed}"
        output.mkdir(parents=True)
        count = 256 if passes else 255
        result = {
            "schema": "freqduet-v23-projection-feasibility-v1",
            "config": CONFIG,
            "train_seed": seed,
            "checkpoint_episode": 39,
            "constraint_cost_mode": "hf_aggregate_gain_shortfall_v4",
            "action_bins_s": [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0],
            "configured_regularity_budget": 0.05,
            "configured_passenger_budget": 0.08,
            "locked_regularity_replay_target": 0.036,
            "locked_passenger_replay_target": 0.075,
            "passes": passes,
            "full_replay": {
                "passes": True,
                "joint_feasible": True,
                "converged": True,
                "learned_regularity_cost": 0.05,
                "learned_passenger_cost": 0.10,
                "projected_regularity_cost": 0.036,
                "projected_passenger_cost": 0.075,
                "projection_action_change_mean_s": -1.0,
                "projection_kl": 0.2,
            },
            "minibatches": {
                "batch_size": 512,
                "batch_count": 256,
                "sample_seed": 230908,
                "pass_count": count,
                "joint_feasible_count": count,
                "converged_count": count,
                "budget_satisfied_count": count,
                "passes": passes,
                "metrics": {
                    "projected_regularity_cost": {
                        "mean": 0.036, "minimum": 0.035,
                        "maximum": 0.036},
                    "projected_passenger_cost": {
                        "mean": 0.075, "minimum": 0.074,
                        "maximum": 0.075},
                    "projection_action_change_mean_s": {
                        "mean": -1.0, "minimum": -1.2, "maximum": -0.8},
                    "projection_kl": {
                        "mean": 0.2, "minimum": 0.1, "maximum": 0.3},
                },
            },
        }
        (output / "result.json").write_text(json.dumps(result))

    def test_requires_exact_inventory_and_aggregates_pass(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, 11)
            self._write(root, 13)
            result = aggregate_v23_projection_feasibility(
                root, specs=SPECS, train_seeds=[11, 13])

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["checkpoint_count"], 2)
        self.assertEqual(result["minibatch_count"], 512)

    def test_preserves_substantive_no_pass(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, 11, passes=False)
            result = aggregate_v23_projection_feasibility(
                root, specs=SPECS, train_seeds=[11])

        self.assertEqual(result["status"], "no_pass")
        self.assertFalse(result["v23_training_projection_supported"])

    def test_rejects_missing_checkpoint(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "missing V23"):
                aggregate_v23_projection_feasibility(
                    tmp, specs=SPECS, train_seeds=[11])


if __name__ == "__main__":
    unittest.main()
