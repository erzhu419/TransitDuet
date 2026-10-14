import unittest
from pathlib import Path

import pandas as pd

from scripts.run_freqduet_protocol_v6_checkpoint_trajectory import (
    METRICS,
    assert_same_deployment,
    checkpoint_change,
    paired_trajectory,
    runtime_config,
    validate_grid,
)


class CheckpointTrajectoryTest(unittest.TestCase):
    def frame(self):
        rows = []
        for config in ("current", "reference"):
            for seed in (11, 13):
                for evaluation in (21, 23):
                    for checkpoint in (39, 199):
                        value = 0.4
                        if config == "current":
                            value -= 0.03 if checkpoint == 39 else 0.01
                        rows.append(dict(config=config, train_seed=seed,
                                         eval_seed=evaluation, checkpoint_ep=checkpoint,
                                         scenario_tape_id=f"scenario{evaluation}",
                                         **{metric: value for metric in METRICS}))
        return pd.DataFrame(rows)

    def test_change_pairs_same_seed_and_scenario(self):
        frame = self.frame().sample(frac=1, random_state=5)
        paired = paired_trajectory(frame, "current", "reference")
        change = checkpoint_change(paired, 39, 199)
        self.assertEqual(len(change), 4)
        self.assertAlmostEqual(change.headway_cv.mean(), 0.02)

    def test_missing_checkpoint_cannot_be_aggregated(self):
        frame = self.frame().iloc[:-1]
        with self.assertRaisesRegex(ValueError, "grid is incomplete"):
            validate_grid(frame, ("current", "reference"), (11, 13), (21, 23), (39, 199))

    def test_unmatched_policy_rows_are_rejected(self):
        frame = self.frame().iloc[:-1]
        with self.assertRaisesRegex(ValueError, "unmatched policy rows"):
            paired_trajectory(frame, "current", "reference")

    def test_different_scenario_tapes_are_rejected(self):
        frame = self.frame()
        frame.loc[0, "scenario_tape_id"] = "another-demand"
        with self.assertRaisesRegex(ValueError, "scenario tapes differ"):
            validate_grid(frame, ("current", "reference"), (11, 13), (21, 23), (39, 199))

    def test_instrumentation_preserves_learning_configuration(self):
        config = {"lower": {"reward_weight": 2.0}, "training": {"updates": 100}}
        result = runtime_config(config, 11, Path("/tmp/trajectory"), 40)
        self.assertEqual(result["lower"], config["lower"])
        self.assertEqual(result["training"]["updates"], 100)
        self.assertEqual(result["training"]["checkpoint_freq"], 40)
        self.assertNotIn("checkpoint_freq", config["training"])

    def test_instrumentation_metadata_is_validated_separately(self):
        left = {"config_fingerprint_sha256": "intermediate", "belief": [0.25, 0.75]}
        right = {"config_fingerprint_sha256": "final-only", "belief": [0.25, 0.75]}
        assert_same_deployment(left, right, ("intermediate", "final-only"))
        with self.assertRaisesRegex(AssertionError, "instrumentation configuration"):
            assert_same_deployment(left, right, ("unexpected", "final-only"))

    def test_deployment_difference_still_fails(self):
        left = {"config_fingerprint_sha256": "intermediate", "belief": [0.25, 0.75]}
        right = {"config_fingerprint_sha256": "final-only", "belief": [0.5, 0.5]}
        with self.assertRaisesRegex(AssertionError, "checkpoint value differs"):
            assert_same_deployment(left, right, ("intermediate", "final-only"))


if __name__ == "__main__":
    unittest.main()
