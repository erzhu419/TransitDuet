import unittest
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import pandas as pd

from scripts.run_freqduet_protocol_v6_checkpoint_trajectory import (
    METRICS,
    EVAL_SEEDS,
    assert_same_deployment,
    checkpoint_change,
    evaluate_checkpoint,
    paired_trajectory,
    runtime_config,
    validate_grid,
)


class CheckpointTrajectoryTest(unittest.TestCase):
    def test_precomputed_checkpoint_is_reused_without_running_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "logs/current_seed11/frozen_trajectory/ep39"
            output.mkdir(parents=True)
            rows = pd.DataFrame([
                dict(eval_seed=seed, checkpoint_ep=39, scenario_tape_id=f"tape{seed}",
                     protocol_version="freqduet-eval-v6", policy_digest="frozen",
                     **{metric: 0.2 for metric in METRICS}) for seed in EVAL_SEEDS])
            rows.to_csv(output / "evaluation.csv", index=False)
            runner, load_config = Mock(), Mock()
            evaluate_checkpoint(Path("/unused"), root, "current", 11, 39,
                                runtime=(None, runner, load_config, None))
            runner.assert_not_called()
            load_config.assert_not_called()
            record = pd.read_csv(root / "records/current_seed11_ep39.csv")
            self.assertEqual(list(record.eval_seed), list(EVAL_SEEDS))
            self.assertTrue(record.train_seed.eq(11).all())

    def test_checkpoint_worker_keeps_original_evaluation_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, load_config, torch = Mock(), Mock(return_value={}), Mock()
            runner.return_value.upper_warmup = 30
            rows = pd.DataFrame([
                dict(eval_seed=seed, checkpoint_ep=39, scenario_tape_id=f"tape{seed}",
                     **{metric: 0.2 for metric in METRICS}) for seed in EVAL_SEEDS])
            with patch("scripts.run_freqduet_protocol_v6_checkpoint_trajectory.seed_runtime") as seed_fn, \
                    patch("scripts.run_freqduet_protocol_v6_checkpoint_trajectory.evaluation_rows",
                          return_value=rows):
                evaluate_checkpoint(Path("/model"), root, "current", 11, 39,
                                    runtime=(None, runner, load_config, torch))
            seed_fn.assert_called_once_with(torch, 11)
            runner.return_value.load_checkpoint.assert_called_once_with(
                root / "logs/current_seed11/checkpoints", ep=39, require_deployment_state=True)
            runner.return_value.evaluate.assert_called_once_with(
                EVAL_SEEDS, output_dir=root / "logs/current_seed11/frozen_trajectory/ep39",
                policy_ep=39)
            runner.return_value.train.assert_not_called()

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
        left = {"config_fingerprint_sha256": "intermediate", "belief": [0.25, 0.75],
                "fixed_selector_prev_diag": {
                    "config_fingerprint_sha256": "intermediate", "headway_cv": 0.2,
                    "wall_env_s": 100.0, "wall_train_s": 3.0}}
        right = {"config_fingerprint_sha256": "final-only", "belief": [0.25, 0.75],
                 "fixed_selector_prev_diag": {
                     "config_fingerprint_sha256": "final-only", "headway_cv": 0.2,
                     "wall_env_s": 105.0, "wall_train_s": 4.0}}
        assert_same_deployment(left, right, ("intermediate", "final-only"))
        with self.assertRaisesRegex(AssertionError, "instrumentation configuration"):
            assert_same_deployment(left, right, ("unexpected", "final-only"))
        right["fixed_selector_prev_diag"]["config_fingerprint_sha256"] = "wrong-config"
        with self.assertRaisesRegex(AssertionError, "cached diagnostics"):
            assert_same_deployment(left, right, ("intermediate", "final-only"))

    def test_cached_operational_outcome_difference_still_fails(self):
        left = {"config_fingerprint_sha256": "intermediate", "fixed_selector_prev_diag": {
            "config_fingerprint_sha256": "intermediate", "headway_cv": 0.2}}
        right = {"config_fingerprint_sha256": "final-only", "fixed_selector_prev_diag": {
            "config_fingerprint_sha256": "final-only", "headway_cv": 0.3}}
        with self.assertRaisesRegex(AssertionError, "fixed_selector_prev_diag.headway_cv"):
            assert_same_deployment(left, right, ("intermediate", "final-only"))

    def test_deployment_difference_still_fails(self):
        left = {"config_fingerprint_sha256": "intermediate", "belief": [0.25, 0.75]}
        right = {"config_fingerprint_sha256": "final-only", "belief": [0.5, 0.5]}
        with self.assertRaisesRegex(AssertionError, "checkpoint value differs"):
            assert_same_deployment(left, right, ("intermediate", "final-only"))


if __name__ == "__main__":
    unittest.main()
