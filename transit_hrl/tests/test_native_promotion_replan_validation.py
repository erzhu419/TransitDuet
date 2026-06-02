import json
import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.transit.merge_native_promotion_shards import merge_native_promotion_shards
from freq_hrl.experiments.transit.native_promotion_replan_validation import paired_checks


class NativePromotionReplanValidationTest(unittest.TestCase):
    def test_paired_checks_gate_native_reward_and_wait(self):
        rows = [
            {
                "seed": 1,
                "variant": "interval_only",
                "ep_reward": -10.0,
                "avg_wait_min": 5.0,
                "score": -6.0,
                "upper_plan_decisions": 3.0,
            },
            {
                "seed": 1,
                "variant": "native_promotion_replan",
                "ep_reward": -8.0,
                "avg_wait_min": 4.0,
                "score": -5.0,
                "upper_plan_decisions": 5.0,
            },
            {
                "seed": 2,
                "variant": "interval_only",
                "ep_reward": -9.0,
                "avg_wait_min": 4.5,
                "score": -5.5,
                "upper_plan_decisions": 4.0,
            },
            {
                "seed": 2,
                "variant": "native_promotion_replan",
                "ep_reward": -7.5,
                "avg_wait_min": 4.1,
                "score": -5.0,
                "upper_plan_decisions": 6.0,
            },
        ]
        checks = {row["metric"]: row for row in paired_checks(rows, min_pairs=2)}
        self.assertGreater(checks["ep_reward"]["delta_mean"], 0.0)
        self.assertLess(checks["avg_wait_min"]["delta_mean"], 0.0)
        self.assertEqual(checks["ep_reward"]["n_common"], 2)

    def test_paired_checks_include_wait_aware_replan_dispatch_metrics(self):
        rows = []
        for seed in (1, 2):
            rows.extend([
                {
                    "seed": seed,
                    "variant": "interval_only",
                    "ep_reward": -10.0,
                    "avg_wait_min": 5.0,
                    "score": -6.0,
                    "upper_plan_decisions": 3.0,
                    "shared_ppo_gate_replans": 0.0,
                    "shared_ppo_wait_replan_count": 0.0,
                    "shared_ppo_wait_replan_shift_abs_mean_s": 0.0,
                    "shared_ppo_wait_replan_shift_mean_s": 0.0,
                    "upper_plan_target_mean": 360.0,
                    "terminal_launch_shift_mean": 0.0,
                },
                {
                    "seed": seed,
                    "variant": "native_wait_aware_replan",
                    "ep_reward": -9.0,
                    "avg_wait_min": 4.7,
                    "score": -5.7,
                    "upper_plan_decisions": 3.0,
                    "shared_ppo_gate_replans": 1.0,
                    "shared_ppo_wait_replan_count": 1.0,
                    "shared_ppo_wait_replan_shift_abs_mean_s": 12.0,
                    "shared_ppo_wait_replan_shift_mean_s": -12.0,
                    "upper_plan_target_mean": 348.0,
                    "terminal_launch_shift_mean": -8.0,
                },
            ])
        checks = {
            row["metric"]: row
            for row in paired_checks(
                rows,
                min_pairs=2,
                treatment="native_wait_aware_replan",
            )
        }
        self.assertIn("terminal_launch_shift_mean", checks)
        self.assertLess(checks["terminal_launch_shift_mean"]["delta_mean"], 0.0)
        self.assertIn("upper_plan_target_mean", checks)
        self.assertLess(checks["upper_plan_target_mean"]["delta_mean"], 0.0)

    def test_merge_native_promotion_shards_recomputes_checks(self):
        rows = []
        for seed in (1, 2):
            rows.extend([
                {
                    "seed": seed,
                    "variant": "interval_only",
                    "status": "ok",
                    "ep_reward": -10.0,
                    "avg_wait_min": 5.0,
                    "headway_cv": 0.2,
                    "score": -5.0,
                    "upper_plan_decisions": 3.0,
                    "upper_plan_reuse_ratio": 0.5,
                    "freq_promotion_strength": 0.0,
                    "shared_ppo_lower_samples": 10.0,
                    "shared_ppo_gate_evaluations": 0.0,
                    "shared_ppo_gate_replans": 0.0,
                    "shared_ppo_gate_value_mean": 0.0,
                    "shared_ppo_loss": 0.0,
                },
                {
                    "seed": seed,
                    "variant": "native_promotion_replan",
                    "status": "ok",
                    "ep_reward": -8.0,
                    "avg_wait_min": 4.5,
                    "headway_cv": 0.2,
                    "score": -4.5,
                    "upper_plan_decisions": 5.0,
                    "upper_plan_reuse_ratio": 0.2,
                    "freq_promotion_strength": 1.0,
                    "shared_ppo_lower_samples": 10.0,
                    "shared_ppo_gate_evaluations": 0.0,
                    "shared_ppo_gate_replans": 0.0,
                    "shared_ppo_gate_value_mean": 0.0,
                    "shared_ppo_loss": 0.0,
                },
            ])
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "shard"
            shard.mkdir()
            with (shard / "summary.json").open("w", encoding="utf-8") as f:
                json.dump({
                    "rows": rows,
                    "payloads": {},
                    "episodes": 1,
                    "lower_hf_wait_action_gain_s": 45.0,
                    "offpolicy_replay_updates": 3,
                }, f)
            merged = merge_native_promotion_shards(
                input_dirs=[shard],
                output_dir=Path(tmp) / "merged",
                min_pairs=2,
            )
            self.assertEqual(len(merged["rows"]), 4)
            self.assertTrue(merged["paired_checks"])
            self.assertTrue((Path(tmp) / "merged" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
