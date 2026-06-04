import json
import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.transit.merge_native_promotion_shards import merge_native_promotion_shards
from freq_hrl.experiments.transit.native_promotion_replan_validation import (
    COMMON_OVERRIDES,
    VARIANTS,
    apply_persistent_stress_preset,
    paired_checks,
    stress_subset_checks,
    write_outputs,
)


class NativePromotionReplanValidationTest(unittest.TestCase):
    def test_persistent_stress_preset_installs_wait_safe_gate(self):
        old_common = json.loads(json.dumps(COMMON_OVERRIDES))
        old_variants = json.loads(json.dumps(VARIANTS))
        try:
            apply_persistent_stress_preset()
            self.assertEqual(set(VARIANTS), {"interval_only", "native_wait_aware_replan"})
            wait_aware = VARIANTS["native_wait_aware_replan"]
            self.assertEqual(wait_aware["_promotion_replan_same_hold_max"], 0.25)
            self.assertEqual(wait_aware["_promotion_replan_same_wait_min"], 0.812)
            self.assertEqual(wait_aware["_promotion_replan_same_wait_max"], 0.85)
            self.assertEqual(wait_aware["_promotion_replan_gap_risk_cap_start"], 0.05)
            self.assertEqual(wait_aware["_promotion_replan_gap_risk_cap_full"], 0.20)
            self.assertTrue(wait_aware["_promotion_replan_terminal_early_relax"])
            self.assertEqual(
                COMMON_OVERRIDES["upper"]["timetable_planner"]["terminal_shift_min_s"],
                0.0,
            )
        finally:
            COMMON_OVERRIDES.clear()
            COMMON_OVERRIDES.update(old_common)
            VARIANTS.clear()
            VARIANTS.update(old_variants)

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
                    "variant_private_overrides": {
                        "native_promotion_replan": {
                            "promotion_replan_max_shift_s": 2.0,
                        },
                    },
                }, f)
            merged = merge_native_promotion_shards(
                input_dirs=[shard],
                output_dir=Path(tmp) / "merged",
                min_pairs=2,
            )
            self.assertEqual(len(merged["rows"]), 4)
            self.assertTrue(merged["paired_checks"])
            self.assertEqual(
                merged["variant_private_overrides"]["native_promotion_replan"]["promotion_replan_max_shift_s"],
                2.0,
            )
            self.assertTrue((Path(tmp) / "merged" / "summary.json").exists())

    def test_write_outputs_accepts_variant_specific_config_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            payload = {
                "rows": [
                    {
                        "seed": 1,
                        "variant": "interval_only",
                        "ep_reward": -10.0,
                        "avg_wait_min": 5.0,
                        "headway_cv": 0.2,
                        "score": -5.0,
                        "upper_plan_decisions": 3.0,
                        "shared_ppo_gate_replans": 0.0,
                        "shared_ppo_gate_value_mean": 0.0,
                        "freq_promotion_strength": 0.0,
                        "shared_ppo_lower_samples": 0.0,
                    },
                    {
                        "seed": 1,
                        "variant": "native_wait_aware_replan",
                        "ep_reward": -9.0,
                        "avg_wait_min": 4.8,
                        "headway_cv": 0.2,
                        "score": -4.8,
                        "upper_plan_decisions": 3.0,
                        "shared_ppo_gate_replans": 1.0,
                        "shared_ppo_gate_value_mean": 0.9,
                        "freq_promotion_strength": 1.0,
                        "shared_ppo_lower_samples": 0.0,
                        "cfg_promotion_replan_same_wait_max": 0.82,
                    },
                ],
                "paired_checks": [],
            }
            write_outputs(out, payload)
            csv_text = (out / "summary.csv").read_text(encoding="utf-8")
            self.assertIn("cfg_promotion_replan_same_wait_max", csv_text.splitlines()[0])

    def test_stress_subset_checks_use_control_side_selector(self):
        rows = []
        for seed, control_strength in [(1, 1.0), (2, 0.0)]:
            rows.extend([
                {
                    "seed": seed,
                    "variant": "interval_only",
                    "ep_reward": -10.0,
                    "avg_wait_min": 5.0,
                    "score": -5.0,
                    "shared_ppo_wait_replan_count": 0.0,
                    "freq_promotion_strength": control_strength,
                    "headway_cv": 0.1,
                    "upper_plan_target_mean": 340.0,
                },
                {
                    "seed": seed,
                    "variant": "native_wait_aware_replan",
                    "ep_reward": -8.0,
                    "avg_wait_min": 4.8,
                    "score": -4.8,
                    "shared_ppo_wait_replan_count": 1.0,
                    "freq_promotion_strength": 1.0,
                    "headway_cv": 0.1,
                    "upper_plan_target_mean": 340.0,
                },
            ])
        checks = stress_subset_checks(rows, min_pairs=1)
        reward = next(
            row for row in checks
            if row["check"] == "native_wait_aware_replan_control_promotion_strength_ge1_vs_interval_ep_reward"
        )
        self.assertEqual(reward["n_common"], 1)
        self.assertEqual(reward["delta_mean"], 2.0)


if __name__ == "__main__":
    unittest.main()
