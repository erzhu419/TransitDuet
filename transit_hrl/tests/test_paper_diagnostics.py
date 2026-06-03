import tempfile
import unittest
import json
from pathlib import Path

from freq_hrl.experiments.paper_diagnostics import (
    build_claim_matrix,
    build_statistical_checks,
    write_report,
)
from freq_hrl.experiments.statistics import (
    claim_status,
    noninferiority_status,
    paired_delta_stats,
    sign_test_p_value,
)


class PaperDiagnosticsTest(unittest.TestCase):
    def test_claim_matrix_builds_with_missing_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            claims = build_claim_matrix(root / "results", root / "transit")
            self.assertGreaterEqual(len(claims), 1)
            self.assertIn("claim", claims[0])
            checks = build_statistical_checks(root / "results")
            write_report(root / "report.md", claims, checks)
            self.assertTrue((root / "report.md").exists())

    def test_paired_statistics_capture_direction(self):
        rows = []
        for seed in [1, 2, 3, 4]:
            rows.append({"variant": "base", "seed": seed, "wait": 5.0 + seed})
            rows.append({"variant": "freq", "seed": seed, "wait": 4.0 + seed})
        stats = paired_delta_stats(
            rows,
            variant_key="variant",
            pair_keys=("seed",),
            metric="wait",
            treatment="freq",
            control="base",
            lower_is_better=True,
            n_boot=100,
            seed=7,
        )
        self.assertEqual(stats["n_common"], 4)
        self.assertAlmostEqual(stats["delta_mean"], -1.0)
        self.assertEqual(stats["win_rate"], 1.0)
        self.assertIn(claim_status(stats, min_pairs=4), {"supported", "positive_mixed"})
        self.assertLess(sign_test_p_value([1.0, 1.0, 1.0, 1.0]), 0.2)
        self.assertLess(sign_test_p_value([1.0] * 1024), 1e-250)
        self.assertEqual(sign_test_p_value([1.0] * 512 + [-1.0] * 512), 1.0)

    def test_noninferiority_status_uses_loss_margin(self):
        stats = {
            "n_common": 5,
            "improvement_mean": -0.002,
            "improvement_ci95_low": -0.004,
            "improvement_ci95_high": 0.001,
        }
        self.assertEqual(noninferiority_status(stats, max_loss=0.005, min_pairs=5), "supported")
        self.assertEqual(noninferiority_status(stats, max_loss=0.001, min_pairs=5), "inconclusive")

    def test_real_demand_control_rows_enter_statistical_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "results" / "transit_real_demand_control"
            out.mkdir(parents=True)
            rows = []
            for source in ("afc", "apc"):
                for seed in (1, 2, 3):
                    rows.append({
                        "source": source,
                        "seed": seed,
                        "variant": "base_real_ema",
                        "control_objective": -10.0,
                        "reward_mean": -8.0,
                        "wait_proxy": 7.0,
                        "LowerLFDrift": 2.0,
                        "RawLowerLFDriftAbs": 2.5,
                    })
                    rows.append({
                        "source": source,
                        "seed": seed,
                        "variant": "full_real_freqhrl",
                        "control_objective": -8.0,
                        "reward_mean": -6.0,
                        "wait_proxy": 5.0,
                        "LowerLFDrift": 1.5,
                        "RawLowerLFDriftAbs": 1.8,
                    })
            (out / "summary.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
            checks = {
                row["check"]: row
                for row in build_statistical_checks(root / "results")
            }
            self.assertEqual(
                checks["transit_real_demand_control_objective_vs_base"]["status"],
                "supported",
            )
            self.assertLess(
                checks["transit_real_demand_control_wait_vs_base"]["delta_mean"],
                0.0,
            )

    def test_native_real_demand_and_l2_matching_enter_statistical_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native_out = root / "results" / "transit_native_real_demand_control"
            native_out.mkdir(parents=True)
            rows = []
            for source in ("afc", "apc"):
                for seed in (1, 2, 3):
                    rows.append({
                        "source": source,
                        "seed": seed,
                        "variant": "native_real_interval",
                        "control_score": -20.0,
                        "ep_reward": -100.0,
                        "avg_wait_min": 5.0,
                        "native_avg_board_wait_min": 4.0,
                        "native_alighted_pax": 50.0,
                        "native_avg_onboard_load": 0.8,
                    })
                    rows.append({
                        "source": source,
                        "seed": seed,
                        "variant": "native_real_freqhrl",
                        "control_score": -12.0,
                        "ep_reward": -90.0,
                        "avg_wait_min": 4.0,
                        "native_avg_board_wait_min": 3.0,
                        "native_alighted_pax": 55.0,
                        "native_avg_onboard_load": 0.6,
                    })
            (native_out / "summary.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")

            match_out = root / "results" / "trading_order_book_matching_validation"
            match_out.mkdir(parents=True)
            match_check = {
                "check": "state_space_vs_ema_sharpe",
                "claim": "L2 order-book matching encoder paired check",
                "status": "supported",
                "metric": "sharpe",
                "treatment": "state_space",
                "control": "ema",
                "direction": "increase",
                "n_common": 3,
                "delta_mean": 1.0,
                "delta_ci95_low": 0.4,
                "delta_ci95_high": 1.4,
                "improvement_mean": 1.0,
                "improvement_ci95_low": 0.4,
                "improvement_ci95_high": 1.4,
                "win_rate": 1.0,
                "sign_p_value": 0.125,
            }
            (match_out / "summary.json").write_text(
                json.dumps({"paired_checks": [match_check]}),
                encoding="utf-8",
            )
            l3_out = root / "results" / "trading_order_book_l3_replay_validation"
            l3_out.mkdir(parents=True)
            l3_check = {
                "check": "adaptive_wavelet_vs_ema_fill_rate",
                "claim": "L3 order-event replay encoder paired check",
                "status": "supported",
                "metric": "fill_rate",
                "treatment": "adaptive_wavelet",
                "control": "ema",
                "direction": "increase",
                "n_common": 3,
                "delta_mean": 0.1,
                "delta_ci95_low": 0.05,
                "delta_ci95_high": 0.2,
                "improvement_mean": 0.1,
                "improvement_ci95_low": 0.05,
                "improvement_ci95_high": 0.2,
                "win_rate": 1.0,
                "sign_p_value": 0.125,
            }
            (l3_out / "summary.json").write_text(
                json.dumps({"paired_checks": [l3_check]}),
                encoding="utf-8",
            )

            checks = {
                row["check"]: row
                for row in build_statistical_checks(root / "results")
            }
            self.assertEqual(
                checks["transit_native_real_demand_control_score_vs_interval"]["status"],
                "supported",
            )
            self.assertLess(
                checks["transit_native_real_demand_board_wait_vs_interval"]["delta_mean"],
                0.0,
            )
            self.assertEqual(
                checks["order_book_matching_state_space_vs_ema_sharpe"]["status"],
                "supported",
            )
            self.assertEqual(
                checks["order_book_l3_adaptive_wavelet_vs_ema_fill_rate"]["status"],
                "supported",
            )

    def test_expanded_native_promotion_preferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_out = root / "results" / "transit_native_promotion_replan"
            expanded_out = root / "results" / "transit_native_promotion_replan_expanded"
            learned_waitaware_out = root / "results" / "transit_native_learned_wait_replan_same070_082_cap2_512seed"
            waitaware_out = root / "results" / "transit_native_wait_aware_replan_fair"
            old_out.mkdir(parents=True)
            expanded_out.mkdir(parents=True)
            learned_waitaware_out.mkdir(parents=True)
            waitaware_out.mkdir(parents=True)
            old_rows = [
                {"source": "old", "seed": 1, "variant": "interval_only", "ep_reward": 10.0, "avg_wait_min": 5.0, "score": 0.0, "upper_plan_decisions": 0.0, "shared_ppo_gate_replans": 0.0},
                {"source": "old", "seed": 1, "variant": "native_learned_gate", "ep_reward": 1.0, "avg_wait_min": 5.0, "score": 0.0, "upper_plan_decisions": 0.0, "shared_ppo_gate_replans": 0.0},
            ]
            expanded_rows = []
            for seed in range(1, 6):
                expanded_rows.append({
                    "source": "expanded",
                    "seed": seed,
                    "variant": "interval_only",
                    "ep_reward": 10.0,
                    "avg_wait_min": 5.0,
                    "score": 0.0,
                    "upper_plan_decisions": 0.0,
                    "shared_ppo_gate_replans": 0.0,
                })
                expanded_rows.append({
                    "source": "expanded",
                    "seed": seed,
                    "variant": "native_learned_gate",
                    "ep_reward": 20.0,
                    "avg_wait_min": 4.0,
                    "score": 1.0,
                    "upper_plan_decisions": 1.0,
                    "shared_ppo_gate_replans": 1.0,
                })
            waitaware_rows = []
            for seed in range(1, 6):
                waitaware_rows.append({
                    "source": "waitaware",
                    "seed": seed,
                    "variant": "interval_only",
                    "ep_reward": 10.0,
                    "avg_wait_min": 5.0,
                    "score": 0.0,
                    "shared_ppo_gate_replans": 0.0,
                    "shared_ppo_wait_replan_count": 0.0,
                    "shared_ppo_wait_replan_shift_abs_mean_s": 0.0,
                    "upper_plan_target_mean": 360.0,
                    "terminal_launch_shift_mean": 0.0,
                })
                waitaware_rows.append({
                    "source": "waitaware",
                    "seed": seed,
                    "variant": "native_wait_aware_replan",
                    "ep_reward": 20.0,
                    "avg_wait_min": 4.0,
                    "score": 1.0,
                    "shared_ppo_gate_replans": 1.0,
                    "shared_ppo_wait_replan_count": 1.0,
                    "shared_ppo_wait_replan_shift_abs_mean_s": 12.0,
                    "upper_plan_target_mean": 348.0,
                    "terminal_launch_shift_mean": -8.0,
                })
            learned_waitaware_rows = []
            for seed in range(1, 7):
                learned_waitaware_rows.append({
                    "source": "learned_waitaware",
                    "seed": seed,
                    "variant": "interval_only",
                    "ep_reward": 10.0,
                    "avg_wait_min": 5.0,
                    "score": 0.0,
                    "shared_ppo_gate_replans": 0.0,
                    "shared_ppo_wait_replan_count": 0.0,
                    "shared_ppo_wait_replan_shift_abs_mean_s": 0.0,
                    "upper_plan_target_mean": 360.0,
                    "terminal_launch_shift_mean": 0.0,
                })
                learned_waitaware_rows.append({
                    "source": "learned_waitaware",
                    "seed": seed,
                    "variant": "native_wait_aware_replan",
                    "ep_reward": 21.0,
                    "avg_wait_min": 3.9,
                    "score": 1.2,
                    "shared_ppo_gate_replans": 1.0,
                    "shared_ppo_wait_replan_count": 1.0,
                    "shared_ppo_wait_replan_shift_abs_mean_s": 10.0,
                    "upper_plan_target_mean": 346.0,
                    "terminal_launch_shift_mean": -9.0,
                })
            (old_out / "summary.json").write_text(json.dumps({"rows": old_rows}), encoding="utf-8")
            (expanded_out / "summary.json").write_text(json.dumps({"rows": expanded_rows}), encoding="utf-8")
            (learned_waitaware_out / "summary.json").write_text(
                json.dumps({"rows": learned_waitaware_rows}), encoding="utf-8")
            (waitaware_out / "summary.json").write_text(json.dumps({"rows": waitaware_rows}), encoding="utf-8")

            checks = {
                row["check"]: row
                for row in build_statistical_checks(root / "results")
            }
            learned_reward = checks["transit_native_learned_gate_reward_vs_interval"]
            self.assertEqual(learned_reward["n_common"], 5)
            self.assertEqual(learned_reward["status"], "supported")
            waitaware_shift = checks["transit_native_wait_aware_replan_shift_vs_interval"]
            self.assertEqual(waitaware_shift["n_common"], 6)
            self.assertEqual(waitaware_shift["status"], "supported")
            waitaware_target = checks["transit_native_wait_aware_replan_target_vs_interval"]
            self.assertEqual(waitaware_target["status"], "supported")


if __name__ == "__main__":
    unittest.main()
