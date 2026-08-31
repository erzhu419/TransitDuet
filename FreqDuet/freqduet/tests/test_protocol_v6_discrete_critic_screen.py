import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.audit_protocol_v6_discrete_critic_screen import (
    CANDIDATE_SPECS,
    CONFIGS,
    CURRENT_MAIN,
    EVAL_SEEDS,
    HARD_MAIN,
    MATCHED_CONTEXT,
    REFERENCE,
    SAME_ENTROPY,
    TRAIN_SEEDS,
    V13_ANCHOR,
    V14_ANCHOR,
    V14_ZERO_HOLD_ADVANTAGE,
    evaluate_discrete_critic_screen,
)


class ProtocolV6DiscreteCriticScreenTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> None:
        manifest = {
            "strict_complete": True,
            "run_manifests_verified": True,
            "common_random_numbers_verified": True,
            "stage": "exploratory",
            "independent_confirmation": False,
            "configs": CONFIGS,
            "train_seeds": TRAIN_SEEDS,
            "eval_seeds": EVAL_SEEDS,
            "train_episodes": 40,
            "checkpoint_ep": 39,
            "reference": V13_ANCHOR,
            "run_git_provenance": {"tracked_dirty": False},
            "expected_rollouts": (
                len(CONFIGS) * len(TRAIN_SEEDS) * len(EVAL_SEEDS)),
        }
        (root / "matrix_manifest.json").write_text(json.dumps(manifest))
        (root / "frozen_summary.csv").write_text("config\n")

        baselines = {
            HARD_MAIN: (21.0, 0.28, 20.0, 200.0, 30.0),
            REFERENCE: (21.0, 0.25, 12.0, 120.0, 15.0),
            MATCHED_CONTEXT: (20.5, 0.23, 11.0, 115.0, 14.0),
            CURRENT_MAIN: (20.5, 0.202, 11.0, 115.0, 14.0),
            SAME_ENTROPY: (20.5, 0.202, 11.0, 115.0, 14.0),
            V13_ANCHOR: (20.0, 0.205, 10.0, 110.0, 12.0),
            V14_ANCHOR: (19.8, 0.200, 9.0, 105.0, 11.0),
        }
        candidates = {
            name: (
                baselines[anchor][0] - 0.10,
                baselines[anchor][1] - 0.002,
                baselines[anchor][2] - 0.5,
                baselines[anchor][3] - 1.0,
                baselines[anchor][4] - 1.0,
            )
            for name, anchor, _, _ in CANDIDATE_SPECS
        }
        rows = []
        for config in CONFIGS:
            values = (
                candidates[config]
                if config in candidates else baselines[config])
            spec = next(
                (item for item in CANDIDATE_SPECS if item[0] == config),
                None,
            )
            expected_critic = spec[2] if spec else "continuous_action"
            capacity_gain = bool(spec and spec[3])
            for train_seed in TRAIN_SEEDS:
                for eval_seed in EVAL_SEEDS:
                    rows.append({
                        "config": config,
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "restricted_total_journey_horizon_min": values[0],
                        "headway_cv": values[1],
                        "lower_action_mean": values[2],
                        "holding_vehicle_seconds": values[3],
                        "fleet_denied_dispatch_events": values[4],
                        "lower_discrete_critic": expected_critic,
                        "lower_policy_frozen": 1.0,
                        "lower_critic_frozen": 1.0,
                        "upper_policy_frozen": 1.0,
                        "lower_causal_guard_adjustment_mean_s": 0.0,
                        "lower_regularity_policy_enabled": (
                            1.0 if spec else 0.0),
                        "lower_regularity_policy_mode": (
                            "analytic_two_sided_capacity_gain_regret_dual_v3"
                            if capacity_gain else
                            "analytic_two_sided_zero_hold_regret_dual_v2"
                            if spec else "disabled"),
                        "lower_regularity_policy_constraint_cost_mode": (
                            "zero_hold_regret_v2" if spec else "disabled"),
                        "lower_regularity_policy_constraint_scale_mode": (
                            "cost_limit_ratio_v1" if spec else "raw_cost_v1"),
                        "lower_regularity_policy_initial_lambda": (
                            0.01 if spec else 0.0),
                        "lower_regularity_policy_cost_limit": (
                            0.00025 if spec else 0.0),
                        "lower_regularity_policy_scaled_limit": (
                            1.0 if spec else 0.0),
                        "lower_regularity_policy_action_regret_mean": (
                            0.0001 if spec else 0.0),
                        "lower_regularity_policy_evidence_valid_mean": (
                            0.75 if spec else 0.0),
                        "lower_regularity_policy_capacity_gain_enabled": (
                            1.0 if capacity_gain else 0.0),
                        "lower_regularity_policy_capacity_gain_mode": (
                            "positive_zero_hold_gain_v1"
                            if capacity_gain else "disabled"),
                        "lower_regularity_policy_capacity_gain_weight": (
                            0.02 if capacity_gain else 0.0),
                        "lower_regularity_policy_capacity_gain_scale": (
                            0.002 if capacity_gain else 1.0),
                        "lower_regularity_policy_capacity_exponent": 1.0,
                        "lower_regularity_policy_capacity_gain_mean": (
                            0.001 if capacity_gain else 0.0),
                        "lower_regularity_policy_scaled_capacity_gain_mean": (
                            0.5 if capacity_gain else 0.0),
                        "lower_regularity_policy_capacity_gain_bonus": (
                            0.01 if capacity_gain else 0.0),
                        "lower_regularity_policy_capacity_gate_mean": (
                            0.8 if capacity_gain else 0.0),
                    })
        pd.DataFrame(rows).to_csv(root / "frozen_per_eval.csv", index=False)
        pd.DataFrame([
            {
                "candidate": config,
                "reference": V13_ANCHOR,
                "n_pairs": len(TRAIN_SEEDS) * len(EVAL_SEEDS),
            }
            for config in CONFIGS if config != V13_ANCHOR
        ]).to_csv(root / "frozen_paired_deltas.csv", index=False)

    def test_selects_zero_hold_advantage_capacity_candidate_first(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            result = evaluate_discrete_critic_screen(root)

        self.assertEqual(
            result["status"], "exploratory_candidate_selected")
        self.assertEqual(
            result["selected_for_confirmation"],
            V14_ZERO_HOLD_ADVANTAGE,
        )
        self.assertTrue(all(
            row["passes"] for row in result["candidate_results"]))

    def test_rejects_wrong_critic_telemetry(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            per_eval = pd.read_csv(root / "frozen_per_eval.csv")
            per_eval.loc[
                per_eval["config"] == V14_ZERO_HOLD_ADVANTAGE,
                "lower_discrete_critic",
            ] = "continuous_action"
            per_eval.to_csv(root / "frozen_per_eval.csv", index=False)
            result = evaluate_discrete_critic_screen(root)

        candidate = next(
            row for row in result["candidate_results"]
            if row["config"] == V14_ZERO_HOLD_ADVANTAGE)
        self.assertFalse(candidate["passes"])
        self.assertFalse(
            candidate["mechanism_checks"]["candidate_critic_locked"])

    def test_rejects_journey_cv_tradeoff(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            per_eval = pd.read_csv(root / "frozen_per_eval.csv")
            candidate_names = [item[0] for item in CANDIDATE_SPECS]
            per_eval.loc[
                per_eval["config"].isin(candidate_names),
                "headway_cv",
            ] += 0.01
            per_eval.to_csv(root / "frozen_per_eval.csv", index=False)
            result = evaluate_discrete_critic_screen(root)

        self.assertEqual(result["status"], "no_pass")
        self.assertIsNone(result["selected_for_confirmation"])


if __name__ == "__main__":
    unittest.main()
