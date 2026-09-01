import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.audit_protocol_v6_passenger_holding_screen import (
    CANDIDATE_SPECS,
    CANDIDATES,
    CONFIGS,
    CURRENT_MAIN,
    EVAL_SEEDS,
    HARD_MAIN,
    MATCHED_CONTEXT,
    PASSENGER_CONFIGS,
    PASSENGER_SCALAR_SPECS,
    REFERENCE,
    SAME_ENTROPY,
    TRAIN_SEEDS,
    V13_ANCHOR,
    V13_ZERO_HOLD_ADVANTAGE,
    evaluate_passenger_holding_screen,
)


class ProtocolV6PassengerHoldingScreenTest(unittest.TestCase):
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
            V13_ZERO_HOLD_ADVANTAGE: (20.1, 0.203, 10.5, 112.0, 13.0),
        }
        for config, _ in PASSENGER_SCALAR_SPECS:
            baselines[config] = (20.0, 0.205, 10.0, 110.0, 12.0)
        for config in CANDIDATES:
            baselines[config] = (19.9, 0.202, 9.5, 109.0, 11.0)
        budgets = {
            config: budget for config, budget in PASSENGER_SCALAR_SPECS}
        budgets.update({
            config: budget for config, _, budget in CANDIDATE_SPECS})

        rows = []
        for config in CONFIGS:
            values = baselines[config]
            passenger_enabled = config in PASSENGER_CONFIGS
            regularity_enabled = (
                config in PASSENGER_CONFIGS
                or config in {V13_ANCHOR, V13_ZERO_HOLD_ADVANTAGE})
            critic = (
                "zero_hold_advantage"
                if config == V13_ZERO_HOLD_ADVANTAGE or config in CANDIDATES
                else "continuous_action")
            budget = budgets.get(config, 0.0)
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
                        "lower_discrete_critic": critic,
                        "lower_policy_frozen": 1.0,
                        "lower_critic_frozen": 1.0,
                        "upper_policy_frozen": 1.0,
                        "lower_causal_guard_adjustment_mean_s": 0.0,
                        "lower_regularity_policy_enabled": float(
                            regularity_enabled),
                        "lower_regularity_policy_mode": (
                            "analytic_two_sided_zero_hold_regret_dual_v2"
                            if regularity_enabled else "disabled"),
                        "lower_regularity_policy_constraint_cost_mode": (
                            "zero_hold_regret_v2"
                            if regularity_enabled else "disabled"),
                        "lower_regularity_policy_constraint_scale_mode": (
                            "cost_limit_ratio_v1"
                            if regularity_enabled else "raw_cost_v1"),
                        "lower_regularity_policy_initial_lambda": (
                            0.01 if regularity_enabled else 0.0),
                        "lower_regularity_policy_cost_limit": (
                            0.00025 if regularity_enabled else 0.0),
                        "lower_regularity_policy_scaled_limit": (
                            1.0 if regularity_enabled else 0.0),
                        "lower_regularity_policy_action_regret_mean": (
                            0.0001 if regularity_enabled else 0.0),
                        "lower_regularity_policy_evidence_valid_mean": (
                            0.75 if regularity_enabled else 0.0),
                        "lower_regularity_passenger_holding_enabled": float(
                            passenger_enabled),
                        "lower_regularity_passenger_holding_mode": (
                            "causal_apc_person_delay_dual_v1"
                            if passenger_enabled else "disabled"),
                        "lower_regularity_passenger_constraint_scale_mode": (
                            "cost_limit_ratio_v1"
                            if passenger_enabled else "raw_cost_v1"),
                        "lower_regularity_passenger_initial_lambda": (
                            0.01 if passenger_enabled else 0.0),
                        "lower_regularity_passenger_cost_limit": budget,
                        "lower_regularity_passenger_scaled_limit": (
                            1.0 if passenger_enabled else 0.0),
                        "lower_regularity_passenger_expected_cost_mean": (
                            0.8 * budget if passenger_enabled else 0.0),
                        "lower_regularity_passenger_selected_cost_mean": (
                            0.5 * budget if passenger_enabled else 0.0),
                        "lower_regularity_passenger_load_mean": (
                            0.5 if passenger_enabled else 0.0),
                        "lower_regularity_passenger_lambda": (
                            0.2 if passenger_enabled else 0.0),
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

    def test_selects_weakest_passing_passenger_budget_first(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            result = evaluate_passenger_holding_screen(root)

        self.assertEqual(
            result["status"], "exploratory_candidate_selected")
        self.assertEqual(
            result["selected_for_confirmation"], CANDIDATES[-1])
        self.assertTrue(all(
            row["passes"] for row in result["candidate_results"]))
        self.assertTrue(all(
            row["passes"] for row in result["mechanism_control_results"]))

    def test_rejects_candidate_that_exceeds_expected_passenger_budget(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            frame = pd.read_csv(root / "frozen_per_eval.csv")
            for config, _, budget in CANDIDATE_SPECS:
                frame.loc[
                    frame["config"] == config,
                    "lower_regularity_passenger_expected_cost_mean",
                ] = budget + 0.001
            frame.to_csv(root / "frozen_per_eval.csv", index=False)
            result = evaluate_passenger_holding_screen(root)

        self.assertEqual(result["status"], "no_pass")
        self.assertTrue(all(
            not row["mechanism_checks"][
                "expected_passenger_budget_satisfied_every_rollout"]
            for row in result["candidate_results"]
        ))

    def test_rejects_missing_passive_apc_telemetry(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            frame = pd.read_csv(root / "frozen_per_eval.csv")
            frame.loc[
                frame["config"].isin(CANDIDATES),
                "lower_regularity_passenger_load_mean",
            ] = 0.0
            frame.to_csv(root / "frozen_per_eval.csv", index=False)
            result = evaluate_passenger_holding_screen(root)

        self.assertEqual(result["status"], "no_pass")
        self.assertTrue(all(
            not row["mechanism_checks"][
                "passive_passenger_telemetry_active"]
            for row in result["candidate_results"]
        ))


if __name__ == "__main__":
    unittest.main()
