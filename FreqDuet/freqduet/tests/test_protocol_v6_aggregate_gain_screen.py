import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.audit_protocol_v6_aggregate_gain_screen import (
    CONFIGS,
    CURRENT_MAIN,
    EVAL_SEEDS,
    FACTORIAL_CONFIGS,
    FACTORIAL_SPECS,
    REFERENCE,
    TRAIN_SEEDS,
    V13_ANCHOR,
    V13_ZERO_HOLD_ADVANTAGE,
    V20_QADV_B080,
    evaluate_aggregate_gain_screen,
)


class ProtocolV6AggregateGainScreenTest(unittest.TestCase):
    def _write_fixture(self, root: Path, *, fail_budgets: bool = False) -> None:
        expected_pairs = len(TRAIN_SEEDS) * len(EVAL_SEEDS)
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
            "expected_rollouts": len(CONFIGS) * expected_pairs,
        }
        (root / "matrix_manifest.json").write_text(json.dumps(manifest))
        (root / "frozen_summary.csv").write_text("config\n")

        baseline_values = {
            CURRENT_MAIN: (20.5, 0.202, 11.0, 115.0, 14.0),
            REFERENCE: (21.0, 0.250, 12.0, 120.0, 15.0),
            V13_ANCHOR: (20.0, 0.205, 10.0, 110.0, 12.0),
            V13_ZERO_HOLD_ADVANTAGE: (20.0, 0.203, 10.0, 110.0, 12.0),
            V20_QADV_B080: (19.65, 0.230, 8.0, 95.0, 10.0),
        }
        factor_lookup = {
            config: (allocation, dual, rho)
            for config, allocation, dual, rho, _ in FACTORIAL_SPECS
        }
        rows = []
        for config in CONFIGS:
            factor_enabled = config in factor_lookup
            journey, cv, action, holding, denied = baseline_values.get(
                config, (19.5, 0.180, 9.5, 108.0, 11.0))
            allocation, dual, rho = factor_lookup.get(
                config, ("relative", "log_adam_v1", 0.0))
            for train_index, train_seed in enumerate(TRAIN_SEEDS):
                for eval_index, eval_seed in enumerate(EVAL_SEEDS):
                    hf_pressure = (
                        0.40 + 0.01 * train_index + 0.002 * eval_index
                        if factor_enabled else 0.0)
                    required_fraction = (
                        0.30 + 0.30 * hf_pressure
                        if factor_enabled else 0.0)
                    required_gain = 0.002 if factor_enabled else 0.0
                    regularity_cost = (
                        0.06 if fail_budgets and config
                        in FACTORIAL_CONFIGS[1:] else 0.04)
                    expected_absolute = required_gain * regularity_cost
                    selected_absolute = required_gain * 0.03
                    rows.append({
                        "config": config,
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "restricted_total_journey_horizon_min": journey,
                        "headway_cv": cv,
                        "lower_action_mean": action,
                        "holding_vehicle_seconds": holding,
                        "fleet_denied_dispatch_events": denied,
                        "lower_discrete_critic": (
                            "zero_hold_advantage"
                            if factor_enabled else "continuous_action"),
                        "lower_policy_frozen": 1.0,
                        "lower_critic_frozen": 1.0,
                        "upper_policy_frozen": 1.0,
                        "lower_causal_guard_adjustment_mean_s": 0.0,
                        "lower_regularity_policy_enabled": float(
                            factor_enabled),
                        "lower_regularity_policy_mode": (
                            "analytic_two_sided_hf_aggregate_gain_floor_dual_v9"
                            if factor_enabled and allocation == "aggregate"
                            else "analytic_two_sided_hf_gain_floor_dual_v8"
                            if factor_enabled else "disabled"),
                        "lower_regularity_policy_constraint_cost_mode": (
                            "hf_aggregate_gain_shortfall_v4"
                            if factor_enabled and allocation == "aggregate"
                            else "hf_relative_gain_shortfall_v3"
                            if factor_enabled else "disabled"),
                        "lower_regularity_policy_constraint_scale_mode": (
                            "cost_limit_ratio_v1"
                            if factor_enabled else "raw_cost_v1"),
                        "lower_regularity_policy_dual_update_mode": dual,
                        "lower_regularity_policy_augmented_lagrangian_rho": rho,
                        "lower_regularity_policy_initial_lambda": (
                            0.01 if factor_enabled else 0.0),
                        "lower_regularity_policy_cost_limit": (
                            0.05 if factor_enabled else 0.0),
                        "lower_regularity_policy_scaled_limit": (
                            1.0 if factor_enabled else 0.0),
                        "lower_regularity_policy_action_regret_mean": (
                            0.0002 if factor_enabled else 0.0),
                        "lower_regularity_policy_evidence_valid_mean": (
                            0.60 if factor_enabled else 0.0),
                        "lower_regularity_policy_augmented_penalty": 0.0,
                        "lower_regularity_lambda": (
                            0.20 if factor_enabled else 0.0),
                        "lower_regularity_gain_floor_enabled": float(
                            factor_enabled),
                        "lower_regularity_gain_floor_mode": (
                            "causal_hf_aggregate_gain_floor_v2"
                            if factor_enabled and allocation == "aggregate"
                            else "causal_hf_relative_gain_floor_v1"
                            if factor_enabled else "disabled"),
                        "lower_regularity_gain_floor_base_fraction": (
                            0.30 if factor_enabled else 0.0),
                        "lower_regularity_gain_floor_hf_increment": (
                            0.30 if factor_enabled else 0.0),
                        "lower_regularity_gain_floor_hf_energy_scale": (
                            0.04 if factor_enabled else 1.0),
                        "lower_regularity_gain_floor_hf_energy_exponent": 1.0,
                        "lower_regularity_gain_floor_actor_required_fraction_mean": 0.0,
                        "lower_regularity_gain_floor_actor_hf_energy_mean": 0.0,
                        "lower_regularity_gain_floor_actor_hf_pressure_mean": 0.0,
                        "lower_regularity_gain_floor_actor_expected_gain_fraction_mean": 0.0,
                        "lower_regularity_gain_floor_actor_expected_shortfall_mean": 0.0,
                        "lower_regularity_gain_floor_actor_required_gain_mean": 0.0,
                        "lower_regularity_gain_floor_actor_expected_absolute_shortfall_mean": 0.0,
                        "lower_regularity_gain_floor_actor_aggregate_shortfall_ratio": 0.0,
                        "lower_regularity_gain_floor_actor_eligible_fraction": 0.0,
                        "lower_regularity_gain_floor_required_fraction_mean": required_fraction,
                        "lower_regularity_gain_floor_hf_pressure_mean": hf_pressure,
                        "lower_regularity_gain_floor_expected_gain_fraction_mean": (
                            0.60 if factor_enabled else 0.0),
                        "lower_regularity_gain_floor_selected_gain_fraction_mean": (
                            0.70 if factor_enabled else 0.0),
                        "lower_regularity_gain_floor_expected_shortfall_mean": (
                            regularity_cost if factor_enabled else 0.0),
                        "lower_regularity_gain_floor_selected_shortfall_mean": (
                            0.03 if factor_enabled else 0.0),
                        "lower_regularity_gain_floor_required_gain_mean": required_gain,
                        "lower_regularity_gain_floor_expected_absolute_shortfall_mean": expected_absolute,
                        "lower_regularity_gain_floor_selected_absolute_shortfall_mean": selected_absolute,
                        "lower_regularity_gain_floor_expected_aggregate_shortfall_ratio": (
                            regularity_cost if factor_enabled else 0.0),
                        "lower_regularity_gain_floor_selected_aggregate_shortfall_ratio": (
                            0.03 if factor_enabled else 0.0),
                        "lower_regularity_gain_floor_eligible_mean": (
                            0.70 if factor_enabled else 0.0),
                        "lower_regularity_passenger_holding_enabled": float(
                            factor_enabled),
                        "lower_regularity_passenger_holding_mode": (
                            "causal_apc_person_delay_dual_v1"
                            if factor_enabled else "disabled"),
                        "lower_regularity_passenger_constraint_scale_mode": (
                            "cost_limit_ratio_v1"
                            if factor_enabled else "raw_cost_v1"),
                        "lower_regularity_passenger_dual_update_mode": dual,
                        "lower_regularity_passenger_augmented_lagrangian_rho": rho,
                        "lower_regularity_passenger_initial_lambda": (
                            0.01 if factor_enabled else 0.0),
                        "lower_regularity_passenger_cost_limit": (
                            0.08 if factor_enabled else 0.0),
                        "lower_regularity_passenger_scaled_limit": (
                            1.0 if factor_enabled else 0.0),
                        "lower_regularity_passenger_expected_cost_mean": (
                            0.07 if factor_enabled else 0.0),
                        "lower_regularity_passenger_selected_cost_mean": (
                            0.06 if factor_enabled else 0.0),
                        "lower_regularity_passenger_load_mean": (
                            0.50 if factor_enabled else 0.0),
                        "lower_regularity_passenger_actor_augmented_penalty": 0.0,
                        "lower_regularity_passenger_lambda": (
                            0.30 if factor_enabled else 0.0),
                    })
        pd.DataFrame(rows).to_csv(root / "frozen_per_eval.csv", index=False)
        pd.DataFrame([
            {
                "candidate": config,
                "reference": V13_ANCHOR,
                "n_pairs": expected_pairs,
            }
            for config in FACTORIAL_CONFIGS
        ]).to_csv(root / "frozen_paired_deltas.csv", index=False)

    def test_selects_preregistered_fully_specified_candidate(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            result = evaluate_aggregate_gain_screen(root)

        self.assertEqual(result["status"], "exploratory_candidate_selected")
        self.assertEqual(
            result["selected_for_confirmation"], FACTORIAL_CONFIGS[-1])
        self.assertTrue(all(result["strict_checks"].values()))
        self.assertFalse(result["claim_eligible"])
        anchor = next(
            row for row in result["candidate_results"]
            if row["config"] == FACTORIAL_CONFIGS[0])
        self.assertFalse(anchor["promotion_eligible"])
        self.assertFalse(anchor["passes"])

    def test_budget_failure_cannot_promote_a_factorial_control(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root, fail_budgets=True)
            result = evaluate_aggregate_gain_screen(root)

        self.assertEqual(result["status"], "no_pass")
        self.assertIsNone(result["selected_for_confirmation"])
        self.assertFalse(any(
            row["passes"] for row in result["candidate_results"]))

    def test_manifest_seed_drift_fails_closed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            manifest_path = root / "matrix_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["train_seeds"] = [1, 2, 3, 4]
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "strict checks failed"):
                evaluate_aggregate_gain_screen(root)


if __name__ == "__main__":
    unittest.main()
