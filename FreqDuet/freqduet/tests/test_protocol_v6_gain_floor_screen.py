import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.audit_protocol_v6_gain_floor_screen import (
    CANDIDATE_SPECS,
    CANDIDATES,
    CONFIGS,
    CURRENT_MAIN,
    EVAL_SEEDS,
    FLOOR_CONFIGS,
    FLOOR_ONLY_SPECS,
    HARD_MAIN,
    MATCHED_CONTEXT,
    REFERENCE,
    SAME_ENTROPY,
    TRAIN_SEEDS,
    V13_ANCHOR,
    V13_ZERO_HOLD_ADVANTAGE,
    V20_QADV_B080,
    V20_SCALAR_B080,
    evaluate_gain_floor_screen,
)


class ProtocolV6GainFloorScreenTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> None:
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

        baselines = {
            HARD_MAIN: (21.0, 0.28, 20.0, 200.0, 30.0),
            REFERENCE: (21.0, 0.25, 12.0, 120.0, 15.0),
            MATCHED_CONTEXT: (20.5, 0.23, 11.0, 115.0, 14.0),
            CURRENT_MAIN: (20.5, 0.202, 11.0, 115.0, 14.0),
            SAME_ENTROPY: (20.5, 0.202, 11.0, 115.0, 14.0),
            V13_ANCHOR: (20.0, 0.205, 10.0, 110.0, 12.0),
            V13_ZERO_HOLD_ADVANTAGE: (20.0, 0.203, 10.0, 110.0, 12.0),
            V20_SCALAR_B080: (19.9, 0.225, 8.5, 100.0, 11.0),
            V20_QADV_B080: (19.65, 0.230, 8.0, 95.0, 10.0),
        }
        for config, _, _ in FLOOR_ONLY_SPECS:
            baselines[config] = (19.9, 0.198, 9.7, 108.0, 11.0)
        for config in CANDIDATES:
            baselines[config] = (19.8, 0.197, 9.5, 109.0, 11.0)
        floor_specs = {
            name: (base, increment)
            for name, base, increment in FLOOR_ONLY_SPECS + CANDIDATE_SPECS
        }
        passenger_configs = {
            V20_SCALAR_B080,
            V20_QADV_B080,
            *CANDIDATES,
        }
        rows = []
        for config in CONFIGS:
            journey, cv, action, holding, denied = baselines[config]
            floor_enabled = config in FLOOR_CONFIGS
            passenger_enabled = config in passenger_configs
            regularity_enabled = floor_enabled or config in {
                V13_ANCHOR,
                V13_ZERO_HOLD_ADVANTAGE,
                V20_SCALAR_B080,
                V20_QADV_B080,
            }
            base, increment = floor_specs.get(config, (0.0, 0.0))
            for train_index, train_seed in enumerate(TRAIN_SEEDS):
                for eval_index, eval_seed in enumerate(EVAL_SEEDS):
                    hf_pressure = (
                        0.40 + 0.01 * train_index + 0.002 * eval_index
                        if floor_enabled else 0.0)
                    required = (
                        base + increment * hf_pressure
                        if floor_enabled else 0.0)
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
                            if config in FLOOR_CONFIGS or config in {
                                V13_ZERO_HOLD_ADVANTAGE, V20_QADV_B080}
                            else "continuous_action"),
                        "lower_policy_frozen": 1.0,
                        "lower_critic_frozen": 1.0,
                        "upper_policy_frozen": 1.0,
                        "lower_causal_guard_adjustment_mean_s": 0.0,
                        "lower_regularity_policy_enabled": float(
                            regularity_enabled),
                        "lower_regularity_policy_mode": (
                            "analytic_two_sided_hf_gain_floor_dual_v8"
                            if floor_enabled
                            else "analytic_two_sided_zero_hold_regret_dual_v2"
                            if regularity_enabled else "disabled"),
                        "lower_regularity_policy_constraint_cost_mode": (
                            "hf_relative_gain_shortfall_v3"
                            if floor_enabled else "zero_hold_regret_v2"
                            if regularity_enabled else "disabled"),
                        "lower_regularity_policy_constraint_scale_mode": (
                            "cost_limit_ratio_v1"
                            if regularity_enabled else "raw_cost_v1"),
                        "lower_regularity_policy_initial_lambda": (
                            0.01 if regularity_enabled else 0.0),
                        "lower_regularity_policy_cost_limit": (
                            0.05 if floor_enabled else 0.00025
                            if regularity_enabled else 0.0),
                        "lower_regularity_policy_scaled_limit": (
                            1.0 if regularity_enabled else 0.0),
                        "lower_regularity_policy_action_regret_mean": (
                            0.0001 if regularity_enabled else 0.0),
                        "lower_regularity_policy_evidence_valid_mean": (
                            0.75 if regularity_enabled else 0.0),
                        "lower_regularity_lambda": (
                            0.2 if regularity_enabled else 0.0),
                        "lower_regularity_gain_floor_enabled": float(
                            floor_enabled),
                        "lower_regularity_gain_floor_mode": (
                            "causal_hf_relative_gain_floor_v1"
                            if floor_enabled else "disabled"),
                        "lower_regularity_gain_floor_base_fraction": base,
                        "lower_regularity_gain_floor_hf_increment": increment,
                        "lower_regularity_gain_floor_hf_energy_scale": (
                            0.04 if floor_enabled else 1.0),
                        "lower_regularity_gain_floor_hf_energy_exponent": 1.0,
                        "lower_regularity_gain_floor_actor_required_fraction_mean": 0.0,
                        "lower_regularity_gain_floor_actor_hf_energy_mean": 0.0,
                        "lower_regularity_gain_floor_actor_hf_pressure_mean": 0.0,
                        "lower_regularity_gain_floor_actor_expected_gain_fraction_mean": 0.0,
                        "lower_regularity_gain_floor_actor_expected_shortfall_mean": 0.0,
                        "lower_regularity_gain_floor_actor_eligible_fraction": 0.0,
                        "lower_regularity_gain_floor_required_fraction_mean": required,
                        "lower_regularity_gain_floor_hf_pressure_mean": hf_pressure,
                        "lower_regularity_gain_floor_expected_gain_fraction_mean": (
                            0.65 if floor_enabled else 0.0),
                        "lower_regularity_gain_floor_selected_gain_fraction_mean": (
                            0.70 if floor_enabled else 0.0),
                        "lower_regularity_gain_floor_expected_shortfall_mean": (
                            0.04 if floor_enabled else 0.0),
                        "lower_regularity_gain_floor_selected_shortfall_mean": (
                            0.02 if floor_enabled else 0.0),
                        "lower_regularity_gain_floor_eligible_mean": (
                            0.75 if floor_enabled else 0.0),
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
                        "lower_regularity_passenger_cost_limit": (
                            0.08 if passenger_enabled else 0.0),
                        "lower_regularity_passenger_scaled_limit": (
                            1.0 if passenger_enabled else 0.0),
                        "lower_regularity_passenger_expected_cost_mean": (
                            0.06 if passenger_enabled else 0.0),
                        "lower_regularity_passenger_selected_cost_mean": (
                            0.05 if passenger_enabled else 0.0),
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
                "n_pairs": expected_pairs,
            }
            for config in CONFIGS if config != V13_ANCHOR
        ]).to_csv(root / "frozen_paired_deltas.csv", index=False)

    def test_selects_preregistered_mild_hf_floor_first(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            result = evaluate_gain_floor_screen(root)

        self.assertEqual(
            result["status"], "exploratory_candidate_selected")
        self.assertEqual(
            result["selected_for_confirmation"], CANDIDATE_SPECS[1][0])
        self.assertTrue(all(
            row["passes"] for row in result["floor_control_results"]))
        self.assertTrue(all(
            row["passes"] for row in result["candidate_results"]))

    def test_rejects_floor_that_exceeds_expected_shortfall_budget(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            frame = pd.read_csv(root / "frozen_per_eval.csv")
            frame.loc[
                frame["config"].isin(CANDIDATES),
                "lower_regularity_gain_floor_expected_shortfall_mean",
            ] = 0.051
            frame.to_csv(root / "frozen_per_eval.csv", index=False)
            result = evaluate_gain_floor_screen(root)

        self.assertEqual(result["status"], "no_pass")
        self.assertTrue(all(
            not row["mechanism_checks"][
                "expected_floor_budget_satisfied_every_rollout"]
            for row in result["candidate_results"]
        ))

    def test_rejects_inconsistent_hf_required_fraction_telemetry(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            frame = pd.read_csv(root / "frozen_per_eval.csv")
            frame.loc[
                frame["config"] == CANDIDATE_SPECS[1][0],
                "lower_regularity_gain_floor_required_fraction_mean",
            ] += 0.01
            frame.to_csv(root / "frozen_per_eval.csv", index=False)
            result = evaluate_gain_floor_screen(root)

        row = next(
            item for item in result["candidate_results"]
            if item["config"] == CANDIDATE_SPECS[1][0])
        self.assertFalse(
            row["mechanism_checks"]["passive_floor_telemetry_active"])
        self.assertFalse(row["passes"])


if __name__ == "__main__":
    unittest.main()
