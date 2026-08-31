import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.audit_protocol_v6_capacity_gain_screen import (
    CURRENT_MAIN,
    HARD_MAIN,
    MATCHED_CONTEXT,
    REFERENCE,
    SAME_ENTROPY,
    V13_ANCHOR,
)
from scripts.audit_protocol_v6_efficiency_gain_screen import (
    CANDIDATE_SPECS,
    CONFIGS,
    EVAL_SEEDS,
    PRIORITY,
    TRAIN_SEEDS,
    V14_ANCHOR,
    evaluate_efficiency_gain_screen,
)


class ProtocolV6EfficiencyGainScreenTest(unittest.TestCase):
    def _artifacts(self, root: Path, *, active_efficiency: bool = True):
        expected_pairs = len(TRAIN_SEEDS) * len(EVAL_SEEDS)
        (root / "matrix_manifest.json").write_text(json.dumps({
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
            "expected_rollouts": len(CONFIGS) * expected_pairs,
            "run_git_provenance": {"tracked_dirty": False},
        }))
        values = {
            HARD_MAIN: (21.0, 0.18, 500.0, 500.0, 18.0),
            REFERENCE: (21.0, 0.25, 100.0, 100.0, 15.0),
            MATCHED_CONTEXT: (20.9, 0.24, 100.0, 100.0, 14.0),
            CURRENT_MAIN: (20.8, 0.21, 150.0, 150.0, 13.0),
            SAME_ENTROPY: (20.6, 0.21, 140.0, 140.0, 13.0),
            V13_ANCHOR: (20.2, 0.23, 120.0, 120.0, 12.0),
            V14_ANCHOR: (20.3, 0.215, 160.0, 160.0, 13.0),
        }
        rows = []
        spec_by_name = {
            name: (weight, penalty)
            for name, weight, penalty in CANDIDATE_SPECS
        }
        for config in CONFIGS:
            candidate = config in spec_by_name
            if candidate:
                journey, cv, holding, denied, action = (
                    20.15, 0.20, 130.0, 130.0, 12.5)
                weight, penalty = spec_by_name[config]
                efficiency = (
                    1.0 / (1.0 + 0.5 * penalty)
                    if active_efficiency else 1.0)
            else:
                journey, cv, holding, denied, action = values[config]
                weight, penalty, efficiency = 0.0, 0.0, 1.0
            for train_seed in TRAIN_SEEDS:
                for eval_seed in EVAL_SEEDS:
                    gain = 0.001 if candidate else 0.0
                    rows.append({
                        "config": config,
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "restricted_total_journey_horizon_min": journey,
                        "headway_cv": cv,
                        "holding_vehicle_seconds": holding,
                        "fleet_denied_dispatch_events": denied,
                        "lower_action_mean": action,
                        "lower_causal_guard_adjustment_mean_s": 0.0,
                        "lower_regularity_policy_enabled": float(candidate),
                        "lower_regularity_policy_mode": (
                            "analytic_two_sided_efficiency_gain_regret_dual_v4"
                            if candidate else "disabled"),
                        "lower_regularity_policy_constraint_scale_mode": (
                            "cost_limit_ratio_v1" if candidate else
                            "raw_cost_v1"),
                        "lower_regularity_policy_initial_lambda": (
                            0.01 if candidate else 0.0),
                        "lower_regularity_policy_cost_limit": (
                            0.00025 if candidate else 0.0),
                        "lower_regularity_policy_scaled_limit": (
                            1.0 if candidate else 0.0),
                        "lower_regularity_policy_action_regret_mean": (
                            0.0001 if candidate else 0.0),
                        "lower_regularity_policy_evidence_valid_mean": (
                            0.8 if candidate else 0.0),
                        "lower_regularity_policy_capacity_gain_enabled": (
                            float(candidate)),
                        "lower_regularity_policy_capacity_gain_mode": (
                            "positive_zero_hold_efficiency_gain_v2"
                            if candidate else "disabled"),
                        "lower_regularity_policy_capacity_gain_weight": weight,
                        "lower_regularity_policy_capacity_gain_scale": (
                            0.002 if candidate else 1.0),
                        "lower_regularity_policy_capacity_exponent": 1.0,
                        "lower_regularity_policy_action_efficiency_penalty": (
                            penalty),
                        "lower_regularity_policy_actor_capacity_gain_mean": gain,
                        "lower_regularity_policy_actor_scaled_capacity_gain_mean": (
                            gain / 0.002 if candidate else 0.0),
                        "lower_regularity_policy_actor_capacity_gain_bonus": (
                            weight * gain / 0.002 if candidate else 0.0),
                        "lower_regularity_policy_actor_action_efficiency_gate_mean": (
                            efficiency),
                        "lower_regularity_policy_capacity_gain_mean": gain,
                        "lower_regularity_policy_scaled_capacity_gain_mean": (
                            gain / 0.002 if candidate else 0.0),
                        "lower_regularity_policy_capacity_gain_bonus": (
                            weight * gain / 0.002 if candidate else 0.0),
                        "lower_regularity_policy_capacity_gate_mean": (
                            0.5 if candidate else 0.0),
                        "lower_regularity_policy_action_efficiency_gate_mean": (
                            efficiency),
                    })
        pd.DataFrame(rows).to_csv(root / "frozen_per_eval.csv", index=False)
        pd.DataFrame([{"config": config} for config in CONFIGS]).to_csv(
            root / "frozen_summary.csv", index=False)
        paired = []
        for candidate, _, _ in CANDIDATE_SPECS:
            paired.append({
                "candidate": candidate,
                "reference": V13_ANCHOR,
                "n_pairs": expected_pairs,
                "delta_restricted_total_journey_horizon_min_mean": -0.05,
                "delta_restricted_total_journey_horizon_min_ci_low": -0.10,
                "delta_restricted_total_journey_horizon_min_ci_high": 0.05,
                "delta_headway_cv_mean": -0.03,
                "delta_headway_cv_ci_low": -0.04,
                "delta_headway_cv_ci_high": -0.02,
            })
        pd.DataFrame(paired).to_csv(
            root / "frozen_paired_deltas.csv", index=False)

    def test_selects_the_weakest_passing_efficiency_candidate(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._artifacts(root)
            result = evaluate_efficiency_gain_screen(root)

        self.assertEqual(result["status"], "exploratory_candidate_selected")
        self.assertEqual(result["selected_for_confirmation"], PRIORITY[0])
        self.assertTrue(all(result["strict_checks"].values()))
        self.assertTrue(all(result["candidate_results"][0][
            "mechanism_checks"].values()))

    def test_inactive_efficiency_gate_cannot_pass(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._artifacts(root, active_efficiency=False)
            result = evaluate_efficiency_gain_screen(root)

        self.assertEqual(result["status"], "no_pass")
        self.assertTrue(all(
            not item["mechanism_checks"][
                "efficiency_gate_active_in_every_rollout"]
            for item in result["candidate_results"]))

    def test_wrong_seed_grid_fails_before_effect_selection(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._artifacts(root)
            path = root / "matrix_manifest.json"
            manifest = json.loads(path.read_text())
            manifest["train_seeds"] = [1, 2, 3, 4]
            path.write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "strict checks"):
                evaluate_efficiency_gain_screen(root)


if __name__ == "__main__":
    unittest.main()
