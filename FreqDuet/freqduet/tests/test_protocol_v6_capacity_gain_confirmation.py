import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.audit_protocol_v6_capacity_gain_confirmation import (
    CONFIGS,
    EVAL_SEEDS,
    SELECTED,
    TRAIN_SEEDS,
    evaluate_capacity_gain_confirmation,
)
from scripts.audit_protocol_v6_capacity_gain_screen import (
    CONFIGS as SCREEN_CONFIGS,
    CURRENT_MAIN,
    EVAL_SEEDS as SCREEN_EVAL_SEEDS,
    HARD_MAIN,
    MATCHED_CONTEXT,
    REFERENCE,
    SAME_ENTROPY,
    TRAIN_SEEDS as SCREEN_TRAIN_SEEDS,
    V13_ANCHOR,
)


class ProtocolV6CapacityGainConfirmationTest(unittest.TestCase):
    def _artifacts(self, root: Path, *, journey_ci_high: float = 0.15):
        screen = root / "screen"
        screen.mkdir(parents=True)
        screen_manifest = {
            "strict_complete": True,
            "stage": "exploratory",
            "independent_confirmation": False,
            "configs": SCREEN_CONFIGS,
            "train_seeds": SCREEN_TRAIN_SEEDS,
            "eval_seeds": SCREEN_EVAL_SEEDS,
            "train_episodes": 40,
            "checkpoint_ep": 39,
            "run_git_provenance": {
                "commit": "abc123",
                "tracked_dirty": False,
            },
            "run_source_fingerprint": {"sha256": "source"},
            "scenario_contract": {"sha256": "scenario"},
        }
        (screen / "matrix_manifest.json").write_text(
            json.dumps(screen_manifest))
        (screen / "capacity_gain_screen.json").write_text(json.dumps({
            "gate_version": "freqduet-v14-capacity-gain-screen-v1",
            "status": "exploratory_candidate_selected",
            "claim_eligible": False,
            "selected_for_confirmation": SELECTED,
            "candidate_results": [{
                "config": SELECTED,
                "passes": True,
                "mechanism_checks": {"mechanism": True},
                "outcome_checks": {"outcome": True},
            }],
        }))

        aggregate = root / "confirmation"
        aggregate.mkdir()
        expected_pairs = len(TRAIN_SEEDS) * len(EVAL_SEEDS)
        (aggregate / "matrix_manifest.json").write_text(json.dumps({
            "strict_complete": True,
            "run_manifests_verified": True,
            "common_random_numbers_verified": True,
            "stage": "confirmation",
            "independent_confirmation": True,
            "configs": CONFIGS,
            "train_seeds": TRAIN_SEEDS,
            "eval_seeds": EVAL_SEEDS,
            "train_episodes": 200,
            "checkpoint_ep": 199,
            "reference": V13_ANCHOR,
            "expected_rollouts": len(CONFIGS) * expected_pairs,
            "run_git_provenance": {
                "commit": "abc123",
                "tracked_dirty": False,
            },
            "run_source_fingerprint": {"sha256": "source"},
            "scenario_contract": {"sha256": "scenario"},
        }))
        values = {
            HARD_MAIN: (21.0, 0.18, 200.0, 500.0),
            REFERENCE: (22.0, 0.25, 100.0, 250.0),
            MATCHED_CONTEXT: (21.5, 0.23, 95.0, 230.0),
            CURRENT_MAIN: (21.4, 0.205, 110.0, 260.0),
            SAME_ENTROPY: (21.2, 0.200, 105.0, 240.0),
            V13_ANCHOR: (20.7, 0.230, 80.0, 200.0),
            SELECTED: (20.8, 0.190, 90.0, 210.0),
        }
        rows = []
        for config in CONFIGS:
            journey, cv, holding, denied = values[config]
            selected = config == SELECTED
            for train_seed in TRAIN_SEEDS:
                for eval_seed in EVAL_SEEDS:
                    rows.append({
                        "config": config,
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "restricted_total_journey_horizon_min": journey,
                        "headway_cv": cv,
                        "holding_vehicle_seconds": holding,
                        "fleet_denied_dispatch_events": denied,
                        "lower_causal_guard_adjustment_mean_s": 0.0,
                        "lower_regularity_policy_enabled": float(selected),
                        "lower_regularity_policy_mode": (
                            "analytic_two_sided_capacity_gain_regret_dual_v3"
                            if selected else "disabled"),
                        "lower_regularity_policy_constraint_scale_mode": (
                            "cost_limit_ratio_v1" if selected else
                            "raw_cost_v1"),
                        "lower_regularity_policy_initial_lambda": (
                            0.01 if selected else 0.0),
                        "lower_regularity_policy_cost_limit": (
                            0.00025 if selected else 0.0),
                        "lower_regularity_policy_scaled_limit": (
                            1.0 if selected else 0.0),
                        "lower_regularity_policy_action_regret_mean": (
                            0.0001 if selected else 0.0),
                        "lower_regularity_policy_action_regret_max": 0.01,
                        "lower_regularity_policy_evidence_valid_mean": (
                            0.8 if selected else 0.0),
                        "lower_regularity_policy_capacity_gain_enabled": (
                            float(selected)),
                        "lower_regularity_policy_capacity_gain_weight": (
                            0.02 if selected else 0.0),
                        "lower_regularity_policy_capacity_gain_scale": (
                            0.002 if selected else 1.0),
                        "lower_regularity_policy_capacity_exponent": 1.0,
                        "lower_regularity_policy_capacity_gain_mean": (
                            0.0004 if selected else 0.0),
                        "lower_regularity_policy_scaled_capacity_gain_mean": (
                            0.2 if selected else 0.0),
                        "lower_regularity_policy_capacity_gain_bonus": (
                            0.004 if selected else 0.0),
                        "lower_regularity_policy_capacity_gate_mean": (
                            0.4 if selected else 0.0),
                    })
        pd.DataFrame(rows).to_csv(
            aggregate / "frozen_per_eval.csv", index=False)
        pd.DataFrame([{"config": config} for config in CONFIGS]).to_csv(
            aggregate / "frozen_summary.csv", index=False)
        anchor = values[V13_ANCHOR]
        candidate = values[SELECTED]
        pd.DataFrame([{
            "candidate": SELECTED,
            "reference": V13_ANCHOR,
            "n_pairs": expected_pairs,
            "delta_restricted_total_journey_horizon_min_mean": (
                candidate[0] - anchor[0]),
            "delta_restricted_total_journey_horizon_min_ci_high": (
                journey_ci_high),
            "delta_headway_cv_mean": candidate[1] - anchor[1],
            "delta_headway_cv_ci_high": -0.02,
        }]).to_csv(aggregate / "frozen_paired_deltas.csv", index=False)
        return aggregate, screen

    def test_confirms_the_preregistered_candidate(self):
        with TemporaryDirectory() as tmp:
            aggregate, screen = self._artifacts(Path(tmp))
            result = evaluate_capacity_gain_confirmation(
                aggregate,
                screen_dir=screen,
            )

        self.assertEqual(result["status"], "capacity_gain_confirmed")
        self.assertTrue(result["confirmation_claim_eligible"])
        self.assertTrue(all(result["strict_checks"].values()))
        self.assertTrue(all(result["mechanism_checks"].values()))
        self.assertTrue(all(result["outcome_checks"].values()))

    def test_journey_ci_noninferiority_controls_confirmation(self):
        with TemporaryDirectory() as tmp:
            aggregate, screen = self._artifacts(
                Path(tmp), journey_ci_high=0.21)
            result = evaluate_capacity_gain_confirmation(
                aggregate,
                screen_dir=screen,
            )

        self.assertEqual(result["status"], "capacity_gain_not_confirmed")
        self.assertFalse(result["confirmation_claim_eligible"])
        self.assertFalse(result["outcome_checks"][
            "journey_noninferior_to_v13_with_ci"])

    def test_source_change_fails_before_effect_audit(self):
        with TemporaryDirectory() as tmp:
            aggregate, screen = self._artifacts(Path(tmp))
            manifest_path = aggregate / "matrix_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["run_git_provenance"]["commit"] = "different"
            manifest_path.write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "strict checks"):
                evaluate_capacity_gain_confirmation(
                    aggregate,
                    screen_dir=screen,
                )


if __name__ == "__main__":
    unittest.main()
