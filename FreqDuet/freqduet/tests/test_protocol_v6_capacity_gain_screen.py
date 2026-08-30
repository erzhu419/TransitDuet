import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.audit_protocol_v6_capacity_gain_screen import (
    CANDIDATE_SPECS,
    CANDIDATES,
    CONFIGS,
    CURRENT_MAIN,
    EVAL_SEEDS,
    HARD_MAIN,
    MATCHED_CONTEXT,
    REFERENCE,
    SAME_ENTROPY,
    TRAIN_SEEDS,
    V13_ANCHOR,
    evaluate_capacity_gain_screen,
)


class ProtocolV6CapacityGainScreenTest(unittest.TestCase):
    def _aggregate(self, root: Path, *, dirty: bool = False) -> Path:
        root.mkdir(parents=True, exist_ok=True)
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
            "run_git_provenance": {
                "commit": "abc123",
                "tracked_dirty": dirty,
            },
            "expected_rollouts": len(CONFIGS) * expected_pairs,
        }
        (root / "matrix_manifest.json").write_text(json.dumps(manifest))

        values = {
            HARD_MAIN: (21.0, 0.18, 200.0, 500.0),
            REFERENCE: (22.0, 0.25, 100.0, 250.0),
            MATCHED_CONTEXT: (21.5, 0.23, 95.0, 230.0),
            CURRENT_MAIN: (21.4, 0.205, 110.0, 260.0),
            SAME_ENTROPY: (21.2, 0.200, 105.0, 240.0),
            V13_ANCHOR: (20.7, 0.230, 80.0, 200.0),
        }
        for candidate in CANDIDATES:
            values[candidate] = (20.8, 0.195, 90.0, 210.0)
        specs = {
            candidate: (weight, exponent)
            for candidate, weight, exponent in CANDIDATE_SPECS
        }
        rows = []
        for config in CONFIGS:
            journey, cv, holding, denied = values[config]
            is_candidate = config in CANDIDATES
            weight, exponent = specs.get(config, (0.0, 1.0))
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
                        "lower_regularity_policy_enabled": float(is_candidate),
                        "lower_regularity_policy_mode": (
                            "analytic_two_sided_capacity_gain_regret_dual_v3"
                            if is_candidate else "disabled"),
                        "lower_regularity_policy_constraint_scale_mode": (
                            "cost_limit_ratio_v1" if is_candidate
                            else "raw_cost_v1"),
                        "lower_regularity_policy_initial_lambda": (
                            0.01 if is_candidate else 0.0),
                        "lower_regularity_policy_cost_limit": (
                            0.00025 if is_candidate else 0.0),
                        "lower_regularity_policy_scaled_limit": (
                            1.0 if is_candidate else 0.0),
                        "lower_regularity_policy_action_regret_mean": (
                            0.0001 if is_candidate else 0.0),
                        "lower_regularity_policy_action_regret_max": 0.01,
                        "lower_regularity_policy_evidence_valid_mean": (
                            0.8 if is_candidate else 0.0),
                        "lower_regularity_policy_capacity_gain_enabled": (
                            float(is_candidate)),
                        "lower_regularity_policy_capacity_gain_weight": weight,
                        "lower_regularity_policy_capacity_gain_scale": (
                            0.002 if is_candidate else 1.0),
                        "lower_regularity_policy_capacity_exponent": exponent,
                        "lower_regularity_policy_capacity_gain_mean": (
                            0.0004 if is_candidate else 0.0),
                        "lower_regularity_policy_scaled_capacity_gain_mean": (
                            0.2 if is_candidate else 0.0),
                        "lower_regularity_policy_capacity_gain_bonus": (
                            weight * 0.2 if is_candidate else 0.0),
                        "lower_regularity_policy_capacity_gate_mean": (
                            0.4 if is_candidate else 0.0),
                    })
        pd.DataFrame(rows).to_csv(root / "frozen_per_eval.csv", index=False)
        pd.DataFrame([{"config": name} for name in CONFIGS]).to_csv(
            root / "frozen_summary.csv", index=False)

        paired_rows = []
        anchor = values[V13_ANCHOR]
        for config in CONFIGS:
            if config == V13_ANCHOR:
                continue
            journey, cv, holding, denied = values[config]
            paired_rows.append({
                "candidate": config,
                "reference": V13_ANCHOR,
                "n_pairs": expected_pairs,
                "delta_restricted_total_journey_horizon_min_mean": (
                    journey - anchor[0]),
                "delta_restricted_total_journey_horizon_min_ci_high": 0.15,
                "delta_headway_cv_mean": cv - anchor[1],
                "delta_headway_cv_ci_high": (
                    -0.01 if config in CANDIDATES else 0.1),
                "delta_holding_vehicle_seconds_mean": holding - anchor[2],
                "delta_fleet_denied_dispatch_events_mean": denied - anchor[3],
            })
        pd.DataFrame(paired_rows).to_csv(
            root / "frozen_paired_deltas.csv", index=False)
        return root

    def test_selects_weakest_passing_capacity_gain(self):
        with TemporaryDirectory() as tmp:
            result = evaluate_capacity_gain_screen(
                self._aggregate(Path(tmp) / "aggregate"))

        self.assertEqual(result["status"], "exploratory_candidate_selected")
        self.assertFalse(result["claim_eligible"])
        self.assertEqual(
            result["selected_for_confirmation"],
            "F_freqduet_protocol_v6_w2adcapgain_l001_e25_r00025_"
            "w0005_x2_hiro",
        )

    def test_dirty_source_fails_closed(self):
        with TemporaryDirectory() as tmp:
            aggregate = self._aggregate(Path(tmp) / "aggregate", dirty=True)
            with self.assertRaisesRegex(ValueError, "strict checks"):
                evaluate_capacity_gain_screen(aggregate)

    def test_configured_but_inactive_capacity_gain_does_not_pass(self):
        with TemporaryDirectory() as tmp:
            aggregate = self._aggregate(Path(tmp) / "aggregate")
            path = aggregate / "frozen_per_eval.csv"
            frame = pd.read_csv(path)
            candidate_rows = frame["config"].isin(CANDIDATES)
            for column in (
                    "lower_regularity_policy_capacity_gain_mean",
                    "lower_regularity_policy_scaled_capacity_gain_mean",
                    "lower_regularity_policy_capacity_gain_bonus"):
                frame.loc[candidate_rows, column] = 0.0
            frame.to_csv(path, index=False)

            result = evaluate_capacity_gain_screen(aggregate)

        self.assertEqual(result["status"], "no_pass")
        self.assertTrue(all(
            not item["mechanism_checks"][
                "capacity_gain_active_in_every_rollout"]
            for item in result["candidate_results"]
        ))

    def test_inconsistent_realized_gain_arithmetic_does_not_pass(self):
        with TemporaryDirectory() as tmp:
            aggregate = self._aggregate(Path(tmp) / "aggregate")
            path = aggregate / "frozen_per_eval.csv"
            frame = pd.read_csv(path)
            candidate_rows = frame["config"].isin(CANDIDATES)
            frame.loc[
                candidate_rows,
                "lower_regularity_policy_scaled_capacity_gain_mean",
            ] = 0.25
            frame.to_csv(path, index=False)

            result = evaluate_capacity_gain_screen(aggregate)

        self.assertEqual(result["status"], "no_pass")
        self.assertTrue(all(
            not item["mechanism_checks"][
                "capacity_gain_arithmetic_verified"]
            for item in result["candidate_results"]
        ))


if __name__ == "__main__":
    unittest.main()
