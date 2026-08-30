import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from scripts.audit_protocol_v6_zero_hold_regret_screen import (
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
    V12_ABSOLUTE_TARGET,
    evaluate_zero_hold_regret_screen,
)


class ProtocolV6ZeroHoldRegretScreenTest(unittest.TestCase):
    def _aggregate(self, root: Path, *, dirty: bool = False) -> Path:
        root.mkdir(parents=True, exist_ok=True)
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
            "reference": REFERENCE,
            "run_git_provenance": {
                "commit": "abc123",
                "tracked_dirty": dirty,
            },
            "scenario_contract": {"version": "test-scenario"},
            "expected_rollouts": (
                len(CONFIGS) * len(TRAIN_SEEDS) * len(EVAL_SEEDS)),
        }
        (root / "matrix_manifest.json").write_text(json.dumps(manifest))

        values = {
            HARD_MAIN: (21.0, 0.18, 80.0, 200.0),
            REFERENCE: (22.0, 0.25, 100.0, 250.0),
            MATCHED_CONTEXT: (21.5, 0.23, 90.0, 220.0),
            CURRENT_MAIN: (21.2, 0.22, 85.0, 210.0),
            SAME_ENTROPY: (21.05, 0.210, 82.0, 200.0),
            V12_ABSOLUTE_TARGET: (21.5, 0.190, 83.0, 205.0),
        }
        for candidate in CANDIDATES:
            values[candidate] = (20.9, 0.195, 80.0, 190.0)

        specs = {
            candidate: (initial, limit)
            for candidate, initial, limit in CANDIDATE_SPECS
        }
        rows = []
        for config in CONFIGS:
            journey, cv, holding, denied = values[config]
            is_candidate = config in CANDIDATES
            initial, limit = specs.get(config, (0.0, 0.002))
            is_dual = config in {
                SAME_ENTROPY, V12_ABSOLUTE_TARGET, *CANDIDATES}
            regret = (
                0.5 * limit if is_candidate
                else 0.001 if config == SAME_ENTROPY
                else 0.0002)
            entropy = 0.25 * np.log(7.0) if (
                is_candidate or config == SAME_ENTROPY) else 1.8
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
                        "lower_regularity_policy_enabled": float(is_dual),
                        "lower_regularity_policy_mode": (
                            "analytic_two_sided_zero_hold_regret_dual_v2"
                            if is_candidate
                            else "analytic_two_sided_target_dual_v1"
                            if is_dual else "disabled"),
                        "lower_regularity_policy_constraint_scale_mode": (
                            "cost_limit_ratio_v1"
                            if is_candidate else "raw_cost_v1"),
                        "lower_regularity_policy_initial_lambda": initial,
                        "lower_regularity_policy_scaled_limit": (
                            1.0 if is_candidate else limit),
                        "lower_regularity_policy_action_cost_mean": 0.004,
                        "lower_regularity_policy_zero_hold_action_cost_mean": (
                            0.001),
                        "lower_regularity_policy_action_regret_mean": regret,
                        "lower_regularity_policy_action_regret_max": 0.01,
                        "lower_regularity_policy_evidence_valid_mean": 0.8,
                        "lower_regularity_entropy_split_enabled": float(
                            is_candidate or config == SAME_ENTROPY),
                        "lower_regularity_entropy_target_fraction": (
                            0.25 if is_candidate or config == SAME_ENTROPY
                            else 0.98),
                        "lower_regularity_alpha": (
                            0.02 if is_candidate or config == SAME_ENTROPY
                            else 0.0),
                        "lower_alpha": 0.07,
                        "lower_regularity_policy_entropy_valid_mean": entropy,
                    })
        pd.DataFrame(rows).to_csv(root / "frozen_per_eval.csv", index=False)
        pd.DataFrame([{"config": name} for name in CONFIGS]).to_csv(
            root / "frozen_summary.csv", index=False)

        paired_rows = []
        for candidate in CANDIDATES:
            journey, cv, holding, denied = values[candidate]
            paired_rows.append({
                "candidate": candidate,
                "reference": REFERENCE,
                "n_pairs": len(TRAIN_SEEDS) * len(EVAL_SEEDS),
                "delta_restricted_total_journey_horizon_min_mean": (
                    journey - values[REFERENCE][0]),
                "delta_restricted_total_journey_horizon_min_ci_high": -0.5,
                "delta_headway_cv_mean": cv - values[REFERENCE][1],
                "delta_headway_cv_ci_high": -0.01,
                "delta_holding_vehicle_seconds_mean": (
                    holding - values[REFERENCE][2]),
                "delta_fleet_denied_dispatch_events_mean": (
                    denied - values[REFERENCE][3]),
            })
        pd.DataFrame(paired_rows).to_csv(
            root / "frozen_paired_deltas.csv", index=False)
        return root

    def test_weakest_passing_regret_constraint_is_selected(self):
        with TemporaryDirectory() as tmp:
            result = evaluate_zero_hold_regret_screen(
                self._aggregate(Path(tmp) / "aggregate"))

        self.assertEqual(result["status"], "exploratory_candidate_selected")
        self.assertFalse(result["claim_eligible"])
        self.assertEqual(
            result["selected_for_confirmation"],
            "F_freqduet_protocol_v6_w2adregret_l001_e25_r0010_hiro",
        )

    def test_dirty_source_fails_closed(self):
        with TemporaryDirectory() as tmp:
            aggregate = self._aggregate(Path(tmp) / "aggregate", dirty=True)
            with self.assertRaisesRegex(ValueError, "strict checks"):
                evaluate_zero_hold_regret_screen(aggregate)


if __name__ == "__main__":
    unittest.main()
