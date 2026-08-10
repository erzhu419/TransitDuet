import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from scripts.audit_protocol_v6_normalized_constraint_screen import (
    CANDIDATE_SPECS,
    CANDIDATES,
    CONFIGS,
    CURRENT_MAIN,
    EVAL_SEEDS,
    HARD_MAIN,
    MATCHED_CONTEXT,
    NO_ENTROPY,
    REFERENCE,
    SAME_ENTROPY,
    SCENARIO_SHA256,
    TRAIN_SEEDS,
    evaluate_normalized_constraint_screen,
)


class ProtocolV6NormalizedConstraintScreenTest(unittest.TestCase):
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
            "source_fingerprint": {"sha256": "source"},
            "scenario_contract": {"sha256": SCENARIO_SHA256},
            "launch_analysis_fingerprint": {"sha256": "analysis"},
            "expected_rollouts": (
                len(CONFIGS) * len(TRAIN_SEEDS) * len(EVAL_SEEDS)),
        }
        (root / "matrix_manifest.json").write_text(json.dumps(manifest))

        values = {
            HARD_MAIN: (21.0, 0.18, 80.0, 200.0),
            REFERENCE: (22.0, 0.25, 100.0, 250.0),
            MATCHED_CONTEXT: (21.5, 0.23, 90.0, 220.0),
            CURRENT_MAIN: (21.2, 0.22, 85.0, 210.0),
            NO_ENTROPY["0010"]: (21.1, 0.215, 84.0, 205.0),
            NO_ENTROPY["0020"]: (21.1, 0.215, 84.0, 205.0),
            SAME_ENTROPY["0010"]: (21.05, 0.210, 82.0, 200.0),
            SAME_ENTROPY["0020"]: (21.05, 0.210, 82.0, 200.0),
        }
        for candidate in CANDIDATES:
            values[candidate] = (20.9, 0.195, 80.0, 190.0)

        specs = {
            candidate: (initial, fraction, limit)
            for candidate, initial, fraction, limit, _ in CANDIDATE_SPECS
        }
        same_entropy_specs = {
            SAME_ENTROPY["0010"]: (0.50, 0.001),
            SAME_ENTROPY["0020"]: (0.25, 0.002),
        }
        rows = []
        for config in CONFIGS:
            journey, cv, holding, denied = values[config]
            is_candidate = config in CANDIDATES
            is_same_entropy = config in same_entropy_specs
            initial, fraction, limit = specs.get(
                config,
                (0.0, *same_entropy_specs.get(config, (0.98, 0.01))),
            )
            is_dual = (
                is_candidate or is_same_entropy
                or config in NO_ENTROPY.values())
            entropy = fraction * np.log(7.0) if (
                is_candidate or is_same_entropy) else 1.8
            action_cost = (
                0.8 * limit if is_candidate
                else limit + 0.0005 if is_same_entropy
                else 0.0025)
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
                        "lower_regularity_policy_constraint_scale_mode": (
                            "cost_limit_ratio_v1"
                            if is_candidate else "raw_cost_v1"),
                        "lower_regularity_policy_initial_lambda": initial,
                        "lower_regularity_policy_scaled_limit": (
                            1.0 if is_candidate else limit),
                        "lower_regularity_policy_action_cost_mean": action_cost,
                        "lower_regularity_policy_oracle_action_cost_mean": (
                            0.05 * limit),
                        "lower_regularity_policy_evidence_valid_mean": 0.8,
                        "lower_regularity_entropy_split_enabled": float(
                            is_candidate or is_same_entropy),
                        "lower_regularity_entropy_target_fraction": fraction,
                        "lower_regularity_alpha": (
                            0.02 if is_candidate or is_same_entropy else 0.0),
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

    def test_weakest_passing_normalized_constraint_is_selected(self):
        with TemporaryDirectory() as tmp:
            result = evaluate_normalized_constraint_screen(
                self._aggregate(Path(tmp) / "aggregate"))

        self.assertEqual(result["status"], "exploratory_candidate_selected")
        self.assertFalse(result["claim_eligible"])
        self.assertEqual(
            result["selected_for_confirmation"],
            "F_freqduet_protocol_v6_w2adnorm_l005_e25_c0020_hiro",
        )

    def test_dirty_source_fails_closed(self):
        with TemporaryDirectory() as tmp:
            aggregate = self._aggregate(
                Path(tmp) / "aggregate", dirty=True)
            with self.assertRaisesRegex(ValueError, "strict checks"):
                evaluate_normalized_constraint_screen(aggregate)


if __name__ == "__main__":
    unittest.main()
