import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.audit_protocol_v6_actiondual_screen import (
    CANDIDATES,
    CONFIGS,
    CURRENT_MAIN,
    EVAL_SEEDS,
    HARD_MAIN,
    MATCHED_CONTEXT,
    REFERENCE,
    SCENARIO_SHA256,
    TRAIN_SEEDS,
    evaluate_actiondual_screen,
)


class ProtocolV6ActionDualScreenTest(unittest.TestCase):
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
            "run_source_fingerprint": {"sha256": "source"},
            "scenario_contract": {"sha256": SCENARIO_SHA256},
            "launch_analysis_sha256": "analysis",
            "expected_rollouts": (
                len(CONFIGS) * len(TRAIN_SEEDS) * len(EVAL_SEEDS)),
        }
        (root / "matrix_manifest.json").write_text(json.dumps(manifest))

        values = {
            HARD_MAIN: (21.0, 0.18, 80.0, 200.0, 0.004),
            REFERENCE: (22.0, 0.25, 100.0, 250.0, 0.004),
            MATCHED_CONTEXT: (21.5, 0.23, 90.0, 220.0, 0.004),
            CURRENT_MAIN: (21.2, 0.22, 85.0, 210.0, 0.003),
        }
        for candidate in CANDIDATES:
            values[candidate] = (20.9, 0.20, 80.0, 190.0, 0.0004)
        rows = []
        for config in CONFIGS:
            journey, cv, holding, denied, action_cost = values[config]
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
                        "lower_regularity_policy_enabled": (
                            1.0 if config in CANDIDATES else 0.0),
                        "lower_regularity_policy_action_cost_mean": action_cost,
                        "lower_regularity_policy_evidence_valid_mean": 0.8,
                        "lower_regularity_lambda": 1.5,
                    })
        pd.DataFrame(rows).to_csv(root / "frozen_per_eval.csv", index=False)
        pd.DataFrame([{"config": name} for name in CONFIGS]).to_csv(
            root / "frozen_summary.csv", index=False)

        paired_rows = []
        for candidate in CANDIDATES:
            journey, cv, holding, denied, _ = values[candidate]
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

    def test_weakest_passing_dual_only_variant_is_selected_for_confirmation(self):
        with TemporaryDirectory() as tmp:
            result = evaluate_actiondual_screen(
                self._aggregate(Path(tmp) / "aggregate"))

        self.assertEqual(result["status"], "exploratory_candidate_selected")
        self.assertFalse(result["claim_eligible"])
        self.assertEqual(
            result["selected_for_confirmation"],
            "F_freqduet_protocol_v6_actiondual_c0020_hiro",
        )

    def test_dirty_source_fails_closed(self):
        with TemporaryDirectory() as tmp:
            aggregate = self._aggregate(
                Path(tmp) / "aggregate", dirty=True)
            with self.assertRaisesRegex(ValueError, "strict checks"):
                evaluate_actiondual_screen(aggregate)


if __name__ == "__main__":
    unittest.main()
