import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.audit_protocol_v6_incremental_selection import (
    evaluate_selection,
)


MAIN = "F_freqduet_protocol_v6_main_hiro"
REFERENCE = "F_freqduet_protocol_v6_noguard_hiro"
CANDIDATE = "F_freqduet_protocol_v6_avlbal_w4_hiro"
CONTEXT = "F_freqduet_protocol_v6_avlcompact_hiro"


class ProtocolV6IncrementalSelectionTest(unittest.TestCase):
    def make_aggregate(
        self,
        root: Path,
        *,
        cv_delta: float = -0.03,
        follower: float = 0.8,
        adjustment: float = 0.0,
        stage: str = "exploratory",
        context_cv: float = 0.29,
    ) -> Path:
        aggregate = root / "combined_summary"
        aggregate.mkdir()
        manifest = {
            "strict_complete": True,
            "common_random_numbers_verified": True,
            "run_manifests_verified": True,
            "stage": stage,
            "independent_confirmation": stage == "confirmation",
            "expected_rollouts": 8,
            "configs": [MAIN, REFERENCE, CONTEXT, CANDIDATE],
            "reference": REFERENCE,
            "train_seeds": [1, 2],
            "eval_seeds": [11],
            "protocol_version": "freqduet-eval-v6",
            "run_source_fingerprint": {"sha256": "a" * 64},
            "scenario_contract": {"sha256": "b" * 64},
            "launch_analysis_sha256": "c" * 64,
            "run_git_provenance": {
                "commit": "d" * 40,
                "tracked_dirty": False,
            },
        }
        (aggregate / "matrix_manifest.json").write_text(json.dumps(manifest))

        rows = []
        journey = {
            MAIN: 18.0,
            REFERENCE: 17.0,
            CONTEXT: 16.75,
            CANDIDATE: 16.8,
        }
        headway_cv = {
            MAIN: 0.20,
            REFERENCE: 0.30,
            CONTEXT: context_cv,
            CANDIDATE: 0.30 + cv_delta,
        }
        for config in (MAIN, REFERENCE, CONTEXT, CANDIDATE):
            for seed in (1, 2):
                rows.append({
                    "config": config,
                    "train_seed": seed,
                    "eval_seed": 11,
                    "lower_causal_guard_adjustment_mean_s": (
                        adjustment if config == CANDIDATE else 0.0),
                    "lower_departure_regularity_evidence_valid_mean": (
                        1.0 if config == CANDIDATE else 0.0),
                    "lower_departure_regularity_follower_valid_mean": (
                        follower if config == CANDIDATE else 0.0),
                    "lower_departure_regularity_baseline_loss_mean": (
                        0.2 if config == CANDIDATE else 0.0),
                    "lower_departure_regularity_post_loss_mean": (
                        0.1 if config == CANDIDATE else 0.0),
                    "restricted_total_journey_horizon_min": journey[config],
                    "headway_cv": headway_cv[config],
                })
        pd.DataFrame(rows).to_csv(
            aggregate / "frozen_per_eval.csv", index=False)
        pd.DataFrame([
            {
                "config": MAIN,
                "restricted_total_journey_horizon_min_mean": 18.0,
                "headway_cv_mean": 0.20,
                "holding_vehicle_seconds_mean": 100.0,
                "fleet_denied_dispatch_events_mean": 100.0,
            },
            {
                "config": REFERENCE,
                "restricted_total_journey_horizon_min_mean": 17.0,
                "headway_cv_mean": 0.30,
                "holding_vehicle_seconds_mean": 50.0,
                "fleet_denied_dispatch_events_mean": 50.0,
            },
            {
                "config": CONTEXT,
                "restricted_total_journey_horizon_min_mean": 16.75,
                "headway_cv_mean": context_cv,
                "holding_vehicle_seconds_mean": 45.0,
                "fleet_denied_dispatch_events_mean": 45.0,
            },
            {
                "config": CANDIDATE,
                "restricted_total_journey_horizon_min_mean": 16.8,
                "headway_cv_mean": 0.30 + cv_delta,
                "holding_vehicle_seconds_mean": 45.0,
                "fleet_denied_dispatch_events_mean": 45.0,
            },
        ]).to_csv(aggregate / "frozen_summary.csv", index=False)
        pd.DataFrame([{
            "candidate": CANDIDATE,
            "reference": REFERENCE,
            "n_pairs": 2,
            "delta_restricted_total_journey_horizon_min_mean": -0.2,
            "delta_restricted_total_journey_horizon_min_ci_low": -0.3,
            "delta_restricted_total_journey_horizon_min_ci_high": -0.1,
            "delta_headway_cv_mean": cv_delta,
            "delta_headway_cv_ci_low": cv_delta - 0.01,
            "delta_headway_cv_ci_high": cv_delta + 0.01,
            "delta_holding_vehicle_seconds_mean": -5.0,
            "delta_fleet_denied_dispatch_events_mean": -5.0,
        }]).to_csv(aggregate / "frozen_paired_deltas.csv", index=False)
        return aggregate

    def test_unique_candidate_passes_all_preregistered_gates(self):
        with TemporaryDirectory() as tmp:
            aggregate = self.make_aggregate(Path(tmp))
            result = evaluate_selection(
                aggregate, candidates=[CANDIDATE])
        self.assertEqual(result["status"], "unique_pass")
        self.assertEqual(result["selected_candidate"], CANDIDATE)
        self.assertTrue(result["candidate_results"][0]["passes"])

    def test_follower_coverage_fails_closed(self):
        with TemporaryDirectory() as tmp:
            aggregate = self.make_aggregate(Path(tmp), follower=0.49)
            result = evaluate_selection(
                aggregate, candidates=[CANDIDATE])
        self.assertEqual(result["status"], "no_pass")
        gates = result["candidate_results"][0]["gates"]
        self.assertFalse(gates["follower_coverage_every_rollout"])

    def test_nonzero_execution_adjustment_fails_closed(self):
        with TemporaryDirectory() as tmp:
            aggregate = self.make_aggregate(Path(tmp), adjustment=0.01)
            result = evaluate_selection(
                aggregate, candidates=[CANDIDATE])
        self.assertEqual(result["status"], "no_pass")
        gates = result["candidate_results"][0]["gates"]
        self.assertFalse(gates["zero_execution_adjustment"])

    def test_confirmation_requires_matching_stage_and_one_candidate(self):
        with TemporaryDirectory() as tmp:
            aggregate = self.make_aggregate(
                Path(tmp), stage="confirmation")
            result = evaluate_selection(
                aggregate,
                candidates=[CANDIDATE],
                expected_stage="confirmation",
            )
            with self.assertRaisesRegex(ValueError, "strict checks"):
                evaluate_selection(
                    aggregate,
                    candidates=[CANDIDATE],
                    expected_stage="exploratory",
                )
        self.assertEqual(result["audit_stage"], "confirmation")
        self.assertEqual(result["status"], "unique_pass")

    def test_confirmation_rejects_multiple_candidates(self):
        with TemporaryDirectory() as tmp:
            aggregate = self.make_aggregate(
                Path(tmp), stage="confirmation")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                evaluate_selection(
                    aggregate,
                    candidates=[CANDIDATE, REFERENCE],
                    expected_stage="confirmation",
                )

    def test_matched_context_gate_requires_incremental_cv_improvement(self):
        with TemporaryDirectory() as tmp:
            aggregate = self.make_aggregate(Path(tmp), context_cv=0.29)
            result = evaluate_selection(
                aggregate,
                candidates=[CANDIDATE],
                matched_context=CONTEXT,
            )
        self.assertEqual(result["status"], "unique_pass")
        candidate = result["candidate_results"][0]
        self.assertAlmostEqual(
            candidate["matched_context_deltas"]["headway_cv_delta"],
            -0.02,
        )

        with TemporaryDirectory() as tmp:
            aggregate = self.make_aggregate(Path(tmp), context_cv=0.27)
            result = evaluate_selection(
                aggregate,
                candidates=[CANDIDATE],
                matched_context=CONTEXT,
            )
        self.assertEqual(result["status"], "no_pass")
        self.assertFalse(result["candidate_results"][0]["gates"]
                         ["headway_cv_improves_over_matched_context"])


if __name__ == "__main__":
    unittest.main()
