import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis_provenance import csv_artifact_record, sha256_file
from scripts.decide_freqduet_protocol_v6_screen import (
    ALLOCATION_CONTROLS,
    ARTIFACT_PRIMARY_KEYS,
    CONFIGS,
    DECISION_CONTRACT,
    FREQUENCY_CONTROLS,
    MATRIX_MANIFEST_VERSION,
    MECHANISM_ABLATIONS,
    PAIRED_FILE,
    PER_EVAL_FILE,
    PRIMARY,
    PROTOCOL,
    REFERENCE,
    REFERENCE_NO_HARM_LIMITS,
    SIMPLE_CONFIG,
    SUMMARY_FILE,
    decide,
)


METRICS = list(dict.fromkeys([PRIMARY, *REFERENCE_NO_HARM_LIMITS]))


def _paired_frame() -> pd.DataFrame:
    rows = []
    for name in CONFIGS:
        if name == REFERENCE:
            continue
        if name in FREQUENCY_CONTROLS:
            mean, low, high = 0.50, 0.20, 0.80
        elif name in ALLOCATION_CONTROLS:
            mean, low, high = 0.30, 0.10, 0.50
        elif name in MECHANISM_ABLATIONS:
            mean, low, high = 0.30, 0.10, 0.50
        else:
            mean, low, high = 0.10, -0.10, 0.20
        row = {
            "candidate": name,
            "reference": REFERENCE,
            f"delta_{PRIMARY}_mean": mean,
            f"delta_{PRIMARY}_ci_low": low,
            f"delta_{PRIMARY}_ci_high": high,
        }
        for metric in REFERENCE_NO_HARM_LIMITS:
            row[f"delta_{metric}_ci_low"] = -0.001
            row[f"delta_{metric}_ci_high"] = 0.001
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_frame() -> pd.DataFrame:
    rows = []
    for config in CONFIGS:
        row = {
            "config": config,
            "n_train_seeds": 2,
            "n_eval_seeds": 2,
            "n_rollouts": 4,
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = 18.0
            row[f"{metric}_ci_low"] = 17.5
            row[f"{metric}_ci_high"] = 18.5
        rows.append(row)
    return pd.DataFrame(rows)


def _per_eval_frame(train_seeds, eval_seeds) -> pd.DataFrame:
    rows = []
    for config in CONFIGS:
        for train_seed in train_seeds:
            for eval_seed in eval_seeds:
                row = {
                    "config": config,
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "protocol_version": PROTOCOL,
                    "scenario_tape_id": f"scenario-{eval_seed}",
                    PRIMARY: 18.0,
                    "restricted_wait_horizon_min": 5.0,
                    "passenger_unserved_rate": 0.01,
                    "headway_cv": 0.10,
                    "fleet_denied_trip_rate": 0.01,
                    "fleet_readiness_delay_mean_s": 10.0,
                    "holding_passenger_min_per_generated": 0.8,
                    "trip_launch_rate": 0.99,
                    "trip_completion_rate": 0.98,
                    "upper_plan_projected_delta_sum_abs_mean_s": 0.0,
                    "lower_causal_guard_enabled": 1.0,
                    "lower_causal_guard_evidence_mode": (
                        "pre_action_departure_v6"),
                    "passengers_generated": 20000.0,
                    "passengers_unserved": 200.0,
                    "physical_vehicle_count": 12.0,
                    "fleet_capacity": 12.0,
                    "peak_fleet": 12.0,
                    "fleet_denied_dispatch_events": 30.0,
                    "fleet_denied_retry_trip_seconds": 30.0,
                    "fleet_denied_trips": 2.0,
                    "fleet_readiness_delay_max_s": 40.0,
                }
                rows.append(row)
    return pd.DataFrame(rows)


class MatrixFixture:
    def __init__(self, root: Path, *, stage="development",
                 train_seeds=None, eval_seeds=None):
        self.root = root
        self.stage = stage
        self.train_seeds = list(train_seeds or [101, 103])
        self.eval_seeds = list(eval_seeds or [1001, 1003])
        self.frames = {
            SUMMARY_FILE: _summary_frame(),
            PAIRED_FILE: _paired_frame(),
            PER_EVAL_FILE: _per_eval_frame(
                self.train_seeds, self.eval_seeds),
        }
        self.manifest_path = root / "matrix_manifest.json"
        self._write_artifacts()
        self.manifest = {
            "manifest_version": MATRIX_MANIFEST_VERSION,
            "protocol_version": PROTOCOL,
            "stage": stage,
            "independent_confirmation": stage == "confirmation",
            "strict_complete": True,
            "run_manifests_verified": True,
            "common_random_numbers_verified": True,
            "configs": list(CONFIGS),
            "reference": REFERENCE,
            "primary_metric": PRIMARY,
            "metrics": METRICS,
            "train_seeds": self.train_seeds,
            "eval_seeds": self.eval_seeds,
            "expected_rollouts": (
                len(CONFIGS) * len(self.train_seeds) * len(self.eval_seeds)),
            "artifacts": self._artifact_records(),
        }
        self.write_manifest()

    def _write_artifacts(self):
        self.root.mkdir(parents=True, exist_ok=True)
        for name, frame in self.frames.items():
            frame.to_csv(self.root / name, index=False)

    def _artifact_records(self):
        return {
            name: csv_artifact_record(
                self.root / name, frame, ARTIFACT_PRIMARY_KEYS[name])
            for name, frame in self.frames.items()
        }

    def rewrite_artifact(self, name):
        self.frames[name].to_csv(self.root / name, index=False)
        self.manifest["artifacts"][name] = csv_artifact_record(
            self.root / name,
            self.frames[name],
            ARTIFACT_PRIMARY_KEYS[name],
        )
        self.write_manifest()

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2) + "\n")


class ProtocolV6DecisionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_development_produces_candidate_and_binds_inputs(self):
        fixture = MatrixFixture(self.root / "development")

        result = decide(fixture.manifest_path, stage="development")

        self.assertEqual(result["decision_contract"], DECISION_CONTRACT)
        self.assertEqual(
            result["status"],
            "frequency_supported_simple_optimizer_candidate",
        )
        self.assertEqual(result["candidate_config"], SIMPLE_CONFIG)
        self.assertNotIn("selected_config", result)
        self.assertNotIn("final", json.dumps(result).lower())
        self.assertEqual(
            result["matrix_manifest"]["sha256"],
            sha256_file(fixture.manifest_path),
        )
        for name, key in ARTIFACT_PRIMARY_KEYS.items():
            binding = result["input_artifacts"][name]
            self.assertEqual(binding["primary_key"], list(key))
            self.assertEqual(
                binding["sha256"], sha256_file(fixture.root / name))

    def test_independent_confirmation_selects_development_candidate(self):
        development = MatrixFixture(self.root / "development")
        development_result = decide(
            development.manifest_path, stage="development")
        development_path = self.root / "development_decision.json"
        development_path.write_text(json.dumps(development_result) + "\n")
        confirmation = MatrixFixture(
            self.root / "confirmation",
            stage="confirmation",
            train_seeds=[211, 223],
            eval_seeds=[2003, 2011],
        )

        result = decide(
            confirmation.manifest_path,
            stage="confirmation",
            development_manifest=development_path,
        )

        self.assertEqual(result["status"], "confirmation_supported")
        self.assertEqual(result["selected_config"], SIMPLE_CONFIG)
        self.assertNotIn("candidate_config", result)
        self.assertEqual(
            result["development_decision"]["sha256"],
            sha256_file(development_path),
        )

    def test_rejects_tampered_csv(self):
        fixture = MatrixFixture(self.root / "development")
        with (fixture.root / SUMMARY_FILE).open("a") as handle:
            handle.write("tampered\n")

        with self.assertRaisesRegex(ValueError, "SHA256"):
            decide(fixture.manifest_path, stage="development")

    def test_rejects_command_and_manifest_stage_mismatch(self):
        fixture = MatrixFixture(self.root / "development")

        with self.assertRaisesRegex(ValueError, "does not match command stage"):
            decide(
                fixture.manifest_path,
                stage="confirmation",
                development_manifest=self.root / "unused.json",
            )

    def test_rejects_confirmation_without_independent_marker(self):
        development = MatrixFixture(self.root / "development")
        development_result = decide(
            development.manifest_path, stage="development")
        development_path = self.root / "development_decision.json"
        development_path.write_text(json.dumps(development_result) + "\n")
        confirmation = MatrixFixture(
            self.root / "confirmation",
            stage="confirmation",
            train_seeds=[211],
            eval_seeds=[2003],
        )
        confirmation.manifest["independent_confirmation"] = False
        confirmation.write_manifest()

        with self.assertRaisesRegex(ValueError, "independent_confirmation"):
            decide(
                confirmation.manifest_path,
                stage="confirmation",
                development_manifest=development_path,
            )

    def test_rejects_any_seed_overlap_with_development(self):
        development = MatrixFixture(self.root / "development")
        development_result = decide(
            development.manifest_path, stage="development")
        development_path = self.root / "development_decision.json"
        development_path.write_text(json.dumps(development_result) + "\n")
        confirmation = MatrixFixture(
            self.root / "confirmation",
            stage="confirmation",
            train_seeds=[211],
            eval_seeds=[101],
        )

        with self.assertRaisesRegex(ValueError, "overlap development seeds"):
            decide(
                confirmation.manifest_path,
                stage="confirmation",
                development_manifest=development_path,
            )

    def test_rejects_development_manifest_with_selection(self):
        development = MatrixFixture(self.root / "development")
        development_result = decide(
            development.manifest_path, stage="development")
        development_result["selected_config"] = SIMPLE_CONFIG
        development_path = self.root / "invalid_development_decision.json"
        development_path.write_text(json.dumps(development_result) + "\n")
        confirmation = MatrixFixture(
            self.root / "confirmation",
            stage="confirmation",
            train_seeds=[211],
            eval_seeds=[2003],
        )

        with self.assertRaisesRegex(ValueError, "illegally contains"):
            decide(
                confirmation.manifest_path,
                stage="confirmation",
                development_manifest=development_path,
            )

    def test_rejects_tampered_development_candidate(self):
        development = MatrixFixture(self.root / "development")
        development_result = decide(
            development.manifest_path, stage="development")
        development_result["candidate_config"] = REFERENCE
        development_path = self.root / "tampered_development_decision.json"
        development_path.write_text(json.dumps(development_result) + "\n")
        confirmation = MatrixFixture(
            self.root / "confirmation",
            stage="confirmation",
            train_seeds=[211],
            eval_seeds=[2003],
        )

        with self.assertRaisesRegex(ValueError, "inconsistent"):
            decide(
                confirmation.manifest_path,
                stage="confirmation",
                development_manifest=development_path,
            )

    def test_rejects_missing_locked_config(self):
        fixture = MatrixFixture(self.root / "development")
        fixture.manifest["configs"] = list(CONFIGS[:-1])
        fixture.write_manifest()

        with self.assertRaisesRegex(ValueError, "locked V6 configs"):
            decide(fixture.manifest_path, stage="development")

    def test_each_main_rollout_invariant_is_fail_closed(self):
        cases = {
            "budget": (
                "upper_plan_projected_delta_sum_abs_mean_s", 0.01),
            "guard": ("lower_causal_guard_enabled", 0.0),
            "evidence": (
                "lower_causal_guard_evidence_mode", "arrival_event_v5"),
            "passengers": ("passengers_generated", np.nan),
            "fleet": ("fleet_capacity", np.inf),
            "metric": (PRIMARY, np.nan),
        }
        for label, (column, value) in cases.items():
            with self.subTest(label=label):
                fixture = MatrixFixture(self.root / label)
                frame = fixture.frames[PER_EVAL_FILE]
                index = frame.index[frame["config"].eq(REFERENCE)][0]
                frame.loc[index, column] = value
                fixture.rewrite_artifact(PER_EVAL_FILE)

                result = decide(fixture.manifest_path, stage="development")

                self.assertEqual(
                    result["status"], "implementation_contract_failed")
                self.assertIsNone(result["candidate_config"])
                self.assertFalse(result["evidence"]["main_invariant_pass"])

    def test_requires_confirmation_development_manifest(self):
        fixture = MatrixFixture(
            self.root / "confirmation",
            stage="confirmation",
            train_seeds=[211],
            eval_seeds=[2003],
        )

        with self.assertRaisesRegex(ValueError, "requires"):
            decide(fixture.manifest_path, stage="confirmation")


if __name__ == "__main__":
    unittest.main()
