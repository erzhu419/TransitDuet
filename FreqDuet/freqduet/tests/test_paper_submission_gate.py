import copy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import pandas as pd
import yaml

from scripts import build_freqduet_paper_package
from scripts import curate_freqduet_paper_panels
from scripts.analysis_provenance import csv_artifact_record
from scripts.decide_freqduet_protocol_v6_screen import (
    ARTIFACT_PRIMARY_KEYS,
    CONFIGS,
    REFERENCE,
)
from scripts.paper_submission_gate import (
    ACTIVE_HOLD_STATUS,
    ACTIVE_PROTOCOL,
    ARTIFACT_BINDING_FIELDS,
    CONFIRMATION_DECISION_STATUS,
    DECISION_CONTRACT,
    DECISION_CONTRACT_SHA256,
    MATRIX_MANIFEST_VERSION,
    PRIMARY_METRIC,
    READY_STATUS,
    REQUIRED_MATRIX_ARTIFACTS,
    require_no_missing_artifacts,
    require_submission_ready,
)
from scripts.run_freqduet_protocol_v2_matrix import (
    analysis_fingerprint,
    scenario_contract,
    source_fingerprint,
)


SOURCE_COMMIT = "a" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReadyFixture:
    def __init__(self, root: Path):
        self.root = root
        frames = {
            "frozen_per_eval.csv": pd.DataFrame([{
                "config": REFERENCE, "train_seed": 211,
                "eval_seed": 2003, "value": 1.0,
            }]),
            "frozen_summary.csv": pd.DataFrame([{
                "config": REFERENCE, "value": 1.0,
            }]),
            "frozen_paired_deltas.csv": pd.DataFrame([{
                "candidate": CONFIGS[1], "reference": REFERENCE,
                "value": 1.0,
            }]),
        }
        for filename, frame in frames.items():
            frame.to_csv(root / filename, index=False)
        artifact_records = {
            filename: csv_artifact_record(
                root / filename,
                frame,
                ARTIFACT_PRIMARY_KEYS[filename],
            )
            for filename, frame in frames.items()
        }

        self.matrix_path = root / "matrix_manifest.json"
        analysis_record = analysis_fingerprint()
        self.matrix = {
            "manifest_version": MATRIX_MANIFEST_VERSION,
            "protocol_version": ACTIVE_PROTOCOL,
            "stage": "confirmation",
            "strict_complete": True,
            "run_manifests_verified": True,
            "common_random_numbers_verified": True,
            "independent_confirmation": True,
            "configs": list(CONFIGS),
            "reference": REFERENCE,
            "primary_metric": PRIMARY_METRIC,
            "train_seeds": [211],
            "eval_seeds": [2003],
            "train_episodes": 200,
            "checkpoint_ep": 199,
            "expected_rollouts": len(CONFIGS),
            "run_source_fingerprint": source_fingerprint(),
            "scenario_contract": scenario_contract(REFERENCE),
            "launch_analysis_sha256": analysis_record["sha256"],
            "analysis_fingerprint": analysis_record,
            "run_git_provenance": {
                "commit": SOURCE_COMMIT,
                "tracked_dirty": False,
            },
            "git": {"commit": SOURCE_COMMIT, "tracked_dirty": False},
            "artifacts": artifact_records,
        }
        self._write_matrix()

        self.decision_path = root / "confirmation_decision.json"
        self.decision = {
            "decision_contract": DECISION_CONTRACT,
            "decision_contract_sha256": DECISION_CONTRACT_SHA256,
            "status": CONFIRMATION_DECISION_STATUS,
            "protocol": ACTIVE_PROTOCOL,
            "stage": "confirmation",
            "primary_metric": PRIMARY_METRIC,
            "matrix_manifest": {
                "sha256": _sha256(self.matrix_path),
                "manifest_version": MATRIX_MANIFEST_VERSION,
            },
            "input_artifacts": {
                filename: {
                    key: record[key] for key in ARTIFACT_BINDING_FIELDS
                }
                for filename, record in artifact_records.items()
            },
            "selected_config": "F_freqduet_protocol_v6_main_hiro",
        }
        self._write_decision()

        self.manifest = {
            "version": "2026-08-08-protocol-v6-confirmation",
            "submission_status": READY_STATUS,
            "active_protocol": ACTIVE_PROTOCOL,
            "active_source_commit": SOURCE_COMMIT,
            "active_metrics": {"primary": PRIMARY_METRIC},
            "confirmation": {
                "status": "confirmed",
                "stage": "confirmation",
                "source_commit": SOURCE_COMMIT,
                "decision": {
                    "path": self.decision_path.name,
                    "sha256": _sha256(self.decision_path),
                },
                "source_manifest": {
                    "path": self.matrix_path.name,
                    "sha256": _sha256(self.matrix_path),
                },
            },
        }
        self.manifest_path = root / "paper_manifest.yaml"
        self._write_manifest()

    def _write_matrix(self) -> None:
        self.matrix_path.write_text(
            json.dumps(self.matrix, indent=2, sort_keys=True) + "\n"
        )

    def _write_decision(self) -> None:
        self.decision_path.write_text(
            json.dumps(self.decision, indent=2, sort_keys=True) + "\n"
        )

    def _write_manifest(self) -> None:
        self.manifest_path.write_text(
            yaml.safe_dump(self.manifest, sort_keys=False)
        )

    def rewrite_decision_and_bind(self) -> None:
        self._write_decision()
        self.manifest["confirmation"]["decision"]["sha256"] = _sha256(
            self.decision_path
        )
        self._write_manifest()

    def rewrite_matrix_and_bind(self) -> None:
        self._write_matrix()
        matrix_sha = _sha256(self.matrix_path)
        self.decision["matrix_manifest"]["sha256"] = matrix_sha
        self.manifest["confirmation"]["source_manifest"]["sha256"] = (
            matrix_sha
        )
        self.rewrite_decision_and_bind()


class PaperSubmissionGateTest(unittest.TestCase):
    def test_hold_blocks_default_generation(self):
        with self.assertRaisesRegex(RuntimeError, "submission_status"):
            require_submission_ready({
                "version": "2026-06-26-historical",
                "submission_status": "hold_pending_protocol_v5",
                "active_protocol": "freqduet-eval-v5",
            })

    def test_historical_override_requires_explicit_historical_manifest(self):
        require_submission_ready(
            {
                "version": "2026-06-26-historical",
                "submission_status": "hold_pending_protocol_v5",
                "active_protocol": "freqduet-eval-v5",
            },
            allow_historical=True,
        )
        with self.assertRaisesRegex(RuntimeError, "explicitly historical"):
            require_submission_ready(
                {
                    "version": "2026-08-08-protocol-v6",
                    "submission_status": "hold_pending_protocol_v5",
                    "active_protocol": "freqduet-eval-v5",
                },
                allow_historical=True,
            )
        with self.assertRaisesRegex(RuntimeError, "locked historical hold"):
            require_submission_ready(
                {
                    "version": "2026-06-26-historical",
                    "submission_status": "hold_unknown",
                },
                allow_historical=True,
            )

    def test_active_v6_hold_preserves_explicit_historical_reproduction(self):
        manifest = {
            "version": "2026-08-08-protocol-v6-development",
            "submission_status": ACTIVE_HOLD_STATUS,
            "active_protocol": ACTIVE_PROTOCOL,
            "historical_reproduction": {
                "version": "2026-06-26-historical",
                "submission_status": "hold_pending_protocol_v5",
                "active_protocol": "freqduet-eval-v5",
            },
        }
        require_submission_ready(manifest, allow_historical=True)
        with self.assertRaisesRegex(RuntimeError, "submission_status"):
            require_submission_ready(manifest)

        broken = copy.deepcopy(manifest)
        del broken["historical_reproduction"]
        with self.assertRaisesRegex(RuntimeError, "historical_reproduction"):
            require_submission_ready(broken, allow_historical=True)

    def test_confirmed_v6_manifest_is_accepted(self):
        with TemporaryDirectory() as tmp:
            fixture = ReadyFixture(Path(tmp))
            require_submission_ready(
                fixture.manifest, manifest_path=fixture.manifest_path
            )

    def test_old_v5_and_legacy_ready_statuses_fail_closed(self):
        for manifest in (
            {},
            {"submission_status": "ready"},
            {"submission_status": "pending"},
            {"submission_status": "ready_protocol_v5"},
            {"submission_status": "ready_protocol_v5_confirmed"},
        ):
            with self.subTest(manifest=manifest):
                with self.assertRaisesRegex(RuntimeError, "fail-closed"):
                    require_submission_ready(manifest, allow_historical=True)

    def test_ready_manifest_requires_v6_protocol_and_nonhistorical_version(self):
        with TemporaryDirectory() as tmp:
            fixture = ReadyFixture(Path(tmp))
            cases = (
                ("active_protocol", "freqduet-eval-v5", "active_protocol"),
                ("version", "2026-06-26-historical", "non-historical"),
            )
            for key, value, error in cases:
                broken = copy.deepcopy(fixture.manifest)
                broken[key] = value
                with self.subTest(key=key):
                    with self.assertRaisesRegex(RuntimeError, error):
                        require_submission_ready(
                            broken, manifest_path=fixture.manifest_path
                        )

            broken = copy.deepcopy(fixture.manifest)
            broken["active_metrics"] = {"primary": "composite"}
            with self.assertRaisesRegex(RuntimeError, "primary metric"):
                require_submission_ready(
                    broken, manifest_path=fixture.manifest_path
                )

    def test_decision_file_tampering_is_rejected(self):
        with TemporaryDirectory() as tmp:
            fixture = ReadyFixture(Path(tmp))
            fixture.decision_path.write_text(
                fixture.decision_path.read_text() + "\n"
            )
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                require_submission_ready(
                    fixture.manifest, manifest_path=fixture.manifest_path
                )

    def test_confirmation_decision_contract_and_hash_are_locked(self):
        cases = (
            (
                "decision_contract",
                "freqduet-protocol-v5-confirmation-v1",
                "decision_contract",
            ),
            ("decision_contract_sha256", "c" * 64, "contract_sha256"),
        )
        for key, value, error in cases:
            with self.subTest(key=key), TemporaryDirectory() as tmp:
                fixture = ReadyFixture(Path(tmp))
                fixture.decision[key] = value
                fixture.rewrite_decision_and_bind()
                with self.assertRaisesRegex(RuntimeError, error):
                    require_submission_ready(
                        fixture.manifest, manifest_path=fixture.manifest_path
                    )

    def test_matrix_file_tampering_is_rejected(self):
        with TemporaryDirectory() as tmp:
            fixture = ReadyFixture(Path(tmp))
            fixture.matrix_path.write_text(fixture.matrix_path.read_text() + "\n")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                require_submission_ready(
                    fixture.manifest, manifest_path=fixture.manifest_path
                )

    def test_decision_must_bind_confirmation_source_manifest_sha(self):
        with TemporaryDirectory() as tmp:
            fixture = ReadyFixture(Path(tmp))
            fixture.matrix["audit_note"] = "changed after decision"
            fixture._write_matrix()
            fixture.manifest["confirmation"]["source_manifest"]["sha256"] = (
                _sha256(fixture.matrix_path)
            )
            fixture._write_manifest()

            with self.assertRaisesRegex(
                RuntimeError, "matrix_manifest_sha256"
            ):
                require_submission_ready(
                    fixture.manifest, manifest_path=fixture.manifest_path
                )

    def test_development_or_candidate_decision_is_never_ready(self):
        with TemporaryDirectory() as tmp:
            fixture = ReadyFixture(Path(tmp))
            fixture.decision["stage"] = "development"
            fixture.decision["candidate_config"] = fixture.decision.pop(
                "selected_config"
            )
            fixture.rewrite_decision_and_bind()
            with self.assertRaisesRegex(RuntimeError, "stage"):
                require_submission_ready(
                    fixture.manifest, manifest_path=fixture.manifest_path
                )

        with TemporaryDirectory() as tmp:
            fixture = ReadyFixture(Path(tmp))
            fixture.decision["candidate_config"] = (
                "F_freqduet_protocol_v6_main_hiro"
            )
            fixture.rewrite_decision_and_bind()
            with self.assertRaisesRegex(RuntimeError, "never submission-ready"):
                require_submission_ready(
                    fixture.manifest, manifest_path=fixture.manifest_path
                )

    def test_development_matrix_is_rejected(self):
        with TemporaryDirectory() as tmp:
            fixture = ReadyFixture(Path(tmp))
            fixture.matrix["stage"] = "development"
            fixture.matrix["independent_confirmation"] = False
            fixture.rewrite_matrix_and_bind()
            with self.assertRaisesRegex(RuntimeError, "matrix stage"):
                require_submission_ready(
                    fixture.manifest, manifest_path=fixture.manifest_path
                )

    def test_matrix_requires_v2_completion_and_crn_contract(self):
        cases = (
            ("manifest_version", "freqduet-matrix-manifest-v1", "version"),
            ("protocol_version", "freqduet-eval-v5", "protocol"),
            ("strict_complete", False, "strict_complete"),
            ("run_manifests_verified", False, "run_manifests_verified"),
            ("common_random_numbers_verified", False,
             "common_random_numbers_verified"),
            ("independent_confirmation", False, "independent_confirmation"),
        )
        for key, value, error in cases:
            with self.subTest(key=key), TemporaryDirectory() as tmp:
                fixture = ReadyFixture(Path(tmp))
                fixture.matrix[key] = value
                fixture.rewrite_matrix_and_bind()
                with self.assertRaisesRegex(RuntimeError, error):
                    require_submission_ready(
                        fixture.manifest, manifest_path=fixture.manifest_path
                    )

    def test_matrix_requires_all_three_artifacts_with_valid_sha(self):
        with TemporaryDirectory() as tmp:
            fixture = ReadyFixture(Path(tmp))
            del fixture.matrix["artifacts"]["frozen_summary.csv"]
            fixture.rewrite_matrix_and_bind()
            with self.assertRaisesRegex(RuntimeError, "frozen_summary.csv"):
                require_submission_ready(
                    fixture.manifest, manifest_path=fixture.manifest_path
                )

        with TemporaryDirectory() as tmp:
            fixture = ReadyFixture(Path(tmp))
            artifact = Path(tmp) / "frozen_per_eval.csv"
            artifact.write_text(artifact.read_text() + "tampered,2\n")
            with self.assertRaisesRegex(RuntimeError, "SHA256 does not match"):
                require_submission_ready(
                    fixture.manifest, manifest_path=fixture.manifest_path
                )

    def test_confirmation_source_commit_is_bound(self):
        with TemporaryDirectory() as tmp:
            fixture = ReadyFixture(Path(tmp))
            broken = copy.deepcopy(fixture.manifest)
            broken["confirmation"]["source_commit"] = "c" * 40
            with self.assertRaisesRegex(RuntimeError, "source_commit"):
                require_submission_ready(
                    broken, manifest_path=fixture.manifest_path
                )

    def test_matrix_source_fingerprints_and_run_git_are_bound(self):
        with TemporaryDirectory() as tmp:
            fixture = ReadyFixture(Path(tmp))
            fixture.matrix["run_source_fingerprint"]["sha256"] = "e" * 64
            fixture.rewrite_matrix_and_bind()
            with self.assertRaisesRegex(RuntimeError, "locked source"):
                require_submission_ready(
                    fixture.manifest, manifest_path=fixture.manifest_path
                )

        with TemporaryDirectory() as tmp:
            fixture = ReadyFixture(Path(tmp))
            fixture.matrix["launch_analysis_sha256"] = "e" * 64
            fixture.rewrite_matrix_and_bind()
            with self.assertRaisesRegex(RuntimeError, "launch analysis"):
                require_submission_ready(
                    fixture.manifest, manifest_path=fixture.manifest_path
                )

        with TemporaryDirectory() as tmp:
            fixture = ReadyFixture(Path(tmp))
            fixture.matrix["run_git_provenance"]["tracked_dirty"] = True
            fixture.rewrite_matrix_and_bind()
            with self.assertRaisesRegex(RuntimeError, "tracked-dirty"):
                require_submission_ready(
                    fixture.manifest, manifest_path=fixture.manifest_path
                )

    def test_missing_required_artifacts_fail(self):
        with self.assertRaisesRegex(RuntimeError, "missing 2 required"):
            require_no_missing_artifacts(["table.csv", "figure.png"])

    def test_package_builder_fails_when_required_artifact_is_missing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = ReadyFixture(root)
            out_dir = root / "package"

            def mark_missing(*args):
                args[-1].append("required-table.csv")

            with mock.patch.object(
                sys,
                "argv",
                [
                    "build_freqduet_paper_package.py",
                    "--manifest",
                    str(fixture.manifest_path),
                    "--out-dir",
                    str(out_dir),
                ],
            ), mock.patch.object(
                build_freqduet_paper_package,
                "copy_core_tables",
                side_effect=mark_missing,
            ), mock.patch.object(
                build_freqduet_paper_package, "copy_figures"
            ), mock.patch.object(
                build_freqduet_paper_package, "copy_manuscript_notes"
            ), mock.patch.object(
                build_freqduet_paper_package, "copy_config_snapshots"
            ), mock.patch.object(
                build_freqduet_paper_package, "copy_data_sources"
            ), mock.patch.object(
                build_freqduet_paper_package, "copy_paper_scripts"
            ), mock.patch.object(
                build_freqduet_paper_package, "copy_paper_curation"
            ), mock.patch.object(
                build_freqduet_paper_package, "build_negative_appendix"
            ):
                with self.assertRaisesRegex(RuntimeError, "required artifact"):
                    build_freqduet_paper_package.main()
            self.assertFalse((out_dir / "package_manifest.json").exists())

    def test_panel_curator_fails_when_required_artifact_is_missing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = ReadyFixture(root)
            out_dir = root / "curation"

            def missing_manifest(specs, destination, missing):
                missing.append("required-panel.png")
                return object()

            with mock.patch.object(
                sys,
                "argv",
                [
                    "curate_freqduet_paper_panels.py",
                    "--manifest",
                    str(fixture.manifest_path),
                    "--out-dir",
                    str(out_dir),
                ],
            ), mock.patch.object(
                curate_freqduet_paper_panels,
                "build_manifest",
                side_effect=missing_manifest,
            ):
                with self.assertRaisesRegex(RuntimeError, "required artifact"):
                    curate_freqduet_paper_panels.main()
            self.assertFalse(
                (out_dir / "paper_curation_manifest.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
