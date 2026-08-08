import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd

from scripts.analysis_provenance import (
    csv_artifact_record,
    validate_csv_artifact,
)
from scripts.run_freqduet_protocol_v2_matrix import (
    METRICS,
    V6_MECHANISM_METRICS,
    V6_SAFETY_METRICS,
    aggregate,
    config_name,
    git_provenance,
    run_dir,
    run_manifest,
)


CONFIGS = [
    "F_freqduet_protocol_v6_main_hiro",
    "F_freqduet_protocol_v6_nofreq_hiro",
]
EVAL_SEEDS = [101, 103]


def _write_run(logs: Path, config: str, train_episodes: int = 3) -> None:
    path = run_dir(logs, config, 1)
    evaluation_dir = path / "frozen_evaluation"
    evaluation_dir.mkdir(parents=True)
    rows = []
    for seed in EVAL_SEEDS:
        row = {
            "protocol_version": "freqduet-eval-v6",
            "eval_seed": seed,
            "checkpoint_ep": train_episodes - 1,
            "policy_digest": f"digest-{config_name(config)}",
            "scenario_tape_id": f"tape-{seed}",
            "lower_causal_guard_evidence_mode": (
                "pre_action_departure_v6"),
        }
        row.update({metric: 1.0 for metric in METRICS})
        row.update({metric: 1.0 for metric in V6_SAFETY_METRICS})
        row.update({metric: 1.0 for metric in V6_MECHANISM_METRICS})
        rows.append(row)
    frame = pd.DataFrame(rows)
    evaluation_path = evaluation_dir / "evaluation.csv"
    frame.to_csv(evaluation_path, index=False)
    artifact = csv_artifact_record(
        evaluation_path, frame, ["eval_seed"])
    (evaluation_dir / "evaluation_manifest.json").write_text(json.dumps({
        "manifest_version": "freqduet-evaluation-manifest-v2",
        "protocol_version": "freqduet-eval-v6",
        "config_name": config_name(config),
        "training_seed": 1,
        "checkpoint_ep": train_episodes - 1,
        "scenario_seeds": EVAL_SEEDS,
        "n_episodes": len(EVAL_SEEDS),
        "policy_digest": f"digest-{config_name(config)}",
        "artifacts": {"evaluation_csv": artifact},
    }))
    (path / "protocol_run_manifest.json").write_text(json.dumps(
        run_manifest(
            config=config,
            train_seed=1,
            eval_seeds=EVAL_SEEDS,
            train_episodes=train_episodes,
            stage="development",
        )
    ))


class ProtocolV6ProvenanceTest(unittest.TestCase):
    def test_git_provenance_falls_back_when_remote_has_no_git_binary(self):
        environment = {
            "FREQDUET_SOURCE_COMMIT": "a" * 40,
            "FREQDUET_SOURCE_BRANCH": "HEAD",
            "FREQDUET_SOURCE_TRACKED_DIRTY": "0",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            with mock.patch(
                    "scripts.run_freqduet_protocol_v2_matrix.subprocess.run",
                    side_effect=FileNotFoundError("git")):
                record = git_provenance()

        self.assertEqual(record["commit"], "a" * 40)
        self.assertEqual(record["branch"], "HEAD")
        self.assertIs(record["tracked_dirty"], False)

    def test_aggregate_locks_checkpoint_source_scenario_and_csvs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            out = root / "out"
            for config in CONFIGS:
                _write_run(logs, config)

            aggregate(
                CONFIGS,
                [1],
                EVAL_SEEDS,
                [logs],
                out,
                CONFIGS[0],
                3,
                "development",
            )

            manifest = json.loads((out / "matrix_manifest.json").read_text())
            self.assertEqual(
                manifest["manifest_version"],
                "freqduet-matrix-manifest-v2",
            )
            self.assertEqual(manifest["checkpoint_ep"], 2)
            self.assertEqual(manifest["stage"], "development")
            self.assertTrue(manifest["run_manifests_verified"])
            self.assertEqual(
                manifest["run_git_provenance"]["commit"],
                manifest["git"]["commit"],
            )
            self.assertEqual(
                manifest["launch_analysis_sha256"],
                manifest["analysis_fingerprint"]["sha256"],
            )
            validate_csv_artifact(
                out / "frozen_per_eval.csv",
                manifest["artifacts"]["frozen_per_eval.csv"],
                expected_primary_key=["config", "train_seed", "eval_seed"],
            )

    def test_tampered_evaluation_csv_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            for config in CONFIGS:
                _write_run(logs, config)
            target = (
                run_dir(logs, CONFIGS[0], 1)
                / "frozen_evaluation/evaluation.csv"
            )
            target.write_text(target.read_text().replace("1.0", "2.0", 1))

            with self.assertRaisesRegex(ValueError, "SHA256"):
                aggregate(
                    CONFIGS,
                    [1],
                    EVAL_SEEDS,
                    [logs],
                    root / "out",
                    CONFIGS[0],
                    3,
                    "development",
                )

    def test_wrong_training_length_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            for config in CONFIGS:
                _write_run(logs, config)

            with self.assertRaisesRegex(ValueError, "checkpoint episode"):
                aggregate(
                    CONFIGS,
                    [1],
                    EVAL_SEEDS,
                    [logs],
                    root / "out",
                    CONFIGS[0],
                    4,
                    "development",
                )

    def test_mixed_run_git_commits_are_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            for config in CONFIGS:
                _write_run(logs, config)
            manifest_path = (
                run_dir(logs, CONFIGS[0], 1)
                / "protocol_run_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text())
            manifest["git"]["commit"] = "b" * 40
            manifest_path.write_text(json.dumps(manifest))

            with self.assertRaisesRegex(RuntimeError, "Git provenance"):
                aggregate(
                    CONFIGS,
                    [1],
                    EVAL_SEEDS,
                    [logs],
                    root / "out",
                    CONFIGS[0],
                    3,
                    "development",
                )


if __name__ == "__main__":
    unittest.main()
