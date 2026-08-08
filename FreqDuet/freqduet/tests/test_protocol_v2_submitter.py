import importlib.util
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "submit_freqduet_protocol_v2_scheduleurm.py"
REFERENCE = "F_freqduet_protocol_v2_uppercompact_disc_hist_hiro"
CANDIDATE = "F_freqduet_protocol_v2_uppercompact_disc_hist_physnorm_hiro"

SPEC = importlib.util.spec_from_file_location("freqduet_submitter", SCRIPT)
SUBMITTER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUBMITTER)


class ProtocolV2SubmitterTest(unittest.TestCase):
    def run_submitter(self, *extra_args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--configs", f"{REFERENCE},{CANDIDATE}",
                "--train-seeds", "7",
                "--eval-seeds", "10001",
                "--run-name", "protocol_v2_submitter_test",
                "--shard-size", "1",
                "--dry-run",
                *extra_args,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )

    def test_first_config_is_forwarded_as_default_reference(self):
        result = self.run_submitter()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"--reference {REFERENCE}", result.stdout)
        self.assertIn("--allow-no-ckpt", result.stdout)
        self.assertIn("--allow-no-resume", result.stdout)
        self.assertIn("FREQDUET_SOURCE_COMMIT=", result.stdout)
        self.assertRegex(
            result.stdout, r"FREQDUET_SOURCE_TRACKED_DIRTY=[01]")

    def test_reference_must_be_in_config_matrix(self):
        result = self.run_submitter("--reference", "missing_config")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "reference config must be included in --configs", result.stderr)

    def test_matrix_stage_is_forwarded_to_every_shard(self):
        result = self.run_submitter("--stage", "confirmation")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--stage confirmation", result.stdout)
        self.assertIn("--train-episodes 40", result.stdout)

    def test_summary_result_sync_excludes_training_logs_and_checkpoints(self):
        result = self.run_submitter("--result-sync", "summary")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "--result-dir "
            + str(SUBMITTER.REMOTE_ROOT)
            + "/results_freqduet/protocol_v2_submitter_test/"
            "shard_summaries/shard_0000_0001",
            result.stdout,
        )
        submit_lines = [
            line for line in result.stdout.splitlines()
            if "scheduler.py submit" in line
        ]
        self.assertTrue(submit_lines)
        self.assertTrue(all(
            "/logs_shards/" not in line.split("--result-dir", 1)[-1]
            for line in submit_lines
        ))

    def test_none_result_sync_omits_scheduler_pullback(self):
        result = self.run_submitter("--result-sync", "none")
        self.assertEqual(result.returncode, 0, result.stderr)
        submit_lines = [
            line for line in result.stdout.splitlines()
            if "scheduler.py submit" in line
        ]
        self.assertTrue(submit_lines)
        self.assertTrue(all("--result-dir" not in line for line in submit_lines))

    def test_defaults_select_the_locked_v6_development_matrix(self):
        self.assertEqual(len(SUBMITTER.DEFAULT_CONFIGS), 12)
        self.assertEqual(
            SUBMITTER.DEFAULT_CONFIGS[0],
            "F_freqduet_protocol_v6_main_hiro",
        )
        self.assertEqual(SUBMITTER.DEFAULT_TRAIN_SEEDS, [503, 521, 541, 557])
        self.assertEqual(
            SUBMITTER.DEFAULT_EVAL_SEEDS, [41011, 41017, 41023, 41039])

    def test_v6_rejects_overlapping_train_and_eval_seeds(self):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--configs", "F_freqduet_protocol_v6_main_hiro",
                "--train-seeds", "503",
                "--eval-seeds", "503",
                "--run-name", "protocol_v6_overlap_test",
                "--dry-run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "V6 train and evaluation seed sets must be disjoint",
            result.stderr,
        )

    def test_protocol_label_follows_submitted_configs(self):
        config = "F_freqduet_protocol_v3_compact_b30_hiro"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--configs", config,
                "--train-seeds", "7",
                "--eval-seeds", "10001",
                "--run-name", "protocol_v3_submitter_test",
                "--shard-size", "1",
                "--dry-run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("protocol-v3 run=protocol_v3_submitter_test", result.stdout)
        self.assertIn("FreqDuet protocol-v3 protocol_v3_submitter_test", result.stdout)

    def test_default_submission_uses_one_trusted_bulk_transaction(self):
        captured = {}

        def fake_run(command, **kwargs):
            if command[:2] == ["git", "rev-parse"]:
                return subprocess.CompletedProcess(
                    command, 0, stdout="a" * 40 + "\n", stderr="")
            if command[:2] == ["git", "status"]:
                return subprocess.CompletedProcess(
                    command, 0, stdout="", stderr="")
            self.assertIn("submit-jsonl", command)
            specs = json.loads(kwargs["input"])
            captured["command"] = command
            captured["specs"] = specs
            payload = {
                "count": len(specs),
                "submitted": [
                    {"id": f"t{index + 1}"}
                    for index in range(len(specs))
                ],
            }
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps(payload), stderr="")

        argv = [
            str(SCRIPT),
            "--configs", f"{REFERENCE},{CANDIDATE}",
            "--train-seeds", "7",
            "--eval-seeds", "10001",
            "--run-name", "protocol_bulk_submitter_test",
            "--shard-size", "1",
        ]
        with patch.object(SUBMITTER.subprocess, "run", side_effect=fake_run):
            with patch.object(sys, "argv", argv):
                with redirect_stdout(io.StringIO()):
                    SUBMITTER.main()

        self.assertEqual(len(captured["specs"]), 2)
        self.assertIn("--trusted", captured["command"])
        self.assertEqual(
            [spec["require_node"] for spec in captured["specs"]],
            ["node001", "node002"],
        )
        self.assertTrue(all(
            spec["skip_resume_scan"] for spec in captured["specs"]
        ))

    def test_bulk_summary_sync_targets_only_shard_summaries(self):
        captured = {}

        def fake_run(command, **kwargs):
            if command[:2] == ["git", "rev-parse"]:
                return subprocess.CompletedProcess(
                    command, 0, stdout="b" * 40 + "\n", stderr="")
            if command[:2] == ["git", "status"]:
                return subprocess.CompletedProcess(
                    command, 0, stdout="", stderr="")
            specs = json.loads(kwargs["input"])
            captured["specs"] = specs
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({
                    "count": len(specs),
                    "submitted": [
                        {"id": f"t{index + 1}"}
                        for index in range(len(specs))
                    ],
                }),
                stderr="",
            )

        argv = [
            str(SCRIPT),
            "--configs", f"{REFERENCE},{CANDIDATE}",
            "--train-seeds", "7",
            "--eval-seeds", "10001",
            "--run-name", "protocol_bulk_summary_test",
            "--shard-size", "1",
            "--result-sync", "summary",
        ]
        with patch.object(SUBMITTER.subprocess, "run", side_effect=fake_run):
            with patch.object(sys, "argv", argv):
                with redirect_stdout(io.StringIO()):
                    SUBMITTER.main()

        self.assertTrue(all(
            "/shard_summaries/" in spec["result_dir"]
            for spec in captured["specs"]
        ))
        self.assertTrue(all(
            "/logs_shards/" not in spec["result_dir"]
            for spec in captured["specs"]
        ))


if __name__ == "__main__":
    unittest.main()
