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

    def test_reference_must_be_in_config_matrix(self):
        result = self.run_submitter("--reference", "missing_config")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "reference config must be included in --configs", result.stderr)

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


if __name__ == "__main__":
    unittest.main()
