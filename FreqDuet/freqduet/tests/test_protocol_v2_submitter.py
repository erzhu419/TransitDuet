import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "submit_freqduet_protocol_v2_scheduleurm.py"
REFERENCE = "F_freqduet_protocol_v2_uppercompact_disc_hist_hiro"
CANDIDATE = "F_freqduet_protocol_v2_uppercompact_disc_hist_physnorm_hiro"


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

    def test_reference_must_be_in_config_matrix(self):
        result = self.run_submitter("--reference", "missing_config")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "reference config must be included in --configs", result.stderr)


if __name__ == "__main__":
    unittest.main()
