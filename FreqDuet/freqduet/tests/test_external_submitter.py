import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "submit_freqduet_external_baselines_scheduleurm.py"


class ExternalSubmitterTest(unittest.TestCase):
    def test_dry_run_injects_frozen_git_provenance(self):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--configs", "F_freqduet_protocol_v6_main_hiro",
                "--variants", "fixed_headway",
                "--seeds", "45007",
                "--episodes", "1",
                "--last-k", "1",
                "--direct-scenario-seeds",
                "--run-name", "external_submitter_test",
                "--shard-size", "1",
                "--dry-run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(
            result.stdout,
            r"FREQDUET_SOURCE_COMMIT=[0-9a-f]{40}",
        )
        self.assertIn("FREQDUET_SOURCE_BRANCH=", result.stdout)
        self.assertRegex(
            result.stdout,
            r"FREQDUET_SOURCE_TRACKED_DIRTY=[01]",
        )


if __name__ == "__main__":
    unittest.main()
