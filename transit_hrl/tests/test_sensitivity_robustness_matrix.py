import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.trading.sensitivity_robustness_matrix import (
    run_sensitivity_robustness_matrix,
    write_outputs,
)


class SensitivityRobustnessMatrixTest(unittest.TestCase):
    def test_runner_emits_profile_checks(self):
        payload = run_sensitivity_robustness_matrix(
            scenarios=["persistent_shift"],
            profiles=["default", "plan_curve"],
            train_seeds=[42],
            eval_seeds=[123],
            steps=24,
            assets=2,
            iterations=1,
            optimizer_seed=7,
            min_pairs=1,
        )
        self.assertEqual(payload["summary"]["rows"], 2)
        self.assertEqual(payload["summary"]["robustness_check_count"], 4)
        self.assertTrue(any(row["profile"] == "plan_curve" for row in payload["per_seed"]))
        self.assertTrue(all("noninferiority_margin" in row for row in payload["paired_checks"]))

    def test_write_outputs_creates_artifacts(self):
        payload = run_sensitivity_robustness_matrix(
            scenarios=["persistent_shift"],
            profiles=["default", "leakage_reward"],
            train_seeds=[42],
            eval_seeds=[123],
            steps=20,
            assets=2,
            iterations=1,
            optimizer_seed=7,
            min_pairs=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_outputs(out, payload)
            self.assertTrue((out / "per_seed.csv").exists())
            self.assertTrue((out / "paired_checks.csv").exists())
            self.assertTrue((out / "profile_summary.csv").exists())
            self.assertTrue((out / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
