from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from freq_hrl.experiments.plan_validity_boundary_credit import boundary_state_report
from scripts.analyze_plan_validity_boundary_variance import analyze, readout
from test_plan_validity_boundary_variance import row


def fixture():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "protocols/plan_validity_boundary_variance_2026-11-11.json").read_text())
    records = []
    for cell in manifest["cells"]:
        for shard in range(2):
            seeds = cell["development_prefix_seeds"][4 * shard:4 * (shard + 1)]
            records.append({"run_name": manifest["run_name"], "task": cell["task"],
                "optimizer_seed": cell["optimizer_seed"], "shard_index": shard,
                "status": "complete", "stage": "boundary_variance_complete", "sampled_seeds": seeds,
                "state_reports": [boundary_state_report(row(s, t)) for s in seeds for t in (50, 100, 150)],
                "new_environment_steps": manifest["per_job_environment_steps"],
                "measured_cost": {"counts": {"environment_steps": manifest["per_job_environment_steps"]},
                    "cpu_seconds": 10, "wall_seconds": 11}, "peak_process_rss_mb": 400,
                "confirmation_released": False, "performance_claim_supported": False})
    return manifest, records


class BoundaryVarianceReadoutTest(unittest.TestCase):
    def test_rosters_cost_and_no_work(self):
        manifest, records = fixture()
        result = analyze(manifest, records)
        self.assertEqual(result["measured_counts"]["environment_steps"], 3955200)
        self.assertEqual(result["acquisition_cpu_seconds"], 80)
        self.assertEqual(sum(c["summary"]["states"] for c in result["cells"]), 96)
        self.assertEqual(sum(len(c["prefix_reports"]) for c in result["cells"]), 32)
        self.assertEqual(result["new_model_fits"], 0)
        self.assertFalse(result["performance_claim_supported"])

    def test_missing_duplicate_wrong_state_and_fit_are_rejected(self):
        manifest, original = fixture()
        variants = [original[:-1], original + original[:1]]
        for field, value in (("status", "running"), ("sampled_seeds", [1]), ("new_environment_steps", 1)):
            changed = deepcopy(original)
            changed[0][field] = value
            variants.append(changed)
        changed = deepcopy(original)
        changed[0]["state_reports"][0]["seed"] = 1
        variants.append(changed)
        changed = deepcopy(original)
        changed[0]["measured_cost"]["counts"]["ridge_fit_calls"] = 1
        variants.append(changed)
        for records in variants:
            with self.subTest(records=len(records)), self.assertRaises(ValueError):
                analyze(manifest, records)

    def test_merges_state_dispersion_instead_of_averaging_shard_dispersion(self):
        reports = [boundary_state_report(row(s)) for s in range(4)]
        for r in reports[2:]:
            r["targets"]["W"]["mean"] += 100
        result = readout(reports)["targets"]["W"]
        self.assertAlmostEqual(result["across_fixed_state_mean_variance"], np.var([2.5, 2.5, 102.5, 102.5], ddof=1))
        self.assertAlmostEqual(result["nested_over_independent_variance_projection"], 4)
        self.assertEqual(result["suffix_fraction_of_total_corrected"], 0)

    def test_half_axes_and_negative_components_are_not_clipped(self):
        r = boundary_state_report(row())
        r["targets"]["G"]["between_window_corrected"] = -0.5
        result = readout([r, deepcopy(r)])["targets"]["G"]
        self.assertEqual(result["between_window_corrected"], -0.5)
        self.assertEqual(result["half_means"]["window"], r["targets"]["G"]["window_half_means"])
        self.assertEqual(result["half_means"]["suffix"], r["targets"]["G"]["suffix_half_means"])


if __name__ == "__main__":
    unittest.main()
