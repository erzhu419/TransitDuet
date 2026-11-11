from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments.compute_budget import ComputeLedger
from freq_hrl.experiments.plan_validity_boundary_tail import dataset, fit_probe
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from scripts import analyze_plan_validity_boundary_noise as analysis
from test_plan_validity_boundary_tail import bank


class BoundaryNoiseTest(unittest.TestCase):
    def test_paired_loss_and_suffix_variance_use_shared_label_cancellation(self):
        tail = np.asarray([[0., 0., 0., 2.], [0., 0., 2., 6.]])
        p, q = np.asarray([1., 2.]), np.asarray([0., 1.])
        result = analysis.paired_parts(p, q, tail)
        self.assertEqual(result["squared_prediction_term"], 2)
        self.assertEqual(result["A_delta"], 2)
        self.assertEqual(result["B_delta"], -3)
        self.assertAlmostEqual(result["B_suffix_mc_se"], np.sqrt(5))
        self.assertEqual(result["B_replica_deltas"], [0, -6])
        same = analysis.paired_parts(p, p, tail)
        self.assertEqual(same["B_suffix_mc_se"], 0)
        self.assertEqual(same["B_delta"], 0)
        self.assertIsNone(same["largest_window_share_of_B_mc_variance"])

    def test_prefix_deletion_and_negative_dispersion_are_readout_only(self):
        data = {"groups": np.asarray([1, 2]), "windows": np.asarray([0, 2]),
            "tail": np.asarray([[0., 0., 0., 2.], [0., 0., 2., 6.]])}
        before = deepcopy(data)
        result = analysis.comparison(np.asarray([1., 2.]), np.asarray([0., 1.]), data)
        self.assertEqual(result["B_prefix_variance_observed"], 8)
        self.assertAlmostEqual(result["B_prefix_suffix_mc_variance"], 10)
        self.assertAlmostEqual(result["B_prefix_dispersion_corrected"], -2)
        self.assertEqual([p["B_delta_without_prefix"] for p in result["prefix_reports"]], [-5, -1])
        self.assertEqual(result["B_improving_prefixes"], 2)
        self.assertAlmostEqual(result["largest_beneficial_prefix_share_of_gross_benefit"], 5 / 6)
        for key in data:
            np.testing.assert_array_equal(data[key], before[key])
        data["tail"][:, :2] += 1000
        changed = analysis.comparison(np.asarray([1., 2.]), np.asarray([0., 1.]), data)
        self.assertEqual(changed["B_delta"], result["B_delta"])
        self.assertEqual(changed["B_suffix_mc_se"], result["B_suffix_mc_se"])

    def test_cached_worker_reproduces_losses_without_fits_or_simulation(self):
        @contextmanager
        def runtime():
            yield {"fixture": True}

        rows = bank()
        saved, private = fit_probe(dataset(rows, horizon=1200, period=50), total_df=3)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("private", "public"):
                (root / "source" / name).mkdir(parents=True)
            saved.update(run_name="source", task="fixture", optimizer_seed=17, status="complete", stage="boundary_parent_scale_complete")
            (root / "source/public/result.json").write_text(json.dumps(_json_ready(saved)))
            raw = json.dumps(_json_ready(private))
            (root / "source/private/models_and_predictions.json").write_text(raw)
            (root / "source.json").write_text(json.dumps({"run_name": "source", "source_hardware_profile": {"fixture": True},
                "cells": [{"task": "fixture", "optimizer_seed": 17, "output_directory": "source"}]}))
            (root / "manifest.json").write_text(json.dumps({"run_name": "noise", "source_manifest": "source.json", "output_directory": "out"}))
            with patch.object(analysis, "ROOT", root), patch.object(analysis, "load_cache", return_value=rows), \
                 patch.object(analysis, "single_thread_runtime", runtime):
                args = ["--manifest", str(root / "manifest.json")]
                self.assertEqual(analysis.main(args), 0)
                with self.assertRaises(FileExistsError):
                    analysis.main(args)
            result = json.loads((root / "out/public/result.json").read_text())
            self.assertEqual(result["measured_cost"]["counts"], {})
            self.assertEqual((result["new_fits"], result["new_environment_steps"]), (0, 0))
            self.assertEqual(result["cells"][0]["prefixes"], 8)
            self.assertNotIn("weights", json.dumps(result))
            self.assertNotIn("causal_features", json.dumps(result))
            self.assertEqual((root / "source/private/models_and_predictions.json").read_text(), raw)


if __name__ == "__main__":
    unittest.main()
