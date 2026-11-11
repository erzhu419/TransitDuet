from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments import plan_validity_boundary_tail as tail
from freq_hrl.experiments.compute_budget import ComputeLedger
from freq_hrl.experiments.pointmaze_timing_baselines import _CAUSAL_DYNAMIC_NAMES
from scripts import run_plan_validity_boundary_tail as worker
from scripts import submit_plan_validity_boundary_tail_scheduleurm as submitter


def bank():
    rng = np.random.default_rng(11403)
    names = list(_CAUSAL_DYNAMIC_NAMES)
    rows = []
    for seed in range(8):
        for step in (50, 105, 150):
            row = {"seed": seed, "opportunity_step": step, "boundary_step": step + 50,
                "feature_names": names, "causal_features": rng.normal(size=len(names)).tolist(), "windows": []}
            for j in range(4):
                contexts = {arm: {"feature_names": names, "causal_features": rng.normal(size=len(names)).tolist(),
                    "offset_fraction": (step % 50) / 25, "remaining_fraction": (1150 - step) / 1200,
                    "call_spent": bool(j % 2)} for arm in ("now", "wait_one_check")}
                effect = (tail.basis([contexts["wait_one_check"]]) - tail.basis([contexts["now"]]))[0, 0]
                row["windows"].append({"boundary_context": contexts,
                    "suffix_draws": [{"tail": float(effect + rng.normal(0, .1))} for _ in range(4)]})
            rows.append(row)
    return rows


class BoundaryTailTest(unittest.TestCase):
    def test_feature_sides_and_swap_antisymmetry(self):
        rows = bank()
        data = tail.dataset(rows, horizon=1200, period=50)
        changed = deepcopy(rows)
        for row in changed:
            for w in row["windows"]:
                c = w["boundary_context"]
                c["now"], c["wait_one_check"] = c["wait_one_check"], c["now"]
                for d in w["suffix_draws"]:
                    d["tail"] += 100
        altered = tail.dataset(changed, horizon=1200, period=50)
        np.testing.assert_array_equal(data["causal_t"], altered["causal_t"])
        np.testing.assert_array_equal(data["boundary_delta"], -altered["boundary_delta"])
        self.assertEqual(data["tail"].shape, (96, 4))
        self.assertEqual(data["causal_t"].shape, (96, 120))
        self.assertEqual(set(data["groups"]), set(range(8)))

    def test_suffix_B_and_held_labels_cannot_change_their_fitted_predictions(self):
        data = tail.dataset(bank(), horizon=1200, period=50)
        train = data["groups"] != 0
        p, models = tail.fit_views(data, train, total_df=3)
        changed = deepcopy(data)
        changed["tail"][:, 2:] += 1000
        changed["tail"][~train, :2] -= 1000
        q, other = tail.fit_views(changed, train, total_df=3)
        self.assertEqual(models, other)
        for name in tail.METHODS:
            np.testing.assert_array_equal(p[name], q[name])
        for name in ("causal_t", "boundary_delta"):
            self.assertAlmostEqual(models[name]["total_effective_df"], 3)
        model = models["boundary_delta"]
        np.testing.assert_allclose(tail.predict(model, -data["boundary_delta"]), -p["boundary_delta"], atol=1e-12)
        np.testing.assert_array_equal(tail.predict(model, np.zeros_like(data["boundary_delta"])), 0)

    def test_noise_correction_and_cross_block_errors(self):
        values = np.asarray([[-1., 1., -1., 1.], [-1., 1., -1., 1.]])
        train = tail.errors(np.zeros(2), values, independent=False)
        held = tail.errors(np.zeros(2), values, independent=True)
        self.assertIsNone(train["A"]["noise_corrected_mse"])
        self.assertIsNone(train["conditional_squared_error_cross_AB"])
        self.assertEqual(train["B"]["noise_corrected_mse"], -1)
        self.assertEqual(held["A"]["noise_corrected_mse"], -1)
        self.assertEqual(held["conditional_squared_error_cross_AB"], 0)

    def test_native_free_worker_closes_folds_and_exact_fit_budget(self):
        @contextmanager
        def runtime():
            yield {"fixture": True}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = bank()
            for shard in (0, 1):
                d = root / "cache" / f"shard{shard}"
                (d / "public").mkdir(parents=True)
                (d / "private").mkdir()
                seeds = list(range(4 * shard, 4 * shard + 4))
                (d / "public/result.json").write_text(json.dumps({"run_name": "source", "task": "fixture",
                    "optimizer_seed": 17, "status": "complete", "stage": "boundary_variance_complete", "sampled_seeds": seeds}))
                (d / "private/rows.json").write_text(json.dumps([r for r in rows if r["seed"] in seeds]))
            manifest = {"run_name": "probe", "source_run_name": "source", "source_hardware_profile": {"fixture": True},
                "horizon": 1200, "period": 50, "total_effective_df": 3, "ridge_fits_per_root": 18, "constant_fits_per_root": 9,
                "cells": [{"task": "fixture", "optimizer_seed": 17, "cache_directory": "cache",
                    "output_directory": "out", "development_prefix_seeds": list(range(8))}]}
            (root / "manifest.json").write_text(json.dumps(manifest))
            with patch.object(worker, "ROOT", root), patch.object(worker, "single_thread_runtime", runtime):
                self.assertEqual(worker.main(["--manifest", str(root / "manifest.json"), "--cell-index", "0"]), 0)
                with self.assertRaises(FileExistsError):
                    worker.main(["--manifest", str(root / "manifest.json"), "--cell-index", "0"])
            result = json.loads((root / "out/public/result.json").read_text())
            counts = result["measured_cost"]["counts"]
            self.assertEqual(counts, {"ridge_fit_calls": 18, "constant_fit_calls": 9})
            self.assertEqual(result["new_environment_steps"], 0)
            self.assertEqual(result["prefixes"], 8)
            private = json.loads((root / "out/private/models_and_predictions.json").read_text())
            self.assertEqual(len(private["fold_models"]), 8)
            for fold in private["fold_models"]:
                self.assertNotIn(fold["held_prefix"], fold["training_prefixes"])
            self.assertNotIn("weights", json.dumps(result))
            self.assertNotIn("causal_features", json.dumps(result))

    def test_missing_cache_prevents_any_fit(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(worker, "ROOT", Path(folder)), \
             patch.object(worker, "fit_probe") as fit:
            with self.assertRaises(FileNotFoundError):
                worker.load_cache({"cache_directory": "missing"}, {})
            fit.assert_not_called()

    def test_scheduler_uses_only_existing_banks_without_hard_node_pins(self):
        source = json.loads((submitter.ROOT / "protocols/plan_validity_boundary_variance_2026-11-11.json").read_text())
        frozen = submitter.prepare(source, {}, "probe", Path("protocols/probe.json"))
        self.assertEqual(len(frozen["jobs"]), 4)
        self.assertEqual(frozen["total_ridge_fits"], 72)
        self.assertEqual(frozen["new_environment_steps"], 0)
        for old, cell, job in zip(source["cells"], frozen["cells"], frozen["jobs"]):
            self.assertEqual(cell["cache_directory"], old["output_directory"])
            self.assertEqual(cell["development_prefix_seeds"], old["development_prefix_seeds"])
            self.assertEqual((job["cpu"], job["ram_mb"], job["vram"]), (1, 2048, 0))
            self.assertIsNone(job["require_node"])
            self.assertIsNone(job["preferred_node"])
            self.assertTrue(job["protect_from_auto_eviction"])
            self.assertTrue(job["result_dir"].endswith("/public"))


if __name__ == "__main__":
    unittest.main()
