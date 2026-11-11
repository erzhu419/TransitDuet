from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments import plan_validity_boundary_parent_scale as parent
from freq_hrl.experiments import plan_validity_boundary_tail as old
from freq_hrl.experiments.compute_budget import ComputeLedger
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from freq_hrl.experiments.pointmaze_timing_baselines import _ridge_at_df
from scripts import run_plan_validity_boundary_parent_scale as worker
from scripts import run_plan_validity_boundary_tail as old_worker
from scripts import submit_plan_validity_boundary_parent_scale_scheduleurm as submitter
from test_plan_validity_boundary_tail import bank


class BoundaryParentScaleTest(unittest.TestCase):
    def test_training_only_parent_scaling_labels_and_antisymmetry(self):
        data = parent.parent_dataset(bank(), horizon=1200, period=50)
        train = data["groups"] != 0
        scores, model = parent.fit_parent(data, train, total_df=3)
        changed = deepcopy(data)
        changed["tail"][:, 2:] += 1000
        changed["tail"][~train, :2] -= 1000
        changed["now"][~train, :39] += 1000
        changed["wait_one_check"][~train, :39] *= 1000
        other_scores, other = parent.fit_parent(changed, train, total_df=3)
        self.assertEqual(model, other)
        np.testing.assert_array_equal(scores[train], other_scores[train])
        np.testing.assert_allclose(parent.predict_parent(model, data["wait_one_check"], data["now"]), -scores, atol=1e-12)
        np.testing.assert_array_equal(parent.predict_parent(model, data["now"], data["now"]), 0)
        self.assertAlmostEqual(model["total_effective_df"], 3)
        expected_scale = np.r_[data["now"][train, :39], data["wait_one_check"][train, :39]].std(axis=0)
        np.testing.assert_allclose(model["parent_scale"], np.where(expected_scale > 1e-8, expected_scale, 1))

    def test_near_zero_training_gate_contrast_is_not_amplified(self):
        rng = np.random.default_rng(11404)
        n = 81
        now, wait = rng.normal(size=(n, 39)), rng.normal(size=(n, 39))
        wait[:, 0] = now[:, 0] + rng.normal(0, 1e-6, n)

        def arms(f, spent):
            clock = np.column_stack((np.zeros(n), np.full(n, .5), spent))
            return np.column_stack((f, clock, f * .5, f * spent[:, None]))

        spent_now, spent_wait = np.ones(n), np.ones(n)
        spent_now[-1] = 0
        a, b = arms(now, spent_now), arms(wait, spent_wait)
        y = .01 * (wait[:, 0] - now[:, 0]) / 1e-6
        data = {"now": a, "wait_one_check": b, "tail": np.repeat(y[:, None], 4, axis=1)}
        train = np.arange(n) < n - 1
        x = b - a
        old_scores, _ = _ridge_at_df(np.r_[x[train], -x[train]], np.r_[y[train], -y[train]], x, 3)
        repaired, model = parent.fit_parent(data, train, total_df=3)
        self.assertGreater(abs(old_scores[-1]), 100)
        self.assertLess(abs(repaired[-1]), 1)
        self.assertGreater(model["parent_scale"][0], .5)

    def test_prior_roster_and_losses_are_checked_before_reuse(self):
        data = parent.parent_dataset(bank(), horizon=1200, period=50)
        saved, private = old.fit_probe(data, total_df=3)
        parent.verify_prior(data, private, saved)
        bad = deepcopy(private)
        bad["training_predictions"]["constant"][0] += 1
        with self.assertRaisesRegex(ValueError, "frozen prior losses"):
            parent.verify_prior(data, bad, saved)
        bad = deepcopy(private)
        bad["prefix_by_window"][0] = -1
        with self.assertRaisesRegex(ValueError, "window roster"):
            parent.verify_prior(data, bad, saved)

    def test_worker_reuses_all_baselines_and_fits_only_nine_potentials(self):
        @contextmanager
        def runtime():
            yield {"fixture": True}

        rows = bank()
        data = parent.parent_dataset(rows, horizon=1200, period=50)
        saved, private = old.fit_probe(data, total_df=3)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for shard in (0, 1):
                d = root / "cache" / f"shard{shard}"
                (d / "public").mkdir(parents=True)
                (d / "private").mkdir()
                seeds = list(range(4 * shard, 4 * shard + 4))
                (d / "public/result.json").write_text(json.dumps({"run_name": "source", "task": "fixture", "optimizer_seed": 17,
                    "status": "complete", "stage": "boundary_variance_complete", "sampled_seeds": seeds}))
                (d / "private/rows.json").write_text(json.dumps([r for r in rows if r["seed"] in seeds]))
            for name in ("public", "private"):
                (root / "previous" / name).mkdir(parents=True)
            saved.update(run_name="previous", task="fixture", optimizer_seed=17, status="complete", stage="boundary_tail_probe_complete")
            (root / "previous/public/result.json").write_text(json.dumps(_json_ready(saved)))
            frozen = json.dumps(_json_ready(private))
            (root / "previous/private/models_and_predictions.json").write_text(frozen)
            manifest = {"run_name": "parent", "prior_probe_run_name": "previous", "source_run_name": "source",
                "source_hardware_profile": {"fixture": True}, "horizon": 1200, "period": 50, "total_effective_df": 3,
                "ridge_fits_per_root": 9, "cells": [{"task": "fixture", "optimizer_seed": 17, "cache_directory": "cache",
                    "prior_probe_directory": "previous", "output_directory": "out", "development_prefix_seeds": list(range(8))}]}
            (root / "manifest.json").write_text(json.dumps(manifest))
            with patch.object(worker, "ROOT", root), patch.object(old_worker, "ROOT", root), \
                 patch.object(worker, "single_thread_runtime", runtime):
                args = ["--manifest", str(root / "manifest.json"), "--cell-index", "0"]
                self.assertEqual(worker.main(args), 0)
                with self.assertRaises(FileExistsError):
                    worker.main(args)
            result = json.loads((root / "out/public/result.json").read_text())
            self.assertEqual(result["measured_cost"]["counts"], {"ridge_fit_calls": 9})
            self.assertEqual(result["new_environment_steps"], 0)
            self.assertTrue(result["frozen_baseline_losses_reproduced"])
            for panel in ("training", "prefix_oof"):
                for method in ("zero", "constant", "causal_t"):
                    self.assertEqual(result["panels"][panel][method], saved["panels"][panel][method])
            self.assertEqual((root / "previous/private/models_and_predictions.json").read_text(), frozen)
            self.assertNotIn("weights", json.dumps(result))

    def test_scheduler_preserves_the_frozen_cohort_and_baseline_cache(self):
        source = json.loads((submitter.ROOT / "protocols/plan_validity_boundary_tail_2026-11-11.json").read_text())
        m = submitter.prepare(source, "parent", Path("protocols/parent.json"))
        self.assertEqual((m["total_ridge_fits"], m["total_constant_fits"], m["new_environment_steps"]), (36, 0, 0))
        for old_cell, c, job in zip(source["cells"], m["cells"], m["jobs"]):
            self.assertEqual(c["cache_directory"], old_cell["cache_directory"])
            self.assertEqual(c["prior_probe_directory"], old_cell["output_directory"])
            self.assertEqual(c["development_prefix_seeds"], old_cell["development_prefix_seeds"])
            self.assertEqual((job["cpu"], job["ram_mb"], job["vram"]), (1, 2048, 0))
            self.assertIsNone(job["require_node"])
            self.assertIsNone(job["preferred_node"])


if __name__ == "__main__":
    unittest.main()
