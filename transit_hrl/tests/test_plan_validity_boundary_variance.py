from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments import plan_validity_boundary_credit as credit
from freq_hrl.experiments.compute_budget import ComputeLedger, count_work
from scripts import run_plan_validity_boundary_variance as worker
from scripts.submit_plan_validity_independent_calibration_scheduleurm import literal_seeds
from test_plan_validity_phase_support import clock_rule
from test_plan_validity_second_task import Controller, options


def row(seed=1, step=50):
    w = np.array([1, 2, 3, 4.])
    tail = np.array([[1, -1, 2, -2], [3, 1, 4, 0], [5, 3, 6, 2], [7, 5, 8, 4.]])
    return {"seed": seed, "opportunity_step": step, "boundary_step": step + 50,
        "windows": [{"suffix_draws": [{"W": float(w[j]), "tail": float(tail[j, k]),
            "G": float(w[j] + tail[j, k])} for k in range(4)]} for j in range(4)]}


class BoundaryVarianceTest(unittest.TestCase):
    def test_frozen_rosters_resources_and_full_replay_cost(self):
        root = Path(__file__).resolve().parents[1]
        m = json.loads((root / "protocols/plan_validity_boundary_variance_2026-11-11.json").read_text())
        previous = json.loads((root / m["source_manifest"]).read_text())
        precheck = json.loads((root / m["precheck_manifest"]).read_text())
        forbidden = literal_seeds(previous) | literal_seeds(precheck)
        self.assertEqual(len(m["cells"]), 4)
        self.assertEqual(len(m["jobs"]), 8)
        for cell in m["cells"]:
            self.assertEqual(len(cell["development_prefix_seeds"]), 8)
            self.assertFalse(set(cell["development_prefix_seeds"]) & forbidden)
            forbidden.update(cell["development_prefix_seeds"])
        expected = 1200 * (1 + 3 * (2 + 2 * 4 * 4))
        self.assertEqual(m["per_prefix_environment_steps"], expected)
        self.assertEqual(m["per_job_environment_steps"], 4 * expected)
        self.assertEqual(m["total_environment_steps"], 32 * expected)
        self.assertEqual(m["new_model_fits"], 0)
        self.assertFalse(m["E8_reused_for_training"])
        for job in m["jobs"]:
            self.assertEqual((job["cpu"], job["ram_mb"], job["vram"]), (1, 2048, 0))
            self.assertIsNone(job["require_node"])
            self.assertIsNone(job["preferred_node"])
            self.assertTrue(job["result_dir"].endswith("/public"))

    def test_variance_covariance_identity_and_cost_projection(self):
        r = credit.boundary_state_report(row())
        w, t, g = (r["targets"][k] for k in ("W", "tail", "G"))
        self.assertEqual(w["within_window"], 0)
        self.assertAlmostEqual(g["within_window"], t["within_window"])
        self.assertAlmostEqual(g["between_window_corrected"],
            w["between_window_corrected"] + t["between_window_corrected"] + 2 * r["between_window_W_tail_covariance"])
        self.assertAlmostEqual(g["nested_mean_variance"] - g["equal_pair_count_independent_projection"],
            g["between_window_corrected"] * 3 / 16)

    def test_half_groups_are_different_axes_and_raw_features_are_not_public(self):
        original = row()
        original["causal_features"] = [987654321]
        report = credit.boundary_state_report(original)
        w = report["targets"]["W"]
        self.assertEqual(w["window_half_means"], [1.5, 3.5])
        self.assertEqual(w["suffix_half_means"], [2.5, 2.5])
        self.assertNotIn("causal_features", report)
        self.assertNotIn("suffix_draws", json.dumps(report))
        self.assertEqual(original["causal_features"], [987654321])

    def test_negative_corrections_survive_and_zero_sign_agreement_is_disclosed(self):
        raw = row()
        for window in raw["windows"]:
            window["suffix_draws"] = [{"W": 0., "tail": t, "G": t} for t in (-1., 1., -1., 1.)]
        report = credit.boundary_state_report(raw)
        self.assertLess(report["targets"]["G"]["between_window_corrected"], 0)
        summary = credit.summarize_boundary_reports([report, {**deepcopy(report), "seed": 2}])
        self.assertEqual(summary["prefixes"], 2)
        self.assertEqual(summary["states"], 2)
        self.assertEqual(summary["targets"]["G"]["negative_between_window_estimates"], 2)
        self.assertEqual(summary["targets"]["G"]["suffix_half_both_zero"], 2)

    def test_missing_precheck_blocks_loading_any_controller(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "precheck.json").write_text(json.dumps({"cells": [{"output_directory": "missing"}]}))
            with patch.object(worker, "ROOT", root), patch.object(worker, "load_sources") as load:
                with self.assertRaises(FileNotFoundError):
                    worker.run_cell({}, {"precheck_manifest": "precheck.json"}, 0, root, lambda _: None)
                load.assert_not_called()

    def test_sampling_preserves_roster_and_private_rows_without_fitting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "precheck/public").mkdir(parents=True)
            (root / "precheck/public/result.json").write_text(json.dumps({"run_name": "pre", "task": "pointmaze",
                "optimizer_seed": 17, "status": "complete", "stage": "boundary_precheck_passed"}))
            (root / "precheck.json").write_text(json.dumps({"run_name": "pre", "cells": [
                {"task": "pointmaze", "optimizer_seed": 17, "output_directory": "precheck"}]}))
            (root / "phase").mkdir()
            frozen = json.dumps({"checks_clock": clock_rule().to_dict()})
            (root / "phase/frozen_rules.json").write_text(frozen)
            (root / "source.json").write_text(json.dumps({"cells": [
                {"task": "pointmaze", "optimizer_seed": 17, "phase_cache_directory": "phase"}]}))
            (root / "output/private").mkdir(parents=True)
            cell = {"task": "pointmaze", "optimizer_seed": 17, "development_prefix_seeds": list(range(100, 108))}
            manifest = {"precheck_manifest": "precheck.json", "source_manifest": "source.json", "horizon": 300,
                "window_draws": 4, "suffix_draws_per_window": 4}
            ledger = ComputeLedger()
            with patch.object(worker, "ROOT", root), \
                 patch.object(worker, "load_sources", return_value=({}, {"history": Controller()}, {}, options())), \
                 patch.object(worker, "observe", side_effect=lambda *a, seed, **kw: {"seed": seed, "check_steps": [50, 100, 150]}), \
                 patch.object(worker, "collect_boundary_pair", side_effect=lambda a, b, obs, step, **kw: row(obs["seed"], step)), ledger.recording("flow"):
                result = worker.run_cell(cell, manifest, 0, root / "output", lambda _: None)
            self.assertEqual(result["sampled_seeds"], [100, 101, 102, 103])
            self.assertEqual(result["summary"]["states"], 12)
            self.assertEqual(result["summary"]["prefixes"], 4)
            self.assertEqual(len(result["state_reports"]), 12)
            self.assertEqual(len(json.loads((root / "output/private/rows.json").read_text())), 12)
            self.assertEqual((root / "phase/frozen_rules.json").read_text(), frozen)
            self.assertEqual(ledger.summary()["counts"].get("ridge_fit_calls", 0), 0)

    def test_failed_attempt_keeps_measured_work_and_cannot_overwrite(self):
        @contextmanager
        def runtime():
            yield {"cpu": "fixture"}

        def unexpected_fit(*_):
            count_work("environment_steps", 7)
            count_work("ridge_fit_calls")
            return {"stage": "boundary_variance_complete"}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "source.json").write_text(json.dumps({"source_hardware_profile": {"cpu": "fixture"}}))
            path = root / "manifest.json"
            path.write_text(json.dumps({"run_name": "fixture", "source_manifest": "source.json", "per_job_environment_steps": 7,
                "cells": [{"task": "pointmaze", "optimizer_seed": 1, "output_directory": "out"}]}))
            with patch.object(worker, "ROOT", root), patch.object(worker, "single_thread_runtime", runtime), \
                 patch.object(worker, "run_cell", side_effect=unexpected_fit):
                args = ["--manifest", str(path), "--cell-index", "0", "--shard-index", "0"]
                self.assertEqual(worker.main(args), 1)
                with self.assertRaises(FileExistsError):
                    worker.main(args)
            d = json.loads((root / "out/shard0/public/result.json").read_text())
            self.assertEqual(d["status"], "failed")
            self.assertEqual(d["new_environment_steps"], 7)
            self.assertFalse(d["confirmation_released"])


if __name__ == "__main__":
    unittest.main()
