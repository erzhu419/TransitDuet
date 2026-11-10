from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments import plan_validity_decision_calibration as calibration
from freq_hrl.experiments import plan_validity_label_continuation as labels
from freq_hrl.experiments.compute_budget import ComputeLedger, count_work
from scripts import run_plan_validity_decision_calibration as worker
from scripts import submit_plan_validity_decision_calibration_scheduleurm as submitter
from test_plan_validity_label_continuation import banks


class DecisionCalibrationTest(unittest.TestCase):
    def test_G_objective_keeps_the_original_cut_feasible_and_enforces_W_floor(self):
        scores = np.array([0, 1, 2, 3], dtype=float)
        selected = calibration.calibrate_cut(scores, np.array([1, -3, 1, 1]), np.array([1, 8, 1, 1]), 2.5)
        # The unconstrained all-now choice has larger G but violates the W floor.
        self.assertGreaterEqual(selected["delta_W_A"], 0)
        self.assertGreaterEqual(selected["delta_G_A"], 0)
        brute = []
        for cut in (2.5, np.nextafter(0., -np.inf), 0.5, 1.5, np.nextafter(3., np.inf)):
            delta = (scores >= cut).astype(int) - (scores >= 2.5).astype(int)
            w, g = np.mean(delta * [1, -3, 1, 1]), np.mean(delta * [1, 8, 1, 1])
            if w >= 0:
                brute.append(g)
        self.assertAlmostEqual(selected["delta_G_A"], max(brute))
        self.assertEqual(selected["cut"], 1.5)

    def test_zero_G_ties_retain_the_exact_original_cut(self):
        result = calibration.calibrate_cut(np.array([-1., 0., 1., 2.]), np.ones(4), np.zeros(4), 0.37)
        self.assertEqual(result["cut"], 0.37)
        self.assertEqual(result["changed_A_actions"], 0)

    def test_OOF_reconstruction_does_not_refit_full_models_and_B_cannot_change_selection(self):
        data = banks()
        rules, _ = labels.fit_labels(data, effective_df=8, threshold_quantile=0.75)
        frozen = {k: v.to_dict() for k, v in rules.items()}
        ledger = ComputeLedger()
        with ledger.recording("reconstruct"):
            first = calibration.reconstruct_oof(data, rules, effective_df=8, quantile=0.75)
        self.assertEqual(ledger.summary()["counts"]["ridge_fit_calls"], 16)
        changed = deepcopy(data)
        for rows in changed.values():
            for row in rows:
                for r in row["B_replicas"]:
                    r["adaptive_window_advantage"] += 1000
                    r["full_advantage"] -= 1000
        second = calibration.reconstruct_oof(changed, rules, effective_df=8, quantile=0.75)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
            self.assertAlmostEqual(np.quantile(first[name], 0.75), rules[name].threshold)
            a = calibration.values(data["candidate"], "W", "A").mean(axis=1)
            g = calibration.values(data["candidate"], "G", "A").mean(axis=1)
            self.assertEqual(calibration.calibrate_cut(first[name], a, g, rules[name].threshold),
                calibration.calibrate_cut(second[name], a, g, rules[name].threshold))
        self.assertEqual(frozen, {k: v.to_dict() for k, v in rules.items()})

    def test_B_pairs_cancel_before_MC_error_and_prefixes_are_not_futures(self):
        data = banks()
        rows = data["candidate"]
        rules, _ = labels.fit_labels(data, effective_df=8, threshold_quantile=0.75)
        for row in rows:
            for r in row["B_replicas"]:
                r["adaptive_window_advantage"], r["full_advantage"] = 2, 7
        oof = {name: np.zeros(len(rows)) for name in rules}
        for rule in rules.values():
            rule.threshold = 1e6
        selected = {name: {"cut": -1e6} for name in rules}
        result = calibration.decision_readout(rows, oof, rules, selected)
        for head in rules:
            for view in ("oof_scores", "frozen_full_scores"):
                gain = result[head][view]["B_panels"]["B_all"]["utility_minus_quantile"]["G"]
                self.assertEqual(gain["mean"], 7)
                self.assertEqual(gain["conditional_MC_standard_error"], 0)
                self.assertEqual(gain["training_prefix_interval"]["n"], 8)
                self.assertEqual(gain["training_prefix_interval"]["interval"], [7, 7])
                self.assertEqual(result[head][view]["choose_now"]["utility"], 24)

    def test_manifest_inputs_contain_only_training_and_fit_paths_and_never_launch_by_default(self):
        source_path = submitter.ROOT / "protocols/plan_validity_label_continuation_2026-11-10.json"
        source = json.loads(source_path.read_text())
        m = submitter.prepare(source, Path("protocols/source.json"), "fixture", Path("protocols/fixture.json"))
        self.assertEqual(len(m["jobs"]), 4)
        self.assertEqual(m["fold_reconstruction_ridge_calls"], 256)
        self.assertEqual(m["new_environment_steps"], 0)
        for c in m["cells"]:
            self.assertNotIn("evaluation_seeds", c)
            self.assertNotIn("probe", json.dumps(c))
        for job in m["jobs"]:
            self.assertEqual((job["cpu"], job["ram_mb"], job["vram"]), (1, 2048, 0))
            self.assertIsNone(job["require_node"])
            self.assertIsNone(job["preferred_node"])
            self.assertTrue(job["result_dir"].endswith("/public"))
        with tempfile.TemporaryDirectory(dir=submitter.ROOT / "protocols") as folder:
            with patch.object(submitter, "execute_bulk") as scheduled:
                submitter.main(["--source-manifest", str(source_path), "--run-name", "fixture",
                    "--manifest", str(Path(folder) / "manifest.json")])
                scheduled.assert_not_called()

    def test_native_access_fails_and_preserves_measured_attempt(self):
        @contextmanager
        def runtime():
            yield {"cpu": "fixture"}

        def unexpected_native(*_):
            count_work("ridge_fit_calls", 4)
            count_work("environment_steps")
            return {"stage": "decision_calibration_complete"}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "manifest.json"
            path.write_text(json.dumps({"run_name": "fixture", "source_hardware_profile": {"cpu": "fixture"},
                "cells": [{"task": "pointmaze", "optimizer_seed": 1, "fit_prefix_seeds": [1, 2], "output_directory": "out"}]}))
            with patch.object(worker, "ROOT", root), patch.object(worker, "single_thread_runtime", runtime), \
                 patch.object(worker, "run_cell", side_effect=unexpected_native):
                self.assertEqual(worker.main(["--manifest", str(path), "--cell-index", "0"]), 1)
            public = json.loads((root / "out/public/result.json").read_text())
            private = json.loads((root / "out/private/result.json").read_text())
            self.assertEqual(public["status"], "failed")
            self.assertEqual(public["measured_cost"], private["measured_cost"])
            self.assertEqual(public["measured_cost"]["counts"]["environment_steps"], 1)
            self.assertEqual(public["new_environment_steps"], 1)
            self.assertFalse(public["confirmation_released"])

    def test_actual_cache_flow_freezes_before_B_and_keeps_full_models_unchanged(self):
        source = banks()
        seeds = list(range(1000, 1032))
        expanded = {c: [] for c in labels.CONTINUATIONS}
        for c in labels.CONTINUATIONS:
            for i, seed in enumerate(seeds):
                expanded[c].extend({**deepcopy(r), "seed": seed} for r in source[c][3 * (i % 8):3 * (i % 8 + 1)])
        rules, reports = labels.fit_labels(expanded, effective_df=8, threshold_quantile=0.75)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fit = root / "source/fit"
            for sub in ("private", "public"):
                (fit / sub).mkdir(parents=True)
            frozen = json.dumps({k: r.to_dict() for k, r in rules.items()})
            (fit / "private/frozen_rules.json").write_text(frozen)
            (fit / "public/result.json").write_text(json.dumps({"status": "complete", "run_name": "source",
                "task": "pointmaze", "optimizer_seed": 17, "training_readout": reports}))
            for shard in (0, 1):
                directory = root / f"source/sample/shard{shard}"
                for sub in ("private", "public"):
                    (directory / sub).mkdir(parents=True)
                (directory / "public/result.json").write_text(json.dumps({"status": "complete", "run_name": "source",
                    "optimizer_seed": 17, "sampled_seeds": seeds[16 * shard:16 * (shard + 1)]}))
                (directory / "private/label_banks.json").write_text(json.dumps({c: rows[48 * shard:48 * (shard + 1)] for c, rows in expanded.items()}))
            output = root / "output"
            (output / "private").mkdir(parents=True)
            cell = {"task": "pointmaze", "optimizer_seed": 17, "fit_directory": "source/fit",
                "sample_directory": "source/sample", "fit_prefix_seeds": seeds}
            manifest = {"source_run_name": "source", "effective_df": 8, "original_quantile": 0.75}
            actual = worker.decision_readout

            def B_readout(*args):
                self.assertEqual(set(json.loads((output / "private/frozen_cuts.json").read_text())), set(labels.HEADS))
                return actual(*args)

            ledger = ComputeLedger()
            with patch.object(worker, "ROOT", root), patch.object(worker, "decision_readout", side_effect=B_readout), ledger.recording("flow"):
                result = worker.run_cell(cell, manifest, output, lambda _: None)
            self.assertEqual(result["states"], 96)
            self.assertEqual(result["training_prefixes"], 32)
            self.assertEqual(ledger.summary()["counts"]["ridge_fit_calls"], 64)
            self.assertEqual(ledger.summary()["counts"].get("environment_steps", 0), 0)
            self.assertEqual((fit / "private/frozen_rules.json").read_text(), frozen)


if __name__ == "__main__":
    unittest.main()
