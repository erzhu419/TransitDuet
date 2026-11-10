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
from freq_hrl.experiments.plan_validity_phase_support import predict
from scripts import run_plan_validity_score_source as worker
from scripts import submit_plan_validity_score_source_scheduleurm as submitter
from test_plan_validity_label_continuation import banks


class ScoreSourceTest(unittest.TestCase):
    def test_direct_contrast_pairs_the_two_selected_actions_on_identical_scores(self):
        rows = banks()["candidate"]
        rules, _ = labels.fit_labels(banks(), effective_df=8, threshold_quantile=0.75)
        oof = {head: np.arange(len(rows), dtype=float) for head in rules}
        selected = {source: {head: {"cut": cut} for head in rules}
            for source, cut in (("oof_scores", 4.), ("frozen_full_scores", 15.))}
        result = calibration.score_source_readout(rows, oof, rules, selected)
        for head, rule in rules.items():
            for view, scores in (("oof_scores", oof[head]), ("frozen_full_scores", predict(rule, rows))):
                expected = ((scores >= 15).astype(int) - (scores >= 4).astype(int))[:, None] * calibration.values(rows, "G", "B")
                actual = result["same_score_source_contrast"][head][view]["B_full_selected_minus_oof_selected"]["B_all"]["G"]
                self.assertAlmostEqual(actual["mean"], expected.mean())
                self.assertEqual(actual["training_prefix_interval"]["n"], 8)
                a = result["by_calibration_source"]["frozen_full_scores"][head][view]["B_panels"]["B_all"]["utility_minus_quantile"]["G"]["mean"]
                b = result["by_calibration_source"]["oof_scores"][head][view]["B_panels"]["B_all"]["utility_minus_quantile"]["G"]["mean"]
                self.assertAlmostEqual(actual["mean"], a - b)

    def test_actual_cache_flow_freezes_before_B_never_fits_and_B_cannot_change_cuts(self):
        source = banks()
        seeds = list(range(1000, 1032))
        expanded = {c: [] for c in labels.CONTINUATIONS}
        for c in labels.CONTINUATIONS:
            for i, seed in enumerate(seeds):
                expanded[c].extend({**deepcopy(r), "seed": seed} for r in source[c][3 * (i % 8):3 * (i % 8 + 1)])
        rules, _ = labels.fit_labels(expanded, effective_df=8, threshold_quantile=0.75)
        oof = calibration.reconstruct_oof(expanded, rules, effective_df=8, quantile=0.75)
        w, g = (calibration.values(expanded["candidate"], k, "A").mean(axis=1) for k in ("W", "G"))
        old_cuts = {head: calibration.calibrate_cut(oof[head], w, g, rules[head].threshold) for head in rules}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fit, cached = root / "fit/private", root / "calibration"
            fit.mkdir(parents=True)
            for sub in ("public", "private"):
                (cached / sub).mkdir(parents=True)
            frozen = json.dumps({head: r.to_dict() for head, r in rules.items()})
            (fit / "frozen_rules.json").write_text(frozen)
            (cached / "public/result.json").write_text(json.dumps({"status": "complete", "run_name": "source",
                "task": "pointmaze", "optimizer_seed": 17}))
            score_cache = json.dumps({"state_roster": [[r["seed"], r["opportunity_step"]] for r in expanded["candidate"]],
                "scores": {head: scores.tolist() for head, scores in oof.items()}})
            (cached / "private/reconstructed_oof.json").write_text(score_cache)
            (cached / "private/frozen_cuts.json").write_text(json.dumps(old_cuts))

            def write_samples():
                for shard in (0, 1):
                    sample = root / f"sample/shard{shard}"
                    for sub in ("public", "private"):
                        (sample / sub).mkdir(parents=True, exist_ok=True)
                    (sample / "public/result.json").write_text(json.dumps({"status": "complete", "run_name": "labels",
                        "optimizer_seed": 17, "sampled_seeds": seeds[16 * shard:16 * (shard + 1)]}))
                    (sample / "private/label_banks.json").write_text(json.dumps({c: rows[48 * shard:48 * (shard + 1)] for c, rows in expanded.items()}))

            cell = {"task": "pointmaze", "optimizer_seed": 17, "fit_prefix_seeds": seeds,
                "fit_directory": "fit", "sample_directory": "sample", "calibration_directory": "calibration"}
            manifest = {"source_run_name": "source", "label_run_name": "labels", "original_quantile": 0.75}
            selections = []
            for attempt in range(2):
                write_samples()
                output = root / f"output{attempt}"
                (output / "private").mkdir(parents=True)
                original_readout = calibration.score_source_readout

                def B_readout(*args):
                    cuts = json.loads((output / "private/frozen_cuts.json").read_text())
                    self.assertEqual(set(cuts), {"oof_scores", "frozen_full_scores"})
                    self.assertEqual(cuts["oof_scores"], old_cuts)
                    return original_readout(*args)

                ledger = ComputeLedger()
                with patch.object(worker, "ROOT", root), patch.object(worker, "score_source_readout", side_effect=B_readout), ledger.recording("flow"):
                    result = worker.run_cell(cell, manifest, output, lambda _: None)
                selections.append(result["selected"])
                self.assertEqual(result["states"], 96)
                for key in ("ridge_fit_calls", "environment_steps", "controller_optimizer_updates", "termination_optimizer_updates"):
                    self.assertEqual(ledger.summary()["counts"].get(key, 0), 0)
                for head in rules:
                    self.assertGreaterEqual(result["readout"]["by_calibration_source"]["frozen_full_scores"][head]["frozen_full_scores"]["apparent_A_utility_minus_quantile"]["W"], 0)
                for rows in expanded.values():
                    for row in rows:
                        for future in row["B_replicas"]:
                            future["adaptive_window_advantage"] += 1000
                            future["full_advantage"] -= 2000
            self.assertEqual(selections[0], selections[1])
            self.assertEqual((fit / "frozen_rules.json").read_text(), frozen)
            self.assertEqual((cached / "private/reconstructed_oof.json").read_text(), score_cache)

    def test_manifest_reuses_scores_and_contains_no_probe_paths_or_default_launch(self):
        path = submitter.ROOT / "protocols/plan_validity_decision_calibration_2026-11-10.json"
        source = json.loads(path.read_text())
        m = submitter.prepare(source, Path("protocols/source.json"), "fixture", Path("protocols/fixture.json"))
        self.assertEqual(len(m["jobs"]), 4)
        self.assertEqual(m["ridge_fit_calls"], 0)
        self.assertEqual(m["new_environment_steps"], 0)
        self.assertFalse(m["confirmation_released"])
        for old, cell, job in zip(source["cells"], m["cells"], m["jobs"]):
            self.assertEqual(cell["calibration_directory"], old["output_directory"])
            self.assertNotIn("probe", json.dumps(cell))
            self.assertNotIn("evaluation_seeds", cell)
            self.assertEqual((job["cpu"], job["ram_mb"], job["vram"]), (1, 2048, 0))
            self.assertIsNone(job["require_node"])
            self.assertIsNone(job["preferred_node"])
            self.assertTrue(job["result_dir"].endswith("/public"))
        with tempfile.TemporaryDirectory(dir=submitter.ROOT / "protocols") as folder:
            with patch.object(submitter, "execute_bulk") as scheduled:
                submitter.main(["--source-manifest", str(path), "--run-name", "fixture",
                    "--manifest", str(Path(folder) / "manifest.json")])
                scheduled.assert_not_called()

    def test_budget_violation_is_failed_and_its_measured_cost_is_retained(self):
        @contextmanager
        def runtime():
            yield {"cpu": "fixture"}

        def unexpected_work(*_):
            count_work("ridge_fit_calls")
            count_work("environment_steps", 7)
            return {"stage": "score_source_complete"}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "manifest.json"
            path.write_text(json.dumps({"run_name": "fixture", "source_hardware_profile": {"cpu": "fixture"},
                "cells": [{"task": "pointmaze", "optimizer_seed": 1, "output_directory": "out"}]}))
            with patch.object(worker, "ROOT", root), patch.object(worker, "single_thread_runtime", runtime), \
                 patch.object(worker, "run_cell", side_effect=unexpected_work):
                self.assertEqual(worker.main(["--manifest", str(path), "--cell-index", "0"]), 1)
                with self.assertRaises(FileExistsError):
                    worker.main(["--manifest", str(path), "--cell-index", "0"])
            result = json.loads((root / "out/public/result.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["new_environment_steps"], 7)
            self.assertEqual(result["measured_cost"]["counts"]["ridge_fit_calls"], 1)
            self.assertFalse(result["confirmation_released"])


if __name__ == "__main__":
    unittest.main()
