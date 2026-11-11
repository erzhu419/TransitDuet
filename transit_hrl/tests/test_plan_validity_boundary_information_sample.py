from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from freq_hrl.experiments.compute_budget import ComputeLedger, count_work
from scripts import run_plan_validity_boundary_information_sample as worker
from scripts import submit_plan_validity_boundary_information_sample_scheduleurm as submitter


def frozen():
    design = json.loads((worker.ROOT / "protocols/plan_validity_boundary_information_2026-11-11.json").read_text())
    release = submitter.prepare(design, Path("protocols/design.json"), Path("protocols/release.json"))
    return design, release


def acquire(_controller, _rule, reference, indices, *, args, spec):
    count_work("environment_steps", 2 * args.horizon * sum(len(v) for v in indices.values()))
    return {"seed": reference["seed"], "opportunity_step": reference["opportunity_step"],
        "windows": [{"window_index": j, "A_replicas": [{"suffix_index": k} for k in ks]} for j, ks in indices.items()]}


class BoundaryInformationSampleTest(unittest.TestCase):
    def test_exact_release_resources_rosters_and_no_eval_jobs(self):
        design, release = frozen()
        self.assertEqual(len(release["jobs"]), 32)
        self.assertEqual(Counter(s["method"] for s in release["shards"]), {"dense": 8, "wide": 24})
        self.assertEqual(sum(s["environment_steps"] for s in release["shards"]), 11289600)
        for i, cell in enumerate(design["cells"]):
            for method, key in (("dense", "development_prefix_seeds"), ("wide", "wide_training_prefix_seeds")):
                selected = [s for s in release["shards"] if s["cell_index"] == i and s["method"] == method]
                self.assertEqual([seed for s in selected for seed in s["prefix_seeds"]], cell[key])
                self.assertEqual(sum(s["environment_steps"] for s in selected), 1411200)
                self.assertFalse(set(seed for s in selected for seed in s["prefix_seeds"]) & set(cell["readout_prefix_seeds"]))
        for job in release["jobs"]:
            self.assertEqual((job["cpu"], job["ram_mb"], job["vram"]), (1, 2048, 0))
            self.assertIsNone(job["require_node"])
            self.assertIsNone(job["preferred_node"])
            self.assertTrue(job["protect_from_auto_eviction"])
            self.assertTrue(job["allow_cpu_training"])
            self.assertTrue(job["result_dir"].endswith("/public"))
            self.assertIn("/sample/", job["signature"])
        self.assertEqual(release["new_fits"], 0)
        self.assertFalse(release["readout_released"])

    def test_dense_canonical_state_order_balanced_indices_and_full_cost(self):
        design, release = frozen()
        cell = design["cells"][0]
        cached = [{"seed": s, "opportunity_step": step} for s in cell["development_prefix_seeds"] for step in (300, 100, 200)]
        for shard in release["shards"][:2]:
            rows, ledger = [], ComputeLedger()
            with patch.object(worker, "acquire_windows", side_effect=acquire), patch.object(worker, "observe") as observe, ledger.recording("dense"):
                worker.sample_rows(cell, design, shard, None, None, SimpleNamespace(horizon=1200), None, cached, rows.append)
            observe.assert_not_called()
            self.assertEqual(ledger.summary()["counts"]["environment_steps"], shard["environment_steps"])
            self.assertEqual(len(rows), 12)
            self.assertEqual(sum(len(w["A_replicas"]) for r in rows for w in r["windows"]), shard["new_pairs"])
            expected_extra = {(k["seed"], [100, 200, 300][k["state_index"]], k["window_index"])
                for k in cell["dense_extra_windows"] if k["seed"] in shard["prefix_seeds"]}
            actual_extra = set()
            for row in rows:
                for w in row["windows"]:
                    indices = [r["suffix_index"] for r in w["A_replicas"]]
                    self.assertEqual(indices[:6], [4, 5, 6, 7, 8, 9])
                    if len(indices) == 7:
                        self.assertEqual(indices[-1], 10)
                        actual_extra.add((row["seed"], row["opportunity_step"], w["window_index"]))
            self.assertEqual(actual_extra, expected_extra)

    def test_wide_pays_one_factual_per_prefix_and_samples_only_A4_A5(self):
        design, release = frozen()
        cell, shard = design["cells"][0], release["shards"][2]

        def observe(*_, seed, **kwargs):
            self.assertEqual(kwargs["probes_per_path"], 3)
            count_work("environment_steps", 1200)
            return {"seed": seed, "check_steps": [300, 100, 200]}

        def reference(obs, step, **_):
            return {"seed": obs["seed"], "opportunity_step": step}

        rows, ledger = [], ComputeLedger()
        with patch.object(worker, "observe", side_effect=observe) as factual, patch.object(worker, "fresh_reference", side_effect=reference), \
             patch.object(worker, "acquire_windows", side_effect=acquire), ledger.recording("wide"):
            worker.sample_rows(cell, design, shard, None, None, SimpleNamespace(horizon=1200), None, None, rows.append)
        self.assertEqual(factual.call_count, 4)
        self.assertEqual(ledger.summary()["counts"]["environment_steps"], 235200)
        self.assertEqual({r["seed"] for r in rows}, set(shard["prefix_seeds"]))
        self.assertEqual(len(rows), 12)
        self.assertTrue(all([a["suffix_index"] for a in w["A_replicas"]] == [4, 5] for r in rows for w in r["windows"]))

    def test_missing_precheck_blocks_cache_and_controller_access(self):
        design, release = frozen()
        with tempfile.TemporaryDirectory() as folder, patch.object(worker, "ROOT", Path(folder)), \
             patch.object(worker, "load_frozen_C0") as load, patch.object(worker, "load_cache") as cache:
            with self.assertRaises(FileNotFoundError):
                worker.run_shard(design["cells"][0], design, release["shards"][0], Path(folder), lambda _: None)
            load.assert_not_called()
            cache.assert_not_called()

    def test_worker_closes_native_budget_preserves_failure_and_refuses_overwrite(self):
        @contextmanager
        def runtime():
            yield {"fixture": True}

        for should_fail in (False, True):
            def run(*_):
                count_work("environment_steps", 7 if should_fail else 235200)
                if should_fail:
                    raise RuntimeError("fixture sampled-state failure")
                return {"stage": "boundary_information_sample_complete"}

            with self.subTest(fail=should_fail), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "design.json").write_text(json.dumps({"run_name": "fixture", "source_hardware_profile": {"fixture": True},
                    "cells": [{"task": "fixture", "optimizer_seed": 17}]}))
                p = root / "release.json"
                p.write_text(json.dumps({"design_manifest": "design.json", "shards": [
                    {"cell_index": 0, "method": "wide", "shard_index": 0, "output_directory": "out", "environment_steps": 235200}]}))
                with patch.object(worker, "ROOT", root), patch.object(worker, "single_thread_runtime", runtime), patch.object(worker, "run_shard", side_effect=run):
                    args = ["--manifest", str(p), "--shard-index", "0"]
                    self.assertEqual(worker.main(args), int(should_fail))
                    with self.assertRaises(FileExistsError):
                        worker.main(args)
                saved = json.loads((root / "out/public/result.json").read_text())
                self.assertEqual(saved["new_environment_steps"], 7 if should_fail else 235200)
                self.assertEqual(saved["status"], "failed" if should_fail else "complete")
                self.assertEqual(saved["measured_cost"]["counts"].get("ridge_fit_calls", 0), 0)


if __name__ == "__main__":
    unittest.main()
