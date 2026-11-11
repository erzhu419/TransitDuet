from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments import plan_validity_boundary_information as info
from freq_hrl.experiments import plan_validity_transfer as shared
from freq_hrl.experiments.compute_budget import ComputeLedger, count_work
from freq_hrl.experiments.plan_validity_boundary_credit import collect_boundary_pair
from freq_hrl.experiments.plan_validity_phase_support import observe
from freq_hrl.experiments.plan_validity_preregistration import draft_plan
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from scripts import run_plan_validity_boundary_information_precheck as worker
from scripts import submit_plan_validity_boundary_information_scheduleurm as submitter
from test_plan_validity_phase_support import clock_rule
from test_plan_validity_second_task import Controller, DenseTask, options


class BoundaryInformationTest(unittest.TestCase):
    def test_full_replay_budget_and_balanced_outcome_free_extra_allocation(self):
        source = json.loads((submitter.ROOT / "protocols/plan_validity_boundary_parent_scale_2026-11-11.json").read_text())
        m = submitter.prepare(source, draft_plan(), set(), "fixture", Path("protocols/fixture.json"))
        seen = set()
        for c, job in zip(m["cells"], m["jobs"]):
            keys = info.dense_extra_windows(c["development_prefix_seeds"])
            self.assertEqual(keys, c["dense_extra_windows"])
            self.assertEqual(len({tuple(k.values()) for k in keys}), 12)
            self.assertEqual(sorted(Counter(k["seed"] for k in keys).values()), [1] * 4 + [2] * 4)
            new = [c["precheck_seed"], *c["wide_training_prefix_seeds"], *c["readout_prefix_seeds"]]
            self.assertFalse(set(new) & (seen | set(c["development_prefix_seeds"])))
            seen.update(new)
            self.assertEqual((job["cpu"], job["ram_mb"], job["vram"]), (1, 2048, 0))
            self.assertIsNone(job["require_node"])
            self.assertIsNone(job["preferred_node"])
            self.assertIn("precheck", job["signature"])
        dense = 2 * (8 * 3 * 4 * 6 + 12) * 1200
        wide = 24 * (1 + 3 * 4 * 2 * 2) * 1200
        self.assertEqual(dense, wide)
        self.assertEqual(dense, m["dense"]["new_steps_per_root"])
        self.assertEqual(sum(m["dense"]["planned_shard_steps"]), dense)
        self.assertEqual(sum(m["wide"]["planned_shard_steps"]), wide)
        self.assertEqual((m["dense"]["training_pairs"], m["wide"]["training_pairs"]), (780, 768))
        self.assertEqual(m["readout"]["new_steps_per_root"], 8 * (1 + 3 * 4 * 8 * 2) * 1200)
        self.assertEqual(m["formal_total_steps"], 4 * (2 * dense + m["readout"]["new_steps_per_root"]))
        self.assertEqual(m["precheck_total_steps"], 4 * 19 * 1200)
        self.assertEqual(m["total_with_prechecks"], m["formal_total_steps"] + m["precheck_total_steps"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            submitter.prepare(source, draft_plan(), {m["cells"][0]["precheck_seed"]}, "fixture", Path("protocols/fixture.json"))

    def test_old_A_and_exposed_B_indices_are_rejected_before_native_access(self):
        with patch.object(info, "run_episode") as native:
            with self.assertRaisesRegex(ValueError, "exposed B"):
                info.acquire_windows(None, None, {}, {0: [2]}, spec=None, args=options())
            native.assert_not_called()

    def test_native_cached_extension_and_fresh_precheck_preserve_context_and_exact_cost(self):
        args = options()
        args.horizon = 500
        spec = shared.PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id)
        controller, rule = Controller(), clock_rule()
        with patch.object(shared, "_make_task", side_effect=lambda **kw: DenseTask(**{k: v for k, v in kw.items() if k != "env_id"})), \
             patch.object(shared, "pointmaze_goal_bounds", return_value=(np.full(2, -2), np.full(2, 2))):
            source = observe(controller, rule, spec=spec, args=args, seed=71, probes_per_path=1)
            cached = collect_boundary_pair(controller, rule, source, 55, spec=spec, args=args, windows=4, suffixes=4)
            before = deepcopy(cached)
            ledger = ComputeLedger()
            with ledger.recording("new_precheck"):
                rows = info.native_precheck(controller, rule, cached, spec=spec, args=args, seed=72)
            self.assertEqual(ledger.summary()["counts"]["environment_steps"], 19 * args.horizon)
            self.assertEqual(ledger.summary()["counts"].get("ridge_fit_calls", 0), 0)
            self.assertEqual(cached, before)
            for name, r in rows.items():
                self.assertEqual(r["boundary_step"], r["opportunity_step"] + 50)
                for w in r["windows"]:
                    self.assertEqual([d["suffix_index"] for d in w["A_replicas"]], [1000, 1001])
                    for d in w["A_replicas"]:
                        self.assertAlmostEqual(d["W"] + d["tail"], d["G"])
                    if name == "cached_extension":
                        self.assertEqual(w["boundary_context"], cached["windows"][w["window_index"]]["boundary_context"])
            bad = deepcopy(cached)
            bad["windows"][0]["boundary_context"]["now"]["offset_fraction"] += 1
            with self.assertRaisesRegex(RuntimeError, "boundary context"):
                info.acquire_windows(controller, rule, bad, {0: [1000]}, spec=spec, args=args)

    def test_missing_cache_prevents_controller_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "source.json").write_text('{}')
            (root / "variance.json").write_text(json.dumps({"source_manifest": "old.json"}))
            (root / "old.json").write_text(json.dumps({"cells": [{"task": "fixture", "optimizer_seed": 17}]}))
            with patch.object(worker, "ROOT", root), patch.object(worker, "load_cache", side_effect=FileNotFoundError), \
                 patch.object(worker, "load_sources") as load:
                with self.assertRaises(FileNotFoundError):
                    worker.run_cell({"task": "fixture", "optimizer_seed": 17},
                        {"source_manifest": "source.json", "source_variance_manifest": "variance.json"})
                load.assert_not_called()

    def test_worker_preserves_failed_native_cost_and_refuses_overwrite(self):
        @contextmanager
        def runtime():
            yield {"fixture": True}

        def fail(*_):
            count_work("environment_steps", 7)
            raise RuntimeError("fixture context failure")

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            p = root / "manifest.json"
            p.write_text(json.dumps({"run_name": "fixture", "source_hardware_profile": {"fixture": True},
                "precheck_steps_per_root": 22800, "cells": [{"task": "fixture", "optimizer_seed": 17, "output_directory": "out"}]}))
            with patch.object(worker, "ROOT", root), patch.object(worker, "single_thread_runtime", runtime), patch.object(worker, "run_cell", side_effect=fail):
                args = ["--manifest", str(p), "--cell-index", "0"]
                self.assertEqual(worker.main(args), 1)
                with self.assertRaises(FileExistsError):
                    worker.main(args)
            saved = json.loads((root / "out/precheck/public/result.json").read_text())
            self.assertEqual(saved["new_environment_steps"], 7)
            self.assertEqual(saved["status"], "failed")


if __name__ == "__main__":
    unittest.main()
