from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments import plan_validity_independent_calibration as experiment
from freq_hrl.experiments import plan_validity_transfer as shared
from freq_hrl.experiments.compute_budget import ComputeLedger, count_work
from freq_hrl.experiments.plan_validity_preregistration import draft_plan
from scripts import run_plan_validity_independent_calibration as worker
from scripts import submit_plan_validity_independent_calibration_scheduleurm as submitter
from test_plan_validity_label_continuation import banks
from test_plan_validity_phase_support import clock_rule
from test_plan_validity_second_task import Controller, DenseTask, options


def artifacts(root):
    rule = clock_rule()
    heads = {name: deepcopy(rule) for name in ("label_reference", "label_candidate")}
    for name, value in (("phase/frozen_rules.json", {"checks_clock": rule.to_dict()}),
                        ("fit/private/frozen_rules.json", {k: v.to_dict() for k, v in heads.items()}),
                        ("previous/private/frozen_cuts.json", {"frozen_full_scores": {k: {"cut": .2} for k in heads}})):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    return heads


def complete(root, cell, phase, shard=-1, **extra):
    directory = root / cell["output_directory"] / phase
    if shard >= 0:
        directory /= f"shard{shard}"
    (directory / "public").mkdir(parents=True, exist_ok=True)
    (directory / "private").mkdir(exist_ok=True)
    (directory / "public/result.json").write_text(json.dumps({"run_name": "fixture", "status": "complete",
        "task": cell["task"], "optimizer_seed": cell["optimizer_seed"], "phase": phase, "shard_index": shard, **extra}))
    return directory


class IndependentCalibrationTest(unittest.TestCase):
    def test_B_never_changes_A_selected_cuts_or_frozen_models(self):
        rows = banks()["candidate"]
        heads = {k: clock_rule() for k in ("label_reference", "label_candidate")}
        frozen = {k: v.to_dict() for k, v in heads.items()}
        first = experiment.select_cuts(rows, heads)
        changed = deepcopy(rows)
        for row in changed:
            for v in row["B_replicas"]:
                v["adaptive_window_advantage"] += 10000
                v["full_advantage"] -= 10000
        self.assertEqual(first, experiment.select_cuts(changed, heads))
        self.assertEqual(frozen, {k: v.to_dict() for k, v in heads.items()})

    def test_fresh_readout_uses_all_E8_and_pairs_before_prefix_intervals(self):
        rows = banks()["candidate"]
        for row in rows:
            row["A_replicas"] = [{"adaptive_window_advantage": 2, "full_advantage": 7}] * 4
            row["B_replicas"] = [{"adaptive_window_advantage": 3, "full_advantage": 9}] * 4
        heads = {k: clock_rule() for k in ("label_reference", "label_candidate")}
        for head in heads.values():
            head.threshold = 2
        selected, previous = ({k: {"cut": cut} for k in heads} for cut in (-2, 2))
        result = experiment.cut_readout(rows, heads, selected, previous, fresh_prefix=True)
        calibration = experiment.cut_readout(rows, heads, selected, previous, fresh_prefix=False)
        for head in heads:
            self.assertEqual(result[head]["future_count"], 8)
            self.assertEqual(calibration[head]["future_count"], 4)
            for block, expected in (("first_half", 7), ("second_half", 9), ("all", 8)):
                r = result[head]["panels"][block]["independent_minus_quantile"]["G"]
                self.assertEqual(r["mean"], expected)
                self.assertEqual(r["prefix_interval"]["n"], 8)
                self.assertEqual(r["prefix_interval"]["interval"], [expected, expected])
            self.assertEqual(result[head]["panels"]["first_half"]["independent_minus_C0"]["G"]["conditional_MC_standard_error"], 0)

    def test_manifest_freezes_disjoint_roles_exact_budgets_and_no_default_submission(self):
        source = json.loads((submitter.ROOT / "protocols/plan_validity_label_continuation_2026-11-10.json").read_text())
        prior = json.loads((submitter.ROOT / "protocols/plan_validity_score_source_2026-11-10.json").read_text())
        m = submitter.prepare(source, prior, draft_plan(), "fixture", Path("protocols/fixture.json"), set())
        seen = submitter.literal_seeds(source) | submitter.literal_seeds(prior)
        for cell in m["cells"]:
            new = cell["precheck_seeds"] + cell["calibration_prefix_seeds"] + cell["readout_prefix_seeds"]
            self.assertFalse(seen & set(new))
            self.assertEqual(len(new), 28)
            seen.update(new)
        self.assertEqual([len(m["jobs"][p]) for p in ("precheck", "sample", "select", "readout")], [4, 8, 4, 4])
        self.assertEqual(m["total_environment_step_upper_bound"], 4 * 12000 + 12 * 8 * 1200 * (1 + 3 * 2 * 9))
        self.assertEqual(m["ridge_fit_calls"], 0)
        self.assertFalse(m["confirmation_released"])
        for jobs in m["jobs"].values():
            for job in jobs:
                self.assertEqual((job["cpu"], job["ram_mb"], job["vram"]), (1, 2048, 0))
                self.assertIsNone(job["require_node"])
                self.assertIsNone(job["preferred_node"])
                self.assertTrue(job["result_dir"].endswith("/public"))
        with self.assertRaisesRegex(ValueError, "overlap"):
            submitter.prepare(source, prior, draft_plan(), "fixture", Path("protocols/fixture.json"), {source["cells"][0]["optimizer_seed"] + 62000})
        original_root = submitter.ROOT
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "protocols").mkdir()
            with patch.object(submitter, "ROOT", root), patch.object(submitter, "execute_bulk") as scheduled:
                submitter.main(["--source-manifest", str(original_root / "protocols/plan_validity_label_continuation_2026-11-10.json"),
                    "--previous-calibration", str(original_root / "protocols/plan_validity_score_source_2026-11-10.json"),
                    "--preregistration", str(original_root / "protocols/plan_validity_preregistration_2026-10-22.json"),
                    "--run-name", "fixture", "--manifest", str(root / "protocols/manifest.json")])
                scheduled.assert_not_called()

    def test_readout_waits_for_all_roots_before_any_native_access(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = {"task": "pointmaze", "optimizer_seed": 1, "output_directory": "first"}
            second = {"task": "reacher_tracking", "optimizer_seed": 2, "output_directory": "second"}
            complete(root, first, "select")
            with patch.object(worker, "ROOT", root), patch.object(worker, "load_sources") as load:
                with self.assertRaises(FileNotFoundError):
                    worker.run_cell(first, {"run_name": "fixture", "cells": [first, second]}, "readout", -1, root, lambda _: None)
                load.assert_not_called()

    def test_native_precheck_replays_C0_and_both_head_score_paths(self):
        args = options()
        args.horizon = 500
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifacts(root)
            cell = {"task": "pointmaze", "optimizer_seed": args.optimizer_seed, "phase_cache_directory": "phase",
                "fit_directory": "fit", "previous_calibration_directory": "previous", "precheck_seeds": [71, 72, 73, 74]}
            ledger = ComputeLedger()
            with patch.object(worker, "ROOT", root), patch.object(worker, "load_sources", return_value=({}, {"history": Controller()}, {}, args)), \
                 patch.object(shared, "_make_task", side_effect=lambda **kw: DenseTask(**{k: v for k, v in kw.items() if k != "env_id"})), \
                 patch.object(shared, "pointmaze_goal_bounds", return_value=(np.full(2, -2), np.full(2, 2))), ledger.recording("native"):
                result, rows = worker.run_cell(cell, {"horizon": 500}, "precheck", -1, root, lambda _: None)
            self.assertEqual(result["stage"], "precheck_passed")
            self.assertGreater(result["nonzero_phase"], 0)
            self.assertEqual(result["upper_calls"], 10)
            self.assertEqual(len(rows), 1)
            self.assertEqual(ledger.summary()["counts"]["environment_steps"], 3500)
            self.assertEqual(ledger.summary()["counts"].get("ridge_fit_calls", 0), 0)

    def test_paid_calibration_cuts_are_frozen_before_B_without_fitting(self):
        source = banks()["candidate"]
        seeds = list(range(1000, 1016))
        rows = []
        for i, seed in enumerate(seeds):
            for base in source[3 * (i % 8):3 * (i % 8 + 1)]:
                row = {**deepcopy(base), "seed": seed}
                for block, replicas in (("A", [row["A_replicas"][0]] * 4), ("B", row["B_replicas"])):
                    row[block + "_replicas"] = [{**v, "future_seed": 1000 + len(rows) * 8 + j + (4 if block == "B" else 0)} for j, v in enumerate(replicas)]
                rows.append(row)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            heads = artifacts(root)
            frozen = (root / "fit/private/frozen_rules.json").read_text()
            cell = {"task": "pointmaze", "optimizer_seed": 17, "output_directory": "out", "phase_cache_directory": "phase",
                "fit_directory": "fit", "previous_calibration_directory": "previous", "calibration_prefix_seeds": seeds}
            complete(root, cell, "precheck")
            for shard in (0, 1):
                path = complete(root, cell, "sample", shard, sampled_seeds=seeds[8 * shard:8 * (shard + 1)])
                (path / "private/rows.json").write_text(json.dumps(rows[24 * shard:24 * (shard + 1)]))
            output = root / "select"
            (output / "private").mkdir(parents=True)

            def readout(*a, **kw):
                self.assertEqual(set(json.loads((output / "private/frozen_cuts.json").read_text())), set(heads))
                return experiment.cut_readout(*a, **kw)

            ledger = ComputeLedger()
            with patch.object(worker, "ROOT", root), patch.object(worker, "load_sources", return_value=({}, {"history": Controller()}, {}, options())), \
                 patch.object(worker, "cut_readout", side_effect=readout), ledger.recording("select"):
                result, _ = worker.run_cell(cell, {"run_name": "fixture", "cells": [cell], "horizon": 300}, "select", -1, output, lambda _: None)
            self.assertEqual(result["stage"], "cuts_frozen")
            self.assertEqual(ledger.summary()["counts"].get("environment_steps", 0), 0)
            self.assertEqual(ledger.summary()["counts"].get("ridge_fit_calls", 0), 0)
            self.assertEqual((root / "fit/private/frozen_rules.json").read_text(), frozen)

    def test_failed_attempt_preserves_work_and_cannot_overwrite(self):
        @contextmanager
        def runtime():
            yield {"cpu": "fixture"}

        def failed(*_):
            count_work("ridge_fit_calls")
            count_work("environment_steps", 7)
            return {"stage": "precheck_passed"}, None

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "manifest.json"
            path.write_text(json.dumps({"run_name": "fixture", "source_hardware_profile": {"cpu": "fixture"},
                "per_job_environment_steps": {"precheck": 12000},
                "cells": [{"task": "pointmaze", "optimizer_seed": 1, "output_directory": "out"}]}))
            with patch.object(worker, "ROOT", root), patch.object(worker, "single_thread_runtime", runtime), patch.object(worker, "run_cell", side_effect=failed):
                args = ["--manifest", str(path), "--cell-index", "0", "--phase", "precheck"]
                self.assertEqual(worker.main(args), 1)
                with self.assertRaises(FileExistsError):
                    worker.main(args)
            result = json.loads((root / "out/precheck/public/result.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["new_environment_steps"], 7)
            self.assertFalse(result["confirmation_released"])


if __name__ == "__main__":
    unittest.main()
