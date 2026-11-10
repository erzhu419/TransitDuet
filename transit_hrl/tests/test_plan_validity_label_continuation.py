from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments import plan_validity_label_continuation as experiment
from freq_hrl.experiments import plan_validity_phase_support as phase
from freq_hrl.experiments import plan_validity_transfer as shared
from freq_hrl.experiments.compute_budget import count_work
from freq_hrl.experiments.plan_validity_preregistration import draft_plan, seed_cells
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from scripts import analyze_plan_validity_label_continuation as analyzer
from scripts import run_plan_validity_label_continuation as worker
from scripts import submit_plan_validity_label_continuation_scheduleurm as submitter
from test_plan_validity_phase_support import clock_rule
from test_plan_validity_second_task import Controller, DenseTask, options
from test_plan_validity_target_credit import bank


def banks():
    rows = []
    for i, r in enumerate(bank()):
        replicas = [{**v, "future_seed": 100 + i * 8 + j} for j, v in enumerate(r["conditional_replicas"])]
        rows.append(phase.clock_row({**r, "factual_stop": False, "A_replicas": replicas[:1], "B_replicas": replicas[1:5]}, horizon=1200))
    return {c: deepcopy(rows) for c in experiment.CONTINUATIONS}


class LabelContinuationTest(unittest.TestCase):
    def test_candidate_only_checks_are_sampled_with_shared_states_and_futures(self):
        args = options()
        args.horizon = 500
        spec = shared.PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id)
        rules = {c: clock_rule() for c in experiment.CONTINUATIONS}
        rules["reference"].threshold = 0
        with patch.object(shared, "_make_task", side_effect=lambda **kw: DenseTask(**{k: v for k, v in kw.items() if k != "env_id"})), \
             patch.object(shared, "pointmaze_goal_bounds", return_value=(np.full(2, -2), np.full(2, 2))):
            obs = phase.observe(Controller(), rules["candidate"], spec=spec, args=args, seed=71, probes_per_path=3)
            ref = shared.run_episode(Controller(), task_spec=spec, seed=71, args=args, time_scale=shared.time_contract(args), rule=rules["reference"])[1]
            self.assertNotIn(55, [q["step"] for q in ref["queries"]])
            paired = experiment.sample_labels(Controller(), rules, obs, 55, spec=spec, args=args, a=1, b=4, tag=1003)
        result = {c: [paired[c]] for c in experiment.CONTINUATIONS}
        experiment.validate_banks(result, seeds=[71], probes=1, a=1, b=4)
        self.assertEqual(paired["candidate"]["offset_fraction"], 0.2)
        self.assertEqual(len(paired["candidate"]["B_replicas"]), 4)
        for r in paired["candidate"]["A_replicas"] + paired["candidate"]["B_replicas"]:
            self.assertAlmostEqual(r["full_advantage"], r["adaptive_window_advantage"] + r["tail_advantage"])

    def test_B_cannot_fit_or_calibrate_and_both_heads_keep_DF8(self):
        data = banks()
        experiment.validate_banks(data, seeds=list(range(8)), probes=3, a=1, b=4)
        rules, reports = experiment.fit_labels(data, effective_df=8, threshold_quantile=0.75)
        changed = deepcopy(data)
        for rows in changed.values():
            for row in rows:
                for replica in row["B_replicas"]:
                    replica["adaptive_window_advantage"] += 1000
        again, _ = experiment.fit_labels(changed, effective_df=8, threshold_quantile=0.75)
        self.assertEqual({k: v.to_dict() for k, v in rules.items()}, {k: v.to_dict() for k, v in again.items()})
        self.assertEqual(rules["label_reference"].threshold, rules["label_candidate"].threshold)
        for name, rule in rules.items():
            self.assertAlmostEqual(rule.model["model"]["effective_df_slopes"], 8)
            self.assertEqual(rule.model["model"]["coefficient_count"], 42)
            self.assertEqual(reports[name]["own"]["fitting_future_count"], 24)

    def test_mismatched_features_or_overlapping_A_B_futures_are_rejected(self):
        data = banks()
        data["candidate"][0]["causal_features"][0] += 1
        with self.assertRaisesRegex(ValueError, "identical observable"):
            experiment.validate_banks(data, seeds=list(range(8)), probes=3, a=1, b=4)
        data = banks()
        for rows in data.values():
            rows[0]["B_replicas"][0]["future_seed"] = rows[0]["A_replicas"][0]["future_seed"]
        with self.assertRaisesRegex(ValueError, "futures overlap"):
            experiment.validate_banks(data, seeds=list(range(8)), probes=3, a=1, b=4)

    def test_action_contrasts_cancel_shared_G_noise_before_path_intervals(self):
        data = banks()
        for rows in data.values():
            for row in rows:
                row["A_replicas"] = [{"adaptive_window_advantage": v, "full_advantage": 6} for v in (1, 2, 3, 4)]
                row["B_replicas"] = [{"adaptive_window_advantage": v, "full_advantage": 6} for v in (5, 6, 7, 8)]
        rules = {k: clock_rule() for k in ("source_R", "source_C", *experiment.HEADS)}
        for k, r in rules.items():
            r.threshold = -1 if k == "label_candidate" else 2
        result = experiment.single_step_readout(data, rules)
        for block in ("A", "B", "all"):
            effect = result[block]["candidate"]["candidate_minus_reference_label"]["gains"]["G"]
            self.assertEqual(effect["mean"], 6)
            self.assertEqual(effect["conditional_MC_standard_error"], 0)
        merged = analyzer.aggregate_paths([{"blocks": result}] * 8)
        self.assertEqual(merged["all"]["candidate"]["candidate_minus_reference_label"]["gains"]["G"]["path_interval"]["interval"], [6, 6])

    def test_fresh_probe_roster_and_resource_scope_are_frozen_without_submission(self):
        source = json.loads((submitter.ROOT / "protocols/plan_validity_continuation_2026-11-10.json").read_text())
        plan = draft_plan()
        manifest = submitter.prepare(source, plan, "fixture", Path("protocols/fixture.json"))
        exposed = {s for c in source["cells"] for s in c["evaluation_seeds"] + c["previous_evaluation_seeds"] + c["exposed_held_prefix_seeds"]}
        reserved = {s for c in seed_cells(plan) for seeds in c["seed_roles"].values() for s in seeds}
        self.assertEqual([len(manifest["jobs"][p]) for p in ("precheck", "sample", "fit", "probe")], [4, 8, 4, 8])
        for c, old in zip(manifest["cells"], source["cells"]):
            self.assertEqual(c["fit_prefix_seeds"], old["fit_prefix_seeds"])
            self.assertFalse(set(c["evaluation_seeds"]) & (exposed | reserved))
        for jobs in manifest["jobs"].values():
            for job in jobs:
                self.assertEqual((job["cpu"], job["ram_mb"], job["vram"]), (1, 2048, 0))
                self.assertIsNone(job["require_node"])
                self.assertIsNone(job["preferred_node"])
                self.assertTrue(job["result_dir"].endswith("/public"))
        with tempfile.TemporaryDirectory(dir=submitter.ROOT / "protocols") as folder:
            with patch.object(submitter, "execute_bulk") as scheduled:
                submitter.main(["--source-manifest", str(submitter.ROOT / "protocols/plan_validity_continuation_2026-11-10.json"),
                    "--run-name", "fixture", "--manifest", str(Path(folder) / "manifest.json")])
                scheduled.assert_not_called()
            incomplete = analyzer.summarize(manifest, root=Path(folder))
            self.assertEqual(len(incomplete["missing"]), 24)
            self.assertEqual(incomplete["trained_roots_per_task"]["pointmaze"], 2)

    def test_sampler_failure_keeps_ledger_and_never_overwrites_attempt(self):
        @contextmanager
        def runtime():
            yield {"cpu": "fixture"}

        def unexpected_fit(*_):
            count_work("ridge_fit_calls")
            return {"stage": "sample_complete"}, None

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "manifest.json"
            path.write_text(json.dumps({"run_name": "fixture", "source_hardware_profile": {"cpu": "fixture"},
                "cells": [{"task": "pointmaze", "optimizer_seed": 1, "output_directory": "out"}]}))
            with patch.object(worker, "ROOT", root), patch.object(worker, "single_thread_runtime", runtime), \
                 patch.object(worker, "run_cell", side_effect=unexpected_fit):
                args = ["--manifest", str(path), "--cell-index", "0", "--phase", "sample", "--shard-index", "0"]
                self.assertEqual(worker.main(args), 1)
                with self.assertRaises(FileExistsError):
                    worker.main(args)
            public = json.loads((root / "out/sample/shard0/public/result.json").read_text())
            private = json.loads((root / "out/sample/shard0/private/result.json").read_text())
            self.assertEqual(public["measured_cost"], private["measured_cost"])
            self.assertEqual(public["measured_cost"]["counts"]["ridge_fit_calls"], 1)
            self.assertFalse(public["confirmation_released"])

    def test_fit_reuses_paid_banks_and_new_paths_wait_for_every_frozen_root(self):
        source = json.loads((submitter.ROOT / "protocols/plan_validity_continuation_2026-11-10.json").read_text())
        manifest = submitter.prepare(source, draft_plan(), "fixture", Path("protocols/fixture.json"))
        cell = manifest["cells"][0]
        data = banks()
        expanded = {c: [] for c in experiment.CONTINUATIONS}
        for c in experiment.CONTINUATIONS:
            for i, seed in enumerate(cell["fit_prefix_seeds"]):
                expanded[c].extend({**deepcopy(r), "seed": seed} for r in data[c][3 * (i % 8):3 * (i % 8 + 1)])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            public = root / cell["phase_public_result"]
            public.parent.mkdir(parents=True)
            public.write_text(json.dumps({"status": "complete", "optimizer_seed": cell["optimizer_seed"]}))
            private = root / cell["phase_cache_directory"]
            private.mkdir(parents=True)
            (private / "frozen_rules.json").write_text(json.dumps({"checks_clock": clock_rule().to_dict()}))

            def completed(phase_name, shard, extra):
                directory = root / cell["output_directory"] / phase_name
                if shard >= 0:
                    directory /= f"shard{shard}"
                (directory / "public").mkdir(parents=True, exist_ok=True)
                (directory / "private").mkdir(exist_ok=True)
                (directory / "public/result.json").write_text(json.dumps({"status": "complete", "run_name": "fixture",
                    "task": cell["task"], "optimizer_seed": cell["optimizer_seed"], "phase": phase_name, "shard_index": shard, **extra}))
                return directory

            completed("precheck", -1, {})
            for shard in (0, 1):
                directory = completed("sample", shard, {"sampled_seeds": cell["fit_prefix_seeds"][16 * shard:16 * (shard + 1)]})
                (directory / "private/label_banks.json").write_text(json.dumps({c: rows[48 * shard:48 * (shard + 1)] for c, rows in expanded.items()}))
            directory = completed("fit", -1, {})
            records = {"termination_history": {"termination_readiness": {"eligible_for_fresh_confirmation": False}}}
            with patch.object(worker, "ROOT", root), patch.object(worker, "load_sources", return_value=(records,
                {"history": object()}, {"history": clock_rule()}, SimpleNamespace(**cell["expected_configuration"]))), \
                 patch.object(worker, "observe") as observed:
                report, _ = worker.run_cell(cell, manifest, "fit", -1, directory, lambda _: None)
                observed.assert_not_called()
                self.assertEqual(report["fitting_pairs_per_head"], 96)
                self.assertEqual(report["training_paths"], 32)
                self.assertEqual(set(json.loads((directory / "private/frozen_rules.json").read_text())), set(experiment.HEADS))
                with self.assertRaises(FileNotFoundError):
                    worker.run_cell(cell, manifest, "probe", 0, root / "unopened_probe", lambda _: None)
                observed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
