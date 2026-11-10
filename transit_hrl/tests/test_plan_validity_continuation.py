from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments import plan_validity_continuation as diagnostic
from freq_hrl.experiments import plan_validity_transfer as shared
from freq_hrl.experiments.compute_budget import count_work
from freq_hrl.experiments.plan_validity_preregistration import draft_plan
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from scripts import analyze_plan_validity_continuation as analyzer
from scripts import read_plan_validity_continuation_cache as cache_reader
from scripts import run_plan_validity_continuation as worker
from scripts import submit_plan_validity_continuation_scheduleurm as submitter
from test_plan_validity_phase_support import clock_rule
from test_plan_validity_second_task import Controller, DenseTask, options


def rules():
    result = {k: clock_rule() for k in diagnostic.POLICIES}
    result["reference"].name, result["reference"].threshold = "reference", 0.4
    result["candidate"].name, result["candidate"].threshold = "candidate", 0.8
    return result


def toy_probe():
    shifts = dict(zip(diagnostic.CONTEXTS, (0, 1, 2, 4)))
    return {"candidate_scores": {p: 0 for p in diagnostic.POLICIES},
        "actions": {p: {"reference": False, "candidate": True} for p in diagnostic.POLICIES},
        "effects": {c: [{"W": x, "tail": shift, "G": x + shift} for x in (1, 2, 3, 4)]
                    for c, shift in shifts.items()}}


class ContinuationTest(unittest.TestCase):
    def plant(self):
        return patch.object(shared, "_make_task", side_effect=lambda **kw: DenseTask(**{k: v for k, v in kw.items() if k != "env_id"}))

    def bounds(self):
        return patch.object(shared, "pointmaze_goal_bounds", return_value=(np.full(2, -2), np.full(2, 2)))

    def test_same_continuation_is_identical_and_switch_is_after_the_forced_action(self):
        args, policies = options(), rules()
        spec = shared.PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id)
        late = policies["candidate"]
        early = clock_rule()
        early.name, early.threshold = "early", 0
        with self.plant(), self.bounds():
            def run(**kw):
                return shared.run_episode(Controller(), task_spec=spec, seed=71, args=args,
                    time_scale=shared.time_contract(args), rule=late, intervention_step=55, arm="wait_one_check", **kw)[1]
            old = run()
            same = run(continuation_rule=late)
            switched = run(continuation_rule=early)
        self.assertEqual(old["decision_steps"], same["decision_steps"])
        self.assertEqual(old["tracking_squared_error_integral"], same["tracking_squared_error_integral"])
        np.testing.assert_array_equal(old["prefix"], switched["prefix"])
        self.assertEqual(old["decision_steps"][1], 70)
        self.assertEqual(switched["decision_steps"][1], 60)
        self.assertFalse(next(q for q in switched["queries"] if q["step"] == 55)["stop"])
        self.assertEqual(switched["upper_decision_count"], 6)
        self.assertEqual(switched["continuation_policy"], "early")

    def test_switch_requires_a_timing_intervention(self):
        args = options()
        spec = shared.PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id)
        with patch.object(shared.PlanValidityTaskSpec, "open") as opened:
            with self.assertRaisesRegex(ValueError, "adaptive timing intervention"):
                shared.run_episode(Controller(), task_spec=spec, seed=71, args=args,
                    time_scale=shared.time_contract(args), rule=clock_rule(), continuation_rule=clock_rule())
            opened.assert_not_called()

    def test_paired_probe_keeps_prefixes_and_couples_all_four_context_futures(self):
        args, policies = options(), rules()
        args.horizon = 500
        spec = shared.PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id)
        with self.plant(), self.bounds():
            factual = diagnostic.factual_pair(Controller(), policies, spec=spec, args=args, seed=71)
            checks, support = diagnostic.selected_checks(factual, args=args, seed=71, probes_per_path=3)
            self.assertTrue(all(s in support["common"] for s in checks))
            self.assertGreater(support["counts"]["candidate_only"], 0)
            probe = diagnostic.paired_probe(Controller(), policies, factual, spec=spec, args=args, seed=71, step=105, replicas=4)
        futures = [[r["future_seed"] for r in probe["effects"][c]] for c in diagnostic.CONTEXTS]
        self.assertTrue(all(f == futures[0] for f in futures))
        for context in diagnostic.CONTEXTS:
            for row in probe["effects"][context]:
                self.assertAlmostEqual(row["G"], row["W"] + row["tail"])

    def test_shared_noise_cancels_before_MC_error_and_path_aggregation(self):
        summary = diagnostic.probe_summary([toy_probe(), toy_probe()])
        for block in ("A", "B", "all"):
            self.assertEqual(summary[block]["contrasts"]["continuation_at_reference"]["G"]["mean"], 1)
            self.assertEqual(summary[block]["contrasts"]["continuation_at_reference"]["G"]["conditional_MC_standard_error"], 0)
            self.assertEqual(summary[block]["contrasts"]["interaction"]["G"]["mean"], 1)
        result = analyzer.aggregate_paths([{"blocks": summary} for _ in range(8)])
        interval = result["all"]["contrasts"]["continuation_at_reference"]["G"]["path_interval"]
        self.assertEqual(interval["n"], 8)
        self.assertEqual(interval["interval"], [1, 1])
        self.assertEqual(result["all"]["context_action_path_intervals"]["candidate/candidate"]["candidate_action_gain_G"]["n"], 8)
        self.assertEqual(result["all"]["error_path_contrasts"]["continuation_at_reference"]["candidate_W_noise_corrected_mse"]["interval"], [0, 0])

    def test_cross_future_selection_uses_only_the_opposite_half(self):
        probe = toy_probe()
        for context in diagnostic.CONTEXTS:
            probe["effects"][context] = [{"W": w, "tail": g - w, "G": g}
                for w, g in zip((-2, -2, 1, 1), (1, 1, -3, -3))]
        result = diagnostic.cross_future_decisions([probe] * 3)
        context = "reference/reference"
        self.assertEqual(result["A_to_B"][context]["choices"]["W_sign"]["gain_G"], 0)
        self.assertEqual(result["A_to_B"][context]["choices"]["G_sign"]["gain_G"], -3)
        self.assertEqual(result["B_to_A"][context]["choices"]["W_sign"]["gain_G"], 1)
        self.assertEqual(result["all"][context]["choices"]["W_sign"]["gain_G"], 0.5)
        self.assertEqual(result["all"][context]["choices"]["G_sign"]["gain_G"], -1.5)
        self.assertEqual(result["all"][context]["W_G_choice_disagreement"], 1)
        frozen = diagnostic.probe_summary([probe] * 3)["all"]["contexts"][context]["candidate_action_gain_G"]
        self.assertEqual(result["all"][context]["choices"]["candidate"]["gain_G"], frozen)
        self.assertEqual(result["all"][context]["choices"]["score_zero"]["gain_G"], frozen)
        aggregate = cache_reader.aggregate_choices([{"blocks": result}] * 8)
        interval = aggregate["all"][context]["choices"]["W_sign"]["gain_G"]
        self.assertEqual(interval["n"], 8)
        self.assertEqual(interval["interval"], [0.5, 0.5])
        contrast = aggregate["all"][context]["paired_choice_contrasts"]["W_sign_minus_candidate"]["gain_G"]
        self.assertEqual(contrast["interval"], [1.5, 1.5])
        probe["candidate_scores"] = {p: -1 for p in diagnostic.POLICIES}
        result = diagnostic.cross_future_decisions([probe])
        self.assertEqual(result["all"][context]["choices"]["score_zero"]["gain_G"], 0)
        self.assertEqual(result["all"][context]["choices"]["candidate"]["gain_G"], frozen)

    def test_cross_future_selection_rejects_precheck_two_future_probes(self):
        probe = toy_probe()
        for context in diagnostic.CONTEXTS:
            probe["effects"][context] = probe["effects"][context][:2]
        with self.assertRaisesRegex(ValueError, "two halves"):
            diagnostic.cross_future_decisions([probe])

    def test_seed_and_shard_reservations_preserve_four_roots_and_public_only_sync(self):
        source = json.loads((submitter.ROOT / "protocols/plan_validity_phase_support_2026-11-10.json").read_text())
        manifest = submitter.prepare(source, draft_plan(), "fixture", Path("protocols/fixture.json"))
        self.assertEqual(len(manifest["cells"]), 4)
        self.assertEqual(len(manifest["jobs"]["precheck"]), 4)
        self.assertEqual(len(manifest["jobs"]["run"]), 8)
        for cell in manifest["cells"]:
            self.assertFalse(set(cell["evaluation_seeds"]) & set(cell["fit_prefix_seeds"] + cell["previous_evaluation_seeds"] + cell["exposed_held_prefix_seeds"]))
        for phase_jobs in manifest["jobs"].values():
            for job in phase_jobs:
                self.assertEqual((job["cpu"], job["ram_mb"], job["vram"]), (1, 2048, 0))
                self.assertIsNone(job["require_node"])
                self.assertIsNone(job["preferred_node"])
                self.assertTrue(job["result_dir"].endswith("/public"))
        with tempfile.TemporaryDirectory(dir=submitter.ROOT / "protocols") as folder:
            with patch.object(submitter, "execute_bulk") as schedule:
                submitter.main(["--source-manifest", str(submitter.ROOT / "protocols/plan_validity_phase_support_2026-11-10.json"),
                    "--run-name", "fixture", "--manifest", str(Path(folder) / "manifest.json")])
                schedule.assert_not_called()

    def test_incomplete_shards_are_not_counted_as_complete_or_extra_roots(self):
        source = json.loads((submitter.ROOT / "protocols/plan_validity_phase_support_2026-11-10.json").read_text())
        manifest = submitter.prepare(source, draft_plan(), "fixture", Path("protocols/fixture.json"))
        summary = diagnostic.probe_summary([toy_probe()] * 3)
        cell = manifest["cells"][0]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for shard in (0, 1):
                seeds = cell["evaluation_seeds"][4 * shard:4 * (shard + 1)]
                data = {"run_name": "fixture", "task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
                    "phase": "run", "shard_index": shard, "status": "complete", "evaluation_seeds": seeds,
                    "measured_cost": {"cpu_seconds": 10}, "occupancy": {}, "termination_development_eligible": False,
                    "paths": [{"seed": s, "blocks": summary, "episodes": {"reference": 1, "candidate": 2},
                               "support_counts": {"common": 3}, "probes": 3} for s in seeds]}
                p = root / cell["output_directory"] / f"run/shard{shard}/public/result.json"
                p.parent.mkdir(parents=True)
                p.write_text(json.dumps(data))
                result = analyzer.summarize(manifest, root=root)
                self.assertEqual(result["root_readouts"][0]["status"], "incomplete" if shard == 0 else "complete")
                self.assertEqual(result["trained_roots_per_task"]["pointmaze"], 2)
            self.assertFalse(result["confirmation_released"])
            data["paths"][0]["seed"] -= 4
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "path roster"):
                analyzer.summarize(manifest, root=root)

    def test_fitting_is_rejected_and_failure_keeps_measured_costs(self):
        @contextmanager
        def runtime():
            yield {"cpu": "fixture"}

        def unexpected_fit(*_):
            count_work("ridge_fit_calls")
            return {"stage": "complete"}, None

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "manifest.json"
            path.write_text(json.dumps({"run_name": "fixture", "source_hardware_profile": {"cpu": "fixture"},
                "cells": [{"task": "pointmaze", "optimizer_seed": 1, "output_directory": "out"}]}))
            with patch.object(worker, "ROOT", root), patch.object(worker, "single_thread_runtime", runtime), \
                 patch.object(worker, "run_cell", side_effect=unexpected_fit):
                self.assertEqual(worker.main(["--manifest", str(path), "--cell-index", "0", "--phase", "precheck", "--shard-index", "-1"]), 1)
            public = json.loads((root / "out/precheck/public/result.json").read_text())
            private = json.loads((root / "out/precheck/private/result.json").read_text())
            self.assertEqual(public["status"], "failed")
            self.assertEqual(public["measured_cost"]["counts"]["ridge_fit_calls"], 1)
            self.assertEqual(public["measured_cost"], private["measured_cost"])
            self.assertFalse(public["confirmation_released"])


if __name__ == "__main__":
    unittest.main()
