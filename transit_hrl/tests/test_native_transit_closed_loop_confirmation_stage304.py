import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_closed_loop_confirmation_stage304 as spec
from scripts import submit_native_transit_closed_loop_confirmation_stage304_scheduleurm as submit


class ClosedLoopConfirmationTests(unittest.TestCase):
    def test_remote_freeze_needs_native_report_not_local_dispatch_receipt(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            report = base/spec.AUDIT/"report/result.json"
            report.parent.mkdir(parents=True); report.write_text(json.dumps({"all_pilots_qualified": True}))
            with patch.object(spec, "directory", side_effect=lambda run: base/run), patch.object(spec, "contract", return_value={}):
                spec.freeze("unit")
            r = json.loads((base/"unit/freeze/result.json").read_text())
            self.assertTrue(r["software_qualified"])
            self.assertEqual((r["scene_units"], r["distinct_inner_seeds"]), (320, 10240))

    def fixture(self):
        values = np.zeros((10, 32, 4, 4))
        values[:, :, 1] = [-2., -1., -1., -2.]
        values[:, :, 2] = [-3., -1., -2., -3.]
        values[:, :, 3] = [-4., -1., -3., -3.]
        return values

    def test_full_episode_gates_and_paired_comparison_sign(self):
        r = spec.summarize(self.fixture())
        self.assertTrue(r["service_closed_loop_supported"])
        self.assertTrue(r["joint_closed_loop_supported"])
        self.assertTrue(r["critic_incremental_cost_supported"])
        self.assertAlmostEqual(r["comparisons"]["all32_scenes"]["joint_budgeted_vs_service_only"]["mean"]["cost"], -1.)
        self.assertEqual(r["wait_retention_vs_base"]["service_only"]["all32_scenes"], 1.5)

    def test_half_regression_and_wait_retention_cannot_hide_in_pooled_gain(self):
        values = self.fixture(); values[:, 16:, 2, 0] = 1.
        self.assertFalse(spec.summarize(values)["service_closed_loop_supported"])
        values = self.fixture(); values[:, 16:, 2, 3] = -.5
        self.assertFalse(spec.summarize(values)["service_closed_loop_supported"])

    def test_no_effect_and_nonimproving_base_do_not_create_retention_claim(self):
        result = spec.summarize(np.zeros((10, 32, 4, 4)))
        self.assertFalse(result["service_closed_loop_supported"])
        self.assertFalse(result["joint_closed_loop_supported"])
        self.assertFalse(result["critic_incremental_cost_supported"])
        self.assertIsNone(result["wait_retention_vs_base"]["service_only"]["all32_scenes"])

    def test_shared_scene_noise_cancels_between_controllers(self):
        values = self.fixture()
        noise = np.random.default_rng(3).normal(0, 4., (10, 32))
        values[:, :, 2, 0] = noise-3.; values[:, :, 3, 0] = noise-4.
        r = spec.summarize(values)["comparisons"]["all32_scenes"]["joint_budgeted_vs_service_only"]
        np.testing.assert_allclose(r["episode_bootstrap_CI95"]["cost"], [-1., -1.], atol=1e-12, rtol=0.)
        np.testing.assert_allclose(r["paired_normal_CI95"]["cost"], [-1., -1.], atol=1e-12, rtol=0.)

    def test_formal_context_keeps_p8_s8_but_uses_new_scene_and_inner_seeds(self):
        scenes = {spec.scene_seed(r, s, n) for r in spec.source.source.ROOTS for s in spec.source.source.SCENARIOS for n in spec.REPLICAS}
        inner = {spec.future_seed(r, s, n, p, d) for r in spec.source.source.ROOTS for s in spec.source.source.SCENARIOS for n in spec.REPLICAS for p in spec.source.PROBES for d in range(16)}
        pilot_scenes = {spec.source.scene_seed(r, s) for r in spec.source.source.ROOTS for s in spec.source.source.SCENARIOS}
        pilot_inner = {spec.source.future_seed(r, s, p, d) for r in spec.source.source.ROOTS for s in spec.source.source.SCENARIOS for p in spec.source.PROBES for d in range(16)}
        self.assertEqual((len(scenes), len(inner)), (320, 10240))
        self.assertFalse(scenes & pilot_scenes); self.assertFalse(inner & pilot_inner); self.assertFalse(scenes & inner)
        self.assertLess(max(inner|scenes), 2**31)
        c = spec.context("unit", 7)
        self.assertEqual(c.scene_seed(397, "low_noise"), spec.scene_seed(397, "low_noise", 7))
        self.assertEqual(c.BANKS, spec.source.BANKS)
        self.assertIn("replicas/7", str(c.directory("unit")))

    def test_base_is_the_unchanged_unthresholded_screen(self):
        raw = np.zeros((16, 1), int); proposal = np.ones_like(raw)
        panel = np.zeros((1, 8, 6, 4)); panel[:, :, 1, 1] = -.1
        with patch.object(spec.source.parent.native, "controls", return_value=(raw, raw, np.zeros((16, 1, 6)))), \
                patch.object(spec.source.parent.native, "proposals", return_value=proposal), \
                patch.object(spec.source.source, "read", return_value={"contract": {"locked_policy": {}}}):
            self.assertEqual(spec.source.choose("unit", "zero_abstain", {}, panel, panel), 1)
            panel[:, :, 1, 1] = 0.
            self.assertEqual(spec.source.choose("unit", "zero_abstain", {}, panel, panel), 0)

    def test_budget_and_unpinned_native_scheduler_tasks(self):
        jobs = submit.tasks("unit_stage304", "evaluate")
        self.assertEqual(len(jobs), 1280)
        self.assertEqual(len({t["signature"] for t in jobs}), 1280)
        self.assertEqual((jobs[0]["cpu"], jobs[0]["ram_mb"], jobs[0]["vram"]), (1, 1536, 0))
        self.assertIsNone(jobs[0]["require_node"])
        self.assertEqual(jobs[0]["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        with patch.object(spec.source, "contract", return_value={"locked_policy": {}}):
            c = spec.contract()
        self.assertEqual(c["budget"]["native_episodes_including_inner_and_identity"], 320*(5+3*227))
        self.assertIn("neither_is_old_fixed_prefix_conditional_MC", c["CI_boundary"])


if __name__ == "__main__": unittest.main()
