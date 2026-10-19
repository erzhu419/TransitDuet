import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_service_confirmation_stage298 as spec
from scripts import submit_native_transit_service_confirmation_stage298_scheduleurm as submit


class FreshServiceConfirmationTests(unittest.TestCase):
    def locked(self, root):
        p = root/"inputs/locked_policy.json"; p.parent.mkdir()
        p.write_text(json.dumps({"joint": {"family": "joint", "threshold": .002, "index": 27},
            "risk": {"family": "base", "threshold": None, "index": 0}}))

    def test_new_cohorts_and_future_namespaces(self):
        old = spec.parent.keys()+spec.parent.keys("pilot")
        fresh = spec.keys()+spec.keys("pilot")
        self.assertEqual((len(spec.keys()), len(spec.keys("pilot"))), (400, 10))
        self.assertFalse({spec.scene_seed(k) for k in old} & {spec.scene_seed(k) for k in fresh})
        before = {spec.future_seed(k, p, d) for k in old for p in spec.PROBES for d in range(64)}
        after = {spec.future_seed(k, p, d) for k in fresh for p in spec.PROBES for d in range(64)}
        self.assertFalse(before & after); self.assertEqual(len(after), 410*6*64)
        self.assertLess(max(after), 2**31)

    def test_freeze_records_actual_locked_cutoff(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); self.locked(root)
            with patch.object(spec, "directory", return_value=root):
                spec.freeze("unit")
                result = json.loads((root/"freeze/result.json").read_text())
            self.assertTrue(result["seeds_disjoint_from_Stage296"])
            self.assertEqual(result["contract"]["locked_policy"]["joint"]["threshold"], .002)

    def test_shared_sampler_uses_supplied_cohort_receipts_and_seeds(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); key = ("test", 397, "low_noise", 12)
            (root/"prefix").mkdir(); (root/"prefix/rows.json").write_text(json.dumps([{"probe_index": 8, "original_choice": 0}]))
            path = root/"panel"; path.mkdir()
            records = [{"draw": d, "future_seed": 100+d, "values": {str(a): [0., 0., 0., 0.] for a in range(6)}} for d in range(8)]
            (path/"labels.json").write_text(json.dumps(records))
            (path/"receipt.json").write_text(json.dumps({"contract": {"new": True}, "software_qualified": True,
                "key": list(key), "probe": 8, "candidates": list(range(6)), "draws": list(range(8)),
                "native_episodes": 56, "exact_identities": 8, "state_futures": 8}))
            ctx = SimpleNamespace(keys=lambda role: [key], prefix_path=lambda run, k: root/"prefix",
                PROBES=(8,), BANKS={"proposal": tuple(range(8))}, sample_path=lambda *args: path,
                contract=lambda: {"new": True}, future_seed=lambda k, p, d: 100+d)
            panel, coverage = spec.parent.load_panel("unit", "test", "proposal", protocol=ctx)
            self.assertEqual(panel.shape, (1, 8, 6, 4)); self.assertEqual(coverage["native_episodes"], 56)

    def test_new_arm_names_preserve_old_summary_numerics(self):
        rng = np.random.default_rng(8)
        panels = rng.normal(size=(6, 32, 6, 4))
        raw = np.zeros((2, 6), int); proposal = np.ones_like(raw)
        choices = np.stack((raw, proposal, proposal, raw, proposal, raw), axis=1)
        sealed = {"arms": list(spec.parent.learning.ARMS), "choices": choices.tolist(),
            "proposal": proposal.tolist(), "matched_thin_probabilities": np.full_like(raw, .5, dtype=float).tolist()}
        cube = np.arange(6).reshape(2, 3, 1)
        old = spec.parent.learning.summarize(sealed, panels, cube)
        renamed = spec.parent.learning.summarize({**sealed, "arms": [*spec.ARMS[:5], spec.ARMS[-1]]}, panels, cube)
        self.assertEqual(old["comparisons"]["paired_risk_vs_raw"], renamed["comparisons"]["joint_budgeted_vs_raw"])
        self.assertEqual(old["comparisons"]["paired_risk_vs_bias_calibrated"], renamed["comparisons"]["joint_budgeted_vs_legacy_risk"])
        self.assertEqual(old["wait_retention_vs_zero_abstain"], renamed["wait_retention_vs_zero_abstain"])
        self.assertEqual(old["joint_transfer_supported"], renamed["joint_transfer_supported"])

    def test_fresh_primary_still_rejects_lost_waiting_benefit(self):
        panels = np.zeros((6, 32, 6, 4)); panels[:, :, 1] = [-1., -.2, -.8, -1.]
        raw = np.zeros((1, 6), int); base = np.ones_like(raw); joint = base.copy(); joint[:, 2:] = 0
        choices = np.stack((raw, base, base, joint, joint, joint, raw), axis=1)
        sealed = {"arms": list(spec.ARMS), "choices": choices.tolist(), "proposal": base.tolist(),
            "matched_thin_probabilities": np.full_like(raw, 1/3, dtype=float).tolist()}
        result = spec.parent.learning.summarize(sealed, panels, np.arange(6).reshape(2, 3, 1), seed=298)
        self.assertAlmostEqual(result["wait_retention_vs_zero_abstain"]["all32"], 1/3)
        self.assertFalse(result["joint_transfer_supported"])

    def test_budgets_and_dynamic_scheduler_placement(self):
        self.assertEqual(len(submit.tasks("unit", "pilot")), 10)
        self.assertEqual(len(submit.tasks("unit", "capture")), 400)
        self.assertEqual(len(submit.tasks("unit", "menus")), 4800)
        job = submit.task("unit", "pilot", spec.keys("pilot")[0])
        self.assertIsNone(job["require_node"])
        self.assertEqual(job["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual((job["cpu"], job["ram_mb"]), (1, 1536))
        self.assertEqual(spec.BANKS["evaluation"], tuple(range(32, 64)))

    def test_locked_joint_cutoff_and_thinning_use_the_same_base_actions(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); self.locked(root)
            raw = np.zeros((2, 6), int); proposal = np.ones_like(raw)
            panels = np.zeros((6, 8, 6, 4)); panels[:, :, 1] = [-.5, -.1, -.4, -1.]
            def read(path):
                if path.name == "model_before_calibration.json": return {}
                if path.name == "calibrated_model.json": return {"calibration": {"paired_risk": {"upward_adjustment": 0.}}}
                return json.loads(path.read_text())
            with patch.object(spec, "directory", return_value=root), patch.object(spec, "read", side_effect=read), \
                    patch.object(spec.parent.learning, "features", return_value=np.zeros((2, 6, 26))), \
                    patch.object(spec.parent.learning, "predict", return_value=np.full((2, 6), .2)), \
                    patch.object(spec.parent.native, "cube", return_value=np.arange(6).reshape(2, 3, 1)):
                sealed = spec._seal_proposal("unit", [{} for _ in range(6)], raw, raw, proposal, panels, panels)
            selected = np.asarray(sealed["choices"])
            np.testing.assert_array_equal(selected[:, 3], raw)
            np.testing.assert_array_equal(selected[:, 4], proposal)
            np.testing.assert_array_equal(sealed["matched_thin_probabilities"], np.ones_like(raw))

    def test_service_ranking_control_has_exact_same_groupwise_action_count(self):
        raw = np.zeros((1, 6), int); proposal = np.ones_like(raw)
        eligible = np.ones_like(raw, bool); joint = np.array([[True, False, False, True, False, True]])
        score = np.array([[3., 1., 2., 1., 2., 3.]])
        cube = np.arange(6).reshape(2, 3, 1)
        selected = spec.matched_service_choices(raw, proposal, eligible, joint, score, cube)
        np.testing.assert_array_equal(selected, [[0, 1, 0, 1, 1, 0]])
        np.testing.assert_array_equal((selected != raw)[:, cube].sum(axis=2), joint[:, cube].sum(axis=2))


if __name__ == "__main__": unittest.main()
