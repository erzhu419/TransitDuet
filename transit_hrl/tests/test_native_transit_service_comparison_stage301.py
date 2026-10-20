import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_service_comparison_stage301 as spec
from scripts import submit_native_transit_service_comparison_stage301_scheduleurm as submit


class ProspectiveServiceComparisonTests(unittest.TestCase):
    def test_fresh_prefixes_and_futures_are_disjoint_from_both_previous_cohorts(self):
        old = spec.parent.keys()+spec.parent.keys("pilot")+spec.previous.keys()+spec.previous.keys("pilot")
        fresh = spec.keys()
        self.assertEqual(len(fresh), 400)
        self.assertEqual({k[3] for k in fresh}, set(range(52, 92)))
        self.assertFalse({spec.scene_seed(k) for k in old}&{spec.scene_seed(k) for k in fresh})
        before = {spec.future_seed(k, p, d) for k in old for p in spec.PROBES for d in range(64)}
        after = {spec.future_seed(k, p, d) for k in fresh for p in spec.PROBES for d in range(64)}
        self.assertFalse(before & after)
        self.assertEqual(len(after), 400*6*64)
        self.assertLess(max(after), 2**31)

    def test_only_legacy_arm_is_replaced_and_joint_is_bitwise_unchanged(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); (root/"inputs").mkdir()
            (root/"inputs/locked_policy.json").write_text(json.dumps({"joint": {"threshold": .002}, "service": {"threshold": -1.5}}))
            raw = np.zeros((2, 6), int); proposal = np.ones_like(raw)
            panels = np.zeros((6, 8, 6, 4))
            panels[:, :, 1, 1] = -.1
            panels[:, :, 1, 2] = np.arange(-2, 4)[:, None]
            def read(path):
                if path.name == "model_before_calibration.json": return {}
                if path.name == "calibrated_model.json": return {"calibration": {"paired_risk": {"upward_adjustment": 0.}}}
                return json.loads(path.read_text())
            with patch.object(spec, "directory", return_value=root), patch.object(spec.previous, "directory", return_value=root), \
                    patch.object(spec, "read", side_effect=read), patch.object(spec.previous, "read", side_effect=read), \
                    patch.object(spec.parent.learning, "features", return_value=np.zeros((2, 6, 26))), \
                    patch.object(spec.parent.learning, "predict", return_value=np.full((2, 6), .1)), \
                    patch.object(spec.parent.native, "cube", return_value=np.arange(6).reshape(2, 3, 1)):
                before = spec.previous._seal_proposal("unit", [{}]*6, raw, raw, proposal, panels, panels)
                after = spec.seal_proposal("unit", [{}]*6, raw, raw, proposal, panels, panels)
            a, b = np.asarray(before["choices"]), np.asarray(after["choices"])
            np.testing.assert_array_equal(a[:, [0, 1, 2, 4, 5, 6]], b[:, [0, 1, 2, 4, 5, 6]])
            np.testing.assert_array_equal(b[:, 3], [[1, 0, 0, 0, 0, 0]]*2)
            self.assertEqual(before["matched_thin_probabilities"], after["matched_thin_probabilities"])
            self.assertEqual(after["arms"][3], "service_only")

    def test_scheduler_reuses_qualified_path_but_new_namespaces(self):
        self.assertNotIn("pilot", spec.PHASES)
        self.assertEqual(len(submit.tasks("unit", "capture")), 400)
        self.assertEqual(len(submit.tasks("unit", "menus")), 4800)
        job = submit.task("unit", "capture", spec.keys()[0])
        self.assertIn(spec.PROTOCOL, job["signature"])
        self.assertIn("service_comparison_stage301.py", job["cmd"])
        self.assertIsNone(job["require_node"])
        self.assertEqual(job["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual((job["cpu"], job["ram_mb"]), (1, 1536))


if __name__ == "__main__": unittest.main()
