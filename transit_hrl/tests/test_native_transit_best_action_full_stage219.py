import unittest
from unittest.mock import patch

from scripts import analyze_native_transit_best_action_full_stage219 as spec
from scripts.submit_native_transit_best_action_full_stage219_scheduleurm import task_specification
from tests import test_native_transit_best_action_stage218 as fixture


class FullBestActionTest(unittest.TestCase):
    def test_full_reference_retains_every_original_state_and_no_extra_rows(self):
        rows, registered = fixture.fixture()
        panels = {spec.source.state_key(r): {} for r in rows}
        returned = {"partitions": {p: {"bounds": {k: {"linear_programs": 1, "feasible": True}
            for k in spec.source.PROBLEMS}} for p in spec.source.PARTITIONS}}
        with patch.object(spec.source, "diagnose_view", return_value=returned) as diagnose:
            result = spec.analyze({"training_data": rows+[dict(root=0, scene_seed=0, probe_index=0)]}, panels, registered, lambda p, r: None)
        selected = diagnose.call_args.args[0]
        self.assertEqual(len(selected), 240)
        self.assertEqual({spec.source.state_key(r) for r in selected}, set(panels))
        self.assertEqual(result["linear_programs"], 8)
        self.assertTrue(result["linear_bounds_qualified"])
        self.assertFalse(result["native_deployment_eligible"])

    def test_missing_panels_are_not_silently_dropped(self):
        rows, registered = fixture.fixture()
        with self.assertRaisesRegex(ValueError, "original240"):
            spec.analyze({"training_data": rows}, {}, registered, lambda p, r: None)

    def test_scheduler_reference_has_separate_identity_and_small_budget(self):
        task = task_specification("unit_full_best_action")
        self.assertEqual(task["project"], spec.EXPERIMENT_PROTOCOL)
        self.assertIn("full_stage219.py", task["cmd"])
        self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertNotIn("_raw", task["result_dir"])
        self.assertEqual(spec.contract()["budget"]["max_linear_programs"], 8)
        self.assertEqual(spec.contract()["budget"]["new_native_episodes"], 0)


if __name__ == "__main__":
    unittest.main()
