from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_screen_transfer_stage295 as spec
from scripts import submit_native_transit_screen_transfer_stage295_scheduleurm as submit


class ScreenTransferTest(unittest.TestCase):
    def fixture(self, profiles=2):
        raw, proposal = np.zeros((profiles, 3), int), np.ones((profiles, 3), int)
        screen, evaluation = np.zeros((3, 8, 6, 4)), np.zeros((3, 32, 6, 4))
        for i in range(3):
            screen[i, :, 1, 1] = [-1.]*8 if i == 0 else [-1., 1.]*4 if i == 1 else 0.
            evaluation[i, :, 1, 1] = (1., 0., 3.)[i]
            for bank in (screen, evaluation):
                bank[i, :, 1, 2] = -.1*(i+1)
                bank[i, :, 1, 0] = bank[i, :, 1, 1]+bank[i, :, 1, 2]
                bank[i, :, 1, 3] = -.1*(i+1)
        cube = np.arange(3).reshape(1, 3, 1)
        sealed = spec.source.rule.seal(raw, raw, proposal, screen, cube)
        rows = [{"root": 397, "scenario": "low_noise", "replica": i} for i in range(3)]
        parent = spec.source.rule.summarize(sealed, evaluation, cube)
        return sealed, screen, evaluation, cube, rows, parent

    def test_category_partition_and_independent_fleet_reversal(self):
        result, _ = spec.diagnostics(*self.fixture())
        self.assertLess(result["bridge_max_abs_error"], 1e-13)
        self.assertEqual((result["retained_profile_state_changes"], result["retained_physical_states"], result["retained_prefixes"]), (4, 2, 2))
        categories = result["categories"]
        for phase in ("screen8_contribution", "evaluation32_contribution", "first16_contribution", "last16_contribution"):
            arm_phase = "screen8_selected_mean" if phase.startswith("screen") else "evaluation32_mean"
            for f in spec.source.rule.FIELDS:
                self.assertAlmostEqual(sum(categories[n][phase][f] for n in ("retained_negative_mean", "retained_signed_cancellation")),
                    result["screen_to_evaluation"]["zero_abstain"][arm_phase][f])
                self.assertAlmostEqual(sum(categories[n][phase][f] for n in ("thinning_zero_event_rejects", "thinning_nonzero_retained")),
                    result["screen_to_evaluation"]["matched_thin"][arm_phase][f])
        self.assertLess(categories["retained_negative_mean"]["screen8_contribution"]["fleet"], 0)
        self.assertGreater(categories["retained_negative_mean"]["evaluation32_contribution"]["fleet"], 0)
        self.assertEqual(result["physical_net_fleet_events"]["zero_abstain_vs_raw"]["all32"],
            {"net_fleet_positive": 32, "net_fleet_negative": 0, "net_fleet_zero": 64})

    def test_duplicate_profiles_do_not_multiply_physical_events_or_uncertainty(self):
        a, _ = spec.diagnostics(*self.fixture(1))
        b, _ = spec.diagnostics(*self.fixture(16))
        self.assertEqual(a["physical_net_fleet_events"], b["physical_net_fleet_events"])
        for key in ("retained_physical_states", "retained_prefixes"):
            self.assertEqual(a[key], b[key])
        for pair in a["prefix_attribution"]:
            for f in spec.source.rule.FIELDS:
                self.assertAlmostEqual(a["prefix_attribution"][pair]["variance"][f]["conditional_MC_SE"],
                    b["prefix_attribution"][pair]["variance"][f]["conditional_MC_SE"])

    def test_frozen_mean_and_half_bridge_rejects_changed_results(self):
        args = list(self.fixture())
        args[-1]["halves"]["first_half"]["zero_abstain"]["cost"] += .1
        with self.assertRaisesRegex(ValueError, "changed"):
            spec.diagnostics(*args)

    def test_parent_requires_complete_independent_collection(self):
        parent = {"contract": spec.source.contract(), "coverage_qualified": True}
        receipt = {"scheduler": {"collection_status_counts": {"done": 4196},
            "collection_exit_code_counts": {"0": 4196}, "report": {"status": "done", "exit_code": 0}}}
        with patch.object(spec, "read", return_value=parent):
            self.assertEqual(spec.qualify_parent(receipt), parent)
            receipt["scheduler"]["collection_status_counts"] = {"done": 4195, "running": 1}
            with self.assertRaisesRegex(ValueError, "Complete"):
                spec.qualify_parent(receipt)

    def test_worker_uses_registered_completion_not_local_parent_receipt(self):
        registration = {"contract": spec.contract(), "parent_completion": {"scheduler": "registered_snapshot"}}
        args = self.fixture()
        with patch.object(spec, "read", return_value=registration), \
                patch.object(spec, "qualify_parent", return_value=args[-1]) as qualify, \
                patch.object(spec, "load_cached", return_value=(args[0], args[4], args[1], args[2])), \
                patch.object(spec.source, "cube", return_value=args[3]), patch.object(spec, "write_json") as save:
            spec.analyze("unit295")
            qualify.assert_called_once_with(registration["parent_completion"])
            self.assertEqual(save.call_count, 2)
            self.assertEqual(save.call_args.args[0].parts[-2:], ("report", "result.json"))

    def test_scheduler_is_unpinned_and_cannot_sync_remote_parent_outputs(self):
        task = submit.task_specification("unit295")
        self.assertEqual((task["cpu"], task["ram_mb"], task["require_node"]), (1, 3072, None))
        self.assertEqual(len(task["allowed_nodes"]), 6)
        self.assertEqual(task["stage_excludes"], ["results", "**/__pycache__"])
        self.assertEqual(Path(task["stage_input_paths"][-1]), spec.directory("unit295")/"inputs")
        self.assertNotIn(str(spec.source.directory(spec.SOURCE_RUN)), task["stage_input_paths"])


if __name__ == "__main__":
    unittest.main()
