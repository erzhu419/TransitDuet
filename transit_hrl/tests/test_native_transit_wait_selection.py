import contextlib
import copy
import io
import unittest
from unittest.mock import patch

from scripts import analyze_native_transit_wait_selection_stage201 as spec
from scripts.submit_native_transit_wait_selection_stage201_scheduleurm import task_specification
from tests.test_native_transit_coverage_value import bank_fixture, fake_predictions


class WaitSelectionTest(unittest.TestCase):
    def test_transition_roles_and_positive_margin_cannot_introduce_override(self):
        before = {"original_choice": 0, "selected": 1}
        self.assertEqual(spec.transition(before, before), "retained")
        self.assertEqual(spec.transition(before, {**before, "selected": 0}), "cancelled")
        self.assertEqual(spec.transition(before, {**before, "selected": 2}), "rerouted")
        identity = {**before, "selected": 0}
        self.assertEqual(spec.transition(identity, identity), "identity")
        with self.assertRaises(ValueError): spec.transition(identity, before)

    def test_contributions_use_whole_scope_not_changed_subgroup_denominator(self):
        row = dict(root=397, scene_seed=10, old_cost=-.2, new_cost=0., old_wait=-.1,
            new_wait=0., override=False, residual=0., predicted_wait=0.)
        result = spec.selection_summary([row], 10)
        self.assertAlmostEqual(result["cost_change_contribution"], .02)
        self.assertAlmostEqual(result["wait_change_contribution_min"], .01)
        self.assertIsNone(result["selected_residual_mean_min"])
        self.assertEqual(spec.selection_summary([], 10), {"states": 0})

    def test_candidate_errors_exclude_identity_and_context_uses_only_preaction_inputs(self):
        row = {"root": 397, "scene_seed": 0, "probe_index": 8, "scenario": "low_noise",
            "direction": False, "original_choice": 0, "context": {"queues": {"False": 50}},
            "candidates": [{"commitment": {"prefix_planned_s": p}} for p in ([0, 360], [0, 180], [0, 720])],
            "labels": [{"candidate": i, "wait_delta_min": w} for i, w in enumerate([0., .1, -.2])]}
        pred = {"predicted_wait_delta_min": [0., -.1, .1], "selected": 1}
        records = spec.error_records(row, pred)
        self.assertEqual([r["candidate"] for r in records], [1, 2])
        self.assertEqual([r["headway"] for r in records], ["compressed", "widened"])
        self.assertEqual({r["queue"] for r in records}, {"0to50"})
        value = spec.error_summary(records)
        self.assertEqual((value["false_safe"], value["predicted_unsafe_actual_safe"]), (1, 1))
        self.assertAlmostEqual(value["selected_residual_mean_min"], .2)

    def test_rerouted_labels_not_baseline_labels_determine_new_wait(self):
        row = {"root": 397, "scene_seed": 0, "probe_index": 8, "scenario": "low_noise", "original_choice": 0,
            "labels": [{"candidate": i, "actual_cost_delta": c, "wait_delta_min": w}
                for i, c, w in [(0, 0., 0.), (1, -.2, -.1), (2, -.1, .2)]]}
        before = {**{k: row[k] for k in ("root", "scene_seed", "probe_index")}, "original_choice": 0,
            "selected": 1, "actual_cost_delta": -.2, "wait_delta_min": -.1,
            "predicted_wait_delta_min": [0., -.01, -.2]}
        after = {**before, "selected": 2, "actual_cost_delta": -.1, "wait_delta_min": .2}
        record = spec.selection_record(row, before, after)
        self.assertEqual(record["role"], "rerouted")
        self.assertAlmostEqual(record["residual"], .4)
        bad = copy.deepcopy(after)
        bad["wait_delta_min"] = -.1
        with self.assertRaises(ValueError): spec.selection_record(row, before, bad)

    def test_full_frozen_diagnosis_keeps_240_states_per_split_and_never_fits(self):
        bank = bank_fixture()[0]
        source = spec.source
        registered = source.manifest(source.source.manifest(source.source.source.manifest(
            source.source.diagnostic.mean.manifest(bank["manifest"]))))
        keys = {spec.state_key(r) for r in registered["mean_states"]}
        targets = [r for r in bank["training_data"] if spec.state_key(r) in keys]
        for row in targets:
            for label in row["labels"]:
                if label["candidate"] == row["original_choice"]:
                    label.update(actual_cost_delta=0., wait_delta_min=0.)
        def load(fold):
            _, test, _ = source.source.diagnostic.coverage.split_rows(bank, fold["split"], fold["index"])
            predictions = fake_predictions([r for r in test if spec.state_key(r) in keys])
            return {"predictions": {p: predictions for p in source.POLICIES}, "calibration": {
                "mean_selected_residual_min": 0., "group_ratio_standard_error_min": 0.,
                "changed_choices": 0, "margins": {p: 0. for p in source.POLICIES}}}
        with patch.object(source, "qualify") as qualify, \
                patch.object(source.calibrated.ridge, "fit_value", side_effect=AssertionError("Unexpected fit")), \
                contextlib.redirect_stdout(io.StringIO()):
            result, details = spec.diagnose(bank, targets, registered, load)
        self.assertEqual(qualify.call_count, 47)
        self.assertEqual(result["folds"], 47)
        for split in source.SPLITS:
            overall = result["validation"][split]["overall"]
            self.assertEqual(overall["all_nonidentity_wait_errors"]["overall"]["candidate_evaluations"], 1200)
            for value in overall["policies"].values():
                self.assertEqual(value["overall"]["states"], 240)
                self.assertEqual(value["transitions"]["identity"]["states"], 240)
                self.assertEqual(sum(v["states"] for v in value["transitions"].values()), 240)
        self.assertEqual(len(details["fold_calibration"]), 47)
        self.assertFalse(result["third_scene_screen_eligible"] or result["native_deployment_eligible"])

    def test_scheduler_small_report_only_unbound_zero_training_budget(self):
        task = task_specification("unit_wait_selection")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
        self.assertNotIn("_raw", task["result_dir"])
        self.assertTrue(all(v == 0 for v in spec.contract()["budget"].values()))


if __name__ == "__main__":
    unittest.main()
