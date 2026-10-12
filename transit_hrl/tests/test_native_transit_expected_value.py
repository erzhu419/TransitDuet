import copy
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_expected_value_stage192 as spec
from scripts.analyze_native_transit_expected_value_stage192 import qualify, summarize
from scripts.submit_native_transit_expected_value_stage192_scheduleurm import task_specification
from tests.test_native_transit_coverage_value import bank_fixture, fake_model, fake_predictions


def fixture(bank):
    rows = {spec.state_key(r): r for r in bank["training_data"]}
    cells = {}
    for root, scenario, probe, draw in spec.noise.cells():
        old = rows[root, spec.noise.source.scene_seed(root, scenario, 0), probe]
        labels = copy.deepcopy(old["labels"])
        for label in labels:
            label["future_fleet_cost"] = (draw % 2)/12.
            if label["candidate"] != old["original_choice"]:
                label["nonfleet_cost_delta"] += .001*(draw+1)
                label["actual_cost_delta"] = label["fleet_cost_delta"]+label["nonfleet_cost_delta"]
                label["physical_return_delta"] = -100*label["actual_cost_delta"]
        cells[root, scenario, probe, draw] = {**{k: old[k] for k in ("root", "scenario", "scene_seed", "probe_index")},
            "existing_labels": copy.deepcopy(old["labels"]), "labels": labels}
    return cells


class ExpectedValueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank, _, _, _ = bank_fixture()
        cls.registered = spec.manifest(cls.bank["manifest"])
        cls.targets, cls.decomposition = spec.build_targets(cls.bank, fixture(cls.bank))

    def test_mean_uses_physical_cost_not_squared_average_peak_and_preserves_identity(self):
        self.assertEqual(len(self.targets), 30)
        rows = {spec.state_key(r): r for r in self.bank["training_data"]}
        for target in self.targets:
            old = rows[spec.state_key(target)]
            self.assertEqual(target["labels"][old["original_choice"]]["actual_cost_delta"], 0.)
            self.assertAlmostEqual(target["labels"][0]["future_fleet_cost"], 1/24.)
            self.assertNotAlmostEqual(target["labels"][0]["future_fleet_cost"], (.5**2)/12.)
        changed = spec.replace_targets(self.bank["training_data"], self.targets)
        self.assertEqual(len(changed), 600)
        self.assertEqual(sum(a["labels"] != b["labels"] for a, b in zip(changed, self.bank["training_data"])), 30)
        for old, new in zip(self.bank["training_data"], changed):
            self.assertEqual({k: v for k, v in old.items() if k != "labels"}, {k: v for k, v in new.items() if k != "labels"})

    def test_covariance_decomposition_reconstructs_actual_pairwise_cost_noise(self):
        for scope in (self.decomposition["overall"], *self.decomposition["by_root"].values(), *self.decomposition["by_probe"].values()):
            self.assertAlmostEqual(scope["total_gap_variance"], scope["fleet_gap_variance"]+
                scope["nonfleet_gap_variance"]+scope["twice_gap_covariance"])

    def test_all17_folds_exclude_complete_scenes_not_only_replicated_callbacks(self):
        self.assertEqual(len(self.registered["folds"]), 17)
        keys = {spec.state_key(r) for r in self.targets}
        for fold in self.registered["folds"]:
            train, test, _ = spec.source.split_rows(self.bank, fold["split"], fold["index"])
            self.assertFalse({spec.source.paired.scene_group(r) for r in train} & {spec.source.paired.scene_group(r) for r in test})
            self.assertEqual(len(train)+len(test), 600)
            self.assertEqual(sum(spec.state_key(r) in keys for r in test), {"scene": 3, "regime": 6, "lower_root": 15}[fold["split"]])

    def test_fit_reuses_reference_and_never_trains_on_heldout_mean_labels(self):
        fold = self.registered["folds"][0]
        original, _, _ = spec.source.split_rows(self.bank, fold["split"], fold["index"])
        reference = fake_model(original, "timetable_sequence")
        calls = []
        def fit(rows, method, **kwargs):
            calls.append(rows)
            self.assertEqual((method, kwargs), ("floor_absolute", {"steps": 100, "objective": "proper_mse", "representation": "timetable_sequence"}))
            result = fake_model(rows, "timetable_sequence")
            result["target_scale"] = [2., 2., 2.]
            return result
        with patch.object(spec.source.value, "fit_value", side_effect=fit), \
                patch.object(spec.source.floor, "evaluate", side_effect=lambda rows, model, policy: fake_predictions(rows)):
            result, _ = spec.fit_fold(self.bank, self.targets, "timetable_sequence", fold, reference)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 594)
        self.assertNotIn(tuple(fold["held_out"]), {spec.source.paired.scene_group(r) for r in calls[0]})
        self.assertEqual((result["training_mean_states"], result["evaluation_mean_states"]), (27, 3))
        qualify(result, "timetable_sequence", fold, self.registered)
        broken = copy.deepcopy(result)
        broken["predictions"]["mean8_overlay"]["mean8"]["wait_feasible"][0]["selected"] = -1
        with self.assertRaises(ValueError): qualify(broken, "timetable_sequence", fold, self.registered)

    def test_development_gate_cannot_open_reserved_screen_or_claim600_test_states(self):
        cells = {}
        for mode in spec.MODES:
            for fold in self.registered["folds"]:
                train, test, _ = spec.source.split_rows(self.bank, fold["split"], fold["index"])
                keys = {spec.state_key(r) for r in self.targets}
                test = [r for r in test if spec.state_key(r) in keys]
                predictions = fake_predictions(test)
                cells[mode, fold["split"], fold["index"]] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                    "software_qualified": True, "mode": mode, "fold": fold,
                    "fits": {v: fake_model(train, mode) for v in spec.VARIANTS},
                    "training_mean_states": 30-len(test), "evaluation_mean_states": len(test),
                    "predictions": {v: {label: {p: predictions for p in spec.source.value.POLICIES} for label in ("original", "mean8")}
                        for v in spec.VARIANTS}, "critic_updates": 100, "actor_updates": 0, "new_native_steps": 0, "fit_wall_s": 1.}
        result = summarize(cells, self.registered, {"variance_decomposition": self.decomposition})
        self.assertEqual((result["new_fits"], result["reused_fits"], result["critic_updates"]), (34, 34, 3400))
        self.assertEqual((result["evaluation_states"], result["evaluation_prefix_scene_groups"]), (30, 10))
        self.assertFalse(result["expected_target_development_gate"] or result["third_scene_screen_eligible"] or result["native_deployment_eligible"])
        cells.pop(next(iter(cells)))
        with self.assertRaises(ValueError): summarize(cells, self.registered, {"variance_decomposition": self.decomposition})

    def test_scheduler_keeps_models_mean_labels_and_predictions_remote(self):
        fold = self.registered["folds"][0]
        tasks = [task_specification("unit_expected"), task_specification("unit_expected", mode="tail_summary", fold=fold),
            task_specification("unit_expected", report=True)]
        for task in tasks:
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertNotIn("_raw", task["result_dir"])
            self.assertNotIn("bank.json", task["cmd"])


if __name__ == "__main__":
    unittest.main()
