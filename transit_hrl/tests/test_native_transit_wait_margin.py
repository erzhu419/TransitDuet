import copy
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_wait_margin_stage200 as spec
from scripts.submit_native_transit_wait_margin_stage200_scheduleurm import task_specification
from tests.test_native_transit_coverage_value import bank_fixture, fake_predictions


class WaitMarginTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = bank_fixture()[0]
        stage195 = spec.source.diagnostic.mean.manifest(cls.bank["manifest"])
        stage197 = spec.source.source.manifest(stage195)
        cls.previous = spec.source.manifest(stage197)
        cls.registered = spec.manifest(cls.previous)
        cls.mean_keys = {spec.source.source.state_key(r) for r in cls.registered["mean_states"]}

    def test_all_outer_partitions_are_scene_disjoint_and_reproducible(self):
        self.assertEqual(spec.manifest(self.previous), self.registered)
        for fold in self.registered["folds"]:
            train, test, _ = spec.source.diagnostic.coverage.split_rows(self.bank, fold["split"], fold["index"])
            keys = {spec.source.source.paired.scene_group(r) for r in train}
            held = [tuple(g) for bucket in fold["inner_partition"] for g in bucket]
            self.assertEqual((len(held), set(held)), (len(keys), keys))
            self.assertFalse(set(held) & {spec.source.source.paired.scene_group(r) for r in test})

    def test_own_inner_fit_and_normalization_ignore_its_heldout_labels(self):
        fold = next(f for f in self.registered["folds"] if f["split"] == "lower_root")
        train, _, _ = spec.source.diagnostic.coverage.split_rows(self.bank, fold["split"], fold["index"])
        changed = copy.deepcopy(train)
        held = {tuple(g) for g in fold["inner_partition"][0]}
        for row in changed:
            if spec.source.source.paired.scene_group(row) in held:
                for label in row["labels"]:
                    for field in ("fleet_cost_delta", "nonfleet_cost_delta", "wait_delta_min"): label[field] *= 100.
                    label["actual_cost_delta"] = label["fleet_cost_delta"]+label["nonfleet_cost_delta"]
        actual_fit, captured = spec.calibrated.ridge.fit_value, []
        def fit(*args, **kwargs):
            model = actual_fit(*args, **kwargs)
            captured.append(model)
            return model
        with patch.object(spec.calibrated.ridge, "fit_value", side_effect=fit):
            a = spec.calibrated.crossfit(train, self.mean_keys, fold["inner_partition"])
            b = spec.calibrated.crossfit(changed, self.mean_keys, fold["inner_partition"])
        self.assertEqual(captured[0], captured[4])
        self.assertEqual(a["calibration_states"], 120)
        self.assertEqual(b["calibration_groups"], 20)
        for model, excluded in zip(captured[:4], fold["inner_partition"]):
            self.assertFalse({tuple(g) for g in model["training_groups"]} & {tuple(g) for g in excluded})

    def test_selected_residual_ratio_and_cluster_dispersion_are_in_minutes(self):
        rows, predictions = [], []
        residuals = {0: [.1, .1], 1: [.4], 2: []}
        for group, values in residuals.items():
            for probe in range(6):
                changed = probe < len(values)
                prediction = {"predicted_heads": [[0., 0.], [-1., 0.]] if changed else [[0., 0.], [0., 0.]],
                    "predicted_wait_delta_min": [0., -.1]}
                rows.append({"root": 397, "scene_seed": group, "probe_index": probe, "original_choice": 0,
                    "labels": [{"candidate": 0, "wait_delta_min": 0.}, {"candidate": 1,
                        "wait_delta_min": -.1+values[probe] if changed else 0.}]})
                predictions.append(prediction)
        result = spec.calibrated.residual_summary(rows, predictions)
        self.assertEqual((result["changed_choices"], result["calibration_groups"]), (3, 3))
        self.assertAlmostEqual(result["mean_selected_residual_min"], .2)
        self.assertAlmostEqual(result["group_ratio_standard_error_min"], np.sqrt(.12)/3)
        self.assertAlmostEqual(result["margins"]["cluster_margin90"], .2+spec.calibrated.Z90*np.sqrt(.12)/3)

    def test_margin_only_changes_wait_filter_and_always_retains_identity(self):
        prediction = {"predicted_heads": [[0., 0.], [-2., 0.], [-1., 0.]], "predicted_wait_delta_min": [0., -.01, -.2]}
        original = copy.deepcopy(prediction)
        self.assertEqual(spec.calibrated.choose(prediction, 0, 0.), 1)
        self.assertEqual(spec.calibrated.choose(prediction, 0, .1), 2)
        self.assertEqual(spec.calibrated.choose(prediction, 0, 1.), 0)
        self.assertEqual(spec.calibrated.corrected_wait(prediction, 0, 1.)[0], 0.)
        self.assertEqual(prediction, original)

    def test_identity_only_calibration_has_zero_margin_without_false_improvement(self):
        rows = [{"root": 397, "scene_seed": 0, "probe_index": 0, "original_choice": 0, "labels": []}]
        predictions = [{"predicted_heads": [[0., 0.], [0., 0.]], "predicted_wait_delta_min": [0., 0.]}]
        result = spec.calibrated.residual_summary(rows, predictions)
        self.assertEqual(result["changed_choices"], 0)
        self.assertEqual(set(result["margins"].values()), {0.})

    def test_worker_preserves_frozen_control_and_evaluates_new_choices_from_same_labels(self):
        fold = self.registered["folds"][0]
        train, test, _ = spec.source.diagnostic.coverage.split_rows(self.bank, fold["split"], fold["index"])
        targets = [r for r in self.bank["training_data"] if spec.source.source.state_key(r) in self.mean_keys]
        deployed = {"training_groups": sorted(map(list, {spec.source.source.paired.scene_group(r) for r in train})),
            "training_states": len(train), "method": "pair_ridge", "steps": 0, "parameter_count": 294,
            "center": [0.]*98, "scale": [1.]*98, "target_scale": [1.]*3}
        prior = {"predictions": {"ridge_count_weighted": {"wait_feasible": fake_predictions(test)}}}
        before = copy.deepcopy(deployed)
        result = spec.fit_fold(self.bank, targets, fold, deployed, prior)
        spec.qualify(result, fold, self.registered)
        self.assertEqual(deployed, before)
        for a, b in zip(prior["predictions"]["ridge_count_weighted"]["wait_feasible"], result["predictions"]["uncalibrated"]):
            for key in ("selected", "predicted_heads", "predicted_wait_delta_min", "actual_cost_delta", "wait_delta_min"):
                self.assertEqual(a[key], b[key])

    def test_zero_margin_identity_report_keeps_strict_cost_gate_closed(self):
        cells = {}
        for fold in self.registered["folds"]:
            train, test, _ = spec.source.diagnostic.coverage.split_rows(self.bank, fold["split"], fold["index"])
            test = [r for r in test if spec.source.source.state_key(r) in self.mean_keys]
            membership = {spec.source.source.paired.scene_group(r) for r in train}
            mean_groups = {k[:2] for k in self.mean_keys if k[:2] in membership}
            calibration = {"partition": fold["inner_partition"], "groups": sorted(map(list, mean_groups)),
                "calibration_states": 6*len(mean_groups), "margins": {p: 0. for p in spec.POLICIES},
                "fits": [{"held_groups": bucket, "training_groups": sorted(map(list, membership-{tuple(g) for g in bucket})),
                    "training_states": 6*(len(membership)-len(bucket)), "parameter_count": 294} for bucket in fold["inner_partition"]]}
            predictions = fake_predictions(test)
            for row in predictions: row["calibrated_wait_delta_min"] = row["predicted_wait_delta_min"]
            cells[fold["split"], fold["index"]] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                "software_qualified": True, "fold": fold, "new_closed_form_fits": 4, "critic_optimizer_updates": 0,
                "actor_updates": 0, "new_native_episodes": 0, "calibration": calibration,
                "deployed_fit": {"training_groups": sorted(map(list, membership)), "training_states": len(train), "parameter_count": 294, "steps": 0},
                "predictions": {p: predictions for p in spec.POLICIES}}
        result = spec.summarize(cells, self.registered)
        self.assertEqual(result["new_closed_form_fits"], 188)
        self.assertFalse(result["development_gate"] or result["third_scene_screen_eligible"] or result["native_deployment_eligible"])

    def test_scheduler_uses_only_cpu_nodes_and_syncs_no_raw_artifacts(self):
        for task in (task_specification("unit_wait"), task_specification("unit_wait", {"split": "scene", "index": 0})):
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertNotIn("_raw", task["result_dir"])


if __name__ == "__main__":
    unittest.main()
