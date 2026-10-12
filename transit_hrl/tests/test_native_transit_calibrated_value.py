import copy
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_calibrated_value as value
from scripts import run_native_transit_calibrated_value_stage190 as spec
from scripts.analyze_native_transit_calibrated_value_stage190 import qualify, summarize
from scripts.submit_native_transit_calibrated_value_stage190_scheduleurm import task_specification
from tests.test_native_transit_coverage_value import bank_fixture, fake_model


def fake_predict(row, model):
    delta = np.arange(6, dtype=float)-row["original_choice"]
    return {"predicted_heads": np.c_[-delta*.01, delta*0.].tolist(),
        "predicted_wait_delta_min": (-delta*.02).tolist()}


class CalibratedValueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank, _, _, _ = bank_fixture()
        cls.registered = spec.manifest(cls.bank["manifest"])

    def fit(self, fold, mode="tail_summary", bank=None):
        calls = []
        def fit(rows, method, **kwargs):
            calls.append(rows)
            self.assertEqual((method, kwargs), ("floor_absolute", {"steps": 100,
                "objective": "proper_mse", "representation": mode}))
            return fake_model(rows, mode)
        with patch.object(spec.source.value, "fit_value", side_effect=fit), \
                patch.object(spec.source.value, "predict", side_effect=fake_predict):
            result, _ = spec.fit_fold(bank or self.bank, mode, fold)
        return result, calls

    def test_all_107_partitions_are_grouped_disjoint_and_training_only(self):
        self.assertEqual(len(self.registered["folds"]), 107)
        self.assertEqual(spec.manifest(self.bank["manifest"]), self.registered)
        for fold in self.registered["folds"]:
            train, test, _ = spec.source.split_rows(self.bank, fold["split"], fold["index"])
            train_keys = {spec.source.paired.scene_group(r) for r in train}
            test_keys = {spec.source.paired.scene_group(r) for r in test}
            cal = {tuple(g) for g in fold["partition"]["calibration_groups"]}
            members = [{tuple(g) for g in groups} for groups in fold["partition"]["member_groups"]]
            self.assertFalse(cal & test_keys)
            self.assertGreaterEqual(len(cal), 10)
            self.assertEqual(set.union(*members) | cal, train_keys)
            for member in members:
                self.assertFalse(member & (cal | test_keys))
                self.assertTrue(member)

    def test_calibration_uses_scene_maxima_not_120_correlated_callbacks(self):
        rows, predictions = [], []
        for scene in range(20):
            for probe in range(6):
                rows.append({"root": 397, "scene_seed": scene, "original_choice": 0,
                    "labels": [{"actual_cost_delta": (scene+1) if i == 5 and probe == 5 else 0.,
                        "wait_delta_min": 2.*(scene+1) if i == 2 and probe == 2 else 0.} for i in range(6)]})
                predictions.append({"mean": np.zeros((6, 2)).tolist(), "uncertainty": np.ones((6, 2)).tolist()})
        cal = value.calibrate(rows, predictions)
        self.assertEqual((len(cal["groups"]), cal["states"], cal["rank"]), (20, 120, 19))
        self.assertEqual(cal["multipliers"], [19., 38.])

    def test_uncertainty_accounts_for_cost_head_covariance_and_identity(self):
        row = {"original_choice": 0}
        models = list(range(4))
        estimates = [{"predicted_heads": [[0., 0.], [i+1., -i-1.], *([[1., 1.]]*4)],
            "predicted_wait_delta_min": [0., i*.1, 0., 0., 0., 0.]} for i in models]
        with patch.object(value.value, "predict", side_effect=estimates):
            p = value.ensemble_prediction(row, models, np.asarray([.02, .01]))
        self.assertEqual(p["uncertainty"][1][0], .02)
        self.assertGreater(p["uncertainty"][1][1], .1)
        self.assertEqual(p["mean"][0], [0., 0.])
        self.assertEqual(p["uncertainty"][0], [0., 0.])

    def test_cost_and_wait_margins_reject_optimistic_actions_and_tie_keeps_identity(self):
        p = {"mean": [[0., 0.], [-.1, -.01], [-.05, -.2], [1., 1.], [1., 1.], [1., 1.]],
            "uncertainty": [[0., 0.], [.01, .1], [.1, .01], [.1, .1], [.1, .1], [.1, .1]]}
        self.assertEqual(value.choose(p, 0, [0., 0.]), 1)
        self.assertEqual(value.choose(p, 0, [1., 1.]), 0)
        p["mean"][2][0] = -.2
        self.assertEqual(value.choose(p, 0, [1., 1.]), 2)
        p["mean"][2][0] = -.1
        self.assertEqual(value.choose(p, 0, [1., 1.]), 0)

    def test_outer_labels_never_change_fits_calibration_or_selection(self):
        fold = self.registered["folds"][0]
        result, calls = self.fit(fold)
        qualify(result, "tail_summary", fold, self.registered)
        held = {tuple(g) for g in fold["partition"]["calibration_groups"]}
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(not {spec.source.paired.scene_group(r) for r in rows} & held for rows in calls))
        changed = copy.deepcopy(self.bank)
        _, test, _ = spec.source.split_rows(changed, fold["split"], fold["index"])
        for row in test:
            for label in row["labels"]:
                label["actual_cost_delta"] += 100.
                label["wait_delta_min"] += 100.
        other, _ = self.fit(fold, bank=changed)
        for field in ("fits", "calibration", "uncertainty_floors"):
            self.assertEqual(result[field], other[field])
        for policy in value.POLICIES:
            self.assertEqual([p["selected"] for p in result["predictions"][policy]],
                [p["selected"] for p in other["predictions"][policy]])

    def test_qualification_rejects_leaked_fit_changed_calibration_and_selector(self):
        fold = self.registered["folds"][0]
        result, _ = self.fit(fold, mode="timetable_sequence")
        qualify(result, "timetable_sequence", fold, self.registered)
        for fault in ("member", "calibration", "margin", "selector", "gradient"):
            broken = copy.deepcopy(result)
            if fault == "member": broken["fits"][0]["training_groups"].append(fold["partition"]["calibration_groups"][0])
            elif fault == "calibration": broken["calibration"]["groups"].pop()
            elif fault == "margin": broken["calibration"]["multipliers"][0] += 1.
            elif fault == "selector": broken["predictions"]["calibrated"][0]["selected"] = -1
            else: broken["fits"][0]["encoder_gradient_l2_max"] = 0.
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                qualify(broken, "timetable_sequence", fold, self.registered)

    def test_fallback_only_cannot_pass_strict_cost_gate_and_missing_fold_fails(self):
        cells, references = {}, {}
        for mode in spec.MODES:
            for fold in self.registered["folds"]:
                train, test, _ = spec.source.split_rows(self.bank, fold["split"], fold["index"])
                groups = fold["partition"]["calibration_groups"]
                predictions = [value.evaluate(r, {"mean": np.zeros((6, 2)).tolist(),
                    "uncertainty": np.zeros((6, 2)).tolist()}, [0., 0.]) for r in test]
                cells[mode, fold["split"], fold["index"]] = {"protocol": spec.EXPERIMENT_PROTOCOL,
                    "contract": spec.contract(), "software_qualified": True, "mode": mode, "fold": fold,
                    "fits": [fake_model([r for r in train if list(spec.source.paired.scene_group(r)) in member], mode)
                        for member in fold["partition"]["member_groups"]],
                    "calibration": {"groups": groups, "states": len(groups)*6, "quantile": .9,
                        "rank": min(len(groups), int(np.ceil((len(groups)+1)*.9))), "multipliers": [0., 0.],
                        "group_scores": [{"group": g, "score": [0., 0.]} for g in groups]},
                    "critic_updates": 400, "actor_updates": 0, "new_native_steps": 0, "fit_wall_s": 1.,
                    "predictions": {p: predictions for p in value.POLICIES}}
                references.setdefault((mode, fold["split"]), []).extend(predictions)
        result = summarize(cells, self.registered, references)
        self.assertEqual((result["fits"], result["critic_updates"]), (856, 85600))
        self.assertFalse(result["third_scene_screen_eligible"] or result["native_deployment_eligible"])
        for variant in result["variants"].values():
            self.assertEqual(variant["gates"], {"cost": False, "wait": True})
            for split in spec.SPLITS:
                self.assertEqual(variant["validation"][split]["old_state_subset"]["states"], 120)
        cells.pop(next(iter(cells)))
        with self.assertRaises(ValueError): summarize(cells, self.registered, references)

    def test_scheduler_unbound_small_receipts_exclude_server_models_and_bank(self):
        fold = self.registered["folds"][0]
        for task in (task_specification("unit_cal", mode="tail_summary", fold=fold), task_specification("unit_cal")):
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertNotIn("_raw", task["result_dir"])
            self.assertNotIn("bank.json", task["cmd"])
        self.assertIn("/report", task_specification("unit_cal")["result_dir"])


if __name__ == "__main__":
    unittest.main()
