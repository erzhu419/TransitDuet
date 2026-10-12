import contextlib
import copy
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import analyze_native_transit_fit_transfer_stage198 as spec
from scripts.submit_native_transit_fit_transfer_stage198_scheduleurm import task_specification
from tests.test_native_transit_sequence_value import fixture
from tests.test_native_transit_coverage_value import bank_fixture, fake_predictions


class FitTransferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.rows = fixture()[0]["training_data"][:18]
        cls.train, cls.test = cls.rows[:12], cls.rows[12:]
        cls.raw = np.stack([spec.source.value.features(r, "tail_summary") for r in cls.rows])
        cls.models = {}
        for variant in spec.source.VARIANTS:
            kwargs = dict(steps=3, representation="tail_summary",
                sample_weights=[8]*6+[1]*6 if variant.endswith("count_weighted") else None)
            cls.models[variant] = (spec.source.value.fit_advantage(cls.train, **kwargs) if variant.startswith("advantage") else
                spec.source.value.fit_value(cls.train, "floor_absolute", objective="proper_mse", **kwargs))

    def test_batched_and_cached_inference_match_single_rows_without_feature_recomputation(self):
        for model in self.models.values():
            single = [spec.source.value.predict(row, model) for row in self.rows]
            with patch.object(spec.source.value, "features", side_effect=AssertionError("Cache was ignored")):
                batch = spec.source.value.predict_many(self.rows, model, features_batch=self.raw)
            for a, b, row in zip(single, batch, self.rows):
                self.assertEqual(set(a), set(b))
                for key in a: np.testing.assert_allclose(a[key], b[key], rtol=0, atol=1e-12)
                self.assertEqual(b["predicted_heads"][row["original_choice"]], [0., 0.])
                self.assertEqual(b["predicted_wait_delta_min"][row["original_choice"]], 0.)

    def test_sequence_batch_keeps_existing_encoder_and_absolute_floor(self):
        model = spec.source.value.fit_value(self.train[:2], "floor_absolute", steps=2,
            objective="proper_mse", representation="timetable_sequence")
        batch = spec.source.value.predict_many(self.rows[:2], model)
        for row, prediction in zip(self.rows[:2], batch):
            old = spec.source.value.predict(row, model)
            for key in old: np.testing.assert_allclose(old[key], prediction[key], rtol=0, atol=1e-12)
            self.assertTrue(all(v >= prediction["causal_fleet_floor"] for v in prediction["absolute_fleet_cost"]))

    def test_pair_mse_matches_zero_and_perfect_reference_in_physical_units(self):
        predictions = [{"predicted_heads": [[l["fleet_cost_delta"], l["nonfleet_cost_delta"]] for l in row["labels"]],
            "predicted_wait_delta_min": [l["wait_delta_min"] for l in row["labels"]]} for row in self.rows]
        perfect = spec.summary(spec.pair_records(self.rows, predictions))
        for field in spec.FIELDS: self.assertAlmostEqual(perfect["pair_metrics"][field]["observed_pair_mse"], 0.)
        zero = [{"predicted_heads": [[0., 0.]]*6, "predicted_wait_delta_min": [0.]*6} for _ in self.rows]
        records = spec.pair_records(self.rows, zero)
        result = spec.summary(records+records)
        self.assertEqual(result["model_state_evaluations"], 36)
        self.assertEqual(result["unique_states"], 18)
        self.assertEqual(result["unique_prefix_groups"], 3)
        for field in spec.FIELDS:
            value = result["pair_metrics"][field]
            self.assertEqual(value["observed_pair_mse"], value["zero_gap_pair_mse"])
            self.assertEqual(value["error_change_vs_zero"], 0.)

    def test_fitted_objective_reproduces_actual_weighted_preupdate_loss(self):
        for direct in (False, True):
            kwargs = dict(representation="tail_summary", sample_weights=[8]*6+[1]*6)
            def fit(steps):
                return (spec.source.value.fit_advantage(self.train, steps=steps, **kwargs) if direct else
                    spec.source.value.fit_value(self.train, "floor_absolute", steps=steps, objective="proper_mse", **kwargs))
            initial, one = fit(0), fit(1)
            initial["loss_ledger"] = one["loss_ledger"]
            predictions = spec.source.value.predict_many(self.train, initial)
            objective = spec.fitted_objective(self.train, predictions, initial)
            self.assertAlmostEqual(objective["final_training_loss"], objective["first_step_preupdate_loss"], places=5)
            self.assertEqual(objective["absolute_fleet_anchor_loss"] is None, direct)

    def test_mean_training_and_single_training_stay_separate_and_heldout_stays_frozen(self):
        keys = {spec.source.state_key(r) for r in [*self.train[:6], *self.test]}
        prior = {"predictions": {v: {"mean8": {"wait_feasible": [
            {**r, **spec.source.value.predict(r, model)} for r in self.test]}} for v, model in self.models.items()}}
        with patch.object(spec.source.value, "fit_value", side_effect=AssertionError("Unexpected fit")), \
                patch.object(spec.source.value, "fit_advantage", side_effect=AssertionError("Unexpected fit")):
            records, _, _ = spec.diagnose_fold(self.train, self.test, self.raw[:12], self.raw[12:], keys, self.models, prior)
        for variant, cohorts in records.items():
            self.assertEqual([len(cohorts[c]) for c in spec.COHORTS], [6, 6, 6])
            self.assertEqual(cohorts["heldout_mean8"], spec.pair_records(self.test, prior["predictions"][variant]["mean8"]["wait_feasible"]))
        changed = copy.deepcopy(self.models)
        changed["anchored_uniform"]["training_groups"].append([0, 0])
        with self.assertRaises(ValueError):
            spec.diagnose_fold(self.train, self.test, self.raw[:12], self.raw[12:], keys, changed, prior)

    def test_shift_uses_training_scale_and_exact_outside_range_counts(self):
        train = np.zeros((2, 6, 98))
        test = np.zeros((1, 6, 98))
        test[..., 72] = 2.
        scales = np.ones(98)
        scales[72] = 4.
        shifts = spec.physical_shift(train, test, scales)
        value = shifts[spec.source.value.occupancy.OCCUPANCY_NAMES[0]]
        self.assertEqual(value["standardized_mean_shift"], .5)
        self.assertEqual(value["heldout_outside_train_range_fraction"], 1.)

    def test_all47_folds_count_repeated_training_evaluations_not_new_seeds(self):
        bank = bank_fixture()[0]
        registered = spec.source.manifest(spec.mean.manifest(bank["manifest"]))
        keys = {spec.source.state_key(r) for r in registered["mean_states"]}
        targets = [r for r in bank["training_data"] if spec.source.state_key(r) in keys]
        def load(fold):
            train, test, _ = spec.coverage.split_rows(bank, fold["split"], fold["index"])
            test = [r for r in test if spec.source.state_key(r) in keys]
            models = {v: {"training_groups": sorted(map(list, {spec.source.paired.scene_group(r) for r in train})),
                "training_states": len(train), "scale": [1.]*98, "target_scale": [1.]*3,
                "method": "paired_advantage" if v.startswith("advantage") else "floor_absolute",
                "loss_ledger": [{"absolute_fleet_mse": 1., "paired_mse": 1.}]} for v in spec.source.VARIANTS}
            prior = {"predictions": {v: {"mean8": {"wait_feasible": fake_predictions(test)}} for v in spec.source.VARIANTS}}
            return models, prior
        with patch.object(spec.source.value, "predict_many", side_effect=lambda rows, *a, **k: fake_predictions(rows)), \
                contextlib.redirect_stdout(io.StringIO()):
            result, details = spec.diagnose(bank, targets, registered, load)
        self.assertEqual((result["folds"], result["reused_models"], len(details)), (47, 188, 47))
        scene = result["variants"]["advantage_count_weighted"]["scene"]
        self.assertEqual(scene["train_mean8"]["model_state_evaluations"], 9360)
        self.assertEqual(scene["train_mean8"]["unique_states"], 240)
        self.assertEqual(scene["train_single"]["model_state_evaluations"], 14400)
        self.assertEqual(scene["train_single"]["unique_states"], 360)
        self.assertEqual(scene["heldout_mean8"]["unique_prefix_groups"], 40)
        self.assertEqual(set(result["lower_root_folds"]), {"397", "401"})
        self.assertFalse(result["third_scene_screen_eligible"] or result["native_deployment_eligible"])

    def test_diagnostic_task_is_unbound_small_sync_and_has_zero_training_budget(self):
        task = task_specification("unit_fit_transfer")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
        self.assertNotIn("_raw", task["result_dir"])
        self.assertTrue(all(v == 0 for v in spec.contract()["budget"].values()))


if __name__ == "__main__":
    unittest.main()
