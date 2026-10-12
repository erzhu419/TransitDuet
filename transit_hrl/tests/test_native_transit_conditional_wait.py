import copy
import unittest

import numpy as np

from scripts import run_native_transit_conditional_wait_stage202 as spec
from scripts.submit_native_transit_conditional_wait_stage202_scheduleurm import task_specification
from tests.test_native_transit_coverage_value import bank_fixture, fake_model, fake_predictions
from tests.test_native_transit_sequence_value import fixture


class ConditionalWaitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = fixture()[0]["training_data"][:12]
        cls.bank = bank_fixture()[0]
        cls.registered = spec.manifest(spec.source.manifest(spec.source.source.manifest(
            spec.source.diagnostic.mean.manifest(cls.bank["manifest"]))))
        cls.keys = {spec.state_key(r) for r in cls.registered["mean_states"]}

    def test_geometry_uses_physical_plan_units_and_existing_nominal_anchor(self):
        row = {"nominal_s": [0., 360., 720.], "candidates": [{"commitment": {"prefix_planned_s": p}}
            for p in ([0., 360., 720.], [0., 240., 720.])]}
        geometry = spec.wait.geometry_features(row)
        self.assertEqual(geometry.shape, (2, 8))
        np.testing.assert_allclose(geometry[0], [1., 1., 1., 0., 0., 0., 0., 1.])
        self.assertAlmostEqual(geometry[1, 4], -1/9)
        self.assertAlmostEqual(geometry[1, 5], 1/27)
        self.assertAlmostEqual(geometry[1, 7], 10/9)

    def test_feature_shapes_and_inference_ignore_labels_ids_and_future_values(self):
        row = self.rows[0]
        changed = copy.deepcopy(row)
        changed.update(root=0, scenario="unused", scene_seed=0, labels=[])
        for bus in changed["occupancy_context"]["active_buses"]: bus["actual_remaining_s"] = 99999.
        for variant in spec.wait.REPRESENTATIONS:
            x = spec.wait.features(row, variant)
            self.assertEqual(x.shape, (6, spec.wait.FEATURE_COUNTS[variant]))
            np.testing.assert_array_equal(x, spec.wait.features(changed, variant))
            model = spec.wait.fit_wait(self.rows, variant, sample_weights=[1]*12)
            np.testing.assert_array_equal(spec.wait.predict_wait(row, model), spec.wait.predict_wait(changed, model))
            self.assertEqual(spec.wait.predict_wait(row, model)[row["original_choice"]], 0.)

    def test_load_conditions_affect_action_differences_instead_of_cancelling(self):
        row = self.rows[0]
        changed = copy.deepcopy(row)
        for bus in changed["occupancy_context"]["active_buses"]:
            if bus["direction"] == row["direction"]: bus["load_fraction"] += .5
        original = row["original_choice"]
        a, b = (spec.wait.features(r, "conditional_wait") for r in (row, changed))
        np.testing.assert_allclose((a-a[original])[:, :102], (b-b[original])[:, :102], atol=1e-12)
        self.assertFalse(np.allclose((a-a[original])[:, 102:], (b-b[original])[:, 102:]))

    def test_normal_equations_weights_train_scales_and_physical_wait_units(self):
        variant, weights = "conditional_wait", [8]*6+[1]*6
        model = spec.wait.fit_wait(self.rows, variant, sample_weights=weights)
        raw = np.stack([spec.wait.features(r, variant) for r in self.rows])
        targets = np.asarray([[l["wait_delta_min"] for l in r["labels"]] for r in self.rows])
        x = (raw-model["center"])/model["scale"]
        a, b = np.triu_indices(6, 1)
        z, y = (x[:, a]-x[:, b]).reshape(-1, 198), (targets[:, a]-targets[:, b]).reshape(-1)/model["target_scale_min"]
        w = np.repeat(weights, 15)
        expected = np.linalg.solve(z.T@(w[:, None]*z)/w.sum()+.1*np.eye(198), z.T@(w*y)/w.sum())
        np.testing.assert_allclose(model["coefficients"], expected, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(model["center"], raw.mean(axis=(0, 1)))
        predicted = ((x[0]-x[0, self.rows[0]["original_choice"]])@expected)*model["target_scale_min"]
        np.testing.assert_allclose(spec.wait.predict_wait(self.rows[0], model), predicted, atol=1e-12)
        self.assertEqual((model["parameter_count"], model["steps"], model["sample_weight_sum"]), (198, 0, 54.))

    def test_zero_wait_targets_produce_exact_zero_deltas(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            for label in row["labels"]: label["wait_delta_min"] = 0.
        for variant in spec.wait.REPRESENTATIONS:
            model = spec.wait.fit_wait(rows, variant, sample_weights=[1]*12)
            np.testing.assert_array_equal(spec.wait.predict_wait(rows[0], model), np.zeros(6))

    def test_worker_freezes_cost_and_own_heldout_labels_do_not_change_fits(self):
        fold = self.registered["folds"][0]
        train, test, _ = spec.source.diagnostic.coverage.split_rows(self.bank, fold["split"], fold["index"])
        test = [r for r in test if spec.state_key(r) in self.keys]
        targets = [r for r in self.bank["training_data"] if spec.state_key(r) in self.keys]
        prior = {"fits": {"ridge_count_weighted": {**fake_model(train, "tail_summary"), "method": "pair_ridge",
            "parameter_count": 294, "steps": 0}}, "predictions": {"ridge_count_weighted": {"wait_feasible": fake_predictions(test)}}}
        before = copy.deepcopy(prior)
        result, models = spec.fit_fold(self.bank, targets, fold, prior)
        spec.qualify(result, fold, self.registered)
        self.assertEqual(prior, before)
        self.assertEqual(result["predictions"]["frozen_wait"], prior["predictions"]["ridge_count_weighted"]["wait_feasible"])
        changed = copy.deepcopy(targets)
        for row in changed:
            if list(spec.source.source.paired.scene_group(row)) == fold["held_out"]:
                for label in row["labels"]: label["wait_delta_min"] *= 100
        _, refit = spec.fit_fold(self.bank, changed, fold, prior)
        self.assertEqual(models, refit)
        for variant in spec.wait.REPRESENTATIONS:
            self.assertNotIn(fold["held_out"], models[variant]["training_groups"])
            for old, current in zip(result["predictions"]["frozen_wait"], result["predictions"][variant]):
                self.assertEqual(old["predicted_heads"], current["predicted_heads"])
        bad = copy.deepcopy(result)
        bad["predictions"]["conditional_wait"][0]["predicted_heads"][0][0] += 1.
        with self.assertRaises(ValueError): spec.qualify(bad, fold, self.registered)

    def test_all47_folds_identity_cannot_pass_strict_cost_gate(self):
        targets = [r for r in self.bank["training_data"] if spec.state_key(r) in self.keys]
        cells = {}
        for fold in self.registered["folds"]:
            train, test, _ = spec.source.diagnostic.coverage.split_rows(self.bank, fold["split"], fold["index"])
            test = [r for r in test if spec.state_key(r) in self.keys]
            membership = sorted(map(list, {spec.source.source.paired.scene_group(r) for r in train}))
            fits = {v: {"representation": v, "training_groups": membership, "training_states": len(train),
                "parameter_count": spec.wait.FEATURE_COUNTS[v], "sample_weight_sum": len(train)+7*(240-len(test)),
                "steps": 0, "ridge": .1} for v in spec.wait.REPRESENTATIONS}
            cells[fold["split"], fold["index"]] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                "software_qualified": True, "fold": fold, "new_closed_form_fits": 2, "critic_optimizer_updates": 0,
                "actor_updates": 0, "new_native_episodes": 0,
                "frozen_cost_fit": {"training_groups": membership, "training_states": len(train), "parameter_count": 294},
                "fits": fits, "predictions": {v: fake_predictions(test) for v in spec.VARIANTS}}
        result = spec.summarize(cells, self.registered, targets)
        self.assertEqual(result["new_closed_form_fits"], 94)
        self.assertFalse(result["development_gate"] or result["third_scene_screen_eligible"] or result["native_deployment_eligible"])
        for variant in result["variants"].values():
            for validation in variant["validation"].values(): self.assertEqual(validation["overall"]["states"], 240)

    def test_scheduler_unbound_cpu_only_and_report_sync_excludes_raw_artifacts(self):
        for task in (task_specification("unit_conditional_wait"), task_specification("unit_conditional_wait", {"split": "scene", "index": 0})):
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertNotIn("_raw", task["result_dir"])


if __name__ == "__main__":
    unittest.main()
