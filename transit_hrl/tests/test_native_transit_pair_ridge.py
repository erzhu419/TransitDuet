import copy
import unittest

import numpy as np

from scripts import run_native_transit_pair_ridge_stage199 as spec
from scripts.submit_native_transit_pair_ridge_stage199_scheduleurm import task_specification
from tests.test_native_transit_coverage_value import bank_fixture, fake_model, fake_predictions
from tests.test_native_transit_sequence_value import fixture


class PairRidgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = fixture()[0]["training_data"][:12]
        raw = np.stack([spec.source.value.features(r, "tail_summary") for r in cls.rows])
        scale = raw.std(axis=(0, 1))
        scale[scale < 1e-8] = 1.
        cls.reference = {"center": raw.mean(axis=(0, 1)).tolist(), "scale": scale.tolist(), "target_scale": [1., 2., 3.]}

    def test_weighted_normal_equations_use_training_state_weights_and_restore_units(self):
        weights = np.asarray([8]*6+[1]*6)
        model = spec.ridge.fit_value(self.rows, self.reference, sample_weights=weights)
        raw = np.stack([spec.source.value.features(r, "tail_summary") for r in self.rows])
        x = (raw-model["center"])/model["scale"]
        y = np.asarray([[[l[h] for h in spec.source.value.HEADS] for l in r["labels"]] for r in self.rows])/model["target_scale"]
        a, b = np.triu_indices(6, 1)
        z, gaps = (x[:, a]-x[:, b]).reshape(-1, 98), (y[:, a]-y[:, b]).reshape(-1, 3)
        w = np.repeat(weights, 15)
        expected = np.linalg.solve(z.T@(z*w[:, None])/w.sum()+.1*np.eye(98), z.T@(gaps*w[:, None])/w.sum())
        np.testing.assert_allclose(model["coefficients"], expected, atol=1e-12, rtol=0)
        self.assertEqual((model["parameter_count"], model["steps"], model["sample_weight_sum"]), (294, 0, 54))
        row = self.rows[0]
        physical = ((x[0]-x[0, row["original_choice"]])@expected)*model["target_scale"]
        prediction = spec.ridge.predict(row, model)
        np.testing.assert_allclose(prediction["predicted_heads"], physical[:, :2], atol=1e-12, rtol=0)
        np.testing.assert_allclose(prediction["predicted_wait_delta_min"], physical[:, 2], atol=1e-12, rtol=0)

    def test_identity_and_inference_ignore_labels_group_ids_and_absolute_risk(self):
        model = spec.ridge.fit_value(self.rows, self.reference)
        row = self.rows[0]
        prediction = spec.ridge.predict(row, model)
        self.assertEqual(prediction["predicted_heads"][row["original_choice"]], [0., 0.])
        self.assertEqual(prediction["predicted_wait_delta_min"][row["original_choice"]], 0.)
        self.assertNotIn("absolute_fleet_cost", prediction)
        changed = copy.deepcopy(row)
        changed.update(labels=[], root=0, scene_seed=0, scenario="unused")
        self.assertEqual(spec.ridge.predict(changed, model), prediction)
        changed_rows = copy.deepcopy(self.rows)
        for r in changed_rows:
            for l in r["labels"]: l["future_fleet_cost"] += 100.
        self.assertEqual(spec.ridge.fit_value(changed_rows, self.reference), model)

    def test_zero_targets_keep_identity_in_both_selectors(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            for label in row["labels"]:
                for head in spec.source.value.HEADS: label[head] = 0.
        model = spec.ridge.fit_value(rows, self.reference)
        np.testing.assert_array_equal(model["coefficients"], np.zeros((98, 3)))
        for policy in spec.source.value.POLICIES:
            self.assertEqual(spec.ridge.choose_action(rows[0], model, policy), rows[0]["original_choice"])

    def test_worker_holds_out_whole_scene_and_keeps_neural_predictions_frozen(self):
        bank = bank_fixture()[0]
        prior_manifest = spec.source.manifest(spec.diagnostic.mean.manifest(bank["manifest"]))
        registered = spec.manifest(prior_manifest)
        fold = registered["folds"][0]
        train, test, _ = spec.diagnostic.coverage.split_rows(bank, fold["split"], fold["index"])
        keys = {spec.source.state_key(r) for r in registered["mean_states"]}
        targets = [r for r in bank["training_data"] if spec.source.state_key(r) in keys]
        references = {v: {**fake_model(train, "tail_summary"), "method": "paired_advantage"} for v in ("advantage_uniform", "advantage_count_weighted")}
        references["advantage_count_weighted"]["sample_weight_sum"] = len(train)+7*234
        prior = {"predictions": {v: {"mean8": {p: fake_predictions(test) for p in spec.source.value.POLICIES}} for v in references}}
        result, models = spec.fit_fold(bank, targets, fold, references, prior)
        spec.qualify(result, fold, registered)
        self.assertEqual(result["predictions"]["neural_count_weighted"], prior["predictions"]["advantage_count_weighted"]["mean8"])
        self.assertNotIn(fold["held_out"], models["ridge_uniform"]["training_groups"])
        self.assertEqual(result["fits"]["ridge_count_weighted"]["sample_weight_sum"], len(train)+7*234)

    def test_report_cannot_pass_identity_or_call_294_coefficients_matched_to_1635(self):
        bank = bank_fixture()[0]
        registered = spec.manifest(spec.source.manifest(spec.diagnostic.mean.manifest(bank["manifest"])))
        keys = {spec.source.state_key(r) for r in registered["mean_states"]}
        targets = [r for r in bank["training_data"] if spec.source.state_key(r) in keys]
        cells = {}
        for fold in registered["folds"]:
            train, test, _ = spec.diagnostic.coverage.split_rows(bank, fold["split"], fold["index"])
            test = [r for r in test if spec.source.state_key(r) in keys]
            fits = {}
            for variant in spec.VARIANTS:
                model = fake_model(train, "tail_summary")
                model.update(method="pair_ridge" if variant.startswith("ridge") else "paired_advantage",
                    parameter_count=294 if variant.startswith("ridge") else 1635,
                    steps=0 if variant.startswith("ridge") else 100,
                    sample_weight_sum=len(train)+(7*(240-len(test)) if variant.endswith("count_weighted") else 0))
                fits[variant] = model
            cells[fold["split"], fold["index"]] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                "software_qualified": True, "fold": fold, "fits": fits, "new_closed_form_fits": 2,
                "critic_optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0,
                "predictions": {v: {p: fake_predictions(test) for p in spec.source.value.POLICIES} for v in spec.VARIANTS}}
        previous = {"variants": {"advantage_uniform:wait_feasible": {"pair_error": {s: {"overall": {"pair_metrics": {
            f: {"mean_label_noise": 0.} for f in spec.diagnostic.FIELDS}}} for s in spec.SPLITS}}}}
        result = spec.summarize(cells, registered, targets, previous)
        self.assertEqual(result["new_closed_form_fits"], 94)
        self.assertFalse(result["development_gate"] or result["third_scene_screen_eligible"] or result["native_deployment_eligible"])
        cell = next(iter(cells.values()))
        cell["fits"]["ridge_uniform"]["parameter_count"] = 1635
        with self.assertRaises(ValueError): spec.qualify(cell, cell["fold"], registered)

    def test_scheduler_remains_unbound_and_keeps_raw_models_remote(self):
        for task in (task_specification("unit_ridge"), task_specification("unit_ridge", {"split": "scene", "index": 0})):
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertIsNone(task["require_node"])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertNotIn("_raw", task["result_dir"])


if __name__ == "__main__":
    unittest.main()
