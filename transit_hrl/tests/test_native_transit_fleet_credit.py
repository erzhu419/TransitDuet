import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import run_native_transit_fleet_credit_stage194 as spec
from scripts.analyze_native_transit_fleet_credit_stage194 import COMPARISONS, qualify, summarize
from scripts.submit_native_transit_fleet_credit_stage194_scheduleurm import task_specification
from tests.test_native_transit_coverage_value import bank_fixture, fake_model
from tests.test_native_transit_expected_value import fixture
from tests.test_native_transit_fleet_residual import fleet_model as old_model, predictions as old_predictions


def model_fixture(rows, method):
    m = old_model(rows, method)
    m.update(initialization="same16_hidden_rows_zero_output_shared16_or_disjoint8plus8",
        paired_target_scale=.2, paired_loss_scale=1. if method == "separated_shared_scale" else .2,
        baseline_pair_gradient=method == "shared_pair_scaled")
    return m


def predictions(rows):
    return [{**r, "fleet_pair_mse": 0., "original_fleet_mse": 0.} for r in old_predictions(rows)]


class FleetCreditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.bank, _, _, _ = bank_fixture()
        cls.targets, _ = spec.expected.build_targets(cls.bank, fixture(cls.bank))
        cls.registered = spec.manifest(spec.previous.manifest(spec.expected.manifest(cls.bank["manifest"])))

    def test_initial_budget_hidden_weights_and_zero_predictions_match_cached_control(self):
        rows = self.bank["training_data"][:6]
        rng = torch.get_rng_state().clone()
        cached = spec.value.previous.fit_fleet(rows, "fleet_from_zero", steps=0)
        for method in spec.value.METHODS:
            model = spec.value.fit_fleet(rows, method, steps=0)
            self.assertEqual(model["parameter_count"], 1601)
            np.testing.assert_array_equal(spec.value.predict_future(rows[0], model), np.zeros(6))
            hidden = model["weights"]["shared"]["0.weight"] if method == "shared_pair_scaled" else (
                model["weights"]["risk"]["0.weight"]+model["weights"]["contrast"]["0.weight"])
            self.assertEqual(hidden, cached["weights"]["0.weight"])
            self.assertEqual((model["center"], model["scale"], model["target_scale"]),
                (cached["center"], cached["scale"], cached["target_scale"]))
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_pair_credit_does_not_train_reference_risk_and_anchor_does_not_train_credit(self):
        for target_kind in ("pair_only", "risk_only"):
            rows = copy.deepcopy(self.bank["training_data"][:6])
            for row in rows:
                for label in row["labels"]:
                    label["future_fleet_cost"] = .3 if target_kind == "risk_only" else (
                        0. if label["candidate"] == row["original_choice"] else .2)
            for method in ("separated_shared_scale", "separated_pair_scaled"):
                model = spec.value.fit_fleet(rows, method, steps=4)
                untouched = "risk" if target_kind == "pair_only" else "contrast"
                trained = "contrast" if target_kind == "pair_only" else "risk"
                np.testing.assert_array_equal(model["weights"][untouched]["2.weight"], np.zeros((1, 8)))
                self.assertTrue(np.any(np.asarray(model["weights"][trained]["2.weight"]) != 0))
                self.assertNotIn("2.bias", model["weights"]["contrast"])

    def test_torch_numpy_floor_identity_and_permutation_agree_without_label_inputs(self):
        rows = self.bank["training_data"][:6]
        for method in spec.value.METHODS:
            model = json.loads(json.dumps(spec.value.fit_fleet(rows, method, steps=4)))
            row = rows[0]
            x = torch.tensor((spec.value.floor.features(row, "tail_summary")-model["center"])/model["scale"], dtype=torch.float32)
            def forward(t, name):
                w = {k: torch.tensor(v) for k, v in model["weights"][name].items()}
                return (torch.tanh(t@w["0.weight"].T+w["0.bias"])@w["2.weight"].T+w.get("2.bias", 0.)).ravel()
            if method == "shared_pair_scaled":
                expected = forward(x, "shared").clamp_min(0.)
            else:
                risk = forward(x[row["original_choice"]], "risk").clamp_min(0.)
                effect = forward(x, "contrast")
                expected = (risk+effect-effect[row["original_choice"]]).clamp_min(0.)
            p = spec.value.predict_future(row, model)
            np.testing.assert_allclose(p, expected.numpy()*model["target_scale"], rtol=1e-5, atol=1e-7)
            if method != "shared_pair_scaled":
                baseline = max(float(spec.value.forward_numpy(
                    ((spec.value.floor.features(row, "tail_summary")-model["center"])/model["scale"])[row["original_choice"]],
                    model["weights"]["risk"])[0]), 0.)
                self.assertEqual(p[row["original_choice"]], baseline*model["target_scale"])
            changed = copy.deepcopy(row)
            changed.update(labels=[], root=999, scene_seed=1, scenario="unused")
            np.testing.assert_array_equal(p, spec.value.predict_future(changed, model))
            order = [5, 0, 2, 1, 4, 3]
            changed["candidates"] = [row["candidates"][i] for i in order]
            changed["original_choice"] = order.index(row["original_choice"])
            np.testing.assert_allclose(spec.value.predict_future(changed, model), p[order], rtol=0, atol=1e-12)

    def test_pair_units_change_the_pair_loss_without_changing_output_units(self):
        rows = self.bank["training_data"][:6]
        models = {m: spec.value.fit_fleet(rows, m, steps=1) for m in spec.value.METHODS}
        self.assertEqual(len({m["target_scale"] for m in models.values()}), 1)
        self.assertEqual(len({m["paired_target_scale"] for m in models.values()}), 1)
        for method in ("shared_pair_scaled", "separated_pair_scaled"):
            self.assertAlmostEqual(models[method]["loss_ledger"][0]["paired_fleet_mse"], 1., places=5)
        scaled = models["separated_pair_scaled"]
        self.assertAlmostEqual(models["separated_shared_scale"]["loss_ledger"][0]["paired_fleet_mse"],
            (scaled["paired_target_scale"]/scaled["target_scale"])**2, places=5)

    def test_worker_never_trains_test_groups_and_reproduces_cached_predictions(self):
        fold = self.registered["folds"][0]
        train, _, _ = spec.coverage.split_rows(self.bank, fold["split"], fold["index"])
        with patch.object(spec.value, "fit_fleet", side_effect=lambda rows, method, **_: model_fixture(rows, method)) as fit, \
                patch.object(spec.value, "evaluate", side_effect=lambda rows, *args: predictions(rows)):
            result, _ = spec.fit_fold(self.bank, self.targets, fold, fake_model(train, "tail_summary"), old_model(train, "fleet_from_zero"))
        self.assertEqual(fit.call_count, 3)
        for call in fit.call_args_list:
            self.assertEqual(len(call.args[0]), 594)
            self.assertNotIn(tuple(fold["held_out"]), {spec.coverage.paired.scene_group(r) for r in call.args[0]})
        qualify(result, fold, self.registered)
        spec.match_previous(result, result)
        broken = copy.deepcopy(result)
        broken["predictions"]["fleet_from_zero"]["mean8"]["cost_only"][0]["selected"] = -1
        with self.assertRaises(ValueError): spec.match_previous(broken, result)
        broken = copy.deepcopy(result)
        broken["fits"]["separated_pair_scaled"]["baseline_pair_gradient"] = True
        with self.assertRaises(ValueError): qualify(broken, fold, self.registered)

    def test_report_retains_all_factorial_effects_and_does_not_open_reserved_labels(self):
        cells = {}
        for fold in self.registered["folds"]:
            train, test, _ = spec.coverage.split_rows(self.bank, fold["split"], fold["index"])
            keys = {spec.expected.state_key(r) for r in self.targets}
            test = [r for r in test if spec.expected.state_key(r) in keys]
            cells[fold["split"], fold["index"]] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                "software_qualified": True, "fold": fold,
                "fits": {"fleet_from_zero": old_model(train, "fleet_from_zero"),
                    **{m: model_fixture(train, m) for m in spec.value.METHODS}},
                "training_mean_states": 30-len(test), "evaluation_mean_states": len(test),
                "predictions": {v: {l: {p: predictions(test) for p in spec.coverage.value.POLICIES}
                    for l in ("original", "mean8")} for v in spec.value.VARIANTS},
                "critic_updates": 300, "actor_updates": 0, "new_native_steps": 0,
                "fit_wall_s": {m: 1. for m in spec.value.METHODS}}
        result = summarize(cells, self.registered)
        self.assertEqual((result["new_fits"], result["reused_fits"], result["critic_updates"]), (51, 34, 5100))
        self.assertEqual(len(result["variants"]), 10)
        self.assertEqual(set(result["factorial_effects"]["wait_feasible"]), set(COMPARISONS))
        self.assertFalse(result["expected_target_development_gate"] or result["third_scene_screen_eligible"] or result["native_deployment_eligible"])
        cells.pop(next(iter(cells)))
        with self.assertRaises(ValueError): summarize(cells, self.registered)

    def test_scheduler_uses_all_six_unbound_nodes_without_syncing_raw_artifacts(self):
        tasks = [task_specification("unit_credit", fold=f) for f in self.registered["folds"]]
        self.assertEqual(len({t["signature"] for t in tasks}), 17)
        for task in [*tasks, task_specification("unit_credit")]:
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertNotIn("_raw", task["result_dir"])


if __name__ == "__main__":
    unittest.main()
