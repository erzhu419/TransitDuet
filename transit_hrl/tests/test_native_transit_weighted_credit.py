import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import run_native_transit_weighted_credit_stage197 as spec
from scripts.analyze_native_transit_weighted_credit_stage197 import qualify, summarize
from scripts.submit_native_transit_weighted_credit_stage197_scheduleurm import task_specification
from tests.test_native_transit_coverage_value import bank_fixture, fake_model, fake_predictions
from tests.test_native_transit_sequence_value import fixture


class WeightedCreditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.rows = fixture()[0]["training_data"][:12]

    def fit(self, rows=None, weights=None, advantage=False, steps=3):
        rows = self.rows if rows is None else rows
        kwargs = dict(steps=steps, representation="tail_summary", sample_weights=weights)
        return (spec.value.fit_advantage(rows, **kwargs) if advantage else
            spec.value.fit_value(rows, "floor_absolute", objective="proper_mse", **kwargs))

    def test_sample_counts_are_training_state_only_not_heldout_label_values(self):
        targets = copy.deepcopy(self.rows[:6])
        self.assertEqual(spec.sample_counts(self.rows, targets), [8]*6+[1]*6)
        for row in targets: row["labels"] = []
        self.assertEqual(spec.sample_counts(self.rows[6:], targets), [1]*6)

    def test_weighting_keeps_scales_capacity_initialization_and_updates_all_heads(self):
        for advantage in (False, True):
            initial = self.fit(advantage=advantage, steps=0)
            uniform = self.fit(advantage=advantage)
            weighted = self.fit(weights=[8]*6+[1]*6, advantage=advantage)
            for key in ("center", "scale", "target_scale", "parameter_count", "initialization_seed", "training_groups", "steps"):
                self.assertEqual(uniform[key], weighted[key])
            self.assertEqual(weighted["parameter_count"], 1635)
            self.assertEqual(weighted["sample_weight_sum"], 54)
            for i in range(3):
                self.assertFalse(np.array_equal(weighted["weights"]["2.weight"][i], initial["weights"]["2.weight"][i]))
        anchor_initial = self.fit(steps=0)
        direct_initial = self.fit(advantage=True, steps=0)
        self.assertEqual(anchor_initial["weights"], direct_initial["weights"])

    def test_uniform_weights_reproduce_unweighted_loss_and_optimizer(self):
        for advantage in (False, True):
            uniform = self.fit(advantage=advantage)
            weighted = self.fit(advantage=advantage, weights=[1]*12)
            for key in uniform["weights"]:
                # Pair-only output offsets are unidentifiable and cancel at inference.
                if key == "2.bias": continue
                np.testing.assert_allclose(uniform["weights"][key], weighted["weights"][key], atol=1e-6, rtol=1e-5)
            for row in self.rows:
                a, b = (spec.value.predict(row, m) for m in (uniform, weighted))
                for key in ("predicted_heads", "predicted_wait_delta_min"):
                    np.testing.assert_allclose(a[key], b[key], atol=1e-7, rtol=1e-5)

    def test_weighted_pair_loss_reduces_states_not_actions_or_heads(self):
        weights = np.asarray([8]*6+[1]*6)
        initial = self.fit(advantage=True, steps=0)
        model = self.fit(advantage=True, weights=weights, steps=1)
        a, b = np.triu_indices(6, 1)
        errors = []
        for row in self.rows:
            p = spec.value.predict(row, initial)
            prediction = np.c_[p["predicted_heads"], p["predicted_wait_delta_min"]]
            target = np.asarray([[l[h] for h in spec.value.HEADS] for l in row["labels"]])
            gaps = (prediction[a]-prediction[b]-(target[a]-target[b]))/model["target_scale"]
            errors.append(np.mean(gaps**2))
        self.assertAlmostEqual(model["loss_ledger"][0]["paired_mse"], np.average(errors, weights=weights), places=5)
        self.assertEqual(model["loss_ledger"][0]["absolute_fleet_mse"], 0.)

    def test_relative_credit_has_exact_identity_and_no_absolute_forecast(self):
        model = self.fit(advantage=True)
        row = self.rows[0]
        prediction = spec.value.predict(row, model)
        self.assertNotIn("absolute_fleet_cost", prediction)
        self.assertFalse(model["absolute_fleet_forecast"])
        self.assertEqual(prediction["predicted_heads"][row["original_choice"]], [0., 0.])
        self.assertEqual(prediction["predicted_wait_delta_min"][row["original_choice"]], 0.)
        changed = copy.deepcopy(row)
        changed.update(labels=[], root=0, scenario="unused", scene_seed=0)
        self.assertEqual(spec.value.predict(changed, model), prediction)
        w = {k: torch.tensor(v) for k, v in model["weights"].items()}
        x = torch.tensor((spec.value.features(row, "tail_summary")-model["center"])/model["scale"], dtype=torch.float32)
        physical = (torch.tanh(x@w["0.weight"].T+w["0.bias"])@w["2.weight"].T+w["2.bias"])*torch.tensor(model["target_scale"])
        delta = physical-physical[row["original_choice"]].clone()
        np.testing.assert_allclose(prediction["predicted_heads"], delta[:, :2].numpy(), atol=1e-7, rtol=1e-5)

    def test_absolute_target_offsets_cannot_change_direct_action_credit(self):
        changed = copy.deepcopy(self.rows)
        for row in changed:
            for label in row["labels"]: label["future_fleet_cost"] += 100.
        self.assertEqual(self.fit(advantage=True), self.fit(changed, advantage=True))

    def test_worker_holds_out_complete_scene_and_reuses_cached_control(self):
        bank = bank_fixture()[0]
        registered = spec.manifest(spec.source.manifest(bank["manifest"]))
        fold = registered["folds"][0]
        train, test, _ = spec.source.source.split_rows(bank, fold["split"], fold["index"])
        reference = fake_model(train, "tail_summary")
        reference.update(method="floor_absolute", target="absolute")
        keys = {spec.state_key(r) for r in registered["mean_states"]}
        targets = [r for r in bank["training_data"] if spec.state_key(r) in keys]
        prior = {"predictions": {"mean8_overlay": {label: {p: fake_predictions(test) for p in spec.value.POLICIES}
            for label in ("original", "mean8")}}}
        def fit(rows, *args, **kwargs):
            self.assertFalse({spec.paired.scene_group(r) for r in rows} & {spec.paired.scene_group(r) for r in test})
            result = copy.deepcopy(reference)
            if kwargs["sample_weights"] is not None:
                result["sample_weight_sum"] = sum(kwargs["sample_weights"])
            return result
        with patch.object(spec.value, "fit_value", side_effect=fit) as anchored, \
                patch.object(spec.value, "fit_advantage", side_effect=fit) as direct, \
                patch.object(spec.source.source.floor, "evaluate", side_effect=lambda rows, *args: fake_predictions(rows)):
            result, _ = spec.fit_fold(bank, targets, fold, reference, prior)
        self.assertEqual((anchored.call_count, direct.call_count), (1, 2))
        self.assertEqual(result["critic_updates"], 300)
        self.assertEqual(result["fits"]["anchored_count_weighted"]["sample_weight_sum"], len(train)+7*234)

    def test_report_keeps_identity_from_passing_and_counts_prefix_groups(self):
        bank = bank_fixture()[0]
        registered = spec.manifest(spec.source.manifest(bank["manifest"]))
        keys = {spec.state_key(r) for r in registered["mean_states"]}
        cells = {}
        for fold in registered["folds"]:
            train, test, _ = spec.source.source.split_rows(bank, fold["split"], fold["index"])
            test = [r for r in test if spec.state_key(r) in keys]
            fits, predictions = {}, {}
            for variant in spec.VARIANTS:
                model = fake_model(train, "tail_summary")
                model.update(absolute_fleet_forecast=variant.startswith("anchored"),
                    sample_weight_sum=len(train)+(7*(240-len(test)) if variant.endswith("count_weighted") else 0))
                rows = fake_predictions(test)
                if variant.startswith("advantage"):
                    for row in rows:
                        row.pop("absolute_fleet_cost")
                        row.pop("causal_fleet_floor")
                fits[variant] = model
                predictions[variant] = {label: {p: rows for p in spec.value.POLICIES} for label in ("original", "mean8")}
            cells[fold["split"], fold["index"]] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                "software_qualified": True, "fold": fold, "fits": fits, "predictions": predictions,
                "training_mean_states": 240-len(test), "evaluation_mean_states": len(test),
                "critic_updates": 300, "actor_updates": 0, "new_native_episodes": 0, "fit_wall_s": {"a": 1.}}
        panels = {spec.state_key(r): {"values": np.zeros((8, 6, 4))} for r in registered["mean_states"]}
        report = summarize(cells, registered, panels)
        self.assertEqual((report["new_fits"], report["critic_updates"], report["evaluation_prefix_groups"]), (141, 14100, 40))
        self.assertFalse(report["development_gate"] or report["third_scene_screen_eligible"] or report["native_deployment_eligible"])
        first = registered["folds"][0]
        cell = cells[first["split"], first["index"]]
        cell["fits"]["advantage_uniform"]["training_groups"].append(first["held_out"])
        with self.assertRaises(ValueError): qualify(cell, first, registered)

    def test_scheduler_is_unbound_and_never_syncs_raw_artifacts(self):
        for task in (task_specification("unit_credit"), task_specification("unit_credit", {"split": "scene", "index": 0})):
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertIsNone(task["require_node"])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertNotIn("_raw", task["result_dir"])
            self.assertNotIn("bank.json", task["cmd"])


if __name__ == "__main__":
    unittest.main()
