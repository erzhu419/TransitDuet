import copy
import unittest

import numpy as np

from scripts import analyze_native_transit_fit_transfer_stage216 as spec
from scripts.submit_native_transit_fit_transfer_stage216_scheduleurm import task_specification
from tests import test_native_transit_mean_wait as fixture


class MeanFitTransferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.MeanWaitTest.setUpClass()
        f = fixture.MeanWaitTest
        cls.bank, cls.masses, cls.keys = f.bank, f.masses, f.keys
        cls.registered = spec.source.manifest(f.registered)

    def test_training_reference_membership_and_sizes_are_exact(self):
        sizes = {"scene": (594, 18, 6), "regime": (480, 192, 48), "lower_root": (300, 120, 120)}
        for fold in self.registered["folds"]:
            masks = spec.masks(self.bank["training_data"], fold, self.keys)
            self.assertEqual(tuple(int(masks[k].sum()) for k in ("training_all", "training_reference", "held")), sizes[fold["split"]])
            self.assertFalse(np.any(masks["training_all"] & masks["held"]))
            self.assertTrue(np.all(~masks["training_reference"] | masks["training_mean8"]))
            if fold["split"] == "scene":
                strata = [{(r["root"], r["scenario"]) for r, keep in zip(self.bank["training_data"], masks[k]) if keep}
                    for k in ("training_reference", "held")]
                self.assertEqual(strata[0], strata[1])

    def test_diagnostic_weights_are_optimization_weights_not_future_counts(self):
        rows = [copy.deepcopy(self.bank["training_data"][i]) for i in (0, 1)]
        masses = {spec.state_key(r): {"positive_wait_min": [0., 1., 1., 1., 1., 1.],
            "negative_wait_min": [0.]*6, "draw_count": 8, "sample_weight": w} for r, w in zip(rows, (8, 1))}
        for r in rows: r["original_choice"] = 0
        logits = np.asarray([[0.]*6, [2.]*6])
        rs = spec.transfer.records(rows, masses, logits, np.zeros((2, 6)), 1., {})
        self.assertAlmostEqual(rs[0]["weighted_mass_sum_min"], 40.)
        self.assertAlmostEqual(rs[1]["weighted_mass_sum_min"], 5.)
        expected = 40*np.log(2)+5*np.logaddexp(0, -2)
        self.assertAlmostEqual(sum(r["weighted_loss_sum_min"] for r in rs), expected)

    def test_support_uses_training_only_ranges_and_causal_inputs(self):
        rows = self.bank["training_data"][:3]
        raw, context = np.zeros((3, 6, 198)), np.zeros((3, 12))
        originals = [r["original_choice"] for r in rows]
        candidate = next(i for i in range(6) if i != originals[2])
        raw[2, candidate, 0], context[2, 0] = 100., 100.
        support = spec.feature_support(rows, (raw, context), np.asarray([True, True, False]))
        self.assertEqual(support[0]["cost_pair"], 0.)
        self.assertAlmostEqual(support[2]["cost_pair"], 1/490)
        self.assertAlmostEqual(support[2]["wait_pair"], 1/990)
        self.assertAlmostEqual(support[2]["wait_context"], 1/12)
        changed = copy.deepcopy(rows)
        for r in changed: r["labels"] = []
        self.assertEqual(support, spec.feature_support(changed, (raw, context), np.asarray([True, True, False])))

    def test_half_stability_does_not_turn_sampling_disagreement_into_false_safe(self):
        row = copy.deepcopy(self.bank["training_data"][0])
        row["original_choice"] = 0
        for label in row["labels"]:
            for key in ("fleet_cost_delta", "nonfleet_cost_delta", "actual_cost_delta", "wait_delta_min"): label[key] = 0.
        values = np.zeros((8, 6, 4))
        values[:4, 1, 3], values[4:, 1, 3], values[:, 2, 3] = 1., -2., 1.
        masses = {spec.state_key(row): {"positive_wait_min": [0., .5, 1., 0., 0., 0.],
            "negative_wait_min": [0., 1., 0., 0., 0., 0.], "draw_count": 8}}
        panels = {spec.state_key(row): {"values": values}}
        rs = spec.records([row], masses, panels, np.zeros((1, 6, 2)), np.zeros((1, 6)), np.zeros((1, 6)), [1.]*3)
        self.assertEqual(rs[0]["first_last"]["stable_false_safe"], 1)
        self.assertEqual(rs[0]["first_last"]["sign_disagreement"], 1)
        self.assertEqual(rs[0]["first_last"]["stable_unsafe_candidates"], 1)
        result = spec.summary(rs+rs)
        self.assertEqual((result["state_model_evaluations"], result["unique_states"], result["unique_prefix_groups"]), (2, 1, 1))
        self.assertEqual(result["first_last_stable_false_safe_rate"], 1.)

    def test_cost_residual_metric_keeps_total_cost_cancellation_and_head_errors_separate(self):
        row = self.bank["training_data"][0]
        heads = np.asarray([[[l["fleet_cost_delta"], l["nonfleet_cost_delta"]] for l in sorted(row["labels"], key=lambda x: x["candidate"])]])
        altered = heads.copy()
        altered[0, 1] += [1., -1.]
        rs = spec.records([row], self.masses, {}, altered, np.zeros((1, 6)), np.zeros((1, 6)), [1.]*3)
        result = spec.summary(rs)
        self.assertAlmostEqual(result["cost_pair_mse"], 0.)
        self.assertGreater(result["cost_head_pair_normalized_mse"], 0.)
        self.assertIsNone(result["first_last_cost_inversion_rate"])

    def test_all47_frozen_folds_same_state_contrasts_and_controls(self):
        bank, masses = self.bank, self.masses
        panels = {}
        for r in bank["training_data"]:
            if spec.state_key(r) not in self.keys: continue
            labels = {l["candidate"]: l for l in r["labels"]}
            values = np.asarray([[labels[i][k] for k in ("actual_cost_delta", "fleet_cost_delta", "nonfleet_cost_delta", "wait_delta_min")] for i in range(6)])
            panels[spec.state_key(r)] = {"values": np.repeat(values[None], 8, axis=0)}
        cost = {"center": [0.]*98, "scale": [1.]*98, "target_scale": [1.]*3,
            "coefficients": np.zeros((98, 3)).tolist(), "ridge_loss": 0.}
        glm = {"center": [0.]*198, "scale": [1.]*198, "coefficients": [0.]*199, "ridge": .1}
        shrunk = {"reference": glm, "context_center": [0.]*12, "context_scale": [1.]*12,
            "weights": {"0.weight": np.zeros((16, 210)).tolist(), "0.bias": [0.]*16,
                "2.weight": [[0.]*16], "2.bias": [0.]}, "residual_bound": 1., "final_functional_penalty": 0.}
        calls = []
        def load(fold):
            calls.append(fold)
            rs = [r for r, keep in zip(bank["training_data"], spec.masks(bank["training_data"], fold, self.keys)["held"]) if keep]
            predictions = []
            for r in rs:
                share = [0. if j == r["original_choice"] else .5 for j in range(6)]
                predictions.append({**r, "predicted_heads": [[0., 0.]]*6,
                    "predicted_harm_mass_share": share, "predicted_feasibility_score": [0.]*6})
            return {"predictions": {v: predictions for v in spec.VARIANTS}}, {"old_cost": cost, "mean_cost": cost}, {"glm": glm, "shrunk": shrunk}
        result = spec.analyze(bank, masses, panels, self.registered, load)
        self.assertEqual(len(calls), 47)
        self.assertEqual((result["cost_models"], result["wait_models"]), (94, 94))
        for variant in result["variants"].values():
            for split, validation in variant["validation"].items():
                held = validation["held"]["overall"]
                self.assertEqual((held["state_model_evaluations"], held["unique_states"], held["unique_prefix_groups"]), (240, 240, 40))
            self.assertEqual(variant["validation"]["scene"]["training_reference"]["overall"]["state_model_evaluations"], 720)
            self.assertEqual(variant["validation"]["scene"]["training_reference"]["overall"]["unique_states"], 240)
            self.assertEqual(len(variant["lower_root_directions"]), 2)
            self.assertEqual(len(variant["fold_gaps"]), 47)
            for contrast in variant["same_state_held_minus_scene"].values():
                self.assertEqual(contrast["states"], 240)
                self.assertNotIn("cost_head_pair_normalized_mse", contrast["overall"])
                self.assertTrue(all(v in (None, 0.) for v in contrast["overall"].values()))
            for split in spec.SPLITS:
                held = variant["validation"][split]["held"]["overall"]
                self.assertAlmostEqual(held["mean_selected_harm_min"]-held["mean_selected_benefit_min"], held["mean_selected_wait_min"])

    def test_different_fold_scales_are_not_used_as_same_state_transfer_errors(self):
        train = {k: 1. for k in spec.GAP_METRICS}
        held = {**train, "cost_head_pair_normalized_mse": 100.}
        self.assertEqual(spec.gaps(train, held)["cost_head_pair_normalized_mse"], 99.)
        self.assertTrue(all(v == 0 for v in spec.gaps(train, held, spec.CONTRAST_METRICS).values()))

    def test_scheduler_unbound_read_only_small_results(self):
        task = task_specification("unit_fit_transfer")
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertIsNone(task["require_node"])
        self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
        self.assertNotIn("_raw", task["result_dir"])
        self.assertTrue(all("unit_fit_transfer" not in p for p in task["stage_input_paths"]))
        self.assertEqual(spec.contract()["budget"], {"new_fits": 0, "optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0})


if __name__ == "__main__":
    unittest.main()
