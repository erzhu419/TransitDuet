import ast
import unittest

import numpy as np

from scripts import analyze_native_transit_wait_transfer_stage207 as spec
from scripts.submit_native_transit_wait_transfer_stage207_scheduleurm import task_specification
from tests.test_native_transit_decision_wait import DecisionWaitTest
from tests.test_native_transit_coverage_value import fake_predictions
from tests.test_native_transit_sequence_value import fixture


def row(root=397, scenario="low_noise", seed=1):
    return {"root": root, "scenario": scenario, "scene_seed": seed, "probe_index": 8, "original_choice": 0}


def mass(r, positive, negative, count=8):
    return {spec.source.state_key(r): {"positive_wait_min": positive, "negative_wait_min": negative, "draw_count": count}}


class WaitTransferTest(unittest.TestCase):
    def test_conservative_objective_rejections_are_not_own_loss_errors(self):
        r = row()
        masses = mass(r, [0., .2, 0., .2, 0., .1], [0., .3, 0., 0., .2, .2])
        scores = np.array([[0., .1, .1, -.1, .1, .1]])
        logits = np.where(scores < 0, -1., 1.)
        record = spec.records([r], masses, logits, scores, 2., {})[0]
        self.assertEqual(record["intentional_conservative_rejections"], 1)
        self.assertEqual(record["expected_false_unsafe_informative"], 3)
        self.assertEqual(record["expected_false_unsafe_zero_signal"], 1)
        self.assertEqual(record["own_loss_false_safe"], 1)
        self.assertEqual(record["own_loss_false_unsafe"], 2)
        summary = spec.summary([record])
        self.assertEqual(summary["candidate_evaluations"], 5)
        self.assertEqual(summary["own_loss_error_rate_informative"], .75)

    def test_unrestricted_pointwise_mass_predictions_have_zero_excess_loss(self):
        r = row()
        positive = np.array([[0., .1, .2, .3, .4, .5]])
        negative = np.array([[0., .6, .5, .4, .3, .2]])
        logits = np.zeros((1, 6))
        logits[:, 1:] = np.log(2*positive[:, 1:]/negative[:, 1:])
        scores = np.where(logits <= 0, -.1, .1)
        scores[:, 0] = 0.
        record = spec.records([r], mass(r, positive[0].tolist(), negative[0].tolist()), logits, scores, 2., {})[0]
        summary = spec.summary([record])
        self.assertAlmostEqual(summary["mass_weighted_excess_nats"], 0., places=12)
        self.assertAlmostEqual(summary["mass_weighted_share_MAE"], 0., places=12)
        self.assertEqual(summary["own_loss_error_rate_informative"], 0.)

    def test_zero_signal_and_repeated_model_evaluations_do_not_create_extra_samples(self):
        r = row()
        record = spec.records([r], mass(r, [0.]*6, [0.]*6), np.zeros((1, 6)), np.zeros((1, 6)), 1., {})[0]
        summary = spec.summary([record, record])
        self.assertEqual((summary["state_model_evaluations"], summary["unique_states"], summary["unique_prefix_groups"]), (2, 1, 1))
        self.assertEqual(summary["candidate_evaluations"], 10)
        self.assertIsNone(summary["own_loss_error_rate_informative"])
        self.assertIsNone(summary["mass_weighted_excess_nats"])
        self.assertEqual(spec.gaps(summary, summary)["mean_expected_boundary_regret_min"], 0.)

    def test_disjoint_half_signs_separate_sampling_disagreement_from_repeated_errors(self):
        r = row()
        values = np.zeros((8, 6, 4))
        values[:4, 1, 3], values[4:, 1, 3] = .1, -.2
        scores = np.array([[0., .1, 0., 0., 0., 0.]])
        record = spec.records([r], mass(r, [0., .05, 0., 0., 0., 0.], [0., .1, 0., 0., 0., 0.]),
            np.ones((1, 6)), scores, 1., {spec.source.state_key(r): {"values": values}})[0]
        self.assertEqual(record["first_last"]["sign_disagreement"], 1)
        self.assertEqual(record["first_last"]["stable_false_unsafe"], 0)
        self.assertEqual(record["even_odd"]["stable_false_unsafe"], 1)

    def test_batched_inference_matches_original_causal_row_inference_and_identity(self):
        rows = fixture()[0]["training_data"][:12]
        model = {"representation": "conditional_wait", "center": [0.]*198, "scale": [1.]*198,
            "coefficients": np.linspace(-.03, .03, 199).tolist()}
        raw = np.stack([spec.source.source.wait.features(r, "conditional_wait") for r in rows])
        _, shares, scores = spec.model_scores(raw, np.array([r["original_choice"] for r in rows]), model)
        for i, r in enumerate(rows):
            original = spec.source.feasibility.predict(r, model)
            np.testing.assert_allclose(shares[i], original["predicted_harm_mass_share"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores[i], original["predicted_feasibility_score"], atol=1e-12, rtol=0)
            self.assertEqual(scores[i, r["original_choice"]], 0.)

    def test_selected_harm_benefit_decomposition_keeps_identity_in_denominator(self):
        rows = [{"selected": s, "original_choice": 0, "wait_delta_min": w, "actual_cost_delta": c}
            for s, w, c in ((1, 2., -1.), (1, -3., -1.), (0, 0., 0.))]
        result = spec.selected_summary(rows)
        self.assertEqual((result["states"], result["overrides"], result["false_safe_overrides"]), (3, 2, 1))
        self.assertAlmostEqual(result["mean_harm_min"]-result["mean_benefit_min"], result["mean_wait_min"])

    def test_all47_folds_preserve_unique_cohorts_and_match_scene_training_strata(self):
        DecisionWaitTest.setUpClass()
        bank, registered = DecisionWaitTest.bank, spec.source.manifest(DecisionWaitTest.registered)
        keys = {spec.source.state_key(r) for r in registered["mean_states"]}
        panels = {}
        for r in bank["training_data"]:
            if spec.source.state_key(r) in keys:
                values = np.zeros((8, 6, 4))
                labels = {l["candidate"]: l["wait_delta_min"] for l in r["labels"]}
                values[..., 3] = [labels[i] for i in range(6)]
                panels[spec.source.state_key(r)] = {"original_choice": r["original_choice"], "values": values}
        masses = {spec.source.state_key(r): r for r in spec.source.build_masses(bank["training_data"], panels)}
        calls = []
        def load(fold):
            calls.append(fold)
            train, test, _ = spec.source.source.coverage.split_rows(bank, fold["split"], fold["index"])
            test = [r for r in test if spec.source.state_key(r) in keys]
            membership = sorted(map(list, {spec.source.source.paired.scene_group(r) for r in train}))
            models = {v: {"representation": "conditional_wait", "training_groups": membership, "training_states": len(train),
                "training_mean_states": 240-len(test), "sample_weight_sum": len(train)+7*(240-len(test)), "parameter_count": 199,
                "false_safe_cost": cost, "ridge": .1, "max_iterations": 200, "optimizer_iterations": 0, "solver_success": True,
                "center": [0.]*198, "scale": [1.]*198, "coefficients": [0.]*199}
                for v, cost in spec.source.feasibility.FALSE_SAFE_COSTS.items()}
            predictions = [{**r, "predicted_feasibility_score": [0.]*6,
                "predicted_harm_mass_share": [0. if i == r["original_choice"] else .5 for i in range(6)]} for r in fake_predictions(test)]
            cell = {"protocol": spec.source.EXPERIMENT_PROTOCOL, "contract": spec.source.contract(), "software_qualified": True,
                "fold": fold, "fits": {v: {k: x for k, x in m.items() if k not in ("center", "scale", "coefficients")} for v, m in models.items()},
                "predictions": {v: predictions for v in spec.source.VARIANTS}, "new_wait_fits": 2, "new_cost_fits": 0,
                "actor_updates": 0, "new_native_episodes": 0, "wait_optimizer_iterations": 0, "frozen_cost_fit": {"training_groups": membership}}
            return cell, models
        result = spec.analyze(bank, masses, panels, registered, load)
        self.assertEqual(len(calls), 47)
        for variant in result["variants"].values():
            for split, value in variant["validation"].items():
                held = value["held_out_mean8"]["overall"]
                self.assertEqual((held["state_model_evaluations"], held["unique_states"], held["unique_prefix_groups"]), (240, 240, 40))
            scene = variant["validation"]["scene"]
            self.assertEqual(scene["training_mean8"]["overall"]["state_model_evaluations"], 9360)
            matched = scene["scene_matched_training_mean8"]["overall"]
            self.assertEqual((matched["state_model_evaluations"], matched["unique_states"]), (720, 240))
        self.assertFalse(result["third_scene_screen_eligible"] or result["native_deployment_eligible"])
        with self.assertRaises(ValueError): spec.analyze(bank, {}, panels, registered, load)

    def test_scheduler_unbound_small_read_only_and_python310(self):
        task = task_specification("unit_wait_transfer")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
        self.assertNotIn("_raw", task["result_dir"])
        self.assertEqual(spec.contract()["budget"], dict(new_fits=0, optimizer_updates=0, actor_updates=0, new_native_episodes=0))
        for name in ("analyze_native_transit_wait_transfer_stage207.py", "submit_native_transit_wait_transfer_stage207_scheduleurm.py"):
            ast.parse((spec.ROOT/"scripts"/name).read_text(), feature_version=(3, 10))


if __name__ == "__main__":
    unittest.main()
