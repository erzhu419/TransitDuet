import ast
import copy
import unittest
from unittest.mock import patch

import numpy as np

from scripts import analyze_native_transit_oracle_factorial_stage205 as spec
from scripts.submit_native_transit_oracle_factorial_stage205_scheduleurm import task_specification
from tests.test_native_transit_decision_wait import DecisionWaitTest


def draws(cost, wait):
    values = np.zeros((8, 6, 4))
    values[..., 0] = cost
    values[..., 2] = cost
    values[..., 3] = wait
    return values


class OracleFactorialTest(unittest.TestCase):
    def test_wait_error_is_separate_from_cost_ranking(self):
        cost, true_wait = np.array([0., -2., -1., 0., 0., 0.]), np.array([0., .1, -.1, 0., 0., 0.])
        result = spec.panel(draws(cost, true_wait), cost, np.zeros(6), 0, "first_last")
        self.assertEqual(result["selectors"]["LL"]["selected"], [1, 1])
        self.assertEqual(result["selectors"]["OL"]["selected"], [1, 1])
        self.assertEqual(result["selectors"]["LO"]["selected"], [2, 2])
        self.assertEqual(result["selectors"]["OO"]["cross_wait_min"], -.1)
        self.assertEqual(result["errors"]["stable_false_safe_candidates"], 1)
        self.assertEqual(result["errors"]["learned_selected_both_halves_wait_harm"], 1)

    def test_cost_error_is_separate_from_wait_filtering(self):
        cost, wait = np.array([0., -1., -2., 0., 0., 0.]), np.array([0., -.1, -.1, 0., 0., 0.])
        predicted = np.array([0., -2., -1., 0., 0., 0.])
        result = spec.panel(draws(cost, wait), predicted, wait, 0, "first_last")
        self.assertEqual(result["selectors"]["LO"]["selected"], [1, 1])
        self.assertEqual(result["selectors"]["OL"]["selected"], [2, 2])
        self.assertEqual(result["selectors"]["OO"]["cross_cost"], -2.)
        self.assertEqual(result["errors"]["stable_cost_inversions"], 1)

    def test_action_space_cost_wait_conflict_differs_from_model_error(self):
        cost, wait = np.array([0., -1., 0., 0., 0., 0.]), np.array([0., .1, 0., 0., 0., 0.])
        result = spec.panel(draws(cost, wait), cost, wait, 0, "first_last")
        self.assertEqual(result["selectors"]["OO"]["cross_cost"], 0.)
        self.assertEqual(result["selectors"]["unconstrained"]["cross_cost"], -1.)
        self.assertEqual(result["opportunities"]["mean8_cost_only_conflict_states"], 1)
        self.assertEqual(result["opportunities"]["both_halves_same_joint_action_states"], 0)

    def test_cross_evaluation_exposes_selection_optimism_without_using_eval_half_to_choose(self):
        values = draws(np.zeros(6), np.zeros(6))
        values[:4, 1, [0, 2]] = -1.
        values[4:, 1, [0, 2]] = 1.
        values[:, 1, 3] = -.1
        pred = np.zeros(6)
        result = spec.panel(values, pred, pred, 0, "first_last")["selectors"]["OO"]
        self.assertEqual(result["selected"], [1, 0])
        self.assertEqual((result["selection_cost"], result["cross_cost"], result["hindsight_cost"]), (-.5, .5, 0.))
        changed = values.copy()
        changed[4:, 1, [0, 2]] = 20.
        other = spec.panel(changed, pred, pred, 0, "first_last")["selectors"]["OO"]
        self.assertEqual(result["selected"][0], other["selected"][0])
        self.assertNotEqual(result["direction_cost"][0], other["direction_cost"][0])

    def test_stable_wait_error_and_cost_order_exclude_disagreeing_halves_and_unsafe_pairs(self):
        cost, wait = np.array([0., -1., -2., -3., 0., 0.]), np.array([0., .1, .1, -.1, 0., 0.])
        values = draws(cost, wait)
        values[4:, 2, 3] = -.2
        predicted = np.array([0., -3., -2., -1., 0., 0.])
        predicted_wait = np.zeros(6)
        predicted_wait[3] = .1
        result = spec.panel(values, predicted, predicted_wait, 0, "first_last")
        self.assertEqual(result["errors"]["stable_false_safe_candidates"], 1)
        self.assertEqual(result["errors"]["stable_false_unsafe_candidates"], 1)
        self.assertEqual(result["errors"]["wait_half_disagreement_candidates"], 1)
        self.assertEqual(result["errors"]["stable_cost_inversions"], 0)

    def test_identity_ties_and_candidate_permutation_preserve_outcomes(self):
        zeros = np.zeros(6)
        result = spec.panel(draws(zeros, zeros), zeros, zeros, 3, "first_last")
        self.assertTrue(all(s["selected"] == [3, 3] for s in result["selectors"].values()))
        rng = np.random.default_rng(205)
        values = rng.normal(size=(8, 6, 4))
        values[:, 0] = 0.
        cost, wait = rng.normal(size=6), rng.normal(size=6)
        cost[0] = wait[0] = 0.
        permutation = [4, 2, 0, 5, 1, 3]
        a = spec.panel(values, cost, wait, 0, "even_odd")
        b = spec.panel(values[:, permutation], cost[permutation], wait[permutation], 2, "even_odd")
        self.assertEqual(a["errors"], b["errors"])
        self.assertEqual(a["opportunities"], b["opportunities"])
        for selector in spec.SELECTORS:
            for key, value in a["selectors"][selector].items():
                if key == "selected":
                    self.assertEqual(value, [permutation[s] for s in b["selectors"][selector][key]])
                else:
                    np.testing.assert_allclose(value, b["selectors"][selector][key], rtol=0, atol=1e-12)

    def test_summary_uses_unique_states_and_keeps_nonadditive_interaction(self):
        cost, wait = np.array([0., -1., -2., 0., 0., 0.]), np.array([0., -.1, .1, 0., 0., 0.])
        p = spec.panel(draws(cost, wait), np.array([0., -2., -1., 0., 0., 0.]), np.zeros(6), 0, "first_last")
        rows = [{"root": 397, "scene_seed": 1, "probe_index": i, **p} for i in (8, 9)]
        summary = spec.summarize(rows)
        self.assertEqual((summary["states"], summary["prefix_groups"], summary["directional_evaluations"]), (2, 1, 4))
        self.assertEqual(summary["contrasts"]["interaction"]["cross_cost"], 1.)

    def test_prediction_join_checks_mean8_labels_and_identity(self):
        values = draws(np.array([0., -1., 0., 0., 0., 0.]), np.zeros(6))
        p = {"original_choice": 0, "values": values}
        prediction = {"original_choice": 0, "selected": 1, "predicted_heads": values.mean(axis=0)[:, [1, 2]].tolist(),
            "predicted_wait_delta_min": [0.]*6, "actual_cost_delta": -1., "wait_delta_min": 0.}
        spec.validate_panel_prediction(p, prediction)
        bad = copy.deepcopy(prediction)
        bad["actual_cost_delta"] = -2.
        with self.assertRaises(ValueError): spec.validate_panel_prediction(p, bad)
        p["values"][:, 0, 3] = .1
        with self.assertRaises(ValueError): spec.validate_panel_prediction(p, prediction)

    def test_all_folds_and_partitions_keep_one240state_cohort_and_closed_claims(self):
        DecisionWaitTest.setUpClass()
        registered = DecisionWaitTest.registered
        panels = {spec.source.state_key(r): {**r, "original_choice": 0, "values": np.zeros((8, 6, 4))}
            for r in registered["mean_states"]}
        cells = {}
        for fold in registered["folds"]:
            predictions = [{**r, "original_choice": 0, "selected": 0, "predicted_heads": [[0., 0.]]*6,
                "predicted_wait_delta_min": [0.]*6, "actual_cost_delta": 0., "wait_delta_min": 0.}
                for r in registered["mean_states"] if spec.source.credit.group_key(r, fold["split"]) == fold["held_out"]]
            cells[fold["split"], fold["index"]] = {"predictions": {v: predictions for v in spec.source.VARIANTS}}
        with patch.object(spec.source, "qualify") as qualify:
            result = spec.analyze(panels, cells, registered)
        self.assertEqual(qualify.call_count, 47)
        for partitions in result["variants"].values():
            for partition in partitions.values():
                self.assertFalse(partition["gates"]["OO"]["cost"])
                for split in partition["validation"].values():
                    self.assertEqual((split["overall"]["states"], split["overall"]["prefix_groups"]), (240, 40))
        self.assertFalse(result["third_scene_screen_eligible"] or result["native_deployment_eligible"])
        with self.assertRaises(ValueError): spec.analyze(panels, {}, registered)

    def test_scheduler_small_read_only_unbound_and_python310(self):
        task = task_specification("unit_oracle_factorial")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
        self.assertNotIn("_raw", task["result_dir"])
        self.assertEqual(spec.contract()["budget"], dict(new_fits=0, critic_updates=0, actor_updates=0, new_native_episodes=0))
        for name in ("analyze_native_transit_oracle_factorial_stage205.py", "submit_native_transit_oracle_factorial_stage205_scheduleurm.py"):
            ast.parse((spec.ROOT/"scripts"/name).read_text(), feature_version=(3, 10))


if __name__ == "__main__":
    unittest.main()
