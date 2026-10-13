import ast
import unittest

import numpy as np
from scipy.special import expit

from scripts import analyze_native_transit_bound_ranking_stage214 as spec
from scripts.submit_native_transit_bound_ranking_stage214_scheduleurm import task_specification


def scores(logits, original):
    result = expit(logits)-.5
    result[original] = 0.
    return result


def values(cost, wait):
    result = np.zeros((8, 6, 4))
    result[..., 0], result[..., 3] = cost, wait
    return result


class BoundRankingTest(unittest.TestCase):
    def test_bound_enumeration_respects_locked_classes_original_and_cost_ties(self):
        cost, logits = np.asarray([0., -1., -2., -3., 1., -1.]), np.asarray([99., -2., 0., 2., 0., -.1])
        actions, safe, unsafe = spec.reachable(cost, logits, 0, 1.)
        self.assertEqual(actions, [1, 2])
        self.assertTrue(safe[0] and safe[1] and unsafe[3])
        self.assertFalse(unsafe[0])
        free, _, _ = spec.reachable(cost, logits, 0, None)
        self.assertEqual(free, [0, 1, 2, 3, 5])
        self.assertTrue(set(actions) <= set(free))
        tied, _, _ = spec.reachable(np.zeros(6), np.zeros(6), 3, None)
        self.assertEqual(tied, [3])

    def test_unavoidable_unsafe_choice_does_not_fall_back_to_unreachable_original(self):
        truth = np.asarray([[0., 0.], [-3., .2], [-5., .5], [1., 0.], [1., 0.], [1., 0.]])
        self.assertEqual(spec.oracle_choice([1, 2], truth, 0), 1)
        self.assertEqual(spec.oracle_choice([0, 1, 2], truth, 0), 0)
        truth[2, 1] = -.1
        self.assertEqual(spec.oracle_choice([0, 1, 2], truth, 0), 2)

    def test_other_future_half_only_is_evaluated_and_cannot_change_own_half_selection(self):
        cost, logits = np.asarray([0., -1., -2., -3., 1., -1.]), np.asarray([0., -2., 0., 2., 0., -.1])
        draw = values([0., -1., -2., 0., 0., 0.], [0., 1., -1., 1., 1., 1.])
        draw[4:, 2, 3] = 10.
        result = spec.panel(draw, cost, scores(logits, 0), logits, 0, 1, "first_last")
        best = result["selectors"]["bound_best"]
        self.assertEqual(best["selected"], [2, 1])
        self.assertEqual(best["selection_wait_min"], 0.)
        self.assertEqual(best["cross_wait_min"], 5.5)
        self.assertEqual(best["direction_wait_min"], [10., 1.])
        changed = draw.copy()
        changed[4:, 2, 3] = 20.
        other = spec.panel(changed, cost, scores(logits, 0), logits, 0, 1, "first_last")
        self.assertEqual(other["selectors"]["bound_best"]["selected"][0], best["selected"][0])
        self.assertEqual(result["errors"]["locked_false_safe_candidates"], 1)
        self.assertEqual(result["opportunities"]["no_safe_reachable_B"], 1)

    def test_cost_order_blocks_joint_gain_even_with_unbounded_wait_repair(self):
        cost, logits = np.asarray([0., -1., -2., -3., 1., -1.]), np.zeros(6)
        draw = values([0., 1., 2., 3., -4., 1.], [0., -1., -1., -1., -1., -1.])
        result = spec.panel(draw, cost, scores(logits, 0), logits, 0, 0, "even_odd")
        self.assertEqual(result["selectors"]["OO"]["selected"], [4, 4])
        self.assertEqual(result["selectors"]["free_best"]["selected"], [0, 0])
        self.assertEqual(result["opportunities"]["stable_joint_blocked_by_cost_states"], 1)
        row = {"root": 397, "scene_seed": 123, **result}
        summary = spec.summarize([row])
        self.assertEqual(summary["contrasts"]["cost_order_gap"]["cross_cost"], -4.)

    def test_bound_only_blocks_gain_and_half_sums_retain_physical_units(self):
        cost, logits = np.asarray([0., -1., -2., -3., 1., -1.]), np.asarray([0., -2., 2., 2., 0., 0.])
        draw = values([0., 1., -2., 2., 1., 1.], [0., 1., -1., 1., 1., 1.])
        result = spec.panel(draw, cost, scores(logits, 0), logits, 0, 1, "first_last")
        self.assertEqual(result["opportunities"]["stable_joint_blocked_by_bound_states"], 1)
        self.assertEqual(result["errors"]["locked_false_unsafe_candidates"], 1)
        self.assertEqual(result["selectors"]["free_best"]["selected"], [2, 2])
        summary = spec.summarize([{"root": 397, "scene_seed": 1, **result}, {"root": 397, "scene_seed": 2, **result}])
        self.assertEqual(summary["states"], 2)
        self.assertEqual(summary["directional_evaluations"], 4)
        self.assertEqual(summary["opportunities"]["mean_reachable_actions"], 1.)
        self.assertEqual(summary["selectors"]["free_best"]["mean_cross_wait_min"], -1.)

    def test_scheduler_is_unbound_small_report_only_no_fitting_python310(self):
        task = task_specification("unit_bound_rank")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
        self.assertNotIn("_raw", task["result_dir"])
        self.assertFalse(any("unit_bound_rank/inputs" in p for p in task["stage_input_paths"]))
        self.assertTrue(all(v == 0 for v in spec.contract()["budget"].values()))
        for path in (spec.ROOT/"scripts/analyze_native_transit_bound_ranking_stage214.py",
                spec.ROOT/"scripts/submit_native_transit_bound_ranking_stage214_scheduleurm.py"):
            ast.parse(path.read_text(), feature_version=(3, 10))


if __name__ == "__main__":
    unittest.main()
