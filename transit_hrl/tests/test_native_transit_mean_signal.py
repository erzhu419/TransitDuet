import ast
import unittest

import numpy as np

from scripts import analyze_native_transit_mean_signal_stage196 as spec
from scripts.submit_native_transit_mean_signal_stage196_scheduleurm import task_specification


def prediction(values, selected=1, original=0):
    mean = values.mean(axis=0)
    return {"predicted_heads": mean[:, [1, 2]].tolist(), "predicted_wait_delta_min": mean[:, 3].tolist(),
        "selected": selected, "original_choice": original}


class MeanSignalTest(unittest.TestCase):
    def test_pair_error_subtracts_mean_noise_without_clipping_negative_estimates(self):
        values = np.zeros((8, 6, 4))
        values[:, 1, 0] = [1., -1.]*4
        values[:, 1, 1] = values[:, 1, 0]
        panel = spec.prediction_panel(values, prediction(values))
        for field in ("actual_cost_delta", "fleet_cost_delta"):
            metric = panel["pair_metrics"][field]
            self.assertEqual(metric["observed_pair_mse"], 0.)
            self.assertGreater(metric["mean_label_noise"], 0.)
            self.assertAlmostEqual(metric["noise_corrected_pair_mse"], -metric["mean_label_noise"])
            self.assertAlmostEqual(metric["zero_difference_pair_mse"], -metric["mean_label_noise"])
            self.assertEqual(metric["error_change_vs_zero"], 0.)

    def test_exact_predictor_beats_zero_difference_on_constant_action_effect(self):
        values = np.zeros((8, 6, 4))
        values[:, 1, 0] = -.2
        values[:, 1, 2] = -.2
        values[:, 1, 3] = -.1
        panel = spec.prediction_panel(values, prediction(values))
        for field in ("actual_cost_delta", "nonfleet_cost_delta", "wait_delta_min"):
            metric = panel["pair_metrics"][field]
            self.assertAlmostEqual(metric["noise_corrected_pair_mse"], 0.)
            self.assertGreater(metric["zero_difference_pair_mse"], 0.)
            self.assertLess(metric["error_change_vs_zero"], 0.)
        self.assertEqual(panel["false_feasible"], 0)

    def test_cross_half_choice_exposes_oracle_optimism_and_exact_ties_keep_identity(self):
        values = np.zeros((8, 6, 4))
        values[:, 1, 0] = [-1., 1.]*4
        values[:, 1, 2] = values[:, 1, 0]
        values[:, 1, 3] = -.1
        panel = spec.oracle_panel(values, 0)
        self.assertEqual(panel["half_choice_agreement"], 0)
        self.assertEqual(panel["same_half_cost"], -.5)
        self.assertEqual(panel["cross_half_cost"], .5)
        self.assertEqual(panel["mean8_hindsight_cost"], 0.)
        self.assertEqual(panel["mean8_hindsight_wait_min"], 0.)

    def test_false_feasibility_distinguishes_repeated_harm_from_half_disagreement(self):
        values = np.zeros((8, 6, 4))
        values[:, 1, 3] = [1., -.5]*4
        pred = prediction(values)
        pred["predicted_wait_delta_min"] = [0.]*6
        panel = spec.prediction_panel(values, pred)
        self.assertEqual((panel["false_feasible"], panel["false_feasible_both_halves_harm"],
            panel["selected_wait_half_disagreement"]), (1, 0, 1))
        values[:, 1, 3] = .1
        panel = spec.prediction_panel(values, pred)
        self.assertEqual((panel["false_feasible"], panel["false_feasible_both_halves_harm"],
            panel["selected_wait_half_disagreement"]), (1, 1, 0))

    def test_candidate_permutation_preserves_pair_statistics_and_oracle(self):
        rng = np.random.default_rng(196)
        values = rng.normal(size=(8, 6, 4))
        values[:, 0] = 0.
        values[..., 0] = values[..., 1]+values[..., 2]
        pred = prediction(values, selected=2)
        permutation = [4, 2, 0, 5, 1, 3]
        reordered = prediction(values[:, permutation], selected=1, original=2)
        a, b = spec.prediction_panel(values, pred), spec.prediction_panel(values[:, permutation], reordered)
        for field in spec.FIELDS:
            for key in a["pair_metrics"][field]:
                self.assertAlmostEqual(a["pair_metrics"][field][key], b["pair_metrics"][field][key])
        self.assertEqual(spec.oracle_panel(values, 0), spec.oracle_panel(values[:, permutation], 2))

    def test_summary_counts_prefix_groups_and_preserves_closed_claim_boundary(self):
        values = np.zeros((8, 6, 4))
        rows = [{"root": r, "scenario": s, "scene_seed": r*100+i, "probe_index": 8,
            **spec.prediction_panel(values, prediction(values, selected=0))}
            for r in spec.source.ROOTS for i, s in enumerate(spec.source.SCENARIOS)]
        summary = spec.scopes(rows)
        self.assertEqual((summary["overall"]["states"], summary["overall"]["prefix_groups"]), (10, 10))
        self.assertEqual(summary["overall"]["changed_choice"], 0)
        self.assertEqual(spec.contract()["budget"], {"new_fits": 0, "critic_updates": 0, "actor_updates": 0, "new_native_episodes": 0})

    def test_scheduler_runs_server_only_diagnostics_and_scripts_support_python310(self):
        task = task_specification("unit_signal")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual(task["cpu"], 1)
        self.assertNotIn("_raw", task["result_dir"])
        self.assertNotIn("collect", task["cmd"])
        for name in ("analyze_native_transit_mean_signal_stage196.py", "submit_native_transit_mean_signal_stage196_scheduleurm.py"):
            ast.parse((spec.ROOT/"scripts"/name).read_text(), feature_version=(3, 10))


if __name__ == "__main__":
    unittest.main()
