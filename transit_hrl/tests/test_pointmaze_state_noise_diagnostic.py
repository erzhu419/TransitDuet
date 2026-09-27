from copy import deepcopy
import unittest

import numpy as np

from freq_hrl.domains.mujoco import PointMazeRegimeDriver
from freq_hrl.experiments import pointmaze_state_noise_diagnostic as diagnostic
from scripts import pointmaze_state_noise_diagnostic_spec as spec
from scripts.submit_pointmaze_deployed_pair_diagnostic_scheduleurm import task_specification
from tests.test_pointmaze_paired_value_qualification import pair


class StateNoiseDiagnosticTest(unittest.TestCase):
    def test_future_preserves_past_and_does_not_read_discarded_future(self):
        driver = PointMazeRegimeDriver(seed=71, horizon=400, dt_seconds=0.01)
        fields = ("_targets", "_forces", "_distractors", "_regime_ids")
        original = {k: getattr(driver, k).copy() for k in fields}
        cuts = (0, driver.pulse_start_steps[0], driver.pulse_start_steps[0] + 1,
                driver._pulse_stop_steps[0], driver.regime_change_steps[0], 200)
        for cut in cuts:
            first = driver.conditional_future(step=cut, seed=181)
            second = driver.conditional_future(step=cut, seed=182)
            redrawn = first.conditional_future(step=cut, seed=182)
            for key in fields:
                np.testing.assert_array_equal(getattr(driver, key), original[key])
                np.testing.assert_array_equal(getattr(first, key)[:cut + 1], original[key][:cut + 1])
                np.testing.assert_array_equal(getattr(second, key), getattr(redrawn, key))
            for key in ("_regime_change_steps", "_pulse_start_steps", "_pulse_stop_steps",
                        "_distractor_change_steps"):
                self.assertEqual(getattr(second, key), getattr(redrawn, key))
        self.assertFalse(np.array_equal(first._targets[201:], second._targets[201:]))

    def test_remaining_duration_is_conditioned_on_survival_not_restarted(self):
        class LowerBoundRng:
            def integers(self, low, high):
                self.bounds = (low, high)
                return low

        driver = PointMazeRegimeDriver(seed=71, horizon=400, dt_seconds=0.01)
        rng = LowerBoundRng()
        self.assertEqual(driver._conditional_next_step(rng, (0.8, 1.6), start=20, step=110), 111)
        self.assertEqual(rng.bounds, (91, 161))
        self.assertEqual(driver._conditional_next_step(rng, (0.8, 1.6), start=20, step=30), 100)
        self.assertEqual(rng.bounds, (80, 161))

    def test_pulse_duration_and_gap_law_including_zero_amplitude(self):
        for amplitude in (0.0, 0.18):
            driver = PointMazeRegimeDriver(seed=71, horizon=400, dt_seconds=0.01,
                                          force_pulse_amplitude=amplitude)
            for cut in (driver.pulse_start_steps[0] + 1, driver._pulse_stop_steps[0]):
                future = driver.conditional_future(step=cut, seed=1801)
                for start, stop in zip(future._pulse_start_steps, future._pulse_stop_steps):
                    if stop <= future.horizon:
                        self.assertTrue(4 <= stop - start <= 10)
                    np.testing.assert_array_equal(future._forces[start:stop],
                                                  np.tile(future._forces[start], (stop - start, 1)))
                for end, start in zip(future._pulse_stop_steps, future._pulse_start_steps[1:]):
                    self.assertTrue(45 <= start - end <= 110)
                if amplitude == 0:
                    self.assertTrue(np.all(future._forces == 0.0))

    def test_history_comparison_has_equal_shape_and_parameter_count(self):
        pairs = [pair(seed, step) for seed in (1, 2) for step in (50, 100)]
        for row in pairs:
            for arm in ("now", "wait"):
                point = row[f"{arm}_endpoint"]
                point["features"] = [*np.zeros(37), 0.1, 0.5, 0.0]
                point["feature_names"] = [*[f"summary_{i}" for i in range(37)],
                                          "within_bin_fraction", "remaining_fraction", "budget_spent"]
                point["features"][0] = 0.1 if arm == "now" else 0.2
                point["controller_history"] = (np.arange(384) / 384).tolist()
        original = deepcopy(pairs)
        compact = diagnostic.history_endpoints(pairs, representation="compact")
        full = diagnostic.history_endpoints(pairs, representation="full_history")
        self.assertEqual(pairs, original)
        self.assertEqual(compact[0]["feature_names"], full[0]["feature_names"])
        self.assertEqual(len(full[0]["features"]), 424)
        self.assertTrue(all(v == 0 for v in compact[0]["features"][40:]))
        np.testing.assert_array_equal(full[0]["features"][40:], np.arange(384) / 384)
        result = diagnostic.qualify_states(pairs, root=18, fit_seeds=[1, 2], eval_seeds=[3], epochs=2)
        self.assertTrue(all(f["parameter_count"] == 31425 and f["state_dim"] == 424
                            and f["held_out_path"] not in f["training_paths"] for f in result["folds"]))
        with self.assertRaisesRegex(ValueError, "seed roles"):
            diagnostic.qualify_states(pairs, root=18, fit_seeds=[1, 2], eval_seeds=[2], epochs=2)

    def test_noise_correction_is_unclipped_and_selection_does_not_use_labels(self):
        rows = [{"replicate_tail_advantages": samples, "zero_prediction": 0.,
                 "compact_prediction": 2., "full_history_prediction": mean}
                for samples, mean in (([0., 2.], 1.), ([2., 4.], 3.))]
        summary = diagnostic.noise_summary(rows)
        self.assertEqual(summary["mean_conditional_variance"], 2.)
        self.assertEqual(summary["variance_of_conditional_means_estimate"], 1.)
        self.assertEqual(summary["conditional_mean_mse_estimate"],
                         {"zero": 4., "compact": 0., "full_history": -1.})
        pairs = [pair(1, step) for step in (50, 100, 150)]
        chosen = diagnostic.select_noise_checks(pairs, root=18, seed=1, count=2)
        for row in pairs:
            row["actual_tail_advantage"] += 100
        self.assertEqual(chosen, diagnostic.select_noise_checks(pairs, root=18, seed=1, count=2))

    def test_boundary_check_accepts_future_changes_but_rejects_past_changes(self):
        original = {"value_trace": [{"step": 100, "features": [1.], "controller_history": [2.]}],
                    "decision_steps": [0, 50, 100], "window_ise": 0.1}
        actual = deepcopy(original)
        actual["decision_steps"][-1] = 125
        diagnostic.require_same_boundary(actual, original, step=100)
        actual["value_trace"][0]["controller_history"][0] = 3.
        with self.assertRaisesRegex(RuntimeError, "pre-boundary"):
            diagnostic.require_same_boundary(actual, original, step=100)

    def test_scheduler_stages_both_small_sources_without_pin(self):
        task = task_specification("unit_state_noise", 209011, preflight=False, protocol_spec=spec)
        for key, path in spec.input_results(209011, preflight=False).items():
            self.assertIn("--" + key.replace("_", "-"), task["cmd"])
            self.assertIn(str(path.parent), task["stage_input_paths"])
        self.assertIn("--future-replicates 8", task["cmd"])
        self.assertIn("--noise-pairs-per-seed 2", task["cmd"])
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["cpu"], 1)


if __name__ == "__main__":
    unittest.main()
