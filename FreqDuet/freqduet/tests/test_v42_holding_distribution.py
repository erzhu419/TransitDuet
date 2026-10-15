from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freqduet_closed_loop_forecast import ACTIONS, holding_features, fluid_rollout
from freqduet_holding_distribution import ensemble, fit_distribution, probabilities, sample_action
from test_v41_closed_loop import observation


def values():
    return dict(forward_gap_s=360, target_headway_s=360, follower_gap_s=0, follower_valid=False,
                load_ratio=.5, previous_hold_s=0, station_phase=.5, direction=True, time_s=100)


class HoldingDistributionTest(unittest.TestCase):
    def test_constant_state_preserves_seven_bin_frequencies_and_mean(self):
        x = holding_features(values())
        labels = [0] * 64 + [5] * 16 + [10] * 8 + [15] * 4 + [20] * 4 + [30] * 8 + [45] * 24
        model = fit_distribution([(i, x, action / 45) for i, action in enumerate(labels)], 200)
        probs = probabilities(model, values())
        expected = (np.array([labels.count(a) for a in ACTIONS]) + .5) / (len(labels) + 3.5)
        np.testing.assert_allclose(probs, expected, atol=1e-4)
        self.assertAlmostEqual(probs.sum(), 1)
        self.assertTrue(np.all(probs > 0))
        self.assertAlmostEqual(float(probs @ ACTIONS), float(expected @ ACTIONS), places=3)

    def test_bimodal_actions_are_not_collapsed_to_mean_or_argmax(self):
        probs = np.array([.5, 0, 0, 0, 0, 0, .5])
        actions = [sample_action(probs, u) for u in (.1, .4, .6, .9)]
        self.assertEqual(actions, [0, 0, 45, 45])
        self.assertEqual(np.mean(actions), 22.5)

    def test_same_time_and_future_labels_are_excluded(self):
        x = holding_features(values())
        rows = [(i, x, 0) for i in range(40)]
        before = fit_distribution(rows, 40)
        after = fit_distribution(rows + [(40, x, 1), (41, x, 1)], 40)
        self.assertEqual(before, after)
        self.assertEqual(after["latest_time_s"], 39)

    def test_non_deployed_label_is_a_hard_error(self):
        with self.assertRaises(ValueError):
            fit_distribution([(0, holding_features(values()), 22 / 45)], 1)

    def test_fit_works_inside_read_only_actor_evaluation(self):
        import torch
        with torch.no_grad():
            model = fit_distribution([(i, holding_features(values()), 1) for i in range(40)], 100)
        self.assertIsNotNone(model["coef"])
        self.assertTrue(np.all(np.isfinite(model["coef"])))

    def test_fit_and_ensemble_do_not_consume_simulator_rng(self):
        import torch
        np_before, torch_before = np.random.get_state(), torch.random.get_rng_state().clone()
        model = fit_distribution([(i, holding_features(values()), 1) for i in range(40)], 100)
        first = ensemble(observation(), 100, model, draws=8)
        second = ensemble(observation(), 100, model, draws=8)
        self.assertEqual(first, second)
        after = np.random.get_state()
        self.assertEqual(np_before[0], after[0])
        np.testing.assert_array_equal(np_before[1], after[1])
        self.assertEqual(np_before[2:], after[2:])
        self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))

    def test_discrete_callback_preserves_capacity_mass_and_action_mask(self):
        observed = observation()
        observed["flows"]["True"][1].update(queue=100, rate_per_s=.1)
        result = fluid_rollout(observed, 200, "response", lambda _: 45)
        self.assertLessEqual(result["max_load"], 50)
        self.assertAlmostEqual(result["final_queue"] + result["final_onboard"] + result["alighted_total"], 125)
        self.assertTrue(all(row["action_s"] in ACTIONS for row in result["commands"]))
        observed["routes"]["True"]["holding_mask"][1] = True
        self.assertEqual(fluid_rollout(observed, 200, "response", lambda _: 45)["holding_vehicle_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
