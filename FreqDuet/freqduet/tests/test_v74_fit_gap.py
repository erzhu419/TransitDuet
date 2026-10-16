from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v74_fit_gap as audit


class FitGapTest(unittest.TestCase):
    def test_cached_model_roundtrip_keeps_predictions_and_anchor_without_refit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            fitted = audit.model.IncrementModel("state_quadratic", np.array([1., 2.]), np.array([2., 3.]),
                                               np.arange(30.).reshape(6, 5), 0.)
            audit.old.save_model(path, fitted)
            loaded = audit.load_model(path)
            states, native = np.array([[2., 3.], [4., 5.]]), np.array([1, 3])
            np.testing.assert_array_equal(loaded.predict(states, native), fitted.predict(states, native))
            np.testing.assert_array_equal(loaded.predict(states, native)[np.arange(2), native], 0.)

    def test_support_uses_other_training_days_of_same_actor_and_window(self):
        data = dict(days=np.array([1, 2, 3, 4, 5]), cases=np.array([0, 0, 0, 1, 0]),
                    windows=np.array([7200, 7200, 7200, 7200, 21600]))
        data.update({b: np.array([[0., 7.], [1., 7.], [3., 9.], [3., 9.], [3., 9.]]) for b in audit.BLOCKS})
        result = audit.support(data, np.array([0, 1, 3, 4]), np.array([2]), np.zeros(6), np.ones(6))[0]
        for block in audit.BLOCKS:
            self.assertEqual(result[block], dict(matched_days=2, variable_features=1, outside_fraction=1.,
                                               novel_constant_features=1, nearest_rms=2.))
        data["days"][0] = 3
        result = audit.support(data, np.array([0, 1, 3, 4]), np.array([2]), np.zeros(6), np.ones(6))[0]
        self.assertEqual(result["states"]["matched_days"], 1)
        self.assertEqual(result["states"]["variable_features"], 0)
        self.assertEqual(result["states"]["novel_constant_features"], 2)

    def test_support_does_not_fit_scaling_to_query_or_use_labels(self):
        data = dict(days=np.array([1, 2, 3]), cases=np.zeros(3), windows=np.ones(3) * 7200,
                    outcomes=np.full((3, 32, 7, 11), 999.))
        data.update({b: np.array([[0.], [2.], [4.]]) for b in audit.BLOCKS})
        first = audit.support(data, np.array([0, 1]), np.array([2]), np.zeros(3), np.ones(3) * 2)
        data["outcomes"] *= -10
        self.assertEqual(first, audit.support(data, np.array([0, 1]), np.array([2]), np.zeros(3), np.ones(3) * 2))
        self.assertEqual(first[0]["states"]["nearest_rms"], 1.)

    def test_pairing_averages_repeated_fits_per_original_context_not_new_samples(self):
        def row(index, value):
            return dict(context_index=index, scenario_seed=index + 1, case_index=0, target_time_s=7200,
                channels={c: {m: {loss: value for loss in audit.LOSSES} for m in audit.old.PRIMARY} for c in audit.CHANNELS},
                support={b: dict(nearest_rms=value) for b in audit.BLOCKS})
        folds = [dict(held_day=i + 1, held=[row(i, 100.)],
                      train=[row(j, float(i)) for j in range(3) if j != i]) for i in range(3)]
        paired = audit.pair_rows(folds)
        self.assertEqual(len(paired), 3)
        self.assertEqual(paired[0]["train"]["fleet_plan"]["passenger_person_s"]["mae"], 1.5)
        self.assertEqual(paired[0]["held"]["fleet_plan"]["passenger_person_s"]["mae"], 100.)
        broken = deepcopy(folds)
        broken[0]["train"][0]["scenario_seed"] = 1
        with self.assertRaises(ValueError):
            audit.pair_rows(broken)


if __name__ == "__main__":
    unittest.main()
