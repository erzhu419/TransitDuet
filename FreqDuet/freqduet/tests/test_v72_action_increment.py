from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import freqduet_action_increment as model
import run_freqduet_v72_action_increment as audit


def fitting_data():
    states = np.array([[0., 1.], [1., 1.], [2., 1.], [3., 1.], [100., 1.], [200., 1.]])
    native = np.array([0, 1, 2, 3, 4, 5])
    days = np.array([1, 1, 2, 2, 3, 3])
    x = model.design(states, native, np.zeros(2), np.ones(2), True)
    targets = x @ np.arange(x.shape[-1] * 5).reshape(x.shape[-1], 5) / 10.
    return states, native, targets, days


class ActionIncrementTest(unittest.TestCase):
    def test_causal_features_keep_all_encoded_state_but_exclude_replay_privilege(self):
        capture = dict(encoded_state=np.arange(10), prefix_cv_squared=.1,
                       actor_probabilities=np.full(7, 1 / 7), rng="privileged", exposure="unused",
                       future_return_s=10000, scenario_seed=700006)
        first = model.observable_features(capture)
        capture.update(rng=None, future_return_s=-10000, scenario_seed=-1)
        np.testing.assert_array_equal(first, model.observable_features(capture))
        np.testing.assert_array_equal(first[:10], np.arange(10))
        self.assertEqual(len(first), 18)

    def test_targets_subtract_same_future_native_before_averaging(self):
        outcomes = np.zeros((2, 3, 7, 5))
        for context in range(2):
            for future in range(3):
                outcomes[context, future, :, :] = 100 * future + np.arange(7)[:, None]
        target = model.mean_differences(outcomes, np.array([1, 4]))
        np.testing.assert_array_equal(target[0, :, 0], np.arange(7) - 1)
        np.testing.assert_array_equal(target[1, :, 0], np.arange(7) - 4)

    def test_anchor_antisymmetry_and_transitivity(self):
        states = np.ones((3, 2))
        x = model.design(states, np.array([0, 1, 2]), np.zeros(2), np.ones(2), True)
        np.testing.assert_array_equal(x[np.arange(3), [0, 1, 2]], 0)
        np.testing.assert_allclose(x[0, 1], -x[1, 0], atol=0, rtol=0)
        np.testing.assert_allclose(x[0, 1] + x[1, 2], x[0, 2], atol=1e-15, rtol=0)

    def test_quadratic_can_choose_an_interior_holding_bin(self):
        coefficients = np.zeros((2, 5))
        coefficients[:, 0] = [-2 / 3, 1.]
        fitted = model.IncrementModel("action_only", np.zeros(1), np.ones(1), coefficients, 2.)
        predictions = fitted.predict(np.zeros((1, 1)), np.array([0]))[0]
        self.assertEqual(model.ACTIONS[model.choose_action(predictions, 0)], 15)

    def test_every_constraint_blocks_an_apparent_passenger_gain(self):
        for constraint in range(1, 5):
            values = np.zeros((7, 5))
            values[1, 0], values[1, constraint] = -10, 1e-5
            self.assertEqual(model.choose_action(values, 4), 4)
        self.assertEqual(model.choose_action(np.zeros((7, 5)), 4), 4)

    def test_entire_held_day_is_excluded_from_scaler_and_coefficients(self):
        states, native, targets, days = fitting_data()
        fitted = model.fit_excluding_day(states, native, targets, days, 3, "state_quadratic")
        states[days == 3] = -1e6
        targets[days == 3] = 1e9
        second = model.fit_excluding_day(states, native, targets, days, 3, "state_quadratic")
        np.testing.assert_array_equal(fitted.center, [1.5, 1.])
        np.testing.assert_array_equal(fitted.center, second.center)
        np.testing.assert_array_equal(fitted.coefficients, second.coefficients)
        np.testing.assert_array_equal(fitted.predict(states, native)[np.arange(6), native], 0.)

    def test_whole_day_replication_does_not_change_fitting_weight(self):
        states, native, targets, days = fitting_data()
        first = model.fit_model(states, native, targets, days, "state_quadratic")
        indices = np.r_[np.arange(6), [0, 1, 0, 1]]
        second = model.fit_model(states[indices], native[indices], targets[indices], days[indices], "state_quadratic")
        np.testing.assert_allclose(first.center, second.center, atol=1e-12, rtol=0)
        np.testing.assert_allclose(first.coefficients, second.coefficients, atol=1e-12, rtol=0)

    def test_error_excludes_the_exact_zero_native_label(self):
        truth, predicted = np.ones(7), np.zeros(7)
        truth[3] = 0
        result = model.error_statistics(predicted, truth, 3)
        self.assertEqual((result["mae"], result["mse"], result["sign_pairs"]), (1., 1., 6))

    def test_trip_zero_targets_remain_exactly_zero(self):
        states, native, targets, days = fitting_data()
        targets[:, :, 3:] = 0
        fitted = model.fit_model(states, native, targets, days, "state_quadratic")
        np.testing.assert_array_equal(fitted.predict(states, native)[:, :, 3:], 0.)

    def test_interval_clusters_demand_days_and_not_continuation_rows(self):
        rows = [dict(scenario_seed=day, value=float(day)) for day in (1, 2, 3) for _ in range(12)]
        result = audit.interval(rows, lambda r: r["value"])
        self.assertEqual((result["n_demand_days"], result["mean"]), (3, 2.))

    def test_prediction_and_joint_gates_remain_separate(self):
        contrasts = {c: {m: {loss: dict(ci95=[-2., -1.]) for loss in ("mae", "mse")}
                          for m in audit.PRIMARY} for c in ("zero", "action_only")}
        effects = {m: dict(ci95=[-2., -1.]) for m in model.METRICS}
        self.assertTrue(all(audit.gates(contrasts, effects).values()))
        effects["readiness_delay_s"]["ci95"][1] = .01
        self.assertFalse(audit.gates(contrasts, effects)["joint_effect_gate_passed"])
        contrasts["zero"]["cv_squared"]["mse"]["ci95"][1] = .01
        self.assertFalse(audit.gates(contrasts, effects)["prediction_gate_passed"])


if __name__ == "__main__":
    unittest.main()
