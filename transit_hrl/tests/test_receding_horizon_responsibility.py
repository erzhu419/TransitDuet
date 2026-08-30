import unittest

import numpy as np

from freq_hrl.core.receding_horizon_responsibility import (
    CausalRecedingHorizonResponsibilityPlanner,
    future_rolling_mean_system,
)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    rows = []
    for index in range(values.shape[0]):
        start = max(0, index - int(window) + 1)
        rows.append(np.mean(values[start:index + 1], axis=0))
    return np.stack(rows, axis=0)


class RecedingHorizonResponsibilityTest(unittest.TestCase):
    def test_future_rolling_system_matches_direct_prefix_computation(self):
        rng = np.random.default_rng(7)
        past = rng.normal(size=(5, 2))
        future = rng.normal(size=(9, 2))
        matrix, offset = future_rolling_mean_system(
            past, horizon=9, window=8
        )
        predicted = matrix @ future + offset
        direct = _rolling_mean(np.concatenate((past, future)), 8)[5:]
        np.testing.assert_allclose(predicted, direct, atol=1e-14, rtol=0.0)

    def test_zero_trace_is_jointly_feasible_and_exactly_reconstructed(self):
        planner = CausalRecedingHorizonResponsibilityPlanner(
            planning_horizon=8,
            coordinate_sweeps=24,
            multiplier_bisection_steps=8,
        )
        planner.reset(3)
        for _ in range(12):
            row = planner.split(np.zeros(3))
            self.assertTrue(row["joint_feasible_forecast"])
            self.assertEqual(row["status"], "joint_frequency_budgets_feasible_forecast")
            np.testing.assert_array_equal(row["upper"], np.zeros(3))
            np.testing.assert_array_equal(row["lower"], np.zeros(3))
            np.testing.assert_array_equal(
                row["reconstruction_error"], np.zeros(3)
            )

    def test_component_bounds_and_reconstruction_hold_for_dynamic_trace(self):
        planner = CausalRecedingHorizonResponsibilityPlanner(
            planning_horizon=12,
            coordinate_sweeps=32,
            multiplier_bisection_steps=8,
        )
        planner.reset(2)
        time = np.arange(40, dtype=np.float64)
        trace = np.stack(
            (
                0.65 * np.sin(0.21 * time),
                0.45 * np.cos(0.37 * time),
            ),
            axis=1,
        )
        for total in trace:
            row = planner.split(
                total, upper_limit=0.7, lower_limit=0.3
            )
            self.assertLessEqual(float(np.max(np.abs(row["upper"]))), 0.7 + 1e-7)
            self.assertLessEqual(float(np.max(np.abs(row["lower"]))), 0.3 + 1e-7)
            np.testing.assert_allclose(
                np.asarray(row["upper"], dtype=np.float64)
                + np.asarray(row["lower"], dtype=np.float64),
                total,
                atol=6e-8,
                rtol=0.0,
            )
            self.assertTrue(np.isfinite(row["upper_power_forecast"]))
            self.assertTrue(np.isfinite(row["lower_power_at_upper_floor_forecast"]))

    def test_prefix_outputs_do_not_depend_on_future_totals(self):
        shared = np.asarray(
            [[0.1], [0.25], [-0.05], [0.3], [0.15]], dtype=np.float64
        )
        first = np.concatenate((shared, np.full((5, 1), 0.8)), axis=0)
        second = np.concatenate((shared, np.full((5, 1), -0.8)), axis=0)

        def route(trace: np.ndarray) -> np.ndarray:
            planner = CausalRecedingHorizonResponsibilityPlanner(
                planning_horizon=8,
                coordinate_sweeps=32,
                multiplier_bisection_steps=8,
            )
            planner.reset(1)
            return np.stack([planner.split(row)["upper"] for row in trace])

        np.testing.assert_array_equal(route(first)[:5], route(second)[:5])

    def test_budget_ledger_carries_unused_upper_energy_forward(self):
        planner = CausalRecedingHorizonResponsibilityPlanner(
            planning_horizon=8,
            coordinate_sweeps=24,
            multiplier_bisection_steps=8,
            use_budget_ledger=True,
        )
        planner.reset(1)
        first = planner.split(np.zeros(1))
        second = planner.split(np.zeros(1))
        self.assertAlmostEqual(
            first["upper_budget_power_forecast"], 0.075 ** 2
        )
        self.assertGreater(
            second["upper_budget_power_forecast"],
            first["upper_budget_power_forecast"],
        )

    def test_prefix_projection_enforces_cumulative_upper_budget(self):
        planner = CausalRecedingHorizonResponsibilityPlanner(
            planning_horizon=12,
            forecast_mode="hold",
            coordinate_sweeps=24,
            multiplier_bisection_steps=8,
            enforce_prefix_upper_budget=True,
        )
        planner.reset(2)
        time = np.arange(60, dtype=np.float64)
        trace = np.stack(
            (
                0.55 * np.sin(0.37 * time),
                0.45 * np.cos(0.29 * time),
            ),
            axis=1,
        )
        upper_rows = []
        for total in trace:
            row = planner.split(total)
            self.assertTrue(row["prefix_upper_budget_feasible"])
            upper_rows.append(row["upper"])
            upper = np.stack(upper_rows)
            high = upper - _rolling_mean(upper, 8)
            self.assertLessEqual(
                float(np.mean(np.square(high))),
                0.075 ** 2 + 2e-8,
            )

    def test_policy_context_retains_v17_4_state_shape(self):
        planner = CausalRecedingHorizonResponsibilityPlanner(
            planning_horizon=8,
            coordinate_sweeps=24,
            multiplier_bisection_steps=8,
        )
        planner.reset(4)
        action_blocks, scalars = planner.policy_context
        self.assertEqual(len(action_blocks), 7 + 31)
        self.assertTrue(all(block.shape == (4,) for block in action_blocks))
        self.assertEqual(scalars, (0.0, 0.0))
        planner.split(np.full(4, 0.1))
        _, scalars = planner.policy_context
        self.assertEqual(scalars, (1.0 / 7.0, 1.0 / 31.0))

    def test_exhausted_lower_ledger_keeps_actor_diagnostics_finite(self):
        planner = CausalRecedingHorizonResponsibilityPlanner(
            planning_horizon=8,
            lower_rms_budget=1e-5,
            coordinate_sweeps=16,
            multiplier_bisection_steps=6,
        )
        planner.reset(1)
        rows = [
            planner.split(np.asarray([0.95 if index % 2 else -0.95]))
            for index in range(40)
        ]
        self.assertTrue(any(
            row["lower_budget_power_forecast"] == 0.0 for row in rows
        ))
        self.assertTrue(all(np.isfinite(
            row["actor_floor_ratio_excess_squared"]
        ) for row in rows))

    def test_invalid_configuration_and_unreset_use_fail_closed(self):
        with self.assertRaises(ValueError):
            CausalRecedingHorizonResponsibilityPlanner(
                forecast_mode="future_truth"
            )
        planner = CausalRecedingHorizonResponsibilityPlanner()
        with self.assertRaises(RuntimeError):
            planner.split(np.zeros(1))
        planner.reset(1)
        with self.assertRaises(ValueError):
            planner.split(np.asarray([3.0]))


if __name__ == "__main__":
    unittest.main()
