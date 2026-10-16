from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import freqduet_action_increment as model
import freqduet_fleet_observation as telemetry
import run_freqduet_v73_fleet_state as audit


def environment():
    vehicles = [NS(bus_id=i, direction=bool(i % 2), on_route=i == 1, travel_distance=100.,
        launch_time=100., current_speed=4., trip_id=1 if i == 1 else -1,
        passengers=[object()] * 3, capacity=80., state=NS(name="WAITING_ACTION"),
        back_to_terminal_time=float("-inf"), dwelling_time=25., holding_time=42.) for i in range(3)]
    vehicles[2].trip_id, vehicles[2].back_to_terminal_time = 0, 700.
    plans = [NS(launch_turn=i, direction=bool(i % 2), launch_time=100. + 500 * i,
        target_headway=500., launched=i == 0, _actual_launch_time=110. if i == 0 else 999999.) for i in range(3)]
    plans[1]._freqduet_scheduled_launch, plans[1]._original_launch, plans[1]._delta_t = 610., 600., 20.
    plans[2]._freqduet_terminal_dispatch, plans[2]._freqduet_min_dispatch_headway = True, 250.
    return NS(bus_all=vehicles, timetables=plans, current_time=800., max_agent_num=3,
              routes=[NS(distance=1000.)] * 4, protocol=NS(evaluation_end_time_s=48000.))


class FleetStateTest(unittest.TestCase):
    def test_initial_pool_and_completed_return_are_finite_and_distinct(self):
        snapshot = telemetry.capture_fleet(environment())
        fleet, plans = telemetry.encode_fleet(snapshot)
        rows = fleet.reshape(3, -1)
        self.assertTrue(np.isfinite(fleet).all() and np.isfinite(plans).all())
        self.assertEqual(rows[0, telemetry.FLEET_FIELDS.index("terminal_return_seen")], 0)
        self.assertEqual(rows[2, telemetry.FLEET_FIELDS.index("terminal_return_seen")], 1)
        self.assertEqual(rows[2, telemetry.FLEET_FIELDS.index("terminal_return")], 700 / 48000)

    def test_load_not_destinations_and_no_future_actual_launch_or_route_speed(self):
        env = environment()
        first = telemetry.capture_fleet(env)
        env.bus_all[1].passengers = ["different destination"] * 3
        env.bus_all[1].current_route = NS(speed_limit=9999.)
        env.timetables[1]._actual_launch_time = -9999.
        env.future_arrivals, env.future_od, env.predicted_return = [999], [999], 999
        second = telemetry.capture_fleet(env)
        self.assertEqual(first, second)
        self.assertIsNone(second["plans"][1]["actual_launch_s"])

    def test_exact_grid_has_all_bus_and_trip_slots_without_pooling(self):
        snapshot = telemetry.capture_fleet(environment())
        fleet, plans = telemetry.encode_fleet(snapshot)
        self.assertEqual(fleet.size, 3 * len(telemetry.FLEET_FIELDS))
        self.assertEqual(plans.size, 3 * len(telemetry.PLAN_FIELDS))
        changed = deepcopy(snapshot)
        changed["fleet"][1]["load"] += 1
        changed_fleet, changed_plans = telemetry.encode_fleet(changed)
        self.assertEqual(np.count_nonzero(fleet != changed_fleet), 1)
        np.testing.assert_array_equal(plans, changed_plans)
        changed["plans"].pop()
        with self.assertRaises(ValueError):
            telemetry.encode_fleet(changed)

    def test_command_and_service_countdowns_are_only_active_in_their_states(self):
        env = environment()
        for state, command, service in (("WAITING_ACTION", 0, 0), ("DWELLING", 25, 0), ("HOLDING", 0, 42)):
            env.bus_all[1].state.name = state
            row = telemetry.capture_fleet(env)["fleet"][1]
            self.assertEqual((row["remaining_command_hold_s"], row["remaining_service_s"]), (command, service))

    def test_published_release_and_current_modes_follow_dispatch_semantics(self):
        snapshot = telemetry.capture_fleet(environment())
        self.assertEqual(snapshot["plans"][1]["published_release_s"], 620.)
        self.assertTrue(snapshot["plans"][1]["offset_mode"])
        self.assertEqual(snapshot["plans"][2]["min_gap_s"], 250.)
        self.assertFalse(snapshot["plans"][2]["offset_mode"])

    def test_dual_ridge_matches_primal_objective_and_effective_dimension(self):
        rng = np.random.default_rng(73)
        states, native, days = rng.normal(size=(4, 20)), np.array([0, 1, 3, 6]), np.array([1, 1, 2, 2])
        targets = rng.normal(size=(4, 7, 5))
        fitted = model.fit_model(states, native, targets, days, "state_quadratic")
        x = model.design(states, native, fitted.center, fitted.scale, True)
        mask = np.arange(7)[None, :] != native[:, None]
        weights = np.broadcast_to(model.day_weights(days)[:, None] / 6, mask.shape)[mask]
        x = x[mask]
        gram = x.T @ (weights[:, None] * x)
        expected = np.linalg.solve(gram + model.RIDGE * np.eye(x.shape[1]), x.T @ (weights[:, None] * targets[mask]))
        effective = np.trace(np.linalg.solve(gram + model.RIDGE * np.eye(x.shape[1]), gram))
        np.testing.assert_allclose(fitted.coefficients, expected, rtol=1e-11, atol=1e-11)
        self.assertAlmostEqual(fitted.effective_parameters_per_output, effective, places=11)
        np.testing.assert_array_equal(fitted.predict(states, native)[np.arange(4), native], 0.)

    def test_feature_families_do_not_include_day_case_or_outcomes(self):
        data = dict(states=np.ones((3, 2)), fleet_states=np.ones((3, 4)), plan_states=np.ones((3, 6)),
                    days=np.ones(3) * 700006, cases=np.ones(3), outcomes=np.ones((3, 32, 7, 11)))
        for family, size in zip(audit.FAMILIES, (6, 8, 12)):
            self.assertEqual(audit.inputs(data, family).shape, (3, size))
            np.testing.assert_array_equal(audit.inputs(data, family), 1.)

    def test_primary_gate_also_requires_improvement_over_original_state(self):
        contrasts = {c: {m: {loss: dict(ci95=[-2., -1.]) for loss in ("mae", "mse")}
                          for m in audit.old.PRIMARY} for c in audit.CONTROLS}
        effects = {m: dict(ci95=[-2., -1.]) for m in model.METRICS}
        self.assertTrue(all(audit.gates(contrasts, effects).values()))
        contrasts["state_quadratic"]["passenger_person_s"]["mae"]["ci95"][1] = .1
        self.assertFalse(audit.gates(contrasts, effects)["prediction_gate_passed"])


if __name__ == "__main__":
    unittest.main()
