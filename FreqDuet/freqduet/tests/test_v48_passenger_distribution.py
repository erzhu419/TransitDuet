import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freqduet_closed_loop_forecast import fluid_rollout
from freqduet_passenger_distribution import MODES, PassengerProcess, fit_passengers, passenger_ensemble
from freqduet_travel_distribution import fit_travel
from test_v41_closed_loop import observation


def fixture():
    observed = observation()
    observed["time_s"] = 1800
    for stops in observed["flows"].values():
        for stop in stops:
            stop.update(last_observed_s=None, observed_gaps=0)
    model = dict(now_s=1800, dt_s=1, horizon_s=1800, stops={direction: [
        dict(stock=int(stop["queue"]), age_s=0, rate_per_s=stop["rate_per_s"], alight_fraction=stop["alight_fraction"])
        for stop in stops] for direction, stops in observed["flows"].items()})
    return observed, model


class PassengerDistributionTest(unittest.TestCase):
    def test_fit_preserves_prior_and_excludes_future_apc(self):
        observed, _ = fixture()
        observed["flows"]["True"][1].update(queue=5, rate_per_s=.1, last_observed_s=1790)
        apc = pd.DataFrame([dict(direction=True, station_id=1, time_s=t, left_behind=q)
                            for t, q in [(1790, 4), (1800, 100), (1801, 1000)]])
        model = fit_passengers(apc, observed)
        self.assertEqual(model["samples"], 1)
        self.assertEqual(model["latest_time_s"], 1790)
        self.assertEqual(model["stops"]["True"][1]["stock"], 4)
        self.assertEqual(model, fit_passengers(apc.iloc[:1], observed))

    def test_queue_poisson_mean_and_variance_retain_history_mean(self):
        observed, model = fixture()
        model["stops"]["True"][1].update(stock=4, rate_per_s=.1, age_s=100)
        values = []
        for draw in range(2048):
            flows = copy.deepcopy(observed["flows"])
            PassengerProcess(model, "passenger_joint", draw).initialize(flows, {})
            values.append(flows["True"][1]["queue"])
        self.assertAlmostEqual(np.mean(values), 14, delta=.3)
        self.assertAlmostEqual(np.var(values), 10, delta=1)

    def test_arrival_horizons_share_the_same_process_prefix(self):
        _, model = fixture()
        model["stops"]["True"][1]["rate_per_s"] = .1
        short = list(PassengerProcess(model, "arrival_queue", 3).arrival_events(2700))
        long = list(PassengerProcess(model, "arrival_queue", 3).arrival_events(3600))
        self.assertEqual(short, [event for event in long if event[0] <= 2700])

    def test_binomial_alighting_preserves_mean_and_capacity(self):
        _, model = fixture()
        process = PassengerProcess(model, "passenger_joint", 0)
        values = [process.exchange(dict(load=20, capacity=50), dict(queue=100, alight_fraction=.25), False)
                  for _ in range(4096)]
        self.assertAlmostEqual(np.mean([a for a, b in values]), 5, delta=.15)
        self.assertTrue(all(20 - a + b == 50 for a, b in values))

    def test_all_modes_conserve_mass_under_capacity_pressure(self):
        for mode in MODES:
            observed, model = fixture()
            observed["vehicles"][0]["load"] = 49
            model["stops"]["True"][1].update(stock=1000, rate_per_s=.2)
            observed["flows"]["True"][1].update(queue=1000, rate_per_s=.2, alight_fraction=.25)
            result = fluid_rollout(observed, 300, "response", passenger_process=PassengerProcess(model, mode, 0))
            self.assertLessEqual(result["max_load"], 50)
            self.assertAlmostEqual(result["passenger_mass_error"], 0, places=6)
            self.assertEqual(result["vehicles_created"], 0)

    def test_discrete_waiting_integral_counts_events_exactly(self):
        observed, model = fixture()
        observed["vehicles"][0].update(on_route=False, load=0)
        model["stops"]["True"][1]["stock"] = 2
        process = PassengerProcess(model, "passenger_joint", 0)
        process.arrival_events = lambda end: iter([(1805, "True", 1, 3)])
        result = fluid_rollout(observed, 10, "historical", passenger_process=process)
        self.assertEqual(result["waiting_person_seconds"], 35)
        self.assertEqual(result["generated_passengers"], 3)
        self.assertEqual(result["final_queue"], 5)

    def test_capacity_roundoff_is_tolerated_but_a_real_overflow_is_rejected(self):
        for board in (np.nextafter(45.0, np.inf), 46.0):
            observed, model = fixture()
            model["stops"]["True"][1]["stock"] = 1000
            process = PassengerProcess(model, "arrival_queue", 0)
            # A fractional-alighting rollout produced a one-ULP capacity excess.
            process.exchange = lambda vehicle, stop, terminal: (vehicle["load"], 0) if terminal else (0, board)
            if board < 46:
                result = fluid_rollout(observed, 50, "historical", passenger_process=process)
                self.assertGreater(result["max_load"], 50)
                self.assertLess(result["max_load"], 50 + 1e-8)
            else:
                with self.assertRaisesRegex(ValueError, "passenger invariant"):
                    fluid_rollout(observed, 50, "historical", passenger_process=process)

    def test_private_rng_and_input_snapshot_are_unchanged(self):
        observed, model = fixture()
        before_observed = copy.deepcopy(observed)
        legs = pd.DataFrame(columns=["end_s", "start_s", "direction", "station_id", "duration_s"])
        travel = fit_travel(legs, observed)
        holding = dict(coef=None, prior=[1, 0, 0, 0, 0, 0, 0])
        before = np.random.get_state()
        passenger_ensemble(observed, 100, holding, travel, model, "passenger_joint", draws=2)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(observed, before_observed)

    def test_penultimate_mask_is_preserved_with_integer_passengers(self):
        observed, model = fixture()
        observed["routes"]["True"]["holding_mask"][1] = True
        result = fluid_rollout(observed, 100, "response", passenger_process=PassengerProcess(model, "passenger_joint", 0))
        self.assertEqual(result["holding_vehicle_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
