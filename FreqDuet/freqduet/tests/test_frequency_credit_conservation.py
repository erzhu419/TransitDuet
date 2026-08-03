import math
import unittest
from types import SimpleNamespace

from env.evaluation import compute_wait_metrics
from env.passenger import Passenger
from frequency.demand_frequency import DemandFrequencyTracker


class FrequencyCreditConservationTest(unittest.TestCase):
    def test_arrival_shares_use_preupdate_historical_expectation(self):
        prior = {
            "local": {
                (3, True): [math.log1p(6.0), 0.0],
            }
        }
        tracker = DemandFrequencyTracker(
            update_interval_s=60.0,
            bin_sec=60.0,
            method="harmonic",
            fourier_k=0,
            harmonic_prior=prior,
        )

        low, high = tracker.causal_arrival_band_shares(
            3, True, observed_count=10, observation_interval_s=60.0)

        self.assertAlmostEqual(low, 0.6)
        self.assertAlmostEqual(high, 0.4)
        self.assertEqual(tracker.total_updates, 0)
        tracker.update({(3, True): 100})
        self.assertAlmostEqual(low + high, 1.0)

    def test_wait_and_journey_bands_are_exactly_conserved(self):
        station = SimpleNamespace(station_name="A")
        destination = SimpleNamespace(station_name="B")
        first = Passenger(0.0, station, destination)
        first.set_frequency_shares(0.25, 0.75, "test")
        first.boarding_time = 120.0
        first.boarded = True
        first.arrive_time = 300.0
        first.arrived = True
        second = Passenger(60.0, station, destination)
        second.set_frequency_shares(0.5, 0.5, "test")

        metrics = compute_wait_metrics(
            [SimpleNamespace(total_passenger=[first, second])],
            censor_time_s=360.0,
        )

        self.assertAlmostEqual(
            metrics["avg_wait_lf_observed_min"]
            + metrics["avg_wait_hf_observed_min"],
            metrics["avg_wait_observed_min"],
        )
        self.assertAlmostEqual(
            metrics["restricted_wait_lf_horizon_min"]
            + metrics["restricted_wait_hf_horizon_min"],
            metrics["restricted_wait_horizon_min"],
        )
        self.assertAlmostEqual(
            metrics["frequency_lf_passenger_mass"]
            + metrics["frequency_hf_passenger_mass"],
            metrics["passengers_generated"],
        )
        self.assertAlmostEqual(metrics["frequency_share_max_error"], 0.0)
        self.assertAlmostEqual(metrics["avg_in_vehicle_observed_min"], 3.0)
        self.assertAlmostEqual(metrics["avg_total_journey_observed_min"], 5.0)
        self.assertAlmostEqual(
            metrics["restricted_total_journey_horizon_min"], 5.0)


if __name__ == "__main__":
    unittest.main()
