import math
import unittest
from collections import defaultdict
from types import SimpleNamespace

from env.evaluation import compute_wait_metrics
from env.passenger import Passenger
from frequency.demand_frequency import DemandFrequencyTracker
from runner_v3 import TransitDuetV2Runner


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

    def test_runner_consumes_frozen_credit_not_settlement_snapshot(self):
        runner = object.__new__(TransitDuetV2Runner)
        runner.freq_wait_enable = True
        runner.freq_wait_assignment_mode = "frozen_passenger"
        runner.freq_wait_norm_s = 60.0
        runner.freq_wait_clip = 10.0
        runner.freq_wait_lower_weight = 1.0
        runner.freq_wait_lower_board_norm = 10.0
        runner.freq_wait_lower_board_clip = 10.0
        runner.freq_wait_lower_board_credit_weight = 0.0
        runner.freq_wait_lower_board_credit_adaptive = False
        runner.freq_wait_lower_board_credit_absorbed_min = 0.0
        runner.freq_wait_lower_board_credit_absorbed_width = 1.0
        runner.freq_wait_lower_board_credit_min_gate = 0.0
        runner._ep_trip_wait_stats = defaultdict(lambda: defaultdict(float))
        runner._ep_lower_wait_penalties = []
        runner._ep_lower_board_credits = []
        runner._ep_lower_board_credit_gates = []
        runner._ep_lower_wait_net = []
        runner._ep_freq_wait_low_shares = []
        runner._ep_freq_wait_lower_high_shares = []
        runner._ep_freq_wait_boarded_pax = 0

        penalty = runner._record_frequency_wait_credit(
            trip_id=4,
            wait_sum_s=240.0,
            boarded_count=2,
            low_demand=1000.0,
            local_high=0.0,
            lf_wait_sum_s=60.0,
            hf_wait_sum_s=180.0,
            lf_mass=0.5,
            hf_mass=1.5,
        )

        self.assertAlmostEqual(penalty, 1.5)
        self.assertAlmostEqual(
            runner._ep_trip_wait_stats[4]["upper_wait_norm_sum"], 1.0)
        self.assertAlmostEqual(runner._ep_freq_wait_low_shares[-1], 0.25)
        self.assertAlmostEqual(
            runner._ep_freq_wait_lower_high_shares[-1], 0.75)


if __name__ == "__main__":
    unittest.main()
