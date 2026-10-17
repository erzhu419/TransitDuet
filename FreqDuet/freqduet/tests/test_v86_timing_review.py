from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_freqduet_v86_timing_review import paired_timing


def path():
    return dict(target_time_s=100., target_key=[1, 2, False], target_bus=4,
        samples=[dict(time_s=t, waiting_person_s=0., onboard_person_s=0., holding_vehicle_s=0.)
                 for t in (100., 400., 700., 1000.)],
        returns=[dict(trip_id=1, observed_time_s=810., physical_time_s=809.)],
        launches=[dict(trip_id=2, bus_id=4, time_s=900.)])


class TimingReviewTest(unittest.TestCase):
    def test_cost_is_observed_before_return_without_interpolating(self):
        control, recovery = path(), path()
        recovery["samples"][2]["waiting_person_s"] = -10.
        result = paired_timing(control, recovery)
        self.assertTrue(result["passenger_change_observed_before_target_return"])
        self.assertEqual(result["passenger_at_last_pre_return_grid_s"], -10.)
        self.assertEqual(result["first_change_s"]["passenger"], 600.)

    def test_crossing_return_boundary_does_not_imply_pre_return_effect(self):
        control, recovery = path(), path()
        recovery["samples"][-1]["onboard_person_s"] = 10.
        self.assertFalse(paired_timing(control, recovery)["passenger_change_observed_before_target_return"])

    def test_dispatch_is_matched_by_trip_not_bus(self):
        control, recovery = path(), path()
        recovery["launches"][0].update(bus_id=5, time_s=950.)
        recovery["launches"].append(dict(trip_id=3, bus_id=4, time_s=920.))
        result = paired_timing(control, recovery)
        self.assertEqual(result["shifted_matched_dispatches"], 1)
        self.assertEqual(result["first_matched_dispatch_shift_s"], 800.)
        self.assertEqual(result["reassigned_matched_trips"], 1)
        self.assertEqual(result["unmatched_launches"], 1)


if __name__ == "__main__":
    unittest.main()
