import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_freqduet_v71_fleet as audit


def reference():
    return dict(identity=(100, 7, 1, 2, True), native_action_s=5,
                launches=[dict(trip_id=1, bus_id=7, time_s=0, scheduled_s=0, readiness_s=0),
                          dict(trip_id=2, bus_id=7, time_s=200, scheduled_s=190, readiness_s=10),
                          dict(trip_id=3, bus_id=8, time_s=220, scheduled_s=200, readiness_s=20)],
                returns=[dict(trip_id=1, physical_time_s=190), dict(trip_id=2, physical_time_s=400)],
                holds=[dict(time_s=100, bus_id=7, trip_id=1, action_s=5),
                       dict(time_s=120, bus_id=7, trip_id=1, action_s=10),
                       dict(time_s=250, bus_id=7, trip_id=2, action_s=20),
                       dict(time_s=100, bus_id=8, trip_id=3, action_s=30)])


class FleetAccountingTest(unittest.TestCase):
    def test_readiness_and_plan_changes_are_not_actual_time_changes(self):
        old, new = reference(), reference()
        new["launches"][1]["readiness_s"] += 5
        new["launches"][2]["scheduled_s"] += 2
        result = audit.fleet_effects(old, new, 5)
        self.assertEqual(result["actual_launch_times_changed"], 0)
        self.assertEqual(result["readiness_only_launch_changes"], 1)
        self.assertEqual(result["published_plan_only_changes"], 1)

    def test_other_commands_partition_excludes_only_the_target_intervention(self):
        old, new = reference(), reference()
        for hold, action in zip(new["holds"], (45, 0, 15, 20)):
            hold["action_s"] = action
        result = audit.fleet_effects(old, new, 45)
        self.assertEqual(result["target_trip_other_command_s"], -10)
        self.assertEqual(result["target_vehicle_later_trip_command_s"], -5)
        self.assertEqual(result["other_vehicle_command_s"], -10)

    def test_next_launch_compares_the_reference_logical_trip(self):
        old, new = reference(), copy.deepcopy(reference())
        new["returns"][0]["physical_time_s"] += 6
        new["launches"][1]["time_s"] -= 2
        new["launches"][1]["bus_id"] = 8
        new["launches"][2]["time_s"] += 3
        result = audit.fleet_effects(old, new, 5)
        self.assertEqual(result["target_return_delta_s"], 6)
        self.assertEqual(result["reference_next_trip_launch_delta_s"], -2)
        self.assertEqual(result["actual_launch_times_changed"], 2)
        self.assertEqual(result["actual_launches_advanced"], 1)
        self.assertEqual(result["actual_launches_delayed"], 1)
        self.assertEqual(result["vehicles_reassigned"], 1)


if __name__ == "__main__":
    unittest.main()
