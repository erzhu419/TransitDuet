import unittest
from pathlib import Path

from env.sim import env_bus


class ProtocolV4PhysicalContractsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env_path = str(Path(__file__).resolve().parents[1] / "env")

    def _env(self, **overrides):
        config = {
            "effective_trip_num": 8,
            "fleet_inventory_mode": "fixed_pool",
            "fixed_pool_initial_up": 2,
            "evaluation_end_time_s": 1200,
            "demand_end_time_s": 600,
        }
        config.update(overrides)
        env = env_bus(self.env_path, env_config=config)
        env.enable_plot = False
        env._n_fleet_target = 4
        env.reset()
        return env

    def test_agent_slots_do_not_scale_with_timetable_trip_count(self):
        env = self._env(effective_trip_num=80)
        self.assertEqual(env.max_agent_num, 30)

    def test_fixed_pool_has_exact_inventory_and_never_creates_a_bus(self):
        env = self._env()
        env._initialize_fixed_pool()
        self.assertEqual(len(env.bus_all), 4)
        self.assertEqual(env._ready_vehicle_count(True), 2)
        self.assertEqual(env._ready_vehicle_count(False), 2)

        trip = next(tt for tt in env.timetables if bool(tt.direction))
        bus = env.launch_bus(trip, actual_launch=5.0)
        self.assertIsNotNone(bus)
        self.assertEqual(len(env.bus_all), 4)
        self.assertEqual(sum(item.on_route for item in env.bus_all), 1)

    def test_fixed_pool_denies_dispatch_when_correct_terminal_is_empty(self):
        env = self._env()
        env._initialize_fixed_pool()
        for bus in env.bus_all:
            if bus.direction:
                bus.on_route = True
        trip = next(tt for tt in env.timetables if bool(tt.direction))

        self.assertIsNone(env.launch_bus(trip, actual_launch=10.0))
        self.assertEqual(len(env.bus_all), 4)

    def test_upper_fleet_state_uses_physical_capacity_not_agent_slots(self):
        env = self._env()
        env._initialize_fixed_pool()
        for bus in env.bus_all[:2]:
            bus.on_route = True
        trip = env.timetables[0]

        state = env._build_upper_state(trip)

        self.assertAlmostEqual(float(state[2]), 0.5)

    def test_v4_upper_state_exposes_terminal_readiness(self):
        env = self._env(upper_fleet_state_mode="fixed_pool_readiness_v4")
        env._initialize_fixed_pool()
        trip = env.timetables[0]
        state = env._build_upper_state_v2(trip)

        self.assertEqual(len(state), env.upper_state_dim)
        self.assertAlmostEqual(float(state[-4]), 0.5)
        self.assertAlmostEqual(float(state[-3]), 0.5)
        self.assertAlmostEqual(float(state[-2]), 0.0)
        self.assertGreaterEqual(float(state[-1]), 0.0)

    def test_v4_readiness_state_requires_fixed_inventory(self):
        with self.assertRaisesRegex(ValueError, "requires fleet_inventory_mode"):
            self._env(
                fleet_inventory_mode="elastic_legacy",
                upper_fleet_state_mode="fixed_pool_readiness_v4",
            )


if __name__ == "__main__":
    unittest.main()
