import unittest
from pathlib import Path

from env.sim import env_bus


class SimulationProtocolIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env_path = str(Path(__file__).resolve().parents[1] / "env")

    def _run(self, holding_s):
        env = env_bus(
            self.env_path,
            env_config={
                "effective_trip_num": 4,
                "service_start_hour": 6,
                "service_end_hour": 6,
                "demand_end_time_s": 600,
                "evaluation_end_time_s": 1200,
                "scenario_seed": 991,
            },
        )
        env.enable_plot = False
        env.reset()
        actions = {key: float(holding_s) for key in range(env.max_agent_num)}
        while not env.done:
            env.step(actions)
        return env.measurement_details

    def test_policy_does_not_change_exogenous_demand_realisation(self):
        no_hold = self._run(0.0)
        held = self._run(30.0)
        self.assertEqual(
            no_hold["passengers_generated"],
            held["passengers_generated"],
        )
        self.assertEqual(no_hold["scenario_tape_id"], held["scenario_tape_id"])
        self.assertEqual(no_hold["simulation_end_time_s"], 1200)
        self.assertEqual(held["simulation_end_time_s"], 1200)
        self.assertGreater(no_hold["headway_sample_count"], 0)
        self.assertLessEqual(
            no_hold["trips_completed"], no_hold["trips_launched"])
        self.assertAlmostEqual(
            no_hold["trip_launch_rate"],
            no_hold["trips_launched"]
            / no_hold["timetable_trips_evaluated"],
        )
        self.assertAlmostEqual(
            no_hold["trip_completion_rate"],
            no_hold["trips_completed"]
            / no_hold["timetable_trips_evaluated"],
        )

    def test_all_trip_mode_does_not_apply_legacy_264_cap(self):
        route_env = (
            Path(__file__).resolve().parents[1]
            / "data/external_route_envs/mbta_route_day_v1/_line_envs/104_D0"
        )
        if not route_env.exists():
            self.skipTest("local route-day cache is unavailable")
        env = env_bus(str(route_env), env_config={"effective_trip_num": "all"})
        self.assertGreater(env.effective_trip_num, 264)
        self.assertEqual(env.effective_trip_num, env.total_timetable_rows)

    def test_count_harmonic_alias_receives_historical_od_prior(self):
        env = env_bus(
            self.env_path,
            env_config={
                "effective_trip_num": 4,
                "service_start_hour": 6,
                "service_end_hour": 19,
            },
        )
        env.configure_frequency_features({
            "enable": True,
            "method": "harmonic_nb",
            "bin_sec": 60,
            "use_historical_prior": True,
            "harmonic_period_s": 50400.0,
            "fourier_K": 4,
            "upper_enable": True,
            "lower_enable": True,
        })
        prior = env.frequency_tracker.harmonic_prior
        self.assertIn("global", prior)
        self.assertGreater(len(prior["local"]), 0)
        self.assertTrue((env.frequency_tracker.global_state.theta != 0).any())


if __name__ == "__main__":
    unittest.main()
