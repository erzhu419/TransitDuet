import unittest

from lower.causal_follower_eta import (
    AVLVehicleSnapshot,
    estimate_follower_departure_gap,
)


def vehicle(
    bus_id,
    *,
    progress,
    direction=True,
    on_route=True,
    launch_time=0.0,
    current_speed=0.0,
    route_speed=10.0,
):
    return AVLVehicleSnapshot(
        bus_id=bus_id,
        direction=direction,
        on_route=on_route,
        progress_m=progress,
        launch_time_s=launch_time,
        current_speed_mps=current_speed,
        route_speed_mps=route_speed,
    )


class CausalFollowerEtaTest(unittest.TestCase):
    def test_selects_nearest_active_same_direction_physical_follower(self):
        estimate = estimate_follower_departure_gap(
            current_bus_id=1,
            current_direction=True,
            current_progress_m=3000.0,
            current_time_s=600.0,
            service_dwell_proxy_s=20.0,
            vehicles=[
                vehicle(1, progress=3000.0),
                vehicle(2, progress=2500.0),
                vehicle(3, progress=2000.0),
                vehicle(4, progress=2900.0, direction=False),
                vehicle(5, progress=2800.0, on_route=False),
            ],
        )

        self.assertTrue(estimate.valid)
        self.assertEqual(estimate.follower_bus_id, 2)
        self.assertEqual(estimate.spatial_gap_m, 500.0)
        self.assertAlmostEqual(estimate.speed_mps, 2500.0 / 600.0)
        self.assertAlmostEqual(
            estimate.departure_gap_s,
            500.0 / (2500.0 / 600.0) + 20.0,
        )
        self.assertEqual(
            estimate.source, "same_time_avl_journey_speed_eta")

    def test_uses_current_speed_then_route_speed_as_causal_fallbacks(self):
        current = estimate_follower_departure_gap(
            current_bus_id=1,
            current_direction=True,
            current_progress_m=100.0,
            current_time_s=0.0,
            service_dwell_proxy_s=0.0,
            vehicles=[vehicle(2, progress=50.0, current_speed=5.0)],
        )
        route = estimate_follower_departure_gap(
            current_bus_id=1,
            current_direction=True,
            current_progress_m=100.0,
            current_time_s=0.0,
            service_dwell_proxy_s=0.0,
            vehicles=[vehicle(2, progress=50.0, route_speed=12.0)],
        )

        self.assertEqual(current.source, "same_time_avl_current_speed_eta")
        self.assertEqual(route.source, "same_time_avl_route_speed_eta")
        self.assertAlmostEqual(current.eta_s, 10.0)
        self.assertAlmostEqual(route.eta_s, 50.0 / 6.0)

    def test_fails_closed_without_a_physical_follower(self):
        estimate = estimate_follower_departure_gap(
            current_bus_id=1,
            current_direction=True,
            current_progress_m=1000.0,
            current_time_s=600.0,
            service_dwell_proxy_s=20.0,
            vehicles=[
                vehicle(2, progress=1200.0),
                vehicle(3, progress=900.0, direction=False),
            ],
        )

        self.assertFalse(estimate.valid)
        self.assertIsNone(estimate.departure_gap_s)
        self.assertEqual(estimate.source, "no_same_direction_avl_follower")


if __name__ == "__main__":
    unittest.main()
