import unittest

from lower.causal_follower_eta import (
    AVLVehicleSnapshot,
    HistoricalFollowerTargetCalibrator,
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


class HistoricalFollowerTargetCalibratorTest(unittest.TestCase):
    @staticmethod
    def prediction(calibrator):
        return calibrator.calibrate(
            base_departure_gap_s=200.0,
            forward_departure_gap_s=100.0,
            target_headway_s=180.0,
            action_cap_s=60.0,
            eta_s=180.0,
            spatial_gap_m=900.0,
            speed_mps=5.0,
            service_dwell_s=20.0,
            route_progress=0.4,
            station_phase=0.5,
            current_time_s=7200.0,
            direction=True,
            source="same_time_avl_journey_speed_eta",
        )

    @staticmethod
    def rows(prediction, residual, count=2):
        return [
            {
                "calibration_features": prediction.features,
                "base_predicted_target_action_s": (
                    prediction.base_target_action_s),
                "realized_target_action_s": (
                    prediction.base_target_action_s + residual),
            }
            for _ in range(count)
        ]

    def test_disabled_calibration_is_exact_identity(self):
        calibrator = HistoricalFollowerTargetCalibrator(enabled=False)
        result = self.prediction(calibrator)

        self.assertFalse(result.active)
        self.assertEqual(result.base_departure_gap_s, 200.0)
        self.assertEqual(result.calibrated_departure_gap_s, 200.0)
        self.assertEqual(result.requested_adjustment_s, 0.0)
        self.assertEqual(result.effective_target_adjustment_s, 0.0)

    def test_completed_days_activate_bias_without_same_day_leakage(self):
        calibrator = HistoricalFollowerTargetCalibrator(
            enabled=True,
            mode="historical_target_bias_v1",
            min_history_episodes=2,
            min_samples_per_episode=2,
            history_alpha=1.0,
            ridge=0.0,
            adjustment_cap_s=10.0,
        )
        initial = self.prediction(calibrator)
        self.assertFalse(initial.active)

        first_update = calibrator.update_episode(
            self.rows(initial, residual=-8.0), episode=0)
        self.assertEqual(first_update["episode_updated"], 1)
        self.assertFalse(self.prediction(calibrator).active)

        second_input = self.prediction(calibrator)
        calibrator.update_episode(
            self.rows(second_input, residual=-8.0), episode=1)
        active = self.prediction(calibrator)
        self.assertTrue(active.active)
        self.assertAlmostEqual(active.requested_adjustment_s, -8.0)
        self.assertAlmostEqual(active.calibrated_departure_gap_s, 184.0)
        self.assertAlmostEqual(active.calibrated_target_action_s, 42.0)

    def test_context_features_are_finite_and_adjustment_is_bounded(self):
        calibrator = HistoricalFollowerTargetCalibrator(
            enabled=True,
            mode="historical_target_ridge_v1",
            min_history_episodes=1,
            min_samples_per_episode=1,
            adjustment_cap_s=3.0,
        )
        initial = self.prediction(calibrator)
        self.assertEqual(
            len(initial.features), len(calibrator.feature_names))
        calibrator.update_episode(
            self.rows(initial, residual=30.0, count=1), episode=0)
        active = self.prediction(calibrator)
        self.assertTrue(active.active)
        self.assertLessEqual(abs(active.requested_adjustment_s), 3.0)
        self.assertLessEqual(
            abs(active.calibrated_departure_gap_s - 200.0), 6.0)

    def test_state_round_trip_preserves_prediction_and_update_order(self):
        source = HistoricalFollowerTargetCalibrator(
            enabled=True,
            mode="historical_target_bias_v1",
            min_history_episodes=1,
            min_samples_per_episode=1,
        )
        initial = self.prediction(source)
        source.update_episode(
            self.rows(initial, residual=-4.0, count=1), episode=3)

        restored = HistoricalFollowerTargetCalibrator(
            enabled=True,
            mode="historical_target_bias_v1",
            min_history_episodes=1,
            min_samples_per_episode=1,
        )
        restored.load_state_dict(source.state_dict())
        self.assertEqual(
            self.prediction(source), self.prediction(restored))
        with self.assertRaisesRegex(ValueError, "monotonically"):
            restored.update_episode(
                self.rows(initial, residual=-4.0, count=1), episode=3)


if __name__ == "__main__":
    unittest.main()
