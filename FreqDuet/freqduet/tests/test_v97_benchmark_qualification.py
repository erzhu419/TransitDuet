from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v97_benchmark_qualification as study


class BenchmarkQualificationTest(unittest.TestCase):
    def test_one_fixed_path_per_day_not_per_actor_sampler(self):
        self.assertEqual(len(study.ITEMS), 960)
        self.assertEqual(len(study.DAYS), 32)
        self.assertEqual(study.TOTAL_ITEMS, 992)
        self.assertEqual(study.CONTRASTS["preserve_vs_lower40"], ("preserve", "lower40"))

    def test_fixed_uses_physical_exposure_not_rounded_journey_or_composite(self):
        raw = dict(restricted_total_journey_horizon_min=1.23456789, headway_cv=.25,
            fleet_readiness_delay_mean_s=2.12345678, holding_vehicle_seconds_per_launched_trip=0.,
            trips_unlaunched=0, trips_incomplete=0, holding_passenger_seconds=0., composite=999.)
        exposure = SimpleNamespace(stream=SimpleNamespace(total=np.array([12., 30., 0., 0., 5.])))
        row = study.fixed_metrics(raw, exposure, dict(dispatch_gap_mean_s=360., dispatch_gap_variance_s2=0., dispatch_cv_squared=0.))
        self.assertEqual(row["passenger_person_s"], 42.)
        self.assertEqual(row["journey_min"], 1.234568)
        self.assertEqual(row["cv_squared"], .0625)
        self.assertEqual(set(row), set(study.frozen.METRICS))

    def test_prefix_exposure_preserves_fixed_horizon_censoring_without_actions(self):
        exposure, tracker = study.FixedExposure(), SimpleNamespace(record_step=Mock())
        env = SimpleNamespace(current_time=0., protocol=SimpleNamespace(evaluation_end_time_s=5.),
            _done_reason="service_cleared", _measurement_details=dict(trips_unlaunched=0, trips_incomplete=0, passengers_unserved=3))
        exposure.attach(env, tracker)
        tracker.record_step(dt_s=2., waiting_by_direction={True: 3}, onboard_by_direction={True: 4},
            waiting_low_by_direction={True: 1}, onboard_low_by_direction={True: 2}, dispatch_backlog_by_direction={True: 1})
        audit = exposure.finish(env)
        np.testing.assert_array_equal(exposure.stream.total, [15., 8., 5., 4., 2.])
        self.assertEqual(audit["decisions"], 0)
        self.assertEqual(audit["conservation_max_error"], 0.)

    def test_drift_relief_does_not_erase_residual_to_early_actor(self):
        rows = {mode: dict.fromkeys(study.frozen.METRICS, value) for mode, value in
            (("lower40", 10.), ("late", 12.), ("preserve", 11.), ("fixed", 9.))}
        effects = study.comparisons(rows)
        self.assertEqual(effects["late_vs_lower40"]["journey_min"], 2.)
        self.assertEqual(effects["preserve_vs_lower40"]["journey_min"], 1.)
        self.assertEqual(effects["preserve_vs_fixed"]["journey_min"], 2.)


if __name__ == "__main__":
    unittest.main()
