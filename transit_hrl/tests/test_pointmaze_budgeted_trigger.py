import unittest
from collections import Counter

from freq_hrl.domains.mujoco import POINTMAZE_REGIME_SPEEDS, PointMazeRegimeDriver
from freq_hrl.experiments.pointmaze_budgeted_trigger import (
    balanced_jitter_schedule,
)
from freq_hrl.experiments.pointmaze_plan_validity_branching import (
    BRANCH_CATEGORIES,
    plan_renewal_opportunities,
)
from freq_hrl.experiments.pointmaze_plan_value_qualification import (
    DEFAULT_DISTRACTOR_AMPLITUDE,
    DEFAULT_DISTRACTOR_DWELL_SECONDS,
    DEFAULT_FORCE_PULSE_AMPLITUDE,
    DEFAULT_FORCE_PULSE_DURATION_SECONDS,
    DEFAULT_FORCE_PULSE_GAP_SECONDS,
    DEFAULT_REGIME_DWELL_SECONDS,
)


class PointMazeBudgetedTriggerTest(unittest.TestCase):
    def test_jitter_preserves_one_call_per_bin_and_option_bounds(self):
        for seed in (2159201, 2159211, 2160201):
            schedule = balanced_jitter_schedule(
                seed=seed, horizon=300, period_steps=50,
                max_offset_steps=25,
            )
            self.assertEqual(len(schedule), 6)
            self.assertEqual(schedule[0], 0)
            self.assertEqual(
                schedule,
                balanced_jitter_schedule(
                    seed=seed, horizon=300, period_steps=50,
                    max_offset_steps=25,
                ),
            )
            self.assertTrue(all(
                50 * index <= step <= 50 * index + 25
                for index, step in enumerate(schedule)
            ))
            self.assertTrue(all(
                25 <= right - left <= 75
                for left, right in zip(schedule, (*schedule[1:], 300))
            ))

    def test_branch_fit_opportunities_exclude_variable_plan_calls(self):
        seed = 2159201
        horizon = 300
        driver = PointMazeRegimeDriver(
            seed=seed,
            horizon=horizon,
            dt_seconds=0.01,
            regime_dwell_seconds=DEFAULT_REGIME_DWELL_SECONDS,
            target_speed_modes=POINTMAZE_REGIME_SPEEDS,
            force_pulse_amplitude=DEFAULT_FORCE_PULSE_AMPLITUDE,
            force_pulse_duration_seconds=DEFAULT_FORCE_PULSE_DURATION_SECONDS,
            force_pulse_gap_seconds=DEFAULT_FORCE_PULSE_GAP_SECONDS,
            distractor_amplitude=DEFAULT_DISTRACTOR_AMPLITUDE,
            distractor_dwell_seconds=DEFAULT_DISTRACTOR_DWELL_SECONDS,
        )
        schedule = balanced_jitter_schedule(
            seed=seed, horizon=horizon, period_steps=50,
            max_offset_steps=25,
        )
        rows = plan_renewal_opportunities(
            horizon=horizon,
            period_steps=50,
            branch_window_steps=50,
            regime_change_steps=driver.regime_change_steps,
            force_pulse_steps=driver.pulse_start_steps,
            distractor_change_steps=driver.distractor_change_steps,
            seed=seed,
            max_events_per_class=1,
            blocked_steps=schedule,
        )
        self.assertEqual(set(Counter(row.category for row in rows)), set(BRANCH_CATEGORIES))
        self.assertTrue(set(row.step for row in rows).isdisjoint(schedule))


if __name__ == "__main__":
    unittest.main()
