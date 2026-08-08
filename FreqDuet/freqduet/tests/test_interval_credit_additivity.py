import unittest

from upper.interval_credit import UpperIntervalOutcomeTracker


class IntervalCreditAdditivityTest(unittest.TestCase):
    @staticmethod
    def _score_partition(
        *,
        durations,
        waiting,
        onboard,
        backlog,
        fleet,
        headway_events_by_interval,
        passengers_generated=1,
        episode_headway_samples=1,
        episode_duration_s=100.0,
        n_fleet_target=2.0,
    ):
        tracker = UpperIntervalOutcomeTracker(
            enabled=True,
            assignment_mode="additive",
            wait_weight=1.0,
            onboard_weight=1.0,
            dispatch_backlog_weight=1.0,
            headway_weight=1.0,
            fleet_weight=1.0,
            wait_reference_min=1.0,
            onboard_reference_min=1.0,
            dispatch_backlog_reference_trips=1.0,
            headway_reference=1.0,
            fleet_reference=1.0,
            component_clip=4.0,
        )
        outcomes = []
        event_history = []
        time_s = 0.0
        for index, duration_s in enumerate(durations):
            tracker.begin(True, time_s)
            event_history.extend(headway_events_by_interval[index])
            tracker.record_step(
                dt_s=duration_s,
                waiting_by_direction={True: waiting[index], False: 0.0},
                onboard_by_direction={True: onboard[index], False: 0.0},
                dispatch_backlog_by_direction={
                    True: backlog[index], False: 0.0},
                fleet_by_direction={True: fleet[index], False: 0.0},
                n_fleet_target=n_fleet_target,
                headway_events=event_history,
            )
            time_s += duration_s
            outcome = tracker.close(True, time_s)
            outcomes.append(outcome)
        scores = tracker.score_many(
            outcomes,
            passengers_generated=passengers_generated,
            episode_headway_samples=episode_headway_samples,
            episode_duration_s=episode_duration_s,
            n_fleet_target=n_fleet_target,
        )
        totals = {key: 0.0 for key in scores[0]}
        for score in scores:
            for key in totals:
                totals[key] += score[key]
        return totals

    def test_backlog_800_trip_seconds_is_partition_invariant(self):
        one = self._score_partition(
            durations=[100.0],
            waiting=[0.0],
            onboard=[0.0],
            backlog=[8.0],
            fleet=[1.0],
            headway_events_by_interval=[[]],
        )
        two = self._score_partition(
            durations=[50.0, 50.0],
            waiting=[0.0, 0.0],
            onboard=[0.0, 0.0],
            backlog=[8.0, 8.0],
            fleet=[1.0, 1.0],
            headway_events_by_interval=[[], []],
        )

        self.assertEqual(one["dispatch_backlog_cost"], 4.0)
        self.assertEqual(two["dispatch_backlog_cost"], 4.0)
        self.assertEqual(one, two)

    def test_all_additive_components_are_partition_invariant(self):
        events = [
            {
                "direction": True,
                "headway_s": 720.0,
                "target_headway_s": 360.0,
            }
            for _ in range(4)
        ]
        one = self._score_partition(
            durations=[100.0],
            waiting=[4.8],
            onboard=[4.8],
            backlog=[8.0],
            fleet=[9.0],
            headway_events_by_interval=[events],
            passengers_generated=1,
            episode_headway_samples=4,
        )
        two = self._score_partition(
            durations=[50.0, 50.0],
            waiting=[4.8, 4.8],
            onboard=[4.8, 4.8],
            backlog=[8.0, 8.0],
            fleet=[9.0, 9.0],
            headway_events_by_interval=[events[:2], events[2:]],
            passengers_generated=1,
            episode_headway_samples=4,
        )

        for key in one:
            self.assertAlmostEqual(one[key], two[key], places=12, msg=key)
        self.assertEqual(one["wait_cost"], 4.0)
        self.assertEqual(one["onboard_cost"], 4.0)
        self.assertEqual(one["dispatch_backlog_cost"], 4.0)
        self.assertEqual(one["fleet_cost"], 4.0)
        self.assertEqual(one["headway_cost"], 1.0)
        self.assertEqual(one["reward"], -17.0)

    def test_local_mean_still_clips_each_component(self):
        tracker = UpperIntervalOutcomeTracker(
            enabled=True,
            assignment_mode="local_mean",
            component_clip=4.0,
            local_wait_queue_norm=1.0,
        )
        tracker.begin(True, 0.0)
        tracker.record_step(
            dt_s=10.0,
            waiting_by_direction={True: 10.0, False: 0.0},
            fleet_by_direction={True: 0.0, False: 0.0},
            n_fleet_target=2.0,
            headway_events=[],
        )
        score = tracker.score(
            tracker.close(True, 10.0),
            passengers_generated=1,
            episode_headway_samples=1,
            episode_duration_s=10.0,
            n_fleet_target=2.0,
        )

        self.assertEqual(score["wait_cost"], 4.0)


if __name__ == "__main__":
    unittest.main()
