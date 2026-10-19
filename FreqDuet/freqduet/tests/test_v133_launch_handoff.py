from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_freqduet_v133_launch_handoff as run


class LaunchHandoffTests(unittest.TestCase):
    def prefix(self):
        route = dict(ids=[0, 1, 2], positions=np.array([0., 100., 200.]), index={0: 0, 1: 1, 2: 2})
        p = run.ShadowLaunchETA({True: route})
        p.launch(1, True, 0.)
        p.arrival((1, 1, True), 20., 0.)
        p.ready((1, 1, True), 25.)
        p.departure((1, 1, True), 30., 3.)
        p.arrival((1, 2, True), 60., 3.)
        p.ready((1, 2, True), 66.)
        p.departure((1, 2, True), 68., 3.)
        p.launch(2, True, 80.)
        return p

    def evidence(self, **changes):
        return dict(dict(bus_id=10, ready_time_s=80., follower_gap_prediction_s=140.,
            follower_source=run.SOURCES[0], predicted_follower_trip_id=2,
            predicted_follower_bus_id=20, speed_mps=4., spatial_gap_m=200.), **changes)

    def proposal(self, p, e):
        base = p.predict(e, True, 2)
        return base, p.shadow[run.forecast_key(e, True, 2)]

    def test_observed_same_tick_launch_uses_old_completed_labels_only(self):
        p, e = self.prefix(), self.evidence()
        p.history.add(('running', True, 1), 80., 1000.)
        base, proposal = self.proposal(p, e)
        self.assertFalse(base['component_used'])
        self.assertEqual(base['gap_s'], 140.)
        self.assertTrue(proposal['component_used'])
        self.assertEqual(proposal['gap_s'], 66.)
        self.assertEqual(proposal['latest_completion_s'], 66.)
        self.assertEqual(proposal['launch_observed_s'], 80.)

    def test_first_step_handoff_is_continuous_without_rewriting_journey_prediction(self):
        p = self.prefix()
        _, first = self.proposal(p, self.evidence())
        e = self.evidence(ready_time_s=81., spatial_gap_m=199., speed_mps=1.,
            follower_source=run.model.clock.JOURNEY)
        base, second = self.proposal(p, e)
        self.assertIs(base, second)
        self.assertAlmostEqual(first['gap_s']-second['gap_s'], .2)

    def test_all_nonlaunch_journey_components_stay_identical(self):
        e = self.evidence(ready_time_s=100., spatial_gap_m=150., speed_mps=2.5,
            follower_source=run.model.clock.JOURNEY)
        base, proposal = self.proposal(self.prefix(), e)
        self.assertIs(base, proposal)
        self.assertEqual(proposal['gap_s'], 56.)

    def test_missing_support_retains_raw_gap_not_zero(self):
        p = self.prefix()
        del p.history.groups[('service', True, 1)]
        base, proposal = self.proposal(p, self.evidence())
        self.assertIs(base, proposal)
        self.assertEqual(proposal['gap_s'], 140.)

    def test_unobserved_or_future_launch_is_not_inferred(self):
        for trip in (99, 3):
            p = self.prefix()
            if trip == 3:
                p.launch(3, True, 81.)
            base, proposal = self.proposal(p, self.evidence(predicted_follower_trip_id=trip))
            self.assertIs(base, proposal)

    def test_arrival_boundary_and_later_segments_do_not_extend_eligibility(self):
        p = self.prefix()
        p.arrival((2, 1, True), 100., 0.)
        for progress in (100., 150.):
            e = self.evidence(ready_time_s=100., spatial_gap_m=200.-progress)
            base, proposal = self.proposal(p, e)
            self.assertIs(base, proposal)

    def test_no_follower_does_not_produce_a_component_prediction(self):
        e = self.evidence(follower_gap_prediction_s=None, follower_source='no_same_direction_avl_follower')
        base, proposal = self.proposal(self.prefix(), e)
        self.assertIs(base, proposal)
        self.assertIsNone(proposal['gap_s'])

    def test_future_label_identity_failures_are_not_zero_error(self):
        e = self.evidence()
        row = dict(evidence=e, station_id=2, direction=True)
        result = run.label_same_follower(row, {(2, 2, True): dict(bus_id=999, ready_time_s=150.)})
        self.assertFalse(result['resolved'])
        self.assertEqual(result['reason'], 'bus_identity_mismatch')
        self.assertNotIn('original_error_s', result)


if __name__ == '__main__':
    unittest.main()
