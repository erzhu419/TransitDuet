import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_freqduet_v136_handoff_propagation import (
    direct_accounting, delay_accounting, pooled_profile, pattern_groups, variance_totals)


class HandoffPropagationTests(unittest.TestCase):
    def action_path(self, actions, traces=()):
        return dict(actions={k: dict(action_s=a, arrival_s=100., release_s=0.,
            incoming_holding_s=0., incoming_other_s=100.) for k, a in actions.items()},
            traces=traces, terminal={}, events=dict(arrivals=[], departures=[]))

    def test_same_state_override_is_not_old_new_path_difference(self):
        key, other = (1, 2, True), (1, 3, True)
        trace = dict(trip_id=1, station_id=2, direction=True, chosen_s=0., executed_s=0.,
            raw_shadow_s=30., policy_time_s=101., component_used=True,
            online=dict(status='launch_handoff_component', gap_s=200.), evidence=dict(eta_prediction_s=600.))
        base = self.action_path({key: 20., other: 0.})
        candidate = self.action_path({key: 0., other: 5.}, [trace])
        result, _ = direct_accounting(base, candidate)
        self.assertEqual(result['direct_override_s'], -30.)
        self.assertEqual(result['realized_on_handoff_keys_s'], -20.)
        self.assertEqual(result['realized_off_handoff_keys_s'], 5.)
        self.assertEqual(result['realized_holding_delta_s'], -15.)
        self.assertEqual(result['continuation_accounting_residual_s'], 15.)

    def test_no_direct_action_does_not_silently_allow_unexplained_changes(self):
        base = self.action_path({(1, 2, True): 0.})
        candidate = self.action_path({(1, 2, True): 5.})
        with self.assertRaisesRegex(AssertionError, 'without a recorded direct'):
            direct_accounting(base, candidate)

    def test_delay_uses_both_recorded_reference_clocks(self):
        def path(reference):
            return dict(dispatch={1: dict(trip_id=1, bus_id=1, prior_trip_id=-1,
                actual_s=200., eligible_s=reference, first_dispatch_visible_s=200.)},
                clocks={1: reference}, held={1: 0.},
                returns={1: dict(return_s=400., nonholding_elapsed_s=200.)})
        result, _ = delay_accounting(path(100.), path(110.))
        self.assertEqual(result['delay_unmatched_supply_s'], -10.)
        self.assertEqual(result['actual_shift_s'], 0.)
        self.assertEqual(result['reference_shift_s'], 10.)

    def test_pooled_profile_does_not_replace_pooled_cv_by_local_cv(self):
        def path(mean, variance, h):
            return dict(series=dict(arrival=dict(samples=2, mean=mean, variance=variance,
                cv_squared=variance/mean**2, edges={
                    (1, 2, True): dict(headway=h[0]), (2, 3, True): dict(headway=h[1])})))
        result = pooled_profile(path(10., 4., (8., 12.)), path(10., 9., (7., 13.)), lambda k: k[1])
        self.assertAlmostEqual(sum(result.values()), .05)
        self.assertAlmostEqual(result['2'], .025)

    def test_joint_pattern_keeps_v135_passenger_gain_and_other_nonharm_definition(self):
        row = dict(effects=dict(passenger_person_s=-1., cv_squared=0.,
            inclusive_terminal_delay_s=0., trips_unlaunched=0., trips_incomplete=0.))
        self.assertEqual(pattern_groups([row])['joint_improvement'], [row])

    def test_lower_component_variance_can_be_outweighed_by_cross_terms(self):
        result = variance_totals(dict(release_diagonal=-.02,
            release__realized_holding_cross=.03, mean_normalization=0., cv_squared=.01))
        self.assertEqual(result['diagonal_sum'], -.02)
        self.assertEqual(result['cross_sum'], .03)


if __name__ == '__main__':
    unittest.main()
