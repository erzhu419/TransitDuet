import copy
import unittest

from preregister_freqduet_v246_compensation_supply import joint_gate
from preregister_freqduet_v253_action_supply import (
    CONSTRAINTS, check_future_split, holding_differences, registration, supply_differences)


def cycle():
    return dict(cycle_key=[2, 22, 3], launch_s=100., H_s=60., other_elapsed_s=840.,
        return_s=1000., next_trip=30, next_launch_s=1010., next_public_s=980.)


class ActionSupplyTests(unittest.TestCase):
    def test_direct_later_other_H_close_with_rebound(self):
        r = dict(anchor_executed_s=20., anchor_cycle_H_s=100., all_cycles_H_s=1000.)
        a = dict(anchor_executed_s=5., anchor_cycle_H_s=95., all_cycles_H_s=1005.)
        d = holding_differences(r, a)
        self.assertEqual((d['direct_executed_delta_s'], d['later_same_cycle_delta_s'],
                          d['other_cycles_delta_s'], d['all_cycles_delta_s']), (-15., 10., 10., 5.))
        self.assertTrue(d['holding_rebound_observed'])

    def test_zero_pulse_is_no_effect_not_a_rebound(self):
        r = dict(anchor_executed_s=20., anchor_cycle_H_s=100., all_cycles_H_s=1000.)
        d = holding_differences(r, r)
        self.assertEqual(d['all_cycles_delta_s'], 0.)
        self.assertFalse(d['holding_rebound_observed'])

    def test_return_gain_can_be_absorbed_by_waiting(self):
        r, a = cycle(), cycle()
        a.update(H_s=50., return_s=990.)
        d = supply_differences(r, a, 1., dict(trip_id=30, kind='due'))
        self.assertEqual((d['return_delta_s'], d['return_to_next_wait_delta_s'], d['first_next_delta_s']),
                         (-10., 10., 0.))
        self.assertEqual(d['public_delay_delta_s'], 0.)

    def test_actual_same_service_supply_gain_closes(self):
        r, a = cycle(), cycle()
        a.update(H_s=50., return_s=990., next_launch_s=1000.)
        d = supply_differences(r, a, 1., dict(trip_id=30, kind='due'))
        self.assertEqual(d['first_next_delta_s'], -10.)
        self.assertEqual(d['public_delay_delta_s'], -10.)
        self.assertTrue(d['serves_query_obligation'] and d['query_due_obligation_match'])

    def test_future_watch_is_not_due_criterion(self):
        d = supply_differences(cycle(), cycle(), 1., dict(trip_id=30, kind='future'))
        self.assertTrue(d['serves_query_obligation'])
        self.assertFalse(d['query_due_obligation_match'])

    def test_changed_first_next_does_not_search_later_matching_trip(self):
        r, a = cycle(), cycle()
        a['next_trip'] = 31
        d = supply_differences(r, a, 1., dict(trip_id=30, kind='due'))
        self.assertEqual(d['status'], 'changed_first_next_obligation')
        self.assertIsNone(d['first_next_delta_s'])
        self.assertIsNone(d['public_delay_delta_s'])

    def test_missing_and_incomplete_stay_unknown(self):
        for field, status in (('next_trip', 'missing_first_next_launch'),
                              ('return_s', 'incomplete_anchor_cycle')):
            a = cycle()
            a[field] = None
            d = supply_differences(cycle(), a, 1., None)
            self.assertEqual(d['status'], status)
            self.assertIsNone(d['first_next_delta_s'])

    def test_physical_account_is_not_mutated(self):
        r, a = cycle(), cycle()
        before = copy.deepcopy((r, a))
        supply_differences(r, a, 1., None)
        self.assertEqual((r, a), before)

    def test_future_root_isolation_not_just_role_labels(self):
        self.assertEqual(check_future_split(list(range(8)), list(range(8, 16)))['evaluation_halves'], [4, 4])
        for bank in (list(range(7))+[0], list(range(7, 15))):
            with self.assertRaises(AssertionError):
                check_future_split(list(range(8)), bank)

    def test_passenger_gain_cannot_override_actual_CV_or_public_delay_harm(self):
        rows = {k: dict(ci95=[-1., 0.]) for k in CONSTRAINTS}
        rows['passenger_person_s'] = dict(ci95=[-2., -1.])
        self.assertTrue(joint_gate(rows))
        for k in ('cv_squared', 'public_launch_delay_total_s'):
            bad = copy.deepcopy(rows)
            bad[k]['ci95'][1] = .01
            self.assertFalse(joint_gate(bad))

    def test_protocol_reserves_no_simulation_or_selector(self):
        p = registration('test')
        self.assertEqual(set(p['current_budget'].values()), {0})
        self.assertEqual(p['frozen']['holding_bins_s'], [0, 5, 10, 15, 20, 30, 45])
        self.assertEqual(p['v238_source'], 'b4b1d00203')
        for key in ('runtime_connected', 'selector_defined', 'offline_choices_made',
                    'collection_authorized', 'replay_authorized', 'training_authorized',
                    'future_manifest_reserved', 'main_policy_changed', 'v238_changed'):
            self.assertFalse(p[key])
        self.assertIn('MUST be identical', p['blind_controls']['atomic_sham'])


if __name__ == '__main__':
    unittest.main()
