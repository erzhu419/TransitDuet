import copy
import unittest

import freqduet_v259_selection_bank as bank
from preregister_freqduet_v253_action_supply import CONSTRAINTS
from seal_freqduet_v262_selection_contrasts import lock_context


def rows(native_bin=1):
    plan = bank.action_plan(native_bin, list(range(7)))
    data = []
    for f in range(8):
        paths = {}
        for b in plan['physical']:
            paths[b] = dict(actual_anchor_holding_s=bank.core.ACTIONS[native_bin if b == 'native' else int(b[1:])],
                delta_vs_native={k:0. for k in bank.METRICS},
                holding={k:0. for k in ('direct_executed_delta_s', 'later_same_cycle_delta_s',
                    'other_cycles_delta_s', 'all_cycles_delta_s')},
                supply=dict(status='missing_first_next_launch', return_delta_s=None))
        data.append(dict(unit=bank.selection_unit(0, f), plan=copy.deepcopy(plan), paths=paths))
    return data


def set_delta(data, branch, key, values):
    for r,v in zip(data, values):
        r['paths'][branch]['delta_vs_native'][key] = v


class SelectionTests(unittest.TestCase):
    def choose(self, data):
        return lock_context(0, data, bank, CONSTRAINTS)

    def test_least_passenger_sum_among_jointly_eligible_bins(self):
        d = rows()
        set_delta(d, 'b2', 'passenger_person_s', [-2.]*8)
        set_delta(d, 'b3', 'passenger_person_s', [-3.]*8)
        set_delta(d, 'b2', 'journey_min', [-10.]*8)
        self.assertEqual(self.choose(d)['chosen_branch'], 'b3')

    def test_every_frozen_constraint_rejects_harm(self):
        for k in CONSTRAINTS:
            d = rows()
            set_delta(d, 'b6', 'passenger_person_s', [-100.]*8)
            set_delta(d, 'b6', k, [.001]*8)
            self.assertEqual(self.choose(d)['chosen_branch'], 'native')

    def test_zero_passenger_difference_is_not_improvement(self):
        self.assertEqual(self.choose(rows())['chosen_branch'], 'native')

    def test_means_not_best_future_and_not_all_future_sign_gate(self):
        d = rows()
        set_delta(d, 'b2', 'passenger_person_s', [-100.]+[20.]*7)
        set_delta(d, 'b3', 'passenger_person_s', [-2.]*7+[1.])
        self.assertEqual(self.choose(d)['chosen_branch'], 'b3')

    def test_exact_ties_use_smaller_dose_change_then_bin_order(self):
        d = rows(native_bin=3)
        for branch in ('b0', 'b2', 'b4'):
            set_delta(d, branch, 'passenger_person_s', [-4.]*8)
        self.assertEqual(self.choose(d)['chosen_branch'], 'b2')

    def test_masked_bin_not_added_and_native_alias_not_physical(self):
        d = rows()
        for r in d:
            r['plan'] = bank.action_plan(1, [0,1,2])
            r['paths'] = {b:r['paths'][b] for b in r['plan']['physical']}
        set_delta(d, 'b2', 'passenger_person_s', [-1.]*8)
        result = self.choose(d)
        self.assertEqual(set(result['arms']), {'native', 'b0', 'b2'})
        self.assertEqual(result['chosen_branch'], 'b2')

    def test_missing_future_arm_or_changed_unit_blocks_lock(self):
        for change in ('future', 'arm', 'unit'):
            d = rows()
            if change == 'future': d.pop()
            if change == 'arm': del d[0]['paths']['b2']
            if change == 'unit': d[0]['unit']['phase'] = 'evaluation'
            with self.assertRaises(AssertionError):
                self.choose(d)

    def test_missing_supply_is_not_zero_or_a_selection_filter(self):
        d = rows()
        set_delta(d, 'b2', 'passenger_person_s', [-1.]*8)
        result = self.choose(d)
        self.assertEqual(result['chosen_branch'], 'b2')
        self.assertEqual(result['selected_supply']['first_next_delta_s'], dict(n=0, mean=None))
        self.assertEqual(result['selected_query_due_matches'], 0)


if __name__ == '__main__':
    unittest.main()
