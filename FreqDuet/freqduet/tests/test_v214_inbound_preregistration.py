import copy
import unittest

import preregister_freqduet_v214_inbound_chain as v


def frame():
    return dict(time_s=7300, direction=0, fresh_upper=True, ready=[],
        plans=[dict(direction=0, launched=False, planned_s=7290)],
        active=[dict(bus_id=i, trip_id=10+i, direction=d, remaining_distance_m=m,
                     controlled_stop_ahead=True) for i, d, m in ((1, 0, 20), (2, 1, 500), (3, 1, 1000), (4, 1, 2000))])


def query():
    return dict(time_s=7350, bus_id=2, trip_id=12, direction=1, terminal_masked=False,
                forward_valid=True, follower_valid=True, base_bin=3, feasible_bins=list(range(7)))


class InboundPreregistrationTests(unittest.TestCase):
    def setUp(self):
        self.anchor = dict(time_s=7300, direction=0, inbound=v.choose_supply(frame()))

    def test_direction_and_current_position_not_future_return(self):
        f = frame()
        f['active'][3]['future_return_s'] = 7310
        self.assertEqual([b['bus_id'] for b in v.choose_supply(f)], [2, 3])

    def test_ready_same_terminal_or_not_due_is_not_supply_blocked(self):
        for field, value in (('ready', [dict(direction=0)]),
                             ('plans', [dict(direction=0, launched=False, planned_s=7400)]),
                             ('fresh_upper', False), ('time_s', 8100)):
            f = frame()
            f[field] = value
            self.assertEqual(v.choose_supply(f), [])
        f = frame()
        f['ready'] = [dict(direction=1)]
        self.assertTrue(v.choose_supply(f))

    def test_nearest_two_without_control_are_not_replaced_hindsight(self):
        f = frame()
        f['active'][1]['controlled_stop_ahead'] = False
        f['active'][2]['controlled_stop_ahead'] = False
        self.assertEqual(v.choose_supply(f), [])

    def test_reduce_one_feasible_native_bin(self):
        q = query()
        self.assertEqual(v.reduction(self.anchor, q, set()), 2)
        q['feasible_bins'] = [0, 1, 3, 6]
        self.assertEqual(v.reduction(self.anchor, q, set()), 1)

    def test_zero_is_no_opportunity_not_zero_to_zero_intervention(self):
        q = query()
        q['base_bin'] = 0
        self.assertIsNone(v.reduction(self.anchor, q, set()))

    def test_trip_identity_window_evidence_and_spent_opportunity(self):
        for key, value in (('trip_id', 99), ('direction', 0), ('time_s', 8200),
                           ('time_s', 7299), ('terminal_masked', True),
                           ('forward_valid', False), ('follower_valid', False)):
            q = query()
            q[key] = value
            self.assertIsNone(v.reduction(self.anchor, q, set()))
        self.assertIsNone(v.reduction(self.anchor, query(), {(2, 12)}))

    def test_future_banks_and_budget(self):
        p = v.registration('test')
        select = {v.future_seed(c, f) for c in range(4) for f in p['selection_futures']}
        evaluate = {v.future_seed(c, f) for c in range(4) for f in p['evaluation_futures']}
        smoke = {2149000000+c for c in range(4)}
        self.assertEqual((len(select), len(evaluate)), (64, 64))
        self.assertFalse(select & evaluate or (select | evaluate) & smoke)
        self.assertEqual(len(p['candidates'])*len(select), p['budget']['selection_max'])
        self.assertEqual(p['new_rollouts'], 0)

    def test_actual_cv_readiness_and_public_total_all_have_to_pass(self):
        stats = {k: dict(ci95=[-2., -1.]) for k in ('passenger_person_s', *v.CONSTRAINTS)}
        self.assertTrue(v.joint_gate(stats))
        for key in stats:
            changed = copy.deepcopy(stats)
            changed[key]['ci95'][1] = .01
            self.assertFalse(v.joint_gate(changed))
        stats['passenger_person_s']['ci95'][1] = 0.
        self.assertFalse(v.joint_gate(stats))


if __name__ == '__main__':
    unittest.main()
