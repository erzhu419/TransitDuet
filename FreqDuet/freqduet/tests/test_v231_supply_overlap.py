import unittest

from analyze_freqduet_v231_supply_overlap import compensation, cycle_tally, next_relation, supply_delta


class SupplyOverlapTest(unittest.TestCase):
    def test_pressure_uses_target_direction_and_all_due_categories(self):
        coverage = dict(uncovered={'false': [11], 'true': [12]}, due={'false': [10, 11], 'true': [12]})
        plans = [dict(trip_id=t) for t in (10, 11, 12, 13)]
        self.assertEqual(next_relation(11, coverage, plans, 'false'), 'supplied_query_uncovered')
        self.assertEqual(next_relation(10, coverage, plans, 'false'), 'supplied_query_covered_due')
        self.assertEqual(next_relation(13, coverage, plans, 'false'), 'supplied_future_posted')
        self.assertEqual(next_relation(14, coverage, plans, 'false'), 'not_pending_at_query')
        self.assertEqual(next_relation(None, coverage, plans, 'false'), 'no_later_same_bus_launch')

    def test_local_benefit_does_not_imply_a_nonharm_cut(self):
        f = dict(status='supported', native_bin=2, candidates=[
            dict(holding_s=0, local_gap_cost_delta_s2=600),
            dict(holding_s=5, local_gap_cost_delta_s2=250),
            dict(holding_s=10, local_gap_cost_delta_s2=0)])
        result = compensation(f)
        self.assertEqual(result['proxy'], 'benefit_vs_zero')
        self.assertFalse(result['nonharm_cut'])
        self.assertEqual(result['min_cut_delta_R'], 250)

    def test_unsupported_remains_unknown_not_zero_cost(self):
        f = dict(status='missing_follower', native_bin=0, candidates=[])
        self.assertEqual(compensation(f), dict(proxy='missing_follower',
            nonharm_cut=None, zero_delta_R=None, min_cut_delta_R=None))

    def test_same_bus_one_step_accounting_closes(self):
        b = dict(bus_id=2, next_trip=20, launch_s=100., holding_s=40., other_elapsed_s=60.,
            observed_postreturn_wait_s=5., next_launch_s=205., next_public_delay_s=5., pressure_holding_s=10.)
        c = dict(b, launch_s=101., holding_s=45., other_elapsed_s=58., observed_postreturn_wait_s=6.,
                 next_launch_s=210., next_public_delay_s=10., pressure_holding_s=15.)
        self.assertEqual(supply_delta(c, b)['next_launch_delta_s'], 5.)
        with self.assertRaises(AssertionError):
            supply_delta(dict(c, next_trip=21), b)
        with self.assertRaises(AssertionError):
            supply_delta(dict(c, next_launch_s=209.), b)

    def test_cycle_label_count_is_not_query_count(self):
        c = dict(next_trip=20, holding_s=30., pressure_holding_s=20., matched_pressure_queries=8,
                 next_public_delay_s=12., observed_postreturn_wait_s=4.)
        final = dict(c, next_trip=None, next_public_delay_s=None, observed_postreturn_wait_s=None)
        result = cycle_tally([c, final])
        self.assertEqual(result['cycles'], 2)
        self.assertEqual(result['next_launch_labels'], 1)
        self.assertEqual(result['next_public_delay_s'], 12.)
        self.assertEqual(result['matched_pressure_queries'], 16)


if __name__ == '__main__':
    unittest.main()
