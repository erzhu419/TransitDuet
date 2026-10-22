import unittest

import audit_freqduet_v214_smoke_support as v


def query(time, base, **kwargs):
    return dict(time_s=time, base_bin=base, feasible_bins=list(range(7)), bus_id=1, trip_id=4, direction=0,
        station_id=2, command_s=v.ACTIONS[base], state_captured_s=time-1, terminal_masked=False,
        forward_valid=True, follower_valid=kwargs.get('follower', True))


class SmokeSupportTests(unittest.TestCase):
    def test_zero_invalid_and_outside_window_are_distinct(self):
        rows = [query(110, 0), query(120, 2, follower=False), query(1010, 2)]
        data = dict(result=dict(anchor=dict(time_s=100, direction=1, inbound=[dict(bus_id=1, trip_id=4)]),
                    execution={}), trace=dict(lower_queries=rows))
        counts = v.support(data)['selected'][0]['counts']
        self.assertEqual((counts['zero'], counts['positive'], counts['eligible_positive']), (1, 1, 0))
        self.assertEqual(counts['invalid_follower'], 1)
        self.assertEqual(counts['queries_after_window'], 1)

    def test_same_physical_vehicle_later_trip_is_not_selected_supply_trip(self):
        q = query(110, 2)
        q['trip_id'] = 6
        data = dict(result=dict(anchor=dict(time_s=100, direction=1, inbound=[dict(bus_id=1, trip_id=4)]),
                    execution={}), trace=dict(lower_queries=[q]))
        result = v.support(data)
        self.assertEqual(result['selected'][0]['counts']['queries_in_window'], 0)
        self.assertEqual(result['other_inbound_eligible_positive'], 1)


if __name__ == '__main__':
    unittest.main()
