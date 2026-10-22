import unittest

import freqduet_v219_service_task as law


def query(ready=2):
    return dict(time_s=7200., event_seq=20, trip_id=40,
        plans=[dict(trip_id=40+2*i, direction=True, public_s=7200.+360*i, planned_s=7200.+360*i)
               for i in range(3)],
        ready=[dict(bus_id=i, direction=True, return_observed=True, last_observed_return_s=7000.+i)
               for i in range(ready)])


class ServiceTaskTest(unittest.TestCase):
    def test_ready_bus_cannot_fund_anchor_and_future_trip(self):
        self.assertEqual(law.ready_prefix(query(1))['mutable_ready_trip_ids'], [])
        r = law.ready_prefix(query(2))
        self.assertEqual(r['mutable_ready_trip_ids'], [42])
        self.assertEqual(r['ready_bus_by_trip'], {40: 0, 42: 1})

    def test_overdue_obligations_consume_stock_before_new_task(self):
        q = query(2)
        q['plans'].insert(0, dict(trip_id=38, direction=True, public_s=6840., planned_s=6840.))
        self.assertEqual(law.ready_prefix(q)['mutable_ready_trip_ids'], [])

    def test_native_projection_admission(self):
        q = query()
        native = {p['trip_id']: p['planned_s'] for p in q['plans']}
        proposed = dict(native)
        proposed[42] -= 5
        self.assertEqual(law.admit_preview(q, native, proposed, 1)['task_ids'], [42])
        proposed[44] += 1
        self.assertEqual(law.admit_preview(q, native, proposed, 1)['reason'], 'later_native_eligibility')
        proposed = dict(native)
        proposed[40] += 1
        self.assertEqual(law.admit_preview(q, native, proposed, 1)['reason'], 'immutable_anchor_changed')

    def test_unfunded_advance_and_roundoff_only(self):
        q = query(1)
        native = {p['trip_id']: p['planned_s'] for p in q['plans']}
        proposed = dict(native)
        proposed[42] -= 5
        self.assertEqual(law.admit_preview(q, native, proposed, 1)['reason'], 'unfunded_advanced_trip')
        proposed = dict(native)
        proposed[42] -= 1e-12
        self.assertFalse(law.admit_preview(q, native, proposed, 1)['admitted'])

    def test_lower_uses_actual_outbound_identity_and_preserves_compensation(self):
        q = dict(trip_id=42, bus_id=1, event_seq=30, base_bin=2, feasible_bins=[0, 1, 2],
                 terminal_masked=False, forward_valid=True, follower_valid=True,
                 context=dict(target_headway_s=360.))
        task = dict(trip_id=42, bus_id=1, launch_event_seq=25)
        self.assertEqual(law.lower_request(q, task, 400.), 1)
        self.assertEqual(law.lower_request(q, task, 300.), 2)
        self.assertEqual(law.lower_request(q, dict(task, bus_id=2), 400.), 2)
        self.assertEqual(law.lower_request(q, task, None), 2)


if __name__ == '__main__':
    unittest.main()
