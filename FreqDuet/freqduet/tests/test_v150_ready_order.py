import unittest

import audit_freqduet_v150_ready_order as audit


def label(trip, bus, now=100.):
    return dict(trip_id=trip, station_id=3, direction=True, bus_id=bus, ready_time_s=now)


def query():
    return dict(trip_id=2, station_id=3, direction=True, prediction_time_s=100.,
        evidence=dict(predicted_follower_trip_id=4, predicted_follower_bus_id=7))


class ReadyOrderTest(unittest.TestCase):
    def test_prior_same_clock_callback_is_observed(self):
        order = audit.ready_order([label(4, 7), label(2, 5)])
        self.assertEqual(audit.classify(query(), order), 'ready_observed_before_query')

    def test_later_same_clock_callback_is_not_yet_observed(self):
        order = audit.ready_order([label(2, 5), label(4, 7)])
        self.assertEqual(audit.classify(query(), order), 'ready_later_same_tick')

    def test_future_ready_is_not_current_input(self):
        order = audit.ready_order([label(2, 5), label(4, 7, 102.)])
        self.assertEqual(audit.classify(query(), order), 'ready_after_query_time')

    def test_lost_archive_order_stops_observability_claim(self):
        with self.assertRaises(AssertionError):
            audit.ready_order([label(4, 7, 102.), label(2, 5)])

    def test_other_physical_vehicle_is_not_a_match(self):
        with self.assertRaises(AssertionError):
            audit.classify(query(), audit.ready_order([label(4, 8), label(2, 5)]))


if __name__ == '__main__':
    unittest.main()
