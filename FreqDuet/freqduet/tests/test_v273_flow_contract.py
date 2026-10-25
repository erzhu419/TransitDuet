from dataclasses import replace
from types import SimpleNamespace as NS
import unittest

import numpy as np

from freqduet_apc_flow_evidence import APCArrival, completed_flow, interval_bounds
from freqduet_native_event_graph import CompletedPrefix
import freqduet_event_graph_design as design


def event(i, boarded, time=None, completed=None):
    t = float(i*60 if time is None else time)
    return APCArrival(i, i, i, False, 1, t, t, t+1 if completed is None else completed,
                      0., 0., float(boarded), 50.)


class FlowContractTests(unittest.TestCase):
    def test_clear_to_clear_identifies_only_the_interval_count(self):
        result = interval_bounds(event(0,10),event(1,20))
        self.assertEqual((result['lower_arrivals'],result['upper_arrivals']), (20.,20.))

    def test_clear_to_full_has_no_finite_upper(self):
        result = interval_bounds(event(0,10),event(1,50))
        self.assertEqual((result['lower_arrivals'],result['upper_arrivals']), (50.,None))

    def test_full_to_clear_boarding_can_include_old_backlog(self):
        result = interval_bounds(event(0,50),event(1,40))
        self.assertEqual((result['lower_arrivals'],result['upper_arrivals']), (0.,40.))
        self.assertFalse(result['point_identified'])

    def test_full_to_full_is_not_an_arrival_measurement(self):
        result = interval_bounds(event(0,50),event(1,50))
        self.assertEqual((result['lower_arrivals'],result['upper_arrivals']), (0.,None))

    def test_clearance_block_recovers_arrivals_across_full_service(self):
        # Hidden example: 80 new arrivals, serve50/leave30; then10 new,serve40.
        result = completed_flow([event(0,10),event(1,50),event(2,40)],121.1)
        block, = result['exact_blocks']
        self.assertEqual((block['arrivals'],block['start_s'],block['end_s']), (90.,0.,120.))
        self.assertEqual(block['interval_average_rate'],.75)
        self.assertEqual(block['max_completed_s'],121.)

    def test_no_free_seat_and_zero_boarding_is_not_zero_demand(self):
        full = replace(event(1,0),incoming=50.)
        result = interval_bounds(event(0,10),full)
        self.assertEqual((result['lower_arrivals'],result['upper_arrivals']), (0.,None))

    def test_incomplete_visit_breaks_chain_instead_of_disappearing(self):
        rows = [event(0,10),replace(event(1,50),completed_s=None),event(2,40),event(3,7)]
        result = completed_flow(rows,181.1)
        self.assertEqual(result['unavailable_visits'],1)
        block, = result['exact_blocks']
        self.assertEqual((block['first_order'],block['last_order'],block['arrivals']), (2,3,7.))

    def test_both_clocks_must_be_strictly_before_query(self):
        for row in (event(1,20,completed=100.),replace(event(1,20),available_s=100.)):
            self.assertEqual(completed_flow([event(0,10),row],100.)['exact_blocks'],[])
        self.assertEqual(len(completed_flow([event(0,10),event(1,20)],100.)['exact_blocks']),1)

    def test_same_clock_has_no_new_arrivals_or_division_by_zero(self):
        rows = [event(0,50,time=0),event(1,40,time=0),event(2,0,time=0)]
        result = completed_flow(rows,2.)
        self.assertTrue(all(r['lower_arrivals'] == r['upper_arrivals'] == 0 for r in result['intervals']))
        block, = result['exact_blocks']
        self.assertEqual((block['arrivals'],block['interval_average_rate']),(0.,None))

    def test_missing_or_ambiguous_arrival_order_is_not_repaired(self):
        for rows in ([event(0,10),event(2,20)], [event(0,10),replace(event(1,20),order=0)]):
            with self.assertRaises(AssertionError):
                completed_flow(rows,200.)

    def test_open_censored_block_remains_lower_bounded(self):
        result = completed_flow([event(0,10),event(1,50),event(2,50)],200.)
        block, = result['open_blocks']
        self.assertEqual((block['lower_arrivals'],block['upper_arrivals']),(100.,None))
        self.assertFalse(result['future_rate_upper_bound_identified'])

    def test_passive_extension_preserves_legacy_visits_and_first_arrival(self):
        prefix = CompletedPrefix({0:dict(time_s=-20.),1:dict(time_s=-20.)})
        for i,t in enumerate((0.,60.)):
            bus = NS(bus_id=i,trip_id=i,last_board_count=10,passengers=[None]*10,capacity=50.)
            prefix.arrival(bus,t,1,False,0)
            prefix.complete(i,i,1,False,t+1)
            prefix.complete(i,i,1,False,t+1)
        self.assertEqual((len(prefix.visits),prefix.first_visits,len(prefix.flow_events)),(1,1,2))
        self.assertEqual(len(completed_flow(prefix.flow_events,62.)['exact_blocks']),1)

    def test_unexecuted_downstream_compensation_can_destroy_regularity(self):
        spec = design.fixture('alighting')
        full = np.array([20,10,0,20,20,20])
        ok, _, ref = design.admissible(full,spec,matched_amount=True)
        self.assertTrue(ok[0])
        first_only = spec['baseline'].copy()
        first_only[0] = full[0]
        ok, actual, _ = design.admissible(first_only,spec,matched_amount=False)
        self.assertFalse(ok[0])
        self.assertGreater(actual['model_cv_squared'][0],ref['model_cv_squared'][0])

    def test_promised_future_cut_is_not_executed_return_time_credit(self):
        native, planned, dropped_future = (10,20), (20,10), (20,20)
        launch = lambda h:max(81,50+sum(h)+1)
        self.assertEqual(launch(native),launch(planned))
        self.assertEqual(launch(dropped_future)-launch(native),10)


if __name__ == '__main__':
    unittest.main()
