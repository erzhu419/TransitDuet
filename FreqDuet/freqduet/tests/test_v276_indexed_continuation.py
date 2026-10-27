from dataclasses import replace
import unittest

import freqduet_causal_event_graph as graph
import freqduet_flow_controller as flow
from freqduet_indexed_continuation import IndexedContinuation, IndexedNativeGraph, ReusedNativeRule
from freqduet_native_continuation import native_request
from test_v276_native_continuation import model_fixture, packet


class IndexedContinuationTests(unittest.TestCase):
    def models(self):
        _,reference=model_fixture()
        return reference,IndexedNativeGraph(reference,reference.observed)

    def test_all_seven_current_actions_and_two_stop_transfers_match_exactly(self):
        reference,indexed=self.models()
        for first in graph.ACTIONS:
            for second in graph.ACTIONS:
                commands={(0,0,0):int(first),(0,0,1):int(second)}
                self.assertEqual(reference.evaluate(commands),indexed.evaluate(commands),commands)

    def test_reused_rule_keeps_all_evidence_fields_and_clears_previous_state(self):
        rule=ReusedNativeRule()
        p=packet()
        cases=(p,dict(p,departure=None),dict(p,predecessor=None),dict(p,avl=()),
               dict(p,departure=dict(trip_id=123,time_s=1700.)),p)
        for case in cases:
            for legal in (tuple(graph.ACTIONS),(0,30),(5,10,20,45)):
                self.assertEqual(rule(case,legal),native_request(case,legal))

    def test_full_solver_and_fallback_are_exact_with_same_evaluation_budget(self):
        reference,indexed=self.models()
        self.assertEqual(flow.choose(reference),flow.choose(indexed))

    def test_first_search_audit_checks_every_kernel_without_changing_search(self):
        from run_freqduet_v276_indexed_replacement import CheckedIndexedGraph
        reference,_=self.models()
        checked=CheckedIndexedGraph(reference)
        actual=flow.choose(checked)
        self.assertEqual(actual,flow.choose(reference))
        self.assertEqual(checked.matches,actual['evaluations'])

    def test_replacement_aggregation_retains_the_frozen_strict_and_nonharm_gates(self):
        from run_freqduet_v276_indexed_replacement import paired_effects, frozen
        cube={(day,arm):{m:0. if m.startswith('trips_') else 10. for m in frozen.METRICS}
              for day,arm in frozen.ITEMS}
        for day in frozen.DAYS:
            for m in ('passenger_person_s','cv_squared','readiness_delay_s','public_launch_delay_total_s'):
                cube[day,frozen.ARMS[2]][m]=9.
        self.assertTrue(all(x['joint_gate'] for x in paired_effects(cube).values()))
        cube[frozen.DAYS[0],frozen.ARMS[2]]['trips_unlaunched']=1.
        self.assertTrue(all(not x['joint_gate'] for x in paired_effects(cube).values()))
        cube[frozen.DAYS[0],frozen.ARMS[2]]['trips_unlaunched']=0.
        for day in frozen.DAYS:
            cube[day,frozen.ARMS[2]]['passenger_person_s']=10.
        result=paired_effects(cube)
        self.assertFalse(result['candidate_vs_'+frozen.ARMS[0]]['joint_gate'])
        self.assertTrue(result['candidate_vs_'+frozen.ARMS[1]]['joint_gate'])

    def test_homogeneous_cost_and_existing_committed_hold_are_unchanged(self):
        reference,indexed=self.models()
        fleet=tuple(replace(v,phase='hold',remaining_s=12.,arrival_s=1990.,boarded=0.)
            if v.bus==2 else v for v in reference.snapshot.vehicles)
        reference.snapshot=replace(reference.snapshot,vehicles=fleet)
        indexed=IndexedNativeGraph(reference,reference.observed)
        for homogeneous in (False,True):
            self.assertEqual(reference.evaluate(homogeneous=homogeneous),indexed.evaluate(homogeneous=homogeneous))

    def test_coordinates_keep_the_original_summation_order(self):
        reference,indexed=self.models()
        for d,links in reference.observed['links'].items():
            for i in range(len(links)):
                self.assertEqual(indexed.coordinates[d][i],sum(x for x,_ in links[:i+1]))

    def test_prefix_is_shared_but_candidate_generated_events_are_isolated(self):
        reference,indexed=self.models()
        before={key:tuple(rows) for key,rows in indexed.departure_prefix.items()}
        first=indexed.evaluate({(0,0,0):45})
        indexed.evaluate({(0,0,0):0})
        self.assertEqual(first,indexed.evaluate({(0,0,0):45}))
        self.assertEqual(before,indexed.departure_prefix)
        self.assertEqual(reference.evaluate({(0,0,0):45}),first)

    def test_inline_clock_identity_and_original_first_tie_are_preserved(self):
        reference,_=self.models()
        records=[(False,0,90,1990.,2),(False,0,91,1999.,2),
                 (False,0,92,1999.,0),(False,0,93,1999.,2)]
        reference.observed=dict(reference.observed,departures=records)
        indexed=IndexedNativeGraph(reference,reference.observed)
        state=IndexedContinuation(indexed)
        for prepare in (1980.,1990.,1990.5,1999.,2000.):
            for reader in indexed.originals:
                valid=[r for r in records if r[3]<prepare or r[3]==prepare
                       and indexed.fleet_rank[r[4]]<indexed.fleet_rank[reader]]
                row=max(valid,key=lambda r:(r[3],indexed.fleet_rank[r[4]])) if valid else None
                expected=None if row is None else dict(trip_id=row[2],time_s=row[3])
                self.assertEqual(state.departure_at((False,0),prepare,reader),expected)

    def test_new_departures_use_the_same_clock_cutoff_as_saved_prefix(self):
        reference,indexed=self.models()
        state=IndexedContinuation(indexed)
        key=False,0
        state.future_departures[key]=[(2001.,indexed.fleet_rank[0],-10,100,0),
                                      (2001.,indexed.fleet_rank[2],-11,101,2)]
        self.assertEqual(state.departure_at(key,2001.,1),dict(trip_id=100,time_s=2001.))
        self.assertEqual(state.departure_at(key,2002.,0),dict(trip_id=101,time_s=2001.))
        self.assertIsNone(state.departure_at((True,999),2002.,0))


if __name__=='__main__':
    unittest.main()
