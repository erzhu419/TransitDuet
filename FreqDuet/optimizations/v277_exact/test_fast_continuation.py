from dataclasses import replace
import unittest

import freqduet_causal_event_graph as graph
import freqduet_flow_controller as flow
from freqduet_arrival_metric_control import ArrivalMetricGraph, choose_current
from freqduet_indexed_continuation import IndexedContinuation, IndexedNativeGraph, ReusedNativeRule
from freqduet_native_continuation import native_request
from lower.causal_follower_eta import estimate_follower_departure_gap
from test_v276_native_continuation import model_fixture, packet

from fast_continuation import CachedNativeRule, FastArrivalMetricGraph, FastNativeGraph, SparseContinuation


def fixture():
    _, native = model_fixture()
    arrivals = list(native.observed['arrivals'])
    for v in native.snapshot.vehicles:
        if v.arrival_s is not None:
            arrivals.append((v.direction, native.snapshot.routes[v.direction][v.index].station,
                             v.trip, v.arrival_s, v.bus))
    native.observed = dict(native.observed, arrivals=sorted(arrivals, key=lambda r:(r[3], r[4])))
    return native


class FastContinuationTests(unittest.TestCase):
    def test_all_seven_bins_and_two_stop_commands_keep_every_result_field(self):
        native = fixture()
        for reference_cls, candidate_cls in ((IndexedNativeGraph, FastNativeGraph),
                                             (ArrivalMetricGraph, FastArrivalMetricGraph)):
            old, new = reference_cls(native, native.observed), candidate_cls(native, native.observed)
            for homogeneous in (False, True):
                for first in graph.ACTIONS:
                    for second in graph.ACTIONS:
                        commands = {(0, 0, 0):int(first), (0, 0, 1):int(second)}
                        self.assertEqual(old.evaluate(commands, homogeneous), new.evaluate(commands, homogeneous))

    def test_full_beam_outputs_ties_warm_start_and_logical_budget_stay_exact(self):
        native = fixture()
        for old_cls, new_cls in ((IndexedNativeGraph, FastNativeGraph),
                                (ArrivalMetricGraph, FastArrivalMetricGraph)):
            for homogeneous in (False, True):
                old, new = old_cls(native, native.observed), new_cls(native, native.observed)
                proposals = {(0, 0, 1):10}
                before = flow.choose(old, homogeneous, proposals)
                self.assertEqual(before, flow.choose(new, homogeneous, proposals))
                self.assertLessEqual(before['evaluations'], 128)
                self.assertGreater(new.native_rule.hits, 0)

    def test_current_only_complete_output_is_exact_without_replacing_beam(self):
        native = fixture()
        old = ArrivalMetricGraph(native, native.observed)
        new = FastArrivalMetricGraph(native, native.observed)
        for homogeneous in (False, True):
            self.assertEqual(choose_current(old, homogeneous), choose_current(new, homogeneous))

    def test_existing_irreversible_holding_and_candidate_state_isolation(self):
        native = fixture()
        native.snapshot = replace(native.snapshot, vehicles=tuple(
            replace(v, phase='hold', remaining_s=12., arrival_s=1990., boarded=0.) if v.bus == 2 else v
            for v in native.snapshot.vehicles))
        old, new = IndexedNativeGraph(native, native.observed), FastNativeGraph(native, native.observed)
        before = new.evaluate({(0, 0, 0):45})
        self.assertEqual(before, old.evaluate({(0, 0, 0):45}))
        new.evaluate({(0, 0, 0):0}, True)
        self.assertEqual(before, new.evaluate({(0, 0, 0):45}))

    def test_native_rule_cache_preserves_all_evidence_and_each_input(self):
        p = packet()
        cases = [p, dict(p, departure=None), dict(p, predecessor=None), dict(p, avl=()),
                 dict(p, departure=dict(trip_id=123, time_s=1700.)),
                 dict(p, prepare_s=2040.), dict(p, service_s=60.), dict(p, target_s=300.),
                 dict(p, direction=True), dict(p, progress_m=1250.),
                 dict(p, avl=(*p['avl'][:1], replace(p['avl'][1], progress_m=750.)))]
        cached = CachedNativeRule(ReusedNativeRule())
        for current in cases + [p]:
            for legal in (tuple(graph.ACTIONS), (0, 30), (5, 10, 20, 45)):
                self.assertEqual(cached(current, legal), native_request(current, legal))
                self.assertEqual(cached(current, legal), native_request(current, legal))
        self.assertGreater(cached.hits, 0)
        self.assertEqual(len(CachedNativeRule(ReusedNativeRule()).values), 0)

    def test_sparse_follower_keeps_inline_clocks_identity_and_speed_fallbacks(self):
        native = fixture()
        model = FastNativeGraph(native, native.observed)
        state = SparseContinuation(model)
        vehicles = {v.bus:dict(bus=v.bus, trip=v.trip, d=v.direction, i=v.index,
                              phase=v.phase, frontier=2050.) for v in model.snapshot.vehicles}
        state.initialize(vehicles)
        vehicles[0].update(i=1, phase='travel', frontier=2150.)
        state.event('departure', vehicles[0], 2050.)
        vehicles[0].update(phase='terminal')
        state.event('return', vehicles[0], 2060.)
        for now in (2000., 2020., 2050., 2051., 2052., 2060., 2061.):
            for reader in vehicles:
                for direction in (False, True):
                    for progress in (0., 500., 1000., 1500., 2000.):
                        kwargs = dict(current_bus_id=reader, current_direction=direction,
                            current_progress_m=progress, current_time_s=now, service_dwell_proxy_s=40.)
                        full = estimate_follower_departure_gap(**kwargs, vehicles=state.avl(now, reader))
                        sparse = estimate_follower_departure_gap(**kwargs,
                            vehicles=state.follower_avl(now, reader, direction, progress))
                        self.assertEqual(full, sparse)

    def test_spatial_tie_and_slow_nearest_follower_do_not_pick_fastest_eta(self):
        native = fixture()
        model = FastNativeGraph(native, native.observed)
        state = SparseContinuation(model)
        vehicles = {v.bus:dict(bus=v.bus, trip=v.trip, d=v.direction, i=v.index,
                              phase=v.phase, frontier=2050.) for v in model.snapshot.vehicles}
        state.initialize(vehicles)
        state.history[1][-1].update(direction=False, on_route=True, progress=900., launch=1999., speed=100.)
        state.history[2][-1].update(direction=False, on_route=True, progress=900., travel=False,
                                   launch=1900., speed=0.)
        chosen = state.follower_avl(2020., 0, False, 1000.)
        self.assertEqual(chosen[0].bus_id, 1)
        state.history[1][-1].update(progress=899.)
        self.assertEqual(state.follower_avl(2020., 0, False, 1000.)[0].bus_id, 2)


if __name__ == '__main__':
    unittest.main()
