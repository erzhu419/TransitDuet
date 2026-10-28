"""Detached exact-computation candidate; no existing controller is replaced."""

import math

import freqduet_causal_event_graph as graph
from freqduet_arrival_metric_control import ArrivalMetricGraph
from freqduet_indexed_continuation import IndexedContinuation, IndexedNativeGraph
from lower.causal_follower_eta import AVLVehicleSnapshot


class CachedNativeRule:
    """Reuse the unchanged native rule only for identical query-local inputs."""

    def __init__(self, rule):
        self.rule = rule
        self.values = {}
        self.hits = 0

    def __call__(self, packet, legal):
        departure = packet['departure']
        key = (packet['bus'], packet['trip'], packet['station'], packet['direction'],
               packet['prepare_s'], packet['progress_m'], packet['service_s'],
               packet['target_s'], packet['predecessor'],
               None if departure is None else (departure['trip_id'], departure['time_s']),
               tuple(packet['avl']), tuple(legal))
        if key not in self.values:
            self.values[key] = self.rule(packet, legal)
        else:
            self.hits += 1
        return self.values[key]


class SparseContinuation(IndexedContinuation):
    def follower_avl(self, now, reader, direction, progress):
        """Project only eligible followers; preserve the native spatial/tie order."""
        reader_rank = self.rank[reader]
        best, best_gap, best_row, best_progress = None, math.inf, None, None
        for bid, history in self.history.items():
            if bid == reader:
                continue
            row = history[-1]
            if not (row['visible'] < now or
                    row['visible'] == now and row['rank'] < reader_rank):
                row = next(r for r in reversed(history) if r['visible'] < now or
                           r['visible'] == now and r['rank'] < reader_rank)
            if not row['on_route'] or row['direction'] != direction:
                continue
            value = row['progress']
            if row['travel']:
                endpoint = self.station_progress(row['direction'], row['index'])
                span = max(1., row['arrival_s'] - row['motion_s'] + 1)
                sampled = now + float(self.rank[bid] < reader_rank)
                value += (endpoint - value) * min(1., max(0., (sampled - row['motion_s']) / span))
            if not math.isfinite(value) or value < 0.:
                continue
            gap = progress - value
            if gap > 1e-6 and gap < best_gap:
                best, best_gap, best_row, best_progress = bid, gap, row, value
        if best is None:
            return ()
        route_speed = self.data['links'][best_row['direction']][best_row['index']][1]
        return (AVLVehicleSnapshot(best, best_row['direction'], best_row['on_route'],
                                   best_progress, best_row['launch'], best_row['speed'], route_speed),)

    def action(self, v, command_s, legal):
        s = self.model.snapshot
        station = s.routes[v['d']][v['i']].station
        original = self.model.originals[v['bus']]
        if original.phase == 'prepare' and original.trip == v['trip'] and original.index == v['i']:
            packet = self.data['packets'][v['bus'], v['trip'], station]
        else:
            prepare = command_s - 1
            progress = self.station_progress(v['d'], v['i'])
            packet = dict(bus=v['bus'], trip=v['trip'], station=station, direction=v['d'],
                prepare_s=prepare, progress_m=progress, service_s=self.service[v['bus']],
                target_s=360., predecessor=self.predecessors[v['bus']],
                avl=self.follower_avl(prepare, v['bus'], v['d'], progress),
                departure=self.departure_at((v['d'], station), prepare, v['bus']))
        return self.model.native_rule(packet, legal)[0]


class FastNativeGraph(IndexedNativeGraph):
    def __init__(self, base, observed):
        super().__init__(base, observed)
        self.native_rule = CachedNativeRule(self.native_rule)

    def evaluate(self, commands=None, homogeneous=False):
        return graph._simulate(self, commands or {}, homogeneous, SparseContinuation(self))


class FastArrivalMetricGraph(ArrivalMetricGraph, FastNativeGraph):
    """Use the existing arrival metric unchanged, on the sparse continuation."""
