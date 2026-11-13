"""Passive first-return components, using the frozen model's own event output."""

from collections import Counter

FIELDS = ('travel_s', 'service_prepare_s', 'hold_residence_s')


def model_components(snapshot, row):
    records = []
    field = dict(travel='travel_s', service='service_prepare_s', prepare='service_prepare_s',
                 hold='hold_residence_s')
    for vehicle in snapshot.vehicles:
        if vehicle.phase == 'terminal':
            continue
        end = row['returns'].get(vehicle.bus)
        if end is None:
            records.append(dict(bus=vehicle.bus, trip=vehicle.trip, supported=False))
            continue
        counts, clock, phase = Counter(dict.fromkeys(FIELDS, 0.)), snapshot.now_s, vehicle.phase
        for time, kind, bus in row['nodes']:
            if bus != vehicle.bus or time > end:
                continue
            assert kind in ('arrival', 'prepare', 'departure') and time >= clock
            counts[field[phase]] += time-clock
            if time == end and kind == 'arrival':
                clock = time
                break
            if kind == 'arrival':
                assert phase == 'travel'
                phase = 'service'
            elif kind == 'prepare':
                assert phase in ('service', 'prepare')
                phase = 'hold'
            else:
                assert phase == 'hold'
                phase = 'travel'
            clock = time
        assert clock == end and sum(counts.values()) == end-snapshot.now_s
        requests = [(t, h) for (bus, trip, _), (t, h) in row['actions'].items()
                    if (bus, trip) == (vehicle.bus, vehicle.trip) and t >= snapshot.now_s]
        records.append(dict(bus=vehicle.bus, trip=vehicle.trip, supported=True,
            query_s=snapshot.now_s, return_s=end, components=dict(counts),
            future_requested_h_s=sum(h for _, h in requests), future_commands=len(requests)))
    return records


class ComponentSlots:
    """Tap only the passive observer's local graph; the live controller is untouched."""

    def __init__(self, observer_module, leads):
        self.module = observer_module
        self.base = observer_module.PublicSlots(leads)
        self.records = []

    def before_step(self, env, bridge):
        module, output = self.module, []
        original = module.ReferenceGraph

        class RecordedGraph(original):
            def evaluate(self, commands=None, homogeneous=False):
                row = super().evaluate(commands, homogeneous)
                output.append(model_components(self.snapshot, row))
                return row

        module.ReferenceGraph = RecordedGraph
        start = len(self.base.records)
        try:
            self.base.before_step(env, bridge)
        finally:
            module.ReferenceGraph = original
        if len(self.base.records) != start:
            slots = self.base.records[start:]
            # Unsupported forecasts may stop after a partial scenario evaluation.
            available = bool(slots[0]['scenarios'])
            if available:
                assert len(output) == len(slots[0]['scenarios'])
            self.records.append(dict(query_s=float(env.current_time), available=available,
                slots=[(s['trip'], s['lead_s']) for s in slots], scenarios=output))


def actual_components(trace, bus, trip, query_s):
    launches = [r for r in trace['dispatches'] if r['trip'] == trip and r['bus'] == bus]
    ends = [r['physical_time_s'] for r in trace['returns'] if (r['bus_id'], r['trip_id']) == (bus, trip)]
    assert len(launches) == len(ends) == 1
    launch, end = launches[0]['actual_s'], ends[0]
    assert launch <= query_s <= end
    arrivals = sorted((r for r in trace['arrivals'] if r['trip_id'] == trip and r['time_s'] < end),
                      key=lambda r: r['time_s'])
    departures = {r['station_id']: r['time_s'] for r in trace['departures'] if r['trip_id'] == trip}
    queries = {r['station']: r for r in trace['queries'] if r['trip'] == trip and r['bus'] == bus}
    intervals, previous = [], launch
    for arrival in arrivals:
        station, at = arrival['station_id'], arrival['time_s']
        if station not in queries or station not in departures:
            return dict(supported=False, reason='missing per-stop command or departure', station=station)
        command, depart = queries[station], departures[station]
        now, h = command['time_s'], command['executed_request_s']
        assert previous <= at <= now < depart and depart-now == max(1, h)
        intervals += [(previous, at, 'travel_s'), (at, now, 'service_prepare_s'),
                      (now, depart, 'hold_residence_s')]
        previous = depart
    assert arrivals and previous <= end
    intervals.append((previous, end, 'travel_s'))
    counts = Counter(dict.fromkeys(FIELDS, 0.))
    for start, stop, kind in intervals:
        counts[kind] += max(0., stop-max(query_s, start))
    assert sum(counts.values()) == end-query_s
    future = [q for q in queries.values() if q['time_s'] >= query_s]
    return dict(bus=bus, trip=trip, supported=True, query_s=query_s, return_s=end, components=dict(counts),
        future_requested_h_s=sum(q['executed_request_s'] for q in future), future_commands=len(future))
