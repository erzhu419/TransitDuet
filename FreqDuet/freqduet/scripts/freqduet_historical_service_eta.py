"""Independent-day service laws and observable launch/ready state bridges."""

from bisect import bisect_left
from collections import defaultdict
from itertools import accumulate

import freqduet_phase_clock_eta as frozen


class ServiceLaw:
    def __init__(self, cells, arrivals, ready, cutoff):
        self.cutoff = cutoff
        self.prior = {}
        for cell in cells:
            samples = []
            for day in cell['days']:
                modes = [v for v in day['modes'] if v]
                for values in modes:
                    samples.extend((day['day'], float(v), 1./len(modes)/len(values)) for v in values)
            samples.sort(key=lambda r: r[1])
            maxima = defaultdict(float)
            for day, value, _ in samples:
                maxima[day] = max(maxima[day], value)
            self.prior[cell['direction'], cell['station'], cell['clearance']] = dict(
                values=[d for _, d, _ in samples], maxima=list(maxima.values()),
                weights=[0., *accumulate(w for _, _, w in samples)],
                moments=[0., *accumulate(d*w for _, d, w in samples)])
        groups = defaultdict(list)
        for k, arrival in arrivals.items():
            end = ready[k]['ready_time_s']
            groups[k[2], k[1], arrival['time_s'] >= cutoff].append((end, end-arrival['time_s']))
        self.local = {k: sorted(v) for k, v in groups.items()}
        self.times = {k: [t for t, _ in v] for k, v in self.local.items()}

    def estimate(self, direction, station, now, arrival, elapsed=0.):
        k = direction, station, arrival >= self.cutoff
        times = self.times.get(k, [])
        end = bisect_left(times, now)
        start = bisect_left(times, now-frozen.model.WINDOW_S)
        historical = start == end
        if historical:
            start = 0
        local = self.local.get(k, [])[start:end]
        past = self.prior.get(k)
        prior_n = prior_survivors = prior_days = survivor_days = 0
        mass = excess = 0.
        if past is not None:
            prior_n, prior_days = len(past['values']), len(past['maxima'])
            i = bisect_left(past['values'], elapsed)
            prior_survivors = prior_n-i
            survivor_days = sum(v >= elapsed for v in past['maxima'])
            mass = past['weights'][-1]-past['weights'][i]
            excess = max(0., past['moments'][-1]-past['moments'][i]-elapsed*mass)
        local_survivors = [(d-elapsed) for _, d in local if d >= elapsed]
        if local:
            mass += len(local_survivors)/len(local)
            excess += sum(local_survivors)/len(local)
        if prior_survivors+len(local_survivors) == 0:
            return None
        return dict(mean=excess/mass,
            local_n=len(local), local_survivor_n=len(local_survivors),
            prior_n=prior_n, prior_days=prior_days,
            prior_survivor_n=prior_survivors, prior_survivor_days=survivor_days,
            latest_current_s=local[-1][0] if local else None,
            historical_fallback=historical, conditional_mass=mass)


class ObservedReady:
    def __init__(self, ready):
        # V145 writes the callback dictionary in insertion order, not sorted order.
        self.ready = ready
        self.rank = {k: i for i, k in enumerate(ready)}

    def same_clock_before(self, current, query, now):
        return (self.ready[current]['ready_time_s'] == now and
            self.rank[current] < self.rank[query])


def position(path, row, indexed, geometry, objects, law=None, events=False):
    history, services, durations, _ = objects
    original, status = frozen.state(path, row, indexed, geometry, history, services, durations)
    if law is None and not events:
        return original, status, None
    e, now, direction = row['evidence'], row['prediction_time_s'], bool(row['direction'])
    trip = e['predicted_follower_trip_id']
    if e['follower_gap_prediction_s'] is None or trip not in indexed:
        return original, status, None
    route, motion = geometry[direction], indexed[trip].motion
    target = route['index'][row['station_id']]
    progress = float(route['positions'][target])-e['spatial_gap_m']
    i, j = bisect_left(motion.dt, now), bisect_left(motion.at, now)
    if j and (not i or motion.at[j-1] > motion.dt[i-1]):
        arrival = motion.arrivals[j-1]
        current = int(arrival['trip_id']), int(arrival['station_id']), direction
        origin = route['index'][current[1]]
        if origin > target or abs(progress-route['positions'][origin]) > 1e-6:
            return original, status, None
        query = int(row['trip_id']), int(row['station_id']), direction
        observed = path['_observed_ready']
        if events and observed.same_clock_before(current, query, now):
            if path['ready'][current]['bus_id'] != e['predicted_follower_bus_id']:
                raise AssertionError('observed ready belongs to another physical follower')
            if origin == target:
                remaining = 0.
            else:
                residual = durations.estimate(direction, current[1], now, 0.)
                if residual is None:
                    return None, 'unsupported_observed_ready_residence', None
                remaining = residual['mean']
            kind, support = 'observed_same_clock_ready', None
        elif law is not None and path['ready'][current]['ready_time_s'] >= now:
            support = law.estimate(direction, current[1], now, arrival['time_s'], now-arrival['time_s'])
            if support is None:
                return None, 'unsupported_historical_service_tail', None
            remaining, kind = support['mean'], 'historical_current_service'
            if origin < target:
                for channel in ('holding', 'ticks'):
                    value = history.estimate((channel, direction, current[1]), now)
                    if value is None:
                        return None, 'missing_current_'+channel, support
                    remaining += value['mean']
        else:
            return original, status, None
        return dict(route=route, direction=direction, origin=origin, target=target,
            progress=float(route['positions'][origin]), now=now, initial_s=remaining), kind, support
    if events and original is None and path['dispatch'][trip]['actual_s'] == now:
        # All launches finish before the runtime starts its sequential bus-update loop.
        if 0. <= progress < route['positions'][1] and target >= 1:
            return dict(route=route, direction=direction, origin=0, target=target,
                progress=progress, now=now, initial_s=0.), 'observed_same_clock_launch', None
    return original, status, None


def forecast(path, row, geometry, objects, phases, law=None, events=False):
    history, _, _, indexed = objects
    p, status, current_support = position(path, row, indexed, geometry, objects, law, events)
    if p is None:
        return None, dict(status=status, current_support=current_support)
    elapsed, supports = p['initial_s'], []
    for i in range(p['origin']+1, p['target']+1):
        station, direction, now = p['route']['ids'][i], p['direction'], p['now']
        running = history.estimate(('running', direction, station), now)
        native = history.estimate(('service', direction, station), now)
        if running is None or native is None:
            return None, dict(status='missing_downstream_component')
        fraction = ((p['route']['positions'][i]-p['progress'])/
            (p['route']['positions'][i]-p['route']['positions'][i-1])) if i == p['origin']+1 else 1.
        elapsed += fraction*running['mean']
        proposal = (phases.estimate(direction, station, now, now+elapsed) if law is None else
            law.estimate(direction, station, now, now+elapsed))
        selected = native if proposal is None else proposal
        latest = selected.get('latest', selected.get('latest_current_s'))
        if running['latest'] >= now or native['latest'] >= now or (latest is not None and latest >= now):
            raise AssertionError('historical candidate used an unfinished current-day label')
        elapsed += selected['mean']
        if law is not None:
            supports.append(proposal)
        if i < p['target']:
            for channel in ('holding', 'ticks'):
                value = history.estimate((channel, direction, station), now)
                if value is None:
                    return None, dict(status='missing_downstream_'+channel)
                if value['latest'] >= now:
                    raise AssertionError('unfinished downstream residence')
                elapsed += value['mean']
    return float(elapsed), dict(status=status, current_support=current_support,
        downstream_prior_stops=sum(s is not None and s['prior_n'] > 0 for s in supports),
        downstream_local_only_stops=sum(s is not None and s['prior_n'] == 0 for s in supports),
        downstream_no_phase_stops=sum(s is None for s in supports))
