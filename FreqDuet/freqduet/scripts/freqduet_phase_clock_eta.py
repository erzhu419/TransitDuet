"""Fixed demand-phase service means and source-independent physical ETA clocks."""

from bisect import bisect_left

import audit_freqduet_v141_ready_eta as frozen


model = frozen.model
TOTALS = ('running_s', 'service_s', 'holding_s', 'ticks_s', 'target_service_s')


class PhaseServices:
    def __init__(self, arrivals, ready, cutoff):
        self.cutoff = cutoff
        self.history = model.CompletedMeans([
            ((k[2], k[1], a['time_s'] >= cutoff), ready[k]['ready_time_s'],
             ready[k]['ready_time_s']-a['time_s']) for k, a in arrivals.items()])

    def estimate(self, direction, station, now, arrival):
        return self.history.estimate((direction, station, arrival >= self.cutoff), now)


def tail(history, phases, route, direction, origin, target, progress, now,
         initial_s=0., phase_service=False):
    elapsed, service_delta, changed, unsupported = initial_s, 0., 0, 0
    for i in range(origin+1, target+1):
        station = route['ids'][i]
        running = history.estimate(('running', direction, station), now)
        service = history.estimate(('service', direction, station), now)
        if running is None or service is None:
            return None
        fraction = ((route['positions'][i]-progress)/
            (route['positions'][i]-route['positions'][i-1])) if i == origin+1 else 1.
        elapsed += fraction*running['mean']
        selected = service
        if phase_service:
            proposal = phases.estimate(direction, station, now, now+elapsed)
            if proposal is None:
                unsupported += 1
            else:
                selected = proposal
                changed += 1
        if max(running['latest'], service['latest'], selected['latest']) >= now:
            raise AssertionError('future or simultaneous completed component')
        service_delta += selected['mean']-service['mean']
        elapsed += selected['mean']
        if i < target:
            for channel in ('holding', 'ticks'):
                value = history.estimate((channel, direction, station), now)
                if value is None:
                    return None
                if value['latest'] >= now:
                    raise AssertionError('unfinished future residence label')
                elapsed += value['mean']
    return dict(gap_s=elapsed, service_delta_s=service_delta,
        phase_supported_stops=changed, phase_unsupported_stops=unsupported)


def state(path, row, indexed, geometry, history, services, durations):
    e, now, direction = row['evidence'], row['prediction_time_s'], bool(row['direction'])
    trip = e['predicted_follower_trip_id']
    if e['follower_gap_prediction_s'] is None or trip not in indexed:
        return None, 'unavailable'
    route = geometry[direction]
    target = route['index'][int(row['station_id'])]
    progress = float(route['positions'][target])-e['spatial_gap_m']
    motion = indexed[trip].motion
    i, j = bisect_left(motion.dt, now), bisect_left(motion.at, now)
    if j and (not i or motion.at[j-1] > motion.dt[i-1]):
        arrival = motion.arrivals[j-1]
        k = int(arrival['trip_id']), int(arrival['station_id']), bool(arrival['direction'])
        origin = route['index'][k[1]]
        if abs(progress-route['positions'][origin]) > 1e-6:
            return None, 'pending_stop_position_mismatch'
        stop = frozen.frozen.from_stop(history, services, route, direction, origin,
            target, now, arrival['time_s'], path['ready'][k]['ready_time_s'], path['trace'].get(k))
        if stop['status'] == 'ready_without_prior_command':
            stop = frozen.predict_ready(history, durations, route, direction, origin,
                target, now, path['ready'][k]['ready_time_s'])
            if stop['applied']:
                remaining = stop['ready_residual']['mean']
        elif stop['applied']:
            remaining = stop['current_remaining_s']
        if not stop['applied']:
            return None, stop['status']
        progress = float(route['positions'][origin])
    else:
        origin = motion.origin(now, progress, route)
        if origin is None:
            return None, 'no_strict_physical_segment'
        remaining = 0.
    if origin > target:
        return None, 'target_behind_origin'
    return dict(route=route, direction=direction, origin=origin, target=target,
        progress=progress, now=now, initial_s=remaining), 'structured'


def forecasts(path, row, saved, geometry, objects, phases):
    history, services, durations, indexed = objects
    base = saved['v141_s']
    values = dict(base=base, service=base, clock=base, joint=base)
    position, status = state(path, row, indexed, geometry, history, services, durations)
    if position is None:
        return values, dict(status=status, phase_supported_stops=0, phase_unsupported_stops=0)
    native = tail(history, phases, **position)
    candidate = tail(history, phases, **position, phase_service=True)
    if native is None or candidate is None:
        return values, dict(status='missing_completed_component',
            phase_supported_stops=0, phase_unsupported_stops=0)
    structured_base = (row['online']['component_used'] or
        saved['v140_status'] == 'passive_stop_component' or
        saved['v141_status'] == 'passive_unissued_ready')
    if structured_base:
        if abs(native['gap_s']-base) > 1e-7:
            raise AssertionError('common clock changed an already structured frozen forecast')
        values['service'] = candidate['gap_s']
    values.update(clock=native['gap_s'], joint=candidate['gap_s'])
    return values, dict(status='structured', structured_base=bool(structured_base),
        phase_supported_stops=candidate['phase_supported_stops'],
        phase_unsupported_stops=candidate['phase_unsupported_stops'])
