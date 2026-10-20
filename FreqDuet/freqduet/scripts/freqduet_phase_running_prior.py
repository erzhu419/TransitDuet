"""Leave-day-out running moments with one native-prefix shrinkage anchor."""

from bisect import bisect_left
from math import sqrt
from statistics import fmean

import freqduet_historical_service_eta as frozen


def exclude_day(cells, held):
    return [dict(c, days=[d for d in c['days'] if d['day'] != held]) for c in cells]


class RunningLaw:
    def __init__(self, cells, held, prefix):
        self.prior = {}
        self.prefix = prefix
        for c in cells:
            days = [d for d in c['days'] if d['day'] != held]
            if days:
                self.prior[c['direction'], c['station'], c['hour']] = days

    def estimate(self, direction, station, now, entry, native):
        k = direction, station
        times = self.prefix.times[k]
        end, start = bisect_left(times, now), bisect_left(times, now-3600.)
        if start == end:
            start = 0
        values = [r[2] for r in self.prefix.rows[k][start:end]]
        if not values or times[end-1] >= now:
            raise AssertionError('running anchor requires a strict completed prefix')
        if abs(fmean(values)-native['mean']) > 1e-7:
            raise AssertionError('running anchor differs from frozen native mean')
        days = self.prior.get((direction, station, int(entry//3600)), [])
        n = len(days)
        mean = (sum(d['mean'] for d in days)+native['mean'])/(n+1)
        moment = (sum(d['second'] for d in days)+fmean(v*v for v in values))/(n+1)
        return dict(mean=mean, prior_days=n, historical_weight=n/(n+1),
            dispersion_s=sqrt(max(0., moment-mean*mean)),
            latest_current_s=times[end-1])


def forecast(path, row, geometry, objects, phases, service_law, running_law):
    p, status, _ = frozen.position(path, row, objects[-1], geometry, objects, service_law, True)
    if p is None:
        return None, dict(status=status)
    elapsed, running_total = p['initial_s'], 0.
    supports, dispersions, partial = [], [], 0
    history = objects[0]
    for i in range(p['origin']+1, p['target']+1):
        station, direction, now = p['route']['ids'][i], p['direction'], p['now']
        native = history.estimate(('running', direction, station), now)
        service = history.estimate(('service', direction, station), now)
        if native is None or service is None:
            return None, dict(status='missing_downstream_component')
        fraction = ((p['route']['positions'][i]-p['progress'])/
            (p['route']['positions'][i]-p['route']['positions'][i-1])) if i == p['origin']+1 else 1.
        moving = i == p['origin']+1 and p['progress'] > p['route']['positions'][p['origin']]
        selected = None if moving else running_law.estimate(direction, station, now, now+elapsed, native)
        duration = fraction*native['mean'] if selected is None else selected['mean']
        partial += int(moving)
        if selected is not None:
            supports.append(selected['prior_days'])
            dispersions.append(selected['dispersion_s'])
        running_total += duration
        elapsed += duration
        proposal = service_law.estimate(direction, station, now, now+elapsed)
        selected_service = service if proposal is None else proposal
        latest = selected_service.get('latest', selected_service.get('latest_current_s'))
        if native['latest'] >= now or service['latest'] >= now or (latest is not None and latest >= now):
            raise AssertionError('running shadow used an unfinished current-day label')
        elapsed += selected_service['mean']
        if i < p['target']:
            for channel in ('holding', 'ticks'):
                value = history.estimate((channel, direction, station), now)
                if value is None:
                    return None, dict(status='missing_downstream_'+channel)
                if value['latest'] >= now:
                    raise AssertionError('running shadow used unfinished residence')
                elapsed += value['mean']
    return float(elapsed), dict(status=status, running_s=running_total,
        changed_legs=sum(n > 0 for n in supports), unsupported_legs=supports.count(0),
        single_day_legs=supports.count(1), partial_legs_retained=partial,
        mean_prior_days=fmean(supports) if supports else None,
        mean_leg_dispersion_s=fmean(dispersions) if dispersions else None)
