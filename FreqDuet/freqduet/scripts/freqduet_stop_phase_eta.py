"""Passive stop-phase extension; labels and commands must precede the query."""

from bisect import bisect_left
from collections import defaultdict

import numpy as np

import audit_freqduet_v120_component_eta as model


class ServiceResiduals:
    def __init__(self, samples):
        groups = defaultdict(list)
        for direction, station, completed, duration in samples:
            groups[direction, station].append((completed, duration))
        self.groups = {k: sorted(v) for k, v in groups.items()}
        self.times = {k: [t for t, _ in v] for k, v in self.groups.items()}

    def estimate(self, direction, station, now, elapsed):
        k = direction, station
        if k not in self.groups:
            return None
        times = self.times[k]
        end = bisect_left(times, now)
        start = bisect_left(times, now-model.WINDOW_S)
        historical = start == end
        if historical:
            start = 0
        # A ready event at now is not in the strict prefix: retain equal ages.
        survivors = [(t, d-elapsed) for t, d in self.groups[k][start:end] if d >= elapsed]
        if not survivors:
            return None
        values = np.asarray([v for _, v in survivors])
        return dict(mean=float(values.mean()), p10=float(np.quantile(values, .1)),
            p90=float(np.quantile(values, .9)), count=len(values),
            latest=max(t for t, _ in survivors), historical_fallback=historical)


def from_stop(history, services, route, direction, origin, target, now,
              arrival_s, ready_s=None, command=None):
    if arrival_s >= now or origin > target:
        return dict(applied=False, status='no_completed_current_arrival')
    station = route['ids'][origin]
    labels, inputs = [], [arrival_s]
    ready_known = ready_s is not None and ready_s < now
    residual, known_end = None, None
    if ready_known:
        if origin == target:
            return dict(applied=False, status='target_ready_already_observed')
        inputs.append(ready_s)
        if command is None or command['policy_time_s'] >= now:
            return dict(applied=False, status='ready_without_prior_command')
        inputs.append(command['policy_time_s'])
        known_end = command['policy_time_s']+max(1., command['chosen_s'])
        remaining = max(0., known_end-now)
        phase = 'known_command'
    else:
        residual = services.estimate(direction, station, now, now-arrival_s)
        if residual is None:
            return dict(applied=False, status='no_completed_service_survivor')
        labels.append(residual['latest'])
        remaining = residual['mean']
        phase = 'service_in_progress'
        if origin < target:
            for channel in ('holding', 'ticks'):
                value = history.estimate((channel, direction, station), now)
                if value is None:
                    return dict(applied=False, status='missing_current_'+channel)
                remaining += value['mean']
                labels.append(value['latest'])
    tail = 0.
    if origin < target:
        prediction = model.component_prediction(history, route, direction, origin,
            target, float(route['positions'][origin]), now)
        if prediction is None:
            return dict(applied=False, status='missing_downstream_components')
        tail = sum(prediction[k] for k in ('running_s', 'service_s', 'holding_s', 'ticks_s', 'target_service_s'))
        labels.append(prediction['latest_completion_s'])
    latest_label = max(labels) if labels else None
    if max(inputs) >= now or (latest_label is not None and latest_label >= now):
        raise AssertionError('stop-phase forecast used simultaneous or unfinished information')
    return dict(applied=True, status='passive_stop_component', phase=phase,
        gap_s=remaining+tail, current_remaining_s=remaining, downstream_remaining_s=tail,
        latest_input_s=max(inputs), latest_label_s=latest_label,
        known_command_end_s=known_end, service_residual=residual)
