"""Frozen old-day running/holding diagnostics; never rerun a model or policy."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import importlib.util
import math
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('component_readout',
    HERE.parent / 'component_readout01/component_readout.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
run, read, write = prior.run, prior.read, prior.write
FOCUS = {225: (10, 11), 227: (10,), 236: (10,), 249: (10,), 260: (10,)}


def links(trace):
    arrivals, departures, returned = defaultdict(list), {}, {}
    for order, row in enumerate(trace['arrivals']):
        arrivals[row['trip_id']].append((row['time_s'], order, row['station_id']))
    for row in trace['departures']:
        key = row['trip_id'], row['station_id']
        assert key not in departures
        departures[key] = row['time_s']
    for row in trace['returns']:
        returned[row['trip_id']] = row['physical_time_s']
    sequences, observations, by_trip = {}, defaultdict(list), {}
    for row in trace['dispatches']:
        trip, direction = row['trip'], row['direction']
        visits = sorted(arrivals[trip])
        end = returned[trip]
        assert visits and all(at < end for at, _, _ in visits)
        seq = tuple(s for _, _, s in visits)
        assert sequences.setdefault(direction, seq) == seq and len(seq) == len(set(seq))
        previous, journey = row['actual_s'], []
        for at, order, station in visits + [(end, trip, 'terminal')]:
            done = end if station == 'terminal' else departures[trip, station]
            assert previous < at <= done
            key = direction, station
            journey.append((key, previous, at))
            observations[key].append((at, order, done, at-previous))
            previous = done
        by_trip[trip] = journey
    history = {}
    for key, visits in observations.items():
        # CompletedPrefix excludes each station's first arrival, including terminals.
        ordered = sorted(sorted(visits)[1:], key=lambda v: v[2])
        clocks, running = [], []
        for _, _, done, duration in ordered:
            clocks.append(done)
            running.append(duration + (running[-1] if running else 0))
        history[key] = clocks, running
    return by_trip, history


def running_split(journey, history, query, predicted_travel):
    pieces, current = [], None
    for key, departure, arrival in journey:
        if arrival < query:
            continue
        clocks, cumulative = history[key]
        n = bisect_left(clocks, query)
        assert n >= 3
        frozen = math.ceil(cumulative[n-1]/n)
        if departure < query:
            assert current is None
            current = dict(station=key[1], direction=key[0], actual_s=arrival-query,
                           full_link_model_s=frozen, completed_visits=n)
        else:
            pieces.append(dict(station=key[1], direction=key[0], actual_s=arrival-departure,
                               predicted_s=frozen, completed_visits=n))
    full = sum(r['predicted_s'] for r in pieces)
    remainder = predicted_travel-full
    supported = (remainder == 0 if current is None else
                 0 <= remainder <= current['full_link_model_s']-1)
    if current is not None:
        current['inferred_model_s'] = remainder
    return dict(supported=supported, current=current, full_links=pieces, remainder_s=remainder)


def substitution(predicted, actual, *fields):
    return predicted['return_s'] + sum(actual['components'][k]-predicted['components'][k] for k in fields)


def check():
    one, two = (False, 1), (False, 'terminal')
    history = {one: ([5, 10, 15, 20], [4, 8, 12, 40]), two: ([5, 10, 15], [6, 12, 18])}
    result = running_split([(one, 18, 22), (two, 28, 36)], history, 20, 9)
    assert result['supported'] and result['current']['inferred_model_s'] == 3
    assert result['current']['completed_visits'] == 3 and result['full_links'][0]['predicted_s'] == 6
    assert not running_split([(one, 18, 22), (two, 28, 36)], history, 20, 10)['supported']


def audit(root):
    check()
    gate = read(root/'component_parity/summary/results.json')
    frozen = read(root/'component_readout/summary/results.json')
    assert gate['passed'] and frozen['complete'] and gate['day'] == frozen['day'] == run.PARITY
    folder = root/'component_parity/cases'/run.REGIME/str(run.PARITY)/run.CAP
    trace = run.old.frozen.helper(run.old.frozen.ASSETS, 'gate_default').raw(folder/'raw/path.pkl.gz')
    sidecar = {tuple(k):r for r in read(folder/'raw/predicted_components.json') for k in r['slots']}
    journeys, history = links(trace)
    launches = {r['trip']:r for r in trace['dispatches']}
    previous, pred_trip = {}, {}
    for row in sorted(launches.values(), key=lambda r:(r['actual_s'], r['trip'])):
        pred_trip[row['trip']] = previous.get(row['bus'])
        previous[row['bus']] = row['trip']
    groups, examples = defaultdict(Counter), defaultdict(list)
    switches = {'travel': ('travel_s',), 'holding': ('hold_residence_s',),
                'travel_and_holding': ('travel_s', 'hold_residence_s')}
    for slot in read(folder/'raw/public_slots.json'):
        trip, lead, now, due = (slot[k] for k in ('trip', 'lead_s', 'query_s', 'public_s'))
        if not slot['scenarios']:
            continue
        model = slot['scenarios'][0]
        pred = model['launches'].get(str(trip))
        if model['failure'] or pred is None:
            continue
        by_bus = {r['bus']:r for r in sidecar[trip,lead]['scenarios'][0] if r['supported']}
        ptime, bus = pred
        actual = launches[trip]
        atime = actual['actual_s']
        assert actual['scheduled_s'] == actual['eligible_s'] == due
        status = ('correct_late' if atime > due else 'correct_on_time') if (ptime > due) == (atime > due) else (
            'miss' if atime > due else 'false_alarm')
        matched = (bus == actual['bus'] and bus in by_bus and by_bus[bus]['trip'] == pred_trip[trip]
            and not any(t < ptime and b == bus for other, (t,b) in model['launches'].items() if int(other) != trip))
        focal = FOCUS.get(trip, ()) if lead == 300 else ()
        for candidate in sorted(set(focal) | ({bus} if matched else set())):
            p = by_bus.get(candidate)
            if p is None:
                continue
            a = prior.actual_components(trace, candidate, p['trip'], now)
            if not a['supported']:
                continue
            split = running_split(journeys[p['trip']], history, now, p['components']['travel_s'])
            error = prior.compare(p,a)
            if split['supported']:
                full_error = sum(r['predicted_s']-r['actual_s'] for r in split['full_links'])
                c = split['current']
                partial_error = c['inferred_model_s']-c['actual_s'] if c else 0
                assert full_error+partial_error == error['travel_s']
            corrected = {k:substitution(p,a,*v) for k,v in switches.items()}
            if candidate in focal:
                examples[trip].append(dict(bus=candidate, active_trip=p['trip'], query_s=now,
                    forecast_return=p['return_s'], actual_return=a['return_s'], substitutions=corrected,
                    errors=error, running=split))
            if candidate != bus or not matched:
                continue
            group = groups[f'{lead}/{status}']
            group.update(error)
            if split['supported']:
                group.update(running_split_n=1, full_link_error_s=full_error,
                             current_link_error_s=partial_error, full_link_count=len(split['full_links']),
                             absolute_full_link_sum_error_s=abs(full_error),
                             absolute_current_link_error_s=abs(partial_error))
            if ptime == max(due,p['return_s']+1) and atime == max(due,a['return_s']+1):
                group['clock_identity_n'] += 1
                group['frozen_wrong'] += (ptime > due) != (atime > due)
                for key, clock in corrected.items():
                    group[key+'_wrong'] += (clock >= due) != (atime > due)
    expected = {f'{lead.split("/")[0]}/{status}': row for lead, data in frozen['groups'].items()
                if lead.endswith('/scenario_0') for status, row in data['by_status'].items()}
    assert groups.keys() == expected.keys()
    for key, row in expected.items():
        assert all(groups[key][field] == row[field] for field in ('n', 'return_error_s', *prior.FIELDS))
    pair = {r['bus']:r for r in examples[225]}
    assert set(pair) == {10, 11}
    a, b = pair[10], pair[11]
    gaps = {key:a['substitutions'][key]-b['substitutions'][key] for key in switches}
    gaps.update(frozen=a['forecast_return']-b['forecast_return'], actual=a['actual_return']-b['actual_return'])
    assert gaps['frozen'] == -38 and gaps['actual'] == 19
    write(root/'continuation_audit/summary/results.json', dict(complete=True, physical_starts=0,
        day=gate['day'], training_updates=0, main_policy_changed=False, groups=groups, focal_chain=examples,
        trip225_relative_returns_s=gaps, synthetic_check='strict completed clock and partial-link bound',
        scope='Single consumed day. Realized component substitutions are accounting, not executable counterfactuals.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    audit(parser.parse_args().root)
