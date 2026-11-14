"""Read the two completed V415 paths on the focal day; no simulation."""

import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import pickle


DAY = 2550004
ARMS = ('v406_cap_10', 'v406_cap_10_hourly')
FIELDS = ('launch', 'holding', 'other')


def read(path):
    return json.loads(path.read_text())


def gaps(rows, clock, group):
    previous = {}
    for row in rows:
        key = group(row)
        if key in previous:
            old = previous[key]
            gap = row[clock] - old[clock]
            assert gap >= 0, (key, old, row)
            yield old, row, gap
        previous[key] = row


def moments(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean)**2 for x in values) / n
    return dict(n=n, mean_s=mean, var_s2=var, cv2=var/mean**2)


def describe(trace, receipt):
    result = receipt['endpoints']
    launches = {r['trip']: r for r in trace['dispatches']}
    assert len(launches) == len(trace['dispatches']) == len(trace['returns']) == 262
    assert all(r['public_s'] == r['scheduled_s'] == r['eligible_s'] <= r['actual_s']
               for r in launches.values())
    launch_gaps = list(gaps(trace['dispatches'], 'actual_s', lambda r: r['direction']))
    dispatch = moments([g for _, _, g in launch_gaps])
    assert math.isclose(dispatch['cv2'], result['dispatch_cv_squared'], abs_tol=1e-12)
    assert sum(r['actual_s']-r['public_s'] for r in launches.values()) == result['public_delay_s_per_day']
    arrival_rows = []
    for a, b, g in gaps(trace['arrivals'], 'time_s', lambda r: (r['direction'], r['station_id'])):
        left, right = launches[a['trip_id']], launches[b['trip_id']]
        d = right['actual_s'] - left['actual_s']
        h = b['realized_holding'] - a['realized_holding']
        o = b['other_elapsed'] - a['other_elapsed']
        assert a['release'] == left['actual_s'] and b['release'] == right['actual_s'] and d+h+o == g
        arrival_rows.append(dict(direction=b['direction'], station=b['station_id'],
            predecessor=a['trip_id'], trip=b['trip_id'], time_s=b['time_s'],
            gap_s=g, launch=d, holding=h, other=o))
    arrival = moments([r['gap_s'] for r in arrival_rows])
    assert math.isclose(arrival['cv2'], result['cv_squared'], abs_tol=1e-12)
    n, mean = arrival['n'], arrival['mean_s']
    # Center by each path's full-day means, so the grouped contributions sum to CV2.
    part_means = {k: sum(r[k] for r in arrival_rows)/n for k in FIELDS}
    by_station, by_hour = defaultdict(float), defaultdict(float)
    cv_parts = dict.fromkeys(FIELDS, 0.)
    for r in arrival_rows:
        centered = r['gap_s']-mean
        contribution = centered**2/(n*mean**2)
        r['cv_contribution'] = contribution
        by_station[f"{int(r['direction'])}:{r['station']}"] += contribution
        by_hour[str(int(r['time_s']//3600))] += contribution
        for k in FIELDS:
            cv_parts[k] += (r[k]-part_means[k])*centered/(n*mean**2)
    assert math.isclose(sum(cv_parts.values()), arrival['cv2'], abs_tol=1e-12)
    return dict(dispatch=dispatch, arrival=arrival, cv_parts=cv_parts,
                by_station=by_station, by_hour=by_hour), arrival_rows, launch_gaps


def difference(base, hourly):
    return {k: hourly.get(k, 0)-base.get(k, 0) for k in sorted(base.keys() | hourly.keys())}


def passenger_effect(base, hourly):
    first, second = (r['passenger_detail'] for r in (base, hourly))
    assert first['generated'] == second['generated']
    n = first['generated']
    metrics = [r['clock_accounting'] for r in (base, hourly)]
    stations, hours, cells = defaultdict(float), defaultdict(float), {}
    for k, (count, wait, _) in second['station_hour'].items():
        old_count, old_wait, _ = first['station_hour'][k]
        assert count == old_count
        station, hour = k.split(':')
        cells[k] = wait-old_wait
        stations[station] += cells[k]
        hours[hour] += cells[k]
    assert first['station_hour'].keys() == second['station_hour'].keys()
    stamp = second['waiting_person_s']-first['waiting_person_s']
    assert math.isclose(sum(cells.values()), stamp, abs_tol=1e-6)
    offset = metrics[1]['waiting_person_s']['timestamp_minus_interval'] - metrics[0]['waiting_person_s']['timestamp_minus_interval']
    official = metrics[1]['waiting_person_s']['interval_sum'] - metrics[0]['waiting_person_s']['interval_sum']
    assert math.isclose(stamp-offset, official, abs_tol=1e-6)
    return dict(generated=n, official_wait_delta_s_per_person=official/n, timestamp_wait_delta_s_per_person=stamp/n,
                timestamp_minus_interval_delta_s=offset, wait_by_appearance_hour_person_s=hours,
                wait_by_station_person_s=stations, largest_wait_increases=sorted(cells.items(), key=lambda p:-p[1])[:6])


def cycle_rows(trace, focus):
    returns = {(r['bus_id'], r['trip_id']): r['physical_time_s'] for r in trace['returns']}
    holding = defaultdict(float)
    for q in trace['queries']:
        holding[q['trip']] += q['executed_request_s']
    previous, out = {}, {}
    for r in trace['dispatches']:
        trip, bus = r['trip'], r['bus']
        old = previous.get(bus)
        previous[bus] = r
        if trip not in focus:
            continue
        observed = dict(r)
        if old is not None:
            returned = returns[bus, old['trip']]
            assert returned == r['ready_return_s'] and r['actual_s'] >= returned+1
            observed.update(inbound_trip=old['trip'], inbound_launch_s=old['actual_s'],
                inbound_elapsed_s=returned-old['actual_s'], inbound_holding_s=holding[old['trip']],
                after_return_s=r['actual_s']-returned)
        out[trip] = observed
    return out


def audit(root):
    frozen = read(root / 'aggregate/summary/results.json')
    assert frozen['complete'] and frozen['physical_starts'] == 17 and not frozen['passed']
    pair = frozen['by_day'][str(DAY)]
    arms = []
    for arm in ARMS:
        case = root / 'cases/slow' / str(DAY) / arm
        receipt = read(case / 'summary/results.json')
        assert receipt['completed'] and receipt['physical_account']['passed']
        assert all(pair['arms'][arm]['metrics'][k] == v for k,v in receipt['endpoints'].items())
        with gzip.open(case / 'raw/path.pkl.gz', 'rb') as stream:
            trace = pickle.load(stream)
        arms.append((receipt, trace, *describe(trace, receipt)))
    (br, bt, b, ba, bg), (hr, ht, h, ha, hg) = arms
    assert bt['public_clocks'] == ht['public_clocks']
    assert len(bt['queries']) == len(ht['queries']) == 5200
    assert bt['uniforms'].keys() == ht['uniforms'].keys()
    assert all(row['uniform'] == ht['uniforms'][key]['uniform'] for key,row in bt['uniforms'].items())
    # Join by the observed arrival trip, and report predecessor changes explicitly.
    key = lambda r: (r['direction'], r['station'], r['trip'])
    base_arr = {key(r): r for r in ba}
    hour_arr = {key(r): r for r in ha}
    assert len(base_arr) == len(ba) and len(hour_arr) == len(ha)
    first = next((i, x, y) for i,(x,y) in enumerate(zip(bt['queries'], ht['queries']))
                 if (x['time_s'],x['trip'],x['station'],x['bus'],x['executed_request_s']) !=
                    (y['time_s'],y['trip'],y['station'],y['bus'],y['executed_request_s']))
    i, x, y = first
    fields = ('time_s','trip','station','bus','action_s','executed_request_s','load','boarded')
    divergence = dict(common_prior_commands=i, pooled={k:x[k] for k in fields}, hourly={k:y[k] for k in fields})
    common = base_arr.keys() & hour_arr.keys()
    largest = sorted(hour_arr, key=lambda k:-(hour_arr[k]['cv_contribution'] -
                     base_arr.get(k, {}).get('cv_contribution', 0.)))[:6]
    launch_key = lambda item: item[1]['trip']
    old_gaps, new_gaps = ({launch_key(x): x for x in rows} for rows in (bg, hg))
    gap_change = lambda trip: abs(new_gaps[trip][2] - old_gaps[trip][2])
    focus = sorted(old_gaps.keys() & new_gaps.keys(), key=gap_change, reverse=True)[:6]
    cycles = [cycle_rows(t, focus) for t in (bt, ht)]
    changes = [dict(trip=t, pooled_gap_s=old_gaps[t][2], hourly_gap_s=new_gaps[t][2],
                    pooled_predecessor=old_gaps[t][0]['trip'], hourly_predecessor=new_gaps[t][0]['trip'],
                    pooled=cycles[0][t], hourly=cycles[1][t]) for t in focus]
    report = dict(day=DAY, complete=True, physical_starts=0, training_updates=0, main_policy_changed=False,
                  first_divergence=divergence, pooled=b, hourly=h,
                  cv_parts_delta=difference(b['cv_parts'], h['cv_parts']),
                  cv_delta_by_station=difference(b['by_station'], h['by_station']),
                  cv_delta_by_arrival_hour=difference(b['by_hour'], h['by_hour']),
                  arrival_gap_pairs=len(common),
                  predecessor_changed=sum(base_arr[k]['predecessor'] != hour_arr[k]['predecessor'] for k in common),
                  largest_arrival_cv_increases=[dict(pooled=base_arr.get(k), hourly=hour_arr[k]) for k in largest],
                  largest_launch_gap_changes=changes, passenger=passenger_effect(br, hr))
    assert math.isclose(sum(report['cv_parts_delta'].values()), h['arrival']['cv2']-b['arrival']['cv2'], abs_tol=1e-12)
    output = root / 'divergence/summary/results.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, separators=(',', ':'), allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    audit(parser.parse_args().root)
