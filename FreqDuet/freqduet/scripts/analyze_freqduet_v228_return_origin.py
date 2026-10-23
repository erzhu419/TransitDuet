#!/usr/bin/env python3
"""Locate return-delay origins on existing V224/V225 physical vehicle chains."""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import time
import unittest

from analyze_freqduet_v221_execution import differences, read
from analyze_freqduet_v226_coordination import PAIRS, close, supply


PARTS = ('holding_s', 'nonholding_stop_s', 'interstop_nonholding_s',
         'launch_boundary_nonholding_s', 'return_boundary_nonholding_s')


def event_key(row):
    return row['trip_id'], row['station_id'], bool(row['direction'])


def trip_times(data, chain):
    arrivals, departures = {}, {}
    for kind, table in (('arrivals', arrivals), ('departures', departures)):
        for row in data['trace'][kind]:
            key = event_key(row)
            assert key not in table, 'repeated stop occurrence needs explicit matching'
            table[key] = row
            close(row['release'] + row['realized_holding'] + row['other_elapsed'],
                  row['time_s'], 'native event clock')
    assert arrivals.keys() == departures.keys(), 'unmatched arrival/departure, not zero service'
    grouped = defaultdict(list)
    for key, row in arrivals.items():
        grouped[key[0]].append(row)
    assert grouped.keys() == chain['returns'].keys() == chain['launches'].keys()
    trips = {}
    for trip, rows in grouped.items():
        ret, launch = chain['returns'][trip], chain['launches'][trip]
        assert ret['bus_id'] == launch['bus']
        close(ret['launch_s'], launch['actual_s'], 'trip launch identity')
        prev_s, prev_h = ret['launch_s'], 0.
        pieces, totals = [], Counter()
        for i, arrival in enumerate(sorted(rows, key=lambda r: r['time_s'])):
            departure = departures[event_key(arrival)]
            close(arrival['release'], ret['launch_s'], 'arrival release')
            close(departure['release'], ret['launch_s'], 'departure release')
            assert bool(arrival['direction']) == bool(launch['direction'])
            gap_hold = arrival['realized_holding'] - prev_h
            stop_hold = departure['realized_holding'] - arrival['realized_holding']
            gap = arrival['time_s'] - prev_s - gap_hold
            stop = departure['time_s'] - arrival['time_s'] - stop_hold
            assert min(gap_hold, stop_hold, gap, stop) >= 0., 'negative observed interval'
            gap_name = 'interstop_nonholding_s' if i else 'launch_boundary_nonholding_s'
            totals[gap_name] += gap
            totals['nonholding_stop_s'] += stop
            totals['holding_outside_stop_s'] += gap_hold
            pieces.append(dict(station_id=arrival['station_id'], arrival_s=arrival['time_s'],
                departure_s=departure['time_s'], gap_name=gap_name, gap_nonholding_s=gap,
                gap_holding_s=gap_hold, stop_nonholding_s=stop, stop_holding_s=stop_hold))
            prev_s, prev_h = departure['time_s'], departure['realized_holding']
        tail_hold = ret['holding_s'] - prev_h
        tail = ret['return_s'] - prev_s - tail_hold
        assert min(tail_hold, tail) >= 0., 'return precedes final observed departure'
        totals['return_boundary_nonholding_s'] = tail
        totals['holding_outside_stop_s'] += tail_hold
        totals['holding_s'] = ret['holding_s']
        totals['observed_stop_holding_s'] = math.fsum(p['stop_holding_s'] for p in pieces)
        close(totals['holding_s'], totals['observed_stop_holding_s'] +
              totals['holding_outside_stop_s'], 'located holding')
        close(math.fsum(totals[k] for k in PARTS[1:]), ret['other_elapsed_s'], 'other elapsed split')
        close(math.fsum(totals[k] for k in PARTS), ret['return_s']-ret['launch_s'], 'trip duration')
        trips[trip] = dict(trip_id=trip, bus_id=ret['bus_id'], direction=launch['direction'],
            launch_s=ret['launch_s'], return_s=ret['return_s'], totals=dict(totals), stops=pieces,
            return_boundary_holding_s=tail_hold)
    close(math.fsum(t['totals']['holding_s'] for t in trips.values()),
          data['ledger']['executed_holding_s'], 'all-trip holding ledger')
    return trips


def sign(value):
    return 'later' if value > 0 else 'earlier' if value < 0 else 'same'


def trip_difference(current, reference):
    assert (current['trip_id'], current['bus_id'], current['direction']) == (
        reference['trip_id'], reference['bus_id'], reference['direction'])
    assert [p['station_id'] for p in current['stops']] == [p['station_id'] for p in reference['stops']]
    parts = {k: current['totals'].get(k, 0.)-reference['totals'].get(k, 0.) for k in PARTS}
    launch = current['launch_s']-reference['launch_s']
    returned = current['return_s']-reference['return_s']
    duration = math.fsum(parts.values())
    close(launch+duration, returned, 'return difference')
    timeline, offset = [], launch
    first_positive = dict(kind='launch', station_id=None, offset_s=launch) if launch > 0 else None
    first_increase = None
    for c, b in zip(current['stops'], reference['stops']):
        row = dict(station_id=c['station_id'], gap_name=c['gap_name'])
        for name in ('gap_nonholding_s', 'gap_holding_s', 'stop_nonholding_s', 'stop_holding_s'):
            row[name] = c[name]-b[name]
        offset += row['gap_nonholding_s']+row['gap_holding_s']
        close(offset, c['arrival_s']-b['arrival_s'], 'arrival delta propagation')
        row['arrival_delta_s'] = offset
        if first_positive is None and offset > 0:
            first_positive = dict(kind='arrival', station_id=c['station_id'], offset_s=offset)
        offset += row['stop_nonholding_s']+row['stop_holding_s']
        close(offset, c['departure_s']-b['departure_s'], 'departure delta propagation')
        row['departure_delta_s'] = offset
        if first_positive is None and offset > 0:
            first_positive = dict(kind='departure', station_id=c['station_id'], offset_s=offset)
        if first_increase is None and any(row[k] > 0 for k in
                ('gap_nonholding_s', 'gap_holding_s', 'stop_nonholding_s', 'stop_holding_s')):
            first_increase = dict(row)
        timeline.append(row)
    tail_hold = current['return_boundary_holding_s']-reference['return_boundary_holding_s']
    close(offset+parts['return_boundary_nonholding_s']+tail_hold, returned, 'return boundary propagation')
    if first_positive is None and returned > 0:
        first_positive = dict(kind='return', station_id=None, offset_s=returned)
    return dict(trip_id=current['trip_id'], bus_id=current['bus_id'],
        reference_launch_s=reference['launch_s'], reference_return_s=reference['return_s'],
        launch_delta_s=launch, return_delta_s=returned, new_duration_delta_s=duration,
        **parts, first_positive_clock=first_positive, first_increase=first_increase, stops=timeline)


def aggregate(rows):
    fields = ('launch_delta_s', 'return_delta_s', 'new_duration_delta_s', *PARTS)
    return dict(trips=len(rows), **{k: math.fsum(r[k] for r in rows) for k in fields})


def short(row):
    return {k: v for k, v in row.items() if k not in ('stops', 'first_increase')}


def compare(c, b, c_chain, b_chain):
    assert c.keys() == b.keys()
    rows = {t: trip_difference(c[t], b[t]) for t in c}
    groups = defaultdict(list)
    for row in rows.values():
        groups['return_'+sign(row['return_delta_s'])+'__launch_'+sign(row['launch_delta_s'])].append(row)
    buses = defaultdict(list)
    for trip in c_chain['launches']:
        buses[c[trip]['bus_id']].append(trip)
    origins, delayed, chain_details = [], [], []
    for bus, trips in buses.items():
        assert trips == sorted(trips, key=lambda t: c[t]['launch_s']), 'native chain not chronological'
        first_return, first_new = None, None
        initial, accumulated_duration, accumulated_idle, previous = None, 0., 0., None
        for trip in trips:
            row = rows[trip]
            parent = c_chain['parents'][trip]
            assert parent == b_chain['parents'][trip] == previous, 'physical supply chain changed'
            if previous is None:
                initial = row['launch_delta_s']
                idle = 0.
            else:
                idle = row['launch_delta_s']-rows[previous]['return_delta_s']
                accumulated_idle += idle
                if row['launch_delta_s'] > 0:
                    parent_row = rows[previous]
                    delayed.append(dict(child_trip_id=trip, actual_launch_delta_s=row['launch_delta_s'],
                        terminal_idle_delta_s=idle, **short(parent_row)))
            close(initial+accumulated_duration+accumulated_idle,
                  row['launch_delta_s'], 'nonduplicated prefix launch identity')
            accumulated_duration += row['new_duration_delta_s']
            close(initial+accumulated_duration+accumulated_idle,
                  row['return_delta_s'], 'nonduplicated prefix return identity')
            chain_details.append(dict(bus_id=bus, trip_id=trip, previous_trip_id=previous,
                initial_launch_delta_s=initial, accumulated_new_duration_s=accumulated_duration,
                accumulated_terminal_idle_s=accumulated_idle, terminal_idle_delta_s=idle))
            if first_return is None and row['return_delta_s'] > 0:
                first_return = short(row)
            if first_new is None and row['new_duration_delta_s'] > 0:
                first_new = short(row)
            previous = trip
        origins.append(dict(bus_id=bus, first_positive_return=first_return,
                            first_positive_trip_duration=first_new))
    first = sorted((o['first_positive_return'] for o in origins if o['first_positive_return']),
                   key=lambda r: r['reference_return_s'])
    new = sorted((o['first_positive_trip_duration'] for o in origins if o['first_positive_trip_duration']),
                 key=lambda r: r['reference_launch_s'])
    summary = dict(all_trips=aggregate(list(rows.values())),
        groups={k: aggregate(v) for k,v in sorted(groups.items())}, physical_buses=len(buses),
        buses_with_positive_return=len(first), buses_with_positive_new_duration=len(new),
        first_positive_returns=first[:3], first_positive_new_durations=new[:3],
        delayed_same_chain_launches=dict(count=len(delayed),
            launch_delta_s=math.fsum(r['actual_launch_delta_s'] for r in delayed),
            terminal_idle_delta_s=math.fsum(r['terminal_idle_delta_s'] for r in delayed),
            inbound=aggregate(delayed)))
    close(summary['delayed_same_chain_launches']['launch_delta_s'],
          summary['delayed_same_chain_launches']['inbound']['return_delta_s']+
          summary['delayed_same_chain_launches']['terminal_idle_delta_s'], 'delayed cohort supply identity')
    return summary, dict(trips=list(rows.values()), origins=origins,
                         physical_chain_prefixes=chain_details, delayed_launches=delayed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('joint-root', 'controls-root', 'v226-root', 'v227-root', 'out-root'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                              pattern='test_v228_return_origin.py')
    assert suite.countTestCases() > 0
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tests.wasSuccessful():
        raise SystemExit(1)
    protocol = dict(stage='v228_read_only_return_origin', source=os.environ['FREQDUET_ANALYSIS_COMMIT'],
        controller_source='f61f6c9734', new_replays=0, training_updates=0, parameter_changes=0,
        main_policy_changed=False, decomposition=list(PARTS),
        scope='All 12 existing V224/V225 paths; physical-chain identities, no causal mediation. '
        'Launch/return boundaries stay explicit. Nonholding stop residence is not pure passenger '
        'service. Interstop elapsed is not a continuous-motion travel-time label. '
        'First positive return and first positive trip-duration increment are separate; '
        'ancestral contributions are never summed over descendants as independent causes.')
    a.out_root.mkdir(parents=True, exist_ok=True)
    assert not (a.out_root/'protocol.json').exists(), 'preserve completed audit'
    (a.out_root/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    old = read(a.v226_root/'summary/status.json')
    wait = read(a.v227_root/'summary/status.json')
    assert wait['status'] == 'v227_terminal_wait_complete'
    assert len(old['contexts']) == len(wait['contexts']) == 4
    summaries, start = [], time.monotonic()
    (a.out_root/'details').mkdir(exist_ok=True)
    for saved, waiting in zip(old['contexts'], wait['contexts']):
        ci = saved['context']
        assert ci == waiting['context']
        data = {arm: read((a.joint_root if arm == 'joint' else a.controls_root)/'records'/
                f'c{ci}_{arm}.json.gz') for arm in ('v107', 'upper', 'joint')}
        chains = {arm: supply(d) for arm,d in data.items()}
        trips = {arm: trip_times(d, chains[arm]) for arm,d in data.items()}
        comparisons, details = {}, {}
        for current, reference in PAIRS:
            name = current+'_vs_'+reference
            row, detail = compare(trips[current], trips[reference], chains[current], chains[reference])
            previous = saved['comparisons'][name]
            effect = differences(data[current]['results'], data[reference]['results'])
            assert effect == previous['effects'], 'V226 endpoint changed'
            close(row['all_trips']['holding_s'], previous['holding']['total_holding_delta_s'], 'V226 holding')
            close(row['all_trips']['launch_delta_s'],
                  previous['supply_chain']['totals']['actual_launch_delta_s'], 'V226 actual launches')
            delayed = waiting['comparisons'][name]['launch_return_groups']
            # V227 groups include same-parent chains only, as does this cohort.
            cohort = [v for k,v in delayed.items() if k.startswith('launch_later/')]
            count = sum(v['chains'] for v in cohort)
            close(row['delayed_same_chain_launches']['count'], count, 'V227 delayed launch count')
            for field in ('launch_delta_s', 'terminal_idle_delta_s'):
                old_field = 'idle_delta_s' if field == 'terminal_idle_delta_s' else field
                close(row['delayed_same_chain_launches'][field], sum(v[old_field] for v in cohort),
                      'V227 delayed cohort '+field)
            row['effects'] = effect
            comparisons[name], details[name] = row, detail
        (a.out_root/'details'/f'c{ci}.json').write_text(json.dumps(dict(paths=trips, comparisons=details), indent=2)+'\n')
        summaries.append(dict(context=ci, actor=saved['actor'], demand_day=saved['demand_day'],
            path_support={arm: dict(trips=len(t), stops=sum(len(r['stops']) for r in t.values()),
                holding_outside_stop_s=sum(r['totals']['holding_outside_stop_s'] for r in t.values()))
                for arm,t in trips.items()}, comparisons=comparisons))
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(summaries), total=4,
            unit='context', elapsed_seconds=time.monotonic()-start)), flush=True)
    summary = dict(protocol, status='v228_return_origin_complete', tests=tests.testsRun,
        contexts=summaries, v226_endpoints_and_v227_delayed_cohort_reproduced=True)
    (a.out_root/'summary').mkdir(exist_ok=True)
    (a.out_root/'summary/status.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('V228_RETURN_ORIGIN_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
