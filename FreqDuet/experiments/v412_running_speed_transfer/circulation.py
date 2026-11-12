"""Decompose archived departure differences; no simulation or policy calls."""

from collections import Counter, defaultdict
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source02'))
from study412 import ARMS, CAP, DAYS, RULE, old, read, write

FIXED = old.FIXED
TRIPS = 262
COMPONENTS = ('previous_launch_s', 'previous_holding_s', 'previous_other_cycle_s',
              'terminal_gap_s', 'negative_eligibility_s')


def reconcile(trace, receipt):
    commands = Counter()
    for q in trace['queries']:
        commands[q['trip']] += q['executed_request_s']
    launches = sorted(trace['dispatches'], key=lambda d: (d['actual_s'], d['trip']))
    returns = {(r['bus_id'], r['trip_id']): r['physical_time_s'] for r in trace['returns']}
    assert len(launches) == len(returns) == TRIPS
    assert len(trace['queries']) == 5200 and receipt['physical_account']['passed']
    assert set(commands) <= {d['trip'] for d in launches}
    previous, rows = {}, {}
    for d in launches:
        trip, bus, actual = d['trip'], d['bus'], d['actual_s']
        eligible, public = d['eligible_s'], d['public_s']
        assert public == d['scheduled_s'] <= eligible <= actual
        r = dict(d, readiness_s=actual-eligible, public_delay_s=actual-public,
                 eligibility_delay_s=eligible-public, return_s=returns[bus, trip],
                 holding_s=commands[trip], predecessor=None)
        if bus in previous:
            pred = previous[bus]
            gap = actual - pred['return_s']
            cycle = pred['return_s'] - pred['actual_s']
            assert gap >= 0 and cycle >= pred['holding_s']
            r.update(predecessor=pred['trip'], predecessor_return_s=pred['return_s'],
                     return_adjacent=gap <= 1,
                     previous_launch_s=pred['actual_s'], previous_holding_s=pred['holding_s'],
                     previous_other_cycle_s=cycle-pred['holding_s'], terminal_gap_s=gap,
                     negative_eligibility_s=-eligible)
            assert sum(r[k] for k in COMPONENTS) == r['readiness_s']
        previous[bus] = rows[trip] = r
    assert len(rows) == TRIPS and len(previous) == 12
    m = receipt['endpoints']
    for key, endpoint, scale in (('holding_s', 'holding_s_per_trip', TRIPS),
                                 ('readiness_s', 'readiness_s_per_trip', TRIPS),
                                 ('public_delay_s', 'public_delay_s_per_day', 1)):
        assert math.isclose(sum(d[key] for d in rows.values()), scale*m[endpoint], abs_tol=1e-8)
    return rows


def compare(candidate, reference):
    assert candidate.keys() == reference.keys()
    all_rows, hours, groups = [], defaultdict(Counter), defaultdict(Counter)
    for trip, c in candidate.items():
        b = reference[trip]
        assert (c['public_s'], c['direction']) == (b['public_s'], b['direction'])
        delta = {k: c[k]-b[k] for k in ('readiness_s', 'public_delay_s', 'eligibility_delay_s')}
        assert delta['readiness_s'] + delta['eligibility_delay_s'] == delta['public_delay_s']
        if c['predecessor'] is None or b['predecessor'] is None:
            group = 'initial_pool_in_either'
        elif c['predecessor'] == b['predecessor']:
            group = 'same_predecessor'
            delta.update({k: c[k]-b[k] for k in COMPONENTS})
            assert sum(delta[k] for k in COMPONENTS) == delta['readiness_s']
        else:
            group = 'different_predecessor'
        d = delta['readiness_s']
        counts = dict(trips=1, later=int(d > 0), earlier=int(d < 0),
                      positive_readiness_s=max(d, 0), negative_readiness_s=min(d, 0),
                      changed_vehicle=int(c['bus'] != b['bus']),
                      candidate_return_adjacent=int(c.get('return_adjacent', False)),
                      reference_return_adjacent=int(b.get('return_adjacent', False)))
        for tally in (groups[group], hours[str(int(c['public_s'])//3600)]):
            tally.update(counts)
            tally.update(delta)
        all_rows.append(dict(trip=trip, public_s=c['public_s'], direction=c['direction'],
                             group=group, delta=delta, candidate=c, reference=b))
    by_delay = sorted(all_rows, key=lambda r: r['delta']['readiness_s'], reverse=True)
    total = Counter()
    for values in groups.values():
        total.update(values)
    return dict(totals=total, by_hour=hours, by_predecessor=groups,
                top_later=[r for r in by_delay[:5] if r['delta']['readiness_s'] > 0],
                top_earlier=[r for r in by_delay[:-6:-1] if r['delta']['readiness_s'] < 0])


def audit(root):
    agg = read(root / 'aggregate/summary/results.json')
    assert agg['complete'] and agg['physical_starts'] == 48
    archive = old.frozen.helper(old.frozen.ASSETS, 'gate_default')
    by_day = {}
    for day in DAYS['slow']:
        rows = {}
        for arm in ARMS:
            path = root / 'cases/slow' / str(day) / arm
            rows[arm] = reconcile(archive.raw(path / 'raw/path.pkl.gz'),
                                  read(path / 'summary/results.json'))
        by_day[day] = {}
        for ref in (RULE, FIXED):
            name = f'{CAP}_minus_{ref}'
            result = compare(rows[CAP], rows[ref])
            t = result['totals']
            effects = agg['regimes']['slow']['effects'][name]
            assert math.isclose(t['readiness_s'], TRIPS *
                   effects['readiness_s_per_trip']['by_day'][str(day)], abs_tol=1e-8)
            assert t['public_delay_s'] == effects['public_delay_s_per_day']['by_day'][str(day)]
            by_day[day][ref] = result
    write(root / 'circulation/summary/results.json', dict(complete=True, physical_starts=0,
          contrast='capped V406 minus each reference, slow setting only', by_day=by_day,
          interpretation='Exact realized-path accounting; predecessor changes are not causal attribution.'))


if __name__ == '__main__':
    audit(Path(sys.argv[1]))
