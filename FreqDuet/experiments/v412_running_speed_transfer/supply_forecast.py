"""Audit realized FIFO supply and decision-time clocks from paid V412 paths."""

from collections import Counter, defaultdict
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'source02')]
from circulation import reconcile
from study412 import CAP, DAYS, RULE, old, read, write
import numpy as np

FOCAL = 2520208
CHAIN = (225, 227, 236, 249, 260)


def fifo(trace, rows):
    launches = trace['dispatches']
    assert all(a['actual_s'] <= b['actual_s'] for a, b in zip(launches, launches[1:]))
    initial = {}
    for d in launches:
        initial.setdefault(d['bus'], d['direction'])
    assert len(initial) == 12
    latest, supply = {}, {}
    for d in launches:
        actual, direction, chosen = d['actual_s'], d['direction'], d['bus']
        ready, inbound = [], []
        for bus, start_direction in initial.items():
            pred = latest.get(bus)
            if pred is None:
                if start_direction == direction:
                    ready.append((None, bus, None))
            elif pred['direction'] != direction:
                rec = (pred['return_s'], bus, pred['trip'])
                (ready if rec[0] < actual else inbound).append(rec)
        ready.sort(key=lambda x: (-1 if x[0] is None else x[0], x[1]))
        used = next(r for r in ready if r[1] == chosen)
        assert used[0] == ready[0][0]
        row = rows[d['trip']]
        assert used[2] == row['predecessor']
        if used[0] is not None:
            assert used[0] == row['predecessor_return_s'] == d['ready_return_s']
        supply[d['trip']] = dict(actual_s=actual, eligible_s=d['eligible_s'], chosen=used,
                                ready=ready, still_inbound=sorted(inbound))
        latest[chosen] = row
    return initial, supply


def evidence(trace, rows):
    key = lambda t, b, p, s: (t, b, p, s)
    accepted = {key(g['time_s'], g['bus'], g['trip'], g['station']): g
                for g in trace['graph_queries'] if g['status'] == 'coordination_improvement'}
    next_trip = {r['predecessor']: r for r in rows.values() if r['predecessor'] is not None}
    counts, errors, accepted_keys = Counter(), defaultdict(list), set()
    focal = defaultdict(list)
    for p in trace['resource_predictions']:
        k = key(p['query_s'], p['bus'], p['trip'], p['station'])
        g = accepted.get(k)
        if g is None:
            counts['non_override_receipts'] += 1
            continue
        assert k not in accepted_keys and g['shadow_action_s'] == p['request_s']
        accepted_keys.add(k)
        for v in p['vehicles']:
            trip, bus, now = v['trip'], v['bus'], p['query_s']
            r = rows[trip]
            assert r['bus'] == bus and r['actual_s'] <= now <= r['return_s']
            n = next_trip.get(trip)
            target = {'return_s': r['return_s'], 'next_launch_s': n['actual_s'] if n else None}
            rec = dict(query_s=now, query_trip=p['trip'], query_bus=p['bus'], station=p['station'],
                       own=bus == p['bus'], actual_next_trip=n['trip'] if n else None,
                       actual=target, predicted=v['full'], model_reference=v['reference'])
            for metric, actual in target.items():
                if v['actual'].get(metric) != actual:
                    counts[f'{metric}_label_differences'] += 1
                est, ref = v['full'][metric], v['reference'][metric]
                if actual is None or est is None:
                    counts[f'{metric}_unpaired'] += 1
                    continue
                error = est - actual
                rec[metric + '_error'] = error
                lead = est - now
                bucket = '0_5m' if lead <= 300 else '5_15m' if lead <= 900 else 'over_15m'
                errors[metric, 'all'].append(error)
                errors[metric, bucket].append(error)
                if ref is not None and est > ref:
                    counts[f'{metric}_model_increases'] += 1
            if trip in CHAIN or (n and n['trip'] in CHAIN):
                focal[trip].append(rec)
    counts.update(accepted=len(accepted), accepted_with_receipt=len(accepted_keys))
    stats = {}
    for (metric, bucket), values in errors.items():
        a = np.asarray(values)
        stats.setdefault(metric, {})[bucket] = dict(n=len(a), bias_s=float(a.mean()),
            mae_s=float(np.abs(a).mean()), early_over_60s=int((a < -60).sum()))
    snapshots = {}
    for trip, records in focal.items():
        by_time = sorted(records, key=lambda x: x['query_s'])
        follow = next_trip.get(trip)
        before_due = [r for r in by_time if follow and r['query_s'] <= follow['eligible_s']]
        return_scored = [r for r in by_time if 'return_s_error' in r]
        launch_scored = [r for r in by_time if 'next_launch_s_error' in r]
        snapshots[trip] = dict(receipts=len(records), first=by_time[0], last=by_time[-1],
            last_by_next_eligibility=before_due[-1] if before_due else None,
            most_early_return=min(return_scored, key=lambda r: r['return_s_error'], default=None),
            most_early_launch=min(launch_scored, key=lambda r: r['next_launch_s_error'], default=None))
    return counts, stats, snapshots


def audit(root):
    agg = read(root / 'aggregate/summary/results.json')
    assert agg['complete'] and agg['physical_starts'] == 48
    archive = old.frozen.helper(old.frozen.ASSETS, 'gate_default')
    report = {}
    for day in DAYS['slow']:
        rows, supplies, initial = {}, {}, {}
        for arm in (CAP, RULE):
            path = root / 'cases/slow' / str(day) / arm
            trace = archive.raw(path / 'raw/path.pkl.gz')
            rows[arm] = reconcile(trace, read(path / 'summary/results.json'))
            initial[arm], supplies[arm] = fifo(trace, rows[arm])
            if arm == CAP:
                counts, stats, snapshots = evidence(trace, rows[arm])
        assert initial[CAP] == initial[RULE] and rows[CAP].keys() == rows[RULE].keys()
        c, b = rows[CAP], rows[RULE]
        ordered = sorted(c, key=lambda t: (c[t]['public_s'], t))
        switched = [t for t in ordered if c[t]['predecessor'] != b[t]['predecessor']]
        vehicle_switched = [t for t in ordered if c[t]['bus'] != b[t]['bus']]
        report[day] = dict(predecessor_changes=len(switched),
            first_predecessor_change=switched[0] if switched else None,
            first_vehicle_change=vehicle_switched[0] if vehicle_switched else None,
            forecast_counts=counts, forecast_stats=stats,
            readiness_delta_s=sum(c[t]['readiness_s']-b[t]['readiness_s'] for t in ordered))
        if day == FOCAL:
            t = 236
            selected = {*CHAIN, c[t]['predecessor'], b[t]['predecessor']}
            if switched:
                selected.add(switched[0])
            report[day].update(chain={arm: {str(t): dict(trip=rows[arm][t],
                fifo=supplies[arm][t]) for t in sorted(selected)} for arm in rows},
                bus10={arm: [dict(trip=t, actual_s=rows[arm][t]['actual_s'],
                    return_s=rows[arm][t]['return_s']) for t in ordered if rows[arm][t]['bus']==10]
                    for arm in rows}, chain_forecasts=snapshots)
    write(root / 'supply_forecast/summary/results.json', dict(complete=True,
        physical_starts=0, by_day=report, model_scope='first archived scenario; accepted overrides',
        forecast_error='selected first-return/same-bus next launch minus realized clock',
        limitation='No predicted next trip ID or unexecuted native outcome in archive.'))


if __name__ == '__main__':
    audit(Path(sys.argv[1]))
