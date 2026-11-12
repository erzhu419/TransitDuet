"""Read-only V412 return-order and public-slot pressure screen."""

from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from supply_forecast import CAP, DAYS, FOCAL, RULE, old, read, reconcile, write

LOOKBACK_S = 300  # Existing near-term forecast reporting window.
GAP_S = 30  # Original holding action upper bound, not fit to these days.


def order(predicted, realized):
    values = sorted(predicted.values())
    leading = {v for v, eta in predicted.items() if eta == values[0]}
    first_actual = {v for v, clock in realized.items() if clock == min(realized.values())}
    return dict(predicted_first=sorted(leading), realized_first=sorted(first_actual),
        wrong=not leading & first_actual, tie=len(leading) > 1,
        gap_s=values[1] - values[0] if len(values) > 1 else None)


def screen(trace, rows, reference, focal=False):
    # The only time target used for scoring the forecast is the public timetable.
    assert all(r['public_s'] == reference[t]['public_s'] and
               r['direction'] == reference[t]['direction'] for t, r in rows.items())
    by_direction, latest = defaultdict(list), {}
    launches = sorted(rows.values(), key=lambda r: (r['actual_s'], r['trip']))
    for r in sorted(rows.values(), key=lambda r: (r['public_s'], r['trip'])):
        by_direction[r['direction']].append(r)
    launch_ticks = {r['actual_s'] for r in launches}
    return_ticks = {r['return_s'] for r in launches}
    graph = {(q['time_s'], q['bus'], q['trip'], q['station']): q for q in trace['graph_queries']}
    counts, groups, snapshots, focal_pair = Counter(), defaultdict(Counter), {}, []
    cursor, heads, receipts = 0, Counter(), set()
    for p in sorted(trace['resource_predictions'], key=lambda q: q['query_s']):
        now = p['query_s']
        identity = (now, p['bus'], p['trip'], p['station'])
        assert identity not in receipts
        receipts.add(identity)
        g = graph[identity]
        if g['status'] != 'coordination_improvement':
            continue
        assert g['shadow_action_s'] == p['request_s']
        counts['accepted_receipts'] += 1
        while cursor < len(launches) and launches[cursor]['actual_s'] < now:
            r = launches[cursor]
            assert r['trip'] == by_direction[r['direction']][heads[r['direction']]]['trip']
            heads[r['direction']] += 1
            latest[r['bus']] = r
            cursor += 1
        if len(latest) != 12 or now in launch_ticks or now in return_ticks:
            counts['excluded_initial_or_same_tick'] += 1
            continue
        predicted = {v['trip']: v['full']['return_s'] for v in p['vehicles']}
        native = {v['trip']: v['reference']['return_s'] for v in p['vehicles']}
        assert len(predicted) == len(p['vehicles'])
        for v in p['vehicles']:
            assert latest[v['bus']]['trip'] == v['trip'] and rows[v['trip']]['return_s'] > now
        if (focal and {212, 214} <= predicted.keys() and
                all(predicted[t] is not None and native[t] is not None for t in (212, 214))):
            focal_pair.append(dict(query_s=now, full_gap_212_minus_214_s=predicted[212]-predicted[214],
                native_gap_212_minus_214_s=native[212]-native[214]))
        for direction, schedule in by_direction.items():
            index = heads[direction]
            if index == len(schedule):
                continue
            slot = schedule[index]
            due = slot['public_s']
            if not 0 <= due - now <= LOOKBACK_S:
                continue
            trip = slot['trip']
            if trip in snapshots and snapshots[trip]['query_s'] == now:
                continue  # Stable first receipt at the same clock.
            supply = [r for r in latest.values() if r['direction'] != direction]
            ready = any(r['return_s'] < now for r in supply)
            cohort = {r['trip']: r['return_s'] for r in supply if r['return_s'] > now}
            valid = {t: predicted[t] for t in cohort if predicted.get(t) is not None}
            assert all(eta >= now for eta in valid.values())
            snapshots[trip] = dict(query_s=now, public_s=due, ready=ready, realized=cohort,
                                   predicted=valid, pred_trip=slot['predecessor'])

    counts['public_slots'] = len(rows)
    counts['all_predecessor_switches'] = sum(
        r['predecessor'] != reference[t]['predecessor'] for t, r in rows.items())
    counts['with_snapshot'] = len(snapshots)
    for trip, s in snapshots.items():
        realized, predicted, due = s['realized'], s['predicted'], s['public_s']
        if s['ready']:
            counts['already_ready'] += 1
            continue
        if not realized or realized.keys() != predicted.keys():
            counts['no_complete_inbound_forecast'] += 1
            continue
        counts['complete_without_ready'] += 1
        rank = order(predicted, realized)
        # Return at the public clock itself cannot supply a launch at that tick.
        predicted_late = min(predicted.values()) >= due
        realized_late = min(realized.values()) >= due
        actual_slot_late = rows[trip]['readiness_s'] > 0
        rank_group = ('single' if rank['gap_s'] is None else
                      'gap_le_30' if rank['gap_s'] <= GAP_S else 'gap_gt_30')
        switched = rows[trip]['predecessor'] != reference[trip]['predecessor']
        values = dict(n=1, rank_wrong=int(rank['wrong']), predicted_tie=int(rank['tie']),
            predecessor_switch=int(switched), inbound_supplied=int(s['pred_trip'] in realized),
            earlier_clock_false_alarm=int(predicted_late and not realized_late),
            earlier_clock_missed_pressure=int(realized_late and not predicted_late),
            cohort_not_ready_by_public=int(realized_late), slot_readiness_delay=int(actual_slot_late))
        for k in ('all', rank_group, 'forecast_late' if predicted_late else 'forecast_on_time'):
            groups[k].update(values)
    assert sum(counts[k] for k in ('already_ready', 'no_complete_inbound_forecast',
                                   'complete_without_ready')) == len(snapshots)
    return dict(counts=counts, strata=groups), focal_pair


def audit(root):
    aggregate = read(root / 'aggregate/summary/results.json')
    assert aggregate['complete'] and aggregate['physical_starts'] == 48
    archive = old.frozen.helper(old.frozen.ASSETS, 'gate_default')
    by_day, focal = {}, {}
    for day in DAYS['slow']:
        rows = {}
        for arm in (RULE, CAP):
            path = root / 'cases/slow' / str(day) / arm
            trace = archive.raw(path / 'raw/path.pkl.gz')
            rows[arm] = reconcile(trace, read(path / 'summary/results.json'))
        by_day[day], pair = screen(trace, rows[CAP], rows[RULE], day == FOCAL)
        if day == FOCAL:
            clocks = {arm: {t: rows[arm][t]['return_s'] for t in (212, 214)} for arm in rows}
            due = rows[CAP][225]['public_s']
            before = [q for q in pair if q['query_s'] <= due]
            focal = dict(return_clocks=clocks, trip225_public_s=due, paired_receipts=len(pair),
                pre_public_receipts=len(before), first=pair[0] if pair else None,
                last_before_public=before[-1] if before else None, last=pair[-1] if pair else None,
                full_wrong_before_public=sum(q['full_gap_212_minus_214_s'] < 0 for q in before),
                native_wrong_before_public=sum(q['native_gap_212_minus_214_s'] < 0 for q in before))
    write(root / 'supply_order/summary/results.json', dict(complete=True, physical_starts=0,
          query_lookback_s=LOOKBACK_S, holding_gap_s=GAP_S, by_day=by_day, focal=focal,
          scope='Accepted overrides, first model scenario. Latest pre-public query per head slot.'))


if __name__ == '__main__':
    audit(Path(sys.argv[1]))
