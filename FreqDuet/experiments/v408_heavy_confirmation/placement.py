"""Read actual commands and same-state cap projections, not new counterfactuals."""

from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source01'))
from confirm import CANDIDATE, DAYS, REGIME, frozen
from coupled_model import project_native


def readout(root):
    read, write = frozen.study.read, frozen.study.write
    aggregate = read(root / 'aggregate/summary/results.json')
    assert aggregate['complete'] and aggregate['physical_starts'] == 48
    archive = frozen.helper(frozen.ASSETS, 'gate_default')
    days = {}
    stations = defaultdict(Counter)
    hours = defaultdict(Counter)
    for day in DAYS:
        path = root / 'cases' / REGIME / str(day) / CANDIDATE / 'raw/path.pkl.gz'
        trace = archive.raw(path)
        summary = aggregate['by_day'][str(day)]['arms'][CANDIDATE]
        totals, changes = Counter(), Counter()
        assert len(trace['queries']) == len(trace['graph_queries']) == 5200
        for q, g in zip(trace['queries'], trace['graph_queries']):
            assert all(q[k] == g[k] for k in ('trip', 'bus', 'station', 'time_s'))
            native, action = q['action_s'], q['executed_request_s']
            assert native == g['native_action_s'] and action == g['shadow_action_s']
            assert action in q['legal']
            d10 = action - project_native(native, q['legal'], 10)
            d15 = action - project_native(native, q['legal'], 15)
            changed = g['status'] == 'coordination_improvement'
            assert bool(d10) == changed
            group = 'censored' if g['flow'].get('censored_keys', 0) else 'other'
            counts = dict(queries=1, holding_s=action, above_10=int(action > 10),
                          add_vs_10=int(d10 > 0), cut_vs_10=int(d10 < 0),
                          delta_vs_10_s=d10, delta_vs_15_s=d15,
                          added_vs_10_s=max(d10, 0), cut_vs_10_s=max(-d10, 0))
            for total in (totals, stations[f"{int(q['direction'])}:{q['station']}"],
                          hours[str(int(q['time_s']) // 3600)]):
                total.update(counts)
            changes[f'{group}:{g["status"]}'] += 1
        assert totals['holding_s'] == round(262 * summary['metrics']['holding_s_per_trip'])
        days[day] = dict(counts=totals, query_status=changes, wall_s=summary['wall_s'],
                         callback_p95_s=summary['callback_p95_s'])

    totals = Counter()
    statuses = Counter()
    for day in days.values():
        totals.update(day['counts'])
        statuses.update(day['query_status'])
    write(root / 'placement/summary/results.json', dict(
        completed=True, physical_starts=0, totals=totals, query_status=statuses,
        by_day=days, by_direction_station=stations, by_hour=hours,
        interpretation='Cap projections use candidate states; not closed-loop baseline outcomes.'))


if __name__ == '__main__':
    readout(Path(sys.argv[1]))
