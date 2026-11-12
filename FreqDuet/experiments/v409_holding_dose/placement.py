"""Measure actual action changes against native fixed10 on the same states."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source02'))
from study409 import CAP, DAYS, DOSE, FULL, REGIME, frozen, project_native, read, write


def audit(root):
    agg = read(root / 'aggregate/summary/results.json')
    assert agg['complete'] and agg['physical_starts'] == 82
    archive = frozen.helper(frozen.ASSETS, 'gate_default')
    report = {}
    for arm in (FULL, CAP):
        by_day, transitions, totals = {}, Counter(), Counter()
        for day in DAYS:
            out = root / 'cases' / REGIME / str(day) / arm
            trace = archive.raw(out / 'raw/path.pkl.gz')
            counts, table = Counter(), Counter()
            assert len(trace['queries']) == len(trace['graph_queries']) == 5200
            for q, g in zip(trace['queries'], trace['graph_queries']):
                assert all(q[k] == g[k] for k in ('trip', 'bus', 'station', 'time_s'))
                native, action = q['action_s'], q['executed_request_s']
                assert native == g['native_action_s'] and action == g['shadow_action_s']
                assert action in q['legal'] and (arm == FULL or action <= 10)
                ref = project_native(native, q['legal'], 10)
                changed = ref != action
                assert changed == (g['status'] == 'coordination_improvement')
                counts.update(queries=1, changed=int(changed), increased=int(action > ref),
                    decreased=int(action < ref), above_10=int(action > 10),
                    holding_s=action, ref_same_state_s=ref, added_s=max(0, action-ref),
                    removed_s=max(0, ref-action),
                    changed_censored=int(changed and bool(g['flow'].get('censored_keys'))))
                table[f'{ref}->{action}'] += 1
            metrics = agg['by_day'][str(day)]['arms'][arm]['metrics']
            assert counts['holding_s'] == round(262 * metrics[DOSE])
            totals.update(counts)
            transitions.update(table)
            by_day[day] = counts
        report[arm] = dict(totals=totals, transitions=transitions, by_day=by_day)
    write(root / 'placement/summary/results.json', dict(complete=True, physical_starts=0,
          reference='fixed10 projection on each executed candidate state', arms=report))


if __name__ == '__main__':
    audit(Path(sys.argv[1]))
