"""Read existing development geometry and saved quota accounts; no traffic."""

import argparse
import itertools
from pathlib import Path
import statistics

from worker import read, write


def main(root):
    seal = read(root / 'selection/summary/results.json')
    days = list(seal['calibration'].values())
    metrics = list(days[0]['pressure'])
    means = {a: {m: statistics.mean(d[a][m] for d in days) for m in metrics}
             for a in days[0]}
    target = means['pressure']['cv_squared']
    frontier = {}
    for family in ('static', 'cap'):
        arms = [c['arm'] for c in seal['families'][family]['cases']]
        points = [dict(components={a: 1.}, means=means[a]) for a in arms
                  if means[a]['cv_squared'] <= target]
        for a, b in itertools.combinations(arms, 2):
            ca, cb = means[a]['cv_squared'], means[b]['cv_squared']
            if ca == cb or not min(ca, cb) <= target <= max(ca, cb):
                continue
            w = (target - cb) / (ca - cb)
            mixed = {m: w * means[a][m] + (1 - w) * means[b][m] for m in metrics}
            points.append(dict(components={a: w, b: 1 - w}, means=mixed))
        best = min(points, key=lambda r: r['means']['public_delay_s_per_day'])
        frontier[family] = dict(
            feasible_vertices=len(points),
            minimum_calibration_delay_at_pressure_mean_cv2=best,
            differences={m: means['pressure'][m] - best['means'][m] for m in metrics},
            interpretation='development geometry only; not a sealed policy or independent confirmation')
    aggregate = read(root / 'aggregate/summary/results.json')
    accounts = {}
    for arm in next(iter(aggregate['by_day'].values())):
        rows = [read(root / 'confirmation' / arm / f'd{day}' / 'summary/results.json')
                for day in aggregate['by_day']]
        accounts[arm] = dict(
            quota_probes={k: sum(r['quota_probes'][k] for r in rows)
                          for k in rows[0]['quota_probes']},
            mean_budget_issues=statistics.mean(r['slow']['budget_issues'] for r in rows),
            mean_budget_tightenings=statistics.mean(r['slow']['budget_tightenings'] for r in rows),
            mean_budgeted_commands=statistics.mean(r['slow']['budgeted_commands'] for r in rows))
    result = dict(completed=True, calibration_means=means, calibration_frontier=frontier,
                  confirmation_accounts=accounts, physical_starts=0, training_updates=0,
                  protocol_changed=False, main_policy_changed=False,
                  independent_test_used_for_fitting=False)
    write(root / 'decision_review/summary/results.json', result)
    print('V402_DECISION_REVIEW_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
