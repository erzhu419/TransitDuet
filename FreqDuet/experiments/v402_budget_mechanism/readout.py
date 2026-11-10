"""Describe the sealed aggregate without new fitting, tests or traffic."""

import argparse
from pathlib import Path
import statistics

from worker import read, write


def main(root):
    result = read(root / 'aggregate/summary/results.json')
    days = list(result['by_day'].values())
    arms = list(days[0])
    metrics = list(result['effects']['static'])
    means = {arm: {m: statistics.mean(day[arm][m] for day in days)
                   for m in metrics} for arm in arms}
    comparisons = {}
    for family, components in result['sealed_components'].items():
        mixture = {m: sum(c['probability'] * means[c['arm']][m]
                          for c in components) for m in metrics}
        directions = {}
        for metric, effect in result['effects'][family].items():
            values = list(effect['by_day'].values())
            directions[metric] = dict(lower=sum(v < 0 for v in values),
                                      equal=sum(v == 0 for v in values),
                                      higher=sum(v > 0 for v in values))
        comparisons[family] = dict(
            mixture_means=mixture, directions=directions,
            relative_effect_percent={m: 100 * result['effects'][family][m]['mean'] / mixture[m]
                                     for m in metrics if mixture[m] != 0})
    execution = {}
    for arm in arms:
        rows = [r for key, r in result['execution'].items() if key.startswith(arm + '_d')]
        execution[arm] = dict(paths=len(rows), all_physical=all(r['physical'] for r in rows),
            mean_executed_holding_s_per_day=statistics.mean(r['executed_holding_s'] for r in rows),
            mean_supported_queries=statistics.mean(r['supported'] for r in rows),
            mean_overrides=statistics.mean(r['overrides'] for r in rows),
            mean_planned_changes=statistics.mean(r['planned_changes'] for r in rows))
    output = dict(completed=True, original_gate=result['decisions'],
                  overall_passed=result['passed'], days=len(days), means=means,
                  comparisons=comparisons, execution=execution,
                  physical_starts=0, training_updates=0, main_policy_changed=False)
    write(root / 'readout/summary/results.json', output)
    print('V402_READOUT_COMPLETE', result['passed'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
