#!/usr/bin/env python3
"""Describe existing actor/day marginals without changing primary inference."""

import argparse
import json
from pathlib import Path


def describe(root):
    for axis in ('actor', 'day'):
        rows = json.loads((root/'details'/('by_'+axis+'.json')).read_text())
        for mode in ('old', 'handoff'):
            for scope in ('all', 'hour2', 'clearance', 'hour13'):
                selected = [r for r in rows if r['mode'] == mode and r['scope'] == scope]
                valid = [r for r in selected if all(r[m] is not None for m in
                    ('coverage', 'above', 'finite_fraction', 'score_delta_s'))]
                result = dict(axis=axis, mode=mode, scope=scope, groups=len(selected), complete_groups=len(valid))
                for metric, threshold, relation in (('coverage', .8, 'below'), ('above', .1, 'above'),
                        ('finite_fraction', .8, 'below'), ('score_delta_s', 0., 'above')):
                    values = [r[metric] for r in valid]
                    result[metric] = dict(min=min(values) if values else None, max=max(values) if values else None,
                        adverse_groups=sum(v < threshold if relation == 'below' else v > threshold for v in values))
                print('V168_MARGIN '+json.dumps(result), flush=True)
    print('V168_MARGINAL_READONLY_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-root', type=Path, required=True)
    describe(parser.parse_args().result_root)
