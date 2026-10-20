#!/usr/bin/env python3
"""Paired V156 history comparisons using saved cell summaries only."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import audit_freqduet_v156_clearance_support as audit


SCOPES = ('all', 'hour2', 'clearance', 'hour13', 'clearance_components')


def paired(cell):
    result = dict(seed=cell['seed'], day=cell['day'], groups={})
    for mode, arms in cell['modes'].items():
        for scope in SCOPES:
            full, single, none = [arms[a][scope] for a in audit.ARMS]
            if len({d['queries'] for d in (full, single, none)}) != 1:
                raise AssertionError('history arms have different query support')
            for field in full['metrics']:
                if field.startswith('reference_'):
                    for other in (single, none):
                        audit.close(full['metrics'][field], other['metrics'][field])
            audit.close(none['metrics']['delta_full_mae_s'], 0.)
            for arm in (full, single):
                if scope == 'clearance_components':
                    m = arm['metrics']
                    audit.close(m['delta_full_mae_s'], sum(m['delta_'+c+'_mae_s']
                        for c in ('running', 'service', 'residence'))-2*m['delta_cancellation_s'])
            fields = [f for f in full['metrics'] if f.startswith('delta_')]
            result['groups'][mode+'/'+scope] = dict(queries=full['queries'], metrics={
                f: single['metrics'][f]-full['metrics'][f] for f in fields})
        for name in ('full', 'single'):
            d = arms[name]['clearance_components']
            result['groups'][mode+'/'+name+'/clearance_budget'] = dict(queries=d['queries'],
                metrics=dict(cancellation_loss_s=-2*d['metrics']['delta_cancellation_s']))
    return result


def launch_cells(cell):
    result = []
    for mode, arms in cell['modes'].items():
        if 'clearance_launch' not in arms['full']:
            continue
        result.append(dict(seed=cell['seed'], day=cell['day'], mode=mode,
            arms={a: arms[a]['clearance_launch'] for a in audit.ARMS}))
    return result


def run(args):
    parent = audit.read(args.v156_root/'summary/status.json')
    if (parent['status'], parent['execution_source']) != (
            'v156_clearance_support_audit_complete', '8d1e201a13'):
        raise AssertionError('requires complete original V156')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v156_tradeoffs.py', '-v'], check=True)
    cells = [audit.read(p) for p in sorted((args.v156_root/'cells').glob('*.json'))]
    if len(cells) != parent['pairs']:
        raise AssertionError('incomplete V156 saved cells')
    results = audit.stats.aggregate([paired(c) for c in cells], lambda c: c['groups'])
    audit.write(args.out_root/'summary/status.json', dict(
        status='v156_saved_cell_tradeoffs_complete', execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        pairs=len(cells), paired_single_minus_full=results,
        clearance_launch_cells=[r for c in cells for r in launch_cells(c)],
        original_v155_gate=parent['original_v155_gate'], v155_gate_reinterpreted=False,
        new_forecasts=0, new_rollouts=0, calibration_refit=False, main_policy_changed=False,
        v151_model_changed=False, v146_changed=False, window_s=3600., tails=[.1, .1],
        limitations='Read-only consumed V156 cell contrasts, conditional on the fixed earliest '
            'single prior day. Negative single-minus-full MAE favors single. No claim about '
            'alternative prior days or an optimized history weight. Launch rows remain sparse '
            'descriptive cells, not independent crossed confirmation. Cancellation budgets use '
            'the same structured cohort, not opaque full-ETA endpoints. No predictor promotion.'))
    print('V156_SAVED_CELL_TRADEOFFS_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v156-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    run(p.parse_args())
