#!/usr/bin/env python3
"""Read V146 confirmation cells without changing its frozen crossed gates."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys


SUPPORT_FIELDS = ('requests', 'finite_n', 'finite_resolved_n', 'finite_unresolved_n',
    'unresolved_n', 'finite_resolved_episodes', 'label_statuses', 'support_reasons')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def check_support(stats):
    for scope, entry in stats.items():
        b, c = entry['base'], entry['candidate']
        if any(b[k] != c[k] for k in SUPPORT_FIELDS):
            raise AssertionError('native support or labels changed: '+scope)
        if scope in ('service_active', 'ready_all') and b != c:
            raise AssertionError('negative-control interval changed: '+scope)


def margins(cells, scope, axis):
    groups = defaultdict(list)
    for c in cells:
        groups[c[axis]].append(c['stats'][scope])
    rows = []
    for identity, entries in sorted(groups.items()):
        coverage = {arm: [e[arm]['episode_coverage'] for e in entries]
            for arm in ('base', 'candidate')}
        scores = [e['paired'] for e in entries]
        complete = all(v is not None for values in coverage.values() for v in values)
        score_complete = all(s['score_delta'] is not None for s in scores)
        rows.append(dict(identity=identity, cells=len(entries), coverage_complete=complete,
            contributing_cells=sum(v is not None for v in coverage['candidate']),
            base_coverage=mean(coverage['base']) if complete else None,
            candidate_coverage=mean(coverage['candidate']) if complete else None,
            base_score=mean([s['base_score'] for s in scores]) if score_complete else None,
            candidate_score=mean([s['candidate_score'] for s in scores]) if score_complete else None,
            score_delta=mean([s['score_delta'] for s in scores]) if score_complete else None))
    result = dict(groups=len(rows), complete_coverage_groups=sum(r['coverage_complete'] for r in rows))
    for key in ('base_coverage', 'candidate_coverage', 'base_score', 'candidate_score', 'score_delta'):
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = dict(contributing_groups=len(values),
            mean=mean(values) if values else None,
            minimum=min(values) if values else None, maximum=max(values) if values else None)
        if key.endswith('_coverage'):
            result[key]['below_80_n'] = sum(v < .8 for v in values)
        elif key == 'score_delta':
            result[key].update(improved_n=sum(v < 0 for v in values),
                unchanged_n=sum(v == 0 for v in values), worse_n=sum(v > 0 for v in values))
    return rows, result


def run(args):
    parent = read(args.v146_root/'summary/status.json')
    if parent['status'] != 'v146_independent_confirmation_complete':
        raise AssertionError('requires completed V146 confirmation')
    reg, cells = parent['registration'], []
    days, actors, modes = reg['days']['confirmation'], reg['actors'], reg['modes']
    for seed in actors:
        for day in days:
            for mode in modes:
                cell = read(args.v146_root/'confirmation/calibrated_cells'/f'{seed}_{day}_{mode}.json')
                if (cell['seed'], cell['day'], cell['mode']) != (seed, day, mode):
                    raise AssertionError('saved cell differs from registered actor/day/mode')
                check_support(cell['stats'])
                cells.append(cell)
    results, details = {}, {}
    for mode in modes:
        selected = [c for c in cells if c['mode'] == mode]
        results[mode], details[mode] = {}, {}
        for scope in reg['primary_scopes']:
            entry, detail = {}, {}
            for axis in ('seed', 'day'):
                detail[axis], entry[axis] = margins(selected, scope, axis)
            results[mode][scope], details[mode][scope] = entry, detail
    write(args.out_root/'details/marginals.json', details)
    write(args.out_root/'summary/status.json', dict(status='v146_readonly_confirmation_audit_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], paths=len(cells), results=results,
        native_support_and_labels_equal_in_every_scope=True,
        active_service_and_ready_negative_controls_identical=True,
        primary_joint_pass=parent['primary_joint_pass'], original_crossed_gate_replaced=False,
        new_rollouts=0, actor_updates=0, critic_updates=0, dual_updates=0,
        point_parameters_changed=False, main_policy_changed=False,
        limitations='Axis margins are descriptive, not per-group confidence guarantees or new gates. '
            'Missing coverage stays inconclusive, not zero or subset-averaged. Original episode/actor/day '
            'crossed confirmation remains authoritative. Confirmation days are consumed; no refitting.'))
    print('V146_READONLY_CONFIRMATION_AUDIT_COMPLETE', len(cells), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v146-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    args = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v146_confirmation_audit.py', '-v'], check=True)
    run(args)
