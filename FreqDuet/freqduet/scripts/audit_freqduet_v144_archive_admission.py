#!/usr/bin/env python3
"""Inventory existing server archives without scoring forecasts or replaying."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ARTIFACTS = {'traces', 'launch_traces', 'records', 'ready', 'ready_labels', 'terminal'}
META_KEYS = {'paths', 'pairs', 'cases', 'contexts', 'items', 'demand_days', 'days',
    'evaluation_days', 'eval_seeds', 'evaluation_seeds', 'demand_seeds', 'cohorts'}
QUERY_FIELDS = ('trip_id', 'station_id', 'direction', 'policy_time_s', 'chosen_s', 'executed_s')
EVIDENCE_FIELDS = ('ready_time_s', 'bus_id', 'follower_source',
    'predicted_follower_trip_id', 'predicted_follower_bus_id')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def metadata_days(value):
    days = set()
    if isinstance(value, dict):
        if isinstance(value.get('day'), int):
            days.add(value['day'])
        for k, v in value.items():
            if k in META_KEYS:
                if isinstance(v, list) and all(isinstance(x, int) for x in v):
                    days.update(v)
                else:
                    days.update(metadata_days(v))
    elif isinstance(value, list):
        for v in value:
            days.update(metadata_days(v))
    return days


def result_roots(parents):
    found = set()
    for parent in parents:
        if not parent.is_dir():
            continue
        for project in parent.iterdir():
            if 'FreqDuet' not in project.name and project.name != 'TransitDuet':
                continue
            for relative in ('FreqDuet/freqduet/results_freqduet', 'freqduet/results_freqduet', 'results_freqduet'):
                root = project/relative
                if root.is_dir():
                    found.add(root.resolve())
    return sorted(found)


def artifact_bases(experiment):
    # Current recorders write at out_root or its immediate smoke/shard child.
    candidates = [experiment, *(p for p in experiment.iterdir() if p.is_dir() and p.name != 'source')]
    return [p for p in candidates if any((p/d).is_dir() for d in ARTIFACTS)]


def trace_schema(path):
    count = 0
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            missing = [k for k in QUERY_FIELDS if k not in r]
            missing += ['evidence.'+k for k in EVIDENCE_FIELDS if k not in r.get('evidence', {})]
            missing += [k+'.'+v for k, v in (('online', 'gap_s'), ('original_prediction', 'status'))
                if v not in r.get(k, {})]
            if missing:
                return dict(complete=False, rows_checked=count+1, missing=missing)
            count += 1
    return dict(complete=count > 0, rows_checked=count, missing=[] if count else ['empty_trace'])


def ready_schema(path):
    count = 0
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            missing = [k for k in ('trip_id', 'station_id', 'direction', 'bus_id', 'ready_time_s') if k not in r]
            if missing:
                return dict(complete=False, rows_checked=count+1, missing=missing)
            count += 1
    return dict(complete=count > 0, rows_checked=count, missing=[] if count else ['empty_ready_labels'])


def case_id(name):
    parts = name.split('_')
    if len(parts) < 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), '_'.join(parts[2:])
    except ValueError:
        return None


def inspect_bundle(base, name, exposures):
    identity = case_id(name)
    row = dict(base=str(base), name=name, identity=identity, missing=[], schema={})
    files = dict(trace=base/'traces'/(name+'.jsonl.gz'),
        ready=base/'ready_labels'/(name+'.jsonl.gz'), record=base/'records'/(name+'.npz'),
        terminal=base/'terminal'/(name+'.json'))
    # V137 deliberately reuses V135 terminal clocks after exact passive replay.
    if base.name == 'protocol_v6_v137_passive_ready_exposure':
        receipt = base/'paths'/(name+'.json')
        if receipt.is_file() and json.loads(receipt.read_text()).get('terminal_path_exact') is True:
            files['terminal'] = base.parent/'protocol_v6_v135_handoff_confirmation'/'terminal'/(name+'.json')
    row['terminal_source'] = str(files['terminal'])
    # V118/V124 used ready/, but sparse launch_traces never represent all queries.
    if not files['ready'].exists():
        files['ready'] = base/'ready'/(name+'.jsonl.gz')
    row['missing'] = [k for k, p in files.items() if not p.is_file()]
    if files['trace'].is_file():
        row['schema']['trace'] = trace_schema(files['trace'])
    if files['ready'].is_file():
        row['schema']['ready'] = ready_schema(files['ready'])
    if not row['missing'] and all(v['complete'] for v in row['schema'].values()):
        with np.load(files['record']) as record:
            needed = {'event_fields', 'arrivals', 'departures'}
            present = needed <= set(record.files)
            fields = record['event_fields'].tolist() if present else []
            row['schema']['record'] = dict(complete=present and
                {'trip_id', 'station_id', 'direction', 'time_s'} <= set(fields), fields=fields)
        terminal = json.loads(files['terminal'].read_text())
        row['schema']['terminal'] = dict(complete=all(k in terminal for k in
            ('dispatches', 'returns', 'reference_clocks', 'plans')))
    row['fields_complete'] = not row['missing'] and all(v['complete'] for v in row['schema'].values())
    row['prior_summary_evidence'] = [] if identity is None else sorted(exposures.get(identity[1], []))
    row['use_status'] = 'prior_aggregate_or_diagnostic' if row['prior_summary_evidence'] else 'unconsumed_not_established'
    return row


def run(args):
    roots = result_roots(args.parent)
    experiments, compressed, aliases = [], [], []
    for root in roots:
        older = 0
        for p in sorted(root.iterdir()):
            if p.is_file() and p.name.endswith(('.tar', '.tar.gz', '.tgz', '.zip', '.tar.zst')):
                compressed.append(str(p))
            elif p.is_dir() and p.resolve() != args.out_root.resolve():
                experiments.append(p)
                older += not p.name.startswith('protocol_v6')
        aliases.append(dict(root=str(root), other_result_directories=older))
    exposures, metadata, errors = defaultdict(set), [], []
    bases = []
    for i, experiment in enumerate(experiments):
        bases.extend(artifact_bases(experiment))
        compressed.extend(str(p) for p in experiment.iterdir() if p.is_file()
            and p.name.endswith(('.tar', '.tar.gz', '.tgz', '.zip', '.tar.zst')))
        for base in (experiment/'summary', experiment/'prepare_summary'):
            for name in ('status.json', 'summary.json'):
                path = base/name
                if not path.is_file():
                    continue
                try:
                    days = metadata_days(json.loads(path.read_text()))
                    for day in days:
                        exposures[day].add(str(path))
                    metadata.append(dict(path=str(path), days=sorted(days)))
                except (ValueError, OSError) as exc:
                    errors.append(dict(path=str(path), error=str(exc)))
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=i+1, total=len(experiments),
            unit='archive_metadata', experiment=experiment.name,
            elapsed_seconds=time.monotonic()-args.started)), flush=True)
    rows, groups = [], []
    for i, base in enumerate(sorted(set(bases))):
        names = set()
        for directory in ('traces', 'launch_traces', 'ready', 'ready_labels'):
            names.update(p.name.removesuffix('.jsonl.gz') for p in (base/directory).glob('*.jsonl.gz'))
        record_count = len(list((base/'records').glob('*.npz')))
        group = dict(base=str(base), records=record_count, trace_or_ready_bundles=len(names))
        local = []
        for name in sorted(names):
            try:
                local.append(inspect_bundle(base, name, exposures))
            except (ValueError, OSError, KeyError) as exc:
                errors.append(dict(path=str(base/name), error=str(exc)))
        rows.extend(local)
        group.update(fields_complete=sum(r['fields_complete'] for r in local),
            missing_counts=dict(Counter(k for r in local for k in r['missing'])),
            trace_missing_fields=dict(Counter(k for r in local for k in r['schema'].get('trace', {}).get('missing', []))),
            use_status_counts=dict(Counter(r['use_status'] for r in local)),
            days=sorted({r['identity'][1] for r in local if r['identity'] is not None}))
        groups.append(group)
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=i+1, total=len(set(bases)),
            unit='archive_schema', elapsed_seconds=time.monotonic()-args.started)), flush=True)
    complete = [r for r in rows if r['fields_complete']]
    unknown = [r for r in complete if r['use_status'] == 'unconsumed_not_established']
    write(args.out_root/'details/admission.json', dict(bundles=rows, metadata=metadata, errors=errors))
    write(args.out_root/'summary/status.json', dict(status='v144_archive_inventory_complete',
        search_parents=[str(p) for p in args.parent], result_roots=aliases, experiments=len(experiments),
        groups=groups, bundles=len(rows), fields_complete=len(complete),
        fields_complete_with_prior_use=len(complete)-len(unknown),
        fields_complete_use_unknown=len(unknown),
        complete_cases=[dict(base=r['base'], name=r['name'], identity=r['identity'],
            terminal_source=r['terminal_source'], use_status=r['use_status'],
            prior_summary_evidence=r['prior_summary_evidence']) for r in complete],
        compressed_archives_not_opened=compressed, errors=errors,
        independently_admitted_paths=0, validation_run=False,
        admission='No archive is declared unused merely because no summary reference was found. '
            'Unknown-use complete bundles require provenance review before scoring. '
            'The frozen 32-day/12-actor/two-mode design is not replaced by a convenient partial grid.',
        point_forecasts_recomputed=False, interval_updates=0, new_rollouts=0, new_demand_days=0,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        limitations='Known FreqDuet server result stores, current recorder layouts at experiment roots '
            'and immediate smoke/shard children, not recursive counterfactual cells or training caches. '
            'Other-protocol directories are also inspected but their names cannot certify frozen physics. '
            'Aggregate/diagnostic day references establish prior outcome exposure; their absence '
            'does not prove independence. Schema checks do not score forecast outcomes. '
            'No records, raw traces or checkpoints are exported.'))
    print('V144_ARCHIVE_INVENTORY_COMPLETE', len(complete), len(unknown), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, action='append', required=True)
    p.add_argument('--out-root', type=Path, required=True)
    args = p.parse_args()
    args.started = time.monotonic()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v144_*.py', '-v'], check=True)
    run(args)
