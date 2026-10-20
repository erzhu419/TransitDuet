#!/usr/bin/env python3
"""Read-only V159 metadata sidecars and uncollected independent-day reservation."""

import argparse
import ast
from bisect import bisect_left
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
from itertools import zip_longest
import json
from math import isfinite
from multiprocessing import get_context
import os
from pathlib import Path
from statistics import fmean
import subprocess
import sys

import audit_freqduet_v159_residual_archive as checked


extract, audit = checked.extract, checked.audit
read, write, require, close, stream = checked.read, checked.write, checked.require, checked.close, checked.stream
CALIBRATION = tuple(range(2851001, 2851033))
CONFIRMATION = tuple(range(2852001, 2852033))
SAMPLERS = {d: 1600000000+i for i, d in enumerate((*CALIBRATION, *CONFIRMATION))}


def residence_support(durations, key, now, elapsed):
    direction, station = key
    rows, times = durations.groups.get(key, []), durations.times.get(key, [])
    end, start = bisect_left(times, now), bisect_left(times, now-3600.)
    historical = start == end
    if historical:
        start = 0
    selected = rows[start:end]
    survivors = [(t, d-elapsed) for t, d in selected if d >= elapsed]
    require(bool(survivors), 'recorded current residence lacks its completed conditional support')
    latest = max(t for t, _ in survivors)
    require(latest < now and elapsed >= 0., 'residence support is not strictly completed')
    # Reconcile the frozen mean only; no empirical quantiles are computed.
    return dict(direction=direction, station=station, query_time_s=now, elapsed_ready_s=elapsed,
        selected_n=len(selected), survivor_n=len(survivors), latest_survivor_s=latest,
        latest_selected_s=selected[-1][0], historical_fallback=historical,
        mean_remaining_s=fmean(v for _, v in survivors))


def follower_clock(row, record):
    e, context, now = row['evidence'], row['context'], record['query_time_s']
    close(now, e['ready_time_s'])
    require(row['policy_time_s'] > now, 'V159 late-snapshot contract changed')
    source = str(e['follower_source']).strip().lower()
    require(source == str(context['follower_evidence_source']).strip().lower(),
        'policy context no longer refers to the ready-cached follower source')
    gap = e['follower_gap_prediction_s']
    require(gap == context['follower_departure_gap_s'], 'cached pre-action follower gap changed')
    valid = source.startswith('same_time_avl_') and gap is not None and isfinite(gap) and gap >= 0.
    require(valid == context['follower_evidence_valid'], 'cached follower-validity rule differs')
    return dict(follower_evidence_valid=valid, source=source,
        gap_s=gap, value_origin='ready_cached_pre_action_fields',
        ready_snapshot_time_s=now, ready_callback_order=record['query_ready_order'],
        context_capture_time_s=row['policy_time_s'], policy_snapshot_time_s=row['policy_time_s'],
        load_sample_time_s=row['policy_time_s'], query_load=None, query_load_available=False,
        target_label_available_time_s=record['label']['available_time_s'])


def shadow_clock(call, running):
    leg = running.get((bool(call['direction']), int(call['station'])))
    if leg is None:
        return call['predicted_arrival_time_s'], 'unchanged_current_or_partial_leg'
    return leg['predicted_entry_time_s']+leg['fixed_prior']['mean'], 'recorded_running_entry_plus_frozen_mean'


def compare_support(actual, saved):
    require((actual is None) == (saved is None), 'V151 service support availability differs')
    if actual is not None:
        require(set(actual) == set(saved), 'V151 service support fields differ')
        for field, value in actual.items():
            if value is None or isinstance(value, bool):
                require(value == saved[field], 'V151 service support differs: '+field)
            else:
                close(value, saved[field])


def shadow_services(record, law):
    now, support = record['query_time_s'], record['support']
    running = {(bool(r['direction']), int(r['station'])): r for r in support['running']}
    native = {int(r['station']): r['components']['service'] for r in support['completed_components']}
    calls, delta = [], 0.
    for call in support['service']:
        direction, station = bool(call['direction']), int(call['station'])
        baseline = law.estimate(direction, station, now, call['predicted_arrival_time_s'], call['elapsed_service_s'])
        compare_support(baseline, call['support'])
        arrival, origin = shadow_clock(call, running)
        value = law.estimate(direction, station, now, arrival, call['elapsed_service_s'])
        selected = native[station] if value is None else value
        reference = native[station] if baseline is None else baseline
        require(selected is not None and reference is not None, 'shadow native fallback support is absent')
        latest = selected.get('latest_current_s', selected.get('latest'))
        require(latest is None or latest < now, 'shadow service support includes unfinished labels')
        delta += selected['mean']-reference['mean']
        calls.append(dict(direction=direction, station=station, query_time_s=now,
            predicted_arrival_time_s=arrival, elapsed_service_s=call['elapsed_service_s'],
            clock_origin=origin, clearance_arrival=arrival >= law.cutoff,
            phase_changed=(arrival >= law.cutoff) != (call['predicted_arrival_time_s'] >= law.cutoff),
            historical_support=value, native_fallback=selected if value is None else None))
    before, after = [record['components'][m]['prediction_s']['service'] for m in ('v151', 'eight_day')]
    close(delta, after-before)
    return calls, delta


def case(args, seed, day, services):
    geometry = audit.frozen.native.model.load_geometry(args.runtime)
    counts = Counter()
    for mode in ('old', 'handoff'):
        name = f'{seed}_{day}_{mode}'
        path, objects = audit.frozen.prepare(args.archive_root, seed, day, mode, geometry)
        law = audit.candidate.frozen.ServiceLaw(services['actors'][str(seed)], path['arrivals'], path['ready'], path['cutoff'])
        destination = args.out_root/'sidecars'/(name+'.jsonl.gz')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(destination, 'wt', encoding='ascii', compresslevel=1) as handle:
            for order, (item, record) in enumerate(zip_longest(path['trace'].items(),
                    stream(args.v159_root/'residuals'/(name+'.jsonl.gz')))):
                require(item is not None and record is not None, 'sidecar/residual cardinality differs')
                key, row = item
                require(record['query_key'] == list(key) and record['query_order'] == order and
                    (record['actor'], record['day'], record['mode']) == (seed, day, mode), 'sidecar query identity differs')
                clock = follower_clock(row, record)
                sidecar = dict(actor=seed, day=day, mode=mode, query_key=key, query_order=order,
                    clocks_and_validity=clock, current_residence=None, shadow_service=None,
                    support_status='no_scored_joint_components')
                counts[mode+'/queries'] += 1
                counts[mode+'/valid_follower'] += int(clock['follower_evidence_valid'])
                counts[mode+'/late_load_kept_unavailable'] += int(clock['query_load'] is None)
                if record['components'] is not None:
                    sidecar['support_status'] = 'structured_metadata_completed'
                    calls, delta = shadow_services(record, law)
                    sidecar['shadow_service'] = calls
                    counts[mode+'/shadow_service_queries'] += 1
                    counts[mode+'/shadow_service_calls'] += len(calls)
                    counts[mode+'/shadow_phase_changes'] += sum(c['phase_changed'] for c in calls)
                    counts[mode+'/shadow_service_mean_changed'] += int(abs(delta) > 1e-7)
                    joint = record['components']['v151']['inseparable_current_ready_key']
                    if joint is not None:
                        current = tuple(joint)
                        ready_s, now = path['ready'][current]['ready_time_s'], record['query_time_s']
                        require(path['ready'][current]['bus_id'] == record['predicted_follower'][1],
                            'residence support belongs to another physical follower')
                        if ready_s == now:
                            require(path['_observed_ready'].rank[current] < record['query_ready_order'],
                                'current ready callback was not observed at query')
                        require(ready_s <= now, 'residence ready time was still in the future')
                        value = residence_support(objects[2], (current[2], current[1]), now, now-ready_s)
                        close(value['mean_remaining_s'], record['observable']['position']['initial_residence_prediction_s'])
                        sidecar['current_residence'] = dict(current_ready_key=current,
                            ready_time_s=ready_s, ready_callback_order=path['_observed_ready'].rank[current], **value)
                        counts[mode+'/current_residence_support_completed'] += 1
                handle.write(json.dumps(sidecar, allow_nan=False, separators=(',', ':'))+'\n')
    receipt = read(args.v159_root/'cells'/f'{seed}_{day}.json')['counts']
    for mode in ('old', 'handoff'):
        for fresh, old in (('queries', 'queries'), ('shadow_service_queries', 'structured'),
                ('current_residence_support_completed', 'joint_residence_not_pure_holding')):
            require(counts[mode+'/'+fresh] == receipt[mode+'/'+old], 'metadata support changed query cohort')
    write(args.out_root/'cells'/f'{seed}_{day}.json', dict(seed=seed, day=day, counts=dict(counts),
        existing_service_support_reproduced=True, shadow_service_delta_reproduced=True,
        residence_mean_reproduced=True))


def literal_conflicts(path, reserved):
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value in reserved:
            found.add(node.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'range' and node.args and all(
                isinstance(a, ast.Constant) and isinstance(a.value, int) for a in node.args):
            value = range(*(a.value for a in node.args))
            found.update(n for n in reserved if n in value)
    return found


def sampler_ids(value):
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('sampler_seed', 'sampler_seeds', 'samplers'):
                pending = [item]
                while pending:
                    v = pending.pop()
                    if isinstance(v, int):
                        found.add(v)
                    elif isinstance(v, dict):
                        pending.extend(v.values())
                    elif isinstance(v, list):
                        pending.extend(v)
            else:
                found.update(sampler_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(sampler_ids(item))
    return found


def preregister(args, actors):
    inventory = read(args.v159_root/'details/archive_inventory.json')
    reserved_days, reserved_samplers = set(CALIBRATION) | set(CONFIRMATION), set(SAMPLERS.values())
    used_days = {int(k) for k in inventory['day_references']}
    used_samplers, files = set(), set()
    for group in inventory['groups']:
        used_days.update(group['days'])
        base = Path(group['base'])
        for directory in ('paths', 'pairs'):
            files.update((base/directory).glob('*.json'))
    for reference in inventory['day_references'].values():
        files.update(Path(r['path']) for r in reference)
    for path in sorted(files):
        value = read(path)
        used_samplers.update(sampler_ids(value))
        used_days.update(extract.day_ids(value))
    require(not reserved_days & used_days, 'reserved demand days already occur in archive metadata')
    require(not reserved_samplers & used_samplers, 'reserved sampler seeds already occur in archive metadata')
    scripts = Path(__file__).parent
    for path in scripts.glob('*.py'):
        if path.name != Path(__file__).name:
            require(not literal_conflicts(path, reserved_days | reserved_samplers),
                'reserved day/sampler already declared in generator '+path.name)
    require(not set(CALIBRATION) & set(CONFIRMATION), 'independent cohorts overlap')
    plan = dict(status='preregistered_not_collected', stage='v160_independent_day_reservation',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        calibration_days=list(CALIBRATION), confirmation_days=list(CONFIRMATION),
        sampler_seeds={str(d): SAMPLERS[d] for d in (*CALIBRATION, *CONFIRMATION)},
        actors=actors, modes=['old', 'handoff'], checkpoint_ep=199,
        split_unit='entire demand day and its sampler across all actors and both modes',
        paths_per_cohort=len(actors)*len(CALIBRATION)*2, total_future_paths=len(actors)*64*2,
        service_point_source='1b783ea8f3', running_shadow_source='5be4cea534',
        reference_point='fixed V151', shadow_role='unchanged diagnostic only',
        v146_extensions_s=[3., 0.], window_s=3600., tails=[.1, .1],
        coverage_target=.8, scopes=['all', 'active', 'clearance', 'hour2', 'hour13'],
        required_artifacts=['path_receipt', 'all_query_trace', 'same_vehicle_ready_labels',
            'arrival_departure_records', 'terminal_clocks', 'clocked_support_sidecars'],
        context='query-time phase, route/stop, current service/source, frozen predicted horizon and support',
        load_context='excluded: late policy snapshots are not ready-time measurements',
        label_eligibility='available_time_s < later query time',
        low_support='retain abstention and opaque full-residual stratum; no zero component imputation',
        weighting='episodes within actor/day; actor and demand-day clusters, not iid queries',
        release='Interval estimator and final inference gates must be sealed before fresh collection. '
            'Confirmation outcomes stay unopened until calibration and estimator are sealed.',
        registration_scope='day split and frozen collection contract, not an interval estimator registration',
        freshness=dict(metadata_files_checked=len(files), scripts_checked=len(list(scripts.glob('*.py')))-1,
            prior_days_checked=len(used_days), overlap_days=[], overlap_samplers=[],
            scope='V159 known server stores/receipts and prior generator literals/ranges'),
        collector_submitted=False, new_rollouts=0, new_training=0, interval_fit=False,
        main_policy_changed=False)
    write(args.out_root/'preregistration.json', plan)
    print('V160_NEW_DAY_COHORTS_PREREGISTERED_NOT_COLLECTED', flush=True)
    return plan


def run(args):
    parent = read(args.v159_root/'summary/status.json')
    acceptance = read(args.readback_root/'summary/status.json')
    services = read(args.v151_root/'model.json')
    require(parent['registration']['execution_source'] == '6b5dfe58a8' and
        acceptance['status'] == 'v159_residual_readback_complete' and acceptance['v158_readback_reproduced'],
        'requires accepted original V159 archive')
    require((services['status'], services['execution_source']) == ('sealed', '1b783ea8f3'), 'V151 model differs')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v160_metadata.py', '-v'], check=True)
    actors, days = parent['registration']['actors'], parent['registration']['days']
    cases = [(s, d) for s in actors for d in days]
    case(args, *cases[0], services)
    print('V160_FIRST_METADATA_PAIR_RECONCILED', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(case, args, s, d, services) for s, d in cases[1:]]
        for n, job in enumerate(as_completed(jobs), 2):
            job.result()
            if n % 32 == 0:
                print('V160_METADATA_PROGRESS', n, len(cases), flush=True)
    cells = [read(args.out_root/'cells'/f'{s}_{d}.json') for s, d in cases]
    registration = preregister(args, actors)
    write(args.out_root/'summary/status.json', dict(status='v160_metadata_and_preregistration_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], pairs=len(cases), paths=2*len(cases),
        counts=dict(sum((Counter(c['counts']) for c in cells), Counter())), registration=registration,
        original_residual_files_changed=False, frozen_support_moments_reconciled=True,
        full_forecasts_recomputed=False, new_rollouts=0, actor_updates=0, critic_updates=0, dual_updates=0,
        interval_fit=False, new_days_collected=0, main_policy_changed=False, v151_changed=False,
        v146_changed=False, window_s=3600., tails=[.1, .1],
        limitations='Metadata augmentation on consumed development paths. Opaque residuals remain '
            'unallocated; late policy load remains unavailable at the query. Fresh split is reserved '
            'only; an interval estimation/inference protocol remains to be frozen before collection.'))
    print('V160_METADATA_AND_PREREGISTRATION_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive-root', 'v159-root', 'readback-root', 'v151-root', 'out-root', 'runtime'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    run(parser.parse_args())
