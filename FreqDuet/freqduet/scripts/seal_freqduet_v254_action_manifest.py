#!/usr/bin/env python3
"""Reserve V253 metadata and preflight interfaces; no simulator entrypoint."""

import argparse
import ast
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import re
import unittest

from preregister_freqduet_v224_prepare_handoff import CONTEXTS
from preregister_freqduet_v253_action_supply import check_future_split, registration as parent_protocol


ROOT_BASE = 3540000000
SMOKE_ROOT = 3540999999
PHASE_OFFSETS = {'selection': 0, 'evaluation': 20}
BASE_STREAMS = dict(scenario=0, lower_event_and_sampling=10000000, upper_sampling=20000000,
                    python=30000000, numpy_global=40000000, torch_global=50000000)
PAYLOADS = dict(legacy='records', v224='sidecars', v230='resource_sidecars',
                v235='cycle_sidecars', v241='request_sidecars', v248='prepare_timing_sidecars')
RAW = {'records', 'traces', 'sidecars', 'resource_sidecars', 'cycle_sidecars', 'request_sidecars',
       'prepare_timing_sidecars', 'queries', 'query_snapshots', 'cells', 'cases', 'groups',
       'residuals', 'atoms', 'maps', 'logs', 'checkpoints', 'ckpt', 'ckpts', 'details', 'figures', 'plots'}
META_NAMES = {'protocol.json', 'registration.json', 'preregistration.json', 'manifest.json',
              'reservation.json', 'reservation_intent.json', 'reserved_slots.json',
              'future_manifest.json', 'execution_authorization.json', 'status.json', 'summary.json'}
RECEIPT_DIRS = {'started', 'checks', 'progress', 'paths', 'pairs'}


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return json.load(stream)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def stream_offsets(names):
    assert names and len(names) == len(set(names))
    return dict(BASE_STREAMS, **{'runtime/'+name: 60000000+i*1000000 for i, name in enumerate(names)})


def runtime_stream_names(path):
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_runtime_numpy_streams')
    returned = next(n.value for n in method.body if isinstance(n, ast.Return))
    assert isinstance(returned, ast.Dict)
    return [ast.literal_eval(k) for k in returned.keys]


def contexts():
    return [dict(context=i, actor=actor, demand_day=day, checkpoint_ep=199,
        prefix_sampler=187000000+actor+100*day, engine_arm='v107',
        prefix_status='consumed_development_not_new_day') for i, (actor, day) in enumerate(CONTEXTS)]


def future_manifest(names):
    offsets = stream_offsets(names)
    rows = []
    for c in range(4):
        for phase, delta in PHASE_OFFSETS.items():
            for i in range(8):
                root = ROOT_BASE+100*c+delta+i
                rows.append(dict(context=c, phase=phase, future=i, root=root,
                    half=None if phase == 'selection' else 'A' if i < 4 else 'B',
                    streams={name: root+offset for name, offset in offsets.items()}))
        check_future_split([r['root'] for r in rows if r['context'] == c and r['phase'] == 'selection'],
                           [r['root'] for r in rows if r['context'] == c and r['phase'] == 'evaluation'])
    rows.append(dict(context=0, phase='smoke', future=0, root=SMOKE_ROOT, half=None,
                     streams={name: SMOKE_ROOT+offset for name, offset in offsets.items()}))
    inventories = [set(r['streams'].values()) for r in rows]
    assert sum(map(len, inventories)) == len(set().union(*inventories)), 'root/derived stream overlap'
    assert all(0 <= seed < 2**32 for values in inventories for seed in values)
    return rows


def reserved_slots(futures):
    slots = [dict(slot='passive_c0_v107', phase='passive', context=0, root=None,
                  kind='observer_only', requires='separate launch approval'),
             dict(slot='noop_c0_v107', phase='passive', context=0, root=None,
                  kind='forced_native_once', requires='passive_c0_v107 exact pass')]
    for f in futures:
        stem = f"c{f['context']}_{f['phase']}_f{f['future']}"
        slots.append(dict(slot=stem+'_native', context=f['context'], phase=f['phase'], root=f['root'],
                          kind='native', first=True))
        for b in range(7):
            slots.append(dict(slot=stem+'_b'+str(b), context=f['context'], phase=f['phase'], root=f['root'],
                kind='fixed_bin_label', bin=b, skip_if_native_alias=True, require_native_mask=True))
        if f['phase'] == 'smoke':
            slots.append(dict(slot=stem+'_forced_native', context=0, phase='smoke', root=f['root'],
                kind='explicit_engineering_native_alias', requires='smoke native exact fork pass'))
    return slots


def preflight_interface():
    return dict(reference_collection='protocol_v6_v249_prepare_timing_preflight_collection',
        reference_source='96960db57a', reference_file='c0_v107.json.gz', old_payloads=PAYLOADS,
        new_payload='independent_pulse_audit_only',
        slots=['passive_c0_v107', 'noop_c0_v107'],
        query_fields=['native query seq/time/occurrence', 'exact PREPARE seq/order/time',
            'bus/trip/observed-launch key', 'sampled native bin and feasible mask',
            'separate PREPARE/query H', 'frozen F/B/front timing',
            'current plans/versions,ready inventory,incoming positions'],
        equality='Entire six old payloads preserve values,key sets,list order,data types,'
            'actions,uniforms,inputs,all native/sidecar clocks,plans,physical ledgers and '
            'endpoints. Use frozen full-payload equality and recording validators;no '
            'tolerance,rounding,dropped fields or digest substitution. New data is separate.',
        no_op='At FIRST V253-qualified query,after native action/mask are captured,record '
            'one external-native command but return the exact original action object/value. '
            'One original policy call,one native execution;no extra stamp,query,estimator,'
            'snapshot,RNG draw or step. Observer-only path does not dispatch a command.',
        required_checks=['six_payloads_exact', 'anchor_occurrence_complete',
            'prepare_query_clock_domains_preserved', 'cycle_H_and_query_plan_fields_complete',
            'native_calls_and_rng_unchanged', 'exact_native_action_passthrough',
            'original_fleet_mask_and_authority_preserved'],
        expected_boundary='No future switch in either passive slot;original prefix sampler '
            'for the WHOLE path. Native-bin alias under a NEW future is tested ONLY in '
            'the separate smoke slot,not against the old-future endpoint.',
        next='Implementation plus synthetic checks only next;each real slot needs separate approval.')


def manifest(source, names):
    future_rows = future_manifest(names)
    return dict(stage='v254_v253_context_future_budget_and_passive_interface', source=source,
        protocol_source='4667fd4668', contexts=contexts(), futures=future_rows,
        contexts_are_fresh=False, future_namespace='FreqDuet/v254',
        runtime_stream_names=names, stream_offsets=stream_offsets(names),
        frozen=dict(v238_source='b4b1d00203', controller_source='2b0115af7b',
            runtime_optimization_source='9a59ed3b0d', lower='V107 bilateral025',
            upper='native B200 feedback,NOT V220 projection', fleet=12,
            holding_bins_s=[0,5,10,15,20,30,45],
            other='Main,trained/prediction/calibration states,windows,quantiles,masks unchanged.'),
        preflight=preflight_interface(),
        physical_start_caps=dict(passive=2, smoke=8, selection=224, evaluation=224, total=458),
        accounting='Formal:4 contexts x (8S+8E) x AT MOST7 physical starts. Native first,'
            'then legal fixed bins other than captured native. Eight logical labels '
            'native+b0..b6 mean <=7 starts,NOT8. Masks are conditional;illegal labels and '
            'native aliases are unstarted,reported,not replaced. Smoke:<=7 plus ONE '
            'explicit forced-native engineering alias. No separate blind-metadata replay.',
        engineering_release='Passive observer-only then whole-old-future native no-op. '
            'Only exact passes admit the separate smoke root. Smoke native first then '
            'forced-native fork alias and remaining legal bins;verify common pre-command '
            'state/future,current-window preservation,logical-stream coupling,actual '
            'execution,identity/clock/holding closure and at least one nonzero executed '
            'difference. Outcome quality does not choose or tune smoke. Missing anchor '
            'or support stops,not a later anchor or larger gain.',
        bank_release='After engineering admission collect ALL fixed S units,then seal '
            'V253 offline contrast IDs/masks/complete S results BEFORE ANY E path. '
            'Collect all fixed E units/commands without reselection or adaptive expansion. '
            'Other contexts must match the causal anchor contract;never require eventual '
            'due exposure or matching supply. New innovations are conditional on consumed '
            'day profiles,not new-day efficacy. Two actor/day clusters per axis are weak '
            'pilot evidence;report conditional-future and crossed results separately.',
        failure_rule='Every simulator start consumes one slot,even failed/incomplete. '
            'No automatic retry,extra reference,context/root replacement or phase transfer. '
            'Recover completed artifacts read-only;unused mask/alias slots never fund '
            'additional trials. Original service population and completion gates stay fixed.',
        concurrency=dict(max_workers=8, passive_max_workers=1, scheduler_priority='high',
            manual_dispatch=True, nodes=['node001','node002','node003','node004','node005','node006']),
        blind_scope='Atomic resource-hidden metadata aliases the same fixed command;'
            'resource-specific policy efficacy remains closed until genuinely distinct '
            'aware/blind rules are separately sealed under V253. No online action map here.',
        current_execution_budget=dict(simulator_starts=0, future_starts=0, training_updates=0,
                                      model_fits=0, shadow_actions_executed=0),
        reserved_slots=reserved_slots(future_rows), launch_authorized=False,
        collector_implemented=False, runtime_connected=False, selector_defined=False,
        main_policy_changed=False, v238_changed=False)


def numbers(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from numbers(v)
    elif isinstance(value, list):
        for v in value:
            yield from numbers(v)
    elif isinstance(value, int):
        yield value
    elif isinstance(value, str):
        yield from map(int, re.findall(r'(?<![0-9])[0-9]{9,10}(?![0-9])', value))


def seed_hits(value, targets, offsets):
    # Include derived V68/V69 seeds from old declared roots,not just exact root reuse.
    found = set()
    for old in set(numbers(value)):
        for delta in offsets:
            if old+delta in targets:
                found.add((old, delta, old+delta))
    return [dict(recorded_seed=a, derived_offset=b, reserved_seed=c) for a,b,c in sorted(found)]


def audit_namespace(roots, excluded, spec, scheduler_snapshot):
    targets = {s for f in spec['futures'] for s in f['streams'].values()}
    offsets = set(spec['stream_offsets'].values())
    collisions, errors, files, names = [], [], 0, 0
    tasks = scheduler_snapshot['tasks']
    for task in tasks:
        hits = seed_hits(task, targets, offsets)
        if hits:
            collisions.append(dict(kind='scheduler', task=task['id'], hits=hits))
    for root in roots:
        assert root.is_dir(), ('archive root missing', str(root))
        for directory, children, filenames in os.walk(root):
            path = Path(directory)
            children[:] = sorted(d for d in children if path/d != excluded and
                not d.startswith('source') and d not in {'.git','__pycache__','node_modules'} and
                len((path/d).relative_to(root).parts) <= 6)
            if path.name in RAW:
                children[:] = []
                for name in filenames:
                    names += 1
                    hits = seed_hits(name, targets, {0})
                    if hits:
                        collisions.append(dict(kind='artifact_name', path=str(path/name), hits=hits))
                continue
            for name in filenames:
                if name not in META_NAMES and not (path.name in RECEIPT_DIRS and name.endswith('.json')):
                    continue
                target = path/name
                try:
                    value = read(target)
                    files += 1
                    hits = seed_hits(value, targets, offsets)
                    if hits:
                        collisions.append(dict(kind='metadata', path=str(target), hits=hits))
                except (OSError, ValueError) as exc:
                    errors.append(dict(path=str(target), error=str(exc)))
    return dict(decision='fresh' if not collisions and not errors else 'blocked',
        archive_roots=[str(p) for p in roots], metadata_files=files, artifact_names_checked=names,
        scheduler_records=len(tasks), scheduler_sources=scheduler_snapshot['source_files'],
        roots_reserved=len(spec['futures']), root_and_derived_seeds=len(targets),
        collisions=collisions, errors=errors,
        scope='Known server FreqDuet canonical metadata/receipts,raw artifact NAMES only,'
            'and local scheduler hot+archived metadata. No raw contents,logs,CSV,models '
            'or checkpoints opened. Reused development prefix seeds are NOT claimed fresh.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent-root','archive-parent','results-root','runtime-runner','scheduler-snapshot','out-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    parent = read(a.parent_root/'summary.json')
    assert parent['status'] == 'v253_action_difference_protocol_sealed' and parent['source'] == '4667fd4668'
    assert read(a.parent_root/'protocol.json') == parent_protocol('4667fd4668')
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                               pattern='test_v254_action_manifest.py')
    assert suite.countTestCases() > 0
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    assert tested.wasSuccessful()
    spec = manifest(a.source, runtime_stream_names(a.runtime_runner))
    a.out_root.mkdir(parents=True, exist_ok=False)
    write(a.out_root/'reservation_intent.json', spec)
    # Metadata only:prove the passive reference exists,without opening its six payloads.
    reference = a.results_root/spec['preflight']['reference_collection']
    accepted = read(reference/'summary/status.json')
    assert accepted['status'] == 'v249_passive_pair_passed' and accepted['source'] == '96960db57a'
    reference_files = [reference/folder/spec['preflight']['reference_file'] for folder in PAYLOADS.values()]
    assert all(path.is_file() for path in reference_files)
    roots = {a.results_root}
    roots.update(p for p in a.archive_parent.glob('FreqDuet*/FreqDuet/freqduet/results_freqduet') if p.is_dir())
    roots.update(p for p in a.archive_parent.glob('TransitDuet/FreqDuet/freqduet/results_freqduet') if p.is_dir())
    freshness = audit_namespace(sorted(roots), a.out_root, spec, read(a.scheduler_snapshot))
    write(a.out_root/'details/freshness.json', freshness)
    summary = dict(status='v254_manifest_reserved_no_replay' if freshness['decision'] == 'fresh'
        else 'v254_reservation_blocked', source=a.source, protocol_source='4667fd4668',
        tests=tested.testsRun, contexts=spec['contexts'], future_roots=len(spec['futures']),
        phase_roots=dict(Counter(f['phase'] for f in spec['futures'])),
        runtime_stream_names=spec['runtime_stream_names'], physical_start_caps=spec['physical_start_caps'],
        logical_slot_labels=len(spec['reserved_slots']), launch_authorized=False,
        reference_payload_files_available=len(reference_files), raw_payloads_opened=0,
        freshness={k:v for k,v in freshness.items() if k not in ('collisions','errors')},
        collisions=len(freshness['collisions']), errors=len(freshness['errors']),
        simulator_starts=0, future_starts=0, training_updates=0, model_fits=0,
        collector_implemented=False, runtime_connected=False, selector_defined=False,
        main_policy_changed=False, v238_changed=False,
        next_gate='Implement the frozen passive/no-op interface and synthetic tests only;'
            'real passive paths,smoke,S and E each still require separate launch approval.')
    write(a.out_root/'summary.json', summary)
    assert freshness['decision'] == 'fresh', 'metadata collision/error;retain intent and do not replace roots'
    write(a.out_root/'manifest.json', spec)
    write(a.out_root/'future_manifest.json', spec['futures'])
    write(a.out_root/'reserved_slots.json', spec['reserved_slots'])
    print('V254_RESERVED_NO_REPLAY '+json.dumps(summary, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
