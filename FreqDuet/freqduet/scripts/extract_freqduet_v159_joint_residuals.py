#!/usr/bin/env python3
"""Extract paired, clocked development residuals; inventory without new scoring."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import audit_freqduet_v144_archive_admission as archives
import audit_freqduet_v158_prior_dependence as frozen


audit = frozen.audit
read, write, close = audit.read, audit.write, audit.close
EVIDENCE = ('bus_id', 'ready_time_s', 'follower_source', 'service_dwell_proxy_s',
    'predicted_follower_bus_id', 'predicted_follower_trip_id', 'eta_prediction_s',
    'spatial_gap_m', 'speed_mps')


class RecordedService(audit.candidate.frozen.ServiceLaw):
    def __init__(self, *args):
        super().__init__(*args)
        self.calls = []

    def estimate(self, direction, station, now, arrival, elapsed=0.):
        value = super().estimate(direction, station, now, arrival, elapsed)
        self.calls.append(dict(direction=direction, station=station, query_time_s=now,
            predicted_arrival_time_s=arrival, elapsed_service_s=elapsed, support=value))
        return value


def day_ids(value, day_context=False):
    """Read declared demand-day fields, not arbitrary integer actor/sampler IDs."""
    days = set()
    if isinstance(value, int) and not isinstance(value, bool) and day_context:
        days.add(value)
    elif isinstance(value, list):
        for item in value:
            days.update(day_ids(item, day_context))
    elif isinstance(value, dict):
        for key, item in value.items():
            semantic = key == 'day' or key == 'days' or key.endswith('_days') or key in (
                'eval_seeds', 'evaluation_seeds', 'demand_seeds')
            days.update(day_ids(item, day_context or semantic))
    return days


def outcome_day_ids(value):
    def outcomes(item):
        if isinstance(item, dict):
            if item.get('status') == 'preregistered_not_collected':
                return {}
            return {k: outcomes(v) for k, v in item.items()}
        if isinstance(item, list):
            return [outcomes(v) for v in item]
        return item

    return day_ids(outcomes(value))


def metadata_files(experiment):
    # Bounded recorder layouts: no traversal into old counterfactual/training caches.
    stages = [experiment, *(p for p in experiment.iterdir() if p.is_dir() and
        p.name not in archives.ARTIFACTS | {'source', 'cells', 'details', 'paths', 'pairs',
            'candidates', 'candidate_shadows', 'residuals', 'summary', 'prepare_summary'})]
    found = set()
    for stage in stages:
        for path in (stage/'protocol.json', stage/'summary/status.json',
                stage/'summary/summary.json', stage/'prepare_summary/status.json',
                stage/'prepare_summary/summary.json'):
            if path.is_file():
                found.add(path)
    return sorted(found)


def inventory(args, running, services):
    roots = archives.result_roots(args.parent)
    exposed, references, errors, bases = set(), defaultdict(list), [], set()
    for root in roots:
        for experiment in sorted(p for p in root.iterdir() if p.is_dir() and
                p.resolve() != args.out_root.parent.resolve()):
            bases.update(archives.artifact_bases(experiment))
            files = metadata_files(experiment)
            summaries = {p: read(p) for p in files if p.parent.name in ('summary', 'prepare_summary')}
            completed = any(not any(token in str(v.get('status', '')).lower()
                for token in ('inventory', 'admission')) for v in summaries.values())
            for path in files:
                try:
                    value = summaries[path] if path in summaries else read(path)
                    if any(token in str(value.get('status', '')).lower()
                            for token in ('inventory', 'admission')):
                        continue
                    days = day_ids(value)
                    consumed = outcome_day_ids(value) if completed else set()
                    # Registrations alone reserve days; completed summaries establish exposure.
                    for day in days:
                        role = 'completed_experiment' if day in consumed else 'registration_only'
                        references[day].append(dict(path=str(path), role=role))
                    exposed.update(consumed)
                except (ValueError, OSError) as exc:
                    errors.append(dict(path=str(path), error=str(exc)))
    prior_days = set(running['days']) | set(services['calibration_days'])
    # V145 supplies known consumed development paths even if a historical summary omits days.
    exposed.update(read(args.archive_root/'summary/status.json')['registration']['demand_days'])
    exposed.update(prior_days)
    rows = []
    for base in sorted(bases):
        names = set()
        for directory in ('traces', 'launch_traces', 'ready', 'ready_labels'):
            names.update(p.name.removesuffix('.jsonl.gz') for p in (base/directory).glob('*.jsonl.gz'))
        group = dict(base=str(base), bundles=len(names), days=[], counts=Counter(), unknown=[])
        for name in sorted(names):
            identity = archives.case_id(name)
            if identity is None:
                group['counts']['unrecognized_identity'] += 1
                continue
            day = identity[1]
            group['days'].append(day)
            if day in prior_days:
                group['counts']['prior_fit_day'] += 1
            elif day in exposed:
                group['counts']['previous_outcome_exposure'] += 1
            else:
                # Unknown use is not permission to compute residuals or fit intervals.
                try:
                    entry = archives.inspect_bundle(base, name, {})
                    group['unknown'].append(entry)
                    role = 'fields_complete_use_unknown' if entry['fields_complete'] else 'incomplete_use_unknown'
                    group['counts'][role] += 1
                except (ValueError, OSError, KeyError) as exc:
                    group['counts']['schema_error'] += 1
                    errors.append(dict(path=str(base/name), error=str(exc)))
        group['days'] = sorted(set(group['days']))
        group['counts'] = dict(group['counts'])
        rows.append(group)
    counts = dict(sum((Counter(r['counts']) for r in rows), Counter()))
    write(args.out_root/'details/archive_inventory.json', dict(groups=rows,
        day_references={str(k): v for k, v in references.items()}, errors=errors))
    summary = dict(status='v159_inventory_complete', result_roots=len(roots), groups=len(rows),
        counts=counts, prior_fit_days=sorted(prior_days), exposed_days=len(exposed),
        independently_certified_calibration_days=[], independently_certified_confirmation_days=[],
        unknown_complete=[dict(base=r['base'], days=sorted({e['identity'][1] for e in r['unknown']
            if e['fields_complete']})) for r in rows if any(e['fields_complete'] for e in r['unknown'])],
        errors=errors, unknown_days_scored=False,
        missing='No new independent split is certified by an absent use reference. Requires field-complete '
            'frozen-deployment archives and documented unexposed calibration/confirmation day provenance.',
        scope='Known server result roots and bounded current recorder layouts; compressed archives not opened.')
    write(args.out_root/'summary/inventory.json', summary)
    print('V159_INVENTORY_COMPLETE', json.dumps(counts), flush=True)
    return summary


def observable(row, position, status, cutoff):
    now = row['prediction_time_s']
    value = dict(query_time_s=now, hour=int(now//3600), demand_end_time_s=cutoff,
        clearance=now >= cutoff, evidence={k: row['evidence'].get(k) for k in EVIDENCE},
        position_status=status, position=None)
    if position is not None:
        p = position
        value['position'] = dict(origin_index=p['origin'], target_index=p['target'],
            progress_m=p['progress'], remaining_stops=p['target']-p['origin'],
            remaining_distance_m=p['route']['positions'][p['target']]-p['progress'],
            initial_residence_prediction_s=p['initial_s'])
    # Load was captured at policy callback, not necessarily at the earlier ready query.
    if row['policy_time_s'] <= now:
        value['query_load'] = row.get('query_load')
    return value


def label_record(path, row):
    status, truth = audit.stats.target_label(row, path['ready'])
    target = None
    if status == 'resolved':
        trip, bus = audit.stats.follower(row)
        target = path['ready'][trip, int(row['station_id']), bool(row['direction'])]
        close(target['ready_time_s']-row['prediction_time_s'], truth)
        if int(target['bus_id']) != bus:
            raise AssertionError('target physical follower identity changed')
    return dict(status=status, full_eta_s=truth if status == 'resolved' else None,
        target_ready_time_s=None if target is None else target['ready_time_s'],
        available_time_s=None if target is None else target['ready_time_s'])


def components(packet, actual, joint):
    prediction, truth = audit.accounting.coarse(packet), audit.accounting.coarse(actual)
    errors = {c: prediction[c]-truth[c] for c in frozen.COARSE}
    close(sum(errors.values()), sum(packet.values())-sum(actual.values()))
    return dict(prediction_s=prediction, truth_s=truth, residual_s=errors,
        holding_separable=joint is None,
        holding_residual_s=packet['holding']-actual['holding'] if joint is None else None,
        execution_wait_residual_s=packet['execution_wait']-actual['execution_wait'] if joint is None else None,
        inseparable_current_ready_key=joint)


def completed_support(position, history):
    values = []
    if position is None:
        return values
    p = position
    for i in range(p['origin'], p['target']+1):
        station = p['route']['ids'][i]
        support = {channel: history.estimate((channel, p['direction'], station), p['now'])
            for channel in ('running', 'service', 'holding', 'ticks')}
        if any(v is not None and v['latest'] >= p['now'] for v in support.values()):
            raise AssertionError('component support contains an unfinished label')
        values.append(dict(station=station, components=support))
    return values


def reconcile(groups, expected):
    if set(groups) != set(expected):
        raise AssertionError('joint residual cohort support differs from V158')
    for key, entry in groups.items():
        old = expected[key]
        if (entry['queries'], entry['episodes']) != (old['queries'], old['episodes']):
            raise AssertionError('joint residual query/episode support differs from V158')
        for field, value in entry['metrics'].items():
            close(value, old['metrics'][field])


def case(args, seed, day, running, services):
    geometry = audit.frozen.native.model.load_geometry(args.runtime)
    expected = read(args.v158_root/'cells'/f'{seed}_{day}.json')
    counts, modes = Counter(), {}
    for mode in audit.frozen.frozen.MODES:
        path, objects = audit.frozen.prepare(args.archive_root, seed, day, mode, geometry)
        phases = audit.frozen.frozen.candidate.frozen.PhaseServices(path['arrivals'], path['ready'], path['cutoff'])
        service = RecordedService(services['actors'][str(seed)], path['arrivals'], path['ready'], path['cutoff'])
        law = frozen.launch.RecordedRunning(running['actors'][str(seed)], day,
            audit.frozen.prior.RunningPrefix(path, geometry))
        groups = defaultdict(list)
        destination = args.out_root/'residuals'/f'{seed}_{day}_{mode}.jsonl.gz'
        destination.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(destination, 'wt', encoding='ascii') as stream:
            for order, (key, row) in enumerate(path['trace'].items()):
                now = row['prediction_time_s']
                label = label_record(path, row)
                record = dict(actor=seed, day=day, mode=mode, query_key=key,
                    predicted_follower=audit.stats.follower(row), query_time_s=now, query_order=order,
                    query_ready_order=path['_observed_ready'].rank.get(key),
                    episode=(*audit.stats.follower(row), *key[1:]),
                    label=label, observable=observable(row, None, 'not_reconstructed', path['cutoff']),
                    support=None, forecasts=None, components=None)
                counts[mode+'/queries'] += 1
                counts[mode+'/label/'+label['status']] += 1
                if label['status'] == 'resolved':
                    service.calls.clear()
                    before, detail0 = audit.candidate.frozen.forecast(path, row, geometry, objects, phases, service, True)
                    service_calls = list(service.calls)
                    if before is None:
                        before = audit.frozen.fallback(path, row, geometry, objects, phases)
                    if before is not None:
                        law.calls.clear()
                        after, detail = audit.candidate.forecast(path, row, geometry, objects, phases, service, law)
                        after = before if after is None else after
                        record['forecasts'] = {name: dict(point_s=point, residual_s=point-label['full_eta_s'])
                            for name, point in (('v151', before), ('eight_day', after))}
                        running_support = []
                        for (direction, station), call in law.calls.items():
                            prefix = law.prefix.selected(direction, station, now)
                            if prefix['latest_age_s'] <= 0.:
                                raise AssertionError('running support includes unfinished label')
                            running_support.append(dict(direction=direction, station=station,
                                predicted_entry_time_s=call['entry_time_s'], current_prefix=prefix,
                                fixed_prior=call['selected']))
                        record['support'] = dict(service=service_calls, running=running_support,
                            v151_detail=detail0, eight_day_detail=detail)
                        if 'running_s' in detail:
                            reference, joint, status = audit.accounting.structured_packet(
                                path, row, geometry, objects, phases, service, True)
                            actual = audit.parts.actual_parts(path, row, geometry[bool(row['direction'])])
                            close(sum(reference.values()), before)
                            close(sum(actual.values()), label['full_eta_s'])
                            shadow = audit.shadow_packet(reference, before, after, detail['running_s'])
                            position, _, _ = audit.accounting.candidate.position(
                                path, row, objects[-1], geometry, objects, service, True)
                            record['observable'] = observable(row, position, status, path['cutoff'])
                            record['support']['completed_components'] = completed_support(position, objects[0])
                            record['components'] = {name: components(packet, actual, joint)
                                for name, packet in (('v151', reference), ('eight_day', shadow))}
                            counts[mode+'/structured'] += 1
                            counts[mode+'/joint_residence_not_pure_holding'] += int(joint is not None)
                            scopes = ['all', 'clearance' if now >= path['cutoff'] else 'active']
                            if int(now//3600) in (2, 13):
                                scopes.append('hour'+str(int(now//3600)))
                            for name, packet in (('v151', reference), ('eight_day', shadow)):
                                for cohort, errors in frozen.error_views(packet, actual, joint).items():
                                    entry = frozen.error_record(record['episode'], errors, sum(packet.values())-label['full_eta_s'])
                                    for scope in scopes:
                                        groups[name+'/'+cohort+'/'+scope].append(entry)
                        else:
                            counts[mode+'/opaque_full_endpoint'] += 1
                    else:
                        counts[mode+'/no_frozen_point'] += 1
                stream.write(json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n')
        summarized = {g: audit.stats.describe(rows) for g, rows in groups.items()}
        reconcile(summarized, {g: v for g, v in expected['modes'][mode]['groups'].items()
            if '/coarse/' in g or '/holding_identified/' in g})
        modes[mode] = dict(groups={k: dict(queries=v['queries'], episodes=v['episodes'])
            for k, v in summarized.items()})
    write(args.out_root/'cells'/f'{seed}_{day}.json', dict(actor=seed, day=day,
        counts=dict(counts), modes=modes, v158_moments_reproduced=True))


def run(args):
    running, services = read(args.v154_root/'model.json'), read(args.v151_root/'model.json')
    parent = read(args.v158_root/'summary/status.json')
    if (parent['status'], parent['execution_source'], services['execution_source'], running['status']) != (
            'v158_prior_dependence_complete', 'e5a6705971', '1b783ea8f3', 'sealed_development_moments'):
        raise AssertionError('requires frozen V158 audit and sealed V151/V154 priors')
    cases = audit.prior.archive_cases(read(args.archive_root/'summary/status.json'), running, services)
    plan = dict(stage='v159_readonly_joint_residual_extraction',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        actors=sorted({s for s, d in cases}), days=sorted({d for s, d in cases}),
        development_only=True, models=['v151', 'eight_day_unpromoted_shadow'],
        components=['running', 'service', 'residence'], error='prediction_minus_truth_seconds',
        residence='post-ready holding plus execution waits and any inseparable current-ready remainder',
        label_clock='same physical follower target-ready; usable only when available_time_s < next_query_time_s',
        weights='preserve episode identifiers: equal episode means within actor/day, then equal actor/day',
        service_source='1b783ea8f3', running_source='5be4cea534', v146_extensions_s=[3., 0.],
        window_s=3600., tails=[.1, .1], interval_fit=False, interval_selection=False,
        new_rollouts=0, actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False)
    write(args.out_root/'protocol.json', plan)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v159_*.py', '-v'], check=True)
    inventory_summary = inventory(args, running, services)
    case(args, *cases[0], running, services)
    print('V159_FIRST_PAIR_MOMENTS_REPRODUCED', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(case, args, s, d, running, services) for s, d in cases[1:]]
        for n, job in enumerate(as_completed(jobs), 2):
            job.result()
            if n % 32 == 0:
                print('V159_PROGRESS', n, len(cases), flush=True)
    cells = [read(args.out_root/'cells'/f'{s}_{d}.json') for s, d in cases]
    write(args.out_root/'summary/status.json', dict(status='v159_joint_residual_extraction_complete',
        registration=plan, pairs=len(cases), paths=2*len(cases), v158_moments_reproduced=True,
        counts=dict(sum((Counter(c['counts']) for c in cells), Counter())),
        inventory=inventory_summary, server_residual_directory=str(args.out_root/'residuals'),
        limitations='Consumed development archive, not new calibration or confirmation. Label outcomes '
            'are separate from query-observable inputs; opaque/missing labels retain null components. '
            'No interval fitted, no controller promoted, no raw archive exported.'))
    print('V159_JOINT_RESIDUAL_EXTRACTION_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive-root', 'v151-root', 'v154-root', 'v158-root', 'out-root', 'runtime'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--parent', type=Path, action='append', required=True)
    parser.add_argument('--workers', type=int, default=8)
    run(parser.parse_args())
