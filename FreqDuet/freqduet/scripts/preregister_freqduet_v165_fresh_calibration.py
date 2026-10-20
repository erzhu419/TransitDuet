#!/usr/bin/env python3
"""Reserve a fixed fresh 64-day calibration cohort; no collection or fitting."""

import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import run_freqduet_v162_calibration_capture as capture


read, write, require = capture.read, capture.write, capture.require
metadata, extract = capture.metadata, capture.metadata.extract
archives = extract.archives
CALIBRATION = tuple(range(2951001, 2951065))
SAMPLERS = {day: 1650000000+i for i, day in enumerate(CALIBRATION)}
METHOD_FIELDS = ('frozen', 'target', 'conditioning', 'support', 'estimator', 'comparator', 'inference', 'gates')
RAW = {'traces', 'launch_traces', 'records', 'ready', 'ready_labels', 'terminal',
    'query_snapshots', 'queries', 'residuals'}
SKIP = {'source', 'atoms', 'maps', 'details', 'checkpoints', 'ckpt', 'ckpts', 'logs',
    'figures', 'plots', '__pycache__'}
RECEIPTS = {'paths', 'pairs', 'cases', 'cells', 'capture_cells'}
META_NAMES = {'model.json', 'protocol.json', 'preregistration.json', 'registration.json',
    'status.json', 'summary.json', 'manifest.json'}


def conflicts(value, confirmation):
    return dict(new_day_conflicts=sorted(set(CALIBRATION) & extract.day_ids(value)),
        new_sampler_conflicts=sorted(set(SAMPLERS.values()) & metadata.sampler_ids(value)),
        confirmation_outcome_conflicts=sorted(set(confirmation) & extract.outcome_day_ids(value)))


def filename_day(name):
    pieces = Path(name).stem.split('_')
    if len(pieces) >= 2 and pieces[0].isdigit() and pieces[1].isdigit():
        return int(pieces[1])
    return None


def inventory(args, confirmation):
    old = read(args.v159_root/'details/archive_inventory.json')
    roots = set(archives.result_roots(args.parent))
    roots.add(args.v163_root.parent.parent)
    for group in old['groups']:
        roots.update(p for p in Path(group['base']).parents if p.name == 'results_freqduet' and p.is_dir())
    known_days = {int(k) for k in old['day_references']} | {d for g in old['groups'] for d in g['days']}
    found = {k: set() for k in ('new_day_conflicts', 'new_sampler_conflicts', 'confirmation_outcome_conflicts')}
    found['new_day_conflicts'].update(set(CALIBRATION) & known_days)
    found['confirmation_outcome_conflicts'].update(set(confirmation) &
        {d for g in old['groups'] for d in g['days']})
    references, files, bundles, errors = [], set(), 0, []
    targets = set(CALIBRATION) | set(confirmation)
    for root in sorted(roots):
        require(root.is_dir(), 'known result root missing: '+str(root))
        for directory, children, names in os.walk(root):
            path = Path(directory)
            children[:] = sorted(d for d in children if d not in SKIP and
                path/d != args.out_root.parent and len((path/d).relative_to(root).parts) <= 5)
            if path.name in RAW:
                children[:] = []
            for name in names:
                if path.name in RAW | RECEIPTS:
                    bundles += 1
                    day = filename_day(name)
                    if day in targets:
                        field = 'new_day_conflicts' if day in CALIBRATION else 'confirmation_outcome_conflicts'
                        found[field].add(day)
                        references.append(dict(path=str(path/name), field=field, value=day))
                if path.name not in RAW and name.endswith('.json') and (
                        path.name in RECEIPTS or name in META_NAMES):
                    files.add(path/name)
    # Only metadata is opened; target-day artifact names already establish a conflict.
    for index, path in enumerate(sorted(files), 1):
        if filename_day(path.name) in targets:
            continue
        try:
            value = read(path)
            for field, items in conflicts(value, confirmation).items():
                found[field].update(items)
                references.extend(dict(path=str(path), field=field, value=v) for v in items)
        except (ValueError, OSError) as exc:
            errors.append(dict(path=str(path), error=str(exc)))
        if index % 2000 == 0:
            print('V165_METADATA_PROGRESS', index, len(files), flush=True)
    scripts = 0
    for root in args.generator_root:
        for path in sorted(root.glob('*.py')):
            if path.name == Path(__file__).name:
                continue
            scripts += 1
            hits = metadata.literal_conflicts(path, set(CALIBRATION) | set(SAMPLERS.values()))
            for hit in hits:
                field = 'new_day_conflicts' if hit in CALIBRATION else 'new_sampler_conflicts'
                found[field].add(hit)
                references.append(dict(path=str(path), field=field, value=hit))
    summary = dict(**{k: sorted(v) for k, v in found.items()}, errors=errors,
        result_roots=len(roots), metadata_files=len(files), artifact_names=bundles, generators=scripts,
        scope='V159 recorded days plus current known FreqDuet result metadata, artifact names and generator '
            'literals/ranges; raw arrays, traces and compressed archive contents are not opened')
    write(args.out_root/'details/freshness.json', dict(summary=summary, references=references))
    return summary


def registration(reference, model, diagnosis, freshness):
    require(reference['execution_source'] == '09553bc920', 'V161 source changed')
    require(model['status'] == 'sealed' and model['execution_source'] == '35ae15b185' and
        not model['confirmation_started'] and not model['confirmation_outcomes_read'] and
        not model['confirmation_release_eligible'] and
        model['support_admission']['decision'] == 'insufficient_support', 'requires blocked V163 seal')
    require(diagnosis['status'] == 'v164_readonly_sparse_key_audit_complete' and
        diagnosis['execution_source'] == '4366c6c4b1' and diagnosis['v163_counts_and_weighted_support_reproduced'],
        'requires completed V164 diagnosis')
    require(all(model['frozen_protocol'][k] == reference[k] for k in METHOD_FIELDS), 'frozen method changed')
    require(all(not freshness[k] for k in ('new_day_conflicts', 'new_sampler_conflicts',
        'confirmation_outcome_conflicts', 'errors')), 'fresh cohort or unopened confirmation is not established')
    old = reference['reservation']
    require(model['calibration_days'] == old['calibration_days'], 'old calibration cohort changed')
    confirm, actors = old['confirmation_days'], old['actors']
    require(len(CALIBRATION) == 64 and len(confirm) == 32 and len(actors) == 12 and
        not set(CALIBRATION) & set(old['calibration_days']+confirm), 'cohort size or isolation changed')
    samplers = {str(d): SAMPLERS[d] for d in CALIBRATION}
    samplers.update({str(d): old['sampler_seeds'][str(d)] for d in confirm})
    require(len(set(samplers.values())) == 96, 'calibration/confirmation sampler overlap')
    result = dict(status='preregistered_not_collected', stage='v165_fixed_fresh_64day_calibration',
        **{k: deepcopy(reference[k]) for k in METHOD_FIELDS},
        reservation=dict(calibration_days=list(CALIBRATION), confirmation_days=deepcopy(confirm),
            sampler_seeds=samplers, actors=deepcopy(actors), modes=deepcopy(old['modes']), checkpoint_ep=199,
            excluded_development_days=deepcopy(old['calibration_days']),
            calibration_paths=64*len(actors)*2, confirmation_paths=32*len(actors)*2,
            split_unit='whole demand day and its sampler, shared across actors and old/handoff',
            size_reason='fixed doubling of independent calibration days after V164 widespread sparse follower '
                'contexts; a resource/support design, not a power guarantee or target-until-pass rule'),
        isolation=deepcopy(reference['isolation']), freshness=deepcopy(freshness),
        parents=dict(protocol='09553bc920', capture='25a2eddeaf', sealed_maps='35ae15b185', diagnosis='4366c6c4b1'),
        execution=deepcopy(reference['execution']))
    result['isolation'].update(
        calibration_seal='all 1536 new calibration paths and metadata admitted, then freeze candidate/coarse '
            'joint maps and support; no old V162 atom is included in this fit',
        fixed_budget='collect exactly 64 assigned new days; no outcome-driven additions, replacements, '
            'subsets or optional stopping; a support failure ends this registered attempt',
        first='protocol only now; a separately authorized collector must pass passive equivalence admission',
        development='retain V163 failed admission unchanged; V162 days are development only, not added to new fit')
    return result


def run(args):
    require(not (args.out_root/'protocol.json').exists(), 'protocol already sealed; do not overwrite')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v165_*.py', '-v'], check=True)
    reference = read(args.v161_root/'protocol.json')
    model = read(args.v163_root/'model.json')
    diagnosis = read(args.v164_root/'summary/status.json')
    freshness = inventory(args, reference['reservation']['confirmation_days'])
    protocol = registration(reference, model, diagnosis, freshness)
    protocol['execution_source'] = os.environ['FREQDUET_DIAGNOSTIC_COMMIT']
    write(args.out_root/'protocol.json', protocol)
    require(read(args.out_root/'protocol.json') == protocol, 'protocol JSON roundtrip differs')
    write(args.out_root/'summary/status.json', dict(status='v165_fresh_64day_protocol_sealed_not_collected',
        execution_source=protocol['execution_source'], reservation=protocol['reservation'], freshness=freshness,
        frozen_method_sections=list(METHOD_FIELDS), frozen_method_exactly_matches_v161=True,
        estimator_implementation_source='35ae15b185', old_v163_maps_changed=False,
        confirmation_days_and_samplers_unchanged=True, collector_submitted=False, interval_fit=False,
        new_days_collected=0, new_rollouts=0, new_training=0, confirmation_outcomes_read=False,
        v151_changed=False, v146_changed=False, window_s=3600., tails=[.1, .1], main_policy_changed=False,
        next='Separately authorize and admit a passive new-cohort collector; confirmation remains unopened.'))
    print('V165_FIXED_FRESH_64DAY_PROTOCOL_SEALED_NO_COLLECTION', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('v159-root', 'v161-root', 'v163-root', 'v164-root', 'out-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--parent', type=Path, action='append', required=True)
    parser.add_argument('--generator-root', type=Path, action='append', required=True)
    run(parser.parse_args())
