#!/usr/bin/env python3
"""Reserve V182's independent confirmation cohort using server metadata only."""

import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import preregister_freqduet_v165_fresh_calibration as ledger


read, write, require = ledger.read, ledger.write, ledger.require
DAYS = tuple(range(3452001, 3452033))
SAMPLERS = {day: 1831000000+i for i, day in enumerate(DAYS)}
HEADERS = {'protocol.json', 'preregistration.json', 'registration.json', 'model.json',
    'manifest.json', 'reservation_intent.json'}
SKIP = ledger.SKIP | {'summary', 'prepare_summary'}
RAW = ledger.RAW | {'interval_predictions'}


def hits(value):
    return dict(days=sorted(set(DAYS) & ledger.extract.day_ids(value)),
        samplers=sorted(set(SAMPLERS.values()) & ledger.metadata.sampler_ids(value)))


def audit(roots, generators, inventory, excluded):
    known = {int(k) for k in inventory['day_references']} | {
        d for group in inventory['groups'] for d in group['days']}
    collisions = [dict(path='V159 archive inventory', days=sorted(known & set(DAYS)), samplers=[])] if known & set(DAYS) else []
    files, errors, names, source_dirs = set(), [], 0, set(generators)
    for root in sorted(roots):
        if not root.is_dir():
            errors.append(dict(path=str(root), error='known result root unavailable'))
            continue
        print('V183_METADATA_ROOT', str(root), flush=True)
        def walk_error(exc):
            errors.append(dict(path=exc.filename, error=str(exc)))
        for directory, children, filenames in os.walk(root, onerror=walk_error):
            path = Path(directory)
            if (path/'source/scripts').is_dir():
                source_dirs.add(path/'source/scripts')
            children[:] = sorted(d for d in children if d not in SKIP and path/d != excluded and
                len((path/d).relative_to(root).parts) <= 5)
            if path.name in RAW | ledger.RECEIPTS:
                children[:] = []
                for name in filenames:
                    names += 1
                    day = ledger.filename_day(name)
                    if day in DAYS:
                        collisions.append(dict(path=str(path/name), days=[day], samplers=[]))
            for name in filenames:
                if path.name not in RAW and name.endswith('.json') and (
                        name in HEADERS or path.name in ledger.RECEIPTS):
                    files.add(path/name)
    for index, path in enumerate(sorted(files), 1):
        try:
            found = hits(read(path))
            if found['days'] or found['samplers']:
                collisions.append(dict(path=str(path), **found))
        except (OSError, ValueError) as exc:
            errors.append(dict(path=str(path), error=str(exc)))
        if index % 5000 == 0:
            print('V183_METADATA_PROGRESS', index, len(files), flush=True)
    scripts = 0
    for root in sorted(source_dirs):
        if not root.is_dir():
            errors.append(dict(path=str(root), error='known generator root unavailable'))
            continue
        for path in sorted(root.glob('*.py')):
            if path.name == Path(__file__).name:
                continue
            scripts += 1
            try:
                values = ledger.metadata.literal_conflicts(path, set(DAYS) | set(SAMPLERS.values()))
                if values:
                    collisions.append(dict(path=str(path), days=sorted(set(DAYS) & values),
                        samplers=sorted(set(SAMPLERS.values()) & values)))
            except (OSError, ValueError, SyntaxError) as exc:
                errors.append(dict(path=str(path), error=str(exc)))
    return dict(decision='fresh' if not collisions and not errors and files and scripts else 'blocked',
        result_roots=len(roots), metadata_files=len(files), artifact_names=names,
        generator_directories=len(source_dirs), generator_files=scripts,
        collisions=collisions, errors=errors,
        scope='V159 recorded IDs, current known result roots (supported layouts to depth5), protocol/model/'
            'reservation headers, path/pair/capture receipts, raw artifact names and generator literals/ranges; '
            'reservations also conflict. Outcome summaries, raw bodies, interval maps/atoms and checkpoints not opened.')


def validate(protocol, support, model):
    require(protocol['source'] == 'a57afaa337' and
        protocol['confirmation']['future_days'] == 32 and protocol['confirmation']['paths'] == 768 and
        protocol['confirmation']['status'] == 'not_reserved_not_started' and
        not protocol['future_estimator']['authorized_now'] and not any(protocol['execution'].values()),
        'requires unchanged, unfitted V182 protocol')
    require(support['source'] == 'a57afaa337' and support['protocol'] == protocol and
        support['status'] == 'v182_shared_lower_preregistered_support_complete' and
        support['support']['decision'] == 'pass' and support['support']['candidate_mask_equals_v172'] and
        len(support['parameters']) == 48 and all(p['support_eligible'] and not p['fitted'] and p['days'] >= 19
            for p in support['parameters']) and not support['interval_fit'] and not support['confirmation_started'],
        'V182 support admission missing or execution already started')
    require(model['execution_source'] == 'a5b45df520' and model['status'] == 'sealed' and
        model['calibration_days'] == list(ledger.CALIBRATION) and model['actors'] == protocol['actors'] and
        model['frozen_protocol']['frozen'] == protocol['frozen'], 'frozen V172 parent differs')
    calibration = model['frozen_protocol']['reservation']['sampler_seeds']
    require(len(DAYS) == len(set(DAYS)) == len(SAMPLERS) == len(set(SAMPLERS.values())) == 32 and
        not set(DAYS) & set(model['calibration_days']) and
        not set(SAMPLERS.values()) & set(calibration.values()), 'cohort overlaps calibration or has duplicates')


def reservation(protocol, freshness):
    require(freshness['decision'] == 'fresh' and not freshness['collisions'] and not freshness['errors'],
        'day/sampler collision or incomplete audit; stop without replacing days')
    return dict(stage='v183_v182_confirmation_reservation', source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        status='reserved_not_fitted_not_collected', v182_protocol=deepcopy(protocol),
        confirmation_days=list(DAYS), sampler_seeds={str(d): SAMPLERS[d] for d in DAYS},
        actors=protocol['actors'], modes=protocol['modes'], paths=protocol['confirmation']['paths'],
        split_unit='whole demand day and sampler shared across all actors and both modes',
        fixed_budget=protocol['confirmation']['fixed_budget'], freshness=freshness,
        execution=deepcopy(protocol['execution']),
        release='separate authorization for original64-day fitting/seal; confirmation collection separately '
            'authorized only after seal admission. This reservation authorizes neither.',
        scope='metadata freshness/reservation only; no prediction, calibration or controller performance claim')


def repair_description_only(text):
    broken = "    result['estimator']['implementation'] = 'same V161 empirical ranks and joint atoms; future mixtures must recompute '\n        'equal-day/episode/query weights within each new key, never reuse old-key atom weights'"
    fixed = "    result['estimator']['implementation'] = (\n        'same V161 empirical ranks and joint atoms; future mixtures must recompute '\n        'equal-day/episode/query weights within each new key, never reuse old-key atom weights')"
    require(text.count(broken) == 1, 'not the observed V171 description-only syntax defect')
    return text.replace(broken, fixed)


def resolve_previous(args, protocol):
    old = read(args.previous_root/'summary/status.json')
    intent = read(args.previous_root/'reservation_intent.json')
    fresh = read(args.previous_root/'details/freshness.json')
    require(old['source'] == '4ed654a9b9' and old['status'] == 'v183_reservation_blocked' and
        intent['confirmation_days'] == list(DAYS) and
        intent['sampler_seeds'] == {str(d): SAMPLERS[d] for d in DAYS} and intent['v182_protocol'] == protocol and
        not fresh['collisions'] and len(fresh['errors']) == 1, 'requires the unchanged zero-conflict V183 audit')
    broken = Path(fresh['errors'][0]['path'])
    reference = args.reference_generator
    require(reference is not None and broken.parents[2].name == 'protocol_7c38e514fd' and
        reference.parents[2].name == 'protocol_f5d152f0bc' and
        broken.name == reference.name == 'preregister_freqduet_v171_temporal_tails.py' and
        repair_description_only(broken.read_text()) == reference.read_text(),
        'archived generators differ beyond the observed description parentheses; stop')
    require(not ledger.metadata.literal_conflicts(reference, set(DAYS) | set(SAMPLERS.values())),
        'equivalent V171 declarations conflict with fixed cohort')
    fresh.update(decision='fresh', errors=[], syntax_resolution=dict(original_errors=fresh['errors'],
        reference=str(reference), declarations_unchanged=True, original_files_modified=False,
        method='whole-source equality after only the known description-string parentheses/indentation repair; '
            'literal/range check on the equivalent valid source; no archive rescan or cohort replacement'))
    require(not (args.out_root/'reservation_intent.json').exists(), 'never overwrite cohort intent')
    write(args.out_root/'reservation_intent.json', intent)
    print('V183_SINGLE_SYNTAX_GAP_RESOLVED_NO_ARCHIVE_OR_COHORT_CHANGE', flush=True)
    return fresh


def run(args):
    protocol = read(args.v182_root/'protocol.json')
    support = read(args.v182_root/'summary/status.json')
    model = read(args.v172_root/'model.json')
    validate(protocol, support, model)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v183_*.py', '-v'], check=True)
    if args.previous_root:
        fresh = resolve_previous(args, protocol)
        finish(args, protocol, support, model, fresh)
        return
    require(not (args.out_root/'reservation_intent.json').exists(), 'never overwrite a fixed cohort intent')
    write(args.out_root/'reservation_intent.json', dict(status='fixed_before_metadata_audit',
        confirmation_days=list(DAYS), sampler_seeds={str(d): SAMPLERS[d] for d in DAYS}, v182_protocol=protocol))
    print('V183_FIXED_COHORT_BEFORE_AUDIT', DAYS[0], DAYS[-1], flush=True)
    inventory = read(args.inventory_file)
    require(all(p.is_dir() for p in args.parent), 'archive parent unavailable')
    roots = set(ledger.archives.result_roots(args.parent)) | {args.v182_root.parent.parent}
    for group in inventory['groups']:
        roots.update(p for p in Path(group['base']).parents if p.name == 'results_freqduet' and p.is_dir())
    fresh = audit(roots, args.generator_root, inventory, args.out_root.parent)
    finish(args, protocol, support, model, fresh)


def finish(args, protocol, support, model, fresh):
    write(args.out_root/'details/freshness.json', fresh)
    require(read(args.v182_root/'protocol.json') == protocol and
        read(args.v182_root/'summary/status.json') == support and read(args.v172_root/'model.json') == model,
        'frozen protocol/support/model changed')
    compact = {k: v for k, v in fresh.items() if k not in ('collisions', 'errors')}
    compact.update(collision_count=len(fresh['collisions']), error_count=len(fresh['errors']),
        collision_examples=fresh['collisions'][:5], error_examples=fresh['errors'][:5])
    result = dict(status='v183_confirmation_reserved' if fresh['decision'] == 'fresh' else 'v183_reservation_blocked',
        source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], v182_source=protocol['source'],
        confirmation_days=list(DAYS), sampler_seeds={str(d): SAMPLERS[d] for d in DAYS},
        actors=protocol['actors'], modes=protocol['modes'], future_paths=768, freshness=compact,
        v182_protocol_unchanged=True, frozen_parents_unchanged=True,
        interval_fit=False, fitted_parameter_values=0, new_bounds=0, raw_files_opened=0,
        new_rollouts=0, new_training=0, confirmation_outcomes_read=False,
        confirmation_started=False, collector_submitted=False, main_policy_changed=False,
        next='Separate authorization for original64-day fitting/seal; no confirmation collection yet.'
            if fresh['decision'] == 'fresh' else 'Resolve metadata admission gap; no fitting or collection authorization.')
    if fresh['decision'] == 'fresh':
        registered = reservation(protocol, fresh)
        write(args.out_root/'registration.json', registered)
        require(registered['v182_protocol'] == protocol, 'V182 snapshot differs')
    write(args.out_root/'summary/status.json', result)
    require(fresh['decision'] == 'fresh', 'metadata conflict/error; retain failed intent without substitution')
    print('V183_RESERVED_32_DAYS_768_FUTURE_PATHS_NO_FIT_NO_COLLECTION', compact, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v182-root', 'v172-root', 'inventory-file'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('parent', 'generator-root'):
        parser.add_argument('--'+name, type=Path, action='append', required=True)
    for name in ('previous-root', 'reference-generator'):
        parser.add_argument('--'+name, type=Path)
    run(parser.parse_args())
