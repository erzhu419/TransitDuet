#!/usr/bin/env python3
"""Seal conditional-moment requirements and audit existing original64 summaries."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v197_key_states as previous


saved, base, common = previous.saved, previous.base, previous.common
read, write, require = previous.read, previous.write, previous.require
DAYS, MIN_DAYS = previous.DAYS, 19
STAGES = (179, 180, *range(187, 198))
LAYOUTS = {179: (21, 'completed component prefix/full-ETA residual,not J/first-road Y'),
    180: (27, 'service prefix and full-ETA/service residuals,not J/first-road Y'),
    187: (30, 'running/service prefix and summed remaining-running error,not same-road J/first-road Y'),
    188: (210, 'prefix/clocks and summed remaining-running truth,not same-road J/first-road Y'),
    189: (7, 'leg-support counts,not joint outcome moments'),
    190: (base.prior.SIZE, 'J and full-ETA tail outcomes,not first-road Y'),
    194: (saved.SIZE, 'unconditional39-variable joint moments,not query-state conditional cells')}
BINARIES = ('predicted_entry', 'stage_command')


def registration():
    return dict(stage='v198_conditional_moment_requirements_and_availability', source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        calibration_days=list(DAYS), frozen=saved.registration()['frozen'], moment_source='51a9409ec2',
        required=dict(unit=['actor', 'day', 'mode', 'scope', 'direction', 'road', 'reference_entry_hour', 'fraction_bin', 'horizon_bin'],
            state_bins=dict(first_fraction=['[0,1/3)', '[1/3,2/3)', '[2/3,1]'],
                horizon_min=['[0,5)', '[5,10)', '[10,infinity)']),
            fields=list(saved.VARS), cells_per_base_key=9, statistics=['sum(w)', 'sum(w*v)', 'sum(w*v*v_transpose)'],
            alignment='Same admitted queries and weights for ALL cross terms;never synthesize a joint table by joining marginal summaries.',
            weights='Original V194 query/eligible-episode denominators,then fixed V191 support-only balance; '
                'sum ALL9 cells reproduces parent mass/first/second moments;never renormalize selected cells.',
            clocks='All bin inputs observable at query;strict completed sources and same physical follower/first-arrival Y label as V194. '
                'Keep admission and original eligible-episode metadata;future label times never define bins.',
            support='Each actor/mode/scope/base-common-key/conditional-cell has>=19 distinct complete-input days in EACH half. '
                'Fix masks before inference,no new key pooling,bin merging or19-day relaxation.'),
        estimand='Registered finite-bin resolution only,not complete conditioning on every continuous state or causality.',
        future_readout=dict(centering='Use FULL original base-key/day means to center J,Y before conditional aggregation,including '
                'queries later excluded for conditional support. Per-day conditional moments are essential.',
            algebra='For eligible cell s in each fixed actor/base-key: M_hs=original balanced mass; '
                't_hs=E[(J-mu_base_day_J)*(Y-mu_base_day_Y)|s]. '
                'Delta(sum M*t)=sum DeltaM*average(t)+sum average(M)*Delta(t). '
                'Split t into conditional covariance plus product of conditional centered means.',
            interpretation='At balanced point base-key mass,first term describes stratum exposure mix;second is conditional cross-moment change. '
                'Neither is causal. Preserve unsupported mass and its parent covariance contribution separately.',
            inference='If data later become available:10000 shared-actor/independent-half-day draws,seed20260803, '
                'fixed support/balance;8 two-sided primary delta contrasts(2 modes x2 scopes xmix/conditional),alpha=.05/8. '
                'Other decompositions diagnostic95%;V195 gates unchanged.'),
        audit=dict(stages=list(STAGES), scope='Canonical original64 moment lineage plus immediate run snapshots in these families; '
                'one schema sample per snapshot,ALL24 canonical V194 actor/mode cells for binary support. '
                'No queries,residual streams,records,NPZ,checkpoints or confirmation outcomes opened.',
            binary='M0,M1(B),M2(B,J),M2(B,Y) can provide binary support/marginal means,but '
                'sum(w*B*J*Y),sum(w*B*J^2),sum(w*B*Y^2) are not in ordinary second moments. '
                'Separate binary support is diagnostic,not a substitute for fraction/horizon cells.',
            binaries=list(BINARIES), rounding_mass_tolerance=1e-12,
            continuous='Neither overall moments nor third moments without indicator-conditioned sums identify the registered fixed-bin moments. '
                'Missing conditional moments imply UNKNOWN conditional19-day support,never zero days.'),
        limitations='Finite-bin conditional changes still cannot separate within-bin nonlinear state mixing from every conditional-law change. '
            'This stage audits availability/support only,does not estimate conditional associations or collect moments.',
        interval_fit=False, predictor_fit=False, conditional_associations_estimated=False,
        collection_started=False, confirmation_started=False, new_replays=0, new_training=0)


def stratum(fraction, horizon):
    require(-1e-8 <= fraction <= 1.+1e-8 and np.isfinite(horizon) and horizon >= 0., 'invalid query-time bin inputs')
    fraction = np.clip(fraction, 0., 1.)
    return int(np.searchsorted([1./3, 2./3], fraction, side='right')), int(np.searchsorted([5., 10.], horizon, side='right'))


def binary_mass(m, name):
    i = saved.VARS.index(name)
    mass, one = m[..., 0], m[..., 1+i]
    packed = next(k for k, pair in enumerate(zip(*saved.TRI)) if pair == (i, i))
    require(np.allclose(m[..., 1+saved.N+packed], one, rtol=1e-9, atol=1e-12)
        and np.all(one >= -1e-12) and np.all(one <= mass+1e-12), 'binary indicator moment differs')
    return np.stack((np.maximum(mass-one, 0.), np.maximum(one, 0.)), axis=-1)


def supported(mass):
    counts = np.stack([(v > 1e-12).sum(axis=0) for v in (mass[:32], mass[32:])])
    return counts, np.all(counts >= MIN_DAYS, axis=0)


def schema_sample(path):
    payload = read(path)
    rows = payload.get('rows', payload.get('cells', []))
    days, widths, lengths, names = set(), set(), set(), set(payload)
    for row in rows:
        if not isinstance(row, dict):
            continue
        if 'day' in row:
            days.add(row['day'])
        names.update(row)
        for scope in row.get('scopes', [row]):
            names.update(scope)
            for group in scope.get('groups', scope.get('signals', [])):
                names.update(group)
                v = group.get('moments', group.get('values'))
                if v is not None:
                    widths.add(len(v))
                if 'key' in group:
                    lengths.add(len(group['key']))
    return dict(source=payload.get('source', payload.get('execution_source')), original64_days_in_sample=days == set(DAYS),
        sample_days=len(days), vector_widths=sorted(widths), key_widths=sorted(lengths), field_names=sorted(names))


def inventory(root):
    results = []
    for stage in STAGES:
        for family in sorted(root.glob(f'protocol_v6_v{stage}_*')):
            for run in sorted(p for p in family.iterdir() if p.is_dir()):
                cells = sorted((run/'cells').glob('*.json'))
                if not cells and not (run/'summary/status.json').exists():
                    continue
                row = dict(stage=stage, root=str(run), cell_files=len(cells))
                if cells:
                    row['sample'] = schema_sample(cells[0])
                    row['sample_file'] = cells[0].name
                else:
                    row['sample'] = dict(note='summary-only derived statistics,no primitive per-day conditional cells')
                if stage in LAYOUTS:
                    width, description = LAYOUTS[stage]
                    row.update(known_vector_width=width, known_semantics=description)
                    if cells:
                        row['layout_matches_known_moments'] = row['sample']['vector_widths'] in ([], [width])
                else:
                    row['known_semantics'] = 'derived readout of earlier moments,no new query-state moment partition in its producer'
                row['requires_schema_review'] = (row.get('layout_matches_known_moments') is False or
                    any('conditional' in k or 'fraction_bin' in k or 'horizon_bin' in k for k in row['sample'].get('field_names', [])))
                results.append(row)
    return results


def binary_support(args, actors, mode):
    support = {scope: read(args.v191_root/'details'/f'{mode}_{scope}.json')['common_key_support'] for scope in saved.SCOPES}
    records = []
    for actor in actors:
        payload = read(args.v194_root/'cells'/f'{actor}_{mode}.json')
        require(payload['source'] == '51a9409ec2' and all((r['actor'], r['mode']) == (actor, mode)
            and r['v190_moments_reproduced'] for r in payload['rows']), 'V194 source/identity differs')
        for scope in saved.SCOPES:
            m = base.actor_matrix(payload['rows'], scope, support[scope], actor)
            require(m.shape[-1] == saved.SIZE, 'V194 schema width differs')
            for row in payload['rows']:
                s = next(s for s in row['scopes'] if s['scope'] == scope)
                require(all(set(g) == {'key', 'moments', 'facts'} and len(g['key']) == 3 for g in s['groups']),
                    'unexpected V194 group fields;review before declaring conditional fields unavailable')
            keys = sorted(tuple(k['key']) for k in support[scope] if k['actor'] == actor)
            for name in BINARIES:
                counts, keep = supported(binary_mass(m, name))
                records.extend(dict(actor=actor, mode=mode, scope=scope, key=list(key), binary=name, level=level,
                    first_days=int(counts[0, ki, level]), last_days=int(counts[1, ki, level]),
                    supported_both=bool(keep[ki, level])) for ki, key in enumerate(keys) for level in (0, 1))
        print('V198_EXISTING_MOMENT_SUPPORT', mode, actor, flush=True)
    return records


def run(args):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
        '-p', 'test_v198_*.py', '-v'], check=True)
    parent = read(args.v197_root/'summary/status.json')
    require(parent['source'] == '87cad6a8a8' and parent['protocol']['calibration_days'] == list(DAYS)
        and parent['v196_points_and_intervals_reproduced'] and not parent['interval_fit'] and not parent['main_policy_changed'],
        'requires frozen V197 original64 diagnostic')
    require(not (args.out_root/'protocol.json').exists(), 'do not overwrite preregistration')
    plan = registration()
    write(args.out_root/'protocol.json', plan)
    print('V198_CONDITIONAL_REQUIREMENTS_SEALED_NO_COLLECTION', flush=True)
    artifacts = inventory(args.results_root)
    actors = sorted({r['actor'] for r in read(args.v191_root/'details/old_clearance.json')['common_key_support']})
    require(len(actors) == 12, 'actor grid differs')
    records = [r for mode in ('old', 'handoff') for r in binary_support(args, actors, mode)]
    support = []
    for mode in ('old', 'handoff'):
        for scope in saved.SCOPES:
            for name in BINARIES:
                for level in (0, 1):
                    rows = [r for r in records if (r['mode'], r['scope'], r['binary'], r['level']) == (mode, scope, name, level)]
                    support.append(dict(mode=mode, scope=scope, binary=name, level=level, parent_common_keys=len(rows),
                        supported_both=sum(r['supported_both'] for r in rows),
                        first_day_range=[min(r['first_days'] for r in rows), max(r['first_days'] for r in rows)],
                        last_day_range=[min(r['last_days'] for r in rows), max(r['last_days'] for r in rows)]))
    review = [r['root'] for r in artifacts if r['requires_schema_review']]
    write(args.out_root/'details/binary_support.json', dict(records=records))
    write(args.out_root/'summary/status.json', dict(status='v198_availability_audit_complete', source=plan['source'], protocol=plan,
        inventory=artifacts, schema_review_required=review, binary_support=support,
        conditional_readout='requires_schema_review' if review else 'not_evaluable_from_existing_moments',
        observed_required_conditional_cells=None, supported_required_conditional_cells=None,
        missing=['fraction_bin x horizon_bin per-day joint sums for J,first-road Y and inherited controls',
            'conditional JY/J2/Y2 and remaining full39 second-moment block',
            'distinct complete-input days in EACH half for those conditional cells'],
        parent_common_key_support_checked=True, existing_v194_paths_checked=1536,
        raw_capture_files_opened=0, conditional_associations_estimated=False, collection_started=False,
        interval_fit=False, predictor_fit=False, confirmation_outcomes_read=False, confirmation_started=False,
        new_replays=0, new_training=0, map_updates=0, main_policy_changed=False))
    print('V198_AVAILABILITY_COMPLETE_NO_COLLECTION', 'review' if review else 'missing_conditional_joint_moments', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'results-root', 'v194-root', 'v191-root', 'v197-root'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
