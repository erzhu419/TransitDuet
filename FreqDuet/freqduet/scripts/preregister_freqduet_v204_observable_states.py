#!/usr/bin/env python3
"""Seal fixed-key observable-state tests; inventory existing original64 moments."""

import argparse
from itertools import combinations_with_replacement, product
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v203_parent_centering as previous


compact = previous.previous
base, saved = compact.base, compact.saved
read, write, require = compact.read, compact.write, compact.require
KEYS, DAYS = compact.KEYS, tuple(base.DAYS)
MODES, SCOPES = ('old', 'handoff'), ('clearance', 'hour13_moving_launch')
STATES = ('query_offset_min', 'first_fraction', 'horizon_min', 'native_leg_mean_min',
    'matched_mean_age_min', 'matched_n', 'service_prefix_under_min', 'predicted_entry', 'stage_command')
FIELDS = (*STATES, 'signed_remaining_s', 'first_running_under_s')
BINS = tuple(product(range(3), repeat=2))


def registration():
    return dict(stage='v204_fixed_key_observable_state_preregistration', source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        calibration_days=list(DAYS), keys=[list(k) for k in KEYS], modes=list(MODES), scopes=list(SCOPES),
        parents=dict(v199='71af51ae4d', v191='e9efbe9bae', v202='8829065fe1', v203='c26aa1f39b'),
        frozen=previous.registration()['frozen'],
        support='Unchanged V199 actor/key/bin masks:19 complete-input days in EACH fixed32-day half. '
            'Same9 fraction/horizon cells;no new bins,pooling,mask search or threshold changes.',
        weights='Raw V199 moments use ORIGINAL eligible-episode/query denominators. Multiply V191 saved '
            'parent/half factors ONCE,shared by ALL9 bins. Nominal equal12 actors/day;no selected-mass renormalization.',
        states=list(STATES),
        observability=dict(query_offset_min='query clock relative to observable reference-entry hour',
            first_fraction='current physical progress and route geometry', horizon_min='frozen point ETA,not future realized horizon',
            native_leg_mean_min='current native forecast from strictly completed running prefix',
            matched_mean_age_min='mean completion age of matched strictly completed first-road prefix',
            matched_n='count in that same completed prefix',
            service_prefix_under_min='strictly completed global service-prefix residual mean,truth minus forecast',
            predicted_entry='query-time observable entry-clock source indicator', stage_command='current known-command stage indicator'),
        hypotheses=['Within the fixed physical keys and admitted cells,do query-time state covariance and its '
            'J/Y loadings account linearly for the half-difference,including the V203 daily-mean/centering cancellation?',
            'Does that difference instead remain in mixed or residual terms under ONE common operator?'],
        future_test=dict(
            operator='Per mode/scope,ONE9-state additive linear operator for J and Y,pooled across BOTH fixed keys, '
                '12 actors,both halves and admitted cells. Reference is equal-half weighted within-cell/day covariance; '
                'not parent-centered cross mass. Standardize by pooled state SD,pinv rcond=1e-10 as V195. '
                'No per-key/per-half/per-bin fits,no feature selection,interactions or coefficient tuning.',
            degeneracy='A zero-variance state has no identifiable individual loading;report it,do not replace its signal. '
                'Collinear coefficients are not separately identifiable;only the common column-space projection is interpreted. '
                'No usable state variation or undefined resample means unknown,not zero effect or permission to change support.',
            centering='Compute FULL parent-key/day means of ALL selected variables from ALL9 cells BEFORE masking. '
                'Let fJ=betaJ*Z,fY=betaY*Z,rJ=J-fJ,rY=Y-fY. Apply the SAME full-parent/day centering to all four. '
                'Retain original raw and parent means to reproduce every V203 term.',
            algebra='For each supported cell evaluate (fJ,fY),(rJ,fY),(fJ,rY),(rJ,rY). '
                'Apply unchanged V203 within/between-cell/four-parent-centering/mean-product algebra and V198 '
                'symmetric average-mass times half-difference to EACH pair. Their sum MUST reproduce V203 '
                'points and individual draws,including mix. Never discard mixed terms or their signed cancellation.',
            unsupported='Unsupported mass and ORIGINAL aggregate JY contribution remain separate;do not project or '
                'decompose unsupported cells,renormalize,or generalize the supported diagnostic to them.',
            inference=dict(draws=10000, seed=20260803,
                clusters='shared actors;shared demand-day weights across actors,keys,modes/scopes;independent32-day halves',
                refit='Recompute the same pooled within-cell/day operator once per mode/scope in EACH draw,then reuse on both halves/keys.',
                primary='16 two-sided conditional-change contrasts:2 keys x2 modes x2 scopes x(state-state,mixed sum); '
                    'Bonferroni percentile alpha=.05/16. Existing V194/V198 gates are NOT replaced.',
                diagnostic='All state summaries,individual mixed/residual terms,V203 subparts and half endpoints use descriptive95% intervals. '
                    'Report direct differences and matched-draw cancellation,not differing significance or selected states.'),
            checks='Before effect reporting,reproduce V202 frozen masks/nominal masses and V203 original points/matched draws. '
                'If algebra or inputs differ,stop readout;do not repair by reweighting or changing masks.'),
        required=dict(fields=list(FIELDS), units='actor/day/mode/scope/fixed-parent-key/existing-bin',
            moments=['M0', 'M1(all11)', 'M2(all11 x all11,symmetric)'], packed_slots=78,
            daily='Aligned per-day cell masses/means supply between-day mean products,not joins of unrelated marginals.',
            parent='ALL9 cells needed for full parent/day means,including unsupported cells.',
            lineage='V199 full39 joint block and metadata inherit V194 query/source/physical-follower label clock admission. '
                'This inventory does not reconstruct individual clocks from aggregate facts.'),
        inventory_only=True, state_operator_fitted=False, state_associations_estimated=False,
        limitations='Reused development keys/days,not an independent confirmation. An additive projection describes linear '
            'co-movement,not causal state composition or conditional-law change. General state-dependent JY/J2/Y2 '
            'effects need indicator-conditioned third moments or finer joint cells AND their19-day support;ordinary '
            'second moments and marginal binary counts do not identify them. No such collection is authorized here. '
            'Clock admission is inherited,not a new per-query timing audit. No interval or policy improvement claim.',
        raw_capture_files_opened=0, confirmation_outcomes_read=False, confirmation_started=False,
        interval_fit=False, predictor_fit=False, new_replays=0, new_training=0, map_updates=0, main_policy_changed=False)


def requirements(names):
    missing = [name for name in FIELDS if name not in names]
    if missing:
        return None, missing
    n = len(names)
    pairs = {tuple(p): i for i, p in enumerate(zip(*np.triu_indices(n)))}
    ix = [names.index(name) for name in FIELDS]
    columns = [0, *[1+i for i in ix]]
    columns += [1+n+pairs[tuple(sorted((i, j)))] for i, j in combinations_with_replacement(ix, 2)]
    return columns, []


def check_support(mass, counts, keep):
    observed = np.stack([(part > 1e-12).sum(axis=0) for part in (mass[:32], mass[32:])])
    require(np.array_equal(counts, observed) and np.array_equal(keep, np.all(counts >= 19, axis=0)),
        'frozen V199 complete-input support differs;do not change mask')


def balanced_compact(moments, factors):
    result = moments[..., compact.COLS].copy()
    for h in range(2):
        result[h*32:(h+1)*32] *= factors[h, None, :, None, None]
    return result


def audit_actor(args, actor, mode, scope, names, columns, support):
    metadata = read(args.v199_root/'metadata'/f'{actor}_{mode}.json')
    require(metadata['source'] == '71af51ae4d' and metadata['paths'] == 64, 'V199 source/cohort differs')
    require(sorted(r['day'] for r in metadata['episode_denominators'] if r['scope'] == scope) == list(DAYS),
        'original eligible-episode denominator metadata incomplete')
    expected_keys = sorted(tuple(r['key']) for r in support if r['actor'] == actor)
    with np.load(args.v199_root/'cells'/f'{actor}_{mode}_{scope}.npz') as z:
        require(z['days'].tolist() == list(DAYS) and z['keys'].tolist() == [list(k) for k in expected_keys]
            and z['bins'].tolist() == [list(b) for b in BINS], 'V199 day/key/bin grid differs')
        width = 1+len(names)+len(names)*(len(names)+1)//2
        m, facts = z['moments'], z['facts']
        require(m.shape == (64, len(expected_keys), 9, width)
            and facts.shape == (64, len(expected_keys), 9, 1+len(saved.FACTS)), 'joint moment/fact block incomplete')
        keep, counts, factors = z['supported_both'], z['distinct_days'], z['parent_balance_factors']
    with np.load(args.v202_root/'days'/f'{actor}_{mode}_{scope}.npz') as z:
        require(z['days'].tolist() == list(DAYS) and z['keys'].tolist() == [list(k) for k in KEYS]
            and z['fields'].tolist() == previous.FIELDS, 'V202 reference grid differs')
        reference, ref_keep = z['raw'], z['supported_both']
    require(keep.shape == (len(expected_keys), 9) and counts.shape == (2, len(expected_keys), 9)
        and factors.shape == (2, len(expected_keys)), 'saved support/factor block incomplete')
    records, error = [], 0.
    for ki, key in enumerate(KEYS):
        if key not in expected_keys:
            require(not ref_keep[ki].any() and not np.any(reference[:, ki]), 'missing parent differs from V202')
            records.append(dict(key=list(key), parent_present=False, supported_cells=0, missing='not in frozen V191 common support'))
            continue
        j = expected_keys.index(key)
        cell = m[:, j:j+1]
        require(np.isfinite(cell[..., columns]).all(), 'required joint block contains missing/nonfinite moments')
        require(np.isfinite(facts[:, j]).all() and np.allclose(cell[:, 0, :, 0], facts[:, j, :, 0], rtol=0., atol=1e-12),
            'joint states and clock-admission mass differ')
        check_support(cell[:, 0, :, 0], counts[:, j], keep[j])
        mass = cell[..., 0].sum(axis=2)
        half_mass = np.stack([part.mean(axis=0) for part in (mass[:32], mass[32:])])
        expected_factors = half_mass.mean(axis=0)/half_mass
        require(np.allclose(factors[:, j:j+1], expected_factors, rtol=0., atol=1e-12), 'V191 saved balance differs')
        got = balanced_compact(cell, factors[:, j:j+1])[:, 0]
        require(np.array_equal(keep[j], ref_keep[ki]) and np.allclose(got, reference[:, ki], rtol=0., atol=1e-8),
            'V202 original moments/masks/once-only weights differ')
        error = max(error, float(np.max(np.abs(got-reference[:, ki]))))
        balanced_mass = got[..., 0]
        masses = np.stack([part.mean(axis=0) for part in (balanced_mass[:32], balanced_mass[32:])])
        records.append(dict(key=list(key), parent_present=True, required_joint_block_complete=True,
            supported_cells=int(keep[j].sum()), distinct_days=counts[:, j].tolist(), supported_both=keep[j].tolist(),
            parent_mass=masses.sum(axis=1).tolist(), retained_mass=(masses*keep[j]).sum(axis=1).tolist()))
    print('V204_EXISTING_JOINT_BLOCK', actor, mode, scope, flush=True)
    return dict(actor=actor, mode=mode, scope=scope, v202_max_abs_error=error, keys=records)


def run(args):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
        '-p', 'test_v204_*.py', '-v'], check=True)
    require(not (args.out_root/'protocol.json').exists(), 'do not overwrite preregistration')
    protocol = registration()
    write(args.out_root/'protocol.json', protocol)
    print('V204_STATE_TEST_SEALED_BEFORE_MOMENT_INVENTORY', flush=True)
    parent = read(args.v199_root/'summary/status.json')
    require(parent['source'] == '71af51ae4d' and parent['protocol']['calibration_days'] == list(DAYS)
        and parent['parent_all9_moments_reproduced'] and parent['conditional_support_frozen']
        and parent['source_and_label_clocks_checked'] and parent['original_episode_denominators_preserved'], 'requires admitted V199')
    ref = read(args.v203_root/'summary/status.json')
    require(ref['source'] == 'c26aa1f39b' and ref['v202_points_and_matched_draws_reproduced']
        and ref['masks_and_weights_unchanged'], 'requires V203 fixed estimand')
    names = parent['protocol']['required']['fields']
    columns, missing = requirements(names)
    require(names == list(saved.VARS) and parent['protocol']['packed_second_order'] == 'numpy.triu_indices(39)',
        'joint schema lineage differs;review before interpreting missing slots')
    actors = sorted({r['actor'] for r in read(args.v191_root/'details/old_clearance.json')['common_key_support']})
    require(len(actors) == 12, 'original actor grid differs')
    results = []
    if not missing:
        require(len(columns) == len(set(columns)) == 78 and set(STATES) <= set(saved.prior.CONTROLS),
            'required joint slots or query-time state provenance differ')
        for mode, scope in product(MODES, SCOPES):
            support = read(args.v191_root/'details'/f'{mode}_{scope}.json')['common_key_support']
            for actor in actors:
                results.append(audit_actor(args, actor, mode, scope, names, columns, support))
    absent = [dict(actor=r['actor'], mode=r['mode'], scope=r['scope'], key=k['key'], reason=k['missing'])
        for r in results for k in r['keys'] if not k['parent_present']]
    write(args.out_root/'summary/status.json', dict(status='v204_preregistration_and_joint_inventory_complete',
        source=protocol['source'], protocol=protocol, moment_files_checked=len(results), actors=actors,
        required_slots=len(columns) if columns is not None else None, missing_required_fields=missing,
        missing_parent_support=absent, results=results,
        additive_state_test_available=not missing and bool(results),
        nonlinear_state_test_available=False,
        nonlinear_gap='No per-bin Z*J*Y,Z*J^2,Z*Y^2 or new state-indicator joint blocks/distinct-day support. '
            'Do not reconstruct conditional effects from marginal counts;no archive scan authorized.',
        v202_max_abs_error=max((r['v202_max_abs_error'] for r in results), default=None),
        clock_admission='Inherited V194/V199;aggregate facts do not re-establish individual query clocks.',
        masks_and_weights_unchanged=True, state_operator_fitted=False, state_associations_estimated=False,
        raw_capture_files_opened=0, interval_fit=False, predictor_fit=False, confirmation_outcomes_read=False,
        confirmation_started=False, new_replays=0, new_training=0, map_updates=0, main_policy_changed=False))
    print('V204_INVENTORY_COMPLETE_NO_STATE_FIT', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v199-root', 'v191-root', 'v202-root', 'v203-root'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
