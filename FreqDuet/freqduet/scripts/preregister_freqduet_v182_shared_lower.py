#!/usr/bin/env python3
"""Seal a shared lower-bound-error design; inspect calibration support metadata only."""

import argparse
from itertools import product
import os
from pathlib import Path
import subprocess
import sys

import preregister_freqduet_v181_lower_tail as prior


read, write, require, close = prior.read, prior.write, prior.require, prior.close
CALIBRATION, SIGNS, MIN_DAYS = prior.CALIBRATION, prior.SIGNS, prior.MIN_DAYS
TARGET_SCOPES = prior.previous.prior.SCOPES


def registration(model):
    result = prior.registration(model)
    old = result['candidate']
    result.update(stage='v182_shared_lower_bound_error', source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'])
    result['parents']['v181_support'] = '51b3f32e00'
    result['candidate'] = dict(count=1, parameter_key=['actor', 'mode', 'completed_global_service_sign'],
        maximum_parameters=len(model['actors'])*2*2, parameters_per_actor_mode=2,
        baseline_key=model['frozen_protocol']['conditioning']['candidate_key'],
        sign=list(SIGNS), threshold_s=0.,
        **{k: old[k] for k in ('target', 'outside_target', 'history', 'completion',
            'minimum_prefix_episodes', 'prefix_fallback', 'signal')},
        sharing='one lower-bound-error quantile per sign shared across all V172 keys and the target union '
            'within an actor/mode; no cross-actor or cross-mode pooling',
        excluded='V181 raw-error branch quantiles, fitted V181 parameters, extra thresholds, per-stop or per-scope offsets')
    result['support'].update(unit='distinct original calibration days per actor/mode/service-sign, unioning target scopes once',
        minimum_original_key_days=MIN_DAYS,
        prerequisite='native supported AND unchanged V172 original-key >=19-day support AND completed prefix >=9 episodes',
        missing='shared parameter <19 days or unseen sign/prefix: target abstains without fallback; '
            'original V172/native abstentions remain; non-target support unchanged',
        release='all48 shared parameters >=19 days AND unchanged six-scope support gate; '
            'if metadata cannot establish the full candidate mask, report unknown and stop, not zero-fill')
    result['future_estimator'] = dict(authorized_now=False,
        error='A=L172-T, where T=V151point-sum(same-query R/S/residence vector); '
            'L172 is the frozen, query-specific, physically truncated V172 lower bound',
        weights='within actor/mode/sign: equal contributing day mass, equal physical episode within day, '
            'equal query within episode, target union once; recompute these weights, do not use old-key weights',
        rank='D distinct days; k=floor((D+1)/10), j=D+1-k; delta_sign=inverse empirical day-mass CDF of A at j/D',
        lower='Lnew=min(U172,max(0,L172-delta_sign))', upper='Unew=U172 exactly',
        signs='delta may be positive or negative; do not tune, scale or clip it except for the fixed physical bound cap',
        geometry='retain each original key through L172/U172; do not pool unanchored total-error quantiles',
        diagnostics='report physical-cap/zero-width frequencies and fixed V180 state strata; no successful-subset selection',
        updating='frozen calibration parameter lookup; completed-prefix input may change causally, '
            'never update fitted parameters or V151/V172 during confirmation',
        finite_sample_scope='empirical recalibration on the original V172 calibration days, not exact conformal coverage')
    result['limitations'] = ('Shared lower-bound-error tails across keys are a modeling assumption, not implied by V180 '
        'linear correlations. Reusing the V172 calibration days is in-sample development; '
        'independent score/dual-tail gates decide adequacy. Support alone is not calibration or controller evidence.')
    result['support_input'] = 'V181 original64-day query-support metadata and frozen V172 map headers/cells only'
    return result


def shared_days(conditions):
    expected = set(product(CALIBRATION, TARGET_SCOPES, ('unavailable', *SIGNS)))
    identities = [(c['day'], c['scope'], c['condition']) for c in conditions]
    require(len(identities) == len(expected) and set(identities) == expected, 'condition metadata grid incomplete/duplicate')
    days = {s: set() for s in SIGNS}
    prefix_complete = True
    for c in conditions:
        fraction = c['baseline_fraction']
        if not c['eligible_episodes']:
            require(fraction is None, 'empty condition has a baseline estimate')
            continue
        require(fraction is not None, 'nonempty condition missing V172 support')
        if fraction > 0.:
            if c['condition'] == 'unavailable':
                prefix_complete = False
            else:
                days[c['condition']].add(c['day'])
    return days, prefix_complete


def release(parameters, baseline, prefix_complete):
    failed = [f"{r['actor']}/{r['mode']}/{r['sign']}" for r in parameters if not r['support_eligible']]
    if failed or not prefix_complete:
        return dict(decision='insufficient_support', failed_parameters=failed,
            prefix_complete=prefix_complete, candidate_mask_equals_v172=None, mode_scopes=None,
            reason='do not reconstruct query/episode masks from conditional means when support is missing')
    return dict(**baseline, failed_parameters=[], prefix_complete=True, candidate_mask_equals_v172=True)


def run(args):
    model = prior.parents(args)
    original_protocol = read(args.v181_root/'protocol.json')
    require(original_protocol['source'] == '51b3f32e00' and
        original_protocol['support']['minimum_days'] == MIN_DAYS and
        original_protocol['candidate']['threshold_s'] == 0. and
        original_protocol['frozen'] == model['frozen_protocol']['frozen'], 'requires unchanged V181/V172 protocol')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
        '-p', 'test_v182_*.py', '-v'], check=True)
    plan = registration(model)
    require(not (args.out_root/'protocol.json').exists(), 'never overwrite a preregistration')
    write(args.out_root/'protocol.json', plan)
    print('V182_PROTOCOL_SEALED_BEFORE_SUPPORT_COUNTS', flush=True)
    admitted = read(args.v181_root/'summary/status.json')
    require(admitted['status'] == 'v181_lower_tail_preregistered_support_complete' and
        admitted['source'] == '51b3f32e00' and admitted['calibration_paths'] == 1536 and
        admitted['original_v172_support_reproduced'] and not admitted['interval_fit'] and
        not admitted['confirmation_outcomes_read'] and admitted['protocol'] == original_protocol,
        'requires completed V181 original-calibration support, not fitted branch artifacts')
    parameters, cells, prefix_complete, checked = [], [], True, 0
    for chunk in model['chunks']:
        actor, mode = chunk['actor'], chunk['mode']
        table = read(args.v172_root/chunk['map_file'])
        source = read(args.v181_root/'cells'/f'{actor}_{mode}.json')
        require((table['execution_source'], table['actor'], table['mode']) == ('a5b45df520', actor, mode) and
            (source['source'], source['actor'], source['mode']) == ('51b3f32e00', actor, mode), 'support bucket source differs')
        require(all(e['day_count'] >= MIN_DAYS for e in table['candidate'] if e['status'] == 'finite'),
            'original V172 key-day requirement weakened')
        require(all(c['actor'] == actor and c['mode'] == mode for c in source['conditions']), 'condition actor/mode differs')
        days, complete = shared_days(source['conditions'])
        prefix_complete &= complete
        for sign in SIGNS:
            parameters.append(dict(actor=actor, mode=mode, sign=sign, days=len(days[sign]),
                contributing_days=sorted(days[sign]), support_eligible=len(days[sign]) >= MIN_DAYS, fitted=False))
        baseline = {(c['day'], c['scope']): c for c in table['cells']}
        for c in source['cells']:
            expected = baseline[c['day'], c['scope']]
            close(c['baseline_fraction'], expected['finite_fraction'])
            for k in ('eligible_queries', 'eligible_episodes', 'native_abstentions'):
                require(c[k] == expected[k], 'original support denominator differs')
        cells.extend(dict(c, baseline_fraction=c['finite_fraction']) for c in table['cells'])
        checked += source['checked_queries']
        print('V182_SUPPORT_METADATA', actor, mode, {s: len(days[s]) for s in SIGNS}, flush=True)
    require(len(parameters) == plan['candidate']['maximum_parameters'] == 48 and
        checked == admitted['checked_queries'], 'parameter grid/calibration input changed')
    baseline = prior.temporal.support_release(cells, model['actors'], list(CALIBRATION))
    support = release(parameters, baseline, prefix_complete)
    require(read(args.v172_root/'model.json') == model and
        read(args.v181_root/'protocol.json') == original_protocol and
        read(args.out_root/'protocol.json') == plan, 'frozen parent/preregistration changed')
    write(args.out_root/'summary/status.json', dict(status='v182_shared_lower_preregistered_support_complete',
        source=plan['source'], protocol=plan, calibration_paths=1536, support=support,
        parameters=parameters, v172_baseline_support=baseline,
        input_original_calibration_eligible_queries=checked,
        v181_record_unchanged=True, original_v172_key_support_preserved=True,
        mask_proof='all original V172-finite target queries have the unchanged strict prefix; '
            'if every shared parameter has >=19 days, candidate support equals V172 exactly; '
            'no per-query masks inferred by averaging condition-specific fractions',
        native_abstentions=admitted['reasons']['native_abstention'],
        v172_abstentions=admitted['reasons']['v172_abstention'],
        residual_files_opened=0, atom_files_opened=0, fitted_parameter_values=0, new_bounds=0,
        interval_fit=False, new_rollouts=0, new_training=0, confirmation_outcomes_read=False,
        confirmation_started=False, collector_submitted=False, main_policy_changed=False,
        next='Support feasibility only. No fitting or confirmation authorization; reserve an independent cohort '
            'with separate metadata freshness checks before any later authorized calibration seal.'))
    print('V182_PREREGISTERED_SHARED_SUPPORT_COMPLETE_NO_FIT', support['decision'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v166-root', 'v166-audit-root', 'v176-root', 'v172-root', 'v167-root', 'v180-root', 'v181-root'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
