#!/usr/bin/env python3
"""Seal query-state covariance tests and inventory saved moment schemas only."""

import argparse
from itertools import product
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import numpy as np

import preregister_freqduet_v204_observable_states as plan


read, write, require = plan.read, plan.write, plan.require
SIGNALS = ('innovation_rms_s', 'service_prefix_under_min')
MIXED_FIELDS = tuple(s+'__residual_J__residual_Y' for s in SIGNALS)


def registration(frozen):
    return dict(stage='v208_query_observable_covariance_preregistration',
        source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], frozen=frozen,
        calibration_days=list(plan.DAYS), keys=[list(k) for k in plan.KEYS],
        modes=list(plan.MODES), scopes=list(plan.SCOPES), signals=list(SIGNALS),
        parents=dict(v199='71af51ae4d', v204='4b85f9159c', v205='09530a00dc', v207='8cb299b8b1'),
        support='Original V1999-bin masks,19 complete-input distinct days in EACH fixed32-day half. '
            'All12 actors/all64 days;no new bins,thresholds,deletions or selected-mass normalization.',
        observability=dict(innovation_rms_s='Remaining fraction times RMS of strictly completed matched '
            'running surprises,relative to their strictly pre-entry forecasts;not sample SD or future variability.',
            service_prefix_under_min='Truth-minus-prediction mean of strictly completed GLOBAL service-prefix '
            'episodes;unchanged V190 prerequisite of at least9 episodes. No current unfinished service label.',
            excluded='Day IDs,full-day means,V207 common shifts/profiles,future realized horizons and labels '
            'are not inputs. Query-time load remains unavailable;no late-load backfill.'),
        estimand=dict(operator='SAVED V205 point beta per mode/scope,shared by keys/halves/actors/bins. '
            'X=J-betaJ*Z,Y=first_running_under_s-betaY*Z. No refit or operator selection.',
            primary='Within EACH actor/day/key/bin:Cov_w(S,(X-pJ)*(Y-pY)),where pJ,pY use ALL9 '
                'full-parent/day cells before masking. S is centered only for the diagnostic covariance,not as a deployed feature.',
            decomposition='Let x,y,s be cell/day means. Primary=Cov_w(S,(X-x)*(Y-y)) '
                '+(x-pJ)*Cov_w(S,Y)+(y-pY)*Cov_w(S,X). Preserve both signed pieces.',
            weights='Same V207 fixed coefficients c_h=average-half cell mass/point-half cell mass. '
                'Average original raw cell cross masses over32 days,then nominal equal12 actors. '
                'Apply V191 parent/half factors ONCE. Unsupported mass remains separate.',
            relation='New fixed-reference observable association,not a replacement for V205/V206 refitted '
                'statistics or their gates. Relative-mean and covariance attribution remain distinct.'),
        inference=dict(draws=10000, seed=20260803,
            clusters='Original shared actor and demand-day multinomial draws;independent32-day halves; '
                'same draws across signals/keys/modes/scopes. Point beta and c_h stay fixed.',
            primary='16 all64 two-sided associations:2 signals x2 keys x2 modes x2 scopes, '
                'Bonferroni percentile alpha=.05/16. Report original-unit nominal cross masses.',
            replication='Only consider a replicated association if the all64 family interval excludes zero '
                'with the same sign in BOTH modes,and95% intervals in BOTH fixed halves have that sign '
                'in BOTH modes. Scope overlap is not independent replication.',
            secondary='Decomposition pieces and half differences descriptive95%;no feature selection, '
                'gate fitting,threshold search,conditional independence or causal-effect claim.',
            missing='Undefined zero-exposure/zero-variation results remain missing,not zero effects; '
                'do not relax support or substitute another signal.'),
        required=dict(unit='actor/day/mode/scope/physical-parent-key/existing-bin',
            existing='M0,all11 V204 first/second moments plus each S,S*X,S*Y. '
                'Full-parent means come from ALL9 existing cells.',
            additional='Exactly TWO aligned mixed sums:sum(w*S*X*Y),one per frozen signal, '
                'using SAVED V205 point beta and EXACT V199 complete-input query cohort/denominators.',
            fields=list(MIXED_FIELDS),
            identity='Cov_w(S,(X-pJ)*(Y-pY))=E[SXY]-s*E[XY] '
                '-pJ*(E[SY]-s*y)-pY*(E[SX]-s*x). Ordinary second moments do not determine E[SXY].',
            clocks='Retain per-path original/complete-input counts and violations for source completion<query, '
                'pre-entry baseline<source entry,service label availability<query,and first-road label>=query; '
                'same physical follower/query join. Aggregate clock sums cannot establish new alignment.',
            admission='Before any future association,match old M0/M1/M2,complete-input counts,masks, '
                'original episode/query denominators and saved beta;reject a changed query cohort. '
                'This registration authorizes neither archive scans nor supplementation.'),
        limitations='Original64 are reused development days. The two signals test covariance co-movement, '
            'not all conditional variance/tail risks. Retrospective centering is an outcome-analysis device, '
            'not a deployable input. Even replication is not held-future predictive gain or controller benefit. '
            'Whole-day feature means times V207 profiles cannot substitute aligned query mixed sums. '
            'No full third/fourth-order tensor is needed for this fixed-operator test.',
        interval_fit=False, predictor_fit=False, associations_estimated=False, confirmation_started=False,
        raw_capture_files_opened=0, new_replays=0, new_training=0, map_updates=0, main_policy_changed=False)


def centered_cross(mass, s, x, y, xy, sx, sy, sxy, parent_x, parent_y):
    """Unnormalized state/cross-product covariance from aligned weighted sums."""
    require(mass > 0., 'zero exposure has no covariance')
    mean_s = s/mass
    return sxy-mean_s*xy-parent_x*(sy-mean_s*y)-parent_y*(sx-mean_s*x)


def npz_headers(path):
    result = {}
    with ZipFile(path) as archive:
        for entry in archive.namelist():
            with archive.open(entry) as f:
                version = np.lib.format.read_magic(f)
                require(version in ((1, 0), (2, 0)), 'unexpected project NPZ array format')
                header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                    else np.lib.format.read_array_header_2_0)(f)
                shape, _, dtype = header
                result[entry.removesuffix('.npy')] = dict(shape=list(shape), dtype=str(dtype))
    return result


def requirements(names):
    return [n for n in (*plan.FIELDS, *SIGNALS) if n not in names]


def audit_cells(root, admission, names):
    records, retained = [], 0
    for row in admission['results']:
        actor, mode, scope = (row[k] for k in ('actor', 'mode', 'scope'))
        path = root/'cells'/f'{actor}_{mode}_{scope}.npz'
        headers = npz_headers(path)
        shape = headers['moments']['shape']
        require(shape[0] == 64 and shape[2:] == [9, 1+len(names)+len(names)*(len(names)+1)//2],
            'V199 schema changed;do not infer availability from old admission')
        require(headers['facts']['shape'] == [64, shape[1], 9, 1+len(plan.saved.FACTS)], 'clock-fact schema differs')
        # Only small cohort/support arrays are read;M0/M1/M2 payload was admitted in V204.
        with np.load(path) as z:
            require(z['days'].tolist() == list(plan.DAYS) and z['bins'].tolist() == [list(b) for b in plan.BINS],
                'original cohort or bins changed')
            keys, keep, counts = z['keys'].tolist(), z['supported_both'], z['distinct_days']
        require(keep.shape == (shape[1], 9) and counts.shape == (2, shape[1], 9), 'support schema differs')
        for key in row['keys']:
            if not key['parent_present']:
                require(key['key'] not in keys, 'absent parent admission changed')
                continue
            ki = keys.index(key['key'])
            require(keep[ki].tolist() == key['supported_both'] and counts[:, ki].tolist() == key['distinct_days'],
                'saved V204 support changed;stop without changing masks')
            retained += int(keep[ki].sum())
        records.append(dict(actor=actor, mode=mode, scope=scope, second_order_slots=shape[-1],
            arrays=list(headers), missing_mixed_fields=list(MIXED_FIELDS)))
        require(set(headers) == {'days', 'keys', 'bins', 'moments', 'facts', 'parent_balance_factors',
            'distinct_days', 'supported_both'}, 'unexpected V199 arrays;review rather than assume no extra moments')
    return records, retained


def run(args):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
        '-p', 'test_v208_*.py', '-v'], check=True)
    parent = read(args.v207_root/'summary/status.json')
    require(parent['source'] == '8cb299b8b1' and parent['shared_covariance_identities_checked']
        and parent['masks_and_weights_unchanged'] and parent['protocol']['calibration_days'] == list(plan.DAYS),
        'requires completed original64 V207')
    require(not (args.out_root/'protocol.json').exists(), 'do not overwrite preregistration')
    protocol = registration(parent['protocol']['frozen'])
    write(args.out_root/'protocol.json', protocol)
    print('V208_QUERY_COVARIANCE_PROTOCOL_SEALED_BEFORE_INVENTORY', flush=True)
    admission = read(args.v204_root/'summary/status.json')
    source = read(args.v199_root/'summary/status.json')
    require(admission['source'] == '4b85f9159c' and admission['additive_state_test_available']
        and len(admission['actors']) == 12, 'requires existing complete second-order admission')
    names = source['protocol']['required']['fields']
    require(source['source'] == '71af51ae4d' and names == list(plan.saved.VARS)
        and set(SIGNALS) <= set(plan.saved.prior.SIGNALS+plan.saved.prior.CONTROLS), 'query-state schema differs')
    missing = requirements(names)
    require(not missing, 'required existing fields missing;stop before interpreting covariance')
    records, retained = audit_cells(args.v199_root, admission, names)
    operators, profiles = [], []
    for mode, scope in product(plan.MODES, plan.SCOPES):
        with np.load(args.v205_root/'details'/f'{mode}_{scope}.npz') as z:
            require(z['states'].tolist() == list(plan.STATES) and z['keys'].tolist() == [list(k) for k in plan.KEYS]
                and z['beta_point'].shape == (2, 9) and np.isfinite(z['beta_point']).all(), 'saved point operator differs')
            beta = z['beta_point']
        with np.load(args.v207_root/'details'/f'{mode}_{scope}.npz') as z:
            require(np.array_equal(z['point_operator'], beta), 'V207 fixed-reference beta differs from V205')
        operators.append(dict(mode=mode, scope=scope, saved_point_beta_available=True))
        headers = npz_headers(args.v207_root/'details'/f'{mode}_{scope}.npz')
        require(headers['daily_profiles']['shape'] == [64, 12, 2, 13], 'V207 profile schema differs')
        profiles.append(dict(mode=mode, scope=scope, arrays=list(headers), query_aligned_state_cross_products=False))
    write(args.out_root/'summary/status.json', dict(status='v208_protocol_sealed_query_mixed_moments_missing',
        source=protocol['source'], protocol=protocol, moment_files_checked=len(records),
        supported_actor_bin_cells=retained, candidate_actor_bin_cells=len(admission['actors'])*2*9*4,
        records=records, operators=operators, daily_profile_schemas=profiles,
        existing_fields_complete=True, saved_second_order_admission_reused=True,
        query_covariance_test_available=False, missing_aligned_fields=list(MIXED_FIELDS),
        next_decision='Existing moments cannot execute the registered test. Minimum next stage is an explicitly '
            'authorized read-only original64 archive pass to add ONLY two fixed-beta mixed sums and same-cohort '
            'clock/support admission,before any associations;no new replay,training or confirmation needed.',
        masks_and_weights_unchanged=True, moment_payloads_read=0, raw_capture_files_opened=0,
        interval_fit=False, predictor_fit=False, associations_estimated=False, confirmation_started=False,
        new_replays=0, new_training=0, map_updates=0, main_policy_changed=False))
    print('V208_COMPLETE_MISSING_TWO_ALIGNED_MIXED_SUMS_NO_FIT', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v199-root', 'v204-root', 'v205-root', 'v207-root'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
