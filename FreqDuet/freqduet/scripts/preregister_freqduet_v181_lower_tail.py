#!/usr/bin/env python3
"""Preregister one lower-tail candidate and count original-calibration support only."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from itertools import product, zip_longest
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v179_calibration_prefix as previous
import preregister_freqduet_v171_temporal_tails as temporal


read, write, require, close = previous.read, previous.write, previous.require, previous.close
CALIBRATION = previous.DAYS
SIGNS = ('nonpositive', 'positive')
MIN_DAYS = 19


def service_sign(history):
    if history['n'] < previous.MINIMUM or history['mean'] is None:
        return -1
    return int(history['mean'][1] > 0.)


def target_mask(bits, moving):
    return ((bits & 4) != 0) | moving


def support_mask(baseline, target, codes, days):
    eligible = baseline & target & (codes >= 0)
    pairs = np.unique(np.column_stack((codes[eligible], days[eligible])), axis=0)
    counts = dict(zip(*np.unique(pairs[:, 0], return_counts=True))) if len(pairs) else {}
    finite = baseline & ~target
    indices = np.flatnonzero(eligible)
    finite[indices] = np.array([counts[k] >= MIN_DAYS for k in codes[indices]], dtype=bool)
    return finite, counts


def parents(args):
    previous.parents(args)
    model = read(args.v172_root/'model.json')
    result = read(args.v180_root/'summary/status.json')
    require(model['execution_source'] == 'a5b45df520' and model['status'] == 'sealed' and
        model['calibration_days'] == list(CALIBRATION) and model['capture_source'] == '20d914af03' and
        Path(model['capture_root']) == args.v166_root, 'requires original V172/V166 calibration')
    require(result['status'] == 'v180_calibration_service_increment_complete' and
        result['source'] == '87c79be22d' and result['paths'] == 1536 and
        result['calibration_days'] == list(CALIBRATION) and not result['interval_fit'] and
        not result['confirmation_outcomes_read'], 'requires completed calibration-only V180')
    return model


def registration(model):
    base = model['frozen_protocol']
    return dict(status='preregistered_support_only', stage='v181_binary_service_lower_tail',
        source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        calibration_days=list(CALIBRATION), actors=model['actors'], modes=['old', 'handoff'],
        parents=dict(v172='a5b45df520', capture='20d914af03', v180='87c79be22d'),
        frozen=deepcopy(base['frozen']),
        candidate=dict(count=1, base_key=base['conditioning']['candidate_key'],
            extra_field='completed_global_service_sign', sign=list(SIGNS), threshold_s=0.,
            target='query_phase clearance OR (query hour13 AND native_stage moving-launch); union once',
            outside_target='copy both V172 bounds and support unchanged; no service-prefix requirement',
            history='V179 first field-complete eligible V151 query per physical follower-trip/bus/stop/direction episode',
            completion='label available_time_s < query_time_s; [query_time_s-3600,query_time_s) label window',
            minimum_prefix_episodes=9, prefix_fallback='none',
            signal='unweighted service-residual mean over those completed episodes; positive iff >0, otherwise nonpositive',
            excluded='same-distance history, full V180 diagnostic state grid, tuned thresholds, extra bins, stop exceptions'),
        support=dict(minimum_days=MIN_DAYS, unit='distinct original calibration days per V172-key/sign',
            prerequisite='native supported AND V172 finite AND required prefix available',
            missing='target prefix unavailable, unseen key/sign or fewer than19 days: abstain; never fall back to V172/coarse',
            no_rescue='original V172/native abstentions remain abstentions, including outside targets',
            denominator=base['support']['denominator'],
            release='same V171 six-scope support gate: complete actor/day grids, finite episode in every cell, '
                'mean original-denominator finite fraction >=0.8 for every mode/scope; otherwise stop without regrouping'),
        future_estimator=dict(authorized_now=False, error='V151 prediction minus truth; sum the same R/S/residence vector',
            weights='recompute equal day / equal physical episode within day / equal query within episode '
                'within each new key/sign; do not reuse V172 atom weights',
            rank='D distinct days; k=floor((D+1)/10), j=D+1-k; Qhigh=inverse empirical day-mass CDF(j/D)',
            lower='Lnew=min(U172,max(0,V151point-Qhigh)); retain physical cap and report cap/zero-width rates',
            upper='Unew=U172 exactly; do not estimate Qlow or use V176 upper envelope',
            non_target='V172 interval unchanged',
            updating='sealed calibration mapping only; strictly completed observations may update the fixed input '
                'prefix, never point predictions or fitted ranks/maps',
            finite_sample_scope='existing empirical day-mass ranks, not an exact conditional coverage theorem'),
        comparator=dict(primary='frozen V172', source='a5b45df520',
            matching='paired candidate/V172 scores and both tails on identical candidate-finite queries; '
                'also report V172 own-mask results and all original-denominator abstentions',
            score=base['inference']['score'],
            identity='delta score=(L172-Lnew)+10*((Lnew-T)+-(L172-T)+); upper-tail penalty delta exactly zero'),
        inference=dict(scopes=list(temporal.SCOPES), modes=['old', 'handoff'],
            reduction=base['inference']['reduction'], bootstrap=base['inference']['bootstrap'],
            draws=5000, seed=20260803, gate=deepcopy(base['gates']),
            score_gate='V172 paired upper95 <0 in all/clearance; <=0 in all other scopes, both modes',
            tails_gate='coverage lower95 >=0.8; each tail upper95 <=0.1; finite fraction lower95 >=0.8',
            sign_diagnostics='report both signs and unavailable exposure/support without choosing successful subsets',
            controller_admitted=False),
        confirmation=dict(status='not_reserved_not_started', future_days=32, paths=32*12*2,
            assignment='separate metadata-only freshness/reservation before any fitting; exclude all consumed days/samplers',
            fixed_budget='one reserved cohort, no outcome-dependent additions/replacements or optional stopping',
            release='separate authorization, support pass, independent cohort reservation, then calibration seal '
                'before any confirmation collection'),
        execution=dict(interval_fit=False, new_rollouts=0, new_training=0, confirmation_outcomes_read=False,
            collector_submitted=False, confirmation_started=False, main_policy_changed=False))


def bucket(args, model, chunk):
    actor, mode = chunk['actor'], chunk['mode']
    table = read(args.v172_root/chunk['map_file'])
    require((table['execution_source'], table['actor'], table['mode']) == ('a5b45df520', actor, mode), 'V172 bucket differs')
    key_index = {tuple(e['key']): i for i, e in enumerate(table['candidate'])}
    with np.load(args.v172_root/chunk['atom_file']) as archive:
        requests, native, baseline = archive['requests'], archive['native_supported'], archive['finite_candidate']
        episodes = archive['episode_keys']
    days, episode_ids = requests[:, 0].astype(int), requests[:, 1].astype(int)
    require(set(days) == set(CALIBRATION), 'not original calibration days')
    signs = np.full(len(requests), -1, dtype=np.int8)
    codes = np.full(len(requests), -1, dtype=np.int64)
    moving = np.zeros(len(requests), dtype=bool)
    checked = 0
    for count, day in enumerate(CALIBRATION, 1):
        root, name = args.v166_root/'calibration', f'{actor}_{day}_{mode}'
        receipt = read(root/'capture_cells'/(name+'.json'))
        require((receipt['execution_source'], receipt['seed'], receipt['day'], receipt['mode']) ==
            ('20d914af03', actor, day, mode) and not receipt['interval_fit'] and not receipt['main_policy_changed'], 'capture differs')
        lookup = {tuple(requests[i, 5:].astype(int)): i for i in np.flatnonzero(days == day)}
        atoms, selected, records = {}, [], 0
        for q, r in zip_longest(*(previous.stream(root/d/(name+'.jsonl.gz')) for d in ('queries', 'residuals'))):
            require(q is not None and r is not None and q['query_key'] == r['query_key'] and
                q['query_time_s'] == r['query_time_s'] and q['episode'] == r['episode'] and
                (q['actor'], q['day'], q['mode']) == (actor, day, mode), 'query/label join differs')
            records += 1
            previous.add_first(atoms, q, r)
            if not q['denominator_eligible']:
                continue
            i = lookup.pop(tuple(q['query_key']))
            require(q['point_s'] == requests[i, 3] and q['query_time_s'] == requests[i, 4] and
                episodes[episode_ids[i]].tolist() == [day, *q['episode']] and q['native_supported'] == bool(native[i]),
                'query coordinates/native mask differ')
            bits = previous.saved.fit.scope_bits(q)
            require(bits == int(requests[i, 2]) and q['observable']['hour'] == int(q['query_time_s']//3600), 'scope clock differs')
            moving[i] = bool(bits & 16) and q['native_stage']['kind'] == 'moving-launch'
            period = temporal.STRATA[int(temporal.strata(np.array([q['query_time_s']]))[0])]
            key = (*q['candidate_key'], period) if q['native_supported'] else None
            ki = key_index.get(key)
            base = bool(native[i] and ki is not None and table['candidate'][ki]['status'] == 'finite')
            require(base == bool(baseline[i]), 'V172 support mask not reproduced')
            require(q['clocks']['query_load'] is None and not q['clocks']['query_load_available'], 'late load backfill')
            position = q['observable']['position']
            selected.append((q['query_time_s'], q['clocks']['query_ready_order'], i, ki,
                None if position is None else position['remaining_stops']))
            checked += 1
        require(not lookup and records == receipt['query_record_count'], 'query cohort incomplete')
        prefix = previous.Prefix(atoms)
        for now, order, i, ki, remaining in sorted(selected):
            history = prefix.query(now, remaining)['global']
            if int(requests[i, 2]) & 4 or moving[i]:
                signs[i] = service_sign(history)
                if baseline[i] and signs[i] >= 0:
                    codes[i] = 2*ki+int(signs[i])
        print('V181_SUPPORT_DAY_READ', actor, mode, count, '/64', flush=True)
    bits = requests[:, 2].astype(int)
    target = target_mask(bits, moving)
    finite, counts = support_mask(baseline, target, codes, days)
    require(not np.any(finite & ~baseline) and np.array_equal(finite[~target], baseline[~target]), 'support expanded/non-target changed')
    masks = {s: ((bits & (1 << i)) != 0) for i, s in enumerate(temporal.SCOPES[:-1])}
    masks['hour13_moving_launch'] = moving
    cells = [dict(actor=actor, mode=mode, **temporal.support_cell(d, s, mask, days, episode_ids, native, finite, baseline))
        for d, (s, mask) in product(CALIBRATION, masks.items())]
    old = {(c['day'], c['scope']): c for c in table['cells']}
    for cell in cells:
        reference = old[cell['day'], cell['scope']]
        for field in ('eligible_queries', 'eligible_episodes', 'native_abstentions'):
            require(cell[field] == reference[field], 'original support denominator differs')
        close(cell['baseline_fraction'], reference['finite_fraction'])
    conditions = [dict(actor=actor, mode=mode, condition=label,
        **temporal.support_cell(d, s, masks[s] & (signs == sign), days, episode_ids, native, finite, baseline))
        for d, s, (sign, label) in product(CALIBRATION, previous.prior.SCOPES,
            [(-1, 'unavailable'), (0, SIGNS[0]), (1, SIGNS[1])])]
    keys = [dict(key=[*table['candidate'][int(code)//2]['key'], SIGNS[int(code) % 2]],
        days=int(n), support_eligible=bool(n >= MIN_DAYS)) for code, n in sorted(counts.items())]
    reason = np.full(len(requests), 'support_eligible', dtype=object)
    reason[~native] = 'native_abstention'
    reason[native & ~baseline] = 'v172_abstention'
    reason[baseline & target & (signs < 0)] = 'prefix_abstention'
    reason[baseline & target & (signs >= 0) & ~finite] = 'branch_less_than_19_days'
    require(np.array_equal(reason == 'support_eligible', finite), 'support reason partition differs')
    write(args.out_root/'details/key_support'/f'{actor}_{mode}.json', keys)
    result = dict(actor=actor, mode=mode, source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], cells=cells,
        conditions=conditions, checked_queries=checked, non_target_support_unchanged=True,
        reasons=dict(Counter(reason)), target_reasons=dict(Counter(reason[target])),
        signs=[dict(sign=label, keys=sum(k['key'][-1] == label for k in keys),
            finite_keys=sum(k['key'][-1] == label and k['support_eligible'] for k in keys),
            day_histogram=dict(sorted(Counter(k['days'] for k in keys if k['key'][-1] == label).items()))) for label in SIGNS])
    write(args.out_root/'cells'/f'{actor}_{mode}.json', result)
    return result


def run(args):
    model = parents(args)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
        '-p', 'test_v181_*.py', '-v'], check=True)
    plan = registration(model)
    require(not (args.out_root/'protocol.json').exists(), 'use a new output root; never replace preregistration')
    write(args.out_root/'protocol.json', plan)
    print('V181_PROTOCOL_SEALED_BEFORE_SUPPORT', flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        for job in as_completed([pool.submit(bucket, args, model, c) for c in model['chunks']]):
            rows.append(job.result())
    cells = [c for r in rows for c in r['cells']]
    release = temporal.support_release(cells, model['actors'], list(CALIBRATION))
    conditions = []
    for mode, scope, label in product(('old', 'handoff'), previous.prior.SCOPES, ('unavailable', *SIGNS)):
        selected = [c for r in rows for c in r['conditions'] if (c['mode'], c['scope'], c['condition']) == (mode, scope, label)]
        nonempty = [c for c in selected if c['eligible_episodes']]
        conditions.append(dict(mode=mode, scope=scope, condition=label, actor_day_cells=len(selected),
            nonempty_cells=len(nonempty), empty_cells=len(selected)-len(nonempty),
            eligible_queries=sum(c['eligible_queries'] for c in selected), finite_queries=sum(c['finite_queries'] for c in selected),
            conditional_nonempty_mean_fraction=None if not nonempty else float(np.mean([c['finite_fraction'] for c in nonempty]))))
    require(read(args.v172_root/'model.json') == model and read(args.out_root/'protocol.json') == plan, 'frozen header/protocol changed')
    write(args.out_root/'summary/status.json', dict(status='v181_lower_tail_preregistered_support_complete',
        source=plan['source'], protocol=plan, calibration_paths=1536, support=release, conditions=conditions,
        signs=[dict(actor=r['actor'], mode=r['mode'], signs=r['signs']) for r in rows],
        reasons=dict(sum((Counter(r['reasons']) for r in rows), Counter())),
        target_reasons=dict(sum((Counter(r['target_reasons']) for r in rows), Counter())),
        checked_queries=sum(r['checked_queries'] for r in rows), original_v172_support_reproduced=True,
        non_target_support_unchanged=True, interval_fit=False, new_rollouts=0, new_training=0,
        confirmation_outcomes_read=False, confirmation_started=False, collector_submitted=False, main_policy_changed=False,
        next='Support only; no fitting or confirmation authorization. Stop without regrouping if support is insufficient.'))
    print('V181_PREREGISTERED_SUPPORT_COMPLETE_NO_FIT', release['decision'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v166-root', 'v166-audit-root', 'v176-root', 'v172-root', 'v167-root', 'v180-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    run(p.parse_args())
