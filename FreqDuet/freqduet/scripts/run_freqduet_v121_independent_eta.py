#!/usr/bin/env python3
"""Confirm the frozen V120 predictor on new days, without changing any action."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

import audit_freqduet_v120_component_eta as model


prior, frozen = model.prior, model.prior.frozen
SEEDS = tuple(frozen.SEEDS)
DAYS = tuple(range(2211001, 2211009))
ITEMS = list(itertools.product(SEEDS, DAYS))
PREFLIGHT = [ITEMS.index((seed, DAYS[0])) for seed in SEEDS]
MODEL_SOURCE = 'a3822ea155'


def name_for(seed, day):
    return f'{seed}_{day}_{frozen.PRIMARY}_0'


def observe(root, index):
    seed, day = ITEMS[index]
    rule, credit = prior.IdentityAudit(seed), frozen.pilot.CreditObserver()
    result, _, _, _ = frozen.pilot.rollout(root, seed, day, 'stochastic', 187000000+seed+100*day,
        [0.]*4, f'v115_{seed}_{day}_{frozen.PRIMARY}_0', probability_rule=rule,
        trace_observer=rule, credit_observer=credit)
    result.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    actual = dict(seed=seed, day=day, method=frozen.PRIMARY, rep=0, results=result,
        contract=rule.contract, coverage=rule.coverage, feedforward=rule.feedforward, accounting=rule.accounting)
    name = name_for(seed, day)
    rule.save(root/'records'/(name+'.npz'))
    prior.write_trace(root/'traces'/(name+'.jsonl.gz'),
        [dict(row, evidence=rule.evidence[prior.audit.key(row)]) for row in rule.decisions])
    prior.write_trace(root/'ready'/(name+'.jsonl.gz'), [dict(trip_id=k[0], station_id=k[1], direction=k[2], **v)
        for k, v in rule.ready.items()])
    next_arr = prior.successors(rule.arrivals)
    next_dep = prior.successors([r for r in rule.departures if prior.audit.key(r) in rule.ready])
    support = dict(forecasts=0, same_ready_resolved=0, next_arrival_matches=0, next_departure_matches=0)
    for row in rule.decisions:
        if not row['follower_valid']:
            continue
        key, e = prior.audit.key(row), rule.evidence[prior.audit.key(row)]
        target = e['predicted_follower_trip_id'], key[1], key[2]
        support['forecasts'] += 1
        support['same_ready_resolved'] += int(target in rule.ready and
            rule.ready[target]['bus_id'] == e['predicted_follower_bus_id'] and
            rule.ready[target]['ready_time_s'] >= e['ready_time_s'])
        support['next_arrival_matches'] += int(next_arr.get(key) == target)
        support['next_departure_matches'] += int(next_dep.get(key) == target)
    receipt = dict(seed=seed, day=day, index=index, frozen_result=actual,
        decisions=len(rule.decisions), ready_events=len(rule.ready), label_support=support,
        model_used_for_actions=False, actor_updates=0, critic_updates=0, dual_updates=0)
    frozen.pilot.write_json(root/'pairs'/f'p{index}.json', receipt)
    print('V121_FRESH_PASSIVE_PATH_DONE', index, seed, day, flush=True)
    return receipt


def preflight(root, index):
    proof = root/'preflight'/f'p{index}.json'
    if proof.exists():
        return
    seed, day = ITEMS[index]
    name = name_for(seed, day)
    # The rollout root must stay next to the existing runtime-equivalence proofs.
    plain = frozen.replay(root, seed, day, frozen.PRIMARY, 0)
    old = root/'preflight_control'
    for folder, suffix in (('records', '.npz'), ('traces', '.jsonl.gz')):
        target = old/folder/(name+suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/folder/(name+suffix), target)
    observed = observe(root, index)
    frozen.pilot.joint.assert_same_state(plain, observed['frozen_result'], 'V121_passive_all_endpoints_and_accounts')
    trace = prior.read_trace(root/'traces'/(name+'.jsonl.gz'))
    prior.assert_decisions(prior.read_trace(old/'traces'/(name+'.jsonl.gz')),
        [{k: v for k, v in row.items() if k != 'evidence'} for row in trace])
    with np.load(old/'records'/(name+'.npz')) as before, np.load(root/'records'/(name+'.npz')) as after:
        if before.files != after.files:
            raise AssertionError('passive telemetry changed timing record fields')
        for field in before.files:
            np.testing.assert_array_equal(before[field], after[field])
    frozen.pilot.write_json(proof, dict(seed=seed, day=day, index=index,
        all_endpoints_exact=True, original_decisions_exact=True, full_timing_arrays_exact=True))
    print('V121_FRESH_ACTOR_PREFLIGHT_EXACT', seed, day, flush=True)


def run_item(root, index):
    if not (root/'pairs'/f'p{index}.json').exists():
        observe(root, index)


def assert_grid(receipts):
    if [(r['seed'], r['day'], r['index']) for r in receipts] != [(*item, i) for i, item in enumerate(ITEMS)]:
        raise AssertionError('independent confirmation requires every registered actor/day once')
    if any(r['model_used_for_actions'] or any(r[k] for k in ('actor_updates', 'critic_updates', 'dual_updates'))
            for r in receipts):
        raise AssertionError('independent predictor confirmation changed a controller')


def regression(root, v118, v120, runtime):
    saved = json.loads((v120/'summary/status.json').read_text())
    if not saved['eligible_for_independent_predictor_confirmation']:
        raise AssertionError('V120 did not qualify for independent confirmation')
    actual = model.summarize_paths(v118, prior.ITEMS, model.load_geometry(runtime))
    for key, value in actual.items():
        if saved[key] != value:
            raise AssertionError(f'dataset-interface extraction changed frozen V120 {key}')
    frozen.pilot.write_json(root/'smoke/v120_reproduced.json', dict(v120_core_summary_exact=True,
        model_source=MODEL_SOURCE, window_s=model.WINDOW_S, new_demand_labels_read=0))
    print('V121_FROZEN_V120_SUMMARY_EXACT', flush=True)


def aggregate(root, runtime):
    receipts = [json.loads((root/'pairs'/f'p{i}.json').read_text()) for i in range(len(ITEMS))]
    assert_grid(receipts)
    proofs = [json.loads((root/'preflight'/f'p{i}.json').read_text()) for i in PREFLIGHT]
    if not all(all(p[k] for k in ('all_endpoints_exact', 'original_decisions_exact', 'full_timing_arrays_exact'))
            for p in proofs):
        raise AssertionError('both archived actors must pass exact passive preflight')
    summary = model.summarize_paths(root, ITEMS, model.load_geometry(runtime))
    qualified = summary.pop('eligible_for_independent_predictor_confirmation')
    support = {k: sum(r['label_support'][k] for r in receipts) for k in receipts[0]['label_support']}
    if support['same_ready_resolved'] != support['forecasts']:
        raise AssertionError('same-object labels are incomplete; do not score missing labels as zero')
    if summary['cohorts']['forecasts']['decisions'] != support['forecasts']:
        raise AssertionError('independent forecast denominator changed during analysis')
    frozen.pilot.write_json(root/'summary/status.json', dict(status='v121_independent_passive_eta_complete',
        **summary, independent_predictor_criterion_pass=qualified, model_source=MODEL_SOURCE,
        window_s=model.WINDOW_S, seeds=SEEDS, demand_days=DAYS, label_support=support,
        passive_preflight=proofs, completed_paths=len(receipts), extra_plain_preflight_paths=len(PREFLIGHT),
        new_demand_days=len(DAYS), altered_action_paths=0, actor_updates=0, critic_updates=0, dual_updates=0,
        quota_changed=False, main_policy_changed=False,
        limitations='Independent demand days, same two archived actors/network/demand family. '
            'The V120 rolling completed-prefix rule is unchanged. Pending/boundary cases retain raw ETA. '
            'Same-object fixed-time pair signs are not changed-control CV2 or passenger returns; '
            'actual-next identity agreement is reported separately. No allocation or controller was tested.'))
    print('V121_INDEPENDENT_PASSIVE_COMPLETE criterion=', qualified, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--v118-root', type=Path, required=True)
    p.add_argument('--v120-root', type=Path, required=True)
    p.add_argument('--runtime-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    tests = Path(__file__).resolve().parents[1]/'tests'
    for version in ('v118', 'v120', 'v121'):
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(tests),
            '-p', f'test_{version}_*.py', '-v'], check=True)
    regression(a.out_root, a.v118_root, a.v120_root, a.runtime_root)
    with ProcessPoolExecutor(max_workers=min(2, a.workers), mp_context=get_context('spawn')) as pool:
        list(pool.map(preflight, itertools.repeat(a.out_root), PREFLIGHT))
    print('V121_BOTH_PASSIVE_PREFLIGHTS_EXACT', flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context('spawn')) as pool:
        list(pool.map(run_item, itertools.repeat(a.out_root), [i for i in range(len(ITEMS)) if i not in PREFLIGHT]))
    aggregate(a.out_root, a.runtime_root)
