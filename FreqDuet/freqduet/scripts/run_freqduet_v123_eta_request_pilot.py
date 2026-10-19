#!/usr/bin/env python3
"""Small matched pilot: frozen raw authorization, streaming ETA on requests only."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

import run_freqduet_v122_online_eta as streamed
import run_freqduet_v94_compensation_confirmation as inference
import run_freqduet_v96_frozen_compensation as axes


frozen, prior, model = streamed.frozen, streamed.prior, streamed.model
pilot, confirmation = frozen.pilot, frozen.confirmation
ACTIONS = frozen.ACTIONS
SEEDS = frozen.SEEDS
DAYS = tuple(range(2231001, 2231005))
PRIMARY, RAW, BILATERAL = 'split_request', frozen.PRIMARY, 'bilateral025'
METHODS = (RAW, PRIMARY, BILATERAL)
ITEMS = list(itertools.product(SEEDS, DAYS, METHODS))
METRICS = frozen.METRICS
CONTRASTS = {PRIMARY+'_vs_'+b: (PRIMARY, b) for b in (RAW, BILATERAL)}
CONTRASTS[RAW+'_vs_'+BILATERAL] = (RAW, BILATERAL)


def request_error(context, prediction):
    if not context.evidence_valid:
        return 0.
    if not context.follower_evidence_valid:
        return context.target_headway_s-context.forward_headway_s
    gap = prediction['gap_s'] if prediction['component_used'] else context.follower_departure_gap_s
    return (gap-context.forward_headway_s)/2


class SplitRequestAudit(streamed.OnlineAudit):
    def __call__(self, policy, states, features, theta):
        # Observe the original raw bilateral shadow once; do not issue two credits.
        baseline = frozen.prior.prior.DisturbanceAudit.__call__(self, policy, states, features, theta)
        context = self.runner._pending_lower_action_context[int(self.runner._pilot_agent)]
        bus = self.runner._bus_for_agent(self.runner._pilot_agent)
        key = int(bus.trip_id), int(bus.last_station.station_id), bool(bus.direction)
        prediction = self.online[key]
        predecessor = int(bus.forward_predecessor_trip_id) if context.evidence_valid else None
        residual = self.observable_coordinates(bus, context)[2] if context.evidence_valid else None
        error = request_error(context, prediction)
        mask = policy.feasible_action_mask(states)[0].numpy()
        held = float(bus.episode_hold_vehicle_seconds)-self.launches[key[0]]['holding_counter_at_launch']
        selected, row = self.budget.request(key, predecessor, residual, error, mask,
            ACTIONS[int(baseline[0].argmax())], held)
        request_shadow = ACTIONS[frozen.prior.prior.quantized_action(.25*error, mask)]
        raw_error = request_error(context, dict(component_used=False))
        row.update(policy_time_s=float(self.env.current_time), bus_id=int(bus.bus_id),
            onboard_passengers=len(bus.passengers), follower_valid=bool(context.follower_evidence_valid),
            raw_feedback_error_s=raw_error, request_shadow_s=request_shadow,
            component_used=bool(context.evidence_valid and context.follower_evidence_valid
                and prediction['component_used']), forecast_status=prediction['status'],
            prediction_time_s=prediction['prediction_time_s'], request_gap_s=prediction['gap_s'])
        self.decisions.append(row)
        self.evidence[key] = dict(self.ready[key], forward_gap_s=context.forward_headway_s,
            target_headway_s=context.target_headway_s, follower_gap_prediction_s=context.follower_departure_gap_s)
        law = torch.zeros_like(baseline)
        law[0, selected] = 1.
        return law

    def finish(self, runner, records, holds):
        frozen.prior.prior.DisturbanceAudit.finish(self, runner, records, holds)
        if self.budget.pending or len(self.decisions) != len(records):
            raise AssertionError('split requests lack completed holding receipts')
        for row, sampled in zip(self.decisions, records):
            if prior.audit.key(row) != prior.audit.key(sampled):
                raise AssertionError('split request and sampled stop differ')
            if row['chosen_s'] != ACTIONS[sampled['action']] or row['executed_s'] != row['chosen_s']:
                raise AssertionError('split request differs from actual execution')
            if row['chosen_s'] > min(max(row['feedback_error_s'], 0.), row['available_s']):
                raise AssertionError('split request overspends or overshoots its predicted balance')
            if row['increment_s'] <= 0 and row['chosen_s'] > row['request_shadow_s']:
                raise AssertionError('old credit overrode a nonpositive innovation')
            if row['prediction_time_s'] != self.evidence[prior.audit.key(row)]['ready_time_s']:
                raise AssertionError('split request used a forecast from another clock')
        audit = self.budget.audit()
        if audit['conservation_error'] != 0 or audit['minimum_prefix_margin_s'] < 0:
            raise AssertionError('split requests violate original non-borrowing authorization')
        rows = self.decisions
        self.accounting = dict(**audit, decisions=len(rows), settled=len(rows),
            above_raw_shadow=sum(r['chosen_s'] > r['shadow_s'] for r in rows),
            below_raw_shadow=sum(r['chosen_s'] < r['shadow_s'] for r in rows),
            changed_request_shadow=sum(r['shadow_s'] != r['request_shadow_s'] for r in rows),
            negative_innovation=sum(r['increment_s'] < 0 for r in rows),
            balanced=sum(r['feedback_error_s'] <= 0 for r in rows),
            balanced_nonzero=sum(r['feedback_error_s'] <= 0 and r['chosen_s'] > 0 for r in rows),
            cap_actions=sum(r['chosen_s'] == ACTIONS[-1] for r in rows),
            component_requests=sum(r['component_used'] for r in rows),
            feedback_delta_sum_s=sum(r['feedback_error_s']-r['raw_feedback_error_s'] for r in rows))


def replay(root, runtime, seed, day, method):
    if method == BILATERAL:
        return frozen.prior.replay(root, seed, day, method, 0)
    cls = SplitRequestAudit if method == PRIMARY else streamed.OnlineAudit
    rule, credit = cls(seed, model.load_geometry(runtime)), pilot.CreditObserver()
    result, _, _, _ = pilot.rollout(root, seed, day, 'stochastic', 187000000+seed+100*day,
        [0.]*4, f'v123_{seed}_{day}_{method}_0', probability_rule=rule,
        trace_observer=rule, credit_observer=credit)
    result.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    actual = dict(seed=seed, day=day, method=method, rep=0, results=result,
        contract=rule.contract, coverage=rule.coverage, feedforward=rule.feedforward, accounting=rule.accounting)
    name = f'{seed}_{day}_{method}_0'
    rule.save(root/'records'/(name+'.npz'))
    prior.write_trace(root/'traces'/(name+'.jsonl.gz'),
        [dict(r, evidence=rule.evidence[prior.audit.key(r)]) for r in rule.decisions])
    actual['online_sources'] = dict(Counter(p['status'] for p in rule.online.values()))
    return actual


def check_old_control(root, old, actual):
    case = actual['seed'], actual['day'], actual['method'], 0
    cached = json.loads((old/'pairs'/f'p{frozen.ITEMS.index(case)}.json').read_text())
    pilot.joint.assert_same_state(cached, {k: actual[k] for k in cached}, 'V123_exact_consumed_control')
    name = '_'.join(map(str, case))
    with np.load(root/'records'/(name+'.npz')) as new, np.load(old/'records'/(name+'.npz')) as before:
        if new.files != before.files:
            raise AssertionError('V123 control changed timing fields')
        for key in before.files:
            np.testing.assert_array_equal(new[key], before[key])
    if actual['method'] == RAW:
        current = prior.read_trace(root/'traces'/(name+'.jsonl.gz'))
        prior.assert_decisions(prior.read_trace(old/'traces'/(name+'.jsonl.gz')),
            [{k: v for k, v in row.items() if k != 'evidence'} for row in current])


def execute(root, runtime, cases, workers):
    started = time.monotonic()
    print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=0, total=len(cases), unit='rollout')), flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        futures = {pool.submit(replay, root, runtime, *case): case for case in cases}
        for future in as_completed(futures):
            case, row = futures[future], future.result()
            path = root/'pairs'/('_'.join(map(str, case))+'.json')
            pilot.write_json(path, row)
            results.append(row)
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(results), total=len(cases),
                unit='rollout', elapsed_seconds=time.monotonic()-started)), flush=True)
    return results


def smoke(root, old, v122, runtime, workers):
    parent = json.loads((v122/'summary/status.json').read_text())
    if parent['completed_paths'] != 16 or parent['max_gap_error_s'] > 1e-8:
        raise AssertionError('V122 online equivalence must pass first')
    for version in ('v115', 'v122', 'v123'):
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1]/'tests'), '-p', f'test_{version}_*.py', '-v'], check=True)
    cases = list(itertools.product(SEEDS, (frozen.DAYS[0],), METHODS))
    rows = execute(root, runtime, cases, workers)
    for row in rows:
        if row['method'] in (RAW, BILATERAL):
            check_old_control(root, old, row)
    candidates = [r['accounting'] for r in rows if r['method'] == PRIMARY]
    if not all(r['component_requests'] and r['changed_request_shadow'] for r in candidates):
        raise AssertionError('both actors must exercise the frozen split-request mechanism')
    pilot.write_json(root/'smoke/done.json', dict(status='exact_controls_streaming_requests_and_prefix_accounts_passed',
        consumed_rollouts=len(cases), fresh_demand_days=0, formal_rollouts=len(ITEMS),
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False))
    print('V123_SMOKE_COMPLETE', flush=True)


def run(root, runtime, start, end, workers):
    if not (root/'smoke/done.json').exists():
        raise AssertionError('fresh days require consumed-path smoke')
    cases = [case for case in ITEMS[start:end]
        if not (root/'pairs'/('_'.join(map(str, case))+'.json')).exists()]
    execute(root, runtime, cases, workers)
    pilot.write_json(root/'shards'/f'{start}-{end}'/'done.json', dict(status='complete'))


def aggregate(root):
    rows = [json.loads((root/'pairs'/('_'.join(map(str, case))+'.json')).read_text()) for case in ITEMS]
    if [(r['seed'], r['day'], r['method']) for r in rows] != ITEMS:
        raise AssertionError('V123 requires all 24 registered paths')
    paired, absolute = [], []
    for seed, day in itertools.product(SEEDS, DAYS):
        cell = [r for r in rows if (r['seed'], r['day']) == (seed, day)]
        for row in cell[1:]:
            pilot.joint.assert_same_state(row['contract'], cell[0]['contract'], 'V123_shared_physical_contract')
        means = {r['method']: {k: r['results'][k] for k in METRICS} for r in cell}
        absolute.append(dict(seed=seed, day=day, values=means))
        paired.append(dict(seed=seed, day=day, effects={name: {k: means[a][k]-means[b][k] for k in METRICS}
            for name, (a, b) in CONTRASTS.items()}))
    stats = inference.describe(paired, 'effects')
    gates = {b: confirmation.joint_gate(stats[PRIMARY+'_vs_'+b]) for b in (RAW, BILATERAL)}
    accounts = {}
    for method in (RAW, PRIMARY):
        values = [r['accounting'] for r in rows if r['method'] == method]
        accounts[method] = {k: sum(r[k] for r in values) for k in values[0]}
        accounts[method]['minimum_prefix_margin_s'] = min(r['minimum_prefix_margin_s'] for r in values)
        accounts[method]['conservation_error'] = max(r['conservation_error'] for r in values)
    pilot.write_json(root/'summary/status.json', dict(status='v123_matched_request_pilot_complete',
        effects=stats, absolute=inference.describe(absolute, 'values'), actor_day=paired,
        axes=axes.axis_summary(paired, 'effects', confirmation.KEY_METRICS), accounting=accounts,
        diagnostic_gates=gates, joint_both_controls=all(gates.values()),
        rollouts=len(rows), archived_actors=list(SEEDS), new_demand_days=list(DAYS),
        action_bins_s=list(ACTIONS), feedback_gain=.25, innovation_gain=.75,
        predictor_window_s=3600, raw_shadow_authorization=True, gains_refitted=False,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        limitations='Small frozen-controller pilot on two archived actors and four fresh demand days. '
            'Requests and balance-triggered expiry use the corrected gap, issuance remains raw bilateral025 '
            'on each visited state; realized issuance may differ across closed-loop paths. '
            'Moving-only completed-prefix ETA retains all original fallbacks. '
            'Strict crossed actor/day diagnostic gates, not automatic policy promotion, new-actor '
            'confirmation, frequency-separation evidence or an external fixed-upper comparison.'))
    print('V123_MATCHED_REQUEST_PILOT_COMPLETE', gates, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'run', 'aggregate'))
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--v115-root', type=Path)
    p.add_argument('--v122-root', type=Path)
    p.add_argument('--runtime-root', type=Path)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=len(ITEMS))
    a = p.parse_args()
    if a.command == 'smoke':
        smoke(a.out_root, a.v115_root, a.v122_root, a.runtime_root, a.workers)
    elif a.command == 'run':
        run(a.out_root, a.runtime_root, a.start, a.end, a.workers)
    else:
        aggregate(a.out_root)
