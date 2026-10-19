#!/usr/bin/env python3
"""Close V123's consumed forecast x holding-rule factorial, without training."""

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

import run_freqduet_v123_eta_request_pilot as prior


pilot, streamed = prior.pilot, prior.streamed
PRIMARY = 'split_bilateral025'
METHODS = (*prior.METHODS, PRIMARY)
ITEMS = list(itertools.product(prior.SEEDS, prior.DAYS))
RUN_ITEMS = [item for item in ITEMS if item[1] != prior.DAYS[0]]
CONTRASTS = dict(prior.CONTRASTS, **{PRIMARY+'_vs_'+b: (PRIMARY, b) for b in prior.METHODS})
INTERACTION = 'forecast_x_funded_package'
disturbance = prior.frozen.prior.prior


def bilateral_action(context, prediction, feasible):
    return disturbance.quantized_action(.25*prior.request_error(context, prediction), feasible)


def factorial_effects(levels):
    effects = {name: {k: levels[a][k]-levels[b][k] for k in prior.METRICS}
        for name, (a, b) in CONTRASTS.items()}
    effects[INTERACTION] = {k: (levels[prior.PRIMARY][k]-levels[prior.RAW][k])
        -(levels[PRIMARY][k]-levels[prior.BILATERAL][k]) for k in prior.METRICS}
    return effects


class BilateralAudit(streamed.OnlineAudit):
    def __init__(self, seed, geometry, arm):
        super().__init__(seed, geometry)
        self.arm = arm

    def __call__(self, policy, states, features, theta):
        raw = disturbance.DisturbanceAudit.__call__(self, policy, states, features, theta)
        context = self.runner._pending_lower_action_context[int(self.runner._pilot_agent)]
        bus = self.runner._bus_for_agent(self.runner._pilot_agent)
        key = int(bus.trip_id), int(bus.last_station.station_id), bool(bus.direction)
        prediction = self.online[key]
        old = int(raw[0].argmax())
        selected = bilateral_action(context, prediction, policy.feasible_action_mask(states)[0].numpy()) if self.arm == PRIMARY else old
        self.evidence[key] = dict(self.ready[key], forward_gap_s=context.forward_headway_s,
            target_headway_s=context.target_headway_s, follower_gap_prediction_s=context.follower_departure_gap_s)
        self.decisions.append(dict(trip_id=key[0], station_id=key[1], direction=key[2],
            policy_time_s=float(self.env.current_time), chosen_s=prior.ACTIONS[selected],
            raw_shadow_s=prior.ACTIONS[old], feedback_error_s=prior.request_error(context, prediction),
            raw_feedback_error_s=prior.request_error(context, dict(component_used=False)),
            executed_before_s=float(bus.episode_hold_vehicle_seconds)-self.launches[key[0]]['holding_counter_at_launch'],
            component_used=bool(self.arm == PRIMARY and context.evidence_valid
                and context.follower_evidence_valid and prediction['component_used']),
            request_gap_s=prediction['gap_s'], prediction_time_s=prediction['prediction_time_s']))
        law = torch.zeros_like(raw)
        law[0, selected] = 1.
        return law

    def finish(self, runner, records, holds):
        disturbance.DisturbanceAudit.finish(self, runner, records, holds)
        if self.budget.trips or self.budget.pending:
            raise AssertionError('the bilateral arm entered funded accounting')
        if len(self.decisions) != len(records):
            raise AssertionError('bilateral action receipts are incomplete')
        ends = {prior.prior.audit.key(r): r for r in self.departures}
        for row, sampled in zip(self.decisions, records):
            key = prior.prior.audit.key(row)
            if key != prior.prior.audit.key(sampled) or row['chosen_s'] != prior.ACTIONS[sampled['action']]:
                raise AssertionError('bilateral command differs from sampled action')
            executed = ends[key]['realized_holding']-row['executed_before_s']
            if executed != row['chosen_s'] or row['prediction_time_s'] != self.ready[key]['ready_time_s']:
                raise AssertionError('bilateral execution or forecast clock differs from its receipt')
            row['executed_s'] = executed
        self.accounting = {}
        self.prediction_use = dict(decisions=len(records), quota_active=False,
            component_requests=sum(r['component_used'] for r in self.decisions),
            changed_commands=sum(r['chosen_s'] != r['raw_shadow_s'] for r in self.decisions),
            command_delta_sum_s=sum(r['chosen_s']-r['raw_shadow_s'] for r in self.decisions),
            online_sources=dict(Counter(p['status'] for p in self.online.values())))


def replay(root, runtime, seed, day, arm):
    rule, credit = BilateralAudit(seed, prior.model.load_geometry(runtime), arm), pilot.CreditObserver()
    result, _, _, _ = pilot.rollout(root, seed, day, 'stochastic', 187000000+seed+100*day,
        [0.]*4, f'v124_{seed}_{day}_{arm}', probability_rule=rule, trace_observer=rule, credit_observer=credit)
    result.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    name = f'{seed}_{day}_{arm}_0'
    rule.save(root/'records'/(name+'.npz'))
    prior.prior.write_trace(root/'traces'/(name+'.jsonl.gz'),
        [dict(r, evidence=rule.evidence[prior.prior.audit.key(r)]) for r in rule.decisions])
    return dict(seed=seed, day=day, method=arm, rep=0, results=result, contract=rule.contract,
        coverage=rule.coverage, feedforward=rule.feedforward, accounting=rule.accounting,
        prediction_use=rule.prediction_use)


def execute(root, runtime, cases, workers):
    started = time.monotonic()
    print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=0, total=len(cases), unit='rollout')), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        futures = [pool.submit(replay, root, runtime, *case) for case in cases]
        for future in as_completed(futures):
            row = future.result()
            pilot.write_json(root/'pairs'/f"{row['seed']}_{row['day']}_{row['method']}.json", row)
            rows.append(row)
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=len(rows), total=len(cases),
                unit='rollout', elapsed_seconds=time.monotonic()-started)), flush=True)
    return rows


def smoke(root, cached, runtime, workers):
    for pattern in ('test_v122_*.py', 'test_v123_eta_request.py', 'test_v124_*.py'):
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1]/'tests'), '-p', pattern, '-v'], check=True)
    day = prior.DAYS[0]
    raw = execute(root, runtime, [(s, day, prior.BILATERAL) for s in prior.SEEDS], min(workers, 2))
    for row in raw:
        name = f"{row['seed']}_{day}_{prior.BILATERAL}"
        old = json.loads((cached/'pairs'/(name+'.json')).read_text())
        pilot.joint.assert_same_state(old, {k: row[k] for k in old}, 'V124_exact_original_bilateral')
        with np.load(root/'records'/(name+'_0.npz')) as new, np.load(cached/'records'/(name+'_0.npz')) as before:
            if new.files != before.files:
                raise AssertionError('passive bilateral changed timing fields')
            for field in before.files:
                np.testing.assert_array_equal(new[field], before[field])
    print('V124_BOTH_RAW_CONTROLS_EXACT', flush=True)
    split = execute(root, runtime, [(s, day, PRIMARY) for s in prior.SEEDS], min(workers, 2))
    if not all(r['prediction_use']['component_requests'] and r['prediction_use']['changed_commands'] for r in split):
        raise AssertionError('both actors must exercise the missing factorial arm')
    pilot.write_json(root/'smoke/done.json', dict(status='tests_exact_bilateral_and_split_execution_passed',
        raw_control_replays=2, split_cells_completed=2, remaining_cells=len(RUN_ITEMS),
        new_demand_days=0, actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False))
    print('V124_SMOKE_COMPLETE', flush=True)


def run(root, runtime, start, end, workers):
    if not (root/'smoke/done.json').exists():
        raise AssertionError('remaining cells require both exact controls and exercised split arms')
    cases = [(*item, PRIMARY) for item in RUN_ITEMS[start:end]
        if not (root/'pairs'/f'{item[0]}_{item[1]}_{PRIMARY}.json').exists()]
    execute(root, runtime, cases, workers)
    pilot.write_json(root/'shards'/f'{start}-{end}'/'done.json', dict(status='complete'))


def aggregate(root, cached):
    parent = json.loads((cached/'summary/status.json').read_text())
    paired, absolute, use = [], [], []
    for seed, day in ITEMS:
        rows = {m: json.loads(((root if m == PRIMARY else cached)/'pairs'/f'{seed}_{day}_{m}.json').read_text()) for m in METHODS}
        if any((r['seed'], r['day'], r['method']) != (seed, day, m) for m, r in rows.items()):
            raise AssertionError('factorial path differs from the consumed V123 grid')
        for row in rows.values():
            pilot.joint.assert_same_state(row['contract'], rows[PRIMARY]['contract'], 'V124_shared_physical_contract')
        levels = {m: {k: r['results'][k] for k in prior.METRICS} for m, r in rows.items()}
        effects = factorial_effects(levels)
        old = next(r for r in parent['actor_day'] if (r['seed'], r['day']) == (seed, day))
        pilot.joint.assert_same_state(old['effects'], {k: effects[k] for k in prior.CONTRASTS}, 'V124_retained_V123_effects')
        paired.append(dict(seed=seed, day=day, effects=effects))
        absolute.append(dict(seed=seed, day=day, values=levels))
        use.append(dict(seed=seed, day=day, **rows[PRIMARY]['prediction_use']))
    stats = prior.inference.describe(paired, 'effects')
    gates = {b: prior.confirmation.joint_gate(stats[PRIMARY+'_vs_'+b]) for b in prior.METHODS}
    pilot.write_json(root/'summary/status.json', dict(status='v124_consumed_eta_factorial_complete',
        effects=stats, absolute=prior.inference.describe(absolute, 'values'), actor_day=paired,
        axes=prior.axes.axis_summary(paired, 'effects', prior.confirmation.KEY_METRICS), prediction_use=use,
        diagnostic_gates=gates, joint_all_controls=all(gates.values()), reused_v123_paths=24,
        missing_arm_rollouts=8, raw_control_replays=2, new_demand_days=0,
        feedback_gain=.25, gains_refitted=False, actor_updates=0, critic_updates=0,
        dual_updates=0, main_policy_changed=False, controller_admitted=False,
        limitations='Consumed two-actor/four-day development factorial, not independent confirmation. '
            'Funding is a package of innovation, credit and expiry, not separate causal effects of each. '
            'Streaming ETA, fallback, bins, masks and learned upper remain frozen. '
            'Bilateral arms have no quota and are not subject to funded authorization. '
            'Crossed actor/day diagnostic gates do not promote a controller or establish frequency/fixed-upper superiority.'))
    print('V124_ETA_FACTORIAL_COMPLETE', gates, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('all', 'smoke', 'run', 'aggregate'))
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--v123-root', type=Path)
    p.add_argument('--runtime-root', type=Path)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=len(RUN_ITEMS))
    a = p.parse_args()
    if a.command == 'all':
        smoke(a.out_root, a.v123_root, a.runtime_root, a.workers)
        run(a.out_root, a.runtime_root, 0, len(RUN_ITEMS), a.workers)
        aggregate(a.out_root, a.v123_root)
    elif a.command == 'smoke':
        smoke(a.out_root, a.v123_root, a.runtime_root, a.workers)
    elif a.command == 'run':
        run(a.out_root, a.runtime_root, a.start, a.end, a.workers)
    else:
        aggregate(a.out_root, a.v123_root)
