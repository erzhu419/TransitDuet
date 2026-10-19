#!/usr/bin/env python3
"""Shadow a launch-origin ETA handoff on four unchanged, consumed paths."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np

from freqduet_online_component_eta import OnlineComponentETA, model
import run_freqduet_v124_eta_factorial as bilateral
import run_freqduet_v132_passive_terminal_trace as prior

pilot, original = prior.pilot, prior.original
ITEMS = tuple((s, 2231004, active) for s in (12011, 933013) for active in (False, True))
SOURCES = ('same_time_avl_route_speed_eta', 'same_time_avl_current_speed_eta')


def forecast_key(evidence, direction, station):
    return int(evidence['bus_id']), bool(direction), int(station), evidence['ready_time_s']


def launch_handoff(stream, evidence, direction, station, base):
    if base['component_used'] or evidence['follower_source'] not in SOURCES or base['gap_s'] is None:
        return base
    now, trip = evidence['ready_time_s'], evidence['predicted_follower_trip_id']
    if trip not in stream.launches or stream.launches[trip] > now:
        return base
    route = stream.geometry[direction]
    target = route['index'][station]
    progress = route['positions'][target]-evidence['spatial_gap_m']
    if abs(progress) < 1e-8:
        progress = 0.
    if not 0 <= progress < route['positions'][1]-1e-8:
        return base
    origin = stream.motion[trip].origin(now, progress, route)
    # A currently observed launch is an origin, not a completed historical label.
    if origin is None and stream.launches[trip] == now and progress == 0:
        origin = 0
    if origin != 0:
        return base
    prediction = model.component_prediction(stream.history, route, direction, origin, target, progress, now)
    if prediction is None:
        return base
    gap = prediction['running_s']+(prediction['service_s']+prediction['holding_s']+prediction['ticks_s'])
    return dict(base, **prediction, status='launch_handoff_component', component_used=True,
        gap_s=gap+prediction['target_service_s'], launch_observed_s=stream.launches[trip], progress_m=progress)


class ShadowLaunchETA(OnlineComponentETA):
    def __init__(self, geometry):
        super().__init__(geometry)
        self.shadow = {}

    def predict(self, evidence, direction, station):
        base = super().predict(evidence, direction, station)
        proposed = launch_handoff(self, evidence, direction, station, base)
        self.shadow[forecast_key(evidence, direction, station)] = proposed
        if proposed['component_used'] and proposed['latest_completion_s'] >= evidence['ready_time_s']:
            raise AssertionError('handoff used a simultaneous or unfinished historical label')
        return base


class HandoffAudit(prior.PassiveTrace):
    def __init__(self, seed, geometry, active):
        super().__init__(seed, geometry)
        self.active = active
        self.stream = ShadowLaunchETA(geometry)

    def __call__(self, policy, states, features, theta):
        law = super().__call__(policy, states, features, theta)
        stop = prior.key(self.decisions[-1])
        captured = self.queries[stop]
        proposed = self.stream.shadow[forecast_key(captured['evidence'], stop[2], stop[1])]
        context = self.runner._pending_lower_action_context[int(self.runner._pilot_agent)]
        selected = bilateral.bilateral_action(context, proposed, policy.feasible_action_mask(states)[0].numpy())
        captured.update(handoff=deepcopy(proposed),
            shadow_chosen_s=float(bilateral.prior.ACTIONS[selected]),
            shadow_request_s=.25*bilateral.prior.request_error(context, proposed))
        return law


def label_same_follower(row, ready):
    e = row['evidence']
    target = e['predicted_follower_trip_id'], row['station_id'], row['direction']
    if target not in ready:
        return dict(resolved=False, reason='missing_same_trip_ready')
    label = ready[target]
    if label['bus_id'] != e['predicted_follower_bus_id']:
        return dict(resolved=False, reason='bus_identity_mismatch')
    gap = label['ready_time_s']-e['ready_time_s']
    if gap < 0:
        return dict(resolved=False, reason='same_follower_already_ready')
    return dict(resolved=True, same_follower_gap_s=gap,
        original_error_s=row['online']['gap_s']-gap, handoff_error_s=row['handoff']['gap_s']-gap)


def run_item(out_root, v124, v130, v132, runtime, index):
    seed, day, active = ITEMS[index]
    label = f'{seed}_{day}_split_'+('plan' if active else 'neutral')
    rule = HandoffAudit(seed, model.load_geometry(runtime), active)
    credit = pilot.CreditObserver()
    result, _, _, _ = pilot.rollout(out_root, seed, day, 'stochastic', 187000000+seed+100*day,
        [0.]*4, 'v130_'+label, probability_rule=rule, trace_observer=rule, credit_observer=credit)
    result.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    values = {k: result[k] for k in original.frozen.prior.METRICS}
    # Future artifacts and same-follower outcomes are read only after the frozen rollout.
    if active:
        values.update({k: rule.terminal[k] for k in ('inclusive_terminal_delay_s', 'qualification_shift_s')})
        cached = json.loads((v130/'pairs'/f'p{original.ITEMS.index((seed, day))}.json').read_text())
        old = next(r for r in cached['paths'] if r['method'] == 'split_plan')
        actual = dict(seed=seed, day=day, method='split_plan', values=values,
            terminal=rule.terminal, contract=rule.contract, neutral_exact=False)
        record_before = v130/'records'/(label+'.npz')
        trace_before = v132/'traces'/(label+'.jsonl.gz')
        terminal = dict(plans=rule.plans, dispatches=rule.dispatches,
            returns=list(rule.returns.values()), reference_clocks=rule.reference_clocks)
        terminal_path = out_root/'terminal'/(label+'.json')
        pilot.write_json(terminal_path, terminal)
        pilot.joint.assert_same_state(json.loads((v130/'terminal'/(label+'.json')).read_text()),
            json.loads(terminal_path.read_text()), 'V133_unchanged_terminal_commands_and_turnover')
    else:
        name = f'{seed}_{day}_{bilateral.PRIMARY}'
        old = json.loads((v124/'pairs'/(name+'.json')).read_text())
        full = dict(seed=seed, day=day, method=bilateral.PRIMARY, rep=0, results=result,
            contract=rule.contract, coverage=rule.coverage, feedforward=rule.feedforward,
            accounting=rule.accounting, prediction_use=rule.prediction_use)
        actual = {k: full[k] for k in old}
        record_before = v124/'records'/(name+'_0.npz')
        trace_before = v124/'traces'/(name+'_0.jsonl.gz')
    pilot.joint.assert_same_state(old, actual, 'V133_unchanged_consumed_endpoints')
    record = out_root/'records'/(label+'.npz')
    rule.save(record)
    prior.assert_arrays_and_actions(record_before, record, rule.decisions)
    rows = [dict(r, **rule.queries[prior.key(r)]) for r in rule.decisions]
    saved = prior.identity.read_trace(trace_before)
    prior.identity.assert_decisions([{k: v for k, v in r.items() if k != 'ready_avl'} for r in saved],
        [{k: r[k] for k in before if k != 'ready_avl'} for before, r in zip(saved, rows)])
    if len(saved) != len(rows):
        raise AssertionError('passive handoff changed the query count')
    changed = []
    for row in rows:
        if row['handoff']['status'] != 'launch_handoff_component':
            pilot.joint.assert_same_state(row['online'], row['handoff'], 'V133_unchanged_nonlaunch_forecast')
            if row['shadow_chosen_s'] != row['chosen_s']:
                raise AssertionError('handoff changed an action outside the launch extension')
            continue
        changed.append(dict(trip_id=row['trip_id'], station_id=row['station_id'], direction=row['direction'],
            evidence=row['evidence'], original=row['online'], proposed=row['handoff'],
            chosen_s=row['chosen_s'], shadow_chosen_s=row['shadow_chosen_s'],
            original_request_s=.25*row['feedback_error_s'], shadow_request_s=row['shadow_request_s'],
            label=label_same_follower(row, rule.ready)))
    prior.identity.write_trace(out_root/'traces'/(label+'.jsonl.gz'), rows)
    pilot.write_json(out_root/'pairs'/f'p{index}.json', dict(seed=seed, day=day, terminal_active=active,
        endpoints_exact=True, full_timing_arrays_exact=True, original_decisions_and_evidence_exact=True,
        all_realized_actions_exact=True, nonlaunch_forecasts_exact=True, decisions=len(rows),
        values=values, modified_forecasts=changed, actor_updates=0, critic_updates=0, dual_updates=0))
    print('V133_PASSIVE_HANDOFF_PATH_DONE', index, seed, active, len(changed), flush=True)


def errors(rows):
    resolved = [r['label'] for r in rows if r['label']['resolved']]
    n = len(resolved)
    return dict(modified=len(rows), resolved=n,
        unresolved=dict(Counter(r['label']['reason'] for r in rows if not r['label']['resolved'])),
        original_mae_s=None if not n else sum(abs(r['original_error_s']) for r in resolved)/n,
        handoff_mae_s=None if not n else sum(abs(r['handoff_error_s']) for r in resolved)/n)


def aggregate(args):
    cells = [json.loads((args.out_root/'pairs'/f'p{i}.json').read_text()) for i in range(len(ITEMS))]
    for row, (seed, day, active) in zip(cells, ITEMS):
        if (row['seed'], row['day'], row['terminal_active']) != (seed, day, active):
            raise AssertionError('handoff audit changed the consumed four-path grid')
        if not all(row[k] for k in ('endpoints_exact', 'full_timing_arrays_exact',
                'original_decisions_and_evidence_exact', 'all_realized_actions_exact', 'nonlaunch_forecasts_exact')):
            raise AssertionError('passive handoff did not preserve the deployed path')
    changed = [r for cell in cells for r in cell['modified_forecasts']]
    anchor = next(r for cell in cells if cell['seed'] == 12011 and cell['terminal_active']
        for r in cell['modified_forecasts'] if r['trip_id'] == 16 and r['station_id'] == 5)
    old_anchor = json.loads((args.v132_root/'summary/status.json').read_text())['paths'][0]['comparison']['first_changed_queries'][0]
    anchor_preserved = anchor['shadow_chosen_s'] == old_anchor['baseline']['chosen_s'] == 0.
    statistics = errors(changed)
    admitted = (anchor_preserved and statistics['modified'] > 0
        and statistics['resolved'] == statistics['modified']
        and statistics['handoff_mae_s'] <= statistics['original_mae_s'])
    for cell in cells:
        cell['prediction_errors'] = errors(cell['modified_forecasts'])
        cell['shadow_changed_actions'] = sum(r['chosen_s'] != r['shadow_chosen_s'] for r in cell['modified_forecasts'])
        cell['modified_forecast_examples'] = cell.pop('modified_forecasts')[:3]
    pilot.write_json(args.out_root/'summary/status.json', dict(status='v133_passive_launch_handoff_complete',
        actor_day_terminal=cells, prediction_errors=statistics, anchor=anchor,
        anchor_baseline_request_s=old_anchor['baseline']['unclipped_request_s'],
        anchor_baseline_gap_s=old_anchor['baseline']['request_gap_s'],
        anchor_removes_30s_step=anchor_preserved, diagnostic_admission=admitted,
        new_rollouts=4, new_demand_days=0, actor_updates=0, critic_updates=0, dual_updates=0,
        main_policy_changed=False,
        limitations='Consumed-path shadow forecast audit only. Same-follower labels are post-rollout '
            'outcomes under the original actions, not candidate control effects. The first-segment '
            'extension uses observed launch/progress and unchanged strictly earlier completed means. '
            'Prediction admission only permits a separate closed-loop pilot; no policy promotion.'))
    print('V133_PASSIVE_HANDOFF_COMPLETE', admitted, statistics, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke', 'run', 'aggregate'))
    parser.add_argument('--out-root', type=Path, required=True)
    for name in ('v124-root', 'v130-root', 'v132-root', 'runtime-root'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'smoke':
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v133_*.py', '-v'], check=True)
        pilot.write_json(args.out_root/'smoke/done.json', dict(status='launch_handoff_tests_passed'))
    elif args.command == 'run':
        if not (args.out_root/'smoke/done.json').exists():
            raise AssertionError('passive handoff requires passed focused tests')
        with ProcessPoolExecutor(max_workers=min(args.workers, len(ITEMS)), mp_context=get_context('spawn')) as pool:
            jobs = [pool.submit(run_item, args.out_root, args.v124_root, args.v130_root,
                args.v132_root, args.runtime_root, i) for i in range(len(ITEMS))]
            for job in as_completed(jobs):
                job.result()
        pilot.write_json(args.out_root/'rollouts/done.json', dict(status='four_exact_passive_paths_complete'))
    else:
        aggregate(args)
