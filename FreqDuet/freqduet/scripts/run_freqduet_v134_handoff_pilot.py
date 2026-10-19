#!/usr/bin/env python3
"""Consumed four-path closed-loop pilot after the frozen V133 shadow audit."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v133_launch_handoff as prior

pilot, original = prior.pilot, prior.original
ITEMS = prior.ITEMS
METRICS = (*original.frozen.prior.METRICS, 'inclusive_terminal_delay_s', 'qualification_shift_s')


class ActiveLaunchETA(prior.ShadowLaunchETA):
    def __init__(self, geometry):
        super().__init__(geometry)
        self.original_predictions = {}

    def predict(self, evidence, direction, station):
        base = super().predict(evidence, direction, station)
        identity = prior.forecast_key(evidence, direction, station)
        self.original_predictions[identity] = base
        return self.shadow[identity]


class PilotAudit(prior.prior.PassiveTrace):
    def __init__(self, seed, geometry, active):
        super().__init__(seed, geometry)
        self.active = active
        self.stream = ActiveLaunchETA(geometry)

    def __call__(self, policy, states, features, theta):
        law = super().__call__(policy, states, features, theta)
        stop = prior.prior.key(self.decisions[-1])
        captured = self.queries[stop]
        forecast = prior.forecast_key(captured['evidence'], stop[2], stop[1])
        base = self.stream.original_predictions[forecast]
        context = self.runner._pending_lower_action_context[int(self.runner._pilot_agent)]
        selected = prior.bilateral.bilateral_action(context, base, policy.feasible_action_mask(states)[0].numpy())
        captured.update(original_prediction=base,
            original_command_same_state_s=float(prior.bilateral.prior.ACTIONS[selected]))
        return law


def run_item(root, v124, v130, v133, runtime, index):
    seed, day, active = ITEMS[index]
    label = f'{seed}_{day}_split_'+('plan' if active else 'neutral')
    rule = PilotAudit(seed, prior.model.load_geometry(runtime), active)
    credit = pilot.CreditObserver()
    result, _, _, _ = pilot.rollout(root, seed, day, 'stochastic', 187000000+seed+100*day,
        [0.]*4, 'v134_'+label, probability_rule=rule, trace_observer=rule, credit_observer=credit)
    result.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    values = {k: result[k] for k in original.frozen.prior.METRICS}
    values.update({k: rule.terminal[k] for k in ('inclusive_terminal_delay_s', 'qualification_shift_s')})
    # Original development labels enter comparison only after the live rollout.
    if active:
        cached = json.loads((v130/'pairs'/f'p{original.ITEMS.index((seed, day))}.json').read_text())
        contract = next(r['contract'] for r in cached['paths'] if r['method'] == 'split_plan')
    else:
        contract = json.loads((v124/'pairs'/f'{seed}_{day}_{prior.bilateral.PRIMARY}.json').read_text())['contract']
    pilot.joint.assert_same_state(contract, rule.contract, 'V134_shared_frozen_physics')
    control = json.loads((v133/'pairs'/f'p{index}.json').read_text())
    if (control['seed'], control['day'], control['terminal_active']) != (seed, day, active):
        raise AssertionError('closed-loop control differs from the admitted V133 path')
    if not all(control[k] for k in ('endpoints_exact', 'full_timing_arrays_exact',
            'original_decisions_and_evidence_exact', 'all_realized_actions_exact', 'nonlaunch_forecasts_exact')):
        raise AssertionError('V134 control was not exactly reproduced in V133')
    baseline = dict(control['values'])
    if not active:
        with np.load(v133/'records'/(label+'.npz')) as record:
            field = record['launch_fields'].tolist().index('readiness_s')
            baseline['inclusive_terminal_delay_s'] = float(np.mean(record['launches'][:, field]))
        baseline['qualification_shift_s'] = 0.
    rows = [dict(r, **rule.queries[prior.prior.key(r)]) for r in rule.decisions]
    rule.save(root/'records'/(label+'.npz'))
    prior.prior.identity.write_trace(root/'traces'/(label+'.jsonl.gz'), rows)
    pilot.write_json(root/'terminal'/(label+'.json'), dict(plans=rule.plans, dispatches=rule.dispatches,
        returns=list(rule.returns.values()), reference_clocks=rule.reference_clocks))
    handoffs = [r for r in rows if r['online']['status'] == 'launch_handoff_component']
    pilot.write_json(root/'pairs'/f'p{index}.json', dict(seed=seed, day=day, terminal_active=active,
        values=values, baseline=baseline, delta={k: values[k]-baseline[k] for k in METRICS},
        launch_forecasts=len(handoffs), direct_command_changes=sum(r['chosen_s']
            != r['original_command_same_state_s'] for r in handoffs),
        first_handoff_examples=[dict(trip_id=r['trip_id'], station_id=r['station_id'],
            policy_time_s=r['policy_time_s'], chosen_s=r['chosen_s'], online=r['online'],
            original_prediction=r['original_prediction'],
            original_command_same_state_s=r['original_command_same_state_s']) for r in handoffs[:3]],
        actor_updates=0, critic_updates=0, dual_updates=0))
    print('V134_HANDOFF_PILOT_PATH_DONE', index, seed, active, values, flush=True)


def joint_improvement(delta):
    return delta['passenger_person_s'] < 0 and all(delta[k] <= 0 for k in
        ('cv_squared', 'inclusive_terminal_delay_s', 'trips_unlaunched', 'trips_incomplete'))


def aggregate(args):
    cells = [json.loads((args.out_root/'pairs'/f'p{i}.json').read_text()) for i in range(len(ITEMS))]
    for row, item in zip(cells, ITEMS):
        if (row['seed'], row['day'], row['terminal_active']) != item:
            raise AssertionError('handoff control pilot changed the consumed four-path grid')
        row['joint_improvement'] = joint_improvement(row['delta'])
    by_terminal = {str(int(active)): dict(cells=2,
        mean_delta={k: float(np.mean([r['delta'][k] for r in cells if r['terminal_active'] == active]))
            for k in METRICS}, both_actors_jointly_improve=all(r['joint_improvement']
                for r in cells if r['terminal_active'] == active)) for active in (False, True)}
    interactions = []
    for seed in (12011, 933013):
        off, on = [next(r for r in cells if r['seed'] == seed and r['terminal_active'] == a) for a in (False, True)]
        interactions.append(dict(seed=seed, terminal_x_handoff={k: on['delta'][k]-off['delta'][k] for k in METRICS}))
    pilot.write_json(args.out_root/'summary/status.json', dict(status='v134_closed_loop_handoff_pilot_complete',
        actor_day_terminal=cells, by_terminal=by_terminal, interactions=interactions,
        all_cells_jointly_improve=all(r['joint_improvement'] for r in cells),
        reused_exact_control_paths=4, new_candidate_paths=4, new_demand_days=0,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        limitations='Two actors and one consumed demand day only. Raw paired deltas and interactions '
            'are development diagnostics, not independent-day inference or policy promotion. '
            'Gains, bins, terminal rule and network weights are unchanged; closed-loop states, '
            'subsequent actions and completed-history means may change. No bias refit after V133.'))
    print('V134_CLOSED_LOOP_HANDOFF_PILOT_COMPLETE', by_terminal, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke', 'run', 'aggregate'))
    parser.add_argument('--out-root', type=Path, required=True)
    for name in ('v124-root', 'v130-root', 'v133-root', 'runtime-root'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'smoke':
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v134_*.py', '-v'], check=True)
        admitted = json.loads((args.v133_root/'summary/status.json').read_text())['diagnostic_admission']
        if not admitted:
            raise AssertionError('closed-loop pilot requires admitted V133 shadow evidence')
        pilot.write_json(args.out_root/'smoke/done.json', dict(status='handoff_pilot_tests_and_v133_admission_passed'))
    elif args.command == 'run':
        if not (args.out_root/'smoke/done.json').exists():
            raise AssertionError('closed-loop pilot requires passed smoke')
        with ProcessPoolExecutor(max_workers=min(args.workers, len(ITEMS)), mp_context=get_context('spawn')) as pool:
            jobs = [pool.submit(run_item, args.out_root, args.v124_root, args.v130_root,
                args.v133_root, args.runtime_root, i) for i in range(len(ITEMS))]
            for job in as_completed(jobs):
                job.result()
        pilot.write_json(args.out_root/'rollouts/done.json', dict(status='four_closed_loop_paths_complete'))
    else:
        aggregate(args)
