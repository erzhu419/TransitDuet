#!/usr/bin/env python3
"""Small frozen-controller screen of bounded, plan-anchored terminal coordination."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v128_terminal_factorial as prior

frozen = prior.frozen
ITEMS, SMOKE_INDEX = prior.ITEMS, prior.SMOKE_INDEX
MAX_WAIT_S = 30.
CONTRASTS = {name: (a, b) for name, a, b in (
    ('raw_plan_vs_raw', 'raw_plan', 'raw'), ('split_plan_vs_split', 'split_plan', 'split'),
    ('split_plan_vs_raw', 'split_plan', 'raw'), ('split_plan_vs_raw_plan', 'split_plan', 'raw_plan'),
    ('raw_plan_vs_raw_hard', 'raw_plan', 'raw_hard'),
    ('split_plan_vs_split_hard', 'split_plan', 'split_hard'))}


def planned_command(published, now, last_actual, ready):
    if last_actual > now:
        raise AssertionError('terminal coordinator received a future actual release')
    if ready <= 0 or last_actual < -9000. or now >= published+MAX_WAIT_S:
        return published
    # Equal squared costs for phase delay and gap error give a half-gap response.
    wait = min(MAX_WAIT_S, math.floor(max(0., last_actual+prior.GAP_S-published)/2.))
    return published+wait if published+wait > now else published


def terminal_stats(dispatches, clocks, plans, active):
    delay = np.asarray([r['actual_s']-clocks[r['trip_id']] for r in dispatches])
    native = np.asarray([r['actual_s']-r['eligible_s'] for r in dispatches])
    shift = delay-native
    if min(delay) < 0 or min(shift) < 0 or max(shift) > (MAX_WAIT_S if active else 0):
        raise AssertionError('terminal control exceeded its independent published-clock budget')
    np.testing.assert_array_equal(delay, native+shift)
    requested = [r['command_s']-r['published_release_s'] for r in plans]
    if len(plans) != len(dispatches) or min(requested) < 0 or max(requested) > MAX_WAIT_S:
        raise AssertionError('terminal commands are missing or outside the per-trip budget')
    profiles = {}
    for direction in (False, True):
        rows = [r for r in dispatches if r['direction'] == direction]
        lateness = [r['actual_s']-clocks[r['trip_id']] for r in rows]
        if any(b['trip_id'] <= a['trip_id'] for a, b in zip(rows, rows[1:])):
            raise AssertionError('bounded coordination changed same-direction trip order')
        profiles[str(int(direction))] = dict(final_delay_s=lateness[-1], maximum_delay_s=max(lateness),
            thirds=[float(np.mean(lateness[len(rows)*i//3:len(rows)*(i+1)//3])) for i in range(3)])
    return dict(inclusive_terminal_delay_s=float(delay.mean()), native_readiness_delay_s=float(native.mean()),
        qualification_shift_s=float(shift.mean()), maximum_qualification_shift_s=float(shift.max()),
        commanded_trips=sum(w > 0 for w in requested), cap_hits=sum(w == MAX_WAIT_S for w in requested),
        mean_requested_wait_s=float(np.mean(requested)), directions=profiles)


class PlanAudit(prior.TerminalAudit):
    def __init__(self, seed, geometry, arm, active):
        super().__init__(seed, geometry, arm, False)
        self.active = active

    def attach(self, runner):
        super().attach(runner)
        self.plan_upper = runner._upper_callback_v2

        def upper(state, trip):
            result = self.plan_upper(state, trip)
            row = self.plans[-1]
            release = row['published_release_s']
            command = planned_command(release, row['query_s'], row['last_actual_release_s'],
                row['current_ready_same']) if self.active else release
            row['command_s'] = command
            if command > release:
                if getattr(trip, '_freqduet_min_dispatch_headway', None) is not None:
                    raise AssertionError('bounded candidate must not inherit a hard spacing gate')
                trip._freqduet_terminal_dispatch = True
                trip._freqduet_scheduled_launch = command
            return result

        runner._upper_callback_v2 = upper

    def finish(self, runner, records, holds):
        prior.prior.TurnoverAudit.finish(self, runner, records, holds)
        self.terminal = terminal_stats(self.dispatches, self.reference_clocks, self.plans, self.active)
        np.testing.assert_allclose(self.terminal['native_readiness_delay_s'],
            runner.env._measurement_details['fleet_readiness_delay_mean_s'], atol=1e-9, rtol=0)
        self.plan_upper = self.spacing_upper = self.spacing_step = None


def run_path(root, v123, v124, runtime, seed, day, lower, active):
    arm = prior.LOWER[lower]
    rule = PlanAudit(seed, frozen.prior.model.load_geometry(runtime), arm, active)
    credit = frozen.pilot.CreditObserver()
    label = f'{seed}_{day}_{lower}_'+('plan' if active else 'neutral')
    result, _, _, _ = frozen.pilot.rollout(root, seed, day, 'stochastic', 187000000+seed+100*day,
        [0.]*4, 'v130_'+label, probability_rule=rule, trace_observer=rule, credit_observer=credit)
    result.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    actual = dict(seed=seed, day=day, method=arm, rep=0, results=result, contract=rule.contract,
        coverage=rule.coverage, feedforward=rule.feedforward, accounting=rule.accounting,
        prediction_use=rule.prediction_use)
    record = root/'records'/(label+'.npz')
    rule.save(record)
    cached = v124 if lower == 'split' else v123
    old_name = f'{seed}_{day}_{arm}'
    old = json.loads((cached/'pairs'/(old_name+'.json')).read_text())
    frozen.pilot.joint.assert_same_state(old['contract'], actual['contract'], 'V130_shared_physics')
    if not active:
        frozen.pilot.joint.assert_same_state(old, {k: actual[k] for k in old}, 'V130_exact_neutral_endpoints')
        with np.load(cached/'records'/(old_name+'_0.npz')) as before, np.load(record) as after:
            if before.files != after.files:
                raise AssertionError('neutral bounded observer changed timing fields')
            for field in before.files:
                np.testing.assert_array_equal(before[field], after[field])
        if lower == 'split':
            trace = frozen.prior.prior.read_trace(cached/'traces'/(old_name+'_0.jsonl.gz'))
            frozen.prior.prior.assert_decisions(trace,
                [dict(r, evidence=rule.evidence[prior.prior.timing.elapsed.key(r)]) for r in rule.decisions])
    frozen.pilot.write_json(root/'terminal'/(label+'.json'), dict(plans=rule.plans, dispatches=rule.dispatches,
        returns=list(rule.returns.values()), reference_clocks=rule.reference_clocks))
    values = {k: result[k] for k in frozen.prior.METRICS}
    values.update({k: rule.terminal[k] for k in ('inclusive_terminal_delay_s', 'qualification_shift_s')})
    return dict(seed=seed, day=day, method=lower+('_plan' if active else ''),
        values=values, terminal=rule.terminal, contract=rule.contract, neutral_exact=not active)


def run_cell(args, index):
    receipt = args.out_root/'pairs'/f'p{index}.json'
    if receipt.exists():
        return
    seed, day = ITEMS[index]
    rows = [run_path(args.out_root, args.v123_root, args.v124_root, args.runtime_root,
        seed, day, lower, True) for lower in ('raw', 'split')]
    frozen.pilot.write_json(receipt, dict(seed=seed, day=day, index=index, paths=rows))
    print('V130_PLAN_TERMINAL_CELL_DONE', index, seed, day, flush=True)


def aggregate(args):
    controls = json.loads((args.v127_root/'summary/status.json').read_text())
    hard = json.loads((args.v128_root/'summary/status.json').read_text())
    paired, receipts = [], []
    for index, (seed, day) in enumerate(ITEMS):
        current = json.loads((args.out_root/'pairs'/f'p{index}.json').read_text())
        if (current['seed'], current['day'], current['index']) != (seed, day, index):
            raise AssertionError('bounded coordinator changed the registered consumed grid')
        levels = {r['method']: r['values'] for r in current['paths']}
        old = next(r for r in controls['actor_day'] if (r['seed'], r['day']) == (seed, day))
        for lower, original in prior.LOWER.items():
            baseline = dict(next(r['values'] for r in old['paths'] if r['method'] == original))
            baseline.update(inclusive_terminal_delay_s=baseline['readiness_delay_s'], qualification_shift_s=0.)
            levels[lower] = baseline
            levels[lower+'_hard'] = next(r['values'] for r in hard['terminal_receipts']
                if (r['seed'], r['day'], r['method']) == (seed, day, lower+'_spacing'))
        effects = {name: {k: levels[a][k]-levels[b][k] for k in prior.METRICS}
            for name, (a, b) in CONTRASTS.items()}
        effects['terminal_x_lower'] = {k: (levels['split_plan'][k]-levels['split'][k])
            -(levels['raw_plan'][k]-levels['raw'][k]) for k in prior.METRICS}
        paired.append(dict(seed=seed, day=day, effects=effects, values=levels))
        receipts.extend(current['paths'])
    stats = frozen.prior.inference.describe(paired, 'effects')
    gates = {name: prior.joint_gate(stats[name]) for name in
        ('raw_plan_vs_raw', 'split_plan_vs_split', 'split_plan_vs_raw')}
    frozen.pilot.write_json(args.out_root/'summary/status.json', dict(status='v130_plan_terminal_complete',
        effects=stats, absolute=frozen.prior.inference.describe(paired, 'values'), actor_day=paired,
        axes=frozen.prior.axes.axis_summary(paired, 'effects', list(prior.METRICS)),
        terminal_receipts=receipts, diagnostic_gates=gates, primary_contrast='split_plan_vs_split',
        maximum_requested_wait_s=MAX_WAIT_S, gap_response_gain=.5, reused_control_paths=16,
        new_treated_paths=8, neutral_replays=2, new_demand_days=0,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        limitations='Consumed two-actor/two-day development screen, no independent confirmation or promotion. '
            'Only current query-time published plans, current ready counts and past actual releases enter '
            'a once-only fixed command. The 30-second cap bounds discretionary eligibility shift, not '
            'endogenous fleet delay. Half-gap response contracts lateness without fresh supply shocks; '
            'system-wide recovery and passenger/CV benefits require realized evaluation. No saved future '
            'labels or V129 hindsight advances set the rule. Policy maps/weights remain fixed, but their '
            'closed-loop states and actions may change.'))
    print('V130_PLAN_TERMINAL_COMPLETE', gates, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke', 'run', 'aggregate'))
    for name in ('out-root', 'v123-root', 'v124-root', 'v127-root', 'v128-root', 'runtime-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.command == 'smoke':
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v130_*.py', '-v'], check=True)
        seed, day = ITEMS[SMOKE_INDEX]
        for lower in ('raw', 'split'):
            run_path(args.out_root, args.v123_root, args.v124_root, args.runtime_root, seed, day, lower, False)
        run_cell(args, SMOKE_INDEX)
        rows = json.loads((args.out_root/'pairs'/f'p{SMOKE_INDEX}.json').read_text())['paths']
        if not all(r['terminal']['commanded_trips'] > 0 for r in rows):
            raise AssertionError('smoke must exercise both bounded coordinators')
        frozen.pilot.write_json(args.out_root/'smoke/done.json', dict(status='v130_exact_neutral_and_bounded_execution_passed', index=SMOKE_INDEX))
    elif args.command == 'run':
        if not (args.out_root/'smoke/done.json').exists():
            raise AssertionError('bounded paired screen requires passed smoke')
        run_cell(args, args.index)
        frozen.pilot.write_json(args.out_root/'shards'/str(args.index)/'done.json', dict(status='complete', index=args.index))
    else:
        aggregate(args)
