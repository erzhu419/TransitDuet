#!/usr/bin/env python3
"""Executable terminal-spacing x frozen lower-rule development factorial."""

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v127_turnover_replay as prior

frozen = prior.frozen
GAP_S = 360.
ITEMS, SMOKE_INDEX = prior.ITEMS, prior.SMOKE_INDEX
METHODS = ('raw', 'split', 'raw_spacing', 'split_spacing')
LOWER = dict(raw=frozen.prior.BILATERAL, split=frozen.PRIMARY)
CONTRASTS = {b+'_vs_'+a: (b, a) for a, b in itertools.combinations(METHODS, 2)}
METRICS = (*frozen.prior.METRICS, 'inclusive_terminal_delay_s', 'qualification_shift_s')


def published_release(trip):
    published = float(getattr(trip, '_freqduet_scheduled_launch', trip.launch_time))
    if not getattr(trip, '_freqduet_terminal_dispatch', False) and hasattr(trip, '_delta_t'):
        published = max(published, float(trip._original_launch+trip._delta_t))
    return published


def install_spacing(trip):
    release = published_release(trip)
    if getattr(trip, '_freqduet_min_dispatch_headway', None) is not None:
        raise AssertionError('V128 requires the original no-spacing-control deployment')
    trip._freqduet_scheduled_launch = release
    trip._freqduet_terminal_dispatch = True
    trip._freqduet_min_dispatch_headway = GAP_S
    return release


def terminal_metrics(dispatches, clocks, coordinated):
    inclusive, native, shifts, gaps, latest = [], [], [], [], {}
    for row in dispatches:
        trip, direction, actual = row['trip_id'], row['direction'], row['actual_s']
        base = clocks[trip]
        inclusive.append(actual-base)
        native.append(actual-row['eligible_s'])
        shifts.append(row['eligible_s']-base)
        if direction in latest:
            before, previous_trip = latest[direction]
            gaps.append(actual-before)
            if coordinated and (gaps[-1] < GAP_S or trip <= previous_trip):
                raise AssertionError('terminal spacing or same-direction trip order was not executed')
        latest[direction] = actual, trip
    if min(inclusive) < 0 or min(shifts) < 0:
        raise AssertionError('terminal control launched before its published eligibility')
    np.testing.assert_array_equal(np.asarray(inclusive), np.asarray(native)+shifts)
    if not coordinated:
        np.testing.assert_array_equal(shifts, np.zeros(len(shifts)))
    return dict(inclusive_terminal_delay_s=float(np.mean(inclusive)),
        native_readiness_delay_s=float(np.mean(native)), qualification_shift_s=float(np.mean(shifts)),
        shifted_trips=sum(v > 0 for v in shifts), maximum_qualification_shift_s=max(shifts),
        dispatches=len(dispatches), minimum_actual_gap_s=min(gaps))


class TerminalAudit(prior.TurnoverAudit):
    def __init__(self, seed, geometry, arm, coordinated):
        super().__init__(seed, geometry, arm)
        self.coordinated = coordinated
        self.reference_clocks, self.pending_clocks, self.plans = {}, {}, []

    def attach(self, runner):
        super().attach(runner)
        self.spacing_upper, self.spacing_step = runner._upper_callback_v2, self.env.step

        def upper(state, trip):
            result = self.spacing_upper(state, trip)
            now, identity = float(self.env.current_time), int(trip.launch_turn)
            # The native gate moves its readiness origin; retain the prior plan clock.
            release = published_release(trip)
            if identity in self.reference_clocks or identity in self.pending_clocks:
                raise AssertionError('upper policy queried a terminal trip twice')
            if release <= now:
                self.reference_clocks[identity] = now
            else:
                self.pending_clocks[identity] = release
            self.plans.append(dict(trip_id=identity, query_s=now, published_release_s=release,
                target_headway_s=float(result), last_actual_release_s=float(self.env._last_dispatch_time[trip.direction]),
                current_ready_same=self.env._ready_vehicle_count(trip.direction)))
            if self.coordinated:
                install_spacing(trip)
            return result

        def step(*args, **kwargs):
            now = float(self.env.current_time)
            for trip, release in list(self.pending_clocks.items()):
                if release <= now:
                    self.reference_clocks[trip] = now
                    del self.pending_clocks[trip]
            return self.spacing_step(*args, **kwargs)

        runner._upper_callback_v2, self.env.step = upper, step

    def detach(self):
        self.runner._upper_callback_v2, self.env.step = self.spacing_upper, self.spacing_step
        super().detach()

    def finish(self, runner, records, holds):
        super().finish(runner, records, holds)
        self.terminal = terminal_metrics(self.dispatches, self.reference_clocks, self.coordinated)
        np.testing.assert_allclose(self.terminal['native_readiness_delay_s'],
            runner.env._measurement_details['fleet_readiness_delay_mean_s'], atol=1e-9, rtol=0)
        if len(self.plans) != len(self.dispatches):
            raise AssertionError('terminal commands do not cover the actual dispatch roster')
        self.spacing_upper = self.spacing_step = None


def run_path(root, v123, v124, runtime, seed, day, lower, coordinated):
    arm = LOWER[lower]
    rule = TerminalAudit(seed, frozen.prior.model.load_geometry(runtime), arm, coordinated)
    credit = frozen.pilot.CreditObserver()
    label = f'{seed}_{day}_{lower}_'+('spacing' if coordinated else 'neutral')
    result, _, _, _ = frozen.pilot.rollout(root, seed, day, 'stochastic', 187000000+seed+100*day,
        [0.]*4, 'v128_'+label, probability_rule=rule, trace_observer=rule, credit_observer=credit)
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
    frozen.pilot.joint.assert_same_state(old['contract'], actual['contract'], 'V128_shared_physics')
    if not coordinated:
        frozen.pilot.joint.assert_same_state(old, {k: actual[k] for k in old}, 'V128_exact_neutral_endpoints')
        with np.load(cached/'records'/(old_name+'_0.npz')) as before, np.load(record) as after:
            if before.files != after.files:
                raise AssertionError('neutral terminal observer changed timing fields')
            for field in before.files:
                np.testing.assert_array_equal(before[field], after[field])
        if lower == 'split':
            trace = frozen.prior.prior.read_trace(cached/'traces'/(old_name+'_0.jsonl.gz'))
            frozen.prior.prior.assert_decisions(trace,
                [dict(r, evidence=rule.evidence[prior.timing.elapsed.key(r)]) for r in rule.decisions])
    frozen.pilot.write_json(root/'terminal'/(label+'.json'), dict(plans=rule.plans, dispatches=rule.dispatches,
        returns=list(rule.returns.values()), reference_clocks=rule.reference_clocks))
    values = {k: result[k] for k in frozen.prior.METRICS}
    values.update({k: rule.terminal[k] for k in ('inclusive_terminal_delay_s', 'qualification_shift_s')})
    row = dict(seed=seed, day=day, method=lower+('_spacing' if coordinated else ''),
        values=values, terminal=rule.terminal, contract=rule.contract, neutral_exact=not coordinated)
    frozen.pilot.write_json(root/'paths'/(label+'.json'), row)
    return row


def run_cell(root, v123, v124, runtime, index):
    receipt = root/'pairs'/f'p{index}.json'
    if receipt.exists():
        return
    seed, day = ITEMS[index]
    rows = [run_path(root, v123, v124, runtime, seed, day, lower, True) for lower in ('raw', 'split')]
    frozen.pilot.write_json(receipt, dict(seed=seed, day=day, index=index, paths=rows))
    print('V128_TERMINAL_FACTORIAL_CELL_DONE', index, seed, day, flush=True)


def factorial_effects(levels):
    effects = {name: {k: levels[a][k]-levels[b][k] for k in METRICS} for name, (a, b) in CONTRASTS.items()}
    effects['terminal_x_lower'] = {k: (levels['split_spacing'][k]-levels['split'][k])
        -(levels['raw_spacing'][k]-levels['raw'][k]) for k in METRICS}
    return effects


def joint_gate(stats):
    return stats['passenger_person_s']['ci95'][1] < 0 and all(stats[k]['ci95'][1] <= 0 for k in
        ('cv_squared', 'inclusive_terminal_delay_s', 'trips_unlaunched', 'trips_incomplete'))


def aggregate(root, v127):
    parent = json.loads((v127/'summary/status.json').read_text())
    paired, absolute, receipts = [], [], []
    for index, (seed, day) in enumerate(ITEMS):
        current = json.loads((root/'pairs'/f'p{index}.json').read_text())
        old = next(r for r in parent['actor_day'] if (r['seed'], r['day']) == (seed, day))
        if (current['seed'], current['day'], current['index']) != (seed, day, index):
            raise AssertionError('terminal factorial changed the consumed actor/day grid')
        levels = {r['method']: r['values'] for r in current['paths']}
        for lower, original in LOWER.items():
            baseline = dict(next(r['values'] for r in old['paths'] if r['method'] == original))
            baseline.update(inclusive_terminal_delay_s=baseline['readiness_delay_s'], qualification_shift_s=0.)
            levels[lower] = baseline
        paired.append(dict(seed=seed, day=day, effects=factorial_effects(levels)))
        absolute.append(dict(seed=seed, day=day, values=levels))
        receipts.extend(current['paths'])
    stats = frozen.prior.inference.describe(paired, 'effects')
    gates = {base: joint_gate(stats['split_spacing_vs_'+base]) for base in ('raw', 'split', 'raw_spacing')}
    frozen.pilot.write_json(root/'summary/status.json', dict(status='v128_terminal_lower_factorial_complete',
        effects=stats, absolute=frozen.prior.inference.describe(absolute, 'values'), actor_day=paired,
        axes=frozen.prior.axes.axis_summary(paired, 'effects', list(METRICS)), terminal_receipts=receipts,
        diagnostic_gates=gates, joint_all_controls=all(gates.values()), reused_control_paths=8,
        new_treated_paths=8, neutral_replays=2, new_demand_days=0, terminal_gap_s=GAP_S,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        limitations='Consumed two-actor/two-day executable controller screen, not independent confirmation. '
            'Only current published commitments, past actual releases and actual inventory drive execution. '
            'The native minimum-spacing gate is not capped; its added waiting is included in the primary '
            'published-eligibility-to-launch delay, while native readiness is reported separately. '
            'Same learned policy maps/weights continue closed-loop; their observations/actions may change. '
            'Any common terminal gain is not evidence for frequency necessity or superiority to fixed.'))
    print('V128_TERMINAL_LOWER_FACTORIAL_COMPLETE', gates, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'run', 'aggregate'))
    for name in ('out-root', 'v123-root', 'v124-root', 'v127-root', 'runtime-root'):
        p.add_argument('--'+name, type=Path, required=name == 'out-root')
    p.add_argument('--index', type=int)
    a = p.parse_args()
    if a.command == 'smoke':
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v128_*.py', '-v'], check=True)
        seed, day = ITEMS[SMOKE_INDEX]
        for lower in ('raw', 'split'):
            run_path(a.out_root, a.v123_root, a.v124_root, a.runtime_root, seed, day, lower, False)
        run_cell(a.out_root, a.v123_root, a.v124_root, a.runtime_root, SMOKE_INDEX)
        rows = json.loads((a.out_root/'pairs'/f'p{SMOKE_INDEX}.json').read_text())['paths']
        if not all(r['terminal']['shifted_trips'] > 0 for r in rows):
            raise AssertionError('both lower rules must exercise executable terminal spacing')
        frozen.pilot.write_json(a.out_root/'smoke/done.json', dict(status='v128_exact_neutrals_and_executed_spacing_passed',
            neutral_paths=2, treated_paths_reused=2, index=SMOKE_INDEX))
    elif a.command == 'run':
        if not (a.out_root/'smoke/done.json').exists():
            raise AssertionError('remaining treated paths require exact-neutral and execution smoke')
        run_cell(a.out_root, a.v123_root, a.v124_root, a.runtime_root, a.index)
        frozen.pilot.write_json(a.out_root/'shards'/str(a.index)/'done.json', dict(status='complete', index=a.index))
    else:
        aggregate(a.out_root, a.v127_root)
