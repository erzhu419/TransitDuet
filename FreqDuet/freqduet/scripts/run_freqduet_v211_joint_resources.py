#!/usr/bin/env python3
"""Frozen same-authority upper/holding opportunity pilot with independent futures."""

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import run_freqduet_v106_service_tradeoff as prior
import run_freqduet_v68_repeated_continuations as repeated


pilot, confirmation = prior.pilot, prior.confirmation
SEEDS, DAYS = prior.SEEDS, (2061001, 2061004)
CONTEXTS = tuple(itertools.product(SEEDS, DAYS))
TARGET_S = 7200.
UPPER = (0., -30., 30.)
LOWER = ((0, 0), (-1, -1), (1, 1), (-1, 1), (1, -1))
CANDIDATES = tuple(itertools.product(range(3), range(5)))
SELECT_FUTURES, EVAL_FUTURES = tuple(range(16)), tuple(range(16, 32))
METHODS = ('bilateral', 'upper_only', 'lower_only', 'joint', 'native_suffix', 'preserve_suffix')
METRICS = (*prior.METRICS, 'inclusive_launch_delay_s')
CONSTRAINTS = ('cv_squared', 'readiness_delay_s', 'inclusive_launch_delay_s', 'trips_unlaunched', 'trips_incomplete')


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def future_seed(context, future):
    return 2110000000+100*context+future


def upper_candidate(native, low, high, direction, shift):
    native = np.asarray(native, np.float32).reshape(-1)
    require(native.shape == (2,), 'V211 requires original two directional antisymmetric coefficients')
    result = native.copy()
    result[0 if direction else 1] += shift
    return np.clip(result, low, high).astype(np.float32)


def lower_candidate(base, offset, feasible):
    allowed = np.flatnonzero(feasible)
    require(base in allowed, 'incumbent lower bin is infeasible')
    return int(allowed[np.clip(int(np.flatnonzero(allowed == base)[0])+offset, 0, len(allowed)-1)])


def time_account(launch, returned, holding):
    elapsed = returned-launch
    require(elapsed >= holding >= 0., 'negative physical vehicle-time ledger')
    return dict(launch_s=launch, return_s=returned, holding_s=holding, other_elapsed_s=elapsed-holding)


def resource_snapshot(env, launches):
    ready, active = [], []
    for bus in env.bus_all:
        if not bus.on_route:
            ready.append([int(bus.bus_id), int(bus.direction)])
        else:
            trip = int(bus.trip_id)
            active.append([int(bus.bus_id), trip, int(bus.direction),
                float(bus.episode_hold_vehicle_seconds)-launches[trip]['holding_counter_at_launch']])
    plans = [[int(t.launch_turn), int(t.direction), float(t.launch_time),
        float(getattr(t, '_freqduet_scheduled_launch', t.launch_time))]
        for t in env.timetables if not t.launched]
    require(len(ready)+len(active) == 12 and min((r[3] for r in active), default=0.) >= 0.,
        'resource snapshot changed fleet or spent holding')
    return dict(query_s=float(env.current_time), ready=sorted(ready), active=sorted(active), plans=plans,
        guaranteed_enroute_slack_s=None, discretionary_credit_issued_s=0.,
        pending_same_direction_commitments=[sum(t[1] == d and t[3] <= env.current_time for t in plans) for d in (0, 1)])


class JointAudit(prior.TradeoffAudit):
    def __init__(self, seed, candidate=0, *, reference=None, fork=None, continuation='bilateral025', uniforms=None):
        super().__init__(seed, 'bilateral025')
        self.candidate, self.reference, self.fork = candidate, reference, fork
        self.continuation, self.uniforms = continuation, uniforms
        self.anchor, self.current_upper_trip, self.policy = None, None, None
        self.upper_prefix, self.lower_prefix, self.lower_changes = [], [], []
        self.first_lower, self.applied_upper, self.return_ledger, self.dispatch_ledger = None, None, {}, []

    def attach(self, runner):
        super().attach(runner)
        env = runner.env
        planner = runner.timetable_planner
        require(env.fleet_inventory_mode == 'fixed_pool' and runner.cfg['upper']['N_fleet'] == 12,
            'V211 cannot add vehicles or fleet buffer')
        require(pilot.step.ACTIONS.tolist() == [0., 5., 10., 15., 20., 30., 45.], 'original seven-bin authority differs')
        require(planner is not None and runner.timetable_terminal_dispatch
            and planner.terminal_schedule_mode == 'exact_headway_curve'
            and planner.headway_budget_mode == 'rolling_zero_sum_delta_v6'
            and planner.coefficient_parameterization == 'antisymmetric_linear_v5'
            and runner.upper_action_dim == 2 and np.all(runner.upper_action_low == -60.)
            and np.all(runner.upper_action_high == 60.), 'original upper authority differs')
        self.authority = dict(fleet=12, inventory='fixed_pool', lower_bins=pilot.step.ACTIONS.tolist(),
            upper_low=runner.upper_action_low.tolist(), upper_high=runner.upper_action_high.tolist(),
            planner=copy.deepcopy(runner.cfg['upper']['timetable_planner']))
        self.frozen_scalars = (float(runner.lower_trainer.alpha), float(runner.lower_trainer.lambda_param),
            float(runner.upper_trainer.alpha))
        self.upper_original, self.get_action_original = runner._upper_callback_v2, runner.upper_trainer.policy_net.get_action
        self.launch_original = env.launch_bus
        from env.bus import Bus
        self.bus_class, self.drive_original = Bus, Bus.drive

        def upper(state, trip):
            self.current_upper_trip = trip
            try:
                value = self.upper_original(state, trip)
            finally:
                self.current_upper_trip = None
            if self.anchor is None:
                self.upper_prefix.append([float(env.current_time), int(trip.launch_turn), float(value)])
            elif self.applied_upper is None:
                self.applied_upper = dict(trip=int(trip.launch_turn), query_s=float(env.current_time),
                    target_s=float(value), planned_s=float(getattr(trip, '_freqduet_scheduled_launch', trip.launch_time)),
                    requested_shift_s=UPPER[CANDIDATES[self.candidate][0]])
            return value

        def get_action(state, *args, **kwargs):
            native = self.get_action_original(state, *args, **kwargs)
            if self.anchor is not None or self.current_upper_trip is None or env.current_time < TARGET_S:
                return native
            require(self.policy is not None, 'upper anchor precedes any lower policy initialization')
            trip = self.current_upper_trip
            anchor = dict(time_s=float(env.current_time), trip=int(trip.launch_turn), direction=bool(trip.direction),
                upper_state=np.asarray(state).tolist(), native_action=np.asarray(native).tolist(),
                resources=resource_snapshot(env, self.launches), upper_prefix=copy.deepcopy(self.upper_prefix),
                lower_prefix=copy.deepcopy(self.lower_prefix), event_prefix=[[list(k), v] for k, v in self.uniforms.records.items()])
            if self.reference is not None:
                pilot.joint.assert_same_state(anchor, self.reference['anchor'], 'V211_exact_common_upper_prefix')
            self.anchor = anchor
            if self.fork is not None:
                repeated.fork_future(runner, self.policy, self.fork)
                self.uniforms.seed = self.fork+10000000
            shift = UPPER[CANDIDATES[self.candidate][0]]
            result = upper_candidate(native, runner.upper_action_low, runner.upper_action_high, trip.direction, shift)
            self.anchor_command = dict(native=np.asarray(native).tolist(), candidate=result.tolist())
            return native if shift == 0. else result

        def launch(trip, actual_launch=None):
            ready = {int(b.bus_id) for b in env.bus_in_terminal if b.direction == bool(trip.direction)}
            bus = self.launch_original(trip, actual_launch=actual_launch)
            if bus is not None:
                require(int(bus.bus_id) in ready and len(env.bus_all) == 12, 'launch created or teleported a vehicle')
                original = float(getattr(trip, '_freqduet_base_launch', getattr(trip, '_original_launch', trip.launch_time)))
                self.dispatch_ledger.append(dict(trip=int(trip.launch_turn), bus=int(bus.bus_id),
                    actual_s=float(env.current_time), public_baseline_s=original,
                    eligible_s=float(trip._freqduet_first_eligible_time)))
            return bus

        def drive(bus, *args, **kwargs):
            active = bool(bus.on_route)
            result = self.drive_original(bus, *args, **kwargs)
            if active and not bus.on_route:
                trip = int(bus.trip_id)
                start = self.launches[trip]
                require(trip not in self.return_ledger, 'vehicle trip returned twice')
                self.return_ledger[trip] = time_account(start['time_s'], float(bus.back_to_terminal_time),
                    float(bus.episode_hold_vehicle_seconds)-start['holding_counter_at_launch'])
            return result

        runner._upper_callback_v2, runner.upper_trainer.policy_net.get_action = upper, get_action
        env.launch_bus, Bus.drive = launch, drive

    def __call__(self, policy, states, features, theta):
        self.policy = policy
        self.method = 'bilateral025' if self.anchor is None else self.continuation
        law = super().__call__(policy, states, features, theta)
        bus = self.runner._bus_for_agent(self.runner._pilot_agent)
        context = self.runner._pending_lower_action_context[int(self.runner._pilot_agent)]
        identity = [int(bus.trip_id), int(bus.last_station.station_id), bool(bus.direction), int(bus.bus_id)]
        base = int(law.argmax(dim=1).item())
        if self.anchor is None:
            self.lower_prefix.append([float(self.runner.env.current_time), *identity, base])
            return law
        if self.continuation != 'bilateral025' or len(self.lower_changes) >= 2:
            return law
        feasible = policy.feasible_action_mask(states)[0].numpy()
        eligible = (bool(bus.direction) == self.anchor['direction'] and context.evidence_valid
            and context.follower_evidence_valid and not self.runner._lower_terminal_action_masked(bus) and feasible.all())
        if not eligible:
            return law
        if self.first_lower is not None and not (identity[1:3] == self.first_lower[1:3] and identity[3] != self.first_lower[3]):
            return law
        index = len(self.lower_changes)
        if index == 0:
            self.first_lower = identity
        offset = LOWER[CANDIDATES[self.candidate][1]][index]
        chosen = lower_candidate(base, offset, feasible)
        self.lower_changes.append(dict(time_s=float(self.runner.env.current_time), identity=identity,
            base_bin=base, selected_bin=chosen, offset=offset, actual_command_s=float(pilot.step.ACTIONS[chosen])))
        changed = torch.zeros_like(law)
        changed[0, chosen] = 1.
        return changed

    def detach(self):
        self.runner._upper_callback_v2 = self.upper_original
        self.runner.upper_trainer.policy_net.get_action = self.get_action_original
        self.env.launch_bus, self.bus_class.drive = self.launch_original, self.drive_original
        super().detach()

    def finish(self, runner, records, holds):
        require(self.anchor is not None and self.applied_upper is not None, 'registered upper anchor missing')
        require(self.continuation != 'bilateral025' or len(self.lower_changes) == 2, 'two causal related lower decisions missing')
        require(len(runner.env.bus_all) == 12 and len(self.return_ledger) == len(self.launches)
            and len(self.dispatch_ledger) == len(self.launches), 'resource ledger does not cover complete service')
        held = sum(r['holding_s'] for r in self.return_ledger.values())
        commands = sum(r['action_s'] for r in holds)
        np.testing.assert_allclose(held, commands, atol=1e-8, rtol=0)
        scalar = (float(runner.lower_trainer.alpha), float(runner.lower_trainer.lambda_param), float(runner.upper_trainer.alpha))
        require(scalar == self.frozen_scalars, 'frozen temperature or dual changed')
        self.ledger = dict(physical_vehicles=12, completed_trips=len(self.return_ledger),
            executed_holding_s=held, commanded_holding_s=commands, accounting_error_s=abs(held-commands),
            inclusive_launch_delay_s=float(np.mean([max(0., r['actual_s']-r['public_baseline_s']) for r in self.dispatch_ledger])),
            positive_control_slack_issued_s=0., guarantee_enroute_slack=False)
        super().finish(runner, records, holds)


def replay(root, ci, candidate=0, *, reference=None, future=None, continuation='bilateral025', label):
    seed, day = CONTEXTS[ci]
    sampler = 187000000+seed+100*day
    uniforms, credit = pilot.event.EventUniforms(sampler), pilot.CreditObserver()
    rule = JointAudit(seed, candidate, reference=reference, fork=future, continuation=continuation, uniforms=uniforms)
    row, events, _, _ = pilot.rollout(root, seed, day, 'stochastic', sampler, [0.]*4, label,
        probability_rule=rule, trace_observer=rule, event_uniforms=uniforms, credit_observer=credit)
    row.update(inclusive_launch_delay_s=rule.ledger['inclusive_launch_delay_s'])
    result = dict(context=ci, seed=seed, day=day, candidate=candidate, continuation=continuation,
        future_seed=future, results={k: row[k] for k in (*METRICS, 'action_counts')}, anchor=rule.anchor,
        upper=rule.applied_upper, anchor_command=rule.anchor_command, lower=rule.lower_changes,
        ledger=rule.ledger, authority=rule.authority, physical_contract=rule.contract)
    return result, rule, events


def registration():
    return dict(stage='v211_same_authority_joint_resource_opportunity', source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
        contexts=[list(c) for c in CONTEXTS], target_s=TARGET_S,
        upper_shifts=list(UPPER), lower_bin_offsets=[list(a) for a in LOWER], candidates=[list(a) for a in CANDIDATES],
        selection_futures=list(SELECT_FUTURES), evaluation_futures=list(EVAL_FUTURES),
        future_namespace='2110000000+100*context+future;smoke2119000000,separate from all formal labels',
        reference='V106 bilateral025 with frozen B200 upper/deployment. Four consumed actor/day prefixes,not fresh days.',
        resource_contract='Original12 fixed-pool vehicles,no creation,teleport,layover reduction or service deletion. '
            'Observable ready IDs/directions,published plans and executed holding tracked. Enroute guaranteed slack unknown; '
            'issue zero new credit. Actual holding+other elapsed=return-launch;match total actual holding to commands.',
        authority='One original fresh upper decision after7200s:perturb current-direction native antisymmetric coefficient '
            'by0/-30/+30 within original[-60,60]. Keep native EMA,exact headway curve,900s zero-sum projection and all '
            'execution gates. No direct launch override,hard spacing gate,new terminal wait or persistent policy change.',
        lower='First eligible bilateral seven-bin stop decision in anchor direction,then next same-stop/direction '
            'different vehicle,identified causally after interventions. Offset0/+-one feasible bin,then resume bilateral025. '
            'Keep original0/5/10/15/20/30/45 bins. Branch-dependent later identities are recorded,not forced to match.',
        prefix='Pre-anchor upper state,public resource ledger and lower/upper/event prefixes exactly match native reference. '
            'Fork future innovations ONLY after matching anchor;current sampled window and latent day profile stay fixed.',
        select='Generate all15 candidates on16 selection futures. Select joint over all15,upper-only over3/lower-only over5. '
            'Mean passenger minimum subject to ALL five constraint mean changes<=0 relative to candidate0. Native is feasible; '
            'ties keep candidate0 then lower candidate ID. Seal choices before any evaluation group is allowed.',
        evaluate='16 DISJOINT futures for baseline and three frozen proposals,deduplicating identical candidates. '
            'Add B200-native/preserve suffix controls on SAME bilateral prefix. Never select a law on evaluation labels.',
        metrics=list(METRICS), constraints=list(CONSTRAINTS),
        gate='Passenger crossed actor/day and fixed-prefix conditional-future95% CI upper<0;ALL constraint upper<=0 '
            'under BOTH intervals. Require nonharmful means in BOTH predetermined8-future evaluation halves. '
            'Primary comparisons:joint minus bilateral and joint minus selection-best single;keep all negative outcomes. '
            'Native/preserve suffix comparisons diagnostic,not whole-day policy comparisons.',
        budget='Four references,960 selection continuations,at most384 evaluation continuations plus smoke. '
            'No adaptive future/day extension. Futures are not independent demand days.',
        limitations='Conditional simulation opportunity benchmark,not an observable-state online planner:future generator '
            'inherits simulator hidden current-window/day state. Independent innovations limit selection optimism,not '
            'state-oracle bias. Does not establish a full-day policy,FreqDuet separation contribution or superiority to fixed. '
            'Parent TransitDuet dynamic-fleet and served-passenger metrics are not imported.',
        actor_updates=0, critic_updates=0, dual_updates=0, interval_fit=False, main_policy_changed=False)


def smoke(root, old):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
        '-p', 'test_v211_*.py', '-v'], check=True)
    require(not (root/'protocol.json').exists(), 'do not overwrite sealed V211 protocol')
    pilot.write_json(root/'protocol.json', registration())
    seed, day = CONTEXTS[0]
    cached = read(old/'pairs'/f"p{prior.ITEMS.index((seed,day,'bilateral025',0))}.json")
    ref, ref_rule, ref_events = replay(root, 0, label='smoke_reference')
    for key in (*prior.METRICS, 'action_counts'):
        pilot.joint.assert_same_state(ref['results'][key], cached['results'][key], 'V211_cached_bilateral_'+key)
    noop, noop_rule, noop_events = replay(root, 0, reference=ref, label='smoke_noop')
    pilot.joint.assert_same_state(ref['results'], noop['results'], 'V211_noop_endpoints')
    pilot.joint.assert_same_state(ref_events, noop_events, 'V211_noop_full_actions')
    for name in ('arrivals', 'departures'):
        pilot.joint.assert_same_state(getattr(ref_rule, name), getattr(noop_rule, name), 'V211_noop_full_timing')
    base, _, base_events = replay(root, 0, reference=ref, future=2119000000, label='smoke_paired_base')
    treated, _, events = replay(root, 0, 14, reference=ref, future=2119000000, label='smoke_joint')
    repeat, _, repeated_events = replay(root, 0, 14, reference=ref, future=2119000000, label='smoke_joint_repeat')
    pilot.event.alignment(base_events, events)
    pilot.joint.assert_same_state(treated, repeat, 'V211_same_future_repeat')
    pilot.joint.assert_same_state(events, repeated_events, 'V211_same_future_full_actions')
    require(treated['anchor_command']['candidate'] != treated['anchor_command']['native']
        and any(r['base_bin'] != r['selected_bin'] for r in treated['lower']), 'smoke did not exercise both levels')
    pilot.write_json(root/'references/c0.json', ref)
    pilot.write_json(root/'smoke/done.json', dict(status='v211_tests_cached_bilateral_noop_and_two_level_execution_passed',
        cached_endpoint_exact=True, noop_actions_and_timing_exact=True, paired_future_repeat_exact=True,
        resources_conserved=True, smoke_rollouts=5, actor_updates=0, main_policy_changed=False))
    print('V211_SMOKE_PASSED', flush=True)


def references(root):
    require((root/'smoke/done.json').exists(), 'references require smoke')
    for ci in range(len(CONTEXTS)):
        path = root/'references'/f'c{ci}.json'
        if not path.exists():
            row, _, _ = replay(root, ci, label=f'reference{ci}')
            pilot.write_json(path, row)
    pilot.write_json(root/'reference_done.json', dict(status='four_common_prefixes_complete', contexts=4))


def collect_group(root, phase, index):
    require((root/'smoke/done.json').exists() and (root/'reference_done.json').exists(), 'groups require smoke and references')
    ci, fi = divmod(index, 16)
    future = (SELECT_FUTURES if phase == 'select' else EVAL_FUTURES)[fi]
    path = root/phase/f'c{ci}_f{future}.json'
    if path.exists():
        return
    ref = read(root/'references'/f'c{ci}.json')
    if phase == 'select':
        candidates, controls = list(range(15)), []
    else:
        choices = read(root/'selection/choices.json')['contexts'][ci]
        candidates = sorted(set([0, choices['joint'], choices['upper_only'], choices['lower_only']]))
        controls = ['native', 'preserve']
    rows, common = [], None
    for candidate, continuation in [(c, 'bilateral025') for c in candidates]+[(0, c) for c in controls]:
        row, _, events = replay(root, ci, candidate, reference=ref, future=future_seed(ci, future),
            continuation=continuation, label=f'{phase}_c{ci}_f{future}_a{candidate}_{continuation}')
        if common is not None:
            pilot.event.alignment(common, events)
        else:
            common = events
        rows.append({k: v for k, v in row.items() if k != 'anchor'})
    pilot.write_json(path, dict(context=ci, future=future, future_seed=future_seed(ci, future), rows=rows, exact_prefix=True))
    print('V211_GROUP_DONE', phase, ci, future, len(rows), flush=True)


def select_candidate(values, pool):
    base = values[0]
    feasible = [i for i in pool if all(values[i, METRICS.index(k)] <= base[METRICS.index(k)] for k in CONSTRAINTS)]
    return min(feasible, key=lambda i: (values[i, 0], i))


def freeze_choices(root):
    require(not (root/'selection/choices.json').exists(), 'do not refit choices')
    require(not (root/'evaluate').exists(), 'evaluation must not precede frozen choices')
    rows = []
    for ci in range(4):
        data = []
        for future in SELECT_FUTURES:
            group = read(root/'select'/f'c{ci}_f{future}.json')
            require(group['context'] == ci and group['future'] == future and len(group['rows']) == 15, 'selection grid differs')
            require([r['candidate'] for r in group['rows']] == list(range(15)), 'candidate order differs')
            data.append([[r['results'][k] for k in METRICS] for r in group['rows']])
        means = np.mean(data, axis=0)
        pools = dict(joint=list(range(15)), upper_only=[i for i, (_, l) in enumerate(CANDIDATES) if l == 0],
            lower_only=[i for i, (u, _) in enumerate(CANDIDATES) if u == 0])
        choices = {name: select_candidate(means, pool) for name, pool in pools.items()}
        single = min((choices['upper_only'], choices['lower_only']), key=lambda i: (means[i, 0], i))
        rows.append(dict(context=ci, **choices, best_single=single,
            predicted={name: dict(zip(METRICS, (means[index]-means[0]).tolist())) for name, index in choices.items()},
            candidate_means=means.tolist()))
    pilot.write_json(root/'selection/choices.json', dict(status='choices_frozen_before_evaluation', contexts=rows,
        selection_futures=list(SELECT_FUTURES), evaluation_futures=list(EVAL_FUTURES), updates=0))
    print('V211_CHOICES_FROZEN_NO_EVALUATION_READ', flush=True)


def conditional_intervals(effects):
    rng = np.random.default_rng(211)
    counts = rng.multinomial(16, np.full(16, 1./16), size=(5000, 4))
    estimates = np.einsum('bcf,cfmk->bmk', counts, effects, optimize=True)/64.
    return np.percentile(estimates, [2.5, 97.5], axis=0)


def joint_gate(stats):
    return stats['passenger_person_s']['ci95'][1] < 0 and all(stats[k]['ci95'][1] <= 0 for k in CONSTRAINTS)


def aggregate(root):
    choices = read(root/'selection/choices.json')['contexts']
    methods = (*METHODS, 'best_single')
    physical, rollouts = [], 0
    for ci in range(4):
        cell = []
        for future in EVAL_FUTURES:
            group = read(root/'evaluate'/f'c{ci}_f{future}.json')
            require((group['context'], group['future'], group['future_seed']) == (ci, future, future_seed(ci, future)), 'evaluation grid differs')
            lookup = {(r['candidate'], r['continuation']): r for r in group['rows']}
            ids = dict(bilateral=(0, 'bilateral025'), native_suffix=(0, 'native'), preserve_suffix=(0, 'preserve'))
            ids.update({m: (choices[ci][m], 'bilateral025') for m in ('upper_only', 'lower_only', 'joint', 'best_single')})
            require(set(lookup) == set(ids.values()), 'missing/extra evaluation arms')
            cell.append([[lookup[ids[m]]['results'][k] for k in METRICS] for m in methods])
            rollouts += len(group['rows'])
        physical.append(cell)
    physical = np.asarray(physical)
    contrasts = [(m+'_vs_bilateral', m, 'bilateral') for m in methods if m != 'bilateral']
    contrasts += [('joint_vs_best_single', 'joint', 'best_single'), ('joint_vs_native_suffix', 'joint', 'native_suffix'),
        ('joint_vs_preserve_suffix', 'joint', 'preserve_suffix')]
    effects = np.stack([physical[:, :, methods.index(a)]-physical[:, :, methods.index(b)] for _, a, b in contrasts], axis=2)
    crossed_rows = [dict(seed=SEEDS[ci//2], day=DAYS[ci%2], effects={name: dict(zip(METRICS, effects[ci, :, j].mean(0).tolist()))
        for j, (name, _, _) in enumerate(contrasts)}) for ci in range(4)]
    crossed = confirmation.describe(crossed_rows, 'effects')
    bounds = conditional_intervals(effects)
    conditional = {name: {k: dict(mean=float(effects[:, :, j, q].mean()), ci95=bounds[:, j, q].tolist())
        for q, k in enumerate(METRICS)} for j, (name, _, _) in enumerate(contrasts)}
    halves = {name: [{k: float(effects[:, h*8:(h+1)*8, j, q].mean()) for q, k in enumerate(METRICS)} for h in range(2)]
        for j, (name, _, _) in enumerate(contrasts)}
    gates = {name: dict(crossed=joint_gate(crossed[name]), conditional_future=joint_gate(conditional[name]),
        both_evaluation_halves_nonharm=all(h['passenger_person_s'] < 0 and all(h[k] <= 0 for k in CONSTRAINTS) for h in halves[name]))
        for name in ('joint_vs_bilateral', 'joint_vs_best_single')}
    pilot.write_json(root/'summary/status.json', dict(status='v211_joint_resources_pilot_complete',
        protocol=read(root/'protocol.json'), effects=crossed, conditional_future=conditional, evaluation_halves=halves,
        gates=gates, joint_gate_passed=all(all(g.values()) for g in gates.values()),
        actor_day=crossed_rows, frozen_choices=choices, selection_continuations=960, evaluation_continuations=rollouts,
        references=4, actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        scope='Four consumed prefixes/independent future banks,conditional opportunity only;not an online controller or a whole-day baseline comparison.'))
    print('V211_COMPLETE', gates, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'references', 'select', 'freeze', 'evaluate', 'aggregate'))
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--v106-root', type=Path)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=64)
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    if a.command == 'smoke':
        smoke(a.out_root, a.v106_root)
    elif a.command == 'references':
        references(a.out_root)
    elif a.command == 'freeze':
        freeze_choices(a.out_root)
    elif a.command == 'aggregate':
        aggregate(a.out_root)
    else:
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=get_context('spawn')) as pool:
            jobs = [pool.submit(collect_group, a.out_root, a.command, i) for i in range(a.start, a.end)]
            for i, job in enumerate(jobs, 1):
                job.result()
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=i, total=len(jobs), unit='paired_future_group')), flush=True)
        pilot.write_json(a.out_root/'shards'/f'{a.command}_{a.start}_{a.end}.json', dict(status='complete'))


if __name__ == '__main__':
    main()
