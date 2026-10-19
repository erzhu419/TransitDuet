#!/usr/bin/env python3
"""Frozen no-terminal handoff confirmation on new days and archived actors."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v134_handoff_pilot as frozen
import run_freqduet_v94_compensation_confirmation as inference
import run_freqduet_v96_frozen_compensation as axes
from run_freqduet_protocol_v6_checkpoint_trajectory import TRAIN_SEEDS

pilot, original = frozen.pilot, frozen.original
DEVELOPMENT_SEEDS = (12011, 933013)
VALIDATION_SEEDS = tuple(s for s in TRAIN_SEEDS if s not in DEVELOPMENT_SEEDS)
DAYS = tuple(range(2351001, 2351017))
ITEMS = tuple(itertools.product(TRAIN_SEEDS, DAYS))
MODES = ('old', 'handoff')
METRICS = frozen.METRICS
PRIMARY = ('passenger_person_s', 'cv_squared', 'inclusive_terminal_delay_s')
COMPLETION = ('trips_unlaunched', 'trips_incomplete')
CONTRAST = 'handoff_vs_old'
SHARDS = 6


class ConfirmationAudit(original.PlanAudit):
    def __init__(self, seed, geometry, mode):
        super().__init__(seed, geometry, frozen.prior.bilateral.PRIMARY, False)
        if mode == 'handoff':
            self.stream = frozen.ActiveLaunchETA(geometry)


def bounds(index):
    return len(ITEMS)*index//SHARDS, len(ITEMS)*(index+1)//SHARDS


def run_path(root, runtime, seed, day, mode, *, artifact_root):
    name = f'{seed}_{day}_{mode}'
    receipt = artifact_root/'paths'/(name+'.json')
    if receipt.exists():
        return json.loads(receipt.read_text())
    rule = ConfirmationAudit(seed, frozen.prior.model.load_geometry(runtime), mode)
    credit = pilot.CreditObserver()
    result, _, _, _ = pilot.rollout(root, seed, day, 'stochastic', 187000000+seed+100*day,
        [0.]*4, 'v135_'+name, probability_rule=rule, trace_observer=rule, credit_observer=credit)
    result.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    if not result['base_actors_unchanged'] or result['training_updates'] or result['dual_updates']:
        raise AssertionError('confirmation updated the frozen deployment')
    values = {k: result[k] for k in original.frozen.prior.METRICS}
    values.update({k: rule.terminal[k] for k in ('inclusive_terminal_delay_s', 'qualification_shift_s')})
    rule.save(artifact_root/'records'/(name+'.npz'))
    selected = [dict(r, evidence=rule.evidence[frozen.prior.prior.key(r)],
        online=rule.online[frozen.prior.prior.key(r)]) for r in rule.decisions
        if rule.evidence[frozen.prior.prior.key(r)]['follower_source'] in frozen.prior.SOURCES]
    frozen.prior.prior.identity.write_trace(artifact_root/'launch_traces'/(name+'.jsonl.gz'), selected)
    pilot.write_json(artifact_root/'terminal'/(name+'.json'), dict(plans=rule.plans, dispatches=rule.dispatches,
        returns=list(rule.returns.values()), reference_clocks=rule.reference_clocks))
    row = dict(seed=seed, day=day, mode=mode, values=values, contract=rule.contract,
        decisions=len(rule.decisions), launch_handoffs=sum(r['status'] == 'launch_handoff_component'
            for r in rule.online.values()), sources=dict(Counter(r['status'] for r in rule.online.values())),
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False)
    pilot.write_json(receipt, row)
    return row


def collect_pair(root, runtime, seed, day, index, *, artifact_root):
    receipt = artifact_root/'pairs'/f'p{index}.json'
    if receipt.exists():
        return json.loads(receipt.read_text())
    paths = {mode: run_path(root, runtime, seed, day, mode, artifact_root=artifact_root) for mode in MODES}
    pilot.joint.assert_same_state(paths['old']['contract'], paths['handoff']['contract'], 'V135_same_physics')
    results = {mode: paths[mode]['values'] for mode in MODES}
    row = dict(index=index, seed=seed, day=day, sampler_seed=187000000+seed+100*day,
        results=results, effects={CONTRAST: {k: results['handoff'][k]-results['old'][k] for k in METRICS}},
        use={mode: {k: paths[mode][k] for k in ('decisions', 'launch_handoffs', 'sources')} for mode in MODES},
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False)
    pilot.write_json(receipt, row)
    print('V135_PAIRED_CELL_DONE', index, seed, day, flush=True)
    return row


def assert_arrays(before_path, after_path):
    with np.load(before_path) as before, np.load(after_path) as after:
        if before.files != after.files:
            raise AssertionError('confirmation collector changed timing-record fields')
        for field in before.files:
            np.testing.assert_array_equal(before[field], after[field])


def smoke_pair(root, runtime, v133, v134, seed, index):
    day = 2231004
    current = collect_pair(root, runtime, seed, day, index, artifact_root=root/'smoke_runs')
    old_index = 0 if seed == 12011 else 2
    old = json.loads((v133/'pairs'/f'p{old_index}.json').read_text())['values']
    candidate = json.loads((v134/'pairs'/f'p{old_index}.json').read_text())['values']
    pilot.joint.assert_same_state(old, {k: current['results']['old'][k] for k in old}, 'V135_exact_old_collector')
    pilot.joint.assert_same_state(candidate, current['results']['handoff'], 'V135_exact_candidate_collector')
    name = f'{seed}_{day}_split_neutral.npz'
    for mode, reference in (('old', v133), ('handoff', v134)):
        assert_arrays(reference/'records'/name, root/'smoke_runs/records'/f'{seed}_{day}_{mode}.npz')
    print('V135_BOTH_CONSUMED_PATHS_EXACT', seed, flush=True)


def validate_grid(rows):
    if len(rows) != len(ITEMS):
        raise AssertionError('confirmation has missing actor/day pairs')
    if [(r['index'], r['seed'], r['day']) for r in rows] != [(i, *v) for i, v in enumerate(ITEMS)]:
        raise AssertionError('confirmation grid contains duplicate or mismatched pairs')


def joint_gate(stats, strict=True):
    passenger = stats['passenger_person_s']['ci95'][1] < 0
    other = all(stats[k]['ci95'][1] < 0 if strict else stats[k]['ci95'][1] <= 0 for k in PRIMARY[1:])
    return passenger and other and all(stats[k]['ci95'][1] <= 0 for k in COMPLETION)


def summarize(rows):
    effects = inference.describe(rows, 'effects')
    stats = effects[CONTRAST]
    return dict(effects=effects, strict_joint_improvement=joint_gate(stats),
        passenger_gain_with_joint_nonharm=joint_gate(stats, strict=False),
        axes=axes.axis_summary(rows, 'effects', PRIMARY),
        absolute={mode: {k: float(np.mean([r['results'][mode][k] for r in rows])) for k in METRICS}
            for mode in MODES},
        cell_joint_improvements=sum(frozen.joint_improvement(r['effects'][CONTRAST]) for r in rows),
        cells=len(rows), cells_without_handoff=sum(r['use']['handoff']['launch_handoffs'] == 0 for r in rows),
        cell_signs={k: dict(negative=sum(r['effects'][CONTRAST][k] < 0 for r in rows),
            zero=sum(r['effects'][CONTRAST][k] == 0 for r in rows),
            positive=sum(r['effects'][CONTRAST][k] > 0 for r in rows)) for k in PRIMARY})


def aggregate(args):
    rows = [json.loads((args.out_root/'pairs'/f'p{i}.json').read_text()) for i in range(len(ITEMS))]
    validate_grid(rows)
    cohorts = {name: summarize([r for r in rows if r['seed'] in seeds]) for name, seeds in (
        ('heldout_actors', VALIDATION_SEEDS), ('development_actor_new_days', DEVELOPMENT_SEEDS),
        ('all_actors_secondary', TRAIN_SEEDS))}
    worst = sorted((r for r in rows if r['seed'] in VALIDATION_SEEDS),
        key=lambda r: r['effects'][CONTRAST]['journey_min'], reverse=True)[:6]
    pilot.write_json(args.out_root/'summary/status.json', dict(status='v135_frozen_handoff_confirmation_complete',
        cohorts=cohorts, primary_cohort='heldout_actors', primary_contrast=CONTRAST,
        primary_joint_improvement=cohorts['heldout_actors']['strict_joint_improvement'],
        primary_passenger_gain_with_joint_nonharm=cohorts['heldout_actors']['passenger_gain_with_joint_nonharm'],
        worst_journey_cells=[dict(seed=r['seed'], day=r['day'], delta=r['effects'][CONTRAST]) for r in worst],
        archived_actors=list(TRAIN_SEEDS), validation_actors=list(VALIDATION_SEEDS),
        development_actors=list(DEVELOPMENT_SEEDS), demand_days=list(DAYS),
        paired_cells=len(ITEMS), evaluation_rollouts=2*len(ITEMS), terminal_coordination=False,
        frozen_candidate_source='366cf193a8', checkpoint_ep=199,
        bootstrap='existing 5000 crossed actor/day draws; one pair per actor/day',
        sampler_repeats=1, actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        limitations='Archived actors held out from V133/V134 candidate development, not unseen training '
            'runs from the entire project. New simulator demand seeds, not field service dates. '
            'Primary cohort excludes both development actors. Identical actor/day sampler seeds and '
            'frozen weights in both arms; no forecast/gain/terminal-rule fit on confirmation outcomes. '
            'Strict joint gate requires all three primary upper CI bounds below zero; nonharm is '
            'separate. No missing pair or arm is zero-filled, and this is not fixed-baseline superiority.'))
    print('V135_CONFIRMATION_COMPLETE', cohorts['heldout_actors']['effects'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke', 'run', 'aggregate'))
    parser.add_argument('--out-root', type=Path, required=True)
    for name in ('v133-root', 'v134-root', 'runtime-root'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--shard', type=int, choices=range(SHARDS))
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    if args.command == 'smoke':
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v135_*.py', '-v'], check=True)
        with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn')) as pool:
            tasks = [pool.submit(smoke_pair, args.out_root, args.runtime_root, args.v133_root,
                args.v134_root, seed, i) for i, seed in enumerate(DEVELOPMENT_SEEDS)]
            for task in tasks:
                task.result()
        pilot.write_json(args.out_root/'smoke/done.json', dict(status='tests_and_four_consumed_replays_exact',
            new_demand_days_read=False, checkpoint_ep=199, actor_updates=0, dual_updates=0))
    elif args.command == 'run':
        if not (args.out_root/'smoke/done.json').exists():
            raise AssertionError('new-day confirmation requires exact consumed-path smoke')
        start, end = bounds(args.shard)
        with ProcessPoolExecutor(max_workers=min(args.workers, end-start), mp_context=get_context('spawn')) as pool:
            tasks = [pool.submit(collect_pair, args.out_root, args.runtime_root, *ITEMS[i], i,
                artifact_root=args.out_root)
                for i in range(start, end)]
            for task in as_completed(tasks):
                task.result()
        pilot.write_json(args.out_root/'shards'/str(args.shard)/'done.json', dict(status='complete',
            shard=args.shard, start=start, end=end, pairs=end-start, rollouts=2*(end-start)))
    else:
        aggregate(args)
