#!/usr/bin/env python3
"""Replay consumed V121 paths and verify ready-time streaming ETA equivalence."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np

from freqduet_online_component_eta import OnlineComponentETA
import run_freqduet_v121_independent_eta as previous


model, prior, frozen = previous.model, previous.prior, previous.frozen
ITEMS, PREFLIGHT = previous.ITEMS, previous.PREFLIGHT
COMPONENTS = ('running_s', 'service_s', 'holding_s', 'ticks_s', 'target_service_s')


class OnlineAudit(prior.IdentityAudit):
    def __init__(self, seed, geometry):
        super().__init__(seed)
        self.stream = OnlineComponentETA(geometry)
        self.online = {}

    def attach(self, runner):
        super().attach(runner)
        self.inner_launch = self.env.launch_bus
        self.inner_arrival = self.recorder.record
        self.inner_departure = self.recorder.record_departure
        self.inner_prepare = self.bus_type._prepare_for_action

        def launch(trip, actual_launch=None):
            bus = self.inner_launch(trip, actual_launch=actual_launch)
            if bus is not None:
                self.stream.launch(int(trip.launch_turn), bool(trip.direction),
                    self.launches[int(trip.launch_turn)]['time_s'])
            return bus

        def arrival(recorder, station_id, direction, arrival_time_s, trip_id, target_headway_s=None):
            result = self.inner_arrival(recorder, station_id, direction, arrival_time_s, trip_id, target_headway_s)
            row = self.arrivals[-1]
            self.stream.arrival(prior.audit.key(row), row['time_s'], row['realized_holding'])
            return result

        def departure(recorder, station_id, direction, departure_time_s, trip_id):
            result = self.inner_departure(recorder, station_id, direction, departure_time_s, trip_id)
            row = self.departures[-1]
            self.stream.departure(prior.audit.key(row), row['time_s'], row['realized_holding'])
            return result

        def prepare(bus, now, buses, debug):
            result = self.inner_prepare(bus, now, buses, debug)
            key = int(bus.trip_id), int(bus.last_station.station_id), bool(bus.direction)
            self.stream.ready(key, float(now))
            raw = bus.pre_action_follower_departure_gap
            evidence = dict(self.ready[key], follower_gap_prediction_s=None if raw is None else float(raw))
            self.online[key] = self.stream.predict(evidence, key[2], key[1])
            return result

        self.env.launch_bus = launch
        self.recorder.record, self.recorder.record_departure = arrival, departure
        self.bus_type._prepare_for_action = prepare


def compare_prediction(online, reference, evidence):
    if online['prediction_time_s'] != evidence['ready_time_s']:
        raise AssertionError('online prediction was not issued at the original ready snapshot')
    if online['component_used'] != reference['component_used']:
        raise AssertionError('online/offline eligibility differs')
    expected = evidence['follower_gap_prediction_s']
    maximum = 0.
    if reference['component_used']:
        expected = reference['running_s']+(reference['service_s']+reference['holding_s']+reference['ticks_s'])+reference['target_service_s']
        for field in (*COMPONENTS, 'minimum_support', 'historical_fallback', 'latest_completion_s'):
            error = abs(float(online[field])-float(reference[field]))
            maximum = max(maximum, error)
        if online['minimum_support'] != reference['minimum_support'] or online['historical_fallback'] != reference['historical_fallback']:
            raise AssertionError('online/offline completed support differs')
        if online['latest_completion_s'] >= online['prediction_time_s']:
            raise AssertionError('online prediction used a simultaneous or unfinished component')
    gap_error = abs(online['gap_s']-expected)
    if max(maximum, gap_error) > 1e-8:
        raise AssertionError(f'online/offline numeric mismatch: gap={gap_error}, component={maximum}')
    return expected, gap_error, maximum


def run_item(root, cached, runtime, index):
    receipt = root/'pairs'/f'p{index}.json'
    if receipt.exists():
        return
    seed, day = ITEMS[index]
    geometry = model.load_geometry(runtime)
    rule, credit = OnlineAudit(seed, geometry), frozen.pilot.CreditObserver()
    result, _, _, _ = frozen.pilot.rollout(root, seed, day, 'stochastic', 187000000+seed+100*day,
        [0.]*4, f'v115_{seed}_{day}_{frozen.PRIMARY}_0', probability_rule=rule,
        trace_observer=rule, credit_observer=credit)
    result.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    actual = dict(seed=seed, day=day, method=frozen.PRIMARY, rep=0, results=result,
        contract=rule.contract, coverage=rule.coverage, feedforward=rule.feedforward, accounting=rule.accounting)
    # Cached future artifacts are opened only after the live replay has ended.
    before = json.loads((cached/'pairs'/f'p{index}.json').read_text())
    frozen.pilot.joint.assert_same_state(before['frozen_result'], actual, 'V122_all_original_endpoints_and_accounts')
    name = previous.name_for(seed, day)
    saved = prior.read_trace(cached/'traces'/(name+'.jsonl.gz'))
    prior.assert_decisions([{k: v for k, v in row.items() if k != 'evidence'} for row in saved], rule.decisions)
    for row in saved:
        frozen.pilot.joint.assert_same_state(row['evidence'], rule.evidence[prior.audit.key(row)],
            'V122_original_ready_snapshot_evidence')
    record = root/'records'/(name+'.npz')
    rule.save(record)
    with np.load(cached/'records'/(name+'.npz')) as old, np.load(record) as new:
        if old.files != new.files:
            raise AssertionError('streaming telemetry changed timing record fields')
        for field in old.files:
            np.testing.assert_array_equal(old[field], new[field])
    forecasts = [row for row in saved if row['follower_valid']]
    references = model.read_cell(cached, (seed, day), geometry)['forecasts']
    departures = {prior.audit.key(row): row for row in rule.departures}
    if len(forecasts) != len(references):
        raise AssertionError('offline forecast support is incomplete')
    max_gap, max_component, cuts, sources = 0., 0., 0, Counter()
    for row, reference in zip(forecasts, references):
        key, e = prior.audit.key(row), row['evidence']
        live = rule.online[key]
        expected, gap_error, component_error = compare_prediction(live, reference, e)
        max_gap, max_component = max(max_gap, gap_error), max(max_component, component_error)
        sources[live['status']] += 1
        executed = departures[key]['time_s']-e['ready_time_s']
        parts = prior.audit.action_parts(row)
        for cut in (parts['negative_cut_s'], -parts['quota']):
            if cut > 0:
                cuts += 1
                if ((prior.prior.pair_restore_change(e['forward_gap_s'], live['gap_s'], executed, cut) < 0) !=
                        (prior.prior.pair_restore_change(e['forward_gap_s'], expected, executed, cut) < 0)):
                    raise AssertionError('streaming implementation changed a holding-restoration sign')
    prior.write_trace(root/'online'/(name+'.jsonl.gz'), [dict(trip_id=k[0], station_id=k[1], direction=k[2], **v)
        for k, v in rule.online.items()])
    frozen.pilot.write_json(receipt, dict(index=index, seed=seed, day=day,
        all_endpoints_exact=True, original_decisions_exact=True, original_evidence_exact=True,
        full_timing_arrays_exact=True, online_prediction_support_exact=True,
        holding_signs_exact=True, ready_events=len(rule.online), decisions=len(saved),
        forecasts=len(forecasts), labelled_cut_signs=cuts, online_sources=dict(sources),
        max_gap_error_s=max_gap, max_component_error_s=max_component,
        prediction_time='ready_not_next_policy_tick', actor_updates=0, critic_updates=0, dual_updates=0))
    print('V122_ONLINE_EQUIVALENT_PATH_DONE', index, seed, day, flush=True)


def aggregate(root, cached):
    receipts = [json.loads((root/'pairs'/f'p{i}.json').read_text()) for i in range(len(ITEMS))]
    if [(r['seed'], r['day'], r['index']) for r in receipts] != [(*item, i) for i, item in enumerate(ITEMS)]:
        raise AssertionError('V122 requires the complete consumed grid')
    fields = ('all_endpoints_exact', 'original_decisions_exact', 'original_evidence_exact',
        'full_timing_arrays_exact', 'online_prediction_support_exact', 'holding_signs_exact')
    if not all(all(r[k] for k in fields) for r in receipts):
        raise AssertionError('streaming equivalence did not pass on every path')
    parent = json.loads((cached/'summary/status.json').read_text())
    if sum(r['forecasts'] for r in receipts) != parent['label_support']['forecasts']:
        raise AssertionError('V121 forecast denominator changed')
    sources = Counter()
    for row in receipts:
        sources.update(row['online_sources'])
    frozen.pilot.write_json(root/'summary/status.json', dict(status='v122_ready_time_streaming_equivalence_complete',
        receipts=receipts, completed_paths=len(receipts),
        decisions=sum(r['decisions'] for r in receipts), forecasts=sum(r['forecasts'] for r in receipts),
        labelled_cut_signs=sum(r['labelled_cut_signs'] for r in receipts), online_sources=dict(sources),
        max_gap_error_s=max(r['max_gap_error_s'] for r in receipts),
        max_component_error_s=max(r['max_component_error_s'] for r in receipts),
        replayed_consumed_paths=len(receipts), new_demand_days=0,
        altered_action_paths=0, actor_updates=0, critic_updates=0, dual_updates=0,
        quota_changed=False, main_policy_changed=False,
        limitations='Streaming implementation equivalence, not an additional independent effect test. '
            'Only consumed V121 paths replayed. Online state receives launch/arrival/ready/departure '
            'callbacks and current AVL evidence; cached future artifacts open after rollout only. '
            'The unchanged moving-only predictor still falls back while stopped/boundary/unavailable. '
            'No prediction was used for actions; no closed-loop passenger/CV2 benefit was tested.'))
    print('V122_STREAMING_EQUIVALENCE_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--v121-root', type=Path, required=True)
    p.add_argument('--runtime-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    for version in ('v120', 'v122'):
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1]/'tests'), '-p', f'test_{version}_*.py', '-v'], check=True)
    parent = json.loads((a.v121_root/'summary/status.json').read_text())
    if not parent['independent_predictor_criterion_pass']:
        raise AssertionError('V121 must qualify before online implementation replay')
    for indices, workers in ((PREFLIGHT, min(2, a.workers)),
            ([i for i in range(len(ITEMS)) if i not in PREFLIGHT], a.workers)):
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
            list(pool.map(run_item, itertools.repeat(a.out_root), itertools.repeat(a.v121_root),
                itertools.repeat(a.runtime_root), indices))
        print('V122_REPLAY_BATCH_COMPLETE', indices, flush=True)
    aggregate(a.out_root, a.v121_root)
