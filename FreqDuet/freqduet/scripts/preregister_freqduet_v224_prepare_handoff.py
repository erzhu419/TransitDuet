#!/usr/bin/env python3
"""Seal V224's passive-recording contract. No collector or rollout entrypoint."""

import argparse
import json
from pathlib import Path
import unittest


CONTEXTS = ((12037, 2151001), (12037, 2151002), (933083, 2151001), (933083, 2151002))
ARMS = ('v107', 'upper')
ARCHIVE = 'protocol_v6_v221_release_smoke_2b0115af7b'


def registration(source):
    slots = [dict(slot=f'c{ci}_{arm}', context=ci, actor=actor, demand_day=day,
        arm=arm, sampler_seed=187000000+actor+100*day,
        archived_reference=f'{ARCHIVE}/records/c{ci}_{arm}.json.gz')
        for ci, (actor, day) in enumerate(CONTEXTS) for arm in ARMS]
    return dict(stage='v224_prepare_identity_phase_handoff_preregistration', source=source,
        current_budget=dict(archive_scans=0, replay_paths=0, training_updates=0, new_futures=0),
        reserved_preflight=dict(max_started_paths=8, slots=slots,
            first_pair=['c0_v107', 'c0_upper'], subsequent_slots=[s['slot'] for s in slots[2:]],
            extra_reference_paths=0, max_workers=2, launch_authorized=False,
            failure_rule='Started attempts consume slots,including failed/incomplete paths. '
                'No auto retry,context replacement or extra reference replay. First pair must '
                'pass all equality and recording gates before the other six start. '
                'Reuse complete saved outputs on analysis recovery;extra attempts need approval.'),
        frozen=dict(controller_source='2b0115af7b',
            dependencies=['protocol_v6_v218_cycle_b6ab0892ff/source_b09088f636',
                          'protocol_v6_v211_joint_resources_c45b2d0a93/source'],
            runtime='FreqDuet-v6-engineering-v9-runtime-optimized/FreqDuet/freqduet',
            shadow_law_source='7939ddaab6',
            authority='Existing V221 v107/upper arms only;V220 supply projection unchanged. '
                'Original12 vehicles,seven bins,all native bounds/caches/EMA/cadence unchanged. '
                'No lower override or V223 action is executed. No actor/critic/dual/filter/'
                'V151/V146/V167 predictor/calibration update. Do not import mutable live runtime.'),
        prepare_fields={
            'clock': ['prepare_s', 'record_read_s', 'pre_action_time_s', 'handoff_order',
                      'legacy_prepare_event_seq', 'prepare_occurrence'],
            'current': ['trip_id', 'bus_id', 'direction', 'station_id'],
            'native': ['forward_gap_s', 'forward_valid', 'forward_source',
                       'follower_gap_s', 'follower_valid', 'follower_source', 'target_headway_s'],
            'front': ['status', 'trip_id', 'bus_id', 'direction', 'station_id',
                      'departure_s', 'departure_order', 'launch_order'],
            'follower': ['status', 'trip_id', 'bus_id', 'direction', 'avl_sample_s',
                         'avl_sample_order', 'avl_sample_origin', 'launch_order'],
            'role_phase': ['status', 'phase_s', 'launch_order', 'admission_order',
                           'native_plan_s', 'proposed_plan_s', 'applied_plan_s',
                           'native_tick_s', 'coordinated_tick_s', 'actual_launch_s',
                           'reserved_bus_id', 'actual_bus_id', 'plan_at_launch_s'],
            'query_join': ['query_s', 'query_order', 'legacy_query_event_seq',
                           'prepare_order', 'query_occurrence', 'feasible_bins',
                           'native_bin', 'shadow_bin', 'shadow_support_reason']},
        clocks='Copy cached native evidence immediately AFTER the original prepare returns, '
            'before any next vehicle update. prepare_s=record_read_s=pre_action_time_s. '
            'Use an observer-only monotonic total order for native tick/AVL freeze,launch,'
            'departure,prepare and query;retain V221 legacy EventClock and every existing '
            'event_seq unchanged. V223 shadow launch_event_seq/prepare_event_seq use ONLY '
            'this same sidecar order. Never compare or mix the two counters. '
            'At query join the exact prepared occurrence;preserve repeated calls and unused '
            'prepares,never overwrite a dictionary by trip/stop alone.',
        identities='Current identity is copied at prepare. Front is the matched native '
            'completed same-stop/direction departure whose trip equals the cached predecessor. '
            'Native departure rows contain trip/time but no bus;resolve bus from the already '
            'observed launch ledger,not a reused live vehicle. Require departure_order<prepare '
            'and F=prepare_s-departure_s. Follower is EXACT cached pre_action_follower_bus_id/'
            'trip_id and the same native AVL snapshot consumed by its estimator. Copy the '
            'original tick-start snapshot,or original inline fallback output if used;do not '
            'call the estimator or resample AVL at prepare/query. Keep snapshot time and '
            'sample_order<=prepare. No fixed trip-number or future actual-neighbour substitute.',
        phases='Record immutable same-admission native/proposed/APPLIED plans and executable '
            'eligibility ticks using the frozen native tick rule. At launch,latch against '
            'actual trip/bus/direction and plan_at_launch. Nonzero phase requires the original '
            'reserved bus,unchanged active plan provenance,and actual launch at coordinated '
            'tick. Cancelled/overwritten/undelivered or unknown provenance is unsupported. '
            'Unmodified launched trips have explicit0 even when late;V107 shadow proposals '
            'are NEVER applied offsets. phase=coordinated_tick-native_tick,not public lateness '
            'or continuous/raw request. At prepare copy all three role phase rows from the '
            'strict launch prefix,launch_order<prepare. No end-of-day backfill.',
        missing='Every listed field exists. Explicit statuses distinguish supported,known_zero,'
            'no_native_front,no_native_follower,missing_launch,undelivered_plan and '
            'identity_unresolved. Inapplicable values are null with their reason,not0. '
            'not_recorded is an engineering failure,not abstention. Missing role phase keeps '
            'the frozen native command. Later labels and late onboard state never fill a packet.',
        equality='Serialize legacy data with EXACTLY the archived key set,all recursively '
            'equal. Includes physical contract,resource ledger,all actions/probabilities/'
            'policy inputs/uniform identities,upper commands/caches/callbacks/plans/tasks/'
            'responses,native event sequence,arrivals/departures/launches/returns,issued/'
            'executed holding and EVERY endpoint field. No tolerance,rounding,resorting,'
            'ignored original fields or wall-time equality. Store new handoff/shadow data '
            'in a SEPARATE sidecar;never append fields into legacy rows.',
        recording_gate='Every executed lower query has exactly one prepare occurrence with '
            'matching identity and cached gaps/validity/source/target,correct clocks and '
            'field presence. Keep all no-support and unused prepares. Validate strict-prefix '
            'role launches and front departures,explicit-zero versus unavailable phases,'
            'snapshot provenance and unchanged RNG states BEFORE/AFTER each observer-only '
            'body. Observer-side native call counters must have zero increments;do not '
            'require unarchived historical RNG states or estimator counters. No estimator,policy,planner,'
            'environment step or random sampler gets an extra call for recording.',
        decision='Report each context/arm separately: legacy equality,join coverage,role/'
            'phase support and abstention reasons,native/shadow bins and positive/negative/'
            'zero continuous changes. All equality AND recording gates must pass. Shadow '
            'bin differences are OFFLINE requests,not executed authority or efficacy. '
            'Zero differences stops intervention progression without gain escalation or '
            'forced bin cuts. Any nonzero differences only motivate a separately budgeted '
            'actual-action smoke;independent efficacy days/futures remain closed.',
        execution='scheduler high,manual dispatch,node001--006. This seal job uses synthetic '
            'contract tests only;no archived trajectories are opened. During any later '
            'authorized passive run,raw legacy/sidecars stay remote;return compact per-path '
            'checks and counts only,not CSV trajectories or checkpoints.',
        collector_implemented=False, main_policy_changed=False)


def require_legacy_equal(reference, recorded):
    """Compare serialized legacy payloads; new observer data lives elsewhere."""
    if reference.keys() != recorded.keys():
        raise AssertionError('V224 legacy key set changed')
    for key in reference:
        if reference[key] != recorded[key]:
            raise AssertionError('V224 legacy mismatch: '+key)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    suite = unittest.defaultTestLoader.discover(
        str(Path(__file__).resolve().parents[1]/'tests'), pattern='test_v224_prepare_handoff.py')
    assert suite.countTestCases() > 0, 'V224 contract tests were not staged'
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    spec = registration(a.source)
    a.out_root.mkdir(parents=True, exist_ok=True)
    path = a.out_root/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(spec)), 'V224 protocol is frozen'
    else:
        path.write_text(json.dumps(spec, indent=2)+'\n')
    summary = dict(status='v224_protocol_sealed_only', source=a.source, tests=result.testsRun,
        reserved_passive_paths=8, started_paths=0, archive_scans=0, training_updates=0,
        new_futures=0, launch_authorized=False, collector_implemented=False, main_policy_changed=False)
    (a.out_root/'summary').mkdir(exist_ok=True)
    (a.out_root/'summary/status.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('V224_PROTOCOL_SEALED', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
