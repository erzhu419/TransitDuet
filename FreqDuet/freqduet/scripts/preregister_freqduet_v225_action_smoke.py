#!/usr/bin/env python3
"""Seal the V223 actual-action smoke budget; no replay entrypoint."""

import argparse
import json
from pathlib import Path
import unittest

from preregister_freqduet_v224_prepare_handoff import CONTEXTS


REFERENCE_ROOT = 'protocol_v6_v224_prepare_collection'
NO_HARM = ('cv_squared', 'readiness_delay_s', 'public_launch_delay_total_s',
           'trips_unlaunched', 'trips_incomplete')
FIRST_PATH_CHECKS = ('physical_contract_exact', 'prepared_recording_complete',
    'prefix_before_first_change_exact', 'first_change_matches_archived_shadow',
    'live_commands_match_frozen_law', 'changed_commands_actually_executed',
    'upper_first_admission_same_reserved_bus_advance',
    'native_calls_and_random_sampling_preserved', 'all_trips_cleared',
    'endpoint_metrics_complete')


def first_path_passed(checks, executed_changes):
    """Engineering admission only; endpoint improvement is not a filter."""
    return executed_changes > 0 and all(checks[k] is True for k in FIRST_PATH_CHECKS)


def descriptive_joint_passed(effects):
    """Consumed-context comparison, not an independent efficacy test."""
    return all(effects[arm]['passenger_person_s'] < 0 and
               all(effects[arm][metric] <= 0 for metric in NO_HARM)
               for arm in ('v107', 'upper'))


def registration(source):
    slots = [dict(slot=f'c{ci}_joint', context=ci, actor=actor, demand_day=day,
        sampler_seed=187000000+actor+100*day,
        references={arm: f'{REFERENCE_ROOT}/records/c{ci}_{arm}.json.gz'
                    for arm in ('v107', 'upper')},
        passive_sidecar=f'{REFERENCE_ROOT}/sidecars/c{ci}_upper.json.gz')
        for ci, (actor, day) in enumerate(CONTEXTS)]
    return dict(stage='v225_common_reference_actual_action_smoke_preregistration', source=source,
        current_budget=dict(archive_scans=0, replay_paths=0, training_updates=0, new_futures=0),
        reserved_smoke=dict(max_started_paths=4, slots=slots,
            first_slot='c0_joint', subsequent_slots=[s['slot'] for s in slots[1:]],
            extra_reference_paths=0, max_workers=2, launch_authorized=False,
            failure_rule='Every started attempt consumes its slot,including failed/incomplete '
                'paths. No auto retry,context replacement,gain escalation or forced bin cuts. '
                'Complete outputs permit analysis-only recovery,not another replay. '
                'c0 must pass the engineering gate before the remaining three start. '
                'Then run all three regardless of favorable/unfavorable service endpoints.'),
        frozen=dict(controller_source='2b0115af7b', law_source='7939ddaab6',
            passive_collector_source='17043d5c03', passive_acceptance_source='97f3ea3954',
            handoff_protocol_source='282f0d884b',
            dependencies=['protocol_v6_v218_cycle_b6ab0892ff/source_b09088f636',
                          'protocol_v6_v211_joint_resources_c45b2d0a93/source'],
            runtime='FreqDuet-v6-engineering-v9-runtime-optimized/FreqDuet/freqduet',
            gain=.25, action_bins_s=[0., 5., 10., 15., 20., 30., 45.],
            fleet=12, upper='V221 upper arm with unchanged V220 supply projection,raw+30 '
                'request,0.35 EMA,native upper bounds/cadence/caches. Preserve no-later '
                'and ready-backed projection constraints on BOTH directions at each admission.',
            unchanged='Actor,critic,dual,filters,V151,V146,V167,window,quantiles and main. '
                'No training,extra vehicles,invented slack,mutable live runtime or new future.'),
        candidate='Run the original upper arm,NOT the old joint/lower forced-cut branch. '
            'At EVERY actual query,resolve identities and delivered launch phases from that '
            'trajectory\'s V224 prepare packet and strict observed prefix. Recompute V223 '
            'command with the native cached context and CURRENT feasible bins. One request '
            'gain/2*((B-F)+(2*u_self-u_front-u_follower));clip and quantize once using native '
            'float64 bins/shorter tie. Do not add a native request or separate bin cut. '
            'Allow positive as well as negative changes. Do not replay saved shadow actions, '
            'reuse an archived later identity,or require later requests to stay 3/11/3/11.',
        unsupported='Use only supported/known_zero phase rows with matched identities and '
            'strict-prefix launch order. Missing/undelivered phase keeps native action; '
            'forward-only/missing-forward branches remain native. Never impute zero,backfill '
            'future labels or rerun an estimator. Recording omissions fail the smoke.',
        controls='Reuse V224\'s exactly matched V221 v107/upper records and passive sidecars. '
            'The lower-only counterpart has no applied upper phase:V223 is exactly V107 '
            'on every supported zero-phase or native-fallback branch. Reuse V107 instead '
            'of spending four duplicate paths. Original V221 joint/lower forced cuts are '
            'NOT the new joint/lower controls.',
        execution='Preserve native estimator/planner/policy calls and event-keyed random '
            'sampling. Make a one-hot feasible command only when V223 differs from native; '
            'record prepared/native/selected bins,raw request,phase support,issued amount '
            'and actually executed amount. Keep native EventClock and sidecar order separate. '
            'At the first changed command,require exact same query/prepare packet and native '
            'inputs as the matching archived upper path,including its archived shadow bin. '
            'Require exact original actions,inputs,plans,uniforms and timed events BEFORE '
            'that command. Later trajectory differences are intended,not equality failures. '
            'If a path has no changes,require full legacy upper equality instead.',
        execution_accounting='For each changed query,join the same trip,bus,direction,stop '
            'and prepared occurrence to the issued action and completed departure. Require '
            'realized cumulative holding at departure minus executed holding before query '
            'equals the selected bin duration;issued requests alone do not pass. Report '
            'requested versus executed amounts,subsequent holding response and trip totals. '
            'Preserve actual same-reserved-vehicle first-admission upper advance versus V107 '
            'and all original per-admission resource constraints;do not count planned advance '
            'as actual launch authority or require downstream upper state to remain identical.',
        first_path_gate=dict(required_checks=list(FIRST_PATH_CHECKS),
            minimum_executed_changed_commands=1,
            selection='Engineering checks only,no passenger/CV/readiness benefit filter. '
                'Failed or zero-action c0 stops progression without selecting another context.'),
        reporting=dict(comparators=['v107', 'upper'],
            metrics=['passenger_person_s', 'journey_min', 'waiting_person_s', 'onboard_person_s',
                'cv_squared', 'readiness_delay_s', 'readiness_delay_total_s',
                'public_launch_delay_total_s', 'public_early_launch_total_s',
                'public_early_launch_count', 'trips_unlaunched', 'trips_incomplete'],
            no_harm_metrics=list(NO_HARM),
            joint_gate='For EACH of four consumed contexts,both joint-minus-V107 and '
                'joint-minus-upper require passenger_person_s<0 and every no-harm metric<=0. '
                'Also require all engineering checks and completed trips. Show every path, '
                'absolute endpoint and paired delta;missing values remain unassessed. '
                'Do not pool four paths as independent training seeds or claim a CI.'),
        decision='Mechanical failure stops for engineering diagnosis,no expanded budget. '
            'Mechanical pass with service tradeoff permits only read-only mechanism analysis, '
            'not tuning these consumed contexts. A full descriptive joint pass only motivates '
            'a separately preregistered independent-future efficacy study. No promotion or '
            'frequency-separation claim from this smoke;main stays frozen.',
        deployment='scheduler high,manual dispatch,node001--006,max2 workers. Current seal '
            'runs synthetic tests only,opens no archives and launches no simulation. Raw '
            'trajectories stay remote;only compact checks/outcomes return locally.',
        replay_implemented=False, independent_futures_opened=False, main_policy_changed=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    suite = unittest.defaultTestLoader.discover(
        str(Path(__file__).resolve().parents[1]/'tests'), pattern='test_v225_action_smoke.py')
    assert suite.countTestCases() > 0, 'V225 contract tests were not staged'
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    spec = registration(a.source)
    a.out_root.mkdir(parents=True, exist_ok=True)
    path = a.out_root/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == spec, 'V225 protocol is frozen'
    else:
        path.write_text(json.dumps(spec, indent=2)+'\n')
    summary = dict(status='v225_actual_action_smoke_protocol_sealed', source=a.source,
        tests=result.testsRun, reserved_joint_paths=4, extra_reference_paths=0,
        started_paths=0, archive_scans=0, training_updates=0, new_futures=0,
        launch_authorized=False, replay_implemented=False, main_policy_changed=False)
    (a.out_root/'summary').mkdir(exist_ok=True)
    (a.out_root/'summary/status.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('V225_PROTOCOL_SEALED', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
