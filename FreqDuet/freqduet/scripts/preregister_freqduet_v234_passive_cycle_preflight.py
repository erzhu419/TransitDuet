#!/usr/bin/env python3
"""Seal a two-path passive cycle-observer preflight; no collection entrypoint."""

import argparse
import json
from pathlib import Path
import unittest

from preregister_freqduet_v224_prepare_handoff import require_legacy_equal


JOIN_FIELDS = ('legacy_query_event_seq', 'query_order', 'prepare_order', 'query_occurrence')
PAIR_CHECKS = ('legacy_exact', 'v224_sidecar_exact', 'v230_sidecar_exact',
    'all_query_occurrences_present', 'query_identity_clock_order_exact',
    'frozen_v232_views_exact', 'plan_versions_strict_prefix',
    'cycle_transitions_and_holding_closed', 'all_terminal_statuses_present',
    'association_and_unmatched_records_complete', 'observed_links_not_backfilled',
    'observer_rng_and_native_calls_unchanged', 'shadow_never_executed')


def pair_passed(rows):
    return (len(rows) == 2 and {r['slot'] for r in rows} == {'c0_v107', 'c0_upper'}
            and all(r.get(k) is True for r in rows for k in PAIR_CHECKS))


def check_originals(reference, recorded):
    """All three complete payloads, not selected endpoints or action columns."""
    assert reference.keys() == recorded.keys() == {'legacy', 'v224', 'v230'}
    for name in reference:
        require_legacy_equal(reference[name], recorded[name])


def check_queries(rows, resource_queries, old_queries, frozen_views):
    """Contract oracle only; the later collector must not call it as a predictor."""
    assert len(rows) == len(resource_queries) == len(old_queries) == len(frozen_views), \
        'missing or duplicate query occurrence'
    for row, resource, old, view in zip(rows, resource_queries, old_queries, frozen_views):
        assert row.keys() == {*JOIN_FIELDS, 'identity', 'view'}, 'new query schema changed'
        assert all(row[k] == resource[k] == old[k] for k in JOIN_FIELDS), 'query occurrence changed'
        assert row['identity'] == old['identity'], 'query identity changed'
        bus, trip, _ = resource['cycle_ref']
        assert (bus, trip) == (row['identity']['bus_id'], row['identity']['trip_id'])
        assert resource['snapshot_read_s'] == old['query_s'] == view['query_s'], 'query clock changed'
        assert resource['snapshot_event_seq'] == row['legacy_query_event_seq'] == view['native_event_seq']
        require_legacy_equal(view, row['view'])


def registration(source):
    root = 'protocol_v6_v230_resource_collection'
    slots = [dict(slot='c0_'+arm, context=0, actor=12037, demand_day=2151001, arm=arm,
        sampler_seed=187000000+12037+100*2151001,
        references={name: f'{root}/{folder}/c0_{arm}.json.gz' for name, folder in
                    (('legacy', 'records'), ('v224', 'sidecars'), ('v230', 'resource_sidecars'))})
        for arm in ('v107', 'upper')]
    return dict(stage='v234_passive_cycle_preflight_protocol', source=source,
        parent_source='f6f6a91d89', frozen_cycle_source='7fdf968a5f',
        authorization='Protocol seal,synthetic contract tests and parent small summaries only.',
        current_budget=dict(archive_scans=0, passive_paths=0, intervention_paths=0,
            training_updates=0, model_fits=0, shadow_actions=0, new_futures=0),
        reserved_preflight=dict(max_started_paths=2, slots=slots, max_workers=1,
            order=['c0_v107', 'c0_upper'], extra_reference_paths=0, launch_authorized=False,
            failure_rule='Any simulator start consumes its fixed slot,including failed,incomplete '
                'or duplicate starts. No automatic retry,substitution,extra reference run or '
                'expansion to the remaining six. Start upper only after V107 passes EVERY '
                'equality/recording check. Complete saved outputs may be re-read for analysis '
                'recovery without another simulation;extra starts need separate approval.'),
        frozen=dict(controller_source='2b0115af7b', passive_capture_source='eec7a20c71',
            fleet=12, holding_bins_s=[0, 5, 10, 15, 20, 30, 45],
            runtime='FreqDuet-v6-engineering-v9-runtime-optimized/FreqDuet/freqduet',
            controls='V107 bilateral025 and V220 ready-backed native upper only. Original '
                'bounds,EMA,cadence,anchors,rolling constraints,direction-local callbacks/'
                'caches,native masks and dispatch unchanged. No incoming bus funds an advance.',
            unchanged='Actors,critics,duals,V151,V146,V167,all other predictors,windows,'
                'quantiles,filters and main policy.'),
        equality='Compare the ENTIRE legacy record,V224 sidecar AND V230 resource sidecar '
            'against the corresponding saved V230 payloads with require_legacy_equal: '
            'recursive key/value/list-order equality,no tolerance,rounding,sorting or ignored '
            'fields. Includes inputs/probabilities/uniforms/actions,commands/callbacks/caches/'
            'plans,native events,physical ledgers,all arrival/departure/launch/return clocks '
            'and every endpoint. Preserve old observer/native-call counters and shadow '
            'enumerations. Wall-clock runtime is not a simulator endpoint.',
        passive_boundary='Add an independent observer and a FOURTH separate cycle sidecar '
            'only;do not refactor frozen collectors or change their fields/counters/marks. '
            'Feed original observed launch/query/return events in native order. At an existing '
            'query read the already captured V230 row BEFORE action issue/draw;no additional '
            'snapshot,estimator,planner,policy,env-step or RNG call. New observer has its own '
            'RNG-before/after and native-call checks,zero deltas. No shadow action is executed.',
        schema=dict(metadata=['frozen_cycle_source', 'native_dt_s', 'native_inventory_order',
                'native_timetable_order', 'initial_published_plans', 'initial_version_id'],
            queries=[*JOIN_FIELDS, 'identity', 'view'],
            view='Exact full CycleState.query output from frozen V232,including cycle key,'
                'query/native clocks,H,delta H,previous query clock,message/plan versions,'
                'complete target_profile and own_association. No extra action or future labels.',
            transitions='Original launch/return native seq/time/identity plus exact V232 '
                'transition output. Keep all cycles,including zero-query cycles.',
            observed_links='Separate rows appended ONLY at the first actual subsequent '
                'same-bus launch:observed_at_seq,current_cycle,previous returned-cycle record. '
                'Never backfill earlier queries;initial stock has no fictional parent.',
            terminal='Keep complete open-cycle and returned-without-next-launch records '
                'at end of path;unsupported/unmatched/missing are explicit,null is not0.',
            observer_checks='Own native-call delta0,RNG unchanged,shadow executed0,complete flag.'),
        query_gate='Exactly one new row per old query occurrence,in original order,including '
            'zero holding,masked,unsupported and unmatched rows. Preserve old prepare/query '
            'ordinals as distinct from native event ordinals. Identity and query clocks must '
            'match original rows;positions are query samples,not prepare samples. Reference '
            'V232 views are independently reconstructed in an acceptance pass from the '
            'new V230 payload and native event stream,not copied from new observer output.',
        ledger_gate='Compare launch/query/return transitions and final states with frozen '
            'V232 on the same ordered events. H persists across actual plan writes,no-write '
            'callbacks,station/rank/association changes and pressure relief. Completed cycle '
            'query deltas plus nonnegative last-query-to-return remainder equal actual '
            'returned H exactly;sum closes to the old executed ledger. Zero-query cycles '
            'keep their full return remainder. Open cycles remain open,not zero-filled. '
            'Only actual next launch starts new H=0 and exposes an observed parent link.',
        association_gate='Exact frozen resource_profile at EVERY query:current-ready FIFO '
            'used once,due/future distinction,return+dt boundary,full pending plan IDs and '
            'strict-prior last-write/message versions,current incoming positions/ranks,'
            'unbacked obligations,unmatched buses/slots and null own obligation retained. '
            'No reservations,ETA,slack or future identity. Delta H between samples is not '
            'attributed to the currently associated obligation.',
        pair_gate=list(PAIR_CHECKS),
        reporting='Per fixed slot:all three exact comparisons,query/transition/cycle/link/'
            'missing counts,holding closure,plan writes with positive H,rank/reassociation '
            'and pressure changes,unsupported/masked rows,observer counters. Association '
            'future-label matches are descriptive only,never a100% pass threshold or '
            'path-selection rule. Unexercised real reassociation is reported,not a reason '
            'to add paths. Passive endpoints cannot establish a control benefit.',
        stop='Any equality difference,missing record,clock/version/reset/closure failure '
            'or observer side effect stops progression. Fix engineering without tuning '
            'gain or choosing a new context;no failed-start refund. Even a full pass opens '
            'no intervention,fit,training or policy promotion.',
        next_step='Implement only the independent passive observer,then obtain execution '
            'authorization for these two reserved starts;this seal is NOT authorization.',
        execution='scheduler high,explicit manual dispatch,node001--006. Tests and parent '
            'small-summary reads only now. All raw trajectories and future sidecars stay '
            'remote;return a compact JSON/log receipt only,no CSV/cache/checkpoint.',
        collector_implemented=False, collection_authorized=False, runtime_connected=False,
        main_policy_changed=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v230-root', 'v232-root', 'v233-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    spec = registration(a.source)
    protocol = a.out_root/'protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text()) == spec
        saved = json.loads((a.out_root/'summary/status.json').read_text())
        assert saved['status'] == 'v234_passive_cycle_preflight_sealed' and saved['source'] == a.source
        print('V234_EXISTING_COMPLETE_SEAL_REUSED', flush=True)
        return
    parents = {name: json.loads((root/'summary/status.json').read_text())
               for name, root in (('v230', a.v230_root), ('v232', a.v232_root), ('v233', a.v233_root))}
    assert parents['v230']['status'] == 'v230_remaining_six_passed'
    assert parents['v230']['source'] == 'eec7a20c71' and parents['v230']['all_checked_passed']
    assert parents['v230']['started_paths'] == parents['v230']['checked_paths'] == 8
    assert parents['v232']['status'] == 'v232_persistent_coordination_protocol_sealed'
    assert parents['v232']['source'] == '7fdf968a5f'
    assert parents['v233']['status'] == 'v233_frozen_cycle_state_complete'
    assert parents['v233']['source'] == 'f6f6a91d89' and parents['v233']['paths'] == 8
    assert not parents['v233']['main_policy_changed']
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                              pattern='test_v234_passive_cycle_preflight.py')
    assert suite.countTestCases() > 0
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tests.wasSuccessful():
        raise SystemExit(1)
    a.out_root.mkdir(parents=True, exist_ok=True)
    protocol.write_text(json.dumps(spec, indent=2)+'\n')
    summary = dict(status='v234_passive_cycle_preflight_sealed', source=a.source,
        parent_sources={k: v['source'] for k, v in parents.items()}, tests=tests.testsRun,
        current_budget=spec['current_budget'], reserved_preflight=spec['reserved_preflight'],
        pair_gate=spec['pair_gate'], collector_implemented=False, collection_authorized=False,
        runtime_connected=False, main_policy_changed=False, protocol_path=str(protocol),
        next_step=spec['next_step'])
    (a.out_root/'summary').mkdir(exist_ok=True)
    (a.out_root/'summary/status.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('V234_PROTOCOL_SEALED_NO_COLLECTION '+json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
