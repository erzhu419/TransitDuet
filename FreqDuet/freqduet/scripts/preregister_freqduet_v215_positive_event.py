#!/usr/bin/env python3
"""V215 positive-event causal selection contract; no simulation entrypoint."""

import itertools


CONTEXTS = tuple(itertools.product((12037, 933083), (2151001, 2151002)))
ACTIONS = (0, 5, 10, 15, 20, 30, 45)
CANDIDATES = tuple(itertools.product((0, -30, 30), (0, 1)))
CONSTRAINTS = ('cv_squared', 'readiness_delay_s', 'public_launch_delay_total_s',
               'trips_unlaunched', 'trips_incomplete')
SNAPSHOT_FIELDS = ('time_s', 'event_seq', 'snapshot_read_s', 'snapshot_event_seq', 'ready', 'active', 'plans')
LOWER_FIELDS = (*SNAPSHOT_FIELDS, 'state_captured_s', 'bus_id', 'trip_id', 'direction', 'station_id',
                'terminal_masked', 'forward_valid', 'follower_valid', 'base_bin', 'feasible_bins')


def future_seed(context, future):
    assert context in range(4) and future in range(32)
    return 2150000000 + 100*context + future


def current_snapshot(event):
    return (all(k in event for k in SNAPSHOT_FIELDS)
            and event['snapshot_read_s'] == event['time_s']
            and event['snapshot_event_seq'] == event['event_seq'])


def resource_link(event, bus_id, trip_id, inbound_direction):
    """Observable queue coverage, not a prediction of return/assignment order."""
    target = not bool(inbound_direction)
    if any(b['direction'] == target for b in event['ready']):
        return None
    due = sorted((p for p in event['plans'] if not p['launched'] and p['direction'] == target
                  and p['planned_s'] <= event['time_s']), key=lambda p: (p['planned_s'], p['trip_id']))
    incoming = sorted((b for b in event['active'] if b['direction'] == inbound_direction),
                      key=lambda b: (b['remaining_distance_m'], b['bus_id'], b['trip_id']))
    ranks = [i+1 for i, b in enumerate(incoming) if (b['bus_id'], b['trip_id']) == (bus_id, trip_id)]
    if not ranks or ranks[0] > len(due):
        return None
    return dict(target_direction=target, due_trip_ids=[p['trip_id'] for p in due],
                inbound_rank=ranks[0], due_count=len(due))


def lower_anchor(event):
    if not all(k in event for k in LOWER_FIELDS) or not current_snapshot(event):
        return None
    if not 7200 <= event['time_s'] < 8100 or event['state_captured_s'] > event['time_s']:
        return None
    base = event['base_bin']
    if (base <= 0 or event['terminal_masked'] or not event['forward_valid'] or not event['follower_valid']
            or base not in event['feasible_bins']):
        return None
    smaller = [i for i in event['feasible_bins'] if i < base]
    if not smaller:
        return None
    link = resource_link(event, event['bus_id'], event['trip_id'], event['direction'])
    if link is None:
        return None
    return dict(time_s=event['time_s'], event_seq=event['event_seq'], bus_id=event['bus_id'],
        trip_id=event['trip_id'], station_id=event['station_id'], inbound_direction=bool(event['direction']),
        base_bin=base, smaller_bin=max(smaller), **link)


def upper_followup(anchor, event):
    if not current_snapshot(event) or not event.get('fresh_upper', False):
        return False
    if ((event['time_s'], event['event_seq']) <= (anchor['time_s'], anchor['event_seq'])
            or not anchor['time_s'] <= event['time_s'] < anchor['time_s']+900
            or event['direction'] != anchor['target_direction']):
        return False
    link = resource_link(event, anchor['bus_id'], anchor['trip_id'], anchor['inbound_direction'])
    return link is not None and bool(set(link['due_trip_ids']) & set(anchor['due_trip_ids']))


def registration(source):
    return dict(stage='v215_current_positive_holding_event', source=source,
        parent='V214 rejected:nearest two incoming trips had eight valid but zero requests;never reopen its banks.',
        contexts=[list(c) for c in CONTEXTS], new_scenario_seeds=[2151001, 2151002],
        baseline='Frozen B200 deployment with bilateral025 lower;same two trained actors,new demand seeds.',
        lower='FIRST lower query in[7200,8100) with CURRENT positive deterministic one-hot bilateral025 bin,strictly smaller feasible '
            'bin,valid bilateral evidence,nonterminal and native mask respected. Capture complete query-time '
            'ready/active/pending-plan snapshot. Target terminal is the opposite of this trip direction;require '
            'zero compatible ready vehicles and currently due executable plans. Rank ALL current incoming '
            'vehicles by remaining distance,bus ID,trip ID;require selected rank<=current due-queue length. '
            'Position is a queue-coverage screen AFTER positive-action support,not a prior lock on two vehicles. '
            'No supported anchor in ANY context stops the pilot;no replacing days or postselecting contexts.',
        queue_link='Current destination/direction and finite inventory establish potential supply to a real '
            'observed obligation. Distance rank is NOT return order,ETA,guaranteed allocation or slack. '
            'Keep original due-trip IDs. Do not declare a future vehicle/dispatch match at selection time.',
        clock='Native observation-ready hook records true state creation time;do not use nonexistent '
            'pre_action_time_s or fill it with query time. Record snapshot-read time and a GLOBAL native '
            'event ordinal across prepare/lower-query/fresh-upper/plan/launch/return callbacks. Same-clock '
            'events are ordered by this ordinal,not by hindsight timestamps. Snapshot captured in its own '
            'query callback;no backfill from end-of-day trajectories or post-query launch/return labels.',
        intervention='Lower0 keeps the anchor command;lower1 reduces ONE adjacent feasible bin ONCE on this '
            'actual query. No second vehicle,second stop,added holding or new negative action. No replay of '
            'V214 paths to locate a winner. Future fork occurs after exact common positive anchor/prefix '
            'match but BEFORE its categorical draw/issued holding;all six arms use this same prefix.',
        upper='After the lower anchor,at FIRST qualifying NATIVE fresh upper decision(strict event order, '
            'possibly same clock),within900s:callback dispatch direction equals target terminal,selected bus '
            'still active on SAME inbound trip,no compatible ready bus,current rank still covered by current '
            'due queue,and at least one original due-trip ID remains currently due/unlaunched. Perturb native '
            'pre-EMA shared coefficient by0/-30/+30 clipped[-60,60],then keep all native execution gates. '
            'Never call the actor early,force a replan,revise a past decision or synthesize a terminal command. '
            'Absent/clipped/erased upper opportunity is retained as such;never retry at a later decision. '
            'Follow-up opportunities may differ by branch and are NOT outcome filters.',
        authority='Original12 fixed-pool vehicles,seven0/5/10/15/20/30/45s bins,single shared antisymmetric '
            'upper coefficient,EMA.35 and native900s rolling-zero-sum exact-headway planner. No service '
            'deletion,vehicle creation,layover reduction,direct release,terminal wait override or slack credit. '
            'Actor/critic/dual/predictors/V146/filters/windows/quantiles and main policy unchanged.',
        candidates=[dict(id=i, upper_shift_s=u, reduce_anchor=l) for i, (u, l) in enumerate(CANDIDATES)],
        selection_futures=list(range(16)), evaluation_futures=list(range(16, 32)),
        future_namespace='2150000000+100*context+future;separate smoke2159000000+context',
        fork='Keep frozen V68 innovation-fork semantics,including the separate event-uniform stream offset '
            '+10000000. Current deterministic lower action is known before its native uniform draw;never '
            'condition anchor selection on a future stochastic draw or restart currently sampled road/demand windows.',
        select='After separately authorized capture/equivalence smoke and four complete anchor references, '
            'run all6 arms on16 selection futures/context. Select minimum mean passenger_person_s under ALL '
            'nonharm mean constraints vsID0;pool all6/upper-only[0,2,4]/lower-only[0,1];ID0 fallback,tie smaller '
            'ID. Freeze all three choices and best-single before any evaluation bank is opened. Independent '
            'future innovations only;hidden currently sampled window/day remains fixed,as in V68/V211.',
        metrics=dict(passenger_person_s='Full-horizon waiting+onboard exposure,including censored passengers; '
                'report all-generated-demand journey too.',
            cv_squared='Frozen full-service actual arrival-gap CV squared,both directions,not local/departure proxy.',
            readiness_delay_s='Original actual-minus-first-eligible mean;also record total and physical clocks.',
            public_launch_delay_total_s='SUM max(actual-original PUBLIC launch,0),including control waiting; '
                'also mean/counts. Missing service is missing,not zero delay.'),
        constraints=list(CONSTRAINTS),
        acceptance='For joint minus baseline AND EACH selection-best single:passenger95%CI upper<0,ALL '
            'constraint95%CI uppers<=0 under BOTH crossed actor/day and conditional-future bootstrap '
            '(5000 draws,seed215;paired futures within contexts,actors/days independently). Same passenger '
            'improvement/nonharm means in EACH fixed evaluation half16--23/24--31. No tradeoff score, '
            'relaxed margins,context dropping or adaptive seed/future extension.',
        execution='Conserve elapsed=executed holding+other elapsed per physical trip;all commands must match '
            'native feasible issued actions. Keep all branches,including absent upper opportunity,clipping, '
            'blocked dispatch,overtaking/assignment mismatch and missing labels. Uncleared service or broken '
            'support/clock/equivalence blocks admission,not dataset retention. EACH evaluation half needs '
            'a genuine joint lower reduction and changed upper plan clock. Supply-chain attribution also '
            'needs earlier selected-trip return and earlier actual SAME-vehicle target-terminal dispatch '
            'serving an anchor due-trip ID versus matched baseline;revised eligibility alone is insufficient. '
            'Future return/assignment labels are evaluation ONLY and cannot select anchors or filter effects.',
        budget=dict(references=4, smoke_attempts_max=6, selection_max=384, evaluation_max=256,
            incomplete_smoke_attempts_count=True, automatic_budget_amendment=False),
        next_gate='No capture/replay authorized NOW. Existing V214 query rows lack live queue/ready/active '
            'inventories and unified event ordinals;they cannot validate new anchors. Next separate stage '
            'must implement passive query snapshots/event ordering,prove observation/action/timing/ledger '
            'equivalence and actual nonzero lower/upper execution on separately reserved smoke data. '
            'Stop on missing support;no switch back to nearest-two or forced fresh decision.',
        current_authorization='Independent preregistration,synthetic tests,static runtime/schema and namespace audit ONLY.',
        limitations='A small conditional opportunity pilot with two actors/two new demand seeds,not full-day '
            'online policy confirmation,strong-fixed superiority or frequency-separation evidence. Actual '
            'query support,follow-up upper opportunity and physical supply linkage remain unmeasured.',
        new_rollouts=0, training_updates=0, interval_fit=False, main_policy_changed=False)
