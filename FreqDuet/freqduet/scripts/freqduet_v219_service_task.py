"""Observable service-task admission for a prospective native-cycle candidate."""

from math import ceil

from freqduet_v218_cycle_policy import ACTIONS, CONTEXTS, START_S, WINDOW_S


def ready_prefix(query):
    """Spend each already-ready vehicle once, including earlier pending trips."""
    origin = next(p['public_s'] for p in query['plans'] if p['trip_id'] == query['trip_id'])
    anchors, covered = set(), {}
    for direction in (False, True):
        pending = sorted((p for p in query['plans'] if p['direction'] == direction),
                         key=lambda p: (p['public_s'], p['trip_id']))
        candidates = [p for p in pending if origin <= p['public_s']]
        if candidates:
            anchors.add(candidates[0]['trip_id'])
        stock = sorted((b for b in query['ready'] if b['direction'] == direction),
                       key=lambda b: (b['last_observed_return_s'] if b['return_observed'] else -float('inf'),
                                      b['bus_id']))
        covered.update({p['trip_id']: b['bus_id'] for p, b in zip(pending, stock)})
    eligible = [p['trip_id'] for p in query['plans'] if p['trip_id'] in covered
                and p['trip_id'] not in anchors and origin <= p['public_s'] < origin+WINDOW_S]
    return dict(origin_s=origin, expiry_public_s=origin+WINDOW_S, anchor_ids=sorted(anchors),
                ready_bus_by_trip=covered, mutable_ready_trip_ids=sorted(eligible))


def admit_preview(query, native, proposed, tick_s):
    """Accept only a native projection with ready-backed, non-delaying execution."""
    resource = ready_prefix(query)
    assert native.keys() == proposed.keys() == {p['trip_id'] for p in query['plans']}
    for trip in resource['anchor_ids']:
        if abs(native[trip]-proposed[trip]) > 1e-6:
            return dict(admitted=False, reason='immutable_anchor_changed', task_ids=[])
    now = query['time_s']
    advanced = []
    for trip, plan in native.items():
        old = max(now, ceil(plan/tick_s)*tick_s)
        new = max(now, ceil(proposed[trip]/tick_s)*tick_s)
        if new > old:
            return dict(admitted=False, reason='later_native_eligibility', task_ids=[])
        if new < old and new < resource['expiry_public_s']:
            if trip not in resource['mutable_ready_trip_ids']:
                return dict(admitted=False, reason='unfunded_advanced_trip', task_ids=[])
            advanced.append(trip)
    return dict(admitted=bool(advanced), reason='ready_backed_release' if advanced else 'no_executable_advance',
                task_ids=sorted(advanced))


def lower_request(query, task, release_gap_s):
    base = query['base_bin']
    if task is None or (query['trip_id'], query['bus_id']) != (task['trip_id'], task['bus_id']):
        return base
    if query['event_seq'] <= task['launch_event_seq'] or release_gap_s is None:
        return base
    smaller = [b for b in query['feasible_bins'] if b < base]
    if (base not in query['feasible_bins'] or not smaller or query['terminal_masked']
            or not query['forward_valid'] or not query['follower_valid']):
        return base
    chosen = max(smaller)
    error = release_gap_s-query['context']['target_headway_s']
    return chosen if (error+ACTIONS[chosen])**2 <= (error+ACTIONS[base])**2 else base


def registration(source):
    return dict(stage='v219_ready_backed_outbound_service_task', source=source,
        contexts=[list(c) for c in CONTEXTS], start_s=START_S, window_s=WINDOW_S,
        upper='At native fresh queries,preview exactly native and native+30s raw command '
            '(clipped[-60,60],unchanged EMA.35 and exact shared rolling-zero-sum projection). '
            'Use native planner copies,never custom schedule writes. First pending anchor in each '
            'direction remains immutable. Current ready stock funds chronological pending prefixes '
            'once per vehicle,including older due/anchor tasks. An advanced mutable trip before '
            'public-origin+900s needs its own already-ready compatible vehicle. No proposed native '
            'eligibility clock may become later. Require at least one executable tick of advance. '
            'Otherwise retain native. Plans beyond the public cycle are not independent service promises.',
        lower='After actual launch of an admitted task,track that SAME outbound trip and actual bus '
            'until return,across native cycle changes. At its positive valid bilateral query reduce '
            'one feasible bin only if the completed-predecessor release+executed-holding squared '
            'gap error does not increase. Missing evidence keeps bilateral025. No enroute return '
            'prediction,blanket later-holding suppression or extra credit is issued.',
        current_work='Read only four existing V216 passive captures. Count query-time ready/mutable '
            'prefixes and positive lower opportunities. Actual assignment and strict plan-binding '
            'are separately labeled after selection;never enter admission or replace contexts. '
            'Do not use V218 endpoints,fit scores or claim this count runs the proposed planner.',
        progression='Require a mutable ready-backed task and a later positive feasible own-trip '
            'query in EVERY fixed context before implementing native preview equivalence. Counts '
            'only admit engineering work,not simulation. No new paths are authorized here. If any '
            'context has no opportunity,stop this structure without anchor replacement or gain search.',
        subsequent_test='Separately freeze budget before replay: V107,upper-only,lower-only,joint. '
            'Verify native preview exactness,own-task bus identity and actual launch advance versus '
            'lower-only while holding the ready supply prefix fixed. Co-movement is insufficient. '
            'Require actual own-trip holding change and report full-day passenger time,arrival CV2, '
            'readiness,original-public total launch delay,completion,wait/onboard and early departures. '
            'Only joint passenger benefit without CV2/readiness/public-delay/completion harm admits '
            'a disjoint-future protocol. Fixed+bilateral remains mandatory for later efficacy.',
        frozen='Original12 vehicles,seven bins,native one-dimensional shared authority and cadence; '
            'V151,V146,V167 and all other predictor/calibration/filter/actor/critic/dual versions unchanged.',
        budget=dict(new_replays=0, new_training=0, new_futures=0), main_policy_changed=False)
