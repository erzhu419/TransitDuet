"""Frozen observable resource-cycle candidate; no fitted scores or forecasts."""

START_S = 7200.
WINDOW_S = 900.
SHIFT_S = 30.
ARMS = ('v107', 'upper', 'lower', 'joint')
CONTEXTS = ((12037, 2151001), (12037, 2151002), (933083, 2151001), (933083, 2151002))
ACTIONS = (0, 5, 10, 15, 20, 30, 45)


def cycle_context(query):
    now = query['time_s']
    tasks = [p for p in query['plans'] if not p['launched'] and p['planned_s'] <= now+WINDOW_S]
    ready = [sum(b['direction'] == d for b in query['ready']) for d in (False, True)]
    due = [sum(p['direction'] == d and p['planned_s'] <= now for p in tasks) for d in (False, True)]
    obligations = [sum(p['direction'] == d for p in tasks) for d in (False, True)]
    # Positive amplitude front-loads BOTH directions; it does not reallocate directions.
    shift = -SHIFT_S if any(n > r for n, r in zip(due, ready)) else (
        SHIFT_S if tasks and all(r >= n for r, n in zip(ready, obligations)) else 0.)
    return dict(time_s=now, event_seq=query['event_seq'], task_ids=[p['trip_id'] for p in tasks],
                ready=ready, due=due, obligations=obligations, shift_s=shift)


def lower_request(query, cycle, release_holding_gap_s):
    """Reduce one feasible bin only on a live task without losing release compensation."""
    base = query['base_bin']
    if cycle is None or query['event_seq'] <= cycle['event_seq']:
        return base, 'no_prior_cycle', None
    if base <= 0 or query['terminal_masked'] or not (query['forward_valid'] and query['follower_valid']):
        return base, 'no_positive_bilateral_support', None
    smaller = [i for i in query['feasible_bins'] if i < base]
    if base not in query['feasible_bins'] or not smaller:
        return base, 'no_smaller_native_bin', None
    target = not query['direction']
    if any(b['direction'] == target for b in query['ready']):
        return base, 'ready_supply', None
    due = [p['trip_id'] for p in query['plans'] if not p['launched'] and p['direction'] == target
           and p['planned_s'] <= query['time_s']]
    task_due = sorted(set(due) & set(cycle['task_ids']))
    incoming = sorted((b for b in query['active'] if b['direction'] == query['direction']),
                      key=lambda b: (b['remaining_distance_m'], b['bus_id'], b['trip_id']))
    rank = next((i+1 for i, b in enumerate(incoming) if (b['bus_id'], b['trip_id']) ==
                 (query['bus_id'], query['trip_id'])), None)
    if not task_due or rank is None or rank > len(due):
        return base, 'no_current_cycle_supply_link', None
    link = dict(target_direction=target, due_trip_ids=task_due, inbound_rank=rank)
    if release_holding_gap_s is None:
        return base, 'missing_completed_predecessor', link
    selected = max(smaller)
    error = release_holding_gap_s-query['context']['target_headway_s']
    if (error+ACTIONS[selected])**2 > (error+ACTIONS[base])**2:
        return base, 'preserve_release_compensation', link
    return selected, 'reduce_current_resource_request', link


def registration(source):
    return dict(stage='v218_persistent_native_upper_resource_cycle', source=source,
        contexts=[list(c) for c in CONTEXTS], arms=list(ARMS), start_s=START_S,
        upper='At EVERY native fresh shared upper callback from7200s onward, record current pending tasks '
            'scheduled by now+900s and ready/due counts by direction. Request native coefficient-30s '
            'if ANY current due queue exceeds compatible ready stock;otherwise+30s only if current ready '
            'stock covers ALL recorded next900s obligations in BOTH directions;otherwise0. Clip to native '
            '[-60,60],retain EMA.35 and zero-sum exact-headway projection. Shared antisymmetric linear '
            'means front/back TIME allocation in both directions,NOT independent directional allocation. '
            'No forced fresh call,anchor change,terminal wait override or extra action dimension.',
        lower='At each actual positive bilateral025 query after the latest native fresh cycle,require '
            'valid bilateral evidence,native smaller feasible bin,nonterminal,zero compatible ready '
            'vehicles,and current incoming-distance rank covered by current due queue. At least one '
            'currently due ID must belong to the cycle tasks. Do not bind nearest vehicles before a '
            'positive query. Reduce ONE adjacent feasible bin only when squared release-gap+executed-'
            'holding-gap error does not increase. Compute this from the actual own launch and an already '
            'completed predecessor departure,not a future label. Missing evidence abstains. All other '
            'requests keep bilateral025,including useful compensation and later feedback.',
        authority='Original12 fixed-pool vehicles,seven0/5/10/15/20/30/45s bins,native900s replanning '
            'and original hard headway/zero-sum constraints. No minted slack,vehicle creation,service '
            'deletion,forecast fitting,actor/critic/dual updates or main switch.',
        smoke='Four consumed V216 contexts only. First four V107 paths must exactly reproduce archived '
            'results,physical contracts,full actions,inputs,uniforms,plans,arrivals,departures and returns. '
            'Only then run upper/lower/joint for all four contexts:16 total paths,failed started attempts '
            'count,no automatic replay extension or replacement. This is engineering/execution evidence, '
            'not fresh demand confirmation or parameter selection.',
        execution_gate='Every context must clear service,conserve issued/executed holding and have a '
            'nonzero upper request and changed plan. Joint needs a genuinely executed lower reduction '
            'supplying a cycle due obligation;the SAME immediate same-vehicle target launch must have '
            'different planned AND actual clocks versus lower-only,with the control identity/assignment '
            'retained. Merely changed eligibility,unrelated launch,return rank or requested0->0 is insufficient. '
            'Retain all unsupported contexts and stop efficacy admission if any context fails.',
        outcomes=['passenger_person_s', 'journey_min', 'cv_squared', 'readiness_delay_s',
                  'public_launch_delay_total_s', 'trips_unlaunched', 'trips_incomplete', 'holding_s_per_trip'],
        future_plan='No future bank is opened here. If execution passes,freeze this SAME law before '
            'a separate matched continuation protocol with disjoint selection/evaluation innovations. '
            'No gain or trigger search on these16 paths. Fixed+bilateral,V107,B200 and exact-fixed/zero '
            'remain required subsequent controls;matched TRAINING ablations are still needed for frequency claims.',
        budget=dict(smoke_paths_max=16, new_training=0, independent_future_paths=0),
        limitations='The protected release/holding component is not the complete headway objective;actual '
            'CV2,full-day passenger time,readiness and original-public total delay remain joint efficacy '
            'constraints. Distance rank is not ETA or guaranteed assignment. Counts do not create vehicles.',
        main_policy_changed=False, future_banks_opened=False)
