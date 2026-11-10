"""Pure archive accounting for the frozen initial-queue estimator."""
from collections import defaultdict

from freqduet_apc_flow_evidence import completed_flow


def estimate(events, q, key=(True, 9)):
    """Reproduce the old queue formula using only its original APC clocks.

    Pass the complete global roster: an unavailable visit must break its chain.
    No passenger truth, model evaluation, or alternative estimator is used.
    """
    # A pseudo-query is before step(q). Same-tick arrivals in a later archive
    # have not occurred yet, even if their recorded availability equals q.
    events = [event for event in events if event.arrival_s < q]
    evidence = completed_flow(events, q)
    blocks = [b for b in evidence['exact_blocks']
        if (b['direction'], b['station']) == key and b['end_s'] > b['start_s']][-3:]
    if len(blocks) < 3:
        return None
    seconds = sum(b['end_s']-b['start_s'] for b in blocks)
    count = sum(b['arrivals'] for b in blocks)
    rate, anchor = count/seconds, blocks[-1]
    amount, clock = 0., anchor['end_s']
    last_order, clearance_clock = anchor['last_order'], clock
    known = max(max(b['max_completed_s'], b['max_available_s']) for b in blocks)
    reconciles = []
    # The archived roster already has native arrival order. Preserve it, as the
    # original controller does, including same-clock boarding order.
    for event in events:
        if (event.key != key or event.available_s >= q
                or event.order <= anchor['last_order']):
            continue
        demand = rate*(event.arrival_s-clock)
        before = amount+demand
        amount = max(0., before-event.boarded)
        if event.cleared:
            amount = 0.
            clearance_clock = event.arrival_s
        reconciles.append(dict(order=event.order, arrival_s=event.arrival_s,
            available_s=event.available_s, boarded=event.boarded,
            cleared=event.cleared, generated=demand,
            queue_before=before, queue_after=amount))
        clock, last_order = event.arrival_s, event.order
        known = max(known, event.available_s)
    tail = rate*(q-clock)
    return dict(predicted_queue=amount+tail, rate_person_per_s=rate,
        observation_age_s=q-clock, clearance_age_s=q-clearance_clock,
        last_event_order=last_order, anchor_order=anchor['last_order'],
        history_count=count, history_seconds=seconds, closed_blocks=blocks,
        reconciles=reconciles, known_through_s=known, clock=clock,
        reconciled_queue_persons=amount, tail_queue_persons=tail)


def actual_queue(passengers, q):
    """Retrospective truth before native step(q), not a predictor input."""
    return sum(float(p['appear_time']) <= q-1+1e-9 and
        (not p['boarded'] or float(p['boarding_time']) >= q)
        for p in passengers)


def summarize(rows):
    """Describe queries and equal-weight APC gaps without fitting or inference."""
    rows = list(rows)
    gaps = defaultdict(list)
    for row in rows:
        gaps[row['last_event_order']].append(row)

    def mean(values):
        values = list(values)
        return sum(values)/len(values) if values else None

    def signs(values):
        values = list(values)
        return dict(over=sum(v > 1e-8 for v in values),
            under=sum(v < -1e-8 for v in values),
            equal=sum(abs(v) <= 1e-8 for v in values))

    errors = [r['error_persons'] for r in rows]
    gap_biases = [mean(r['error_persons'] for r in group)
        for group in gaps.values()]
    gap_maes = [mean(abs(r['error_persons']) for r in group)
        for group in gaps.values()]
    return dict(n_queries=len(rows), n_gaps=len(gaps),
        bias_persons=mean(errors), mae_persons=mean(abs(e) for e in errors),
        mean_actual_queue=mean(r['actual_queue'] for r in rows),
        mean_predicted_queue=mean(r['predicted_queue'] for r in rows),
        query_sign_counts=signs(errors), gap_bias_persons=mean(gap_biases),
        gap_mae_persons=mean(gap_maes), gap_sign_counts=signs(gap_biases),
        query_start_s=min((r['query_s'] for r in rows), default=None),
        query_end_s=max((r['query_s'] for r in rows), default=None))
