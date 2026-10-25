"""Arrival-count evidence from APC conservation, not a demand forecast or policy."""

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class APCArrival:
    order: int
    bus: int
    trip: int
    direction: bool
    station: int
    arrival_s: float
    available_s: float
    completed_s: float | None
    incoming: float
    alighted: float
    boarded: float
    capacity: float

    @property
    def key(self):
        return self.direction, self.station

    @property
    def free_before_boarding(self):
        return self.capacity - self.incoming + self.alighted

    @property
    def cleared(self):
        return self.boarded < self.free_before_boarding

    def validate(self):
        assert self.capacity > 0 and 0 <= self.alighted <= self.incoming <= self.capacity
        assert 0 <= self.boarded <= self.free_before_boarding
        assert self.arrival_s <= self.available_s
        assert self.completed_s is None or self.arrival_s <= self.completed_s


def interval_bounds(previous, current):
    """A = B_current + Q_after_current - Q_after_previous for adjacent arrivals."""
    previous.validate()
    current.validate()
    assert previous.key == current.key and previous.order < current.order
    assert previous.arrival_s <= current.arrival_s
    if previous.arrival_s == current.arrival_s:
        # Native passenger release precedes the whole bus-drive loop. There is
        # no exogenous release between same-clock boarding events.
        assert not previous.cleared or current.boarded == 0
        lower, upper = 0., 0.
    elif previous.cleared and current.cleared:
        lower = upper = current.boarded
    elif previous.cleared:
        lower, upper = current.boarded, None
    elif current.cleared:
        lower, upper = 0., current.boarded
    else:
        lower, upper = 0., None
    return dict(direction=current.direction, station=current.station,
        first_order=previous.order, last_order=current.order,
        start_s=previous.arrival_s, end_s=current.arrival_s,
        lower_arrivals=lower, upper_arrivals=upper,
        point_identified=upper is not None and upper == lower)


def completed_flow(events, query_s):
    """Consume the complete native arrival roster, including incomplete visits.

    Omitting an incomplete/censored event can fabricate a clearance-to-clearance
    label. An unavailable visit breaks that station's chain; later completed
    clearances may start a new one. Same-clock events retain native arrival order.
    """
    observed = [e for e in events if e.arrival_s < query_s]
    assert len({e.order for e in observed}) == len(observed)
    ordered = sorted(observed, key=lambda e: e.order)
    assert all(a.arrival_s <= b.arrival_s for a,b in zip(ordered, ordered[1:]))
    if ordered:
        assert [e.order for e in ordered] == list(range(ordered[0].order, ordered[-1].order+1))
    chains = defaultdict(lambda: dict(previous=None, anchor=None, boarded=0., visits=0,
                                     max_completed=None, max_available=None))
    intervals, blocks = [], []
    unavailable = 0
    for event in ordered:
        state = chains[event.key]
        usable = (event.completed_s is not None and event.completed_s < query_s
                  and event.available_s < query_s)
        if not usable:
            unavailable += 1
            chains[event.key] = dict(previous=None, anchor=None, boarded=0., visits=0,
                                     max_completed=None, max_available=None)
            continue
        event.validate()
        if state['previous'] is not None:
            intervals.append(interval_bounds(state['previous'], event))
        if state['anchor'] is not None:
            state['boarded'] += event.boarded
            state['visits'] += 1
            state['max_completed'] = max(state['max_completed'], event.completed_s)
            state['max_available'] = max(state['max_available'], event.available_s)
        if event.cleared:
            anchor = state['anchor']
            if anchor is not None:
                seconds = event.arrival_s-anchor.arrival_s
                assert seconds > 0 or state['boarded'] == 0
                blocks.append(dict(direction=event.direction, station=event.station,
                    first_order=anchor.order, last_order=event.order,
                    start_s=anchor.arrival_s, end_s=event.arrival_s,
                    arrivals=state['boarded'], visits=state['visits'],
                    max_completed_s=state['max_completed'], max_available_s=state['max_available'],
                    interval_average_rate=None if seconds == 0 else state['boarded']/seconds))
            state.update(anchor=event, boarded=0., visits=0,
                         max_completed=event.completed_s, max_available=event.available_s)
        state['previous'] = event
    open_blocks = []
    for (direction,station),state in chains.items():
        if state['anchor'] is not None and state['visits']:
            open_blocks.append(dict(direction=direction, station=station,
                first_order=state['anchor'].order, last_order=state['previous'].order,
                lower_arrivals=state['boarded'], upper_arrivals=None))
    return dict(intervals=intervals, exact_blocks=blocks, open_blocks=open_blocks,
        unavailable_visits=unavailable, query_s=query_s,
        future_rate_upper_bound_identified=False)
