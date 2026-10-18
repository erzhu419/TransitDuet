"""Incremental nonholding response, settled only by completed departures."""

from dataclasses import dataclass


ACTIONS = (0., 5., 10., 15., 20., 30., 45.)


def bounded_action(request, error, feasible):
    choices = [i for i, ok in enumerate(feasible) if ok and ACTIONS[i] <= max(error, 0.)]
    return min(choices, key=lambda i: abs(ACTIONS[i] - min(max(request, 0.), ACTIONS[-1])))


@dataclass
class Budget:
    station: int = -1
    predecessor: int | None = None
    residual: float | None = None
    target: float = 0.
    delivered: float = 0.
    innovations: float = 0.
    discarded: float = 0.
    actual: float = 0.
    shadow: float = 0.


class Compensation:
    def __init__(self, retain=True):
        self.retain = retain
        self.trips, self.pending = {}, {}

    def request(self, key, predecessor, residual, error, feasible, shadow, executed_before):
        trip, station, _ = key
        if any(k[0] == trip for k in self.pending):
            raise AssertionError('trip requested a new action before its previous departure settled')
        b = self.trips.setdefault(trip, Budget())
        contiguous = (residual is not None and b.residual is not None
            and predecessor == b.predecessor and abs(station - b.station) == 1)
        reason = ('missing_forward' if residual is None else 'new_reference' if not contiguous
                  else 'single_step' if not self.retain else 'continued')
        discarded = 0.
        if not contiguous or not self.retain:
            discarded += b.target - b.delivered
            b.target = b.delivered
        innovation = residual - b.residual if contiguous else 0.
        increment = -.75 * innovation
        b.target += increment
        b.innovations += increment
        if error <= 0.:
            discarded += b.target - b.delivered
            b.target = b.delivered
        b.discarded += discarded
        unserved = b.target - b.delivered
        raw = .25 * error + unserved
        selected = bounded_action(raw, error, feasible)
        row = dict(trip_id=trip, station_id=station, direction=key[2], predecessor_trip_id=predecessor,
            residual_s=residual, innovation_s=innovation, increment_s=increment, reset_reason=reason,
            feedback_error_s=error, target_extra_s=b.target, delivered_extra_before_s=b.delivered,
            unserved_s=unserved, discarded_s=discarded, raw_request_s=raw,
            shadow_s=shadow, chosen_s=ACTIONS[selected], executed_before_s=executed_before)
        self.pending[key] = row
        b.station, b.predecessor, b.residual = station, predecessor, residual
        return selected, row

    def settle(self, key, executed_total):
        row = self.pending.pop(key)
        actual = executed_total - row['executed_before_s']
        if actual < 0.:
            raise AssertionError('trip holding counter decreased before departure')
        b = self.trips[key[0]]
        credit = actual - row['shadow_s']
        b.delivered += credit
        b.actual += actual
        b.shadow += row['shadow_s']
        row.update(executed_s=actual, settled_extra_s=credit, delivered_extra_after_s=b.delivered)
        return row

    def conservation_errors(self):
        return dict(target=max((abs(b.target - b.innovations + b.discarded) for b in self.trips.values()), default=0.),
            delivered=max((abs(b.delivered - b.actual + b.shadow) for b in self.trips.values()), default=0.))
