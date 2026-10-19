"""Causal holding authorization; requests never spend future or unearned credit."""

from dataclasses import dataclass

from freqduet_incremental_compensation import ACTIONS, bounded_action


@dataclass
class Account:
    station: int = -1
    predecessor: int | None = None
    residual: float | None = None
    issued: float = 0.
    spent: float = 0.
    expired: float = 0.
    balance: float = 0.


class FundedQuota:
    def __init__(self):
        self.trips, self.pending = {}, {}
        self.prefix_checks = 0
        self.minimum_margin = None

    def check_actual(self, trip, executed_total):
        b = self.trips[trip]
        margin = b.issued-b.expired-executed_total
        if executed_total < 0 or margin < 0:
            raise AssertionError('actual holding exceeds earned, unexpired authorization')
        self.prefix_checks += 1
        self.minimum_margin = margin if self.minimum_margin is None else min(self.minimum_margin, margin)
        return margin

    def request(self, key, predecessor, residual, error, feasible, shadow, executed_before):
        trip, station, _ = key
        if any(k[0] == trip for k in self.pending):
            raise AssertionError('cannot mint another authorization before the previous action settles')
        b = self.trips.setdefault(trip, Account())
        contiguous = (residual is not None and b.residual is not None
            and predecessor == b.predecessor and abs(station-b.station) == 1)
        reason = 'missing_forward' if residual is None else 'new_reference' if not contiguous else 'continued'
        expired = b.balance if not contiguous or error <= 0 else 0.
        b.expired += expired
        b.balance -= expired
        innovation = residual-b.residual if contiguous else 0.
        increment = -.75*innovation
        raw = .25*error+(increment if error > 0 else 0.)
        before = b.balance
        b.issued += shadow
        b.balance += shadow
        available = b.balance
        selected = bounded_action(raw, min(max(error, 0.), available), feasible)
        self.check_actual(trip, executed_before)
        row = dict(trip_id=trip, station_id=station, direction=key[2], predecessor_trip_id=predecessor,
            residual_s=residual, innovation_s=innovation, increment_s=increment,
            feedback_error_s=error, reset_reason=reason, expired_s=expired,
            shadow_s=shadow, credit_before_s=before, available_s=available,
            issued_total_s=b.issued, expired_total_s=b.expired, spent_before_s=b.spent,
            raw_request_s=raw, chosen_s=ACTIONS[selected], executed_before_s=executed_before)
        self.pending[key] = row
        b.station, b.predecessor, b.residual = station, predecessor, residual
        return selected, row

    def settle(self, key, executed_total):
        row = self.pending.pop(key)
        actual = executed_total-row['executed_before_s']
        if not 0 <= actual <= row['chosen_s']:
            raise AssertionError('execution is outside the authorized current command')
        b = self.trips[key[0]]
        b.spent += actual
        b.balance -= actual
        margin = self.check_actual(key[0], executed_total)
        if b.balance < 0 or b.spent != executed_total:
            raise AssertionError('settled execution differs from the trip holding ledger')
        row.update(executed_s=actual, spent_after_s=b.spent, credit_after_s=b.balance,
            actual_prefix_margin_s=margin)
        return row

    def audit(self):
        return dict(issued_s=sum(b.issued for b in self.trips.values()),
            executed_s=sum(b.spent for b in self.trips.values()),
            expired_s=sum(b.expired for b in self.trips.values()),
            remaining_s=sum(b.balance for b in self.trips.values()),
            conservation_error=max((abs(b.balance-b.issued+b.spent+b.expired)
                                    for b in self.trips.values()), default=0.),
            prefix_checks=self.prefix_checks, minimum_prefix_margin_s=self.minimum_margin)
