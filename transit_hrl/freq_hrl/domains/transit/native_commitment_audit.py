"""Passive comparison of full-tail scores and the next same-direction replan."""

import numpy as np

from freq_hrl.domains.transit.native_distributional_supply_plan import NativeDistributionalSupplyPlan, ensemble_peaks
from freq_hrl.domains.transit.native_supply_plan import occupancy_peak


def window_ensemble_peaks(starts, durations, end_s, historical_peak):
    starts = np.asarray(starts)
    keep = starts < end_s
    bounded = np.minimum(durations[:, keep], end_s-starts[keep])
    return ensemble_peaks(starts[keep], bounded, historical_peak)


def histogram(peaks):
    values, counts = np.unique(peaks, return_counts=True)
    return {str(int(v)): int(n) for v, n in zip(values, counts)}


def fleet_cost(counts, target=12):
    return sum(n*max(int(p)-target, 0)**2/target for p, n in counts.items()) / sum(counts.values())


def observed_window_peak(observations, start_s, end_s, historical_peak):
    starts = np.maximum([r["launch_time_s"] for r in observations], start_s)
    ends = np.minimum([r["completion_time_s"] for r in observations], end_s)
    keep = ends > starts
    return max(historical_peak, occupancy_peak(starts[keep], ends[keep]))


class NativeCommitmentAudit(NativeDistributionalSupplyPlan):
    def __init__(self, env, model):
        super().__init__(env, model)
        self.audits = []

    def preferred_intervals(self, nominal, trip, now):
        self._audit_remaining = sorted((t for t in self.env.timetables if bool(t.direction) == bool(trip.direction)
            and not t.launched and t.launch_turn not in self._targets), key=lambda t: t.launch_time)
        self._candidate_audits = []
        preference, rates = super().preferred_intervals(nominal, trip, now)
        d = self.decisions[-1]
        selected_proposal = self._candidate_audits[d["selected"]]["proposed_planned_s"]
        for candidate in self._candidate_audits:
            del candidate["proposed_planned_s"]
        self.audits.append({"time_s": now, "direction": bool(trip.direction),
            "historical_peak": d["historical_peak"], "selected": d["selected"],
            "proposed_trip_ids": [int(t.launch_turn) for t in self._audit_remaining],
            "selected_full_planned_s": selected_proposal, "candidates": self._candidate_audits})
        return preference, rates

    def candidate_fleet_cost(self, peak, planned_by_id, now):
        original_cost, original_diagnostics = super().candidate_fleet_cost(peak, planned_by_id, now)
        prefix = self._audit_remaining[:self.block_size+1]
        planned = np.asarray([planned_by_id[int(t.launch_turn)] for t in prefix])
        next_trip = prefix[-1] if len(prefix) > self.block_size else None
        # The tentative next-block first departure becomes its fixed anchor.
        end_s = (planned[-1]-self.env._upper_dispatch_lookahead_s if next_trip is not None
            else float(self.env.protocol.evaluation_end_time_s))
        if end_s <= now:
            raise RuntimeError("Replan audit needs a future same-direction boundary")
        starts, durations = self.sampled_intervals(planned_by_id, now)
        counts = histogram(window_ensemble_peaks(starts, durations, end_s, self.env._peak_concurrent))
        gaps = np.diff(planned)
        tracker = self.env.frequency_tracker
        offsets = ((planned[:-1]+planned[1:])/2-now)/tracker.bin_interval_s
        demand = np.asarray([max(float(tracker.global_state.forecast(float(t))), 1.) for t in offsets])
        wait = float(np.sum(demand*gaps**2)/(2*np.sum(demand*gaps))/600)
        self._candidate_audits.append({"next_trip_id": int(next_trip.launch_turn) if next_trip is not None else None,
            "window_end_s": float(end_s), "prefix_planned_s": planned.tolist(),
            "prefix_peak_counts": counts, "prefix_fleet_cost": fleet_cost(counts, self.env._n_fleet_target),
            "prefix_wait_cost": wait, "prefix_score": wait+fleet_cost(counts, self.env._n_fleet_target),
            "full_fleet_cost": original_cost,
            "proposed_planned_s": [float(planned_by_id[int(t.launch_turn)]) for t in self._audit_remaining]})
        return original_cost, original_diagnostics
