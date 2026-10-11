"""Isolate waiting and fleet score horizons with unchanged feasible candidates."""

from freq_hrl.domains.transit.native_commitment_audit import NativeCommitmentAudit

HORIZONS = {
    "full_wait_prefix_fleet": (False, True),
    "prefix_wait_full_fleet": (True, False),
    "prefix_wait_prefix_fleet": (True, True),
}


class NativeCommitmentScoringPlan(NativeCommitmentAudit):
    def __init__(self, env, model, condition):
        self.prefix_wait, self.prefix_fleet = HORIZONS[condition]
        super().__init__(env, model)

    def candidate_score(self, wait_cost, fleet_cost, diagnostics):
        prefix = self._candidate_audits[-1]
        diagnostics["commitment"] = {k: v for k, v in prefix.items() if k != "proposed_planned_s"}
        return ((prefix["prefix_wait_cost"] if self.prefix_wait else wait_cost)
            + (prefix["prefix_fleet_cost"] if self.prefix_fleet else fleet_cost))
