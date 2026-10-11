"""Include a fixed-window forecast primitive in the rolling candidate pool."""

import numpy as np

from freq_hrl.core.time_allocation import budgeted_time_points
from freq_hrl.domains.transit.native_service_plan import NativeServicePlan
from freq_hrl.domains.transit.native_commitment_plan import NativeCommitmentScoringPlan


class NativeIncumbentWindowPlan(NativeCommitmentScoringPlan):
    def __init__(self, env, model, condition):
        if condition not in {"incumbent_only", "rolling_plus_incumbent"}:
            raise ValueError(condition)
        self.proposal_condition = condition
        super().__init__(env, model, "prefix_wait_full_fleet")

    def candidate_preferences(self, nominal, trip, now, remaining, forecast):
        count = min(self.block_size, len(nominal))
        prefix = nominal[:count]
        preferred, _ = NativeServicePlan.preferred_intervals(self, prefix, trip, now)
        planned = nominal.copy()
        planned[:count] = budgeted_time_points(prefix, preferred, minimum=240, maximum=480)
        incumbent = (None, np.diff(planned))
        # Keep the next anchor and tentative tail; only internal window times move.
        return ([incumbent] if self.proposal_condition == "incumbent_only" else
            [*super().candidate_preferences(nominal, trip, now, remaining, forecast), incumbent])
