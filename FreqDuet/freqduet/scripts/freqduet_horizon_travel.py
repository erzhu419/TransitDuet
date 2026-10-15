"""Completed-prefix convex mean reversion by forecast departure lead."""

import numpy as np
from scipy.optimize import lsq_linear

from freqduet_conditioned_travel import ConditionedTravelPath
from freqduet_travel_distribution import HISTORY_S


def horizon_basis(lead):
    late = np.clip((np.asarray(lead) - 450.) / 900., 0., 1.)
    return np.stack((1. - late, late), axis=-1)


def fit_reversion(rows, now):
    past = rows.loc[(rows.end_s < now) & (rows.start_s >= max(0, now - HISTORY_S))].sort_values("end_s").tail(4096)
    if (past.origin >= past.start_s).any():
        raise ValueError("mean-reversion training prediction was not issued before departure")
    model = dict(beta=[0., 0.], samples=len(past), latest_completion_s=float(past.end_s.max()) if len(past) else None)
    if not len(past):
        return model
    x = horizon_basis(past.start_s.to_numpy() - past.origin.to_numpy()) * (
        past.historical.to_numpy() - past.current.to_numpy())[:, None]
    if not np.any(x):
        return model
    result = lsq_linear(x, past.actual.to_numpy() - past.current.to_numpy(), bounds=(0., 1.), tol=1e-10)
    if not result.success:
        raise ValueError(f"mean-reversion fit did not converge: {result.message}")
    model["beta"] = result.x.tolist()
    return model


def reverted_mean(current, historical, lead, beta):
    weight = horizon_basis(lead) @ np.asarray(beta)
    return current + weight * (historical - current)


class HorizonTravelPath(ConditionedTravelPath):
    def mean(self, key, time):
        return float(reverted_mean(self.model["means"][key], self.model["historical_means"][key],
            time - self.model["now_s"], self.model["reversion"]["beta"]))
