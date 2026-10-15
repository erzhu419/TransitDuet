"""Completed-prefix, partially pooled road/time travel-mean persistence."""

import numpy as np
from scipy.optimize import lsq_linear

from freqduet_conditioned_travel import ConditionedTravelPath
from freqduet_horizon_travel import fit_reversion, horizon_basis, reverted_mean
from freqduet_travel_distribution import HISTORY_S


PRIOR_PAIRS = 10.


def time_context(origin, departure):
    return f"{int(origin % 3600 // 900)}:{int(departure // 3600 > origin // 3600)}"


def pooled_fit(rows, parent):
    x = horizon_basis(rows.start_s.to_numpy() - rows.origin.to_numpy()) * (
        rows.historical.to_numpy() - rows.current.to_numpy())[:, None]
    y = rows.actual.to_numpy() - rows.current.to_numpy()
    if not np.any(x):
        return list(parent)
    # Ten parent-centred pseudo-pairs, scaled to this group's actual leverage.
    active = np.any(x != 0, axis=0)
    penalty = np.sqrt(PRIOR_PAIRS * np.mean(x[:, active] ** 2, axis=0))
    design = np.vstack((x[:, active], np.diag(penalty)))
    labels = np.r_[y, penalty * np.asarray(parent)[active]]
    result = lsq_linear(design, labels, bounds=(0., 1.), tol=1e-10)
    if not result.success:
        raise ValueError(f"conditional persistence did not converge: {result.message}")
    beta = np.asarray(parent, dtype=float).copy()
    beta[active] = result.x
    return beta.tolist()


def fit_conditional_persistence(records, now):
    past = records.loc[(records.end_s < now) & (records.start_s >= max(0, now - HISTORY_S))].sort_values("end_s").tail(4096).copy()
    if (past.origin >= past.start_s).any():
        raise ValueError("conditional training prediction was not issued before departure")
    parent = fit_reversion(records, now)
    model = dict(global_beta=parent['beta'], roads={}, contexts={}, samples=len(past),
        latest_completion_s=parent['latest_completion_s'], prior_pairs=PRIOR_PAIRS)
    if not len(past):
        return model
    past['context'] = [time_context(o, t) for o, t in zip(past.origin, past.start_s)]
    for road, group in past.groupby('road'):
        beta = pooled_fit(group, parent['beta'])
        model['roads'][road] = dict(beta=beta, samples=len(group))
        for context, subset in group.groupby('context'):
            model['contexts'][road + '/' + context] = dict(beta=pooled_fit(subset, beta), samples=len(subset))
    return model


def conditional_beta(model, road, origin, departure):
    parent = model['roads'].get(road, dict(beta=model['global_beta']))['beta']
    return model['contexts'].get(road + '/' + time_context(origin, departure), dict(beta=parent))['beta']


def conditional_mean(model, road, current, historical, origin, departure):
    return float(reverted_mean(current, historical, departure - origin,
        conditional_beta(model, road, origin, departure)))


class ConditionalPersistencePath(ConditionedTravelPath):
    def mean(self, key, time):
        return conditional_mean(self.model['persistence'], key, self.model['means'][key],
            self.model['historical_means'][key], self.model['now_s'], time)
