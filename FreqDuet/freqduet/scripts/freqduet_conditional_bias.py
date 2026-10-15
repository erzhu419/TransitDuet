"""Additive road/time bias from completed, previously issued V54 predictions."""

import numpy as np

from freqduet_conditional_persistence import (PRIOR_PAIRS, ConditionalPersistencePath,
    conditional_mean, time_context)
from freqduet_horizon_travel import horizon_basis
from freqduet_travel_distribution import HISTORY_S


def attach_issued_means(records, models):
    rows = records.copy()
    predictions = []
    for row in rows.itertuples():
        model = models[str(row.origin)]
        latest = model['persistence']['latest_completion_s']
        if model['now_s'] != row.origin or latest is not None and latest >= row.origin:
            raise ValueError('residual source is not an issued prefix model')
        predictions.append(conditional_mean(model['persistence'], row.road, row.current,
            row.historical, row.origin, row.start_s))
    rows['issued_mean'] = predictions
    return rows


def bias_coefficients(rows, parent=None):
    x = horizon_basis(rows.start_s.to_numpy() - rows.origin.to_numpy())
    y = rows.actual.to_numpy() - rows.issued_mean.to_numpy()
    if parent is None:
        return np.linalg.lstsq(x, y, rcond=None)[0].tolist()
    active = np.any(x != 0, axis=0)
    x = x[:, active]
    penalty = PRIOR_PAIRS * np.mean(x ** 2, axis=0)
    coef = np.asarray(parent, dtype=float).copy()
    coef[active] = np.linalg.solve(x.T @ x + np.diag(penalty),
        x.T @ y + penalty * coef[active])
    return coef.tolist()


def fit_completed_bias(records, now):
    past = records.loc[(records.end_s < now) & (records.start_s >= max(0, now - HISTORY_S))].sort_values('end_s').tail(4096).copy()
    if (past.origin >= past.start_s).any():
        raise ValueError('bias label was not forecast before departure')
    model = dict(global_bias=[0., 0.], roads={}, contexts={}, samples=len(past),
        latest_completion_s=float(past.end_s.max()) if len(past) else None,
        latest_issue_s=float(past.origin.max()) if len(past) else None, prior_pairs=PRIOR_PAIRS)
    if not len(past):
        return model
    parent = bias_coefficients(past)
    model['global_bias'] = parent
    past['context'] = [time_context(o, t) for o, t in zip(past.origin, past.start_s)]
    for road, group in past.groupby('road'):
        coef = bias_coefficients(group, parent)
        model['roads'][road] = dict(coef=coef, samples=len(group))
        for context, subset in group.groupby('context'):
            model['contexts'][road + '/' + context] = dict(coef=bias_coefficients(subset, coef), samples=len(subset))
    return model


def predicted_bias(model, road, origin, departure):
    parent = model['roads'].get(road, dict(coef=model['global_bias']))['coef']
    coef = model['contexts'].get(road + '/' + time_context(origin, departure), dict(coef=parent))['coef']
    return float(horizon_basis(departure - origin) @ np.asarray(coef))


def calibrated_mean(base_mean, calibration, road, origin, departure, dt):
    # The same positive full-leg duration convention as the travel sampler.
    return max(dt, base_mean + predicted_bias(calibration, road, origin, departure))


class BiasCalibratedTravelPath(ConditionalPersistencePath):
    def mean(self, key, time):
        return calibrated_mean(super().mean(key, time), self.model['bias_calibration'],
            key, self.model['now_s'], time, self.model['dt_s'])
