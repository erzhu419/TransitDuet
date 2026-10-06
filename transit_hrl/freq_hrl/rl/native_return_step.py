"""Fit a bounded policy step from paired native returns, not surrogate KL."""

import numpy as np


def fit_native_return_step(zero, plus, minus):
    zero, plus, minus = (np.asarray(x, dtype=np.float64) for x in (zero, plus, minus))
    slope = float(np.mean((plus - minus) / 2))
    curvature = float(np.mean((plus + minus) / 2 - zero))
    if curvature < 0:
        scale = float(np.clip(-slope / (2 * curvature), 0., 1.))
    else:
        scale = float(slope + curvature > 0.)
    return {"scale": scale, "slope": slope, "quadratic_coefficient": curvature,
        "predicted_gain": slope * scale + curvature * scale ** 2,
        "unit_step_gain": slope + curvature, "paired_samples": len(zero)}
