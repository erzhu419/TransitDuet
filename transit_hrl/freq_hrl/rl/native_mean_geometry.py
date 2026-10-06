"""Damped empirical-Fisher directions for a linear Gaussian policy mean."""

import numpy as np


def native_mean_directions(states, signals, standard_deviation, *, damping):
    states = np.asarray(states, dtype=np.float64)
    mean, scale = states.mean(0), states.std(0)
    active = scale > 0
    width = int(active.sum())
    normalized = (states[:, active] - mean[active]) / (scale[active] * np.sqrt(width)) if width else states[:, :0]
    design = np.c_[normalized, np.ones(len(states))]
    std = np.asarray(standard_deviation, dtype=np.float64)
    targets = np.concatenate([np.asarray(g, dtype=np.float64) * std ** 2 for g in signals.values()], axis=1)
    # Solve in sample space: native query batches have far fewer rows than parameters.
    dual = np.linalg.solve(design @ design.T + damping * np.eye(len(states)), targets)
    coefficients = design.T @ dual
    weight = np.zeros((states.shape[1], targets.shape[1]))
    if width:
        weight[active] = coefficients[:-1] / (scale[active, None] * np.sqrt(width))
    bias = coefficients[-1] - mean @ weight
    directions = {}
    for index, panel in enumerate(signals):
        columns = slice(index * len(std), (index + 1) * len(std))
        directions[panel] = {"weight": weight[:, columns].T, "bias": bias[columns]}
    return directions, {"training_rows": len(states), "state_dimensions": states.shape[1],
        "active_state_dimensions": width, "design_rank": int(np.linalg.matrix_rank(design)),
        "damping": damping}
