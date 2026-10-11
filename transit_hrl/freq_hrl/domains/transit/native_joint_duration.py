"""Empirical trip-path dependence with exactly unchanged duration marginals."""

import numpy as np

from freq_hrl.domains.transit.native_distributional_supply_plan import residual_draws


def joint_residual_draws(model, trips):
    trips = sorted(trips)
    if trips != list(zip(model["joint_trip_ids"], model["joint_directions"])):
        raise ValueError("Joint duration paths must match native trip identities/directions")
    paths = np.asarray(model["joint_fit_residual_paths_s"], dtype=float)
    if paths.shape != (len(model["fit_scene_seeds"]), len(trips)) or not np.all(np.isfinite(paths)):
        raise ValueError("Incomplete paired-direction fitting paths")
    independent = residual_draws(model, trips)
    rng = np.random.default_rng(170000 + model["root"])
    templates = rng.integers(len(paths), size=model["members"])
    # A common tie order keeps repeated draws of the same fit path coherent.
    ties = rng.random(model["members"])
    joint = {}
    for column, (tid, _) in enumerate(trips):
        order = np.lexsort((ties, paths[templates, column]))
        values = np.empty(model["members"])
        values[order] = np.sort(independent[tid])
        joint[tid] = values
    return joint


def marginal_reproduction(model, trips):
    independent, joint = residual_draws(model, trips), joint_residual_draws(model, trips)
    return all(np.array_equal(np.sort(independent[t]), np.sort(joint[t])) for t in independent)
