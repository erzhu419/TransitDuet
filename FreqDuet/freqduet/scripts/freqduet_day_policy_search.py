"""Constrained policy-space learning from complete, nonlinear operating days."""

import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution


ACTIONS = np.asarray((0., 5., 10., 15., 20., 30., 45.))
FEATURES = ('bias', 'load', 'progress', 'return_ready', 'return_overdue', 'gap_imbalance')
CONSTRAINTS = ('cv_squared', 'readiness_delay_s', 'public_launch_delay_total_s',
               'trips_unlaunched', 'trips_incomplete')
POPULATION, GENERATIONS, SEARCH_SEED = 8, 1, 268
MAX_VECTORS = POPULATION * (GENERATIONS + 1)
FAMILIES = ('conditional', 'uniform')


def coefficients(values, family):
    values = np.asarray(values, dtype=float)
    assert family in FAMILIES
    assert values.shape == (len(FEATURES) if family == 'conditional' else 1,)
    return values if family == 'conditional' else np.r_[values, np.zeros(len(FEATURES) - 1)]


def observation(context, load, progress, ready, due):
    imbalance = ((context.follower_departure_gap_s - context.forward_headway_s) /
                 (2. * context.target_headway_s)) if context.evidence_valid and context.follower_evidence_valid else 0.
    return np.asarray((1., np.clip(load, 0., 1.) - .5, np.clip(progress, 0., 1.) - .5,
                       ready / (1. + ready) - .5, due / (1. + due + ready),
                       np.clip(imbalance, -1., 1.)))


def choose(context, features, theta, feasible):
    if not context.evidence_valid:
        desired = 0.
    else:
        base = (.25 * (context.follower_departure_gap_s - context.forward_headway_s) / 2.
                if context.follower_evidence_valid else
                .25 * (context.target_headway_s - context.forward_headway_s))
        # Signed redistribution is available; cumulative holding is not a slack budget.
        desired = np.clip(base, 0., ACTIONS[-1]) + ACTIONS[-1] * np.tanh(np.asarray(features) @ theta)
        desired = np.clip(desired, 0., ACTIONS[-1])
    allowed = np.flatnonzero(feasible)
    assert len(allowed)
    return int(allowed[np.argmin(np.abs(ACTIONS[allowed] - desired))])


def costs(rows):
    return np.asarray([[r['passenger_person_s'], *[r[m] for m in CONSTRAINTS]] for r in rows], dtype=float)


def search(evaluate, family):
    """Every constraint call uses realized full-day outcomes, never a tangent/Q."""
    dimension = len(FEATURES) if family == 'conditional' else 1
    rng = np.random.default_rng(SEARCH_SEED)
    initial = rng.uniform(-1., 1., (POPULATION, len(FEATURES)))[:, :dimension]
    initial[0] = 0.
    cache = {}

    def measured(theta):
        key = tuple(float(v) for v in theta)
        if key not in cache:
            assert len(cache) < MAX_VECTORS
            cache[key] = costs(evaluate(np.asarray(theta), len(cache)))
        return cache[key]

    zero = np.zeros(dimension)
    reference = measured(zero)
    scale = np.maximum(np.abs(reference.mean(0)), 1.)

    def changes(theta):
        return (measured(theta) - reference).mean(0) / scale

    result = differential_evolution(lambda x: changes(x)[0], [(-1., 1.)] * dimension,
        constraints=(NonlinearConstraint(lambda x: changes(x)[1:], -np.inf, 0.),),
        init=initial, maxiter=GENERATIONS, polish=False, seed=SEARCH_SEED,
        updating='deferred', tol=0., atol=0.)
    # The explicit baseline is feasible. Budget exhaustion is not convergence.
    assert np.all(changes(result.x)[1:] <= 0.)
    selected = result.x if changes(result.x)[0] < 0. else zero
    effects = measured(selected) - reference
    return dict(family=family, theta=coefficients(selected, family).tolist(),
        optimizer_parameters=selected.tolist(), measured_vectors=len(cache),
        parameter_budget=MAX_VECTORS, solver_success=bool(result.success),
        solver_message=str(result.message), generations=int(result.nit),
        mean_cost_changes=effects.mean(0).tolist(), day_cost_changes=effects.tolist(),
        cost_names=['passenger_person_s', *CONSTRAINTS],
        update_identified=bool(np.any(selected)),
        per_day_joint_improvement=bool(np.all(effects[:, 0] < 0.) and np.all(effects[:, 1:] <= 0.)),
        objective='Mean once-only full-day waiting plus onboard person-seconds',
        constraint_type='Realized complete-day mean changes, not gradient/predicted constraints')
