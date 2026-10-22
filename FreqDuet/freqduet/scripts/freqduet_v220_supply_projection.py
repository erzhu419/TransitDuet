"""Joint headway projection with query-time, already-ready departure support."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from freqduet_v219_service_task import ready_prefix


def block_ends(offsets, window_s):
    ends, origin = [], 0.
    for i, offset in enumerate(offsets):
        if offset-origin >= window_s-1e-9 or i == len(offsets)-1:
            ends.append(i)
            origin = offset
    return ends


def project(planner, query, trips, native, desired):
    """Project desired headways, not independently masked departure times.

    Variables are cumulative clock shifts at ready-backed non-closure slots.
    All other clock shifts are exactly zero by construction. Difference rows
    constrain adjacent headways; fixed block endpoints conserve native budgets.
    """
    resource = ready_prefix(query)
    by_id = {t.launch_turn: t for t in trips}
    selected = sorted((r for r in query['plans'] if r['direction'] == query['direction']
        and -1e-6 <= r['public_s']-resource['origin_s'] <= planner.horizon_s),
        key=lambda r: (r['public_s'], r['trip_id']))
    anchor, future = selected[0], selected[1:]
    ids = [r['trip_id'] for r in future]
    offsets = [r['public_s']-resource['origin_s'] for r in future]
    ends = block_ends(offsets, planner.headway_budget_window_s)
    clocks = np.array([native[anchor['trip_id']]['planned_s']]+[native[t]['planned_s'] for t in ids])
    headways = np.diff(clocks)
    target = np.array([desired[t]['target_s'] for t in ids])
    bases = np.array([planner._base_headway(by_id[t]) for t in ids])
    lower = np.maximum(planner.min_headway_s, bases+planner.delta_min_s)
    upper = np.minimum(planner.max_headway_s, bases+planner.delta_max_s)
    assert np.all(headways >= lower-1e-6) and np.all(headways <= upper+1e-6)
    variable = [i for i, t in enumerate(ids) if i not in ends
        and t in resource['mutable_ready_trip_ids'] and clocks[i+1] > query['time_s']]
    # Each column changes one cumulative departure clock, hence two intervals.
    cumulative = np.zeros((len(ids)+1, len(variable)))
    for j, i in enumerate(variable):
        cumulative[i+1, j] = 1.
    difference = np.diff(cumulative, axis=0)
    shift = np.zeros(len(variable))
    iterations = 0
    if variable:
        error = headways-target
        objective = lambda x: .5*float(np.dot(error+difference@x, error+difference@x))
        gradient = lambda x: difference.T@(error+difference@x)
        fit = minimize(objective, shift, jac=gradient, method='SLSQP',
            bounds=Bounds([query['time_s']-clocks[i+1] for i in variable], np.zeros(len(variable))),
            constraints=LinearConstraint(difference, lower-headways, upper-headways),
            options=dict(ftol=1e-10, maxiter=100))
        assert fit.success, ('supply projection failed', fit.message)
        shift, iterations = fit.x, int(fit.nit)
        assert np.all(shift <= 0.), 'solver returned a later clock'
    new_clocks = clocks+cumulative@shift
    new_headways = np.diff(new_clocks)
    assert np.all(new_headways >= lower-1e-6) and np.all(new_headways <= upper+1e-6)
    assert np.all(new_clocks[1:][variable] >= query['time_s']-1e-6)
    output = {t: dict(r) for t, r in native.items()}
    for i, t in enumerate(ids):
        output[t]['planned_s'] = float(new_clocks[i+1])
        if new_clocks[i] != clocks[i] or new_clocks[i+1] != clocks[i+1]:
            output[t]['target_s'] = float(new_headways[i])
    residuals, start = [], 0
    for end in ends:
        residuals.append(float(np.sum(new_headways[start:end+1]-bases[start:end+1])))
        assert abs(residuals[-1]) <= 1e-6, 'rolling budget was not conserved'
        start = end+1
    eligible = {ids[i] for i in variable}
    fixed = [t for t in native if t not in eligible]
    assert all(output[t]['planned_s'] == native[t]['planned_s'] for t in fixed)
    assert all(output[t] == native[t] for t in native if by_id[t].direction != query['direction'])
    before = float(np.sum((headways-target)**2))
    after = float(np.sum((new_headways-target)**2))
    assert after <= before+1e-6, 'projection worsened its own objective'
    return output, dict(variable_trip_ids=[ids[i] for i in variable],
        ready_bus_ids=[resource['ready_bus_by_trip'][ids[i]] for i in variable],
        closure_trip_ids=[ids[i] for i in ends], rolling_budget_residual_s=residuals,
        objective_before=before, objective_after=after, solver_iterations=iterations,
        fixed_clocks_exact=True, other_direction_exact=True,
        max_headway_error_s=max((abs(output[t]['target_s']-new_headways[i])
                                 for i, t in enumerate(ids)), default=0.))
