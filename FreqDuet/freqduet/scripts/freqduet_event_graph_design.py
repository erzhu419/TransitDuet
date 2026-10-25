"""Finite event-graph certificate, not a simulator controller or fitted policy."""

import numpy as np


ACTIONS = np.array((0, 5, 10, 15, 20, 30, 45), dtype=np.int64)
CASES = ('alighting', 'through', 'saturated', 'committed')


def fixture(name):
    assert name in CASES
    baseline = np.array((30, 20, 10, *([45] * 3 if name == 'saturated' else [10] * 3)))
    prepare = np.array((0, 360, 720))
    travel = np.array((100, 110, 120))
    service = 4 if name == 'through' else 44
    b = prepare + baseline[:3] + travel + service + baseline[3:] + 100
    return dict(name=name, baseline=baseline, prepare=prepare, travel=travel, service=service,
        load_before=np.full(3, 40.), load_after=np.full(3, 40. if name == 'through' else 0.),
        prefix=np.array((-360, -240, b[0] - 360)), suffix=np.array((1080, 1210, b[-1] + 360)),
        published=b + 60, return_tail=np.array((59, 69, 79)), lock_before=name == 'committed',
        waiting_rates=np.array((0., 0., 1. / 60.)))


def model(actions, spec):
    h = np.atleast_2d(actions)
    assert h.shape[1] == 6 and np.isin(h, ACTIONS).all()
    h = h.astype(np.int64)
    a = spec['prepare'] + h[:, :3] + spec['travel']
    b = a + spec['service'] + h[:, 3:] + 100
    ready = b + spec['return_tail']
    # Equality to the physical earliest dispatch prevents hidden terminal waiting.
    launch = np.maximum(spec['published'], ready + 1)
    arrival_sets = (np.broadcast_to(spec['prepare'], a.shape), a, b)
    gaps = []
    for station, times in enumerate(arrival_sets):
        assert np.all(times > spec['prefix'][station]) and np.all(times < spec['suffix'][station])
        gaps.append(np.diff(np.column_stack((np.full(len(h), spec['prefix'][station]),
            np.sort(times, axis=1), np.full(len(h), spec['suffix'][station]))), axis=1))
    pooled = np.concatenate(gaps, axis=1)
    span = pooled.sum(1)
    numerator = pooled.shape[1] * np.square(pooled).sum(1) - np.square(span)
    waiting = sum(rate * np.square(g).sum(1) / 2. for rate, g in zip(spec['waiting_rates'], gaps))
    onboard = h[:, :3] @ spec['load_before'] + h[:, 3:] @ spec['load_after']
    delay = (launch - spec['published']).sum(1)
    return dict(arrival_a=a, arrival_b=b, ready=ready, launch=launch, gaps=gaps,
        holding_by_vehicle=h[:, :3] + h[:, 3:], cv_numerator=numerator,
        cv_denominator=np.square(span), model_cv_squared=numerator / np.square(span),
        model_waiting_person_s=waiting, model_onboard_variable_person_s=onboard,
        model_passenger_variable_person_s=waiting + onboard,
        model_readiness_mean_s=delay / 3., model_public_delay_total_s=delay)


def admissible(actions, spec, matched_amount):
    h = np.atleast_2d(actions)
    values, ref = model(h, spec), model(spec['baseline'], spec)
    feasible = (values['cv_numerator'] * ref['cv_denominator'][0] <=
                ref['cv_numerator'][0] * values['cv_denominator'])
    feasible &= values['model_public_delay_total_s'] <= ref['model_public_delay_total_s'][0]
    if matched_amount:
        feasible &= np.all(values['holding_by_vehicle'] == ref['holding_by_vehicle'][0], axis=1)
    if spec['lock_before']:
        feasible &= np.all(h[:, :3] == spec['baseline'][:3], axis=1)
    return feasible, values, ref


def global_offsets(baseline):
    breaks = np.unique(np.r_[-45., 0., 45.,
        (((ACTIONS[:-1] + ACTIONS[1:]) / 2.)[:, None] - baseline).ravel()])
    breaks = breaks[(breaks >= -45.) & (breaks <= 45.)]
    offsets = np.r_[breaks, (breaks[:-1] + breaks[1:]) / 2.]
    desired = np.clip(baseline + offsets[:, None], 0., 45.)
    selected = np.argmin(np.abs(desired[:, :, None] - ACTIONS), axis=2)
    return np.unique(ACTIONS[selected], axis=0)


def certificate(spec):
    indices = np.indices((len(ACTIONS),) * 6).reshape(6, -1).T
    allocations = ACTIONS[indices]
    feasible, values, ref = admissible(allocations, spec, matched_amount=True)
    possible = np.flatnonzero(feasible)
    assert len(possible)
    index = possible[np.argmin(values['model_passenger_variable_person_s'][possible])]
    strict = values['model_passenger_variable_person_s'][index] < ref['model_passenger_variable_person_s'][0]
    chosen = allocations[index] if strict else spec['baseline']
    selected = model(chosen, spec)
    uniform = global_offsets(spec['baseline'])
    allowed, uniform_values, _ = admissible(uniform, spec, matched_amount=False)
    uniform_better = allowed & (uniform_values['model_passenger_variable_person_s'] < ref['model_passenger_variable_person_s'][0])
    metrics = ('model_passenger_variable_person_s', 'model_waiting_person_s',
               'model_onboard_variable_person_s', 'model_cv_squared',
               'model_readiness_mean_s', 'model_public_delay_total_s')
    return dict(case=spec['name'], baseline_actions=spec['baseline'].tolist(),
        selected_actions=chosen.tolist(), allocations_examined=len(allocations),
        feasible_matched_amount_allocations=len(possible), strict_improvement=bool(strict),
        model_effects={m: float(selected[m][0] - ref[m][0]) for m in metrics},
        per_vehicle_holding_before=ref['holding_by_vehicle'][0].tolist(),
        per_vehicle_holding_after=selected['holding_by_vehicle'][0].tolist(),
        model_return_times_before=ref['ready'][0].tolist(), model_return_times_after=selected['ready'][0].tolist(),
        model_launch_times_before=ref['launch'][0].tolist(), model_launch_times_after=selected['launch'][0].tolist(),
        universal_offset_aliases=len(uniform), universal_offset_joint_improvements=int(uniform_better.sum()))
