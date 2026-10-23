"""Frozen budget and same-reserved-vehicle execution gate for V220 projection."""

from freqduet_v219_service_task import CONTEXTS


ARMS = ('v107', 'upper', 'lower', 'joint')


def registration(source):
    return dict(stage='v221_ready_backed_release_execution_smoke', source=source,
        contexts=CONTEXTS, arms=ARMS, max_started_paths=16,
        stages=dict(reference=4, upper=4, lower_after_gate=4, joint_after_gate=4),
        upper='Frozen V220 supply projection of native+30 raw preview, clipped[-60,60], EMA.35. '
            'Keep ORIGINAL native actor commands and direction-local EMA cache in every arm; '
            'the +30 is a shadow projection request,not a replacement cached command. '
            'Only the projection geometry changes. Keep native fresh/cached cadence and anchors. '
            'No coefficient escalation, synthetic ready stock or extra fleet.',
        gate='First reproduce all four complete V216 reference traces/endpoints. Then run four '
            'upper-only paths. In EACH context the FIRST admitted query must match its reference '
            'query exactly. EVERY task at that query must launch on its own reserved bus in BOTH '
            'paths,with at least one tick of actual advance and a strictly earlier executed plan. '
            'Require unchanged physical contracts and fully cleared service. Otherwise stop before '
            'lower/joint;do not replace contexts or search a later successful admission.',
        lower='Frozen V219 own-outbound-trip release+executed-holding compensation law. '
            'Create tasks from the same live V220 shadow admission in all arms,even lower-only '
            'without applying the projection. Activate only after actual launch on the reserved bus. '
            'Reduce one feasible bin only when squared completed-predecessor release/holding gap '
            'error does not increase;missing evidence keeps bilateral025. Follow same trip/bus '
            'until return,not a new cycle or other vehicle.',
        outcomes='Report all four arms and joint minus each control: passenger person-seconds and '
            'journey,actual CV2,vehicle readiness,original-public total launch delay including '
            'control waiting,holding,waiting/onboard,early departure and completion. '
            'Audit issued versus executed lower changes and joint versus lower actual launches. '
            'Descriptive joint improvement requires passenger reduction without CV2/readiness/'
            'public-delay/completion harm in EACH fixed context;not independent efficacy.',
        accounting='One full-day attempt per fixed context/arm,including failures;no automatic '
            'replacement or retry. Reuse completed path files on engineering recovery. Tests use '
            'no rollout budget. No training,new scenarios,new future sampling or policy promotion.',
        frozen='Original12-vehicle fixed pool,seven holding bins,native scalar bounds and cadence; '
            'V151,V146,V167 and other frozen predictors/calibrations/filters/actors/critics/duals unchanged.',
        execution='scheduler high,manual dispatch,node001--006;only small summaries return locally',
        model_updates=0, independent_futures_opened=False, main_policy_changed=False)


def task_execution(task, reference_launch, candidate_launch, reference_plan, candidate_plan, tick_s):
    present = reference_launch is not None and candidate_launch is not None
    matched = present and reference_launch['bus'] == candidate_launch['bus'] == task['reserved_bus_id']
    advance = None if not present else reference_launch['actual_s']-candidate_launch['actual_s']
    plan_advance = None if reference_plan is None or candidate_plan is None else reference_plan-candidate_plan
    return dict(trip_id=task['trip_id'], reserved_bus_id=task['reserved_bus_id'],
        reference_bus_id=None if reference_launch is None else reference_launch['bus'],
        candidate_bus_id=None if candidate_launch is None else candidate_launch['bus'],
        actual_advance_s=advance, executed_plan_advance_s=plan_advance,
        native_plan_preserved=reference_plan == task['native_s'],
        proposed_plan_executed=candidate_plan == task['proposed_s'],
        passed=bool(matched and advance >= tick_s and plan_advance is not None and plan_advance > 0
                    and reference_plan == task['native_s'] and candidate_plan == task['proposed_s']))


def overwrite_curve(planner, query, timetables, native, solution, summary):
    """Materialize the jointly solved curve, including adjacent target metadata."""
    rows = sorted((r for r in query['plans'] if r['direction'] == query['direction']
        and -1e-6 <= r['public_s']-next(p['public_s'] for p in query['plans']
            if p['trip_id'] == query['trip_id']) <= planner.horizon_s), key=lambda r: r['public_s'])
    by_id = {t.launch_turn: t for t in timetables}
    previous = solution[rows[0]['trip_id']]['planned_s']
    for row in rows[1:]:
        trip = row['trip_id']
        projected = solution[trip]
        if projected != native[trip]:
            t = by_id[trip]
            t._freqduet_scheduled_launch = projected['planned_s']
            t.target_headway = t._freqduet_projected_target_headway = projected['target_s']
            t._freqduet_desired_headway_s = projected['target_s']
            t._freqduet_predecessor_scheduled_launch = previous
            t._freqduet_phase_displacement_s = projected['planned_s']-row['public_s']
            t._freqduet_projection_mode = 'v220_supply_constrained_curve'
        previous = projected['planned_s']
    import numpy as np
    targets = np.array([solution[r['trip_id']]['target_s'] for r in rows])
    clocks = np.array([solution[r['trip_id']]['planned_s'] for r in rows])
    deltas = np.array([solution[r['trip_id']]['target_s']-planner._base_headway(by_id[r['trip_id']])
                       for r in rows[1:]])
    phases = clocks-np.array([r['public_s'] for r in rows])
    result = dict(summary, planned_mean=float(targets.mean()), planned_std=float(targets.std()),
        scheduled_mean=float(clocks.mean()), scheduled_std=float(clocks.std()),
        phase_displacement_mean_s=float(phases.mean()), phase_displacement_max_abs_s=float(np.max(np.abs(phases))),
        projected_headway_delta_sum_s=float(deltas.sum()),
        projected_headway_delta_mean_s=float(deltas.mean()) if deltas.size else 0.,
        projection_mode='v220_supply_constrained_curve')
    return result
