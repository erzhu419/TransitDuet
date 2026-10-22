#!/usr/bin/env python3
"""Static V220 supply projection on four archived native trajectories; no rollout."""

import argparse
from collections import Counter
import gzip
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import freqduet_v219_projection as preview
import freqduet_v219_service_task as law
import freqduet_v220_supply_projection as supply


def audit(data, planner, alpha, context):
    trace, capture = data['trace'], data['capture']
    callbacks = {(r['time_s'], r['trip_id']): r for r in trace['upper_callbacks'] if r['commands']}
    launches = {e['trip_id']: e['event_seq'] for e in capture['events'] if e['kind'] == 'launch'}
    previous, rows, checked, max_native_error = {}, [], 0, 0.
    tick = data['result']['physical_contract']['dt_s']
    for query in capture['fresh_upper']:
        assert query['snapshot_event_seq'] == query['event_seq']
        assert query['snapshot_read_s'] == query['time_s']
        assert all(not b['return_observed'] or b['last_observed_return_s'] <= query['time_s']
                   for b in query['ready'])
        observed = [r for r in trace['actual_launches'] if launches[r['trip']] < query['event_seq']]
        trips = preview.query_timetable(query, observed)
        immutable = [vars(t).copy() for t in trips]
        raw = np.asarray(query['native_coefficient'], dtype=np.float32)
        direction = bool(query['direction'])
        effective = preview.ema_action(raw, previous.get(direction), alpha)
        native = preview.preview(planner, trips, query, effective)
        callback = callbacks[query['time_s'], query['trip_id']]
        assert callback['commands'][0]['coefficient'] == query['native_coefficient']
        assert all(r['after']['direction'] == direction for r in callback['changes'])
        expected = {r['trip_id']: dict(planned_s=r['planned_s'], target_s=r['target_s']) for r in query['plans']}
        expected.update({r['trip_id']: {k: r['after'][k] for k in ('planned_s', 'target_s')}
                         for r in callback['changes']})
        max_native_error = max(max_native_error, preview.verify_native(native, expected, tick))
        checked += 1
        if query['time_s'] >= law.START_S:
            identity, _ = supply.project(planner, query, trips, native, native)
            preview.verify_native(identity, native, tick)
            request = np.clip(raw+30., planner.action_low, planner.action_high).astype(np.float32)
            desired = preview.preview(planner, trips, query,
                preview.ema_action(request, previous.get(direction), alpha))
            raw_admission = preview.inspect_proposal(query, native, desired, tick)
            proposed, diagnostics = supply.project(planner, query, trips, native, desired)
            evidence = preview.inspect_proposal(query, native, proposed, tick)
            assert not evidence['continuous_later_trip_ids'] and not evidence['clock_later_trip_ids']
            assert not evidence['unfunded_within_cycle_trip_ids'] and not evidence['outside_cycle_advanced']
            assert diagnostics['max_headway_error_s'] <= 1e-6
            rows.append(dict(time_s=query['time_s'], event_seq=query['event_seq'], trip_id=query['trip_id'],
                native_effective=effective.tolist(), raw_native_curve_supported=raw_admission['supported'],
                raw_native_curve_rejection=raw_admission['rejection'], projection=diagnostics, **evidence))
        assert immutable == [vars(t) for t in trips], 'static projection mutated query state'
        previous[direction] = effective
    accepted = [r for r in rows if r['supported']]
    return dict(context=context, actor=law.CONTEXTS[context][0], day=law.CONTEXTS[context][1],
        native_snapshots_checked=checked, max_native_error_s=max_native_error,
        candidate_queries=len(rows), nonempty_solver_queries=sum(bool(r['projection']['variable_trip_ids']) for r in rows),
        raw_native_curve_admitted=sum(r['raw_native_curve_supported'] for r in rows),
        admitted_queries=len(accepted), admitted_trip_ids=sorted({t for r in accepted for t in r['law_admission']['task_ids']}),
        rejected_reasons=dict(Counter(r['rejection'] for r in rows if not r['supported'])),
        executable_advances=sum(len(r['law_admission']['task_ids']) for r in accepted),
        max_executable_advance_s=max((-c['executable_delta_s'] for r in accepted for c in r['changes']), default=0.),
        max_rolling_budget_residual_s=max((abs(v) for r in rows
            for v in r['projection']['rolling_budget_residual_s']), default=0.),
        max_headway_error_s=max((r['projection']['max_headway_error_s'] for r in rows), default=0.),
        fixed_and_other_direction_clocks_exact=True, no_continuous_or_tick_delay=True,
        all_advances_current_cycle_and_funded=True, zero_request_tick_equivalent=True,
        input_timetables_unchanged=True, rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--model-root', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.runtime))
    from runner_v3 import load_config
    from upper.timetable_planner import TimetableCurvePlanner
    assert Path(inspect.getfile(TimetableCurvePlanner)).resolve().is_relative_to(a.runtime.resolve())
    config = load_config(str(a.model_root/'configs_freqduet/F_freqduet_protocol_v6_confirmed_main_hiro.yaml'))
    cfg = config['upper']['timetable_planner']
    planner = TimetableCurvePlanner.from_config(cfg)
    assert planner.shared_directions and not planner.plan_all_directions and planner.action_dim == 1
    assert planner.terminal_schedule_mode == 'exact_headway_curve'
    assert planner.headway_budget_mode == 'rolling_zero_sum_delta_v6'
    assert planner.action_low == [-60.] and planner.action_high == [60.]
    assert cfg['action_ema_alpha'] == .35 and cfg['replan_interval_s'] == law.WINDOW_S
    assert not cfg.get('terminal_headway_floor', {}).get('enable', False)
    assert not config['upper'].get('action_bins')
    assert not config.get('fleet_noharm', {}).get('upper', {}).get('enable', False)
    protocol = a.out_root/'protocol.json'
    assert not protocol.exists(), 'preserve the original run; no overwrite'
    a.out_root.mkdir(parents=True, exist_ok=True)
    protocol.write_text(json.dumps(dict(stage='v220_supply_constrained_static_projection',
        source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], planner_source=inspect.getfile(TimetableCurvePlanner),
        planner_config=cfg, contexts=law.CONTEXTS,
        protocol='md/protocol_v6_v220_supply_projection.md in the sealed source commit',
        objective='Squared headway distance to SAME native+30 curve preview; no weight/gain search.',
        constraints='Only query-time ready-prefix funded non-anchor current900s slots may advance. '
            'All other cumulative clocks and every native rolling block endpoint exactly fixed. '
            'Native headway bounds; no earlier than query; no later continuous or executable clock. '
            'Same single raw scalar,EMA and direction-local callback. Changed projection shape.',
        progression='Nonzero funded executable advance in EACH fixed context admits only separate '
            'replay-budget design,not replay execution or a service-benefit claim.',
        new_replays=0, new_training=0, new_futures=0, main_policy_changed=False), indent=2)+'\n')
    for pattern in ('test_v219_*.py', 'test_v220_*.py'):
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1]/'tests'), '-p', pattern, '-v'], check=True)
    contexts = []
    for ci in range(len(law.CONTEXTS)):
        with gzip.open(a.archive_root/'records'/f'v216_c{ci}_capture1.json.gz', 'rt') as f:
            result = audit(json.load(f), planner, cfg['action_ema_alpha'], ci)
        detail = a.out_root/'details'
        detail.mkdir(exist_ok=True)
        (detail/f'c{ci}.json').write_text(json.dumps(result, indent=2)+'\n')
        contexts.append({k: v for k, v in result.items() if k != 'rows'})
        print('V220_CONTEXT', json.dumps(contexts[-1]), flush=True)
    summary = dict(status='v220_supply_projection_preflight_complete', contexts=contexts,
        native_contract=dict(plan_all_directions=False, action_dim=1, ema_streams='direction-local'),
        static_projection_admission=all(r['admitted_queries'] > 0 for r in contexts),
        next_stage='Separate small four-arm budget registration only if every fixed context passes.',
        proposed_controller_run=False, new_replays=0, new_training=0, new_futures=0,
        main_policy_changed=False,
        limitations='Archived native states only. No persistence under changed states, actual vehicle '
            'assignment, own-trip lower compensation or service improvement established. '
            'No independent future was consumed; no frequency-learning claim.')
    output = a.out_root/'summary'
    output.mkdir(exist_ok=True)
    (output/'status.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(summary['status'], flush=True)


if __name__ == '__main__':
    main()
