#!/usr/bin/env python3
"""Check native V219 projection equivalence and shared-action admission; no rollout."""

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

import freqduet_v219_projection as projection
import freqduet_v219_service_task as law


def audit(data, planner, alpha, context):
    trace, capture = data['trace'], data['capture']
    callbacks = {(r['time_s'], r['trip_id']): r for r in trace['upper_callbacks'] if r['commands']}
    launch_seq = {e['trip_id']: e['event_seq'] for e in capture['events'] if e['kind'] == 'launch'}
    previous, rows, checked, max_error = None, [], 0, 0.
    tick = data['result']['physical_contract']['dt_s']
    for query in capture['fresh_upper']:
        assert query['snapshot_event_seq'] == query['event_seq']
        observed = [r for r in trace['actual_launches'] if launch_seq[r['trip']] < query['event_seq']]
        trips = projection.query_timetable(query, observed)
        immutable = [vars(t).copy() for t in trips]
        raw = np.asarray(query['native_coefficient'], dtype=np.float32)
        native_effective = projection.ema_action(raw, previous, alpha)
        native = projection.preview(planner, trips, query, native_effective)
        callback = callbacks[query['time_s'], query['trip_id']]
        assert callback['commands'][0]['coefficient'] == query['native_coefficient']
        expected = {r['trip_id']: dict(planned_s=r['planned_s'], target_s=r['target_s']) for r in query['plans']}
        expected.update({r['trip_id']: {k: r['after'][k] for k in ('planned_s', 'target_s')}
                         for r in callback['changes']})
        max_error = max(max_error, projection.verify_native(native, expected, tick))
        checked += 1
        if query['time_s'] >= law.START_S:
            proposed_raw = np.clip(raw+30., planner.action_low, planner.action_high).astype(np.float32)
            proposed_effective = projection.ema_action(proposed_raw, previous, alpha)
            proposed = projection.preview(planner, trips, query, proposed_effective)
            evidence = projection.inspect_proposal(query, native, proposed, tick)
            rows.append(dict(time_s=query['time_s'], event_seq=query['event_seq'], trip_id=query['trip_id'],
                native_effective=native_effective.tolist(), proposed_effective=proposed_effective.tolist(), **evidence))
        assert immutable == [vars(t) for t in trips], 'preview mutated its input timetable'
        previous = native_effective
    seed, day = law.CONTEXTS[context]
    accepted = [r for r in rows if r['supported']]
    return dict(context=context, actor=seed, day=day, native_snapshots_checked=checked,
        max_native_error_s=max_error, native_continuous_and_tick_equivalent=True,
        candidate_queries=len(rows), admitted_queries=len(accepted),
        admitted_trip_ids=sorted({t for r in accepted for t in r['law_admission']['task_ids']}),
        rejected_reasons=dict(Counter(r['rejection'] for r in rows if not r['supported'])),
        raw_proposals_with_plan_delay=sum(bool(r['continuous_later_trip_ids']) for r in rows),
        raw_proposals_with_tick_delay=sum(bool(r['clock_later_trip_ids']) for r in rows),
        raw_proposals_with_unfunded_cycle_advance=sum(bool(r['unfunded_within_cycle_trip_ids']) for r in rows),
        accepted_advanced_by_direction=[sum(r['advanced_by_direction'][d] for r in accepted) for d in (0, 1)],
        accepted_outside_cycle_advances=sum(r['outside_cycle_advanced'] for r in accepted),
        all_accepted_advances_funded=True, no_accepted_plan_or_tick_delay=True,
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
    assert planner.shared_directions and planner.plan_all_directions and planner.action_dim == 1
    assert planner.terminal_schedule_mode == 'exact_headway_curve'
    assert planner.headway_budget_mode == 'rolling_zero_sum_delta_v6'
    assert planner.action_low == [-60.] and planner.action_high == [60.]
    assert cfg['action_ema_alpha'] == .35 and cfg['replan_interval_s'] == law.WINDOW_S
    assert not config['upper'].get('action_bins')
    assert not config.get('fleet_noharm', {}).get('upper', {}).get('enable', False)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v219_*.py', '-v'], check=True)
    a.out_root.mkdir(parents=True, exist_ok=True)
    protocol = a.out_root/'protocol.json'
    assert not protocol.exists(), 'inspect the original preflight; no automatic overwrite'
    protocol.write_text(json.dumps(dict(stage='v219_native_projection_preflight',
        source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], planner_source=inspect.getfile(TimetableCurvePlanner),
        planner_config=cfg, contexts=[list(c) for c in law.CONTEXTS],
        acceptance='Native continuous clocks and headways within1e-6s AND identical executable ticks '
            'for EVERY pending trip at EVERY fresh query,including warmup. Same native pre-query EMA '
            'for native/+30 previews;never propagate counterfactual previews into baseline state. '
            'Both directions current900s advances must be ready-prefix funded;no continuous or tick '
            'delay anywhere in the projected timetable. Retain every fixed context and every rejection.',
        replay_budget_condition='Only after all four contexts have a nonzero admissible preview '
            'may a separate small four-arm budget be sealed. No replays are authorized by this job.',
        new_replays=0, new_training=0, main_policy_changed=False), indent=2)+'\n')
    contexts = []
    for ci in range(len(law.CONTEXTS)):
        with gzip.open(a.archive_root/'records'/f'v216_c{ci}_capture1.json.gz', 'rt') as f:
            result = audit(json.load(f), planner, cfg['action_ema_alpha'], ci)
        detail = a.out_root/'details'
        detail.mkdir(exist_ok=True)
        (detail/f'c{ci}.json').write_text(json.dumps(result, indent=2)+'\n')
        contexts.append({k: v for k, v in result.items() if k != 'rows'})
    passed = all(r['admitted_queries'] > 0 for r in contexts)
    summary = dict(status='v219_native_projection_preflight_complete', contexts=contexts,
        projection_equivalence_passed=True, replay_budget_admission=passed,
        proposed_controller_run=False, new_replays=0, new_training=0, main_policy_changed=False,
        limitations='Archived native states only,not a closed-loop candidate trajectory. Ready support '
            'covers the frozen public900s cycle;outside-cycle advances remain unexecuted future plans. '
            'Lower compensation eligibility and passenger/regularity benefit are not established.')
    output = a.out_root/'summary'
    output.mkdir(exist_ok=True)
    (output/'status.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(summary['status'], json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
