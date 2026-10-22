#!/usr/bin/env python3
"""Read saved V214 smoke paths; separate missing support from zero requests."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import subprocess
import sys


ACTIONS = (0, 5, 10, 15, 20, 30, 45)


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    return json.loads(path.read_text())


def eligible(q):
    return (not q['terminal_masked'] and q['forward_valid'] and q['follower_valid']
            and any(i < q['base_bin'] for i in q['feasible_bins']))


def support(data):
    anchor = data['result']['anchor']
    queries = data['trace']['lower_queries']
    start = anchor['time_s']
    pool = {(b['bus_id'], b['trip_id']) for b in anchor['inbound']}
    selected = []
    for b in anchor['inbound']:
        rows = [q for q in queries if (q['bus_id'], q['trip_id']) == (b['bus_id'], b['trip_id'])]
        window = [q for q in rows if start <= q['time_s'] < start+900]
        counts = Counter(queries_in_window=len(window), positive=sum(q['base_bin'] > 0 for q in window),
            zero=sum(q['base_bin'] == 0 for q in window), eligible_positive=sum(eligible(q) for q in window),
            invalid_forward=sum(not q['forward_valid'] for q in window),
            invalid_follower=sum(not q['follower_valid'] for q in window),
            terminal_masked=sum(q['terminal_masked'] for q in window),
            queries_after_window=sum(q['time_s'] >= start+900 for q in rows))
        selected.append(dict(vehicle=b, counts=dict(counts), window_queries=[dict(time_s=q['time_s'],
            station_id=q['station_id'], base_action_s=ACTIONS[q['base_bin']], command_s=q['command_s'],
            forward_valid=q['forward_valid'], follower_valid=q['follower_valid'],
            state_captured_s=q['state_captured_s']) for q in window]))
    others = [q for q in queries if start <= q['time_s'] < start+900 and q['direction'] != anchor['direction']
              and (q['bus_id'], q['trip_id']) not in pool]
    return dict(anchor_s=start, direction=anchor['direction'], selected=selected,
        other_inbound_queries=len(others), other_inbound_eligible_positive=sum(eligible(q) for q in others),
        other_inbound_positive_trips=len({(q['bus_id'], q['trip_id']) for q in others if eligible(q)}),
        other_opportunities_not_proof_of_queue_or_immediate_supply=True,
        registered=data['result']['execution'])


def audit(root, old, out):
    stopped = read(root/'summary/status.json')
    assert stopped['rollouts_completed'] == 5 and stopped['phase'] == 'joint_repeat'
    assert stopped['failure'] == 'smoke has no real nonzero inbound holding opportunity; do not extend/search'
    paths = {name: read(root/'records'/f'{name}.json.gz')
             for name in ('observer_on', 'forked_baseline', 'joint', 'joint_repeat')}
    assert paths['joint'] == paths['joint_repeat'], 'saved joint repeat differs'
    cached = read(old/'references/c0.json')['results']
    assert {k: paths['observer_on']['result']['results'][k] for k in cached} == cached
    base, joint = (paths[k]['result']['results'] for k in ('forked_baseline', 'joint'))
    summary = dict(status='v214_noop_support_audit_complete', smoke_gate_passed=False,
        archived_baseline_endpoint_rechecked=True, saved_full_joint_repeat_rechecked=True,
        passive_full_action_timing_check_passed_in_smoke_before_this_gate=True,
        support={name: support(paths[name]) for name in ('forked_baseline', 'joint')},
        upper_only_smoke_effects={k: joint[k]-base[k] for k in
            ('passenger_person_s', 'journey_min', 'cv_squared', 'readiness_delay_s', 'public_launch_delay_total_s')},
        scope='No registered lower change: these are upper-only effects in one smoke future,not joint control gains.',
        new_rollouts=0, training_updates=0, main_policy_changed=False)
    (out/'summary').mkdir(parents=True, exist_ok=True)
    (out/'summary/status.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('V214_READONLY_SUPPORT_AUDIT_COMPLETE', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('smoke-root', 'v211-root', 'out-root'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v214_smoke_support.py', '-v'], check=True)
    audit(a.smoke_root, a.v211_root, a.out_root)


if __name__ == '__main__':
    main()
