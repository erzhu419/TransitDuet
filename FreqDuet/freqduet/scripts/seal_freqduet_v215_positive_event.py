#!/usr/bin/env python3
"""Seal V215 after synthetic tests and read-only runtime/metadata audits."""

import argparse
import ast
import gzip
import json
from pathlib import Path
import re
import subprocess
import sys

import preregister_freqduet_v215_positive_event as v


RESERVED = re.compile(r'(?<!\d)(?:215100[12]|2150000\d{3}|2159000\d{3})(?!\d)')


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    return json.loads(path.read_text())


def functions(path):
    return {n.name: n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef)}


def runtime_audit(root):
    bus = functions(root/'env/bus.py')
    sim = functions(root/'env/sim.py')
    runner = functions(root/'runner_v3.py')
    returns = [n for n in bus.values() if 'self.direction = int(not self.direction)' in ast.unparse(n)
               and 'self.back_to_terminal_time = current_time' in ast.unparse(n)]
    assert len(returns) == 1 and 'self.on_route = False' in ast.unparse(returns[0])
    launch = ast.unparse(sim['launch_bus'])
    assert 'bus_in_terminal' in launch and 'back_to_terminal_time' in launch
    assert 'fixed_pool' in launch and 'return None' in launch and 'trip.direction' in launch
    assert 'not bus.on_route' in ast.unparse(sim['bus_in_terminal'])
    upper = ast.unparse(runner['_upper_callback_v2'])
    assert 'upper_decision_taken = False' in upper and 'upper_decision_taken = True' in upper
    assert 'self.upper_trainer.policy_net.get_action' in upper
    assert '_prepare_for_action' in bus and '_capture_lower_action_context' in runner
    source = (root/'env/bus.py').read_text()
    assert 'pre_action_time_s' not in source, 'revise native observation clock adapter assumptions'
    return dict(return_method=returns[0].name, return_line=returns[0].lineno,
        terminal_direction_flips=True, ready_is_not_on_route=True, fixed_pool_launch_requires_ready=True,
        assignment='Current compatible terminal buses ordered by observed back_to_terminal_time,not distance.',
        launch_line=sim['launch_bus'].lineno, native_prepare_line=bus['_prepare_for_action'].lineno,
        upper_callback_line=runner['_upper_callback_v2'].lineno, fresh_and_reuse_branches_exist=True,
        live_adapter_implemented=False)


def archive_schema(smoke):
    # Schema only: never apply the new selector to consumed V214 queries.
    rows = read(smoke/'records/forked_baseline.json.gz')['trace']['lower_queries']
    assert rows
    missing = [k for k in v.LOWER_FIELDS if any(k not in q for q in rows)]
    assert {'ready', 'active', 'plans', 'event_seq', 'snapshot_read_s', 'snapshot_event_seq'} <= set(missing)
    assert all(q['state_captured_s'] <= q['time_s'] == q['position_sample_s'] for q in rows)
    return dict(queries=len(rows), missing_query_fields=missing, existing_observation_clocks_valid=True,
        queried_v214_for_new_anchors=False, old_anchor_inventory_not_query_inventory=True,
        empirical_queue_support_verified=False, empirical_assignment_verified=False,
        next_capture='Native query ready/active/pending snapshots and global event order;no retrospective fill.')


def seal(a):
    receipt = read(a.namespace_receipt)
    assert receipt['conflicts'] == [] and receipt['archive_scan_complete']
    assert receipt['scenario_seeds'] == [2151001, 2151002]
    prior = read(a.v214_audit/'summary/status.json')
    assert prior['smoke_gate_passed'] is False, 'V214 stop conclusion must remain explicit'
    matches, count = [], 0
    for pattern in ('*/protocol*.json', '*/registration*.json', '*/manifest*.json'):
        for path in a.results_root.glob(pattern):
            if path.parent == a.out_root:
                continue
            count += 1
            if RESERVED.search(path.read_text()):
                matches.append(str(path))
    assert not matches, 'reserved namespace already in archived protocol metadata: '+str(matches)
    runtime, schema = runtime_audit(a.runtime_root), archive_schema(a.smoke_root)
    protocol = v.registration(a.source)
    protocol['namespace_audit'] = dict(receipt=receipt, server_protocol_files_checked=count)
    a.out_root.mkdir(parents=True, exist_ok=True)
    path = a.out_root/'protocol.json'
    assert not path.exists(), 'sealed registration already exists;do not overwrite'
    path.write_text(json.dumps(protocol, indent=2)+'\n')
    summary = dict(status='v215_preregistered_live_query_capture_pending', source=a.source,
        synthetic_tests_passed=8, runtime=runtime, archive_schema=schema,
        namespace=dict(hot_tasks_checked=receipt['hot_tasks_checked'],
            archive_tasks_checked=receipt['archive_tasks_checked'],
            malformed_archive_lines_text_checked=receipt['malformed_archive_lines_text_checked'],
            server_protocol_files_checked=count, conflicts=[],
            scope='Known scheduler hot/archive FreqDuet records and top-level archived protocol metadata;not an exhaustive raw-cache scan.'),
        contexts=protocol['contexts'], protocol_path=str(path), v214_stays_stopped=True,
        replay_admitted=False, new_rollouts=0, training_updates=0, interval_fit=False, main_policy_changed=False,
        next_step='Separately authorize passive live-query capture/equivalence before new smoke;no V214 repointing.')
    (a.out_root/'summary').mkdir(exist_ok=True)
    (a.out_root/'summary/status.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('V215_PREREGISTRATION_SEALED_NO_REPLAY', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('runtime-root', 'results-root', 'smoke-root', 'v214-audit', 'namespace-receipt', 'out-root'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v215_positive_event.py', '-v'], check=True)
    seal(a)


if __name__ == '__main__':
    main()
