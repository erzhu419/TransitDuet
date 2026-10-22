#!/usr/bin/env python3
"""Recover complete common records, then perform four explicitly approved passive-only repairs."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import run_freqduet_v216_passive_query as v


def written_common(path):
    # The failed V216 writer completed result/trace before entering capture.
    with gzip.open(path, 'rt') as f:
        text = f.read()
    decoder, offset, data = json.JSONDecoder(), 1, {}
    assert text[0] == '{'
    for expected in ('result', 'trace'):
        key, offset = decoder.raw_decode(text, offset)
        assert key == expected and text[offset] == ':'
        data[key], offset = decoder.raw_decode(text, offset+1)
        assert text[offset] == ','
        offset += 1
    return data, text[-120:]


def audit(root, old, registered):
    v.preflight_checks(root, registered)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v216_recovery.py', '-v'], check=True)
    assert not (root/'protocol.json').exists()
    rows = []
    for ci in range(4):
        before, _ = written_common(old/'records'/f'v216_c{ci}_capture0.json.gz')
        after, tail = written_common(old/'records'/f'v216_c{ci}_capture1.json.gz')
        progress = v.prior.read(old/'progress'/f'c{ci}.json')
        assert progress['failure'] == 'Out of range float values are not JSON compliant: -inf'
        assert tail.endswith('"last_observed_return_s":'), 'confirm the actual failed field,not another sentinel'
        v.exact(before, after, 'recovered_all_common_records')
        ready = v.prior.read(old/'ready'/f'v216_c{ci}_capture1.json')
        assert ready['base_actors_unchanged'] and ready['training_updates'] == 0
        rows.append(dict(context=ci, common_endpoint_action_state_plan_and_timing_exact=True,
                         full_simulation_completed=True, live_query_capture_complete=False))
    amendment = dict(source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], original_root=str(old),
        user_approved=True, reason='Initial fixed-pool inventory uses -inf for no observed return;strict JSON failed after common records.',
        original_reference_paths=4, completed_original_smoke_paths=4, approved_repair_paths=4,
        smoke_cap_original=6, smoke_cap_amended=8, additional_smoke_allowance=2,
        total_full_simulation_cap=12, reference_replays=0, added_contexts=0, added_candidates=0,
        added_futures=0, scientific_rules_and_gates_unchanged=True, automatic_extension=False,
        scope='Repair initial-inventory representation only. Reuse all four references;repeat only four passive-on paths. '
            'Do not replay V214,change anchor selection,open future banks or start an intervention matrix.')
    v.write(root/'protocol.json', amendment)
    v.write(root/'preflight/done.json', dict(status='v216_recovery_preflight_and_common_audit_passed',
        contexts=rows, original_full_simulations=8, new_rollouts=0, main_policy_changed=False,
        missing='Query-time inventory/global event sequence and same-vehicle supply labels were not fully persisted.'))
    print('V216_COMMON_RECOVERY_EXACT_REPAIR_SEALED', flush=True)


def repair_one(root, old, ci):
    progress = root/'progress'/f'c{ci}.json'
    assert not progress.exists(), 'do not repeat an attempted repair'
    v.write(progress, dict(context=ci, repair_smoke_started=1, references_started=0))
    before, _ = written_common(old/'records'/f'v216_c{ci}_capture0.json.gz')
    observed = v.replay(root, ci, True)
    v.exact(before['result'], observed[0], 'repair_endpoint_contract_ledger')
    v.exact(before['trace'], observed[1], 'repair_full_action_state_plan_and_timing')
    summary = v.pair_summary(ci, (before['result'], before['trace']), observed,
                            dict(context=ci, references_reused=1, repair_smoke_completed=1))
    v.write(root/'summary'/f'c{ci}.json', summary)
    v.write(progress, dict(context=ci, repair_smoke_started=1, repair_smoke_completed=1))
    print('V216_PASSIVE_REPAIR_EXACT', ci, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('audit', 'capture'))
    for key in ('out-root', 'original-root', 'registered-root'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    if a.command == 'audit':
        audit(a.out_root, a.original_root, a.registered_root)
    else:
        assert v.prior.read(a.out_root/'protocol.json')['user_approved'] is True
        assert v.prior.read(a.out_root/'preflight/done.json')['original_full_simulations'] == 8
        assert not (a.out_root/'capture_started.json').exists()
        v.write(a.out_root/'capture_started.json', dict(repair_paths=4, reference_replays=0))
        with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn')) as pool:
            tasks = [pool.submit(repair_one, a.out_root, a.original_root, ci) for ci in range(4)]
            for task in tasks:
                task.result()
        v.aggregate(a.out_root, smoke_used=8, smoke_cap=8, new_rollouts=4)
        summary = v.prior.read(a.out_root/'summary/status.json')
        summary.update(full_simulations_including_original=12, reused_reference_paths=4,
                       original_failed_write_paths=4, explicitly_approved_additional_smoke_slots=2)
        v.write(a.out_root/'summary/status.json', summary)
        print('V216_REPAIR_COMPLETE_TOTAL12_SMOKE8_NO_INTERVENTION', flush=True)


if __name__ == '__main__':
    main()
