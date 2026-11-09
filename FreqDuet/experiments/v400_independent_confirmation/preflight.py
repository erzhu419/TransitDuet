"""Inspect the frozen confirmation plan and existing JSON metadata; never run traffic."""

import argparse
import json
import os
from pathlib import Path
import re


DAY_KEYS = {'day', 'days', 'scenario_seed', 'scenario_seeds', 'demand_seed',
            'demand_days', 'evaluation_days', 'confirmation_days', 'reserved_days',
            'train_days', 'train_batches', 'scenario_day'}
SKIP = {'raw', 'checkpoints', 'logs', 'tb_logs', 'site-packages', 'node_modules',
        '.git', '__pycache__', 'tape', 'replay_buffer', 'native_train'}


def day_hits(value, days, location=''):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f'{location}.{key}' if location else key
            if location.rsplit('.', 1)[-1] == 'by_day' and str(key).isdigit() and int(key) in days:
                found.append({'field': path, 'day': int(key)})
            if key in DAY_KEYS:
                pending = [child]
                while pending:
                    item = pending.pop()
                    if isinstance(item, list):
                        pending.extend(item)
                    elif isinstance(item, int) and item in days:
                        found.append({'field': path, 'day': item})
            found.extend(day_hits(child, days, path))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            if isinstance(child, (dict, list)):
                found.extend(day_hits(child, days, f'{location}[{i}]'))
    return found


def inspect_plan(plan):
    assert plan['protocol_id'] == 'v400_independent_confirmation'
    days = plan['days']
    assert len(days) == len(set(days)) == 16
    assert not set(days).intersection(range(1780001, 1780033))
    assert not set(days).intersection(range(2077001, 2077005))
    expected = {'K4_r050': ('learned', 4, .5), 'manual_r050': ('manual_joint', 4, .5),
                'K4_r025': ('learned', 4, .25), 'manual_r025': ('manual_joint', 4, .25),
                'K0_r050': ('learned', 0, .5)}
    actual = {a['id']: (a['upper'], a['K'], a['rho']) for a in plan['arms']}
    assert len(plan['arms']) == len(actual) == 5 and actual == expected
    assert len(days)*len(actual) == plan['planned_confirmation_paths'] == 80
    assert plan['candidate_seed'] == 314011 and plan['checkpoint_batch'] == 8
    assert plan['fast_actor'] == 12011 and plan['fast_sampler'] == 27011001
    assert plan['default_rho'] == .5 and plan['authorized_traffic_starts_this_phase'] == 0
    assert plan['training_updates'] == 0 and not plan['main_policy_changed']
    assert plan['priority'] == 'high' and not plan['automatic_retry']
    assert plan['allowed_nodes'] == [f'node{i:03}' for i in range(1, 7)]


def inspect_metadata(roots, days):
    scanned, collisions, errors, present = 0, [], [], []
    for root in map(Path, roots):
        if not root.is_dir():
            errors.append({'root': str(root), 'error': 'metadata root missing'})
            continue
        present.append(str(root))
        for directory, folders, files in os.walk(root):
            folders[:] = [name for name in folders if name not in SKIP
                          and not name.startswith(('source', 'runtime_source', 'v400_', 'protocol_v6_v400'))]
            for name in files:
                if name not in {'results.json', 'started.json', 'config.json', 'manifest.json',
                                'protocol.json', 'completion.json', 'CURRENT.json', 'PIPELINE.json'} \
                        and not re.fullmatch(r'p\d+\.json', name):
                    continue
                path = Path(directory)/name
                try:
                    value = json.loads(path.read_text())
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    errors.append({'path': str(path), 'error': type(exc).__name__})
                    continue
                scanned += 1
                for hit in day_hits(value, days):
                    collisions.append({'path': str(path), **hit})
    return {'roots_present': present, 'json_files_read': scanned,
            'collisions': collisions, 'read_errors': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--gpu-inventory', type=Path, required=True)
    parser.add_argument('--queue-inventory', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.manifest.read_text())
    inspect_plan(plan)
    # Real receipts nest identities, and training configs nest day lists.
    assert len(day_hits({'result': {'day': 2400001}, 'train_batches': [[2400002]]},
                        {2400001, 2400002})) == 2
    assert not day_hits({'passenger_count': 2400001}, {2400001})
    assert len(day_hits({'result': {'by_day': {'2400001': {'journey': 1.}}}}, {2400001})) == 1
    metadata = inspect_metadata(plan['metadata_roots'], set(plan['days']))
    gpu = json.loads(args.gpu_inventory.read_text())
    queue = json.loads(args.queue_inventory.read_text())
    assert gpu['days'] == queue['days'] == plan['days']
    freshness = all(not r['collisions'] and not r['read_errors'] and r['json_files_read'] > 0
                    for r in (metadata, gpu['metadata'])) and not queue['collisions']
    assets = {key: {'path': path, 'exists': Path(path).exists()}
              for key, path in plan['server_assets'].items() if key.startswith('hpc_')}
    missing = [key for key, value in assets.items() if not value['exists']]
    value = {'protocol_id': plan['protocol_id'], 'completed': True,
             'status': 'zero_traffic_preflight_complete', 'plan_checks_passed': True,
             'metadata': metadata, 'hpc_assets': assets, 'missing_hpc_assets': missing,
             'gpu_inventory': gpu, 'queue_inventory': queue,
             'freshness_verified_across_both_servers': freshness,
             'traffic_ready': False, 'physical_starts': 0, 'training_updates': 0,
             'actor_calls': 0, 'critic_calls': 0, 'main_policy_changed': False,
             'next_step': 'Audit GPU metadata, stage frozen assets server-to-server, then validate the separate new-day entry before seeking traffic-start authorization.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print('V400_PREFLIGHT_COMPLETE', metadata['json_files_read'], len(metadata['collisions']), missing, flush=True)


if __name__ == '__main__':
    main()
