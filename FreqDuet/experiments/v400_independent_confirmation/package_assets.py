"""Executed on the GPU archive host: transport frozen assets directly to HPC."""

import argparse
import json
from pathlib import Path
import subprocess
import tarfile

GPU = Path('/home/erzhu419/scheduleurm_work')
DEST = '/home/zhengliang01/scheduleurm_work/FreqDuet-v400-independent-confirmation'
HOST = 'zhengliang01@202.197.46.16'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', type=Path, required=True)
    args = parser.parse_args()
    output = GPU/'FreqDuet-v400-transport01'
    output.mkdir(exist_ok=False)
    rows = []

    def add(source, target):
        source = Path(source)
        assert source.is_file(), str(source)
        rows.append((source, target))

    basis = GPU/'FreqDuet-v350-live-basis/source07'
    for name in ('basis_worker.py', 'basis_law.py', 'basis_integration.py', 'provider.py'):
        add(basis/name, 'basis/'+name)
    add(GPU/'FreqDuet-v347-runtime/run01/queue_history.py', 'history/queue_history.py')
    support = GPU/'FreqDuet-v350-live-basis/matched01/aggregate/support'
    for name in ('physical.py', 'v324.py', 'v326.py', 'credit.py'):
        add(support/name, 'support/'+name)
    add(GPU/'FreqDuet-v352-budget-interface/source01/gate_default.py', 'support/gate_default.py')
    add(GPU/'FreqDuet-v352-budget-interface/source01/aggregate.py', 'support/budget_account.py')
    for kind, checkpoint, config in (
        ('K4', GPU/'FreqDuet-v350-live-basis/inputs/actor/checkpoint',
         GPU/'FreqDuet-v350-live-basis/inputs/actor/config.json'),
        ('K0', GPU/'FreqDuet-v350-live-basis/matched01/checkpoints/s314011/b8', basis/'matched_config.json')):
        add(checkpoint/'policy.pt', kind+'/policy.pt')
        add(checkpoint/'summary/results.json', kind+'/summary/results.json')
        add(config, kind+'/config.json')
    runtime = GPU/'FreqDuet-v350-live-basis/inputs/runtime'
    for folder in ('model', 'native_training'):
        for path in sorted((runtime/folder).rglob('*')):
            if path.is_file() and path.suffix in ('.yaml', '.json', '.xlsx', '.pt'):
                add(path, 'runtime/'+str(path.relative_to(runtime)))
    for name in ('benchmark_summary.json', 'training_equivalence/summary.json'):
        add(runtime/'traffic/runtime_optimization_20261014'/name,
            'runtime/traffic/runtime_optimization_20261014/'+name)
    for arm, folder in (
        ('K4_r050', GPU/'FreqDuet-v352-budget-interface/run01/default_r050_d2077002'),
        ('manual_r050', GPU/'FreqDuet-v399-equal-budget-manual/run01/default_r050_d2077002')):
        add(folder/'summary/results.json', 'reference/'+arm+'/summary/results.json')
        add(folder/'raw/path.pkl.gz', 'reference/'+arm+'/raw/path.pkl.gz')
        if arm == 'K4_r050':
            add(folder/'raw/upper_transitions.pkl.gz', 'reference/'+arm+'/raw/upper_transitions.pkl.gz')
    environment = json.loads(args.environment.read_text())
    (output/'environment.json').write_text(json.dumps(environment))
    add(output/'environment.json', 'environment.json')
    receipt = {'source_host':'jtl110gpu2', 'destination_host':'node001-node006 shared storage',
               'files':[{'source':str(p), 'relative':n, 'bytes':p.stat().st_size} for p,n in rows],
               'physical_starts':0, 'training_updates':0}
    (output/'transport.json').write_text(json.dumps(receipt, separators=(',', ':')))
    add(output/'transport.json', 'transport.json')
    bundle = output/'assets.tar.gz'
    with tarfile.open(bundle, 'w:gz') as archive:
        for path, relative in rows:
            archive.add(path, arcname=relative)
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', HOST]
    subprocess.run(ssh+['mkdir', '-p', DEST+'/transport'], check=True)
    subprocess.run(['rsync', '-a', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=20',
                    str(bundle), HOST+':'+DEST+'/transport/assets.tar.gz'], check=True)
    print(json.dumps({'files':len(rows), 'uncompressed_bytes':sum(p.stat().st_size for p,_ in rows),
                      'bundle_bytes':bundle.stat().st_size, 'destination':DEST+'/transport/assets.tar.gz',
                      'physical_starts':0, 'training_updates':0}), flush=True)


if __name__ == '__main__':
    main()
