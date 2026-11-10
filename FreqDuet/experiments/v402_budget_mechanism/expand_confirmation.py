"""Apply the approved concurrency amendment only to unstarted confirmation."""

import base64
import io
import json
from pathlib import Path
import shlex
import tarfile

from launch import HERE,ROOT,deployment,submit_many,dispatch,scheduler


def main():
    destination=ROOT.parent/'source02'
    old=json.loads((HERE/'submission_confirmation.json').read_text())['tasks']
    assert len(old)==8 and not (HERE/'submission_confirmation32.json').exists()
    assert not (HERE/'review/selection/ready.json').exists()
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name in ('confirmation32.py','check_partition.py','concurrency_amendment.json'):
            path=HERE/name
            if path.suffix=='.py':compile(path.read_bytes(),name,'exec')
            archive.add(path,arcname=name)
    code='import base64,io,tarfile\nfrom pathlib import Path\n'
    code+=f'p=Path({str(destination)!r});p.mkdir(exist_ok=False)\n'
    code+=f'tarfile.open(fileobj=io.BytesIO(base64.b64decode({base64.b64encode(buffer.getvalue()).decode()!r}))).extractall(p)\n'
    d=deployment();rc,out,err=scheduler.run_on('node003',shlex.join([d['python'],'-c',code]),timeout=60,check=False)
    assert rc==0,(out,err[-1800:])
    partition_ready=str(HERE/'review/partition_check/ready.json')
    aggregate_id=json.loads((HERE/'submission_aggregate.json').read_text())['tasks'][0]['id']
    with scheduler.state_lock(timeout_s=600,purpose='V402 approved confirmation-only concurrency'):
        state=scheduler.load_state();tasks={t['id']:t for t in state['tasks']}
        for entry in old:
            t=tasks[entry['id']]
            assert t['status']=='queued' and not t.get('started_at')
            argv=shlex.split(t['cmd']);assert argv[2]==str(ROOT.parent/'source01/shard.py')
            argv[2]=str(destination/'confirmation32.py');t['cmd']=shlex.join(argv)
            t['wait_for_files'].append(partition_ready)
        aggregate=tasks[aggregate_id]
        assert aggregate['status']=='queued' and not aggregate.get('started_at')
        aggregate['wait_for_files']=[str(HERE/f'review/confirmation_shard_{i}/finished.json') for i in range(32)]
        scheduler.save_state(state)
    common=['--root',ROOT,'--assets',d['assets']]
    requests=[('partition_check',destination/'check_partition.py',['--root',ROOT],ROOT/'partition_check/summary',[])]
    requests += [(f'confirmation_shard_{i}',destination/'confirmation32.py',
        [*common,'--phase','confirmation','--shard',i],ROOT/f'confirmation_shard_{i}/summary',
        [HERE/'review/selection/ready.json',partition_ready]) for i in range(8,32)]
    submitted=submit_many(requests)
    all_shards=old+[dict(id=t['id'],signature=t['signature']) for t in submitted[1:]]
    with (HERE/'submission_confirmation32.json').open('x') as stream:
        json.dump(dict(stage='confirmation',tasks=all_shards,parallel_paths=32,
            physics_worker=str(ROOT.parent/'source01/worker.py'),physical_start_cap=150,
            calibration_tasks_unchanged=True),stream,indent=2)
    with (HERE/'submission_partition_check.json').open('x') as stream:
        json.dump(dict(task_id=submitted[0]['id'],physical_starts=0,source=str(destination)),stream,indent=2)
    print('V402_CONFIRMATION32_REGISTERED',[(t['id'],t['signature']) for t in submitted],flush=True)
    dispatch(old+submitted+[dict(id=aggregate_id)])


if __name__=='__main__':main()
