"""Correct only CPU admission on the existing, unstarted V401 training tasks."""

import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,'/home/erzhu419/mine_code/scheduleurm/skill')
import scheduler


if __name__=='__main__':
    receipt=HERE/'scheduler_cpu_admission_fix.json'
    assert not receipt.exists()
    original=json.loads((HERE/'submission_pipeline01.json').read_text())
    selected={t['id'] for t in original['tasks'] if '/train/' in t['signature'] or
              ('/checkpoints/' in t['signature'] and not t['signature'].endswith('/b0'))}
    assert len(selected)==80
    reason='Approved CPU-only transit rollouts and small 181-input learner updates on node001--006; dedicated CPU environment, unchanged training budget.'
    with scheduler.state_lock(timeout_s=120,purpose='V401 explicit CPU training admission'):
        state=scheduler.load_state()
        tasks=[t for t in state['tasks'] if t['id'] in selected]
        assert len(tasks)==80 and all(t['status']=='queued' for t in tasks)
        assert all(t['require_node']=='node004' and t['priority']=='high' and
                   t['no_auto_requeue'] and not t.get('started_at') for t in tasks)
        blocked=[t['id'] for t in tasks if scheduler._queued_cpu_training_block_reason(t)]
        assert blocked
        for task in tasks:
            task['allow_cpu_training']=True
            task['cpu_training_justification']=reason
            assert not scheduler._queued_cpu_training_block_reason(task)
        scheduler.save_state(state)
    ready=[t['id'] for t in tasks if not scheduler._queued_wait_for_file_block_reason(t)]
    assert len(ready)==8
    with receipt.open('x') as stream:
        json.dump(dict(reason='Missing explicit CPU-training admission on an intentionally CPU-only workflow.',
            changed_existing_tasks=80,collection_paths=64,update_tasks=16,
            ready_task_ids=ready,physical_starts_added=0,new_tasks_submitted=0,
            original_task_ids_preserved=True,engineering_gate_passed=True,
            effective_physical_start_cap=92,candidate_training_update_cap=3072,
            priority='high',node='node004',automatic_retry=False,main_policy_changed=False),stream,indent=2)
    command=[sys.executable,str(Path(scheduler.__file__)),'dispatch','--lock-timeout','120']
    for task_id in ready:command+=['--task-id',task_id]
    print('V401_CPU_ADMISSION_FIXED',len(tasks),'READY',ready,flush=True)
    result=subprocess.run(command,text=True,capture_output=True)
    print(result.stdout[-6500:],flush=True)
    if result.returncode:print(result.stderr[-2000:],flush=True)
    raise SystemExit(result.returncode)
