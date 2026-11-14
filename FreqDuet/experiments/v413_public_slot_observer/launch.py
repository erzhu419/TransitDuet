"""Submit each explicitly released V413 stage through the shared scheduler."""

import argparse
import base64
import importlib.util
import io
from pathlib import Path
import shlex
import tarfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('shared', HERE.parent / 'v404_publication_closure/launch.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
shared.HERE = HERE
shared.ROOT = Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v413-public-slot-observer/run01')
shared.SOURCE = shared.ROOT.parent / 'source02'
PREFLIGHT = 'preflight02'
CONFIRMATION = shared.ROOT.parent / 'confirmation01'
READOUT = shared.ROOT.parent / 'readout01'
ERROR_AUDIT = shared.ROOT.parent / 'error_analysis02'
ERROR_LABEL = 'error_audit02'
COMPONENTS = shared.ROOT.parent / 'components02'
COMPONENT_LABEL = 'component_audit02'
COMPONENT_ENTRY = shared.ROOT.parent / 'component_entry01'
COMPONENT_READOUT = shared.ROOT.parent / 'component_readout01'
CONTINUATION_AUDIT = shared.ROOT.parent / 'continuation_audit01'
RUNNING_SHADOW = shared.ROOT.parent / 'running_shadow01'
RUNNING_PARITY = shared.ROOT.parent / 'running_parity01'


def stage(source, files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for name in files:
            path = HERE / name
            if path.suffix == '.py':
                compile(path.read_bytes(), name, 'exec')
            tar.add(path, arcname=name)
    code = ('import base64,io,tarfile\nfrom pathlib import Path\n'
            f'p=Path({str(source)!r});p.mkdir(parents=True,exist_ok=False)\n'
            f'tarfile.open(fileobj=io.BytesIO(base64.b64decode({base64.b64encode(buf.getvalue()).decode()!r}))).extractall(p)\n')
    rc, out, err = shared.scheduler.run_on('node004', shlex.join([
        shared.deployment()['python'], '-c', code]), timeout=60, check=False)
    assert rc == 0, (out, err[-1500:])


def job(label, phase, day=None):
    confirming = phase in ('verify', 'case')
    source, script = ((RUNNING_PARITY, 'running_shadow_parity.py') if phase == 'running_shadow_parity' else
                      (RUNNING_SHADOW, 'running_shadow_checks.py') if phase == 'running_shadow_checks' else
                      (CONTINUATION_AUDIT, 'continuation_audit.py') if phase == 'continuation_audit' else
                      (COMPONENT_READOUT, 'component_readout.py') if phase == 'component_readout' else
                      (COMPONENT_ENTRY, 'component_entry.py') if phase in ('component_checks','component_parity') else
                      (COMPONENTS, 'component_audit.py') if phase == 'component_audit' else
                      (ERROR_AUDIT, 'error_audit.py') if phase == 'error_audit' else
                      (READOUT, 'readout.py') if phase == 'readout' else
                      (CONFIRMATION, 'confirmation.py') if confirming else (shared.SOURCE, 'run.py'))
    result = shared.ROOT / ('confirmation/' + str(day) if day is not None else phase) / 'summary'
    item = shared.spec(label, script, [], result=result)
    item.update(description=f'FreqDuet V413 public-slot {label}', signature=f'FreqDuet/v413/{label}')
    item['cwd'] = str(source)
    item['env'].pop('PYTHONPATH', None)
    cmd = [shared.deployment()['python'], '-B', '-I', '-u', str(source / script)]
    if phase in ('component_checks','component_parity'):
        cmd += [phase.removeprefix('component_')]
    elif phase not in ('readout', 'error_audit', 'component_audit', 'component_readout', 'continuation_audit',
                       'running_shadow_checks', 'running_shadow_parity'):
        cmd += [phase]
    cmd += ['--root', str(shared.ROOT)]
    if phase == 'running_shadow_checks':
        cmd += ['--parent', str(shared.ROOT)]
    if day is not None:
        cmd += ['--day', str(day)]
    item['cmd'] = shlex.join(cmd)
    return item


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', nargs='?', choices=('preflight', 'parity', 'verify', 'confirmation', 'readout', 'error_audit', 'component_audit', 'component_checks', 'component_parity', 'component_readout', 'continuation_audit', 'running_shadow_checks', 'running_shadow_parity'),
                        default='preflight')
    phase = parser.parse_args().phase
    if phase == 'preflight':
        if not (HERE / f'submission_{PREFLIGHT}.json').exists():
            stage(shared.SOURCE, ('observer.py', 'slot_checks.py', 'run.py', 'protocol.json', 'PROTOCOL.md'))
        shared.wait(shared.submit(PREFLIGHT, [job(PREFLIGHT, 'preflight')]))
    elif phase == 'parity':
        protocol = shared.read(HERE / 'protocol.json')
        gate = shared.read(HERE / 'review' / PREFLIGHT / 'results.json')
        assert gate['passed'] and gate['physical_starts'] == 0
        assert gate['parity_day'] == protocol['engineering_day']
        assert protocol['planned_parity_paths'] == 1 and not protocol['automatic_retry']
        print(shared.submit('parity', [job('parity', 'parity')]), flush=True)
    elif phase == 'verify':
        parity = shared.read(HERE / 'review/parity/results.json')
        assert parity['passed'] and not parity['confirmation_started']
        if not (HERE / 'submission_verify.json').exists():
            stage(CONFIRMATION, ('confirmation.py',))
        shared.wait(shared.submit('verify', [job('verify', 'verify')]))
    elif phase == 'confirmation':
        protocol = shared.read(HERE / 'protocol.json')
        gate = shared.read(HERE / 'review/verify/results.json')
        days = protocol['confirmation_days']
        assert gate['passed'] and gate['confirmation_days'] == days and gate['slots']['schedule_exact']
        assert len(set(days)) == len(days) == protocol['planned_confirmation_paths'] == 8
        assert not protocol['automatic_retry']
        if not (HERE / 'submission_confirmation.json').exists():
            signatures = {f'FreqDuet/v413/confirmation_{day}' for day in days}
            assert not any(t.get('signature') in signatures for t in shared.scheduler.load_state()['tasks'])
        tasks = [job(f'confirmation_{day}', 'case', day) for day in days]
        print(shared.submit('confirmation', tasks), flush=True)
    elif phase == 'readout':
        protocol = shared.read(HERE / 'protocol.json')
        for day in protocol['confirmation_days']:
            row = shared.read(HERE / f'review/confirmation_{day}/results.json')
            assert row['completed'] and row['day'] == day and row['slots']['schedule_exact']
        if not (HERE / 'submission_readout.json').exists():
            stage(READOUT, ('readout.py',))
        shared.wait(shared.submit('readout', [job('readout', 'readout')]))
    elif phase == 'error_audit':
        report = shared.read(HERE / 'review/readout/results.json')
        assert report['complete'] and report['paid_confirmation_paths'] == 8
        if not (HERE / f'submission_{ERROR_LABEL}.json').exists():
            stage(ERROR_AUDIT, ('error_audit.py',))
        shared.wait(shared.submit(ERROR_LABEL, [job(ERROR_LABEL, 'error_audit')]))
    elif phase == 'component_audit':
        report = shared.read(HERE / f'review/{ERROR_LABEL}/results.json')
        assert report['complete'] and report['physical_starts'] == 0
        if not (HERE / f'submission_{COMPONENT_LABEL}.json').exists():
            stage(COMPONENTS, ('components.py','component_audit.py'))
        shared.wait(shared.submit(COMPONENT_LABEL, [job(COMPONENT_LABEL, 'component_audit')]))
    elif phase == 'component_checks':
        report = shared.read(HERE/f'review/{COMPONENT_LABEL}/results.json')
        assert report['complete'] and report['physical_starts'] == 0
        if not (HERE/'submission_component_checks.json').exists():
            stage(COMPONENT_ENTRY,('component_entry.py','component_parity_protocol.json'))
        shared.wait(shared.submit(phase,[job(phase,phase)]))
    elif phase == 'component_parity':
        gate = shared.read(HERE/'review/component_checks/results.json')
        assert gate['passed'] and gate['physical_starts'] == 0 and gate['physical_start_cap'] == 1
        if not (HERE/'submission_component_parity.json').exists():
            assert not any(t.get('signature') == 'FreqDuet/v413/component_parity'
                           for t in shared.scheduler.load_state()['tasks'])
        print(shared.submit(phase,[job(phase,phase)]),flush=True)
    elif phase == 'component_readout':
        gate = shared.read(HERE/'review/component_parity/results.json')
        assert gate['passed'] and gate['exact_public_slots'] and gate['exact_endpoints']
        if not (HERE/'submission_component_readout.json').exists():
            stage(COMPONENT_READOUT,('component_readout.py',))
        shared.wait(shared.submit(phase,[job(phase,phase)]))
    elif phase == 'continuation_audit':
        report = shared.read(HERE/'review/component_readout/results.json')
        assert report['complete'] and report['physical_starts'] == 0
        if not (HERE/'submission_continuation_audit.json').exists():
            stage(CONTINUATION_AUDIT,('continuation_audit.py',))
        shared.wait(shared.submit(phase,[job(phase,phase)]))
    elif phase == 'running_shadow_checks':
        report = shared.read(HERE/'review/continuation_audit/results.json')
        assert report['complete'] and report['physical_starts'] == 0
        if not (HERE/'submission_running_shadow_checks.json').exists():
            stage(RUNNING_SHADOW, ('running_shadow.py', 'running_shadow_checks.py', 'running_shadow_protocol.json'))
        shared.wait(shared.submit(phase, [job(phase, phase)]))
    else:
        gate = shared.read(HERE / 'review/running_shadow_checks/results.json')
        protocol = shared.read(HERE / 'running_shadow_protocol.json')
        assert gate['passed'] and gate['physical_starts'] == 0 and not gate['confirmation_started']
        assert gate['engineering_day'] == protocol['engineering_day'] == 2520208
        assert protocol['proposed_old_day_paths'] == 1 and not protocol['automatic_retry']
        if not (HERE / 'submission_running_shadow_parity.json').exists():
            assert not any(t.get('signature') == 'FreqDuet/v413/running_shadow_parity'
                           for t in shared.scheduler.load_state()['tasks'])
            stage(RUNNING_PARITY, ('running_shadow_parity.py',))
        print(shared.submit(phase, [job(phase, phase)]), flush=True)
