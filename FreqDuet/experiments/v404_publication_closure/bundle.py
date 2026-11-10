"""Assemble a relocatable source/input runtime, entirely on the server."""

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def build(args, used_sources=None):
    from study import read, write, HERE
    dest = args.root.parent / 'portable_runtime'
    dest.mkdir(exist_ok=False)
    python = dest / 'python'
    python.mkdir()
    environment = Path(sys.prefix).resolve()
    search = [Path(p) for p in sys.path if p and p.startswith('/home/zhengliang01/scheduleurm_work/')
              and not Path(p).resolve().is_relative_to(environment)]
    origins = {name: Path(module.__file__).resolve() for name, module in list(sys.modules.items())
        if getattr(module, '__file__', None) and str(module.__file__).startswith(
            '/home/zhengliang01/scheduleurm_work/') and str(module.__file__).endswith('.py')
            and not Path(module.__file__).resolve().is_relative_to(environment)}
    origins.update(study=HERE / 'study.py', bundle=HERE / 'bundle.py')
    origins.update({name:Path(path) for name,path in (used_sources or {}).items()
                    if str(path).endswith('.py')})
    origins.update(study=HERE/'study.py',bundle=HERE/'bundle.py')
    pending = list(origins.items())
    copied = {}

    def resolve(name):
        for root in search:
            for p in (root.joinpath(*name.split('.')).with_suffix('.py'),
                      root.joinpath(*name.split('.')) / '__init__.py'):
                if p.is_file():
                    return p.resolve()
        return None

    while pending:
        name, src = pending.pop()
        if name in copied or not src.is_file():
            continue
        target = python.joinpath(*name.split('.'))
        target = target / '__init__.py' if src.name == '__init__.py' else target.with_suffix('.py')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        copied[name] = dict(origin=str(src), file=str(target.relative_to(dest)), bytes=src.stat().st_size)
        tree = ast.parse(src.read_text())
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else [])
            for child in names:
                p = origins.get(child) or resolve(child)
                if p is not None and child not in copied:
                    pending.append((child, p))
                if '.' in child:
                    parent = child.rsplit('.', 1)[0]
                    p = origins.get(parent) or resolve(parent)
                    if p is not None and parent not in copied:
                        pending.append((parent, p))
    assets = dest / 'assets'
    for folder in ('basis', 'history', 'support', 'K4', 'K0', 'runtime/model', 'runtime/native_training'):
        shutil.copytree(args.assets / folder, assets / folder)
    for name in ('benchmark_summary.json', 'training_equivalence/summary.json'):
        target = assets / 'runtime/traffic/runtime_optimization_20261014' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.assets / 'runtime/traffic/runtime_optimization_20261014' / name, target)
    shutil.copyfile(HERE / 'protocol.json', python / 'protocol.json')
    shutil.copyfile(HERE / 'portable_entry.py', dest / 'run.py')
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], text=True, capture_output=True, check=True)
    (dest / 'requirements-frozen.txt').write_text(freeze.stdout)
    write(dest / 'source_manifest.json', dict(modules=copied, count=len(copied),
        frozen_inputs='assets/runtime; K4/K0; support', training_updates=0,
        origin_scope='isolated FreqDuet derivative; no original TransitDuet edits',
        redistribution='Review input-data permissions before public distribution; no API credentials included.'))
    return dest


def seal(dest):
    archive = dest.parent / 'freqduet_runtime.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(dest, arcname='freqduet_runtime', filter=lambda info:
                None if '__pycache__' in info.name.split('/') else info)
    source = dest.parent / 'freqduet_source.tar.gz'
    with tarfile.open(source,'w:gz') as tar:
        for name in ('python','run.py','requirements-frozen.txt','source_manifest.json',
                     'assets/basis','assets/history','assets/support'):
            tar.add(dest/name,arcname='freqduet_runtime/'+name,filter=lambda info:
                None if '__pycache__' in info.name.split('/') else info)
    return dict(bundle=str(archive), bytes=archive.stat().st_size,
                source_bundle=str(source), source_bundle_bytes=source.stat().st_size,
                source_files=len(json.loads((dest / 'source_manifest.json').read_text())['modules']))
