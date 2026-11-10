"""Seal the tested source/input bundle; keep the earlier package untouched."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from study import read,write

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--assets',type=Path,required=True)
a=p.parse_args()
data=read(a.root/'aggregate/summary/results.json')
extra=read(a.root/'extra/summary/results.json')
assert data['completed'] and data['physical_pairing_checks']==144
assert extra['charged_start_account']['total']==146
dest=a.root.parent/'portable_runtime'
shutil.copyfile(a.root/'release_metadata/summary/requirements-runtime.txt',dest/'requirements-runtime.txt')
(dest/'README.md').write_text('''# FreqDuet frozen traffic release

This standalone source/input snapshot ran the V404 frozen nine-arm, four-condition
study. No controller source is changed by this packaging step. The source-only
archive omits the native input spreadsheets and model/checkpoint assets.

Use Python 3.10 in an independent CPU environment. `requirements-runtime.txt`
contains installed version pins; use the official CPU PyTorch wheel index for
torch. `requirements-frozen.txt` is the original conda-environment inventory,
not a pip recipe because some entries contain build-machine URLs. Source-path
isolation was tested in the dedicated existing environment; a clean reinstall
has not been tested.

```bash
python -I run.py import_check
python -I run.py case --root /path/to/fresh/output --regime nominal --day 2440001 --arm pressure
```

Run traffic only on scheduler-managed servers at high priority. This example
day is consumed; another execution is reproduction, not an independent sample.
The frozen design is in `python/protocol.json`. The source manifest records
the actual frozen module origins. Manuscript tables, figures and their compact
source data are supplied in the separate paper package.

The archive runs offline and needs no transit API credential. A technical
archive does not settle inherited input-data redistribution permissions, the
software licence, author details or a public repository deposit.
''')
out=a.root/'release/summary';out.mkdir(parents=True,exist_ok=True)
def keep(info):
    return None if '__pycache__' in info.name.split('/') else info
complete=a.root.parent/'freqduet_runtime_release.tar.gz'
with tarfile.open(complete,'w:gz') as tar:
    tar.add(dest,arcname='freqduet_runtime',filter=keep)
relocated=a.root/'release/relocated'
relocated.mkdir(exist_ok=False)
with tarfile.open(complete,'r:gz') as tar:
    tar.extractall(relocated)
subprocess.run([sys.executable,'-I',str(relocated/'freqduet_runtime/run.py'),
    'import_check','--root',str(a.root/'release/relocated_probe')],check=True)
source=out/'freqduet_source_release.tar.gz'
with tarfile.open(source,'w:gz') as tar:
    for name in ('python','run.py','README.md','requirements-runtime.txt',
            'requirements-frozen.txt','source_manifest.json','assets/basis','assets/history','assets/support'):
        tar.add(dest/name,arcname='freqduet_runtime/'+name,filter=keep)
write(out/'results.json',dict(completed=True,full_server_bundle=str(complete),
    full_bytes=complete.stat().st_size,source_only_bytes=source.stat().st_size,
    source_files=read(dest/'source_manifest.json')['count'],
    physical_starts=0,training_updates=0,controller_source_changed=False,
    relocated_import_passed=True,
    formal_paths_from_standalone_source=144,clean_reinstall_tested=False,
    frozen_input_files=[str(p.relative_to(dest)) for p in sorted((dest/'assets/runtime').rglob('*'))
                        if p.is_file() and '__pycache__' not in p.parts]))
