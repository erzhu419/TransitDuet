"""Record installable runtime versions without touching the active environment."""

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from study import write

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--assets',type=Path,required=True)
a=p.parse_args()
out=a.root/'release_metadata/summary'
out.mkdir(parents=True,exist_ok=True)
versions={}
for name in ('numpy','pandas','scipy','torch','gym','pygame','PyYAML','openpyxl',
             'xlrd','matplotlib','tqdm','tensorboard'):
    try:
        versions[name]=version(name)
    except PackageNotFoundError:
        continue
assert all(name in versions for name in ('numpy','pandas','scipy','torch','gym','pygame','PyYAML','openpyxl'))
(out/'requirements-runtime.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n')
write(out/'results.json',dict(completed=True,versions=versions,physical_starts=0,
    training_updates=0,environment_modified=False,
    recipe='Python 3.10; use the official CPU PyTorch index for the torch pin; then install remaining pins.',
    clean_install_tested=False))
