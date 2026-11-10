"""Package the offline numerical manuscript, not server runtime artifacts."""

from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
paths = []
for pattern in ('*.tex', '*.bib', '*.cls', '*.md', '*.py', '*.sh', '*.json','requirements-runtime.txt',
                'tables/*.tex', 'evidence/*.json', 'figures/*.pdf', 'figures/*.svg',
                'figures/*.json'):
    paths.extend(HERE.glob(pattern))
for pattern in ('analysis_sources/*.py','analysis_sources/*.json','analysis_sources/*.md'):
    paths.extend(HERE.glob(pattern))
target = HERE / 'freqduet_turnover_source.zip'
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(paths)):
        archive.write(path, path.relative_to(HERE))
print(target.name, target.stat().st_size, 'bytes')
