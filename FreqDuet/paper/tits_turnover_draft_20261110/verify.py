"""Check the current TeX build for missing references and clipped text."""

import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
OUT = HERE / 'output/pdf'
log = (OUT / 'main.log').read_text()
bib = (OUT / 'main.blg').read_text()
errors = re.findall(r'Overfull[^\n]*|[^\n]*undefined[^\n]*|[^\n]*multiply defined[^\n]*', log)
assert not errors, errors
assert 'Warning--' not in bib and 'error message' not in bib.lower()
evidence = json.loads((HERE / 'evidence_check.json').read_text())
assert evidence['passed'] and evidence['estimate_cells'] == 37
info = subprocess.check_output(['pdfinfo', str(OUT / 'main.pdf')], text=True)
pages = int(re.search(r'^Pages:\s+(\d+)', info, re.MULTILINE).group(1))
result = dict(passed=True, pages=pages, estimate_cells=37, undefined_references=0,
              overfull_boxes=0, bibliography_warnings=0, statistical_resamples=0)
(HERE / 'build_check.json').write_text(json.dumps(result, indent=2) + '\n')
print('PAPER_BUILD_OK', pages, 'pages; references and table evidence checked')
