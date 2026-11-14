"""Check the current TeX build for missing references and clipped text."""

import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
OUT = HERE / 'output/pdf'
pages = {}
for name in ('main', 'supplementary'):
    log = (OUT / (name + '.log')).read_text()
    errors = re.findall(r'Overfull[^\n]*|[^\n]*undefined[^\n]*|[^\n]*multiply defined[^\n]*', log)
    assert not errors, (name, errors)
    info = subprocess.check_output(['pdfinfo', str(OUT / (name + '.pdf'))], text=True)
    pages[name] = int(re.search(r'^Pages:\s+(\d+)', info, re.MULTILINE).group(1))
bib = (OUT / 'main.blg').read_text()
assert 'Warning--' not in bib and 'error message' not in bib.lower()
evidence = json.loads((HERE / 'evidence_check.json').read_text())
assert evidence['passed'] and evidence['estimate_cells'] == 37
supplement = json.loads((HERE / 'supplement_evidence_check.json').read_text())
assert supplement['passed'] and supplement['mean_interval_cells'] == 32
assert supplement['distinct_factorial_and_component_cells'] == 224
followup = json.loads((HERE / 'evidence_manifest.json').read_text())['forecast_followup']
assert len(followup['cells']) == 42 and followup['confirmed_forecast_only']
assert not followup['v415_closed_loop_gate'] and not followup['v416_development_gate']
result = dict(passed=True, pages=pages, estimate_cells=37, supplemental_mean_interval_cells=224, undefined_references=0,
              forecast_followup_mean_interval_cells=len(followup['cells']),
              overfull_boxes=0, bibliography_warnings=0, statistical_resamples=0)
(HERE / 'build_check.json').write_text(json.dumps(result, indent=2) + '\n')
print('PAPER_BUILD_OK', pages, 'pages; references and table evidence checked')
