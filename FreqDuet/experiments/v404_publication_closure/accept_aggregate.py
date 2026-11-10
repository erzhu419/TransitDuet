"""Accept intact statistics after the unused diagnostic export failed."""

import argparse
from pathlib import Path
from study import read,write

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--assets',type=Path,required=True)
a=p.parse_args()
out=a.root/'aggregate/summary'
d=read(out/'results.json')
plan=read(Path(__file__).parent/'protocol.json')
assert d['completed'] and d['physical_pairing_checks']==144
assert list(d['means'])==list(plan['regimes'])
assert all(set(v)==set(plan['arms']) for v in d['means'].values())
assert all(len(v)==4 for arms in d['details'].values() for v in arms.values())
failed=out/'mechanism.json'
saved=a.root/'aggregate/failed_exports/mechanism_t214475.json'
saved.parent.mkdir(parents=True,exist_ok=True)
assert failed.is_file() and not saved.exists()
failed.rename(saved)
write(out/'acceptance.json',dict(passed=True,statistical_results_reused_unchanged=True,
    statistical_resamples=0,physical_starts=0,training_updates=0,
    failed_diagnostic_export=str(saved),failed_task='t214475',
    diagnostic_tail=saved.read_text()[-350:],
    reason='Unused duplicate diagnostic contained a native -inf sentinel; raw traces retained.'))
