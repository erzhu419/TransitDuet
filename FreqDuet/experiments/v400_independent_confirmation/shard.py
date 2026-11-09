"""One sequential shard; every physical case gets a fresh process and one attempt."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--shard',type=int,choices=range(8),required=True)
    args=parser.parse_args()
    source=Path(__file__).parent
    plan=json.loads((source/'manifest.json').read_text())
    assert json.loads((args.root/'engineering_gate/summary/admitted.json').read_text())['passed']
    cases=[(a['id'],d) for d in plan['days'] for a in plan['arms']]
    completed=[]
    rc=0
    start=time.perf_counter()
    try:
        for index in range(args.shard,len(cases),8):
            arm,day=cases[index]
            rc=subprocess.run([sys.executable,'-u',str(source/'evaluation.py'),
                '--root',str(args.root),'--assets',str(args.assets),'--arm',arm,'--day',str(day)],
                env=os.environ.copy()).returncode
            if rc:
                break
            completed.append({'index':index,'arm':arm,'day':day})
    except Exception:
        rc=1
        raise
    finally:
        out=args.root/f'shard_{args.shard}/summary'
        out.mkdir(parents=True,exist_ok=True)
        value={'completed':rc==0 and len(completed)==10,'cases':completed,
               'returncode':rc,'wall_s':time.perf_counter()-start,'training_updates':0,'automatic_retry':False}
        with (out/'finished.json').open('x') as f:
            json.dump(value,f)
    sys.exit(rc)


if __name__ == '__main__':
    main()
