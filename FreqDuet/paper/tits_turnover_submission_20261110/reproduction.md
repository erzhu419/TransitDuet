# Reproduction and release inventory

## Manuscript

`build.sh` formats the archived estimates and compiles the main manuscript and
supporting information. The original V400--V403 figures and tables retain their
original samples and gates. V404 is a separate preregistered operating-condition
and factorial supplement, with four independent demand days in each condition.
`evidence_manifest.json` records the numerical fields and source-data locations.
No local simulation or bootstrap is needed for a manuscript rebuild.

## Traffic runtime

The V404 server-side release builder copies the actually imported first-party
modules and their available static import closure into `python/`, preserving
package paths. It copies the frozen native inputs, model artifacts and optimized
runtime proof records into `assets/`; installed third-party packages remain
dependencies, not vendored source. `requirements-frozen.txt` records the dedicated
Python 3.10 environment. Its conda build URLs make it an inventory rather than
a portable pip recipe. The accompanying `requirements-runtime.txt` pins the
packages needed by the actual runtime and analysis using installed version
metadata. A clean reinstall has not been tested. No system/default CUDA or
cuDNN package is altered.

The independent `run.py` entrypoint sets only the bundle's source/input locations.
Its `import_check` and `native_engineering` modes remove the historical
PYTHONPATH and reject any first-party module still loaded outside the bundle.
The engineering run and manifest determine whether traffic reproduction passed;
an offline TeX build alone does not establish that result.

All 144 formal V404 paths passed from this standalone snapshot. The sealed
archive was extracted under a second server directory and passed its isolated
import check there (`evidence/runtime_release.json`). This relocation adds no
traffic run. The source-only release is in `FreqDuet/runtime_release/v404`.
The full source/input archive remains at the server path in that receipt.

The core V404 aggregation completed before an unused diagnostic export failed
on the native initial-pool `-inf` return sentinel. The failed partial export is
retained server-side. Acceptance reused the complete statistics unchanged,
without traffic or resampling (`evidence/v404_aggregation_acceptance.json`).
The packaged analysis source removes that unused export; it changes no control
code or statistic. `analysis_sources/README.md` describes source-data rebuilds.

Commands inside the assembled runtime:

```bash
python -I run.py import_check
python -I run.py case --root /path/to/fresh/output --regime nominal --day 2440001 --arm pressure
```

Run an authorized case once in a fresh output directory. Do not overwrite an
old run or treat another execution of a consumed day as independent evidence.
Real experiments still belong on scheduler-managed servers. The archive does
not download MBTA/MTA data or require any transit API key.

## Public release decisions

Compact figure source data, result JSON, manuscript source and control/source
entrypoints are part of the prepared paper materials. Raw traces and frozen
input/model files stay on the research server, not the local disk.
The technical archive is not a public repository deposit. An author must still
confirm the redistribution rights for the inherited demand/route spreadsheets,
the software licence, author information and the final public repository record.
No licence, access restriction or DOI has been invented to fill those fields.
