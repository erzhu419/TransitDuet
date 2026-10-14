# Data and Code Availability

The repository contains the compact CSV and JSON artifacts supporting the V8
confirmation, V9 robustness test, V33 timescale screen, and V9 external-baseline
comparison. It also records exact seed contracts, resolved configurations,
source commits, derived figure data, and assembly scripts. Checkpoints and full
training logs are not part of the manuscript bundle. A persistent archival
identifier will be added after the manuscript scope and release snapshot are
frozen.

The external realism audit uses bounded subsets derived from public MTA AFC and
Halifax Transit APC sources. Source endpoints, selection rules, coverage, and
derived hourly profiles are recorded in the packaged data manifests. MTA Bus
Time route, stop, and AVL records are stored in a separate offline cache; the API
credential used for download is neither stored in the repository nor required
to rebuild the manuscript from the derived data. These public data support only
the descriptive demand-shape audit. They are not same-day AFC/APC/AVL
calibration or observed field outcomes for the simulated network.
