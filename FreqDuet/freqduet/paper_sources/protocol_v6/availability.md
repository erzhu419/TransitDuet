# Data and Code Availability

The repository contains the small immutable CSV and JSON artifacts supporting
the V8 confirmation, V9 robustness test, and V9 external-baseline comparison,
together with exact seed contracts, resolved configurations, source commits,
derived figure data, and assembly scripts. Checkpoints and full training logs
remain on the HPC filesystem and are not part of the manuscript bundle. A
persistent archival identifier will be added after the manuscript scope and
release snapshot are frozen.

The external realism audit uses bounded subsets derived from public MTA AFC and
Halifax Transit APC sources. Source endpoints, selection rules, coverage, and
derived hourly profiles are recorded in the packaged data manifests. MTA Bus
Time route, stop, and AVL records are stored in a separate offline cache; the API
credential used for download is neither stored in the repository nor required
to rebuild the manuscript from the derived data. These public data support only
the descriptive demand-shape audit. They are not same-day AFC/APC/AVL
calibration or observed field outcomes for the simulated network.
