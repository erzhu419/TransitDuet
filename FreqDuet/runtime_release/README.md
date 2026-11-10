# Frozen Runtime Source

`v404/` is the standalone source-only snapshot used for all 144 V404 formal
paths. It includes 129 collected first-party Python modules and source helpers.
The full archive remains on the research server:

`/home/zhengliang01/scheduleurm_work/FreqDuet-v404-publication-closure/freqduet_runtime_release.tar.gz`

Use that full archive for offline traffic reproduction. The source-only tree
does not contain the native spreadsheets or model/checkpoint inputs; its traffic
entrypoint requires those assets restored from the full archive. It is not a
replacement default for historical experiment code or the current main policy.

The sealed archive passed an isolated import check after relocation. All formal
paths used the standalone source in the existing dedicated CPU environment;
a clean dependency reinstall was not tested. See the paper's
`evidence/runtime_release.json` and `reproduction.md` for the release boundary.
The table/figure analysis source is supplied separately with the manuscript.
