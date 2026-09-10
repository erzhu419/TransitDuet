# Anonymous TRC Working Bundle

This flat directory is generated from the frozen Protocol V6 evidence package.
On a server with the isolated paper toolchain, run:

```bash
./build.sh
```

The command writes `manuscript.tex`, `manuscript.pdf`, `supplementary.tex`, and
`supplementary.pdf`. A successful build means that the prose, citations,
tables, and figure paths compile. It does not override the failed V9
long-training gate. Before submission, replace anonymous metadata and re-check
the current TRC submission portal requirements.
`highlights.txt` is the source text for four Elsevier-length bullets; convert it
to the portal's required upload format at final submission.
