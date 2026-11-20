# TMLR review submission

This build uses the unchanged [official TMLR style files](https://github.com/JmlrOrg/tmlr-style-file/tree/7bf90efe3a0debbba703c05c43f3ff7e4d4a2992) and the bibliography style shipped with them. See the [author guide](https://www.jmlr.org/tmlr/author-guide.html) and [submission entry](https://jmlr.org/tmlr/submissions.html).

From the artifact root:

```sh
make -C paper tmlr-package
```

- Manuscript upload: `paper/tmlr/submission.pdf`, compiled in the default anonymous review mode.
- Supplemental upload: `/tmp/tmlr-supplement.zip` (override with `TMLR_SUPPLEMENT=/absolute/path.zip`). It is an offline analysis and reproduction artifact, not a retraining package.

The PDF contains the appendix after the bibliography. TMLR sets no hard
page limit, but its [FAQ](https://www.jmlr.org/tmlr/faq.html) notes that papers
with a main body over 12 pages can take longer to review. Supplementary
material must be anonymous, in PDF or ZIP format, and at most 100 MB.

The submitting authors supply the private OpenReview author, conflict,
funding and research-ethics fields, confirm that no archival venue is
reviewing substantially overlapping work, and approve the final text.
