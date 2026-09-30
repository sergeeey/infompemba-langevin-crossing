# The 2026-09-29 rewrite of the git history

**What happened.** After the repository was made public on 2026-09-29, one unrelated tracked file (draft correspondence
with named third parties, committed in the earliest baseline snapshot) was found in it. At the author's instruction it
was removed from the whole history with `git filter-repo --path <file> --invert-paths`, and the remote branches and the
tag were recreated. Because every commit after the baseline descends from the one that contained the file, **every
commit identifier changed**. The content of every remaining file at every commit is unchanged.

**What this means for the record.**
- Short hashes quoted in earlier versions of the note, in `prereg_v2/RESULTS.md`, `prereg_v2/EXECUTION_LOG.md` and
  `PREREG_v2.md` refer to the *old* history and no longer resolve. They are annotated in place as "old [now new]".
- The mapping below was reconstructed by commit order and commit subject. It is **not** a signed mapping of the old
  git objects (those no longer exist in the repository). Treat it as a reading aid, not as cryptographic evidence.
- The frozen pre-specification is unchanged: `PREREG_v2.sha256` equals the SHA-256 of `PREREG_v2.md` as stored at the tag
  `prereg-v2-frozen` (checked 2026-09-30: `ca40708f3343d7b5990eddcff3c8fd55bce25560b5514159e40d9f0ad5bb53d6`).
  The current `PREREG_v2.md` differs because deviations D1–D2.12 were appended after the freeze.
- The annotated tag object is now `f3406fc61fcec90652638b514b61e6dcc5be85f1`, pointing at commit
  `5cd352b3fbac3747c41eb8cfe6b0fba711c6da68`. The Zenodo snapshot of the frozen tag (10.5281/zenodo.23039440) was made
  after the rewrite, from this tag.
- The rewrite happened after all analyses in the note were finished and does not date or authenticate the
  pre-specification in either direction (git dates are author-controlled).

| Purpose | Old short hash | Current short hash |
|---|---|---|
| baseline snapshot | `3e002b6` | `189f1eb` |
| freeze of the pre-registration (tag `prereg-v2-frozen`) | `db3de4a` | `5cd352b` |
| D1: solver method deviation, tolerances (before the solver code) | `2ac799e` | `a2ee495` |
| solver and validation gates V1–V3, G1, G2 | `ce807a5` | `8e6acd2` |
| D2: main-run protocol and controls (before code) | `7d71290` | `6821a80` |
| D2.6: implementation clarifications | `dc145c9` | `4c9b196` |
| main run, controls, results, literature check, draft note | `a2c0889` | `b82d3db` |
| D2.8: sweep definition (before code) | `bde4119` | `f747f17` |
| D2.9: attribution / W1 control / equivalence definitions (before code) | `6c96eb9` | `c2b42c5` |
| post-review analyses, note v0.2 | `f2f30f1` | `759f480` |
| D2.10: margin correction and refinement check (before code) | `52b6d7b` | `0a44ade` |
| margin-statistic correction, refinement check, note v0.3 | `d013b8c` | `b17dbcf` |
| post-review corrections, note v0.4 | `6fed4e9` | `86b7901` |

Commits made after the rewrite have no "old" hash.
