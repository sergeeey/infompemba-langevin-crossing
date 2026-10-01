# Archiving on Zenodo: state and remaining steps (updated 2026-10-01)

## Done

- Zenodo account linked; this repository switched on in the GitHub integration.
- Release `prereg-v2-frozen` (tag at the freeze commit `5cd352b`) archived as **version 10.5281/zenodo.23039440**;
  the **concept DOI for all versions is 10.5281/zenodo.23039439** (cite this one in running text).
  Its metadata were wrong at first (author "C", automatic title, built from a tree without `CITATION.cff`) and were
  **corrected on 2026-10-01**: creator Sergey Boyko (Ronin Institute for Independent Scholarship 2.0, ORCID), readable title,
  MIT licence. The description text still says "note v0.4" (stale; the note is now v0.5) and was not edited.
- Curve dataset published 2026-10-01 as its own record: **10.5281/zenodo.23077448** (all versions 10.5281/zenodo.23077447), type Dataset, MIT licence, six zip
  parts + `SHA256SUMS.txt` + `out_manifest.csv` + `README.txt`, related identifier "is supplement to" 10.5281/zenodo.23039439.
  The nine uploaded files' MD5 sums on Zenodo equal the local ones.
- Pushed to GitHub: all commits up to `ec3e1f8` (note v0.5, reproducibility package, `.zenodo.json`, `CITATION.cff`, `LICENSE`).
- `deposit_staging/` (git-ignored) holds the staged parts; the single 42.5 MB archive built earlier was replaced by the six parts.

## Remaining steps (each a public action; needs the author's explicit go)

1. **Gates before the release that carries the note** (author): a human reads the note and the checks; someone other than the
   author's workflow runs `REPRODUCE.md` sections 1-2 on another machine; the target venue's rule on AI-assisted work is
   checked and the disclosure text approved; 24 hours between "ready" and sending.
2. **New tag at the then-current HEAD and a GitHub release** -> Zenodo builds a new version under concept DOI
   10.5281/zenodo.23039439, this time with correct metadata from `.zenodo.json`. Then put the version DOI into the note and
   `CITATION.cff` (`date-released`).

Order matters: Zenodo archives only releases created after the repository switch is on, from the tree of the tag, and reads
metadata from files inside that tree. Do not hardcode commit hashes in this file.
