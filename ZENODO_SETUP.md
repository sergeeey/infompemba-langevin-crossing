# Archiving on Zenodo: state and remaining steps (updated 2026-09-30)

## Done

- Zenodo account linked; this repository switched on in the GitHub integration.
- Release `prereg-v2-frozen` (tag at the freeze commit `5cd352b`) archived as **version 10.5281/zenodo.23039440**;
  the **concept DOI for all versions is 10.5281/zenodo.23039439** (cite this one in running text).
  That record was built from a tree that predates `CITATION.cff`/`LICENSE`, so its **metadata are wrong**
  (author shown as "C", automatic title, no licence). Its files cannot be changed; its metadata can be edited.
- `CITATION.cff`, `.zenodo.json` (record metadata for the next release), `LICENSE` (MIT), `REPRODUCE.md`,
  `requirements-v2.txt`, `HISTORY_REWRITE.md` are in the tree for the next release.
- Curve archive prepared (not yet uploaded): `deposit_staging/infompemba_prereg_v2_main_run_curves.zip`
  (42.5 MB, SHA-256 `b1cd3ec0e220bea87888a09afbf8e2bea29a705348c2462145ffd5a873bd0229`, every file verified against
  `prereg_v2/out_manifest.csv`). `deposit_staging/` is git-ignored.

## Remaining steps, in this order (each is a public action and needs the author's explicit go)

1. **Push** the local commits of steps A/B to GitHub (plain fast-forward).
2. **Upload the curve dataset** as its own Zenodo record (upload type: dataset; author, ORCID, licence as in `.zenodo.json`;
   related identifier "is supplement to" the software concept DOI). Put its DOI into the note
   ("Data and code availability") and `REPRODUCE.md` section 3.
3. **Edit the metadata of the frozen-tag record** (10.5281/zenodo.23039440): creator Sergey Boyko (Ronin Institute, ORCID),
   readable title, licence MIT. The DOI does not change.
4. **Gates before the release that carries the note** (author): a human reads the note and the checks; someone other than the
   author's workflow runs `REPRODUCE.md` sections 1-2 on another machine; the target venue's rule on AI-assisted work is
   checked and the disclosure text approved; 24 hours between "ready" and sending.
5. **New tag at the then-current HEAD and a GitHub release** -> Zenodo builds a new version under the same concept DOI, this time
   with correct metadata from `.zenodo.json`. Then put the version DOI into the note and `CITATION.cff` (`date-released`).

Order matters: Zenodo archives only releases created after the repository switch is on, from the tree of the tag, and reads
metadata from files inside that tree. Do not hardcode commit hashes in this file.
