# Archiving this repository on Zenodo — steps for the author

Prepared 2026-09-29; not yet done by the assistant, since each step below needs the
author's own login or an author decision. See PENDING items marked `[AUTHOR]`.

## 1. Before enabling Zenodo (author decisions)

- ~~Settle the author name~~ — **done 2026-09-29**: "Sergey Boyko" (git-config spelling),
  affiliation "Independent researcher, Almaty, Kazakhstan", email sergeikuch80@gmail.com.
  Filled into `CITATION.cff` and `paper/note_v2_draft.md` line 6.
- ~~Choose a license~~ — **done 2026-09-29**: MIT (`LICENSE` at repo root, `CITATION.cff`
  `license: MIT`). Covers the code; the note text (`paper/note_v2_draft.md`) is not
  separately licensed — MIT applies to the whole repository as-is. If a text-specific
  license (e.g. CC-BY-4.0 for the prose) is wanted later, add a second file for it.
- `[AUTHOR]` `CITATION.cff` still needs `date-released` (set it when the release is
  actually made).
- `[AUTHOR]` Decide the venue / whether this is ready to be citable at all: the note
  (`paper/note_v2_draft.md`) has not had a 4th skeptic pass, and references [4] and [11]
  are unverified (`prereg_v2/N1_LITERATURE.md` §5). A Zenodo DOI, once minted, is very
  hard to fully retract — Zenodo's own policy is that a deposit can be closed/removed
  in exceptional cases but the DOI record persists. Treat "create the release" as the
  actual publication step, not this preparation.

## 2. Link GitHub and Zenodo (must be done by the author — needs your login/OAuth)

1. Go to https://zenodo.org/ and log in ("Log in with GitHub").
2. Go to https://zenodo.org/account/settings/github/ (GitHub tab under your account).
3. Find `sergeeey/infompemba-langevin-crossing` in the repository list and flip its
   toggle **on**. If it is not listed, click "Sync now".

## 3. Create the GitHub release (only after step 2 — order matters)

Zenodo only archives releases created *after* the repository toggle is switched on;
a release made before that will NOT be picked up automatically. So:

1. Finish step 1 and step 2 first.
2. Then create a GitHub Release (Releases → "Draft a new release"), either reusing the
   existing tag `prereg-v2-frozen` or a new tag pointing at the branch's current HEAD
   (check `git log -1` — do not hardcode a commit hash in this file, it goes stale
   immediately). Title and notes are free text; Zenodo pulls its metadata primarily from
   `CITATION.cff`.
3. Publishing the release triggers Zenodo to archive that snapshot and mint a DOI. The
   DOI page appears at https://zenodo.org/account/settings/github/repository/sergeeey/infompemba-langevin-crossing
   a few minutes after.
4. Optional: add the resulting DOI badge to `README.md` and the DOI itself to
   `paper/note_v2_draft.md`'s Data and code availability section (currently a placeholder).

## What the assistant did already (2026-09-29)

- Made `sergeeey/infompemba-langevin-crossing` public on GitHub (was private).
- Added `CITATION.cff` (author, affiliation, email, license: MIT all resolved by the
  author; `date-released` is the one remaining placeholder, set it at actual release
  time) and `LICENSE` (MIT).
- Found and removed `paper/endorser_emails.md` from the entire git history (real
  third-party emails were exposed when the repo went public) — see `PREREG_v2.md` D2.11
  and `prereg_v2/EXECUTION_LOG.md` (2026-09-29 entry) for the incident record.
- Did not create a GitHub Release and did not touch Zenodo — both require the
  author's own action per the steps above.
