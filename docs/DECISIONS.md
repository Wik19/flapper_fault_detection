# Decisions log

Each entry: what was decided, why, and where it lives in the code or data.

## D1 — Task: binary healthy vs damaged (2026-10-03)
- One question only: is the wing set healthy or damaged? Damage *type* is not classified.
- **Labels** (`data/manifest.csv`, column `label`): `healthy` = all wings intact;
  `damaged` = any of `hole1`, `hole2`, `tear_n_hole2`; `excluded` = `taped`.
- **Tape excluded:** a taped wing is repaired but not pristine, so it is neither class.
  It may later serve as an out-of-distribution test (does the model call a repaired wing healthy?).

## D2 — Damage codes (2026-10-03)
Damage was applied progressively to the same wing set, in this order:

| Code | Physical state |
|---|---|
| `none` | 4 intact wings |
| `hole1` | one hole in one wing |
| `hole2` | `hole1` + a hole in the diagonally opposite wing |
| `tear_n_hole2` | `hole2` + a tear in a third wing |
| `taped` | all damage patched with tape |

## D3 — Campaigns are wing sets (2026-10-03)
- Two recording campaigns (sessions on different days). Campaign 1: 2026-05-25.
- Each campaign started from a **fresh wing set**, damaged progressively, so campaign ≡ physical
  wing set. Consequences:
  - The demonstrated claim is generalisation to **unseen flights** of these wing sets, not unseen wings.
  - Within a campaign, damage level is confounded with recording order (healthy flights always first).
  - Campaign 1 is 1 healthy / 8 damaged recordings, campaign 2 is 10 / 4: campaign and label are
    correlated, so results must also be checked within campaign 2 alone.

## D4 — Data provenance (2026-10-03)
- `data/raw/` holds the files the pipeline uses; `data/original/` holds the untouched recordings.
  Both are read-only (enforced in `.claude/settings.json`).
- **Crop recovery method:** the legacy crop script copied samples verbatim, so each cropped file
  appears unchanged inside its original. Crop times were found by locating the exact sample run
  (audio) and the exact row block (IMU); the IMU row offset equalled start × 416 Hz in every case.
- `original_file` and crop times in the manifest are filled **only when verified this way**.
  Campaign 2 originals are pending.
