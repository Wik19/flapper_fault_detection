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

## D5 — Both sensor streams have hidden gaps (2026-10-04)
- **Finding (firmware, `microcontroller-firmware/src/`).** Samples are dropped silently *before*
  they are packetised:
  - IMU: a 512-sample queue (1.23 s at 416 Hz), filled with timeout 0 (`imu_driver.c:97`).
    If no data-ready interrupt arrives for 100 ms, one sample is read and discarded
    (`imu_driver.c:105`).
  - Microphone: a 64 × 256-sample queue (1.02 s at 16 kHz), filled with timeout 0
    (`mic_driver.c:83`).
  - Sequence numbers are assigned only after these queues (`main.c:123`, `main.c:149`), and the
    link is TCP, which never loses or reorders packets. Continuous `Packet_ID`s therefore do
    **not** imply a complete stream.
- **Evidence (one-off check, to be re-derived by sync check Q2).** For the 19 recordings with
  real receive timestamps, (receive-time span) − (N − 1)/416 Hz is positive in every case:
  1.2–8.5 s in campaign 1 and 0.4–3.9 s in campaign 2. WiFi stalls longer than the IMU queue
  explain part of it. The rest is either further dropping or a true IMU rate below 416 Hz.
- **Correction.** `legacy/LEGACY_AUDIT.md` §1.3 ("zero skipped Packet_IDs … not lost samples")
  is wrong. Row index × (1/416 s) is not a reliable clock.
- **Consequences.**
  - Fixed-length windows cut by row index can straddle a hidden gap.
  - The audio–IMU offset changes within a recording, because each stream loses different
    amounts at different times.
  - Campaign 1 lost more data than campaign 2: one more property correlated with the label (D3).
- **Decision.** Keep the firmware unchanged for now. Quantify the gaps per recording with the
  sync checks (Q2: steady drift vs steps after stalls; Q3: time-varying alignment). Redesign the
  firmware for future recordings only if the flaws turn out to be large.
