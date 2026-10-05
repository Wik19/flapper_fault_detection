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
- **Update (2026-10-04): recording order.** The originals' file timestamps give the order.
  - Session 1: `Healthy1` → the four `hole1` flights → `hole2` → `tear_n_hole2` → `fixed_all_tape`.
  - Session 2: `H1S1`–`H1S7` (healthy) → `Hole1S1`, `Hole1S2` → `Hole2S1`, `Hole2S2W12` →
    `H2S8`, `H2S9`, `H2S10` (healthy, recorded about 7 minutes after the last damaged flight;
    renamed `H1S8`–`H1S10` by the legacy crop script).
  - **Open question:** holes can't be undone, so the `H2` flights must have used different wings.
    Which ones? If they are a second fresh set, campaign 2 contains two wing sets, and these three
    healthy flights come *after* the damaged ones. That weakens the order confound and gives a
    small held-out-wing test.
- **Resolved (2026-10-05).** No record of that day survives. The user recalls only two wing sets
  in total, so the most plausible explanation is that the holed wings were swapped for intact
  ones before `H2S8`. **Rule:** a recording named `H…` is healthy; the manifest labels stand.
  `H2S8`–`H2S10` are therefore healthy flights on the campaign-2 frame recorded after its damaged
  flights. Revisit when new recordings are made.

## D4 — Data provenance (2026-10-03)
- `data/raw/` holds the files the pipeline uses; `data/original/` holds the untouched recordings.
  Both are read-only (enforced in `.claude/settings.json`).
- **Crop recovery method:** the legacy crop script copied samples verbatim, so each cropped file
  appears unchanged inside its original. Crop times were found by locating the exact sample run
  (audio) and the exact row block (IMU); the IMU row offset equalled start × 416 Hz in every case.
- `original_file` and crop times in the manifest are filled **only when verified this way**.
- **Campaign 2 verified (2026-10-04).** The 9 uncropped files are byte-identical to their
  originals. The 5 crops were recovered the same way, and match the starts predicted from packet
  IDs by sync check Q1. `H1S8`, `H1S9` and `H1S10` were renamed from `H2S8`, `H2S9` and `H2S10`
  by the legacy crop script.

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
- **Update (2026-10-04).** Quantified in `docs/SYNC_CHECKS.md`: 0–7.3 s lost per recording; a
  stall loses (its duration − ~1.25 s), exactly the queue model. The rate question is settled:
  the losses are drops, and the IMU/microphone clocks agree to within ~0.15 s per minute.

## D6 — Exclude windows that contain a hidden gap (proposed 2026-10-04, confirmed 2026-10-05)
- **Rule.** Drop every analysis window that contains the located gap (`drop_at_sample` from
  `hidden_gaps`) of an event that lost more than **0.05 s**. For a burst of stalls (`n_stalls` >
  1), whose earlier gaps are not located, drop every window overlapping the event (from
  `stall_at_sample` to `drop_at_sample`).
- **Why 0.05 s.** 21 samples, less than one wingbeat period (~70 ms at 14 Hz). A shorter gap
  cannot glue together visibly different flight. 73 of the 111 events exceed it.
- **Why "contains the gap" and not a blind zone after each stall.** The gap is located to ±48 ms
  (Q2 step 6, validated against the queue model), so only windows that actually straddle it are
  affected. Cost with 3-s windows and a 1-s hop: **21 %** of windows, against about 35 % for a
  blind 3-s zone after each stall.
- **Side effect.** The excluded share differs by campaign and label (15–30 %); excluding removes
  gaps as a possible shortcut. Report the window counts per class after exclusion.

## D7 — Re-align audio to the IMU before any audio or fusion model (proposed 2026-10-04, confirmed 2026-10-05)
- **Rule.** Shift the audio by the Q3 offset measured for that recording and time: linear
  interpolation between window centres, held constant before the first and after the last window.
  The IMU is the time reference because Q2 locates its gaps.
- **Why.** Offsets reach ±1.4 s and change by up to 0.5 s within a recording, a large fraction
  of a 3-s window (`docs/SYNC_CHECKS.md` §5–6).
- **Exception.** The three recordings with an unreliable offset (`hole1_loss_of_control_cropped`,
  `hole1_loss_of_control2_cropped`, `H1S9`) are left out of audio and fusion experiments.
  IMU-only experiments keep them.

## D8 — Model ladder: same classifier, increasingly free features (2026-10-05)
- **Form.** Every model ends in the same classifier, p(damaged | window x) = σ(wᵀφ(x) + b). The
  rungs differ only in who builds the feature map φ:

  | Rung | φ chosen by | Trained weights |
  |---|---|---|
  | R0 | nothing: the campaign alone (the shortcut to beat) | 0 |
  | R1 | physics: hand-designed features (to be designed next) | ~10–30 |
  | R2 | chance: MiniRocket's fixed random filters (Dempster et al., KDD 2021) | ~10k, regularised |
  | R3 | gradient descent: the legacy `BinaryLateFusionNet`, re-run unchanged | 45.5k |
  | R4 | pre-training on general audio (e.g. PANNs), audio only, optional | ~100–2,000 |

- **Order.** IMU first (R0 → R3), then audio, then fusion by averaging the two models'
  probabilities (no extra weights; each sensor's contribution stays visible).
- **Why.** The independent unit is the recording (n = 23). By Cover's theorem (1965), a linear
  classifier on d features fits a share 2·Σₖ₌₀ᵈ C(n−1, k) / 2ⁿ of all labellings of n points
  perfectly: for n = 23, 0.8 % at d = 5, 58 % at d = 11, 97 % at d = 15, all from d = 22. A
  flexible model can therefore memorise flights; the ladder makes each added freedom earn its
  place, measured within campaign 2.
- **Rules.** All rungs use the same windows (D6), the same protocol and the same metrics. Every
  rung is listed in the evaluation protocol before any is run, and every result is reported.
