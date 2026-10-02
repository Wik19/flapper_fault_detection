# Flapper Wing Fault Detection

Multimodal (acoustic + inertial) fault detection for a flapping-wing drone.
This README is written as a **history of the project's methodology** — what was
built, why it was wrong, and what replaced it — so it can serve as a record for
the accompanying master's thesis. Read it top to bottom; each phase exists
because the previous one failed (or fell short) in an instructive way.

---

## TL;DR

| Phase | Model / protocol | Reported result | Verdict |
| --- | --- | --- | --- |
| 1. Original | 5-class, ~1M params, shuffled split | ~95% accuracy | **Mirage** — data leakage |
| 2. Honest eval | 5-class, group-aware split | ~25% accuracy | **Honest but unusable** — too little data |
| 3. Binary, 1 healthy | Binary, leave-one-damaged-recording-out | AUC ≈ 0.81 (late fusion) | **Real signal**, but specificity un-generalizable |
| 4. **Current** | Binary, **stratified group k-fold** (23 recordings) | **AUC ≈ 0.93, balanced acc ≈ 0.89** (late fusion) | **Usable detector** — honest sensitivity *and* specificity |

The headline finding: **the inertial (IMU) channel carries the fault signature**
(late-fusion AUC ≈ 0.93 vs audio-only ≈ 0.76), measured under a leakage-free,
recording-level cross-validation. Phase 3 established the signal was real but
could not honestly estimate *specificity* from a single healthy flight. Phase 4
collected **10 more healthy and 4 more damaged flights** (now 11 + 12 = 23
recordings, roughly class-balanced), which lifted specificity from ~0.2 to
**0.86** and turned the classifier into a genuinely usable detector.

---

## Phase 1 — The original 5-class classifier, and why 95% was a mirage

**What it was.** Five classes derived from filenames — `Healthy`, `hole1`,
`hole2`, `tear_n_hole2`, `fixed_all_tape` — fed to two ~1M-parameter CNNs:

- **Early fusion:** mel-spectrograms (audio) and STFT spectrograms (IMU) resized,
  stacked as channels, and passed through a single 2D CNN.
- **Late fusion:** audio through a 2D CNN, IMU through a 1D CNN, feature vectors
  concatenated at the classification head.

The recordings were cut into **3 s windows with a 1 s hop** (consecutive windows
overlap by 67%), and the resulting windows were shuffled and split into
train/validation. Validation accuracy reached **~95%**.

**Why it was bad — data leakage.** Because the split was done *after* windowing,
near-identical overlapping windows from the *same* recording landed in both the
training and the validation set. Windows 1 s apart differ by only a third of a
second of flight, so the validation set was effectively a paraphrase of the
training set. The network did not have to learn fault physics; it only had to
recognise *"which recording is this window from?"* — i.e. mic placement, ambient
room noise, battery state, and the identity of the one physical wing used for
that class. The 95% measured **memorisation of recording fingerprints**, not
generalisation to unseen wings. This is the single most important cautionary
result in the project and the reason for everything that follows.

---

## Phase 2 — Honest evaluation, and the hard truth about the dataset

To remove the leakage, the splitting was made **group-aware**
(`src/data_tools/splitting.py`):

- Classes with **≥2 recordings** hold out **whole recordings** for validation, so
  train and validation never share a recording — overlapping windows can no
  longer leak.
- Classes with a **single recording** fall back to a **temporal split with a
  guard gap**: the validation segment is taken from a different part of the
  recording, with a gap wide enough that no training window overlaps a
  validation window. This is strictly weaker (same recording, different time) and
  is the best achievable for those classes without new data.

Three robustness fixes were applied at the same time:

- **Per-window audio RMS normalisation** — removes recording loudness as a
  shortcut feature.
- **Per-channel IMU standardisation** — accelerometer and gyroscope live on very
  different scales.
- **Inverse-frequency class weighting** — so the rare classes are not drowned out.

**The honest result: ~25% validation accuracy** for both architectures (chance
for 5 classes is ~20%), while training accuracy still reached ~98%. The lesson is
not "tune harder." With **1–4 recordings per class** (two classes have exactly
one), there is essentially nothing to generalise *across*: the model can only
memorise. The leaky 95% had been hiding a dataset far too small for an honest
5-way problem.

> The earlier ~95% numbers should not be cited. They are an artifact of leakage.

---

## Phase 3 — The binary detector with a single healthy recording

Rather than chase five classes with insufficient data, the problem was reframed
to the question that is actually answerable and operationally useful:
**is this wing healthy or damaged?**

### Design decisions (each traces back to a Phase 1/2 failure)

- **Two classes only:** `Healthy` vs `Damaged`. `fixed_all_tape` is **excluded**
  (a tape-repaired wing is neither pristine nor clearly faulty).
- **Lean models with global average pooling** (~24k–46k params instead of ~1M).
  Smaller capacity → far less room to memorise recording-specific noise.
  See `src/architectures/binary_net.py`.
- **Leave-one-damaged-recording-out cross-validation**
  (`src/cross_validate.py`). At this stage there were 1 healthy + 8 damaged
  recordings. Each of the 8 damaged recordings was rotated out, one at a time, as
  a held-out **unseen wing**; the model trained on the other 7 (plus part of the
  healthy recording) and was tested on the wing it had never seen. This measures
  *generalisation to new damage*, which is the whole point.
- **The single healthy recording** was split once in time, with a guard gap, and
  its held-out tail added to every fold's test set so each fold could report
  specificity — the Phase-2 single-recording fallback, reused.
- **Augmentation** (audio gain/noise/time-shift + SpecAugment; IMU scale/noise/
  shift) and a **`WeightedRandomSampler`** to balance the ~6:1 class imbalance per
  batch.
- **Threshold-free metrics:** sensitivity, specificity, balanced accuracy, and
  **ROC-AUC**, plus a leak-safe **calibrated decision threshold** (Youden's J on
  each fold's own training data) since the default 0.5 cutoff is meaningless under
  imbalance.

### Results (leave-one-damaged-out, 8 folds; window 3 s, hop 1 s)

| Model | Sensitivity | Specificity | Balanced acc | **ROC-AUC** |
| --- | :--: | :--: | :--: | :--: |
| Audio only | 0.55 | 0.44 | 0.49 | **0.58** |
| **Late fusion (audio + IMU)** | **0.98** | 0.24 | 0.61 | **0.81** |

**Interpretation.**

- **The IMU is doing the work.** Audio-only barely beat chance (AUC 0.58); adding
  the IMU lifted it to 0.81. The fault signature lives primarily in the wing's
  *motion*, not its *sound*.
- **AUC ≈ 0.81** means: given a random damaged window and a random healthy window,
  the model ranks the damaged one as more suspicious ~81% of the time — on a wing
  it has never seen. A genuine, leakage-free result.
- **Sensitivity high (0.98), specificity low (0.24).** The low specificity was
  *expected and was a data artifact*: there was only **one** healthy recording,
  evaluated against itself (a different time segment of the same flight). With one
  healthy example the model had barely learned what "healthy" looks like *in
  general*. Threshold calibration and healthy-window densification each moved it a
  few points at best — **the real fix was collecting more healthy recordings,
  which is Phase 4.**

### Leakage verification

Because Phase 1 was destroyed by leakage, the binary pipeline was audited with
four independent checks: (A) exact byte-containment between files, (B) per-fold
train/test window disjointness, (C) the healthy temporal guard gap at the sample
level, and (D) cross-correlation between similarly-named damaged recordings.
**All clean** — no window appears in both train and test of any fold, and no file
is a re-cut slice of another.

### What did *not* work (recorded for completeness)

Densifying the single healthy recording with **overlapping windows** (37 → 148
healthy training windows) did **not** improve specificity (within noise).
Overlapping windows from one recording add more *views of the same flight*, not
new information about what healthy looks like in general — and the class sampler
already compensated for raw count. The bottleneck was healthy **variety**, which
only new recordings could supply. (This option has since been removed from the
code; it was a dead end.)

---

## Phase 4 — Multiple recordings and honest specificity (current system)

Phase 3 diagnosed the binding constraint precisely: **one** healthy flight. So a
second data-collection campaign was run — same lab, same caged non-stationary
flight, same body-centre microphone — adding **10 healthy sessions**
(`H1S1…H1S10`) and **4 damaged sessions** (`Hole1S1`, `Hole1S2`, `Hole2S1`,
`Hole2S2W12`). The corpus is now:

- **11 healthy** recordings (`Healthy1` + `H1S1…H1S10`) — ~496 windows
- **12 damaged** recordings — ~493 windows
- **≈ class-balanced** (was ~6:1 damaged), so accuracy is no longer degenerate.

### What changed in the protocol

With healthy no longer a single flight, the single-recording temporal split (and
its healthy-densification hack) were **retired** in favour of a proper
**stratified group k-fold** at the recording level (`src/cross_validate.py`):

- The **recording is the unit** — every window of a recording stays together, so
  a recording is entirely in train *or* entirely in test (overlapping-window
  leakage remains impossible).
- Recordings are shuffled and round-robin-assigned to **K = 5 folds**, *stratified
  by class*, so each held-out test fold contains a **mix of unseen healthy and
  damaged recordings**. Every fold therefore reports honest sensitivity,
  specificity, balanced accuracy *and* ROC-AUC — all on recordings never trained
  on. **Specificity is finally a real generalisation estimate**, not a
  same-recording artifact.
- The deployable model is then retrained on all 23 recordings, with its decision
  threshold taken from the pooled out-of-fold predictions (leak-safe, and not
  read off the model's own overconfident training scores).

Two data-handling fixes landed here too: the labeller now recognises the
`H<n>S<n>` healthy-session naming (`src/data_tools/binary_data.py`), and the IMU
CSVs (which carry a `Packet_ID,Rx_Time_Sec,Acc_XYZ,Gyro_XYZ` header) are read with
`imu_has_header=True` so the header row no longer leaks into each recording's
first window.

### Results (stratified group 5-fold CV; window 3 s, hop 1 s, 30 epochs)

Mean ± std across the 5 folds, at the leak-safe calibrated threshold:

| Model | Sensitivity | Specificity | Balanced acc | **ROC-AUC** |
| --- | :--: | :--: | :--: | :--: |
| Audio only | 0.67 | 0.77 | 0.72 | **0.76** |
| **Late fusion (audio + IMU)** | **0.91** | **0.87** | **0.89** | **0.93** |

Pooled confusion matrix (late fusion, calibrated threshold), 989 windows:

| true ⧵ pred | Healthy | Damaged |
| --- | :--: | :--: |
| **Healthy** | 427 | 69 |
| **Damaged** | 44 | 449 |

**Interpretation.**

- **Specificity 0.19 → 0.87.** The Phase-3 artifact is gone. The model now knows
  what "healthy" looks like *across flights*, not just one.
- **Late fusion is now a usable detector:** AUC 0.93, balanced accuracy 0.89, on
  recordings it has never seen. Both error types are low and comparable
  (69 false alarms, 44 misses out of ~989 windows).
- **The IMU still carries the signal** (late 0.93 vs audio 0.76), consistent with
  Phase 3 — but audio also improved markedly (0.58 → 0.76) once it had more than
  one healthy flight to characterise.
- **Class balance is a side benefit:** the *default* 0.5 threshold now already
  reaches balanced accuracy 0.86 for late fusion; calibration adds a small margin
  (specificity 0.81 → 0.86). Audio still relies on calibration (its probabilities
  saturate — specificity 0.04 → 0.76).
- **Honest variance remains.** One fold (fold 3: `Healthy1` + the two most heavily
  cropped damaged clips) drops to AUC 0.67 while the other four sit at 0.97–1.00.
  This is exactly the kind of per-fold spread that the leaky Phase-1 setup hid.

Full per-fold tables and confusion heatmaps are regenerated to
`results/binary/cv_report.md` and `results/binary/confusion_*.png`.

### Verification (Phase-4 audit)

Because the corpus doubled, the leakage audit was redone from scratch on all 24
files and on the new protocol:

- **File independence:** no two recordings are identical (MD5), no recording is a
  byte-exact slice of another (PCM probe search), and no pair exceeds a rolling
  envelope correlation of r = 0.80 (a re-encoded or gain-changed crop would score
  ~1.0) — in particular, the `*_cropped` files are *not* cuts of the `*_full`
  files. The same checks pass on the IMU CSVs.
- **Fold integrity (seed 0, K = 5):** each of the 23 recordings is tested exactly
  once and never appears in its own training fold; train and test share **zero
  windows and zero files** in every fold; the pooled test set covers all 989
  windows exactly once.
- **Pipeline hygiene:** per-window normalisation only (no dataset-wide statistics);
  augmentation and the class sampler verified active only on the training path;
  threshold calibration reads only training-fold predictions; a fresh seeded model
  per fold.
- **Seed stability:** re-running the late-fusion CV with two different fold
  shuffles gives mean AUC **0.93 / 0.94 / 0.91** (seeds 0/1/2) and pooled balanced
  accuracy **0.89 / 0.91 / 0.88** — the result is not an artifact of one lucky
  partition. Each shuffle also shows one weaker fold (AUC 0.6–0.7), so the honest
  summary is "AUC ≈ 0.9 with real per-fold variance", not a uniform 0.99.

**Scope caveat (for the thesis):** all 11 healthy recordings come from the same
physical wing, so the demonstrated claim is generalisation to **unseen flights**,
not unseen wings or drones. (The new healthy and damaged sessions were recorded
in the same lab sessions, which controls the room/day confound between classes.)

---

## Data specifications

Paired files with identical base names live in `data/raw/MIC` (audio) and
`data/raw/IMU` (inertial):

- **Audio:** `.wav`, mono, **16 kHz**, 16-bit PCM (~11–58 s per flight).
- **IMU:** `.csv`, **416 Hz**, with a header row
  `Packet_ID, Rx_Time_Sec, Acc_X/Y/Z, Gyro_X/Y/Z`; the pipeline uses the last 6
  columns (the 6 inertial channels).

Label assignment is derived from the filename
(`src/data_tools/binary_data.py:binary_label`), case-insensitive:

- `healthy` **or** an `H<n>S<n>` session code (e.g. `H1S7`) → **Healthy**
- a damage word (`hole`/`tear`/`crack`/`broken`/`damage`) → **Damaged**
- `tape` → **excluded** (tape-repair recording)

Current corpus: **11 healthy** + **12 damaged** recordings, plus 1 excluded
tape-repair recording.

---

## How to run (current binary system)

```bash
# Full stratified group 5-fold CV for both architectures,
# then train + save the deployable models and the report.
.venv/bin/python src/cross_validate.py

# Just the strong model (audio + IMU):
.venv/bin/python src/cross_validate.py --arch late

# More folds, a different shuffle, or without synthetic augmentation:
.venv/bin/python src/cross_validate.py --folds 8 --seed 1
.venv/bin/python src/cross_validate.py --no-augment
```

**Outputs**

- `models/binary/best_audio_binary_model.pth`, `models/binary/best_late_binary_model.pth`
  — final models trained on *all* data (each stored with its window/hop config and
  a deploy threshold from the out-of-fold predictions).
- `results/binary/cv_report.md` — readable Markdown report.
- `results/binary/confusion_{audio,late}.png` — pooled confusion heatmaps.

The Phase 1/2 five-class scripts (`train_early.py`, `train_late.py`,
`evaluate.py`) are retained for the historical record and still run, but the
binary pipeline is the current system.

---

## Known limitations & next steps

1. **Corpus size.** 23 recordings is enough for an honest binary estimate but
   still modest; results carry real per-fold variance (see fold 3). More
   independent flights — different wings, mounts, and days — would tighten the
   confidence intervals and let per-condition analysis become meaningful.
2. **Revisit the multi-class problem.** With multiple recordings now available for
   several damage types, the 5-class question (which *kind* of damage) could be
   attempted honestly again under the same recording-level protocol.
3. **Deployment / streaming.** The models decide per 3 s window; a real-time
   detector would aggregate windowed scores over a flight (e.g. a rolling vote)
   and report a single health verdict with a confidence.
