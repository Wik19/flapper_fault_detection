# Flapper Wing Fault Detection

Multimodal (acoustic + inertial) fault detection for a flapping-wing drone.
This README is written as a **history of the project's methodology** — what was
built, why it was wrong, and what replaced it — so it can serve as a record for
the accompanying master's thesis. Read it top to bottom; each phase exists
because the previous one failed in an instructive way.

---

## TL;DR

| Phase | Model | Reported result | Verdict |
| --- | --- | --- | --- |
| 1. Original | 5-class, ~1M params | ~95% accuracy | **Mirage** — data leakage |
| 2. Honest eval | 5-class, group-aware split | ~25% accuracy | **Honest but unusable** — too little data |
| 3. Current | Binary (Healthy/Damaged), LORO-CV | **AUC ≈ 0.82** (late fusion) | **Real signal**, dominated by IMU |

The headline finding: **the inertial (IMU) channel carries the fault signature**
(late-fusion AUC ≈ 0.82 vs audio-only ≈ 0.58), measured under a leakage-free,
leave-one-recording-out protocol. The remaining limitation is a *data*
limitation (a single healthy recording), not a modelling one.

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

## Phase 3 — The binary detector (current system)

Rather than chase five classes with insufficient data, the problem was reframed
to the question that is actually answerable and operationally useful:
**is this wing healthy or damaged?**

### Design decisions (each traces back to a Phase 1/2 failure)

- **Two classes only:** `Healthy` vs `Damaged`. `fixed_all_tape` is **excluded**
  (a tape-repaired wing is neither pristine nor clearly faulty).
- **Lean models with global average pooling** (~24k–46k params instead of ~1M).
  Smaller capacity → far less room to memorise recording-specific noise.
  See `src/architectures/binary_net.py`.
- **Leave-one-recording-out cross-validation (LORO-CV)** — the centrepiece
  (`src/cross_validate.py`). There are 1 healthy + 8 damaged recordings. Each of
  the 8 damaged recordings is rotated out, one at a time, as a held-out **unseen
  wing**; the model trains on the other 7 (plus part of the healthy recording)
  and is tested on the wing it has never seen. This measures *generalisation to
  new damage*, which is the whole point.
- **The single healthy recording** is split once in time, with a guard gap, and
  its held-out tail is added to every fold's test set so each fold can report
  specificity. (Optionally densified with overlapping windows via `--healthy-hop`
  — see "What did not work".)
- **Augmentation** (audio gain/noise/time-shift + SpecAugment; IMU scale/noise/
  shift) and a **`WeightedRandomSampler`** to balance the ~6:1 class imbalance per
  batch.
- **Threshold-free metrics:** sensitivity, specificity, balanced accuracy, and
  **ROC-AUC**, reported as mean ± std across folds plus a pooled confusion matrix.
  AUC is used because accuracy is meaningless under class imbalance (always
  predicting "Damaged" scores ~72%).

### Results (LORO-CV, 8 folds; window 3 s, damaged hop 1 s, 30 epochs)

| Model | Sensitivity | Specificity | Balanced acc | **ROC-AUC** |
| --- | :--: | :--: | :--: | :--: |
| Audio only | 0.55 | 0.44 | 0.49 | **0.58** |
| **Late fusion (audio + IMU)** | **0.98** | 0.24 | 0.61 | **0.81** |

(Full per-fold tables and confusion matrices are regenerated to
`results/binary/cv_report.md` and `results/binary/confusion_*.png`.)

**Interpretation.**

- **The IMU is doing the work.** Audio-only barely beats chance (AUC 0.58);
  adding the IMU lifts it to 0.81–0.84. The fault signature lives primarily in
  the wing's *motion*, not its *sound*.
- **AUC ≈ 0.82** means: given a random damaged window and a random healthy
  window, the model ranks the damaged one as more suspicious ~82% of the time —
  on a wing it has never seen. A genuine, leakage-free result.
- **Sensitivity is high (0.98), specificity low (0.24).** The low specificity is
  *expected and is a data artifact*: there is only **one** healthy recording, and
  it is evaluated against itself (a different time segment of the same flight).
  With one healthy example the model has barely learned what "healthy" looks like
  in general. The high AUC says the ranking ability exists; the poor
  default-threshold specificity says the 0.5 cutoff is simply in the wrong place
  (fixable by threshold calibration — see next steps).

### Leakage verification

Because Phase 1 was destroyed by leakage, the binary pipeline was audited with
four independent checks: (A) exact byte-containment between files, (B) per-fold
train/test window disjointness, (C) the healthy temporal guard gap at the sample
level, and (D) cross-correlation between similarly-named damaged recordings.
**All clean** — no window appears in both train and test of any fold, and no file
is a re-cut slice of another.

### What did *not* work (recorded for completeness)

Densifying the single healthy recording with **overlapping windows**
(`--healthy-hop 0.25`, 37 → 148 healthy training windows) did **not** improve
specificity (0.26 → 0.24, within noise). Overlapping windows from one recording
add more *views of the same flight*, not new information about what healthy looks
like in general — and the class sampler already compensated for raw count. The
bottleneck is healthy **variety**, which only new recordings can supply.

---

## Data specifications

Paired files with identical base names live in `data/raw/MIC` (audio) and
`data/raw/IMU` (inertial):

- **Audio:** `.wav`, mono, **16 kHz**, 16-bit PCM.
- **IMU:** `.csv`, **416 Hz**, 6 channels (accelerometer x/y/z + gyroscope x/y/z),
  no header.

Label assignment is derived from the filename
(`src/data_tools/binary_data.py:binary_label`): a name containing `Healthy` →
Healthy; `hole`/`tear` → Damaged; `fixed_all_tape` → excluded.

Current corpus: **1 healthy** recording, **8 damaged** recordings (hole /
two-hole / tear-and-hole conditions), plus 1 excluded tape-repair recording.

---

## How to run (current binary system)

```bash
# Full leave-one-recording-out CV for both architectures,
# then train + save the deployable models and the report.
.venv/bin/python src/cross_validate.py

# Just the strong model (audio + IMU):
.venv/bin/python src/cross_validate.py --arch late

# Disable the (neutral) healthy-overlap densification:
.venv/bin/python src/cross_validate.py --healthy-hop 1.0
```

**Outputs**

- `models/binary/best_audio_binary_model.pth`, `models/binary/best_late_binary_model.pth`
  — final models trained on *all* data (each stored with its window/hop config).
- `results/binary/cv_report.md` — readable Markdown report.
- `results/binary/confusion_{audio,late}.png` — pooled confusion heatmaps.

The Phase 1/2 five-class scripts (`train_early.py`, `train_late.py`,
`evaluate.py`) are retained for the historical record and still run, but the
binary pipeline is the current system.

---

## Known limitations & next steps

1. **One healthy recording.** This is the binding constraint. It makes
   specificity weakly estimated and caps how much "healthy" can be learned.
   *Priority: collect multiple independent healthy recordings.*
2. **Threshold calibration.** AUC ≈ 0.82 shows the separation exists; the default
   0.5 threshold is poorly placed for specificity. A threshold chosen on the
   training folds only (leak-safe) should rebalance sensitivity/specificity.
3. **More independent recordings per condition** (different wings, mounts, days)
   so that held-out-recording evaluation is statistically meaningful and the
   five-class problem could eventually be revisited honestly.
