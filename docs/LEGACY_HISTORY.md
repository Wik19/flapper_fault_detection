# What was tried before the rewrite (May–July 2026)

A chronological account of every attempt in the legacy project: what question it asked, what
was built, what it scored, and why it was abandoned. It complements `legacy/LEGACY_AUDIT.md`,
which describes the *final* legacy code by topic (data, features, architectures,
hyperparameters, flaws).

Written 2026-10-04. Sources: git history, file timestamps of saved models and results, the
saved confusion-matrix images and CV reports in `legacy/results/`, and `legacy/README.md`.

- ✔ = verified in this review from a saved file (image, report, checkpoint, git commit)
- ✱ = claimed in the legacy README only; no saved evidence exists

---

## At a glance

| # | When | Question | Setup in one line | Result | Verdict |
|---|---|---|---|---|---|
| 0 | 05-25 → 05-26 | — | Campaign 1 recorded (10 recordings); 5 files cropped by hand | — | Data |
| 1 | 05-26 → 05-28 | Which of 5 damage states? | Big 5-class CNNs; windows shuffled *then* split | 93 % / 79 % accuracy ✔ | **Invalid** (leakage) |
| 2 | ≈ 06-24 → 06-25 | Same, measured honestly | Same CNNs; whole recordings held out | 25 % / 18 % accuracy ✔ (chance 20 %) | Honest, unusable |
| 3 | 06-30 | Healthy or damaged? | Small binary CNNs; leave-one-damaged-out; **one** healthy recording | Late-fusion AUC 0.81 ✱ / 0.78 ✔ | Signal found; specificity unmeasurable |
| 4 | 06-30 → 07-02 | Same, with more data | + campaign 2 (14 recordings); stratified group 5-fold | Late-fusion AUC 0.93 ✔ | Best so far, but optimistic |

---

## Attempt 0: data collection (2026-05-25 → 05-26)

- **Campaign 1** recorded on 2026-05-25: `Healthy1`, four `hole1` flights, two `hole2`, two
  `tear_n_hole2`, `fixed_all_tape` (10 recordings, one fresh wing set damaged progressively).
- The instructions given to Cursor that day are in `legacy/2026-05-25_Data_Processing_Plan.md`.
  They asked for timestamp-based synchronisation (`Rx_Time_Sec`); that step was never implemented.
- 2026-05-26 20:01–20:09: five recordings cropped by hand with
  `legacy/src/data_tools/edit_data.py` (crop times recovered on 2026-10-03 and recorded in
  `data/manifest.csv`).

## Attempt 1: five classes, leaky split (2026-05-26 → 05-28)

**Question.** Classify every 3 s window into one of 5 classes: healthy, hole1, hole2,
tear_n_hole2, taped.

**What was built** (first commit `12264e5`, 2026-05-28):
- *Early fusion:* audio log-mel spectrogram [1, 64, 188] plus IMU STFT spectrograms, resized to
  the same 64 × 188 grid and stacked as 7 image channels → `EarlyFusionCNN`, **6.16 M params** ✔.
- *Late fusion:* audio 2D CNN + raw-IMU 1D CNN, concatenated → `LateFusionNet`, **14.81 M params** ✔.
- 3 s windows with 1 s hop. AdamW (lr 1e-4, weight decay 1e-4), batch 32, unweighted
  cross-entropy. The checkpoint was chosen by lowest validation loss. The first README documents
  `--split 0.7 --epochs 50` as the usage.
- **Split:** each recording's windows were shuffled (`random.seed(42)`, commented
  `# Stop data leakage!`), then divided train/validation by a ratio.

**Results** ✔ (early fusion; the Attempt 1 late-fusion figure was overwritten in Attempt 2):

| Saved figure | Date | Validation windows | Accuracy | Settings |
|---|---|---|---|---|
| `legacy/results/early_fusion/confusion_matrix_0.png` | 05-26 | 75 | 70/75 = **93.3 %** | could not be reconstructed |
| `legacy/results/early_fusion/confusion_matrix_extended.png` | 05-27 | 115 | 91/115 = **79.1 %** | split 0.7, 3 s / 1 s (per-class counts match exactly) |

The README's "~95 %" corresponds at best to the first run.

**Why it was invalid.** Consecutive windows overlap by 67 %, so near-copies of the same seconds
sat in both train and validation. The model could recognise *which recording* a window came from
(mic placement, battery, room noise) instead of learning the fault. Also, the validation set
chose the checkpoint and was then reported as the result.

## Attempt 2: five classes, honest split (≈ 2026-06-24 → 06-25)

Dated by the checkpoint and result files (late fusion 06-24, early fusion 06-25). This code was
committed only on 06-30, together with Attempt 3 (`4290c3c`).

**What changed:**
- `legacy/src/data_tools/splitting.py`: classes with ≥ 2 recordings hold out whole recordings;
  single-recording classes (healthy, taped) are split in time with a 3-window guard gap.
- Inverse-frequency class weights in the loss.
- Per-window, per-channel IMU standardisation. (The legacy README also lists audio loudness
  normalisation; it is **not** in this code ✔.)

**Validation set** ✔ (reconstructed from the per-class counts: split 0.7):
151 windows = `hole1_loss_of_control_cropped` (16) + `hole2_full2` (56) +
`tear_n_hole2_full_cropped` (53) + the last 13 windows of `Healthy1` and of `fixed_all_tape`.
Each damaged class was judged on a **single** held-out recording.

**Results** ✔ (chance level for 5 classes = 20 %):

| Model | Accuracy | Balanced accuracy |
|---|:--:|:--:|
| Late fusion | 37/151 = **24.5 %** | 0.33 |
| Early fusion | 27/151 = **17.9 %** | 0.15 |

The legacy README's "~25 % for both" holds only for late fusion; early fusion was below chance.
The errors were mostly confusions between damage types (e.g. late fusion called 35 of 56 `hole2`
windows `hole1`).

**Lesson.** With 1–4 recordings per class there was nothing to generalise across. The 93 % of
Attempt 1 had been hiding a dataset far too small for a 5-way problem.

## Attempt 3: binary, one healthy recording (2026-06-30)

Commits `4290c3c` (11:07) and `be33b1f` (11:47).

**What changed:**
- **Question reframed:** healthy vs damaged; the taped wing excluded.
- **Small models** with global average pooling: audio-only **23 650** and late-fusion
  **45 522** params ✔ (vs millions before).
- **Protocol:** leave-one-damaged-recording-out, 8 folds. The single healthy recording was split
  once in time (first 70 % train, guard gap, tail added to every test fold).
- 30 epochs, AdamW (lr 1e-3, weight decay 1e-3), batch 32, `WeightedRandomSampler` to balance
  the ~6:1 class ratio, augmentation (audio gain/noise/shift + SpecAugment; IMU scale/noise/shift).
- **Healthy densification:** healthy training windows taken every 0.25 s instead of 1 s (default).
- `be33b1f` added a `--no-augment` switch.

**Results:**

| Model | Augmentation | Sens | Spec | ROC-AUC | Source |
|---|---|:--:|:--:|:--:|---|
| Audio only | on | 0.55 | 0.44 | 0.58 | ✱ README |
| Late fusion | on | 0.98 | 0.24 | 0.81 | ✱ README |
| Audio only | off | 0.67 | 0.43 | 0.59 | ✔ `legacy/results/binary/cv_report_no_augment.md` |
| Late fusion | off | 0.91 | 0.24 | 0.78 | ✔ same file |

Thresholds were the default 0.5 at this stage.

**Side experiments:**
- **Healthy densification** gave no specificity gain ✱; it was removed in Attempt 4.
- **Augmentation off** scored late AUC 0.78 ✔ vs 0.81 ✱ on. This comparison mixes a saved
  number with an unsaved one.

**Lesson.** Adding the IMU lifted AUC far above audio alone, so the motion signal looked
informative. Specificity was meaningless: the only healthy test data was a later segment of the
same flight the model trained on.

## Attempt 4: binary, 23 recordings (2026-06-30 → 07-02)

The code was **never committed** at the time; it was snapshotted on 2026-10-02 (`5779519`, tag
`v0-vibecoded`).

**What changed:**
- **Campaign 2 data:** 10 healthy (`H1S1…H1S10`) + 4 damaged (`Hole1S1`, `Hole1S2`, `Hole2S1`,
  `Hole2S2W12`) on a fresh wing set. Five healthy files were cropped on 07-02.
- **Protocol:** hand-rolled stratified group 5-fold CV over recordings (seed 0), so every test
  fold holds unseen healthy *and* damaged recordings.
- **Threshold calibration:** Youden's J on each fold's own training predictions. The deployable
  model was retrained on all 23 recordings, with a threshold from pooled out-of-fold predictions.
- IMU header row fixed; healthy densification removed.

**Results** ✔ (`legacy/results/binary/cv_report.md`, 07-02; mean ± std over 5 folds, calibrated
threshold):

| Model | Sens | Spec | Bal-acc | ROC-AUC |
|---|:--:|:--:|:--:|:--:|
| Audio only | 0.67 ± 0.24 | 0.77 ± 0.18 | 0.72 ± 0.12 | 0.76 ± 0.13 |
| Late fusion | 0.91 ± 0.07 | 0.87 ± 0.19 | 0.89 ± 0.12 | 0.93 ± 0.13 |

Late-fusion AUC per fold: 1.00, 0.99, **0.67**, 0.97, 1.00.

**Claimed but not backed by saved files** ✱: seed stability (AUC 0.93 / 0.94 / 0.91 for fold
seeds 0 / 1 / 2) and the "Phase-4 verification audit".

**What is still wrong with it** (details in `legacy/LEGACY_AUDIT.md` §7.2 and
`docs/DECISIONS.md` D3):
- **Campaign confound:** campaign 1 is 1 healthy / 8 damaged, campaign 2 is 10 / 4. The weak fold
  3 is the only fold that tests `Healthy1`, the one campaign-1 healthy flight.
- **Campaign = wing set, and damage follows recording order:** every healthy flight was recorded
  before every damaged flight of the same set.
- **Every window counted as a sample:** 989 overlapping windows from only 23 recordings.
- **Tuned on the test results:** settings were iterated across attempts while watching CV
  results, and no untouched test set exists.
- **Threshold fitted in-sample:** on the model's own overconfident training predictions.

---

## Ideas tried and dropped

| Idea | Attempt | What happened | Evidence |
|---|---|---|---|
| Five-class problem | 1–2 | Leaky 93 % → honest 18–25 % | ✔ confusion matrices |
| Early fusion (spectrograms resized and stacked as channels) | 1–2 | Below chance once honest; audio (0–8 kHz mel) and IMU (0–208 Hz) axes don't line up | ✔ |
| Multi-million-parameter CNNs | 1–2 | Replaced by ~24–46 k-param models | ✔ |
| Inverse-frequency class weights | 2 | Replaced by a balanced sampler | ✔ code |
| Healthy densification (0.25 s hop) | 3 | No specificity gain; removed | ✱ |
| Augmentation on vs off | 3 | Small AUC difference; no Attempt 4 comparison saved | ✔ / ✱ |
| Default 0.5 threshold | 3 | Replaced by Youden's J on training predictions | ✔ code |
| Audio only vs audio + IMU | 3–4 | Fusion better every time (AUC 0.78–0.93 vs 0.58–0.76) | ✔ / ✱ |
| Single-file inference (`legacy/src/test_on_new.py`) | 1 | Broken (imports a module that no longer exists) | ✔ |

## What carries over to the rewrite

- **Split by recording, always.** Attempt 1 is the cautionary tale; Attempt 2 shows what honest
  numbers look like.
- **Binary first.** Five classes are not supportable with this data.
- **The IMU looked informative**, but this is a hypothesis to re-test under a cleaner protocol,
  not a result to assume.
- **Save every result with its exact config and seed.** The settings of several legacy numbers
  could not be recovered.
- **Report per recording**, and check results within campaign 2 alone.
