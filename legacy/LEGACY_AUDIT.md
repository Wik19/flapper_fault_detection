# Legacy pipeline audit

Read-only inspection of everything in `legacy/` (tag `v0-vibecoded`, commit `5779519`),
done 2026-10-02 before the rewrite. No legacy code was modified. Numbers marked
*(measured)* were computed during this audit by instantiating the legacy code or
reading the raw data, not copied from the README.

---

## 0. Inventory

| File | Role | Status |
|---|---|---|
| `src/data_tools/binary_data.py` | Binary labelling, windowing, per-window features, augmentation | **Current** (Phase 3–4) |
| `src/architectures/binary_net.py` | Lean audio-only and late-fusion binary CNNs | **Current** |
| `src/cross_validate.py` | Stratified group k-fold CV, threshold calibration, final model, reports | **Current** |
| `src/data_tools/splitting.py` | 5-class labeller + group/temporal split | Historical (Phase 2) |
| `src/data_tools/preprocess_early.py` | 5-class early-fusion dataset (spectrogram stacking) | Historical (Phase 1–2) |
| `src/data_tools/multimodal_dataset.py` | 5-class late-fusion dataset (raw audio + raw IMU) | Historical (Phase 1–2) |
| `src/architectures/early_fusion.py`, `late_fusion.py` | 5-class CNNs | Historical |
| `src/train_early.py`, `train_late.py`, `evaluate.py`, `run_pipeline.sh` | 5-class train/eval | Historical, **broken on current data** (see §7.3) |
| `src/test_on_new.py` | Single-file inference for early fusion | **Broken** (imports a module that no longer exists) |
| `src/data_tools/verify_data.py` | Plots `Data/processed/*.npy` | **Dead** (nothing produces those files) |
| `src/data_tools/edit_data.py` | Manual crop of a wav/csv pair | Data-prep utility (destructive, see §1.2) |
| `src/data_tools/view_data.py` | Plot audio + IMU of one recording | Data-prep utility |
| `2026-05-25_Data_Processing_Plan.md` | Original instructions given to Cursor | Provenance |
| `models/`, `results/` | Saved checkpoints, confusion PNGs, CV reports | Outputs (git-ignored) |

The Phase 1 (leaky) code no longer exists in the working tree; it survives only in
git commit `12264e5` (§6).

---

## 1. Data

### 1.1 Corpus and labels

- 24 paired recordings: `data/raw/MIC/<name>.wav` + `data/raw/IMU/<name>.csv`, paired by
  identical base name. All 24 pairs exist *(measured)*.
- **Audio:** mono, 16 kHz, 16-bit PCM, 6–59 s per recording.
- **IMU:** CSV with header `Packet_ID, Rx_Time_Sec, Acc_X, Acc_Y, Acc_Z, Gyro_X, Gyro_Y, Gyro_Z`,
  nominal 416 Hz; the code uses only the last 6 columns.
- **Labels come from filenames only** (`binary_data.py:33`, case-insensitive):
  `tape` → excluded; `hole|tear|crack|broken|damage` → Damaged; `healthy` or regex
  `^h\d+s\d+` → Healthy; anything else raises `ValueError`.
- Used corpus: **11 healthy + 12 damaged = 23 recordings, 989 windows** (496 H / 493 D);
  `fixed_all_tape` excluded *(measured)*.
- No metadata beyond the filename: **which physical wing(s) are damaged, and whether the
  same damaged wing is reused across recordings, is not recorded anywhere.**

Two collection campaigns, distinguishable by naming convention *(measured window counts)*:

| Campaign | Healthy recordings (windows) | Damaged recordings (windows) |
|---|---|---|
| 1st (snake_case names) | 1 — `Healthy1` (53) | 8 — `hole1_*`, `hole2_*`, `tear_n_hole2_*` (270) |
| 2nd (`H<n>S<n>`, `Hole<n>S<n>`) | 10 — `H1S1…H1S10` (443) | 4 — `Hole1S1`, `Hole1S2`, `Hole2S1`, `Hole2S2W12` (223) |

### 1.2 Manual preprocessing before the pipeline (`edit_data.py`)

- Crops a wav/csv pair to `[start_sec, end_sec]`: audio by sample index × 16 000, IMU by
  **row index × 416** (assumes uniform 416 Hz), then **overwrites `Rx_Time_Sec`** with
  `linspace(0, n/416, n)` (`edit_data.py:43`).
- Writes the output next to the inputs in `data/raw/`; the originals are not kept there
  (they're on Google Drive). **Crop boundaries are not logged**; only the last
  invocation survives in the script: `H2S10` → `H1S10`, 2–40 s (`edit_data.py:52-60`),
  which also **renames H2 → H1**.
- 10 files carry the rewritten-timestamp signature (exactly uniform 416 Hz), so they went
  through this script *(measured)*: `H1S1, H1S5, H1S8, H1S9, H1S10, hole1_failure_cropped,
  hole1_loss_of_control_cropped, hole1_loss_of_control2_cropped, tear_n_hole2_cropped,
  tear_n_hole2_full_cropped`.

### 1.3 Timing and synchronisation

- **No synchronisation step exists.** The original plan said to align streams with
  `Rx_Time_Sec`; no code ever reads that column. Audio and IMU windows are aligned purely
  by "both files start at t = 0", with sample index × rate.
- IMU uniformity assumption is **supported by the data** *(measured)*: every file has 20
  samples per packet and **zero skipped `Packet_ID`s**; the 0.1–3 s jumps in
  `Rx_Time_Sec` in uncropped files are WiFi receive-time bursts, not lost samples.
- Audio and IMU lengths differ by up to 1.4 s (`H1S1`: 53.0 s audio vs 51.6 s IMU).
  The window count is computed from the **audio** length only, so 3 windows run past the
  end of the IMU and get zero-padded (`H1S1` windows 49–50, `H1S8` window 50; all
  healthy) *(measured)*.

---

## 2. Processing and features: binary pipeline (Phase 3–4, current)

### 2.1 Windowing

- Window **3.0 s**, hop **1.0 s** (67 % overlap) → audio 48 000 samples, IMU 1 248 rows
  per window; audio hop 16 000, IMU hop 416.
- Windows per recording = `floor((N_audio − 48000) / 16000) + 1` (`binary_data.py:56`);
  recordings shorter than 3 s are dropped. Range: 4 to 57 windows per recording.
- Each window is read **from disk on every access** (`wave` seek for audio,
  `pd.read_csv(skiprows=…, nrows=1248)` for IMU), every epoch.

### 2.2 Audio, per window (`binary_data.py:190-202`)

1. int16 → float32 / 32768; multi-channel averaged to mono; zero-padded to 48 000 if short.
2. **RMS normalisation:** `w / (sqrt(mean(w²)) + 1e-8)` (removes loudness).
3. *(train only, if augmenting)* gain × U(0.7, 1.3) → + N(0, 0.01²) noise (≈40 dB SNR)
   → `torch.roll` by U{−2400…+2400} samples (±0.15 s, **wraps around**).
4. **Log-mel spectrogram:** `torchaudio MelSpectrogram(sr=16000, n_fft=1024, hop_length=256,
   n_mels=64)` with defaults (Hann window 1024 = 64 ms, hop 16 ms, f 0–8 kHz, power 2,
   `center=True`, HTK mel, no norm) → `AmplitudeToDB()` (10·log10, no `top_db`).
   Output **[1, 64, 188]**.
5. *(train only)* SpecAugment: one `FrequencyMasking(10)` + one `TimeMasking(25)`.

### 2.3 IMU, per window (`binary_data.py:123-136`, `183-188`)

1. Rows `[416·k + 1, 416·k + 1249)` (the +1 skips the header), last 6 columns,
   `to_numeric(coerce)`, NaN → 0.
2. **Per-window, per-channel z-score:** `(x − mean) / (std + 1e-6)` over the 1 248 samples
   (removes gravity/DC **and absolute vibration amplitude**). Zero-pad to 1 248 if short.
3. *(train only)* per-channel scale × U(0.9, 1.1) → + N(0, 0.05²) → `torch.roll` by
   U{−62…+62} samples (±0.15 s, wraps around).
4. Output **[6, 1248]**, fed raw (time domain) to a 1D CNN. No spectral features.

### 2.4 Sampling

- `WeightedRandomSampler`, weight = 1 / (class count), `num_samples = len(train set)`,
  `replacement=True` (`cross_validate.py:92-96`). With the corpus now ~balanced this is
  ≈ bootstrap resampling. The comment ("damaged outnumbers healthy ~6:1") is stale.

---

## 3. Processing and features: 5-class pipelines (Phase 1–2, historical)

Common: same 3 s / 1 s windowing, 5 classes from `splitting.py:18`
(`Healthy`, `hole1`, `hole2`, `tear_n_hole2`, `fixed_all_tape`; case-sensitive).
**No audio RMS normalisation** in either 5-class dataset. IMU per-window per-channel
z-score in both. `CSV_HAS_HEADER = False` although the CSVs have a header (§7.3).

**Early fusion** (`preprocess_early.py`):
- Audio → log-mel exactly as §2.2 step 4 → [1, 64, 188].
- IMU (z-scored) → `torch.stft(n_fft=64, hop=16, Hann 64, normalized=True)` → |·| →
  `AmplitudeToDB` → [6, 33, 79] (0–208 Hz, 6.5 Hz bins).
- IMU spectrograms **bilinearly resized to [64, 188]** and concatenated with the mel
  as channels → **[7, 64, 188]**.
- *(train only)* `FrequencyMasking(8)` + `TimeMasking(20)` on the fused 7-channel tensor
  (same rows/columns masked across audio and IMU).

**Late fusion** (`multimodal_dataset.py` + `late_fusion.py`):
- Dataset returns raw audio [1, 48000] and z-scored IMU [6, 1248]; the mel transform
  (same parameters) runs **inside the model**. **No augmentation.**

---

## 4. Architectures (exact)

Parameter counts *(measured)* by instantiating the legacy classes.

### 4.1 Binary (current), `binary_net.py`

Conv blocks: 2D = `Conv2d(3×3, pad 1) → BatchNorm2d → ReLU → MaxPool2d(2)`;
1D = `Conv1d(k=5, pad 2) → BatchNorm1d → ReLU → MaxPool1d(p)`.

| Module | Layers | In → out | Params |
|---|---|---|---|
| `AudioCNN` | 2D blocks 1→16→32→64, then `AdaptiveAvgPool2d(1)` | [1,64,188] → [64,8,23] → **64** | 23 520 |
| `ImuCNN` | 1D blocks 6→16→32→64, pools 4, 4, 2, then `AdaptiveAvgPool1d(1)` | [6,1248] → [64,39] → **64** | 13 616 |
| `BinaryAudioNet` | `AudioCNN` → `Dropout(0.5)` → `Linear(64, 2)` | | **23 650** |
| `BinaryLateFusionNet` | concat(`AudioCNN`, `ImuCNN`) = 128 → `Dropout(0.5)` → `Linear(128, 64)` → ReLU → `Dropout(0.5)` → `Linear(64, 2)` | | **45 522** |

- Receptive field before global pooling: IMU ≈ 116 samples ≈ **0.28 s**; audio ≈ 22 mel
  bins × 22 frames (≈ 0.35 s).
- `BinaryAudioNet.forward` ignores the IMU, but the dataset still loads it.
- Output: 2 logits; P(damaged) = `softmax(logits)[:, 1]`.

### 4.2 5-class (historical)

**`EarlyFusionCNN`** (`early_fusion.py`), **6 160 773 params**:
`Conv2d` 7→32→64→128→256 (each 3×3 pad 1 → BN → ReLU → MaxPool 2) → flatten
256×4×11 = 11 264 → `Linear(11264, 512)` → ReLU → `Dropout(0.3)` → `Linear(512, 5)`.

**`LateFusionNet`** (`late_fusion.py`), **14 807 845 params**:
- Audio: mel inside model → `Conv2d` 1→32→64→128 (3×3, BN, ReLU, MaxPool 2) → flatten
  128×8×23 = 23 552.
- IMU: `Conv1d` 6→32 (k5, pool 4) → 64 (k3, pool 4) → 128 (k3, pool 2) → flatten
  128×39 = 4 992.
- Head: concat 28 544 → `Linear(28544, 512)` → BN1d → ReLU → `Dropout(0.4)` →
  `Linear(512, 128)` → ReLU → `Dropout(0.2)` → `Linear(128, 5)`.

---

## 5. Hyperparameters (exact)

### 5.1 Binary CV (`cross_validate.py` CLI defaults, used for the saved Phase 4 report)

| Setting | Value |
|---|---|
| Window / hop | 3.0 s / 1.0 s |
| Folds / fold seed | K = 5 / `numpy.default_rng(0)` |
| Epochs | 30 (fixed, no early stopping, no validation split) |
| Batch size | 32 |
| Optimiser | `AdamW(lr=1e-3, weight_decay=1e-3)`, default betas, **no LR schedule** |
| Loss | `CrossEntropyLoss()` (unweighted; balancing via the sampler) |
| Dropout | 0.5 (head) |
| Augmentation | on (disable with `--no-augment`) |
| Model init seed | `torch.manual_seed(0)` before each fold's model |
| DataLoader workers | 4 |
| Threshold | Youden's J on the fold's own **training** predictions; median of the tied plateau; fallback 0.5 |
| Deploy thresholds (saved in checkpoints) | audio **0.942**, late **0.666** *(read from `.pth`)* |
| Misc | `torch.backends.cudnn.enabled = False` (environment workaround) |

### 5.2 5-class (`train_early.py`, `train_late.py`, `evaluate.py`)

| Setting | Script default | `run_pipeline.sh` |
|---|---|---|
| Window / hop | 3.0 / 1.0 s | 3.0 / 1.0 s |
| Train split ratio | 0.6 | **0.7** |
| Epochs | 100 | **50** |
| Batch / LR | 32 / 1e-4 | 32 / 1e-4 |
| Optimiser | `AdamW(wd=1e-4)` | |
| Loss | Phase 1: `CrossEntropyLoss()`; Phase 2: inverse-frequency class weights `w_c = N / (C·n_c)` | |
| Checkpoint | epoch with lowest **validation loss** | |
| Workers | 24 | |

Which of the two columns produced the reported Phase 1/2 numbers is **not recorded**. The
5-class checkpoints are bare `state_dict`s with no config, and the provenance of
`best_flapper_model_0.pth` / `_extended.pth` is unknown.

---

## 6. Evaluation protocols by phase

| Phase | Split | Unit held out | Reported |
|---|---|---|---|
| 1 | Commit `12264e5`: per recording, `random.seed(42); random.shuffle(chunks)` then first 60 % train / rest val (commented `# Stop data leakage!`) | **Windows** (overlapping) | ~95 % acc (README only) |
| 2 | `splitting.py`: classes with ≥ 2 recordings hold out the alphabetically **last** recordings; single-recording classes (Healthy, tape) get a temporal split with a guard gap | Recordings (or time segments) | ~25 % acc (README only) |
| 3 | Leave-one-damaged-recording-out (8 folds); the single healthy recording is split once in time, its tail added to every test fold | Damaged recordings | `results/binary/cv_report_no_augment.md` + README |
| 4 | Hand-rolled stratified group 5-fold (round-robin per class after shuffle) | **Recordings** | `results/binary/cv_report.md` |

Phase 4 metrics: window-level sensitivity, specificity, balanced accuracy at the
calibrated threshold, window-level ROC-AUC per fold; mean ± population std across folds;
pooled confusion matrix; pooled metrics at 0.5 vs calibrated. The final model is
retrained on all 23 recordings with the same hyperparameters, and its threshold is
picked by Youden's J on the **pooled out-of-fold** probabilities.

---

## 7. Methodological flaws and leakage

### 7.1 Confirmed leakage (historical)

1. **Phase 1, window-level split.** Windows of the same recording were shuffled before
   splitting, so near-identical 67 %-overlapping windows sit on both sides; the model
   could identify the recording instead of the fault. The ~95 % is invalid.
2. **Phases 1–2, validation = test.** The checkpoint was chosen by lowest validation loss,
   and the confusion matrix was then reported on that same validation set (optimistic).
3. **Phase 2, single-recording classes.** Healthy and tape were validated on a later time
   segment of their *only* recording. The guard gap prevents window overlap but not
   recording-fingerprint leakage.

### 7.2 Binary pipeline (Phase 4): no direct train/test window leakage found

Folds are recording-level, normalisation is per window only (no dataset statistics),
augmentation and the sampler run only on the training path, and the threshold is fitted
on training predictions. The issues below are **bias, confounds and statistical
weakness**, not window leakage:

1. **Campaign ↔ class confound (new finding).** Campaign 1 is 1 H / 8 D; campaign 2 is
   10 H / 4 D (§1.1). Anything that differs between campaigns (setup, firmware, battery,
   mic placement, day) is correlated with the label. Evidence that it matters: the weak
   fold, **fold 3 (late AUC 0.67, specificity 0.49), is the only fold whose test set
   contains `Healthy1`**, the one campaign-1 healthy recording *(measured fold
   composition, seed 0)*. Evidence it isn't the whole story: fold 1 tests two
   campaign-2 damaged recordings and still reaches AUC 1.00. Not proven either way; the
   legacy code saves no per-recording predictions.
2. **Pseudo-replication.** All metrics are per window, but the independent unit is the
   recording (n = 23). Windows overlap by 67 % and recordings contribute 4–57 windows
   each, so long recordings dominate pooled numbers; per-fold AUC rests on 4–6
   recordings; "± std" is over only 5 folds.
3. **Model selection on the evaluation data.** Window, hop, epochs, LR, augmentation,
   architecture and the thresholding scheme were iterated across phases while looking at
   CV results, and no untouched test set exists. The reported numbers are therefore
   optimistic by an unknown amount.
4. **In-sample threshold calibration.** The threshold is fitted on the model's own
   (overfit, saturated) training predictions (audio τ = 0.87–1.00), and the plateau-median
   rule patches over that. The deploy threshold comes from pooled OOF probabilities of 5
   *different* models and is applied to a 6th model trained on all data. Its performance
   is never evaluated.
5. **Unseen recording ≠ unseen wing.** Wing identity is unrecorded, so the same physical
   damaged wing can sit in train and test.
6. **Label/scope confound.** Damaged includes `hole1_loss_of_control*` and
   `hole1_failure` crops; the model may partly detect unstable flight rather than wing
   damage.
7. **Hand-chosen crops.** Crop boundaries were chosen by eye and not logged, which is a
   possible selection effect, and they cannot be reproduced.
8. **Normalisation removes amplitude.** Per-window RMS (audio) and z-score (IMU) remove
   absolute vibration/sound level, a physically plausible damage cue (and also a
   plausible shortcut). This design choice was never examined.
9. **Zero-padded IMU windows.** 3 windows, all healthy (§1.3), form a tiny artefact
   correlated with the class.
10. **Augmentation artefact.** `torch.roll` wraps the window end onto its start, creating
    a discontinuity (training only).
11. **No synchronisation.** Audio/IMU alignment is assumed from file start; lengths
    differ by up to 1.4 s.

### 7.3 Implementation defects and README-vs-code discrepancies

- **README says the 5-class nets have "~1M params".** They have **6.16 M** (early) and
  **14.81 M** (late) *(measured)*.
- **README says the 5-class scripts "still run".** `get_class_label` raises `ValueError`
  on 14 of the 24 current files (`H1S*`, `Hole*S*`) *(measured)*, so they crash.
- **README Phase 2 lists per-window audio RMS normalisation.** It is absent from both
  5-class datasets; only `binary_data.py` has it.
- **Header handling.** 5-class scripts use `CSV_HAS_HEADER = False` on CSVs that have a
  header, so the header row becomes a row of zeros in each recording's first window and
  all IMU rows are offset by one.
- **`test_on_new.py`** imports `data_tools.preprocess`, which doesn't exist, and labels
  output "Time {i}s" assuming hop = 1 s.
- **`verify_data.py`** reads `Data/processed/*.npy` (capital D, absolute path), which no
  script produces.
- **Hyperparameters not stored.** The `run_pipeline.sh` and script defaults disagree
  (§5.2), and 5-class checkpoints don't store their config.
- **`cv_report_no_augment.md` is a Phase 3 report** (1 healthy, leave-one-damaged-out,
  healthy hop 0.25 s), not a no-augmentation version of the Phase 4 report; the two are
  not comparable. No Phase 4 no-augmentation result was saved.
- **Slow data loading.** IMU CSVs are re-parsed with pandas for every window, every epoch
  (performance only).

---

## 8. Results as saved

**Phase 4** (`results/binary/cv_report.md`, stratified group 5-fold, augmentation on,
calibrated threshold, mean ± std across folds):

| Model | Sens | Spec | Bal-acc | ROC-AUC |
|---|:--:|:--:|:--:|:--:|
| audio | 0.67 ± 0.24 | 0.77 ± 0.18 | 0.72 ± 0.12 | 0.76 ± 0.13 |
| late | 0.91 ± 0.07 | 0.87 ± 0.19 | 0.89 ± 0.12 | 0.93 ± 0.13 |

Late fusion per-fold AUC: 1.00, 0.99, **0.67**, 0.97, 1.00. Audio at the 0.5 threshold:
specificity 0.04 (probabilities saturate toward "damaged").

**Phase 3** (`cv_report_no_augment.md`, augmentation off): audio AUC 0.59, late AUC 0.78,
late specificity 0.24. The README reports augmentation-on Phase 3 numbers (late AUC 0.81),
but no saved report backs them.

**Phases 1–2:** only confusion-matrix PNGs remain; the accuracy figures exist in the
README only, with unknown exact settings.

---

## 9. Open questions this raises for the rewrite

1. Per recording: which of the 4 wings is damaged, and is the same physical wing reused
   across recordings? (Needed for a wing-aware split and to state the generalisation claim.)
2. Does performance survive the campaign confound? Report per-recording scores, and
   evaluate within campaign 2 alone (10 H / 4 D, recorded in the same sessions).
3. Should damaged-but-unstable flights (`loss_of_control`, `failure`) stay in the
   "damaged" class?
4. Should absolute amplitude be kept as a feature? Make it a deliberate, tested choice.
5. Crop boundaries: recover them from the Google Drive originals and record them in a
   manifest, so the dataset can be rebuilt from untouched raw files.
