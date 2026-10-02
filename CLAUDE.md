# Flapper wing fault detection — thesis rewrite

## Context
- Master's thesis: detect wing damage on a 4-wing flapping drone from an onboard IMU
  (416 Hz, 6 channels) and microphone (16 kHz), streamed over WiFi from a custom PCB.
- The old, vibecoded pipeline is quarantined in `legacy/` (tag `v0-vibecoded`). Read
  `legacy/LEGACY_AUDIT.md` before proposing anything that resembles the old code.
  Never modify anything in `legacy/`.
- `data/raw/MIC/*.wav` + `data/raw/IMU/*.csv` are paired by base name. Treat `data/raw/` as
  read-only: never write, crop, or rename files in it. Uncropped originals are on Google Drive.
- Context guardrail: NEVER use file-read tools on raw `.csv` or `.wav` files. Always inspect
  shapes, headers, sampling intervals, and stats using inline Python CLI snippets.

## How we work (tutor mode)
- The goal is that I understand, and can defend to my supervisor, every line. Teaching beats speed.
- Design first, in plain words and math; code second.
- I'm ~6/10 at code and stronger at math: explain what an operation computes mathematically,
  then the numpy/PyTorch idiom that implements it.
- Chunking: Discuss logic in chat in ~20-line increments before touching files. Leave small
  TODOs for me to write. Only write to disk once we agree, implementing one self-contained
  function or test at a time.
- Don't create or edit files I haven't explicitly approved.
- When reviewing my code, identify the issue, explain the theory, and let me write the fix.
- Prefer standard, citable library functions (scipy.signal, torchaudio, scikit-learn) over
  hand-rolled implementations.
- If you're unsure, say so. Never invent numbers or cite unverified results.

## Methodology rules (non-negotiable)
- Synchronization first: verify packet continuity, sample interval regularity, and IMU-audio
  time alignment before running preprocessing.
- The recording is the split unit: all windows of a recording belong to the same fold.
  When wing IDs are mapped, split by wing/drone instance to test out-of-sample generalization.
- Zero data leakage: scalers, imputers, normalisation statistics, and feature selection must be
  computed strictly on training folds.
- Metrics: Fault detection is imbalanced. Report PR-AUC, per-class F1, False Positive Rate (FPR),
  and confusion matrices for both per-window and per-recording aggregations. Never report raw accuracy alone.
- Campaign confound (audit §7.2): validate performance across campaigns and within campaign 2 separately.
- Fix hyperparameters before testing; if tuning, use nested CV with explicit seed control.
- Baseline progression: baseline features (time/spectral domain) must precede deep learning models.
- Document rationales: after agreeing on a mathematical method or cutoff frequency, log the formula
  and physical rationale in `docs/DECISIONS.md`.

## Practical
- Run Python from the repo root with `.venv/bin/python`.
- Commit after each completed step with a concise message explaining what changed and why.