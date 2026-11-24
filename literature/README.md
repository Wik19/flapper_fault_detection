# Literature: reading notes

PDFs are in `papers/` (not committed: arXiv licences don't generally allow redistribution);
BibTeX in `references.bib`.
Each note: what the paper does, the numbers worth remembering (with where to find them), and what
it means for this project. Notes written 2026-11-24 from the arXiv versions.

---

## Boniol et al. (2024), *Dive into Time-Series Anomaly Detection: A Decade Review*
`boniol2024tsad` · arXiv 2412.20512 · interactive version: https://boniolp.github.io/TSADtaxonomy/

**What it is.** A survey that sorts time-series anomaly detection (TSAD) methods by *how* they
detect, not by which community produced them.

**Key ideas.**
- **Anomaly types (§2.2, Fig. 2):** *point* (one value outside the normal range), *contextual*
  (normal value, abnormal in its local context) and *collective* (a sub-sequence whose shape is
  abnormal). A further case is **whole-sequence detection**: the entire series is judged at once.
- **Supervision (§2.4):** *unsupervised* (nothing known), *semi-supervised* (only normal examples
  are labelled; the authors stress this is often mislabelled "unsupervised"), *supervised*
  (labelled anomalies available).
- **Common pipeline (§2.5, Fig. 4):** pre-processing (always a sliding window) → detection model →
  anomaly score → post-processing (threshold).
- **Taxonomy (§3, Fig. 5):** *distance-based* (proximity, clustering, discord), *density-based*
  (distribution, graph, tree, encoding) and *prediction-based* (forecasting, reconstruction: train
  on normal data and flag what the model cannot predict or reconstruct).

**What it means for us.**
- Our problem in this vocabulary: **multivariate** (6 IMU channels, plus audio),
  **whole-sequence** (a flight is healthy or damaged throughout; no onset to find), and currently
  **supervised** (we have damaged examples).
- Our pipeline already has their four stages: 3-s windows → classifier → P(damaged) →
  aggregation over the flight and a threshold.
- **The semi-supervised framing is worth discussing:** train only on healthy flights and flag
  deviations. That is how a deployed monitor would work (you can't collect every damage type in
  advance). Caveat for our data: 10 of the 11 healthy flights are from campaign 2, so a detector
  trained on healthy flights would flag campaign 1 for being campaign 1. It could only be tested
  fairly within campaign 2.

---

## Gong, Chung & Glass (2021), *AST: Audio Spectrogram Transformer*
`gong2021ast` · Interspeech 2021 · arXiv 2104.01778 · code: https://github.com/YuanGongND/ast

**What it is.** The first audio classifier with no convolutions: a standard Transformer encoder
applied to patches of a spectrogram, like the Vision Transformer (ViT) for images.

**How it works (§2.1, Fig. 1).**
- Input: 128-bin log-mel filterbank, 25-ms window every 10 ms → a 128 × 100t image for t seconds.
- Cut into 16 × 16 patches (overlap 6), each flattened and linearly projected to 768 numbers, plus
  a learned position embedding. A [CLS] token is prepended; its output feeds a linear classifier.
- Encoder: 12 layers, 12 heads, dimension 768; about 87 M parameters (Table 3).

**Numbers worth remembering.**
- **Pre-training is what makes it work.** On the balanced AudioSet subset (22k clips), mAP is
  0.347 with ImageNet pre-training and 0.148 without (Table 2). The authors state the reason
  themselves (§2.2): transformers need more data than CNNs, citing ViT's finding that they only
  overtake CNNs above ~14 M training images.
- State of the art at the time: 0.485 mAP on AudioSet, 95.6 % on ESC-50, 98.1 % on Speech
  Commands V2 (abstract, Tables 1 and 7).

**What it means for us.**
- Training AST (or any transformer) from scratch on 23 flights is ruled out by the paper's own
  Table 2: even 22k clips was too few without pre-training.
- **Use it frozen:** take the AudioSet-trained checkpoint, extract the 768-d [CLS] embedding of
  each audio window, and fit our logistic regression on top. That is rung R4 of the model ladder
  (D8). To check before relying on it: the sample rate the released checkpoints expect (our
  microphone records 16 kHz) and how to handle our 3-s windows (AudioSet clips are 10 s).

---

## Li et al. (2022), *A Hybrid System of Sound Event Detection Transformer and Frame-wise Model for DCASE 2022 Task 4*
`li2022sedt` · DCASE 2022 workshop (challenge technical report) · arXiv 2210.09529

**What it is.** A competition entry for *sound event detection* (SED): find which sounds occur in
a 10-s domestic recording **and when** (onset and offset). It combines two different models and
ranked 6th / 9th in the challenge (§6.2).

**Key ideas.**
- **SEDT (§2):** CNN backbone → transformer encoder–decoder. Each learnable "event query" outputs
  one event (class, centre, duration), matched to the ground truth with the Hungarian algorithm.
- **Semi-supervised training (§2.4, Algorithm 1):** a teacher model, an exponential moving average
  of the student, labels the unlabelled clips. Self-supervised pre-training on unlabelled audio (§2.3).
- **Late fusion (§4.2, Eqs. 7–8):** the final probability is a weighted average of the two models'
  probabilities, p̂_c = Σᵢ wᵢ,c pᵢ,c, with weights proportional to each model's validation score.
  The hybrid beats either model alone (Table 3).
- **Data:** thousands of clips (1,578 weakly labelled, 14,412 unlabelled, 10,000 synthetic; §6.1).

**What it means for us.**
- Different task: we don't need onsets and offsets, because a flight's condition doesn't change
  during the flight. Our task is clip classification ("audio tagging"), the simpler half of SED.
- **It supports our fusion choice:** decision-level fusion, averaging the probabilities of
  separately trained models (D8). Their weights are tuned on a validation set; with 23 flights we
  keep equal weights, so nothing extra is fitted (otherwise it would need nested CV).
- The transformer needed large unlabelled data and pre-training; same lesson as AST.
- Being a challenge technical report, it is weaker as a citation than a peer-reviewed paper.

---

## How the three fit together (for the thesis's related-work chapter)

| Question | Paper | Our answer |
|---|---|---|
| How to frame the problem | Boniol 2024 | multivariate, whole-sequence; supervised now, semi-supervised as an option |
| Can a transformer learn from our data? | Gong 2021 | not from scratch; frozen pre-trained embeddings only (R4) |
| How to combine audio and IMU | Li 2022 | decision-level fusion: average the per-sensor probabilities |
