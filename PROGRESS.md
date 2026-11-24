# PROGRESS

**Current milestone:** M3 — Label-free EDA → R1 features. The figures are ready
(`results/eda/`); next is your answers to the five reading questions (below), then the feature
list (D9). Start in a new session.

M2 — Synchronisation checks: DONE 2026-10-05 (no quiz yet)

M1 — Provenance: DONE 2026-10-04

M0 — Setup and quarantine: DONE 2026-10-02

## Code so far
| File | What it does | Written by | Study? |
|---|---|---|---|
| `src/sync_checks.py` | Q1 packet continuity, Q2 hidden gaps, Q3 audio–IMU lags (method in `docs/SYNC_CHECKS.md`) | Claude, Oct (tutor mode) | **study it**: you'll defend D5–D7 with it |
| `tests/test_sync_checks.py` | 4 tests on synthetic packets and signals | Claude | study it |
| `scripts/run_sync_checks.py` | runs Q1–Q3 on every recording → `results/sync/*.csv` | Claude | helper |
| `scripts/plot_sync_example.py` | `docs/figures/sync_hole2_full.png` | Claude | helper |
| `scripts/eda_spectra.py` | Welch spectra + spectrograms of all flights, no labels → `results/eda/` | Claude; the two `scipy.signal` calls filled by Claude on 2026-11-24 | **study the two calls** (`welch`, `spectrogram`); plotting is helper |
| `scripts/plot_recording.py` | raw signals + spectrograms of one recording → `results/figures/` | Claude | helper |

Not written yet: windows (M5), features (M3/M6), evaluation (M4/M6), models (M6–M9).
`.venv/bin/python -m pytest`: 4 passed (2026-11-24).

## Docs so far
- `docs/DECISIONS.md` — D1–D8 (task, damage codes, campaigns, provenance, gaps, gap
  exclusion, audio re-alignment, model ladder).
- `docs/SYNC_CHECKS.md` — method, validation and results of Q1–Q3.
- `docs/LEGACY_HISTORY.md`, `legacy/LEGACY_AUDIT.md` — what the old pipeline did and why its 0.93 is optimistic.
- `docs/PROGRESS_2026-11-25.md` — summary for the 25 Nov supervisor meeting.
- `literature/` — notes + BibTeX for the papers sent by the supervisor (PDFs git-ignored).

## M3 checklist
- [x] `scripts/eda_spectra.py` written, self-check (14 Hz tone) passes
- [x] figures: `results/eda/psd.png`, `results/eda/spectrograms.png`
- [ ] answer the reading questions (describe, don't compare classes):
  1. `psd.png`: where does the low-frequency (manoeuvre) hump fall to the floor? → boundary between m(t) and v(t)
  2. `psd.png`: wingbeat frequency f? How many multiples 2f, 3f, … are visible?
  3. `psd.png`: peaks that are *not* multiples of f? Possible causes?
  4. `spectrograms.png`: how far and how fast (Hz/s) does the wingbeat line move within a flight? Compare with the ~5-s window that Δf = 0.4 Hz would need.
  5. `spectrograms.png`: one line, or two close lines / beating?
- [ ] R1 feature list (formula + physical rationale each) → D9
- [ ] M3 quiz

## Open decisions / waiting on
- **M4, split:** leave-one-recording-out + train-campaign-2/test-campaign-1 was proposed but
  never explicitly confirmed. Points 3–6 of the protocol are not discussed yet.
- **Supervisor (25 Nov):** supervised vs semi-supervised framing; is "unseen flights of these
  wing sets" an acceptable claim; time for a third recording campaign?
- **H1S9:** its spectrogram is unusually steady (almost no manoeuvring). Was it a different
  kind of flight (tethered, hovering in place)?
- Sensitivity run without H1S8–10 (healthy flights recorded after the damaged ones): suggested, not decided.
- Label-free check of manoeuvre intensity per recording, compared across classes: planned for M4.

## Struggled with
- Window length vs frequency resolution: suggested 16.6-s windows without a derivation.
  Rule: resolving two lines Δf apart needs T ≳ 1/Δf, about 2/Δf with a Hann window; longer
  windows also smear a frequency that moves during manoeuvres and lose more windows to gaps (D6).

## Open bugs / watch list
- `microcontroller-firmware/` (untracked) contains plaintext WiFi credentials: never commit as-is.
- Acc_X hits the ±16 g limit in 0.12 % of samples (max 0.28 % in one recording): clipping adds
  spurious harmonics. Gyro never clips.
- No `requirements.txt` yet (`pyproject.toml` only configures pytest).
- The project `/eda` skill is broken; fix or delete it.
- `results/` is git-ignored: figures shown to others must be regenerated or copied to `docs/figures/`.
