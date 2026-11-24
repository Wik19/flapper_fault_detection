"""Label-free look at the IMU spectra: the first step in designing the R1 features (D8).

Answers three questions without comparing classes, so that no feature is chosen by looking at
what separates healthy from damaged:
  1. where slow manoeuvre motion ends and wingbeat vibration begins,
  2. whether the wingbeat peak is single or split, and how fast it moves during manoeuvres,
  3. how many harmonics stand above the noise.
Hidden gaps (D5) are ignored here: a phase jump smears a peak slightly but cannot create or move one.

Writes results/eda/psd.png and results/eda/spectrograms.png.
Run from the repo root:  .venv/bin/python scripts/eda_spectra.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import signal  # noqa: E402

RATE = 416
CHANNELS = ["Acc_X", "Acc_Y", "Acc_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]    # m/s² and rad/s
WELCH_N = 2048                  # 4.9-s segments: 416 / 2048 = 0.2 Hz per bin
SPEC_N, SPEC_HOP = 832, 104     # 2-s windows (0.5 Hz per bin), one every 0.25 s
SPEC_BAND = (8.0, 40.0)         # wingbeat and first harmonics, as in sync check Q3
OUT = "results/eda"
SURFACE, INK, INK2, MUTED, GRID, AXIS, BLUE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#2a78d6")


def average_spectrum(x):
    """Welch power spectral density of each channel.

    x: array (channels, samples). Returns f (n_freq,) in Hz and p (channels, n_freq).
    """
    # Time is the last axis of x, which is welch's default.
    return signal.welch(x, fs=RATE, nperseg=WELCH_N)


def moving_spectrum(x):
    """Spectrogram of each channel: the power spectrum of consecutive 2-s windows.

    x: array (channels, samples). Returns f (n_freq,) in Hz, t (n_times,) in s and
    s (channels, n_freq, n_times).
    """
    # Consecutive windows share all but one hop. Hann is passed explicitly because
    # spectrogram's default is a Tukey window, unlike welch.
    return signal.spectrogram(x, fs=RATE, nperseg=SPEC_N, noverlap=SPEC_N - SPEC_HOP, window="hann")


def self_check():
    """Both functions must put a pure 14 Hz tone at 14 Hz, to within one frequency bin."""
    x = np.sin(2 * np.pi * 14.0 * np.arange(20 * RATE) / RATE)[None, :]
    f, p = average_spectrum(x)
    assert abs(f[p[0].argmax()] - 14.0) <= RATE / WELCH_N, "average_spectrum: peak not at 14 Hz"
    f, _, s = moving_spectrum(x)
    assert np.all(np.abs(f[s[0].argmax(axis=0)] - 14.0) <= RATE / SPEC_N), \
        "moving_spectrum: peak not at 14 Hz"


def plot_psd(f, psds):
    """psds: (recordings, channels, n_freq). One thin line per recording, their median in bold."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey="row",
                             facecolor=SURFACE, layout="constrained")
    keep = f > 0                                        # 0 Hz cannot sit on a log axis
    for c, ax in enumerate(axes.flat):
        ax.set_facecolor(SURFACE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.loglog(f[keep], psds[:, c, keep].T, color=MUTED, linewidth=0.6, alpha=0.5)
        ax.loglog(f[keep], np.median(psds[:, c, keep], axis=0), color=BLUE, linewidth=2)
        ax.set_title(CHANNELS[c], loc="left", color=INK, fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("frequency (Hz)")
    axes[0, 0].set_ylabel("PSD ((m/s²)²/Hz)")
    axes[1, 0].set_ylabel("PSD ((rad/s)²/Hz)")
    axes[0, 0].legend([Line2D([], [], color=MUTED, linewidth=1), Line2D([], [], color=BLUE, linewidth=2)],
                      ["one recording", f"median of the {len(psds)}"],
                      frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")
    fig.suptitle(f"Average spectrum of every recording (Welch, {RATE / WELCH_N:.2f} Hz bins), "
                 "no labels shown", x=0.01, ha="left", color=INK, fontsize=11)
    fig.savefig(f"{OUT}/psd.png", dpi=150, facecolor=SURFACE)


def plot_spectrograms(names, specs):
    """specs: one (t, f, power) per recording, already cut to SPEC_BAND."""
    n_cols = 4
    n_rows = -(-len(names) // n_cols)                   # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 1.9 * n_rows), sharey=True,
                             facecolor=SURFACE, layout="constrained")
    for ax, name, (t, f, s) in zip(axes.flat, names, specs):
        db = 10 * np.log10(s / s.max())                 # dB below this recording's strongest bin
        mesh = ax.pcolormesh(t, f, db, vmin=-30, vmax=0, cmap="Blues", shading="nearest")
        ax.set_title(name, loc="left", color=INK, fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.colorbar(mesh, ax=axes, shrink=0.5, label="dB below the recording's maximum")
    fig.supxlabel("time in the IMU file (s)", color=INK2)
    fig.supylabel("frequency (Hz)", color=INK2)
    fig.suptitle(f"Accelerometer spectrogram, {SPEC_BAND[0]:.0f}–{SPEC_BAND[1]:.0f} Hz "
                 f"({SPEC_N / RATE:.0f}-s windows every {SPEC_HOP / RATE:.2f} s), power summed over x, y, z",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.savefig(f"{OUT}/spectrograms.png", dpi=130, facecolor=SURFACE)


def main():
    self_check()
    manifest = pd.read_csv("data/manifest.csv").query("label != 'excluded'")
    psds, specs = [], []
    for name in manifest.recording:
        x = pd.read_csv(f"data/raw/IMU/{name}.csv")[CHANNELS].to_numpy(dtype=float).T
        f, p = average_spectrum(x)
        psds.append(p)
        # Summing the power of the three accelerometer axes gives |a(f)|², which does not depend
        # on how the board is oriented: a rotation preserves the length of a vector.
        fs, t, s = moving_spectrum(x[:3])
        band = (fs >= SPEC_BAND[0]) & (fs <= SPEC_BAND[1])
        specs.append((t, fs[band], s.sum(axis=0)[band]))
        print(f"done {name}")
    os.makedirs(OUT, exist_ok=True)
    plot_psd(f, np.array(psds))
    plot_spectrograms(list(manifest.recording), specs)
    print(f"saved {OUT}/psd.png and {OUT}/spectrograms.png")


if __name__ == "__main__":
    main()
