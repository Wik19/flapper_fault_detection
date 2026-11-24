"""Quick look at one or more recordings: raw signals and spectrograms (not part of the pipeline).

Writes three figures per recording to results/figures/ (git-ignored):
  <name>_signals.png       audio, accelerometer and gyroscope over the whole recording
  <name>_signals_zoom.png  the same over a short stretch, to see individual wingbeats
  <name>_spectrograms.png  spectrogram of the audio and of each IMU axis
Caveats: time is sample index / rate, so hidden gaps are ignored (D5), and audio and IMU are not
re-aligned (D7): they can be offset by up to ~1.4 s.

Run from the repo root:
  .venv/bin/python scripts/plot_recording.py                     # H1S2
  .venv/bin/python scripts/plot_recording.py H1S2 Hole1S1
  .venv/bin/python scripts/plot_recording.py hole2_full --zoom 20 --zoom-len 0.5
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import mlab  # noqa: E402
from scipy.io import wavfile  # noqa: E402

IMU_RATE = 416
ACC = ["Acc_X", "Acc_Y", "Acc_Z"]          # m/s²
GYRO = ["Gyro_X", "Gyro_Y", "Gyro_Z"]      # rad/s
AXIS_COLORS = ["#0072b2", "#e69f00", "#009e73"]     # Okabe–Ito, colour-blind safe; always x, y, z
AUDIO_NFFT, AUDIO_HOP = 1024, 256          # 64-ms windows (15.6 Hz bins), every 16 ms
IMU_NFFT, IMU_HOP = 512, 52                # 1.2-s windows (0.8 Hz bins), every 0.125 s
DB_RANGE = 50                              # colour scale: 0 to -50 dB below each panel's maximum
OUT = "results/figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"


def load(name):
    """Audio scaled to [-1, 1) and the IMU table."""
    rate, audio = wavfile.read(f"data/raw/MIC/{name}.wav")
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio / (np.iinfo(audio.dtype).max + 1)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return rate, audio, pd.read_csv(f"data/raw/IMU/{name}.csv")


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_signals(name, rate, audio, imu, t0, t1, path, lw):
    """Three stacked panels over [t0, t1] s: audio, accelerometer, gyroscope."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, facecolor=SURFACE, layout="constrained")
    a = slice(int(t0 * rate), int(t1 * rate))
    axes[0].plot(np.arange(len(audio))[a] / rate, audio[a], color=INK2, linewidth=lw)
    axes[0].set_ylabel("audio (full scale)")
    i = slice(int(t0 * IMU_RATE), int(t1 * IMU_RATE))
    t_imu = np.arange(len(imu))[i] / IMU_RATE
    for ax, cols, unit in [(axes[1], ACC, "acceleration (m/s²)"), (axes[2], GYRO, "angular rate (rad/s)")]:
        for col, color in zip(cols, AXIS_COLORS):
            ax.plot(t_imu, imu[col].to_numpy()[i], color=color, linewidth=lw, label=col)
        ax.set_ylabel(unit)
        ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1),
                  fontsize=9, labelcolor=INK2, borderaxespad=0.2)
    for ax in axes:
        style(ax)
    axes[-1].set_xlabel("time (s)")
    axes[-1].set_xlim(t0, t1)
    fig.suptitle(f"{name}, {t0:.1f}–{t1:.1f} s", x=0.01, ha="left", color=INK)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def show_spectrogram(ax, x, rate, nfft, hop, title):
    """Hann-windowed power spectrogram (mean removed per window), in dB below its own maximum."""
    s, f, t = mlab.specgram(x, NFFT=nfft, Fs=rate, noverlap=nfft - hop, detrend="mean")
    db = 10 * np.log10(s / s.max() + 1e-12)
    dt, df = hop / rate, rate / nfft
    image = ax.imshow(db, origin="lower", aspect="auto", cmap="Blues", vmin=-DB_RANGE, vmax=0,
                      extent=(t[0] - dt / 2, t[-1] + dt / 2, f[0] - df / 2, f[-1] + df / 2))
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.set_ylabel("frequency (Hz)")
    return image


def plot_spectrograms(name, rate, audio, imu, path):
    """Audio across the top, accelerometer axes on the left, gyroscope axes on the right."""
    fig = plt.figure(figsize=(13, 10), facecolor=SURFACE, layout="constrained")
    grid = fig.add_gridspec(4, 2)
    top = fig.add_subplot(grid[0, :])
    image = show_spectrogram(top, audio, rate, AUDIO_NFFT, AUDIO_HOP, "Microphone")
    for col_idx, cols in enumerate([ACC, GYRO]):
        for row, col in enumerate(cols, start=1):
            ax = fig.add_subplot(grid[row, col_idx], sharex=top)
            show_spectrogram(ax, imu[col].to_numpy(dtype=float), IMU_RATE, IMU_NFFT, IMU_HOP, col)
            if row == 3:
                ax.set_xlabel("time (s)")
    fig.colorbar(image, ax=fig.axes, shrink=0.4, label="dB below the panel's maximum")
    fig.suptitle(name, x=0.01, ha="left", color=INK)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", nargs="*", default=["H1S2"], help="base names in data/raw")
    parser.add_argument("--zoom", type=float, help="start of the zoomed stretch in s (default: the middle)")
    parser.add_argument("--zoom-len", type=float, default=1.0, help="length of the zoomed stretch in s")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)

    for name in args.recordings:
        rate, audio, imu = load(name)
        end = max(len(audio) / rate, len(imu) / IMU_RATE)
        zoom = min(len(audio) / rate, len(imu) / IMU_RATE) / 2 if args.zoom is None else args.zoom
        base = f"{OUT}/{name}"
        plot_signals(name, rate, audio, imu, 0, end, f"{base}_signals.png", lw=0.4)
        plot_signals(name, rate, audio, imu, zoom, zoom + args.zoom_len, f"{base}_signals_zoom.png", lw=1.2)
        plot_spectrograms(name, rate, audio, imu, f"{base}_spectrograms.png")
        print(f"saved {base}_signals.png, {base}_signals_zoom.png, {base}_spectrograms.png")


if __name__ == "__main__":
    main()
