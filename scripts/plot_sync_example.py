"""Figure for docs/SYNC_CHECKS.md: Q2 (lateness floor) and Q3 (audio-IMU offset) for one recording.

Needs results/sync/alignment.csv, so run scripts/run_sync_checks.py first.
Run from the repo root:  .venv/bin/python scripts/plot_sync_example.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sync_checks import hidden_gaps  # noqa: E402

NAME, RATE = "hole2_full", 416
OUT = "docs/figures/sync_hole2_full.png"
SURFACE, INK, INK2, MUTED, GRID, AXIS, BLUE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#2a78d6")

imu = pd.read_csv(f"data/raw/IMU/{NAME}.csv")
q2 = hidden_gaps(imu["Packet_ID"], imu["Rx_Time_Sec"], rate=RATE)
x = q2["packet_last_sample"] / RATE
align = pd.read_csv("results/sync/alignment.csv").query("recording == @NAME")

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, facecolor=SURFACE,
                               gridspec_kw={"height_ratios": [3, 2], "hspace": 0.35})
for ax in (ax1, ax2):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)

# Q2: every packet's lateness, and the floor of each segment long enough for its backlog to
# drain (short segments inside a burst of stalls have no reliable floor and are not drawn)
ax1.scatter(x, q2["lateness_s"], s=6, color=MUTED, alpha=0.45, linewidths=0, label="each packet")
size = np.bincount(q2["segment"])
drawn = False
for s, fl in enumerate(q2["floor_s"]):
    if s in (0, len(size) - 1) or size[s] >= 30:
        xs = x[q2["segment"] == s]
        ax1.hlines(fl, xs.min(), xs.max(), color=BLUE, linewidth=2.5,
                   label=None if drawn else "floor of each segment")
        drawn = True
for k in np.argsort(q2["lost_s"])[-3:]:                     # label the three biggest events
    ax1.annotate(f"+{q2['lost_s'][k]:.1f} s lost", (q2["drop_at_sample"][k] / RATE, q2["floor_after_s"][k]),
                 xytext=(4, -14), textcoords="offset points", color=INK2, fontsize=9)
ax1.set_ylabel("lateness (s)")
ax1.set_title(f"Q2 · {NAME}: packet lateness vs a perfect 416 Hz clock — the floor steps up "
              f"at each stall\n(total {q2['total_lost_s']:.1f} s of IMU data lost)",
              loc="left", color=INK, fontsize=10.5)
ax1.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK2)

# Q3: audio-minus-IMU offset along the recording
ax2.axhline(0, color=AXIS, linewidth=1)
ax2.plot(align.centre_s, align.lag_s, color=BLUE, linewidth=2, solid_capstyle="round")
ax2.scatter(align.centre_s, align.lag_s, s=42, color=BLUE, edgecolors=SURFACE, linewidths=2, zorder=3)
ax2.set_ylabel("audio − IMU offset (s)")
ax2.set_xlabel("time in the IMU file (s)")
ax2.set_title(f"Q3 · {NAME}: offset between the streams (one point per 16-s window)",
              loc="left", color=INK, fontsize=10.5)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor=SURFACE)
print(f"saved {OUT}")
