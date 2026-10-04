"""Run the three synchronisation checks (docs/SYNC_CHECKS.md) on every recording in the manifest.

Writes, under results/sync/:
  summary.csv    one row per recording
  gaps.csv       one row per WiFi stall (Q2)
  alignment.csv  one row per alignment window (Q3)
Run from the repo root:  .venv/bin/python scripts/run_sync_checks.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sync_checks import alignment_lags, hidden_gaps, packet_continuity  # noqa: E402

RATE = 416
LOSSY_S = 0.05          # a stall that lost more than this counts as a real discontinuity (D6)
OUT = "results/sync"
IMU_COLS = ["Acc_X", "Acc_Y", "Acc_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]


def receive_times(row, n_rows):
    """Rows with real receive times for this recording, and where the recording starts in them.

    Uncropped files kept their receive times. Cropped files had them overwritten by the legacy
    crop script (D4), so the untouched original is used and the crop start row is the offset.
    Returns None when no file with real receive times is available (campaign-2 crops, for now).
    """
    if row.cropped == "no":
        return pd.read_csv(f"data/raw/IMU/{row.recording}.csv"), 0
    if isinstance(row.original_file, str):
        return pd.read_csv(f"data/original/IMU/{row.original_file}.csv"), round(row.crop_start_s * RATE)
    return None


def main():
    manifest = pd.read_csv("data/manifest.csv")
    summary, gap_rows, align_rows = [], [], []

    for row in manifest.itertuples():
        imu = pd.read_csv(f"data/raw/IMU/{row.recording}.csv")
        _, audio = wavfile.read(f"data/raw/MIC/{row.recording}.wav")
        out = {"recording": row.recording, "label": row.label, "campaign": row.campaign,
               "imu_s": len(imu) / RATE, "audio_s": len(audio) / 16000}

        # Q1: packet continuity
        q1 = packet_continuity(imu["Packet_ID"])
        out["q1_pass"] = (q1["lost_packets"] == 0 and q1["backward_steps"] == 0
                          and set(q1["interior_sizes"]) <= {20})

        # Q2: hidden gaps, from receive times
        src = receive_times(row, len(imu))
        if src is not None:
            full, offset = src
            q2 = hidden_gaps(full["Packet_ID"], full["Rx_Time_Sec"], rate=RATE)
            at = q2["stall_at_sample"] - offset
            drop = q2["drop_at_sample"] - offset
            inside = (at >= 0) & (at < len(imu) - 1)
            lost = q2["lost_s"][inside]
            for a, dr, n, d, s in zip(at[inside], drop[inside], q2["n_stalls"][inside],
                                      q2["stall_duration_s"][inside], lost):
                gap_rows.append({"recording": row.recording, "stall_at_sample": int(a),
                                 "stall_at_s": a / RATE, "drop_at_sample": int(dr),
                                 "drop_at_s": dr / RATE, "n_stalls": int(n),
                                 "stall_duration_s": d, "lost_s": s})
            out.update(q2_stalls=int(inside.sum()), q2_lossy_stalls=int((lost > LOSSY_S).sum()),
                       q2_lost_s=lost.sum(), q2_max_lost_s=lost.max() if lost.size else 0.0)
            if row.cropped == "no":
                # cross-check: total time missing per stream over the whole session
                span = full["Rx_Time_Sec"].iloc[-1] - full["Rx_Time_Sec"].iloc[0]
                out.update(session_s=span, imu_missing_s=span - (len(full) - 1) / RATE,
                           audio_missing_s=span - len(audio) / 16000)

        # Q3: audio-IMU alignment
        a416 = signal.resample_poly(audio.astype(float), 13, 500)      # 16000 Hz -> 416 Hz
        q3 = alignment_lags(a416, imu[IMU_COLS].to_numpy(dtype=float), fs=RATE)
        for c, lag, sc, ru in zip(q3["centre_s"], q3["lag_s"], q3["score"], q3["runner_up"]):
            align_rows.append({"recording": row.recording, "centre_s": c, "lag_s": lag,
                               "score": sc, "runner_up": ru})
        out.update(q3_windows=len(q3["lag_s"]), q3_first_lag_s=q3["lag_s"][0],
                   q3_last_lag_s=q3["lag_s"][-1], q3_max_abs_lag_s=np.abs(q3["lag_s"]).max(),
                   q3_median_score=np.median(q3["score"]),
                   q3_median_margin=np.median(q3["score"] - q3["runner_up"]))
        summary.append(out)
        print(f"done {row.recording}")

    os.makedirs(OUT, exist_ok=True)
    summary = pd.DataFrame(summary)
    summary.to_csv(f"{OUT}/summary.csv", index=False, float_format="%.3f")
    pd.DataFrame(gap_rows).to_csv(f"{OUT}/gaps.csv", index=False, float_format="%.3f")
    pd.DataFrame(align_rows).to_csv(f"{OUT}/alignment.csv", index=False, float_format="%.3f")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
