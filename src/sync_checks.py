"""Synchronisation checks run before any preprocessing (CLAUDE.md methodology rule 1).

Every function here takes plain arrays, not file paths, so it can be tested on toy data.
The reasoning behind each check, and its results, are in docs/SYNC_CHECKS.md.
"""
from collections import Counter

import numpy as np
from scipy import signal


def packet_starts(ids):
    """Row index where each packet begins: row 0, plus every row whose ID differs from the row before."""
    ids = np.asarray(ids)
    return np.concatenate(([0], np.flatnonzero(np.diff(ids) != 0) + 1))


def packet_continuity(packet_ids):
    """Q1: check that IMU packets arrive complete and in order.

    packet_ids: 1-D integer array, the Packet_ID of every row, in file order.
    """
    ids = np.asarray(packet_ids)

    starts = packet_starts(ids)
    sizes = np.diff(np.append(starts, len(ids)))
    steps = np.diff(ids[starts])

    return {
        "n_samples": len(ids),
        "n_packets": len(starts),
        "first_size": int(sizes[0]),
        "last_size": int(sizes[-1]),
        "interior_sizes": Counter(sizes[1:-1].tolist()),  # expect {20: n_packets - 2}
        "lost_packets": int(np.sum(steps[steps > 1] - 1)),
        "backward_steps": int(np.sum(steps < 1)),
    }


def hidden_gaps(packet_ids, rx_times, rate=416.0, stall_s=0.5, tol_s=0.05, min_packets=30):
    """Q2: locate and size the samples dropped on the board (D5), using receive times.

    Lateness of packet k: r_k = t_k - i_k / rate, where t_k is its receive time and i_k the index
    of its last sample. Network delay can only make a packet later, never earlier, so the minimum
    lateness within a stretch without stalls (the "floor") equals the time lost so far plus the
    smallest possible delay. The floor steps up by exactly the time lost at each stall.

    Where the samples went missing: after a stall the board first sends its queued backlog (old
    samples, so very late), then fresh samples (on the new floor). The drop sits between the two,
    so it is located at the first packet that arrives within tol_s of the new floor.

    Bursts: a segment with fewer than min_packets packets (other than the first and last) is too
    short for the backlog to drain, so its floor is unreliable. It is skipped, and the stalls on
    either side of it form one event, measured between the long segments around it.

    packet_ids, rx_times: the Packet_ID and Rx_Time_Sec of every row, in file order.
    """
    ids = np.asarray(packet_ids)
    rx = np.asarray(rx_times, dtype=float)

    starts = packet_starts(ids)
    t = rx[starts]                                   # one receive time per packet
    last = np.append(starts[1:], len(ids)) - 1       # index of each packet's last sample

    late = t - last / rate
    seg = np.concatenate(([0], np.cumsum(np.diff(t) > stall_s)))   # +1 after every stall
    n_seg = seg[-1] + 1
    size = np.bincount(seg)
    floor = np.array([late[seg == s].min() for s in range(n_seg)])

    good = [s for s in range(n_seg) if s in (0, n_seg - 1) or size[s] >= min_packets]
    before, after = np.array(good[:-1], dtype=int), np.array(good[1:], dtype=int)
    lost = floor[after] - floor[before]

    end_before = np.array([np.flatnonzero(seg == s)[-1] for s in before], dtype=int)
    on_floor = np.array([np.flatnonzero((seg == s) & (late <= floor[s] + tol_s))[0]
                         for s in after], dtype=int)
    return {
        "stall_at_sample": last[end_before],         # last sample before the event's first stall
        "drop_at_sample": starts[on_floor],          # first sample after its last gap, +-1 packet
        "n_stalls": after - before,
        "stall_duration_s": t[end_before + 1] - t[end_before],   # duration of the first stall
        "lost_s": lost,
        "total_lost_s": float(lost.sum()),
        "floor_after_s": floor[after],               # floor of the segment following each event
        # per-packet intermediates, for plotting and debugging
        "packet_last_sample": last,
        "lateness_s": late,
        "segment": seg,
        "floor_s": floor,
    }


def spectral_shape(x, fs=416.0, band=(8.0, 40.0), frame_s=2.0, hop_s=0.125):
    """Log-power spectrogram restricted to `band`, each frame normalised over frequency.

    Normalising each frame (zero mean, unit std across frequency) keeps *where* the peaks are
    (the wingbeat and its harmonics) and discards *how loud* the frame is, because loudness does
    not agree between the microphone and the IMU. Returns an array (n_freq, n_frames).
    """
    nper, hop = int(frame_s * fs), int(hop_s * fs)
    f, _, S = signal.spectrogram(x - np.mean(x), fs, nperseg=nper, noverlap=nper - hop)
    L = np.log(S[(f >= band[0]) & (f <= band[1])] + 1e-12)
    return (L - L.mean(axis=0)) / L.std(axis=0)


def alignment_lags(audio, imu, fs=416.0, window_s=16.0, step_s=4.0, max_lag_s=3.0,
                   frame_s=2.0, hop_s=0.125):
    """Q3: time offset between the audio and the IMU, along the recording.

    audio: 1-D array already resampled to `fs`; imu: array (n_samples, n_channels) at `fs`.
    The IMU's shape is the average over its channels, so no single axis is assumed to carry
    the shared signal. For each window of IMU frames, the audio frames are slid by every lag in
    +-max_lag_s and scored by the mean frame-by-frame correlation of the two shapes.

    Sign convention: lag > 0 means an event appears later in the audio file than in the IMU file.
    Returns arrays: window centre (s, IMU time), best lag (s), its score, and the best score at
    least 0.5 s away from it (runner-up; a clear gap between the two means an unambiguous peak).
    """
    A = spectral_shape(audio, fs, frame_s=frame_s, hop_s=hop_s)
    B = np.mean([spectral_shape(imu[:, c], fs, frame_s=frame_s, hop_s=hop_s)
                 for c in range(imu.shape[1])], axis=0)
    B = (B - B.mean(axis=0)) / B.std(axis=0)

    n = min(A.shape[1], B.shape[1])
    w, step, m = (round(v / hop_s) for v in (window_s, step_s, max_lag_s))
    if n < w + 2 * m:                                # short recording: a single window
        m = min(m, n // 4)
        w = n - 2 * m
    lags = np.arange(-m, m + 1)
    away = round(0.5 / hop_s)

    centre, best, score, runner_up = [], [], [], []
    for s in range(m, n - w - m + 1, step):
        b = B[:, s:s + w]
        sc = np.array([np.mean(np.sum(A[:, s + L:s + L + w] * b, axis=0)) for L in lags]) / A.shape[0]
        k = np.argmax(sc)
        centre.append(frame_s / 2 + (s + w / 2) * hop_s)
        best.append(lags[k] * hop_s)
        score.append(sc[k])
        runner_up.append(sc[np.abs(lags - lags[k]) > away].max())
    return {
        "centre_s": np.array(centre),
        "lag_s": np.array(best),
        "score": np.array(score),
        "runner_up": np.array(runner_up),
    }
