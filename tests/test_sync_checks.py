import numpy as np
import pytest

from sync_checks import alignment_lags, hidden_gaps, packet_continuity


def test_toy_stream_with_one_lost_packet():
    ids = np.array([7, 7, 7, 8, 8, 8, 10, 10, 10, 11])
    result = packet_continuity(ids)

    # packets start at rows [0, 3, 6, 9] -> sizes [3, 3, 3, 1]; IDs [7, 8, 10, 11] -> steps [1, 2, 1]
    assert result["n_packets"] == 4
    assert result["first_size"] == 3
    assert result["last_size"] == 1
    assert result["lost_packets"] == 1    # the step 8 -> 10 skips packet 9
    assert result["backward_steps"] == 0


def test_hidden_gap_of_three_samples():
    # The firmware's queue mechanism at toy scale: 10 Hz, 2 samples per packet, a 2-sample queue.
    # WiFi stalls from 0.35 s to 0.9 s. Samples measured at 0.4 and 0.5 s fill the queue
    # (they become samples 4, 5), those at 0.6-0.8 s are dropped, and sampling resumes at 0.9 s
    # (samples 6-9 at 0.9-1.2 s). Packet IDs stay continuous throughout.
    ids = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    rx = [0.15, 0.15, 0.35, 0.35, 0.95, 0.95, 1.05, 1.05, 1.25, 1.25]   # 0.6 s stall before packet 2
    result = hidden_gaps(ids, rx, rate=10.0)

    # lateness [0.05, 0.05, 0.45, 0.35, 0.35]; segments [0, 0, 1, 1, 1]; floors [0.05, 0.35]
    assert result["stall_at_sample"].tolist() == [3]
    assert result["lost_s"] == pytest.approx([0.30])                  # 3 samples at 10 Hz
    assert result["total_lost_s"] == pytest.approx(0.30)
    assert result["drop_at_sample"].tolist() == [6]                   # the gap is between samples 5 and 6


def test_burst_of_two_stalls_counts_as_one_event():
    # As above, but a second stall (0.95-1.55 s) hits before the backlog has drained: samples at
    # 0.9 and 1.0 s are queued (6, 7), those at 1.1-1.4 s are dropped, sampling resumes at 1.5 s.
    # 3 + 4 = 7 samples are lost in total. The single packet between the two stalls is backlog.
    ids = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    rx = [0.15, 0.15, 0.35, 0.35, 0.95, 0.95, 1.6, 1.6, 1.65, 1.65, 1.85, 1.85]
    result = hidden_gaps(ids, rx, rate=10.0, min_packets=2)

    # lateness [0.05, 0.05, 0.45, 0.9, 0.75, 0.75]: packet 2 alone would give a floor of 0.45
    # and split the loss wrongly (0.40 + 0.30); merged, the event's floor is 0.75.
    assert result["n_stalls"].tolist() == [2]
    assert result["lost_s"] == pytest.approx([0.70])                  # 7 samples at 10 Hz
    assert result["drop_at_sample"].tolist() == [8]                   # after the last gap


def test_alignment_recovers_a_known_delay():
    # A wingbeat-like tone whose frequency steps between levels (as throttle changes do),
    # seen by two "sensors" with different gain, sign and noise. The audio lags by 0.5 s.
    rng = np.random.default_rng(0)
    fs, seconds, delay = 416, 60, 208                                 # 208 samples = 0.5 s
    levels = rng.uniform(12, 16, size=seconds)                        # one level per second
    freq = np.repeat(levels, fs)
    phase = 2 * np.pi * np.cumsum(freq) / fs
    tone = np.sin(phase) + 0.6 * np.sin(2 * phase)
    imu = (tone + 0.3 * rng.standard_normal(tone.size))[:, None]
    audio = -2.0 * np.concatenate([np.zeros(delay), tone[:-delay]])
    audio += 0.3 * rng.standard_normal(audio.size)

    result = alignment_lags(audio, imu, fs=fs)

    assert np.all(np.abs(result["lag_s"] - 0.5) <= 0.125)            # within one 0.125-s hop
    assert np.all(result["score"] > result["runner_up"])
