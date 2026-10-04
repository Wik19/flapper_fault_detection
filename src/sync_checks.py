"""Synchronisation checks run before any preprocessing (CLAUDE.md methodology rule 1).

Every function here takes plain arrays, not file paths, so it can be tested on toy data.
"""
from collections import Counter

import numpy as np


def packet_continuity(packet_ids):
    """Check that IMU packets arrive complete and in order.

    packet_ids: 1-D integer array, the Packet_ID of every row, in file order.
    """
    ids = np.asarray(packet_ids)

    starts = np.concatenate(([0], np.flatnonzero(np.diff(ids) != 0) +1))
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
