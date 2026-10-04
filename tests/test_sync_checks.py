import numpy as np

from sync_checks import packet_continuity


def test_toy_stream_with_one_lost_packet():
    ids = np.array([7, 7, 7, 8, 8, 8, 10, 10, 10, 11])
    result = packet_continuity(ids)

    # packets start at rows [0, 3, 6, 9] -> sizes [3, 3, 3, 1]; IDs [7, 8, 10, 11] -> steps [1, 2, 1]
    assert result["n_packets"] == 4
    assert result["first_size"] == 3
    assert result["last_size"] == 1
    assert result["lost_packets"] == 1    # the step 8 -> 10 skips packet 9
    assert result["backward_steps"] == 0
