import os
import glob
import math
import wave

# ==========================================
# Class Label Mapping (single source of truth)
# ==========================================
CLASS_NAMES = {
    0: "Healthy",
    1: "hole1",
    2: "hole2",
    3: "tear_n_hole2",
    4: "fixed_all_tape",
}


def get_class_label(filename):
    """Maps filenames to exact integer classes."""
    if "Healthy" in filename: return 0
    elif "tear_n_hole2" in filename: return 3
    elif "hole1" in filename: return 1
    elif "hole2" in filename: return 2
    elif "fixed_all_tape" in filename: return 4
    else: raise ValueError(f"Could not map {filename}.")


def _count_chunks(audio_path, samples_per_window, hop_samples):
    with wave.open(audio_path, "rb") as wav_file:
        num_frames = wav_file.getnframes()
    if num_frames < samples_per_window:
        return 0
    return ((num_frames - samples_per_window) // hop_samples) + 1


def build_chunk_index(data_dir, is_train, audio_samples_per_window, audio_hop_samples,
                      window_sec, hop_sec, split_ratio, verbose=False):
    """Build the list of chunk records for the train or validation split.

    Splitting strategy (group-aware, leakage-free):

      * Classes with >= 2 recordings -> hold out whole recording(s) for validation.
        Because train and validation never share a recording, overlapping windows
        cannot leak across the split.

      * Classes with a single recording (currently Healthy and fixed_all_tape) ->
        temporal split inside the recording, with a guard gap so that no training
        window overlaps a validation window. This is a weaker protocol (same
        recording, different time segment) and is the best achievable until more
        recordings exist for those classes.

    Returns a list of dicts: {audio_path, csv_path, chunk_idx, label}.
    """
    mic_dir = os.path.join(data_dir, "MIC")
    imu_dir = os.path.join(data_dir, "IMU")

    # --- Group paired recordings by class label ---
    by_label = {}
    for audio_path in sorted(glob.glob(os.path.join(mic_dir, "*.wav"))):
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        csv_path = os.path.join(imu_dir, f"{base_name}.csv")
        if not os.path.exists(csv_path):
            continue
        total_chunks = _count_chunks(audio_path, audio_samples_per_window, audio_hop_samples)
        if total_chunks == 0:
            continue
        by_label.setdefault(get_class_label(base_name), []).append({
            "base_name": base_name,
            "audio_path": audio_path,
            "csv_path": csv_path,
            "total_chunks": total_chunks,
        })

    # Minimum chunk distance for two windows NOT to overlap in time.
    guard = math.ceil(window_sec / hop_sec)

    train_index, val_index = [], []
    plan = []

    def add(index, rec, chunks):
        for chunk_idx in chunks:
            index.append({
                "audio_path": rec["audio_path"],
                "csv_path": rec["csv_path"],
                "chunk_idx": chunk_idx,
                "label": label,
            })

    for label in sorted(by_label):
        recordings = sorted(by_label[label], key=lambda r: r["base_name"])
        n_rec = len(recordings)

        if n_rec >= 2:
            # Group split: hold out whole recording(s) for validation.
            n_val = max(1, round(n_rec * (1.0 - split_ratio)))
            n_val = min(n_val, n_rec - 1)  # always keep at least one in train
            train_recs, val_recs = recordings[:n_rec - n_val], recordings[n_rec - n_val:]
            for rec in train_recs:
                add(train_index, rec, range(rec["total_chunks"]))
            for rec in val_recs:
                add(val_index, rec, range(rec["total_chunks"]))
            plan.append((label, "group", n_rec,
                         sum(r["total_chunks"] for r in train_recs),
                         sum(r["total_chunks"] for r in val_recs),
                         [r["base_name"] for r in val_recs]))
        else:
            # Temporal split inside the single recording, with a guard gap.
            rec = recordings[0]
            total = rec["total_chunks"]
            train_count = int(total * split_ratio)
            train_chunks = list(range(0, train_count))
            val_chunks = list(range(train_count + guard, total))
            add(train_index, rec, train_chunks)
            add(val_index, rec, val_chunks)
            plan.append((label, "temporal", 1,
                         len(train_chunks), len(val_chunks), [rec["base_name"]]))

    if verbose:
        print("\n[split] Group-aware split (window={:.1f}s hop={:.1f}s ratio={:.2f}, guard={} chunks)"
              .format(window_sec, hop_sec, split_ratio, guard))
        print("[split] {:<14} {:<9} {:>5} {:>7} {:>7}  {}".format(
            "class", "mode", "recs", "train", "val", "val recordings"))
        for label, mode, n_rec, n_tr, n_va, val_recs in plan:
            print("[split] {:<14} {:<9} {:>5} {:>7} {:>7}  {}".format(
                CLASS_NAMES.get(label, label), mode, n_rec, n_tr, n_va, ", ".join(val_recs)))
        print("[split] TOTAL train chunks: {} | val chunks: {}\n".format(
            len(train_index), len(val_index)))

    return train_index if is_train else val_index
