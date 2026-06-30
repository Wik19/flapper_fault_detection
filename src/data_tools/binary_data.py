"""Binary (HEALTHY vs DAMAGED) data handling for the lean fault detector.

Differences from the 5-class pipeline:
  * `fixed_all_tape` is excluded (ambiguous: repaired but non-pristine).
  * The dataset receives an explicit list of chunk records, so the caller
    (cross_validate.py) fully controls the train/test split per fold.
  * Audio is RMS-normalized (kills recording-gain as a shortcut) and the
    training set is augmented to discourage memorizing recording-specific cues.
"""
import os
import glob
import wave

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

HEALTHY = 0
DAMAGED = 1
CLASS_NAMES = {HEALTHY: "Healthy", DAMAGED: "Damaged"}


DAMAGE_TOKENS = ("hole", "tear", "crack", "broken", "damage")


def binary_label(filename):
    """Map a recording's base name to a binary label, or None to exclude it.

    Case-insensitive and token-based so newly collected files just need a
    recognisable word in the name: 'healthy' -> Healthy, any damage word
    (hole/tear/crack/broken/damage) -> Damaged, 'tape' -> excluded.
    """
    name = filename.lower()
    if "tape" in name:
        return None  # excluded: repaired-but-modified wing
    if "healthy" in name:
        return HEALTHY
    if any(token in name for token in DAMAGE_TOKENS):
        return DAMAGED
    raise ValueError(
        f"Could not map {filename!r} to a binary label. Put 'healthy' or a "
        f"damage word ({', '.join(DAMAGE_TOKENS)}) in the filename.")


def _count_chunks(audio_path, samples_per_window, hop_samples):
    with wave.open(audio_path, "rb") as wav_file:
        num_frames = wav_file.getnframes()
    if num_frames < samples_per_window:
        return 0
    return ((num_frames - samples_per_window) // hop_samples) + 1


def gather_recordings(data_dir, audio_sr=16000, window_sec=3.0, hop_sec=1.0):
    """Return a list of recording dicts (tape excluded, empty files dropped)."""
    samples_per_window = int(audio_sr * window_sec)
    hop_samples = int(audio_sr * hop_sec)

    mic_dir = os.path.join(data_dir, "MIC")
    imu_dir = os.path.join(data_dir, "IMU")

    recordings = []
    for audio_path in sorted(glob.glob(os.path.join(mic_dir, "*.wav"))):
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        label = binary_label(base_name)
        if label is None:
            continue
        csv_path = os.path.join(imu_dir, f"{base_name}.csv")
        if not os.path.exists(csv_path):
            continue
        total_chunks = _count_chunks(audio_path, samples_per_window, hop_samples)
        if total_chunks == 0:
            continue
        recordings.append({
            "base_name": base_name,
            "audio_path": audio_path,
            "csv_path": csv_path,
            "label": label,
            "total_chunks": total_chunks,
        })
    return recordings


def records_for(recording, chunk_indices):
    """Expand a recording + chunk indices into per-chunk dataset records."""
    return [{
        "audio_path": recording["audio_path"],
        "csv_path": recording["csv_path"],
        "chunk_idx": idx,
        "label": recording["label"],
    } for idx in chunk_indices]


def _read_audio_chunk(audio_path, chunk_idx, hop_samples, samples_per_window):
    # round() so a fractional chunk_idx (used to densify the single healthy
    # recording with overlapping windows) maps to a valid integer sample offset.
    frame_offset = int(round(chunk_idx * hop_samples))
    with wave.open(audio_path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        wav_file.setpos(frame_offset)
        raw_bytes = wav_file.readframes(samples_per_window)

    audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    audio_np = audio_np.reshape(-1, channels)
    waveform = torch.from_numpy(audio_np.copy()).T  # [channels, T]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.shape[1] < samples_per_window:
        waveform = torch.nn.functional.pad(waveform, (0, samples_per_window - waveform.shape[1]))
    return waveform  # [1, samples_per_window]


def _read_imu_chunk(csv_path, chunk_idx, hop_samples, samples_per_window, has_header=False):
    start_row = int(round(chunk_idx * hop_samples)) + (1 if has_header else 0)
    imu_df = pd.read_csv(csv_path, skiprows=start_row, nrows=samples_per_window, header=None)
    imu_df = imu_df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if imu_df.shape[1] > 6:
        imu_df = imu_df.iloc[:, -6:]
    imu_df = imu_df.fillna(0)

    imu = torch.tensor(imu_df.values, dtype=torch.float32).T  # [6, T]
    # Per-channel standardization (accel and gyro live on different scales).
    imu = (imu - imu.mean(dim=1, keepdim=True)) / (imu.std(dim=1, keepdim=True) + 1e-6)
    if imu.shape[1] < samples_per_window:
        imu = torch.nn.functional.pad(imu, (0, samples_per_window - imu.shape[1]))
    return imu  # [6, samples_per_window]


class BinaryMultimodalDataset(Dataset):
    """Yields (mel_spectrogram, imu, label) for an explicit list of records.

    The audio-only model uses just `mel`; the late-fusion model uses both.
    """

    def __init__(self, records, is_train, audio_sr=16000, imu_sr=416,
                 window_sec=3.0, hop_sec=1.0, imu_has_header=False,
                 n_fft=1024, hop_length=256, n_mels=64):
        self.records = records
        self.is_train = is_train
        self.imu_has_header = imu_has_header

        self.audio_samples = int(audio_sr * window_sec)
        self.imu_samples = int(imu_sr * window_sec)
        self.audio_hop = int(audio_sr * hop_sec)
        self.imu_hop = int(imu_sr * hop_sec)

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=25)

    def __len__(self):
        return len(self.records)

    def label_counts(self):
        counts = [0, 0]
        for record in self.records:
            counts[record["label"]] += 1
        return counts

    def _augment_waveform(self, waveform):
        waveform = waveform * (0.7 + 0.6 * torch.rand(1))          # random gain
        waveform = waveform + 0.01 * torch.randn_like(waveform)    # additive noise
        shift = int(torch.randint(-self.audio_samples // 20,
                                  self.audio_samples // 20 + 1, (1,)))
        return torch.roll(waveform, shifts=shift, dims=1)          # time shift

    def _augment_imu(self, imu):
        imu = imu * (0.9 + 0.2 * torch.rand(imu.shape[0], 1))      # per-channel scale
        imu = imu + 0.05 * torch.randn_like(imu)                   # additive noise
        shift = int(torch.randint(-self.imu_samples // 20,
                                  self.imu_samples // 20 + 1, (1,)))
        return torch.roll(imu, shifts=shift, dims=1)

    def __getitem__(self, idx):
        rec = self.records[idx]

        waveform = _read_audio_chunk(rec["audio_path"], rec["chunk_idx"],
                                     self.audio_hop, self.audio_samples)
        # RMS normalization removes recording loudness as a shortcut feature.
        waveform = waveform / (waveform.pow(2).mean().sqrt() + 1e-8)
        if self.is_train:
            waveform = self._augment_waveform(waveform)

        mel = self.to_db(self.mel(waveform))  # [1, n_mels, T]
        if self.is_train:
            mel = self.time_mask(self.freq_mask(mel))

        imu = _read_imu_chunk(rec["csv_path"], rec["chunk_idx"],
                              self.imu_hop, self.imu_samples, self.imu_has_header)
        if self.is_train:
            imu = self._augment_imu(imu)

        return mel, imu, torch.tensor(rec["label"], dtype=torch.long)
