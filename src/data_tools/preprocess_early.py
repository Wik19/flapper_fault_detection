import wave
import pandas as pd
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from data_tools.splitting import build_chunk_index, get_class_label

# ==========================================
# PyTorch Dataset Implementation
# ==========================================
class MultimodalEarlyFusionDataset(Dataset):
    def __init__(self, data_dir, is_train=True, audio_sr=16000, imu_sr=416, 
                 window_sec=3.0, hop_sec=1.0, imu_has_header=False, split_ratio=0.7):
        """
        Args:
            is_train (bool): If True, uses the first 70% of each file and applies Augmentation.
                             If False, uses the last 30% of each file (No Augmentation).
            split_ratio (float): Percentage of the recording used for training.
        """
        self.data_dir = data_dir
        self.is_train = is_train
        self.audio_sr = audio_sr
        self.imu_sr = imu_sr
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.imu_has_header = imu_has_header
        self.split_ratio = split_ratio
        
        self.audio_samples_per_window = int(audio_sr * window_sec)
        self.imu_samples_per_window = int(imu_sr * window_sec)
        self.audio_hop_samples = int(audio_sr * hop_sec)
        self.imu_hop_samples = int(imu_sr * hop_sec)
        
        # Audio Transformers
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.audio_sr, n_fft=1024, hop_length=256, n_mels=64
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Data Augmentation (SpecAugment)
        # Randomly masks up to 8 frequency bins and 20 time frames
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=8)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=20)

        # Build the index based on the split
        self.chunk_index = self._build_index()

    def _build_index(self):
        # Group-aware, leakage-free split (see data_tools/splitting.py).
        return build_chunk_index(
            data_dir=self.data_dir,
            is_train=self.is_train,
            audio_samples_per_window=self.audio_samples_per_window,
            audio_hop_samples=self.audio_hop_samples,
            window_sec=self.window_sec,
            hop_sec=self.hop_sec,
            split_ratio=self.split_ratio,
            verbose=self.is_train,
        )

    def label_counts(self, num_classes=5):
        counts = [0] * num_classes
        for record in self.chunk_index:
            counts[record["label"]] += 1
        return counts

    def __len__(self):
        return len(self.chunk_index)

    def _process_audio(self, audio_path, chunk_idx):
        frame_offset = chunk_idx * self.audio_hop_samples
        with wave.open(audio_path, 'rb') as wav_file:
            channels = wav_file.getnchannels()
            wav_file.setpos(frame_offset)
            raw_bytes = wav_file.readframes(self.audio_samples_per_window)
            
        audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio_np = audio_np.reshape(-1, channels)
        waveform = torch.from_numpy(audio_np).T 
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        mel_spec = self.mel_transform(waveform)
        return self.amplitude_to_db(mel_spec)

    def _process_imu(self, csv_path, chunk_idx):
        start_row = chunk_idx * self.imu_hop_samples
        if self.imu_has_header: start_row += 1 
            
        imu_df = pd.read_csv(
            csv_path, skiprows=start_row, nrows=self.imu_samples_per_window, header=None 
        )
        
        # Data Sanitization
        imu_df = imu_df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
        if imu_df.shape[1] > 6: imu_df = imu_df.iloc[:, -6:]
        imu_df = imu_df.fillna(0)
        
        imu_tensor = torch.tensor(imu_df.values, dtype=torch.float32).T

        # Per-channel standardization before the STFT so accel and gyro
        # (very different scales) contribute comparably to the spectrogram.
        mean = imu_tensor.mean(dim=1, keepdim=True)
        std = imu_tensor.std(dim=1, keepdim=True)
        imu_tensor = (imu_tensor - mean) / (std + 1e-6)

        window = torch.hann_window(64)
        
        imu_stft = torch.stft(
            imu_tensor, n_fft=64, hop_length=16, window=window, return_complex=True, normalized=True
        )
        
        return self.amplitude_to_db(torch.abs(imu_stft))

    def __getitem__(self, idx):
        info = self.chunk_index[idx]
        
        audio_spec = self._process_audio(info["audio_path"], info["chunk_idx"]) 
        imu_specs = self._process_imu(info["csv_path"], info["chunk_idx"])      
        
        target_size = (audio_spec.shape[1], audio_spec.shape[2]) 
        imu_specs_resized = F.interpolate(
            imu_specs.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False
        ).squeeze(0) 
        
        # Stacking [7, 64, 188]
        fused_tensor = torch.cat([audio_spec, imu_specs_resized], dim=0) 
        
        # --- APPLY DATA AUGMENTATION (ONLY IN TRAINING) ---
        if self.is_train:
            fused_tensor = self.freq_masking(fused_tensor)
            fused_tensor = self.time_masking(fused_tensor)
        # --------------------------------------------------
        
        return fused_tensor, torch.tensor(info["label"], dtype=torch.long)

if __name__ == "__main__":
    # Quick Test
    DATA_DIR = "./data/raw" 
    train_dataset = MultimodalEarlyFusionDataset(DATA_DIR, is_train=True)
    val_dataset = MultimodalEarlyFusionDataset(DATA_DIR, is_train=False)
    print(f"Temporal Split: {len(train_dataset)} Train Chunks | {len(val_dataset)} Test Chunks")