import os
import glob
import wave
import pandas as pd
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader

# ==========================================
# Class Label Mapping
# ==========================================
def get_class_label(filename):
    if "Healthy" in filename: return 0
    elif "tear_n_hole2" in filename: return 3
    elif "hole1" in filename: return 1
    elif "hole2" in filename: return 2
    elif "fixed_all_tape" in filename: return 4
    else: raise ValueError(f"Could not map {filename}.")

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
        index = []
        mic_dir = os.path.join(self.data_dir, "MIC")
        imu_dir = os.path.join(self.data_dir, "IMU")
        
        audio_files = glob.glob(os.path.join(mic_dir, "*.wav"))
        
        for audio_path in audio_files:
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            csv_path = os.path.join(imu_dir, f"{base_name}.csv")
            
            if not os.path.exists(csv_path): continue
                
            with wave.open(audio_path, 'rb') as wav_file:
                num_audio_frames = wav_file.getnframes()
                
            if num_audio_frames >= self.audio_samples_per_window:
                total_chunks = ((num_audio_frames - self.audio_samples_per_window) // self.audio_hop_samples) + 1
            else:
                total_chunks = 0
                
            label = get_class_label(base_name)

            all_chunks = list(range(total_chunks))
            random.seed(42)
            random.shuffle(all_chunks)
            
            split_point = int(total_chunks * self.split_ratio)
            
            if self.is_train:
                valid_chunks = all_chunks[:split_point]
            else:
                valid_chunks = all_chunks[split_point:]
            
            for chunk_idx in valid_chunks:
                index.append({
                    "audio_path": audio_path,
                    "csv_path": csv_path,
                    "chunk_idx": chunk_idx,
                    "label": label
                })
                
        return index

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