import os
import glob
import wave
import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import Dataset

def get_class_label(filename):
    """Maps filenames to exact integer classes."""
    if "Healthy" in filename: return 0
    elif "tear_n_hole2" in filename: return 3
    elif "hole1" in filename: return 1
    elif "hole2" in filename: return 2
    elif "fixed_all_tape" in filename: return 4
    else: raise ValueError(f"Could not map {filename}.")

class RawMultimodalDataset(Dataset):
    def __init__(self, data_dir, is_train=True, audio_sr=16000, imu_sr=416, 
                 window_sec=3.0, hop_sec=1.0, imu_has_header=False, split_ratio=0.6):
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
            
            # --- STRATIFIED / RANDOM SPLIT LOGIC ---
            all_chunks = list(range(total_chunks))
            
            random.seed(42) # Stop data leakage!
            random.shuffle(all_chunks)
            
            split_point = int(total_chunks * self.split_ratio)
            
            if self.is_train:
                valid_chunks = all_chunks[:split_point]
            else:
                valid_chunks = all_chunks[split_point:]
            # ---------------------------------------
            
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

    def _get_raw_audio(self, audio_path, chunk_idx):
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
            
        if waveform.shape[1] < self.audio_samples_per_window:
            pad_amount = self.audio_samples_per_window - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            
        return waveform

    def _get_raw_imu(self, csv_path, chunk_idx):
        start_row = chunk_idx * self.imu_hop_samples
        if self.imu_has_header: start_row += 1 
            
        imu_df = pd.read_csv(
            csv_path, skiprows=start_row, nrows=self.imu_samples_per_window, header=None 
        )
        
        imu_df = imu_df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
        if imu_df.shape[1] > 6: imu_df = imu_df.iloc[:, -6:]
        imu_df = imu_df.fillna(0)
        
        imu_tensor = torch.tensor(imu_df.values, dtype=torch.float32).T 
        
        if imu_tensor.shape[1] < self.imu_samples_per_window:
            pad_amount = self.imu_samples_per_window - imu_tensor.shape[1]
            imu_tensor = torch.nn.functional.pad(imu_tensor, (0, pad_amount))
            
        return imu_tensor

    def __getitem__(self, idx):
        info = self.chunk_index[idx]
        audio_waveform = self._get_raw_audio(info["audio_path"], info["chunk_idx"]) 
        imu_tensor = self._get_raw_imu(info["csv_path"], info["chunk_idx"])      
        label = torch.tensor(info["label"], dtype=torch.long)
        return audio_waveform, imu_tensor, label