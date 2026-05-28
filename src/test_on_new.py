import torch
import numpy as np
import importlib
import os
import wave
import pandas as pd
import torch.nn.functional as F

preprocess = importlib.import_module("data_tools.preprocess")
from architectures.early_fusion import EarlyFusionCNN

class SingleFileInference(preprocess.MultimodalEarlyFusionDataset):
    def __init__(self, audio_path, csv_path, **kwargs):
        self.audio_path = audio_path
        self.csv_path = csv_path
        super().__init__(data_dir="./", **kwargs)

    def _get_num_chunks(self):
        with wave.open(self.audio_path, 'rb') as wav:
            return ((wav.getnframes() - self.audio_samples_per_window) // self.audio_hop_samples) + 1

    def get_chunk(self, chunk_idx):
        audio_spec = self._process_audio(self.audio_path, chunk_idx)
        imu_specs = self._process_imu(self.csv_path, chunk_idx)
        
        target_size = (audio_spec.shape[1], audio_spec.shape[2])
        imu_specs_resized = F.interpolate(imu_specs.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
        return torch.cat([audio_spec, imu_specs_resized], dim=0)

# ==========================================
# RUN INFERENCE
# ==========================================
def run_inference(wav_file, csv_file, model_path="models/early_fusion/best_flapper_model_extended.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = False
    
    # 1. Load Model
    model = EarlyFusionCNN(input_shape=(7, 64, 188)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Setup Processor
    processor = SingleFileInference(wav_file, csv_file)
    num_chunks = processor._get_num_chunks()
    
    class_names = ["Healthy", "Hole 1", "Hole 2", "Tear & Hole 2", "Tape Fixed"]
    
    print(f"\n🚀 Running Inference on: {os.path.basename(wav_file)}")
    print("-" * 50)
    
    with torch.no_grad():
        for i in range(num_chunks):
            tensor = processor.get_chunk(i).unsqueeze(0).to(device)
            logits = model(tensor)
            pred = torch.argmax(logits, dim=1).item()
            
            print(f"Time {i}s: Predicted Fault -> {class_names[pred]}")

if __name__ == "__main__":
    NEW_WAV = "data/test/test_1.wav"
    NEW_CSV = "data/test/test_1.csv"
    
    run_inference(NEW_WAV, NEW_CSV)