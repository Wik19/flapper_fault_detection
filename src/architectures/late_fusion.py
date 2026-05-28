import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class LateFusionNet(nn.Module):
    def __init__(self, audio_sr=16000, num_classes=5, dropout_rate=0.4, window_sec=3.0):
        super(LateFusionNet, self).__init__()
        
        # ==========================================
        # 1. AUDIO BRANCH (2D CNN)
        # ==========================================
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_sr, n_fft=1024, hop_length=256, n_mels=64
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        self.audio_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), 
        )

        # ==========================================
        # 2. IMU BRANCH (1D CNN)
        # ==========================================
        self.imu_conv = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4), 
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4), 
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2), 
        )

        # ==========================================
        # 3. DYNAMIC FUSION HEAD SIZING
        # ==========================================
        with torch.no_grad():
            # --- USE THE DYNAMIC window_sec HERE ---
            dummy_audio = torch.zeros(1, 1, int(audio_sr * window_sec)) 
            dummy_imu = torch.zeros(1, 6, int(416 * window_sec))        
            
            a_mel = self.amplitude_to_db(self.mel_transform(dummy_audio))
            a_feat = self.audio_conv(a_mel)
            self.audio_flat_size = a_feat.numel()
            
            i_feat = self.imu_conv(dummy_imu)
            self.imu_flat_size = i_feat.numel()
            
            self.total_fusion_size = self.audio_flat_size + self.imu_flat_size

        # ==========================================
        # 4. FUSION HEAD (Dense Layers)
        # ==========================================
        self.fusion_head = nn.Sequential(
            nn.Linear(self.total_fusion_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            
            nn.Linear(128, num_classes) 
        )

    def forward(self, audio_x, imu_x):
        mel = self.mel_transform(audio_x)
        mel_db = self.amplitude_to_db(mel)
        audio_features = self.audio_conv(mel_db)
        audio_features = audio_features.view(audio_features.size(0), -1) 
        
        imu_features = self.imu_conv(imu_x)
        imu_features = imu_features.view(imu_features.size(0), -1) 
        
        fused_features = torch.cat((audio_features, imu_features), dim=1)
        
        logits = self.fusion_head(fused_features)
        return logits