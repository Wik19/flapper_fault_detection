"""Lean binary (HEALTHY vs DAMAGED) models.

Deliberately small and heavily regularized: with ~250 training windows the
5-class nets (~1M params) just memorized. These use global average pooling
instead of a large flatten+dense head, keeping each model well under 100k
params so there is far less capacity to overfit recording-specific noise.
"""
import torch
import torch.nn as nn


def _conv2d_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


def _conv1d_block(in_ch, out_ch, pool=4):
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(pool),
    )


class AudioCNN(nn.Module):
    """Mel-spectrogram [B,1,n_mels,T] -> 64-d feature vector (global avg pool)."""

    def __init__(self, feat_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            _conv2d_block(1, 16),
            _conv2d_block(16, 32),
            _conv2d_block(32, feat_dim),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = feat_dim

    def forward(self, mel):
        x = self.pool(self.net(mel))
        return x.flatten(1)  # [B, feat_dim]


class ImuCNN(nn.Module):
    """Raw IMU [B,6,T] -> 64-d feature vector (global avg pool)."""

    def __init__(self, feat_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            _conv1d_block(6, 16),
            _conv1d_block(16, 32),
            _conv1d_block(32, feat_dim, pool=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.feat_dim = feat_dim

    def forward(self, imu):
        x = self.pool(self.net(imu))
        return x.flatten(1)  # [B, feat_dim]


class BinaryAudioNet(nn.Module):
    """Audio-only baseline."""

    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        self.audio = AudioCNN()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.audio.feat_dim, num_classes),
        )

    def forward(self, mel, imu=None):
        return self.head(self.audio(mel))


class BinaryLateFusionNet(nn.Module):
    """Audio + IMU late fusion."""

    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        self.audio = AudioCNN()
        self.imu = ImuCNN()
        fused = self.audio.feat_dim + self.imu.feat_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, mel, imu):
        return self.head(torch.cat([self.audio(mel), self.imu(imu)], dim=1))


# Registry used by the cross-validation runner.
MODELS = {
    "audio": BinaryAudioNet,
    "late": BinaryLateFusionNet,
}
