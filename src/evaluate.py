import os
import sys
import torch
import importlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from data_tools.preprocess_early import MultimodalEarlyFusionDataset
from architectures.early_fusion import EarlyFusionCNN
from data_tools.multimodal_dataset import RawMultimodalDataset
from architectures.late_fusion import LateFusionNet

# ==========================================
# 1. Argument Parsing (The Smart Switch)
# ==========================================
parser = argparse.ArgumentParser(description="Evaluate Flapper Fault Detection Models.")
parser.add_argument("mode", choices=["early", "late"], help="Choose the fusion architecture: 'early' or 'late'")

parser.add_argument("--window", type=float, default=3.0, help="Window size in seconds")
parser.add_argument("--hop", type=float, default=1.0, help="Hop size in seconds")
parser.add_argument("--split", type=float, default=0.6, help="Train/Val split ratio")
parser.add_argument("--batch", type=int, default=32, help="Batch size")

args = parser.parse_args()

MODE = args.mode

# ==========================================
# 2. Dynamic Configuration & Setup
# ==========================================
DATA_DIR = "data/raw" 
CSV_HAS_HEADER = False

WINDOW_SEC = args.window
HOP_SEC = args.hop
SPLIT_RATIO = args.split
BATCH_SIZE = args.batch

CLASS_NAMES = [
    "Healthy (0)", 
    "Hole 1 (1)", 
    "Hole 2 (2)", 
    "Tear & Hole 2 (3)", 
    "Tape Fixed (4)"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False 

print(f"\n🚀 Initializing Evaluation for [{MODE.upper()} FUSION] on device: {device}\n" + "="*50)

# Set up paths, visual styles, and classes based on the chosen mode
if MODE == "early":
    MODEL_PATH = "models/early_fusion/best_flapper_model.pth"
    SAVE_FILE = "results/early_fusion/confusion_matrix.png"
    CMAP = 'Blues'
    TITLE = 'Early Fusion CNN: Flapper Fault Confusion Matrix'
    
    # Simple assignment
    DatasetClass = MultimodalEarlyFusionDataset
    ModelClass = EarlyFusionCNN

elif MODE == "late":
    MODEL_PATH = "models/late_fusion/best_late_fusion_model.pth"
    SAVE_FILE = "results/late_fusion/confusion_matrix.png"
    CMAP = 'Purples'
    TITLE = 'Late Fusion CNN: Flapper Fault Confusion Matrix'
    
    # Simple assignment
    DatasetClass = RawMultimodalDataset
    ModelClass = LateFusionNet

# Ensure output directory exists
os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)

# ==========================================
# 3. Load Validation Data
# ==========================================
print("Loading unseen validation dataset...")

val_dataset = DatasetClass(
    data_dir=DATA_DIR, 
    is_train=False, 
    imu_has_header=CSV_HAS_HEADER
)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
print(f"Evaluating on {len(val_dataset)} unseen validation chunks.")

# ==========================================
# 4. Load Trained Model
# ==========================================
model = ModelClass(num_classes=5).to(device)

if not os.path.exists(MODEL_PATH):
    print(f"🚨 Error: Could not find {MODEL_PATH}. Did you run the {MODE} training script?")
    exit(1)

# Load the weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() 

# ==========================================
# 5. Run Inference
# ==========================================
all_preds = []
all_labels = []

print("Running forward passes...")
with torch.no_grad():
    for batch in val_loader:
        
        # ------------------------------------
        # The Switch: Handle data unpacking 
        # based on the chosen architecture
        # ------------------------------------
        if MODE == "early":
            batch_data, batch_labels = batch
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            
        elif MODE == "late":
            audio_data, imu_data, batch_labels = batch
            audio_data = audio_data.to(device)
            imu_data = imu_data.to(device)
            outputs = model(audio_data, imu_data)
        
        # Convert logits to predicted class indices
        _, predicted = torch.max(outputs, 1)
        
        # Store for metrics
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_labels.numpy())

# ==========================================
# 6. Generate Metrics & Confusion Matrix
# ==========================================
print("\n" + "="*50)
print(f"             {MODE.upper()} FUSION REPORT")
print("="*50)

# 1. ADD labels=[0, 1, 2, 3, 4] and zero_division=0 to prevent warnings
print(classification_report(
    all_labels, 
    all_preds, 
    labels=[0, 1, 2, 3, 4], 
    target_names=CLASS_NAMES,
    zero_division=0 
))

# 2. ADD labels=[0, 1, 2, 3, 4] so it always generates a 5x5 grid
cm = confusion_matrix(
    all_labels, 
    all_preds, 
    labels=[0, 1, 2, 3, 4]
)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True,       
    fmt='d',          
    cmap=CMAP,        # Dynamic Color!
    xticklabels=CLASS_NAMES, 
    yticklabels=CLASS_NAMES,
    linewidths=1,
    linecolor='black'
)

plt.title(TITLE, fontsize=14, pad=15)
plt.ylabel('Actual True Fault', fontsize=12, fontweight='bold')
plt.xlabel('Model Predicted Fault', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(SAVE_FILE, dpi=300)
print(f"\n📊 Confusion matrix successfully saved to: {os.path.abspath(SAVE_FILE)}")