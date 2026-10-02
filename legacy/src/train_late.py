import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_tools.multimodal_dataset import RawMultimodalDataset
from architectures.late_fusion import LateFusionNet

# ==========================================
# 1. Command Line Arguments
# ==========================================
parser = argparse.ArgumentParser(description="Train Late Fusion Model")
parser.add_argument("--window", type=float, default=3.0, help="Window size in seconds")
parser.add_argument("--hop", type=float, default=1.0, help="Hop size in seconds")
parser.add_argument("--split", type=float, default=0.6, help="Train/Val split ratio")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--batch", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
args = parser.parse_args()

WINDOW_SEC = args.window
HOP_SEC = args.hop
SPLIT_RATIO = args.split
EPOCHS = args.epochs
BATCH_SIZE = args.batch
LEARNING_RATE = args.lr

DATA_DIR = "data/raw" 
CSV_HAS_HEADER = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False 
print(f"🚀 Training Late Fusion Model on: {device}")

# ==========================================
# 2. Data Loading
# ==========================================
def get_dataloaders():
    print(f"Loading datasets with {WINDOW_SEC}s windows and {HOP_SEC}s hops...")
    
    train_dataset = RawMultimodalDataset(
        data_dir=DATA_DIR, is_train=True, imu_has_header=CSV_HAS_HEADER,
        window_sec=WINDOW_SEC, hop_sec=HOP_SEC, split_ratio=SPLIT_RATIO
    )
    val_dataset = RawMultimodalDataset(
        data_dir=DATA_DIR, is_train=False, imu_has_header=CSV_HAS_HEADER,
        window_sec=WINDOW_SEC, hop_sec=HOP_SEC, split_ratio=SPLIT_RATIO
    )
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Dataset is empty. Check your paths!")
        
    print(f"Dataset split: {len(train_dataset)} Train chunks | {len(val_dataset)} Val chunks")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=24)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=24)

    return train_loader, val_loader, train_dataset


def make_criterion(train_dataset):
    # Inverse-frequency class weights so the imbalanced classes are not drowned
    # out (e.g. hole1 has far more recordings than Healthy / fixed_all_tape).
    counts = torch.tensor(train_dataset.label_counts(), dtype=torch.float32)
    weights = counts.sum() / (len(counts) * counts.clamp(min=1))
    weights[counts == 0] = 0.0
    print(f"Class counts: {counts.tolist()} | weights: {[round(w, 3) for w in weights.tolist()]}")
    return nn.CrossEntropyLoss(weight=weights.to(device))

# ==========================================
# 3. The Training Loop (with TQDM)
# ==========================================
def train():
    train_loader, val_loader, train_dataset = get_dataloaders()

    # --- Pass window_sec to ensure dynamic flattening aligns with bash inputs ---
    model = LateFusionNet(num_classes=5, window_sec=WINDOW_SEC).to(device)

    criterion = make_criterion(train_dataset)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    save_path = "models/late_fusion/best_late_fusion_model.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print("\nStarting Late Fusion Training...\n" + "="*40)
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{EPOCHS} [Train]", leave=False, dynamic_ncols=True, colour='green')
        
        for audio_data, imu_data, labels in train_pbar:
            audio_data, imu_data, labels = audio_data.to(device), imu_data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(audio_data, imu_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            current_loss = train_loss / train_total
            current_acc = (train_correct / train_total) * 100
            train_pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
            
        avg_train_loss = train_loss / train_total
        train_acc = (train_correct / train_total) * 100
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:03d}/{EPOCHS}  [Val] ", leave=False, dynamic_ncols=True, colour='blue')
        
        with torch.no_grad():
            for audio_data, imu_data, labels in val_pbar:
                audio_data, imu_data, labels = audio_data.to(device), imu_data.to(device), labels.to(device)
                
                outputs = model(audio_data, imu_data)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                current_v_loss = val_loss / val_total
                current_v_acc = (val_correct / val_total) * 100
                val_pbar.set_postfix({'loss': f'{current_v_loss:.4f}', 'acc': f'{current_v_acc:.2f}%'})
                
        avg_val_loss = val_loss / val_total
        val_acc = (val_correct / val_total) * 100
        
        # --- CHECKPOINTING ---
        saved_flag = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            saved_flag = "💾 [Saved]"
            
        print(f"Epoch [{epoch+1:03d}/{EPOCHS}] {time.time() - start_time:.1f}s | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}% {saved_flag}")

if __name__ == "__main__":
    train()