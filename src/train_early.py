import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm # <--- NEW IMPORT

from data_tools.preprocess_early import MultimodalEarlyFusionDataset
from architectures.early_fusion import EarlyFusionCNN

# ==========================================
# 1. Command Line Arguments
# ==========================================
parser = argparse.ArgumentParser(description="Train Early Fusion Model")
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
print(f"🚀 Training Early Fusion Model on: {device}")

# ==========================================
# 2. Data Loading
# ==========================================
def get_dataloaders():
    print(f"Loading datasets with {WINDOW_SEC}s windows and {HOP_SEC}s hops...")
    
    train_dataset = MultimodalEarlyFusionDataset(
        data_dir=DATA_DIR, is_train=True, imu_has_header=CSV_HAS_HEADER,
        window_sec=WINDOW_SEC, hop_sec=HOP_SEC, split_ratio=SPLIT_RATIO
    )
    val_dataset = MultimodalEarlyFusionDataset(
        data_dir=DATA_DIR, is_train=False, imu_has_header=CSV_HAS_HEADER,
        window_sec=WINDOW_SEC, hop_sec=HOP_SEC, split_ratio=SPLIT_RATIO
    )
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Dataset is empty. Check your paths!")
        
    print(f"Dataset split: {len(train_dataset)} Train chunks | {len(val_dataset)} Val chunks")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=24)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=24)
    
    return train_loader, val_loader, train_dataset

# ==========================================
# 3. The Training Loop (Now with TQDM)
# ==========================================
def train():
    train_loader, val_loader, train_dataset = get_dataloaders()
    
    sample_tensor, _ = train_dataset[0]
    dynamic_shape = sample_tensor.shape
    print(f"⚙️ Auto-detected input shape: {dynamic_shape}")
    
    model = EarlyFusionCNN(input_shape=dynamic_shape, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    save_path = "models/early_fusion/best_flapper_model.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print("\nStarting Early Fusion Training...\n" + "="*40)
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # -----------------------------------------
        # PHASE 1: TRAINING
        # -----------------------------------------
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        # Wrap the loader in tqdm
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{EPOCHS} [Train]", leave=False, dynamic_ncols=True, colour='green')
        
        for batch_data, labels in train_pbar:
            batch_data, labels = batch_data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Live update the progress bar
            current_loss = train_loss / train_total
            current_acc = (train_correct / train_total) * 100
            train_pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
            
        avg_train_loss = train_loss / train_total
        train_acc = (train_correct / train_total) * 100
        
        # -----------------------------------------
        # PHASE 2: VALIDATION
        # -----------------------------------------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:03d}/{EPOCHS}  [Val] ", leave=False, dynamic_ncols=True, colour='blue')
        
        with torch.no_grad():
            for batch_data, labels in val_pbar:
                batch_data, labels = batch_data.to(device), labels.to(device)
                
                outputs = model(batch_data)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Live update the progress bar
                current_v_loss = val_loss / val_total
                current_v_acc = (val_correct / val_total) * 100
                val_pbar.set_postfix({'loss': f'{current_v_loss:.4f}', 'acc': f'{current_v_acc:.2f}%'})
                
        avg_val_loss = val_loss / val_total
        val_acc = (val_correct / val_total) * 100
        
        # -----------------------------------------
        # CHECKPOINTING & SUMMARY
        # -----------------------------------------
        saved_flag = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            saved_flag = "💾 [Saved]"
            
        # Clean terminal output after the progress bars clear (leave=False)
        print(f"Epoch [{epoch+1:03d}/{EPOCHS}] {time.time() - start_time:.1f}s | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}% {saved_flag}")

if __name__ == "__main__":
    train()