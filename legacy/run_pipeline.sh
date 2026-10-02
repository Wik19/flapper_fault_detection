#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e 

# ==========================================
# 🎛️ MASTER CONTROL PANEL
# ==========================================
WINDOW=3.0
HOP=1.0
SPLIT=0.7
EPOCHS=50
BATCH=32
LR=0.0001
MODE="late" # Options: "early" or "late"
# ==========================================


echo "=================================================="
echo "🚀 STARTING FLAPPER DRONE ML PIPELINE"
echo "=================================================="
echo "Architecture : $MODE Fusion"
echo "Parameters   : Window=${WINDOW}s | Hop=${HOP}s | Split=${SPLIT}"
echo "Training     : Epochs=${EPOCHS} | Batch=${BATCH} | LR=${LR}"
echo "=================================================="

# 1. Run the Training Script
echo -e "\n🔥 [STEP 1/2] Kicking off Training Phase..."
python3 src/train_${MODE}.py \
    --window $WINDOW \
    --hop $HOP \
    --split $SPLIT \
    --epochs $EPOCHS \
    --batch $BATCH \
    --lr $LR

# 2. Run the Evaluation Script
echo -e "\n📊 [STEP 2/2] Generating Confusion Matrix & Metrics..."
python3 src/evaluate.py $MODE \
    --window $WINDOW \
    --hop $HOP \
    --split $SPLIT \
    --batch $BATCH

echo -e "\n✅ PIPELINE COMPLETE! Check the results/ folder for your matrix."