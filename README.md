## 📊 Data Specifications

The pipeline expects paired files with identical base names in the `data/raw/IMU` and `data/raw/MIC` directories.

**Classes (Derived from filenames):**

1. `Healthy1` (Healthy)
2. `hole1_...` (One hole)
3. `hole2_...` (Two holes)
4. `tear_n_hole2_...` (Tear and two holes)
5. `fixed_all_tape` (Patched with tape)

## 🧠 Architectures

1. **Early Fusion CNN (`src/architectures/early_fusion.py`):** Extracts Mel-spectrograms from audio and STFT spectrograms from IMU data. Resizes and stacks them as a multi-channel image before passing through a 2D CNN.
2. **Late Fusion Net (`src/architectures/late_fusion.py`):** Processes Audio via a 2D CNN (Mel-spectrograms) and IMU via a 1D CNN (raw time-series). The extracted feature vectors are concatenated at the fully-connected classification head.

## 🚀 Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt

```

### 2. Run the Full Pipeline

The easiest way to train and evaluate a model is using the master bash script. You can edit the hyperparameters (Window size, batch size, learning rate, and mode) directly at the top of `run_pipeline.sh`.

```bash
bash run_pipeline.sh

```

### 3. Manual Execution

You can also run the scripts individually.

**Training:**

```bash
# For Early Fusion
python src/train_early.py --window 3.0 --hop 1.0 --split 0.7 --epochs 50 --batch 32 --lr 0.0001

# For Late Fusion
python src/train_late.py --window 3.0 --hop 1.0 --split 0.7 --epochs 50 --batch 32 --lr 0.0001

```

**Evaluation:**
Generates a classification report and a confusion matrix saved to the `results/` folder.

```bash
python src/evaluate.py early --window 3.0 --hop 1.0
# OR
python src/evaluate.py late --window 3.0 --hop 1.0

```

**Inference on New Data (work in progress):**

```bash
python src/test_on_new.py

```