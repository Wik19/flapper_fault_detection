**Objective**

Process, synchronize, and extract features from paired multimodal raw data (Inertial and Acoustic) to train a Neural Network for flapper wing fault detection.

**Raw Data Specifications**

- **Base Directory:** `~/Documents/projects/flapper_fault_detection/Data/raw`
    
- **Structure:** Paired files sharing the same base name across two subdirectories:
    
    - `/IMU/` -> `.csv` files
        
    - `/MIC/` -> `.wav` files
        
- **Inertial Data (IMU):**
    
    - Sampling Frequency: 416Hz
        
    - Columns: `Packet_ID`, `Rx_Time_Sec`, `Acc_X`, `Acc_Y`, `Acc_Z`, `Gyro_X`, `Gyro_Y`, `Gyro_Z`
        
- **Acoustic Data (MIC):**
    
    - Sampling Frequency: 16kHz
        
- **Classes / Labels (derived from filenames):**
    
    1. Healthy (`Healthy1`)
        
    2. One hole (`hole1_...`)
        
    3. Two holes (`hole2_...`)
        
    4. A tear and two holes (`tear_n_hole2_...`)
        
    5. Patched - covered tear and holes with tape (`fixed_all_tape`)
        
- _Note: Completely ignore any files named `zero.csv` or `zero.wav`._
    

**Required Processing Steps (Instructions for Cursor)**

1. **File Iteration & Pairing:** Write a script that iterates through the `IMU` directory. For every `.csv` file, locate the exact matching `.wav` file in the `MIC` directory. Explicitly skip the `zero` files.
    
2. **Label Extraction:** Dynamically assign one of the 5 classes (labels) to the current pair of files based on string matching in the filename.
    
3. **Data Loading:** Read the `.csv` using `pandas` and the `.wav` using `librosa` or `scipy`. Handle any missing values in the CSV.
    
4. **Synchronization:** Since the IMU is at 416Hz and the Mic is at 16kHz, use `Rx_Time_Sec` to establish a common timeline and align the acoustic arrays with the inertial arrays.
    
5. **Feature Extraction (IMU):** Normalize the accelerometer and gyroscope data (e.g., standard scaling).
    
6. **Feature Extraction (Audio):** Convert the raw audio waveforms into a format suitable for Neural Networks (e.g., extract Mel-spectrograms or MFCCs).
    
7. **Windowing:** Segment the continuous synchronized streams into fixed-length, overlapping time windows (epochs). Ensure each window is paired with its correct integer label.
    
8. **Export:** Save the final input features ($X$) and labels ($y$) into a `Data/processed/` folder as `.npy` (NumPy) files for easy loading into PyTorch or TensorFlow.