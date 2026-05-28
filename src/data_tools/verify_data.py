import os
import numpy as np
import matplotlib.pyplot as plt

PROCESSED_DIR = "/home/marco/Documents/projects/flapper_fault_detection/Data/processed"
OUTPUT_IMG_PATH = "/home/marco/Documents/projects/flapper_fault_detection/verification.png"

def main():
    print("====================================================")
    print("    VERIFYING PREPROCESSED DATA AND GENERATING PLOT  ")
    print("====================================================")
    
    # 1. Load the preprocessed arrays
    X_imu_path = os.path.join(PROCESSED_DIR, "X_imu.npy")
    X_audio_path = os.path.join(PROCESSED_DIR, "X_audio.npy")
    y_path = os.path.join(PROCESSED_DIR, "y.npy")
    
    if not (os.path.exists(X_imu_path) and os.path.exists(X_audio_path) and os.path.exists(y_path)):
        print("Error: One or more preprocessed files do not exist! Please run 01_preprocess.py first.")
        return
        
    X_imu = np.load(X_imu_path)
    X_audio = np.load(X_audio_path)
    y = np.load(y_path)
    
    # 2. Print shape information
    print("Successfully loaded preprocessed arrays:")
    print(f"  - X_imu shape   : {X_imu.shape} (N_windows, sequence_len, n_channels)")
    print(f"  - X_audio shape : {X_audio.shape} (N_windows, n_mels, time_steps)")
    print(f"  - y shape       : {y.shape} (N_windows,)")
    
    print("\nExtracting first window details:")
    first_imu = X_imu[0]
    first_audio = X_audio[0]
    first_label = y[0]
    
    class_names = {
        0: "Healthy",
        1: "One Hole",
        2: "Two Holes",
        3: "Tear & 2 Holes",
        4: "Patched"
    }
    label_name = class_names.get(first_label, f"Unknown ({first_label})")
    print(f"  - First Window Label: {first_label} ({label_name})")
    
    # 3. Create a premium high-quality dual-panel plot using matplotlib
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), dpi=150)
    
    # Color palette
    colors = {
        'Acc_X': '#1f77b4', 'Acc_Y': '#aec7e8', 'Acc_Z': '#ff7f0e',
        'Gyro_X': '#2ca02c', 'Gyro_Y': '#98df8a', 'Gyro_Z': '#d62728'
    }
    
    # Define timelines (1.0 second duration)
    time_imu = np.linspace(0.0, 1.0, len(first_imu))
    
    # --- Plot 1: IMU Window ---
    ax_imu = axes[0]
    channels = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
    for idx, name in enumerate(channels):
        ax_imu.plot(time_imu, first_imu[:, idx], label=name, color=colors[name], linewidth=1.5)
        
    ax_imu.set_title(f"First Window IMU Signal - Class {first_label}: {label_name} (Normalized)", 
                     fontsize=12, fontweight='bold', pad=10)
    ax_imu.set_xlabel("Time (seconds)", fontsize=10)
    ax_imu.set_ylabel("Normalized Magnitude", fontsize=10)
    ax_imu.grid(True, linestyle='--', alpha=0.6)
    ax_imu.legend(loc='upper right', ncol=3, framealpha=0.9, facecolor='#ffffff')
    ax_imu.set_xlim(0.0, 1.0)
    
    # --- Plot 2: Audio Spectrogram ---
    ax_aud = axes[1]
    # Plot using standard imshow for maximum stability and speed
    img = ax_aud.imshow(
        first_audio, 
        aspect='auto', 
        origin='lower', 
        cmap='magma', 
        extent=[0.0, 1.0, 0, first_audio.shape[0]]
    )
    
    ax_aud.set_title(f"First Window Audio Log-Mel-Spectrogram - Class {first_label}: {label_name}", 
                     fontsize=12, fontweight='bold', pad=10)
    ax_aud.set_xlabel("Time (seconds)", fontsize=10)
    ax_aud.set_ylabel("Mel Frequency Bin", fontsize=10)
    
    # Add beautiful colorbar
    cbar = fig.colorbar(img, ax=ax_aud, pad=0.02)
    cbar.set_label("Magnitude (dB)", fontsize=10)
    
    # Adjust spacing and premium borders
    plt.tight_layout()
    
    # 4. Save the plot as verification.png
    plt.savefig(OUTPUT_IMG_PATH, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Premium plot generated and saved successfully to:\n  {OUTPUT_IMG_PATH}")
    print("====================================================")

if __name__ == "__main__":
    main()
