import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def plot_sensor_data(wav_filepath, csv_filepath):
    # 1. Load Audio Data (16 kHz)
    sample_rate, audio_data = wavfile.read(wav_filepath)
    
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]
        
    # Generate perfect time array for audio
    audio_time = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))

    # 2. Load IMU Data
    imu_df = pd.read_csv(csv_filepath)
    
    # --- THE FIX ---
    # Ignore 'Rx_Time_Sec' and generate a perfect time array based on 416 Hz
    imu_sf = 416.0 
    imu_time = np.linspace(0, len(imu_df) / imu_sf, num=len(imu_df))
    
    # Extract structural columns
    acc_x, acc_y, acc_z = imu_df['Acc_X'], imu_df['Acc_Y'], imu_df['Acc_Z']
    gyro_x, gyro_y, gyro_z = imu_df['Gyro_X'], imu_df['Gyro_Y'], imu_df['Gyro_Z']

    # 3. Create the Plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Microphone & IMU Data Sync', fontsize=16)

    # --- Subplot 1: Audio Waveform ---
    ax1.plot(audio_time, audio_data, color='purple', alpha=0.7)
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Audio Waveform ({sample_rate} Hz)')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Subplot 2: Accelerometer ---
    ax2.plot(imu_time, acc_x, label='Acc_X', color='red', alpha=0.8, linewidth=1)
    ax2.plot(imu_time, acc_y, label='Acc_Y', color='green', alpha=0.8, linewidth=1)
    ax2.plot(imu_time, acc_z, label='Acc_Z', color='blue', alpha=0.8, linewidth=1)
    ax2.set_ylabel('Acceleration')
    ax2.set_title('Accelerometer (416 Hz)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- Subplot 3: Gyroscope ---
    ax3.plot(imu_time, gyro_x, label='Gyro_X', color='red', alpha=0.8, linewidth=1)
    ax3.plot(imu_time, gyro_y, label='Gyro_Y', color='green', alpha=0.8, linewidth=1)
    ax3.plot(imu_time, gyro_z, label='Gyro_Z', color='blue', alpha=0.8, linewidth=1)
    ax3.set_ylabel('Angular Vel')
    ax3.set_xlabel('Time (Seconds)')
    ax3.set_title('Gyroscope (416 Hz)')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.6)

    # Clean up layout and render
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Optional: Force the x-axis limits to match the audio's exact duration
    plt.xlim(0, max(audio_time[-1], imu_time[-1]))
    
    plt.show()

if __name__ == "__main__":
    WAV_FILE = "Data/raw/MIC/tear_n_hole2_full.wav"
    CSV_FILE = "Data/raw/IMU/tear_n_hole2_full.csv"
    
    plot_sensor_data(WAV_FILE, CSV_FILE)