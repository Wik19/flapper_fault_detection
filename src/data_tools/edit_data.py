import pandas as pd
import numpy as np
from scipy.io import wavfile
import os

def crop_sensor_data(wav_in, csv_in, wav_out, csv_out, start_time_sec, end_time_sec, imu_sf=416.0):
    print(f"Cropping data from {start_time_sec}s to {end_time_sec}s...")

    # --- 1. Process Audio Data ---
    sample_rate, audio_data = wavfile.read(wav_in)
    
    # Calculate exact indices
    audio_start_idx = int(start_time_sec * sample_rate)
    audio_end_idx = int(end_time_sec * sample_rate)
    
    # Safety check to prevent slicing beyond the file length
    if audio_end_idx > len(audio_data):
        print("Warning: Requested end time exceeds audio duration. Truncating to end of audio.")
        audio_end_idx = len(audio_data)
        
    # Slice and Save
    audio_cropped = audio_data[audio_start_idx:audio_end_idx]
    wavfile.write(wav_out, sample_rate, audio_cropped)
    print(f"Saved cropped audio: {len(audio_cropped)} samples ({len(audio_cropped)/sample_rate:.2f} seconds)")

    # --- 2. Process IMU Data ---
    imu_df = pd.read_csv(csv_in)
    
    # Calculate exact indices based on the 416 Hz assumption
    imu_start_idx = int(start_time_sec * imu_sf)
    imu_end_idx = int(end_time_sec * imu_sf)
    
    # Safety check
    if imu_end_idx > len(imu_df):
        print("Warning: Requested end time exceeds IMU duration. Truncating to end of IMU data.")
        imu_end_idx = len(imu_df)
        
    # Slice the dataframe by row index
    imu_cropped = imu_df.iloc[imu_start_idx:imu_end_idx].copy()
    
    # Optional but highly recommended: Recalculate the Rx_Time_Sec column
    # so the new file starts at 0.0 seconds and ignores the hardware glitches
    new_time_array = np.linspace(0, len(imu_cropped) / imu_sf, num=len(imu_cropped))
    imu_cropped['Rx_Time_Sec'] = new_time_array
    
    # Save to new CSV
    imu_cropped.to_csv(csv_out, index=False)
    print(f"Saved cropped IMU: {len(imu_cropped)} rows ({len(imu_cropped)/imu_sf:.2f} seconds)")
    print("Cropping complete!\n")

if __name__ == "__main__":
    WAV_FILE = "Data/raw/MIC/tear_n_hole2_full.wav"
    CSV_FILE = "Data/raw/IMU/tear_n_hole2_full.csv"
    
    WAV_OUT = "Data/raw/MIC/tear_n_hole2_full_cropped.wav"
    CSV_OUT = "Data/raw/IMU/tear_n_hole2_full_cropped.csv"
    
    START_SEC = 1.5
    END_SEC = 57.0
    
    try:
        crop_sensor_data(WAV_FILE, CSV_FILE, WAV_OUT, CSV_OUT, START_SEC, END_SEC)
    except FileNotFoundError as e:
        print(f"Error: {e}")