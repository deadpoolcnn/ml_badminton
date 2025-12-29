from pyexpat import features
import pandas as pd
import numpy as np
from scipy import stats
from scipy import signal
from scipy.signal import savgol_filter,find_peaks, welch
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

"""
------ Data Preprocessing Functions ------
"""
# Function to resample and clean timestamp data to ensure consistent sampling frequency
def resample_clean_timestamp(df, target_hz=10):
    """
    Resample and clean timestamp data to ensure consistent sampling rate
    
    Parameters:
    - df: Input DataFrame with sensor data and timestamps
    - target_hz: Target sampling frequency in Hz (default: 10Hz)
    
    Returns:
    - df: Resampled DataFrame with clean, consistent timestamps
    """
    df = df.copy()
    
    # Calculate sampling intervals for resampling
    sampling_interval_ms = int(1000 / target_hz)  # milliseconds
    sampling_interval_s = 1.0 / target_hz         # seconds
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['Timestamp (s)'], unit='s')
    
    # Check time span before resampling
    time_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
    
    # If time span is unreasonable, use relative timestamps
    if time_span > 3600:  # More than 1 hour - likely timestamp issue
        # Use target_hz parameter to calculate time intervals and recreate timestamps
        df['timestamp'] = pd.to_datetime(df.index * sampling_interval_s, unit='s')
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    # Normal processing for reasonable time spans
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.drop_duplicates(subset=['timestamp'], keep='first')
    
    # Only resample if time span is small
    if time_span < 300:  # Less than 5 minutes
        df = df.set_index('timestamp')
        
        # Use target_hz parameter to dynamically calculate resampling interval
        resample_rule = f'{sampling_interval_ms}ms'
        df_resampled = df.resample(resample_rule).first()
        
        # Don't fill all gaps - only interpolate small ones
        df_resampled = df_resampled.interpolate(method='time', limit=3)
        df_resampled = df_resampled.dropna()  # Remove unfilled gaps
        df_resampled = df_resampled.reset_index()
        return df_resampled
    return df
# Function to remove sensor bias
def remove_sensor_bias(df):
    """
    Remove sensor bias from accelerometer and gyroscope data
    
    This function removes the mean bias from sensor readings by calculating
    the average offset and subtracting it from each measurement. It handles
    both standard and alternative column name formats for encoding issues.
    
    Args:
        df (pandas.DataFrame): DataFrame containing sensor measurements
    
    Returns:
        tuple: (df, actual_cols)
            - df (pandas.DataFrame): DataFrame with bias-corrected sensor data
            - actual_cols (list): List of actual sensor column names found in data
    """
    sensor_cols = [
        'Accel_X (g)', 'Accel_Y (g)', 'Accel_Z (g)',
        'Gyro_X (°/s)', 'Gyro_Y (°/s)', 'Gyro_Z (°/s)'
    ]
    actual_cols = []
    for col in sensor_cols:
        if col in df.columns:
            actual_cols.append(col)
        else:
            alt_col = col.replace('°', '�')  # Handle encoding issues
            if alt_col in df.columns:
                actual_cols.append(alt_col)
                df = df.rename(columns={alt_col: col})
    # Remove mean bias from gyroscope and accelerometer data
    for col in actual_cols:
        if 'Accel' in col or 'Gyro' in col:
            bias = df[col].mean()
            df[col] = df[col] - bias
            # print(f"Removed bias from {col}: {bias:.4f}")
    return df, actual_cols
# gravity compensation
def gravity_compensation(df, accel_cols, sampling_rate=10):
    """
    Improved gravity compensation method using high-pass filtering
    
    This function removes gravity and slow drift components from acceleration data
    using a Butterworth high-pass filter. It preserves the dynamic motion components
    that are relevant for badminton shot classification.
    
    Args:
        df (pandas.DataFrame): DataFrame containing acceleration data
        accel_cols (list): List of acceleration column names to process
        sampling_rate (int): Sampling rate of the data in Hz (default: 10)
    
    Returns:
        pandas.DataFrame: DataFrame with gravity-compensated acceleration data
    """
    df_compensated = df.copy()
    
    # Design high-pass filter - remove gravity and slow drift below 0.3Hz
    cutoff_freq = 0.3  # Hz - badminton motions typically >0.5Hz
    nyquist = sampling_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    
    # 2nd order Butterworth high-pass filter
    b, a = signal.butter(2, normalized_cutoff, btype='high')
    
    for col in accel_cols:
        if col in df.columns:
            # Zero-phase filtering - prevents signal delay
            df_compensated[col] = signal.filtfilt(b, a, df_compensated[col])
    return df_compensated
# Noise reduction using Savitzky-Golay filter
def noise_reduction(df, sensor_cols):
    """
    Noise reduction using Savitzky-Golay filter while preserving peaks
    
    This function applies a Savitzky-Golay filter to smooth the sensor data
    while maintaining the important peak characteristics needed for badminton
    shot detection and classification.
    
    Args:
        df (pandas.DataFrame): DataFrame containing sensor data
        sensor_cols (list): List of sensor column names to apply filtering to
    
    Returns:
        pandas.DataFrame: DataFrame with noise-reduced sensor data
    """
    # Savitzky-Golay filter - preserves peaks while reducing noise
    window_size = 5  # Small window to maintain peak characteristics
    poly_order = 2   # Polynomial order for fitting
    for col in sensor_cols:
        df[col] = savgol_filter(df[col], window_size, poly_order)
    return df
# Main preprocessing function
def preprocess_data(csv_path):
    """
    Main preprocessing pipeline for badminton sensor data
    
    This function executes the complete preprocessing pipeline including:
    1. Data loading and timestamp resampling
    2. Sensor bias removal
    3. Gravity compensation for acceleration data
    4. Noise reduction using Savitzky-Golay filtering
    
    Args:
        csv_path (str): Path to the CSV file containing raw sensor data
    
    Returns:
        tuple: (df, sensor_cols)
            - df (pandas.DataFrame): Fully preprocessed sensor data ready for feature extraction
            - sensor_cols (list): List of sensor column names that were processed
    """
    df = pd.read_csv(csv_path, encoding='latin1')
    df = resample_clean_timestamp(df, target_hz=10)  
    df, sensor_cols = remove_sensor_bias(df)
    accel_cols = [col for col in sensor_cols if 'Accel' in col]
    df = gravity_compensation(df, accel_cols) 
    df = noise_reduction(df, sensor_cols)
    # output_path = csv_path.replace('.csv', '_preprocessed.csv')
    # df.to_csv(output_path, index=False)
    return df, sensor_cols
# Preprocess all activity data
def preprocess_all_data():
    activities = ['clear', 'smash', 'drive', 'lift']
    data_dir = Path('./new_dataset')
    all_preprocessed = {}
    all_sensor_cols = set()
    for activity in activities:
        csv_path = data_dir / f'{activity}_10_kyle.csv'
        preprocessed_df, sensor_cols = preprocess_data(csv_path)
        preprocessed_df['shot_type'] = activity
        all_preprocessed[activity] = preprocessed_df
        all_sensor_cols.update(sensor_cols)
    return all_preprocessed, all_sensor_cols

"""
-------- Comprehensive Feature Extraction Functions ------
"""
def extract_comprehensive_features(window_data, sensor_cols, sampling_rate=10):
    """
    Extract comprehensive time-domain, frequency-domain, and directional features
    
    This function extracts a comprehensive 72-dimensional feature vector from
    sensor data windows, including statistical measures, frequency characteristics,
    and correlation features for badminton shot classification.
    
    Parameters:
        window_data (pandas.DataFrame): Windowed sensor data for feature extraction
        sensor_cols (list): List of sensor column names to process
        sampling_rate (int): Sampling rate of the data in Hz (default: 10)
    
    Returns:
        tuple: (features, feature_names)
            - features (list): 72-dimensional feature vector
            - feature_names (list): Corresponding feature name labels
    """
    features = []
    feature_names = []

     # Separate sensor data by type
    accel_cols = [col for col in sensor_cols if 'Accel' in col]
    gyro_cols = [col for col in sensor_cols if 'Gyro' in col]

    # ========== Time-Domain Features ==========
    # 1. Basic statistical features for each axis (Mean, Std, MAD, Min, Max, Range, IQR)
    for sensor_group, group_name in [(accel_cols, 'accel'), (gyro_cols, 'gyro')]:
        for col in sensor_group:
            if col in window_data.columns:
                data = window_data[col].values
                axis_name = col.split('_')[1].lower()  # X, Y, Z
                
                # Mean, Std
                features.extend([np.mean(data), np.std(data)])
                feature_names.extend([f'{group_name}_{axis_name}_mean', f'{group_name}_{axis_name}_std'])
                
                # Median Absolute Deviation
                mad = stats.median_abs_deviation(data, scale='normal')
                features.append(mad)
                feature_names.append(f'{group_name}_{axis_name}_mad')
                
                # Min, Max, Range, IQR
                q1, q3 = np.percentile(data, [25, 75])
                features.extend([np.min(data), np.max(data), np.max(data) - np.min(data), q3 - q1])
                feature_names.extend([
                    f'{group_name}_{axis_name}_min', f'{group_name}_{axis_name}_max', 
                    f'{group_name}_{axis_name}_range', f'{group_name}_{axis_name}_iqr'
                ])
    # 2. Composite magnitude features |a| and |g|
    if len(accel_cols) >= 3:
        accel_mag = np.sqrt(
            window_data[accel_cols[0]]**2 + 
            window_data[accel_cols[1]]**2 + 
            window_data[accel_cols[2]]**2
        )
        features.extend([np.mean(accel_mag), np.std(accel_mag)])
        feature_names.extend(['accel_mag_mean', 'accel_mag_std'])
        
        # Peak count and amplitude for acceleration magnitude
        accel_threshold = np.mean(accel_mag) + 0.5 * np.std(accel_mag)
        peaks, peak_props = find_peaks(accel_mag, height=accel_threshold)
        peak_count = len(peaks)
        peak_amplitude = np.mean(peak_props['peak_heights']) if len(peaks) > 0 else 0
        features.extend([peak_count, peak_amplitude])
        feature_names.extend(['accel_peak_count', 'accel_peak_amplitude'])
    if len(gyro_cols) >= 3:
        gyro_mag = np.sqrt(
            window_data[gyro_cols[0]]**2 + 
            window_data[gyro_cols[1]]**2 + 
            window_data[gyro_cols[2]]**2
        )
        features.extend([np.mean(gyro_mag), np.std(gyro_mag)])
        feature_names.extend(['gyro_mag_mean', 'gyro_mag_std'])
        
        # Peak count and amplitude for gyroscope
        gyro_threshold = np.mean(gyro_mag) + 0.5 * np.std(gyro_mag)
        peaks, peak_props = find_peaks(gyro_mag, height=gyro_threshold)
        peak_count = len(peaks)
        peak_amplitude = np.mean(peak_props['peak_heights']) if len(peaks) > 0 else 0
        features.extend([peak_count, peak_amplitude])
        feature_names.extend(['gyro_peak_count', 'gyro_peak_amplitude'])
    # 3. Jerk approximation (first-order difference of acceleration)
    for i, col in enumerate(accel_cols):
        if col in window_data.columns:
            data = window_data[col].values
            if len(data) > 1:
                jerk = np.diff(data)  # First-order difference approximating jerk
                axis_name = col.split('_')[1].lower()
                features.extend([np.mean(np.abs(jerk)), np.std(jerk)])
                feature_names.extend([f'jerk_{axis_name}_mean', f'jerk_{axis_name}_std'])
            else:
                # Handle case when data is too short
                axis_name = col.split('_')[1].lower()
                features.extend([0, 0])
                feature_names.extend([f'jerk_{axis_name}_mean', f'jerk_{axis_name}_std'])

    # ========== Frequency-Domain Features ==========
    # 4. Dominant frequency and power (acceleration and gyroscope magnitudes)
    if len(accel_cols) >= 3 and len(accel_mag) > 1:
        try:
            nperseg = min(len(accel_mag), 16)  # Adapt to short signals
            freqs, psd = welch(accel_mag, fs=sampling_rate, nperseg=nperseg)
            if len(psd) > 0:
                dominant_freq_idx = np.argmax(psd)
                dominant_freq = freqs[dominant_freq_idx]
                dominant_power = psd[dominant_freq_idx]
                
                # Frequency band energy (0.2-1.5 Hz and 1.5-4 Hz)
                low_band_mask = (freqs >= 0.2) & (freqs <= 1.5)
                mid_band_mask = (freqs > 1.5) & (freqs <= 4.0)
                low_band_energy = np.sum(psd[low_band_mask]) if np.any(low_band_mask) else 0
                mid_band_energy = np.sum(psd[mid_band_mask]) if np.any(mid_band_mask) else 0
                
                features.extend([dominant_freq, dominant_power, low_band_energy, mid_band_energy])
                feature_names.extend(['accel_dominant_freq', 'accel_dominant_power', 
                                    'accel_low_band_energy', 'accel_mid_band_energy'])
            else:
                features.extend([0, 0, 0, 0])
                feature_names.extend(['accel_dominant_freq', 'accel_dominant_power', 
                                    'accel_low_band_energy', 'accel_mid_band_energy'])
        except:
            features.extend([0, 0, 0, 0])
            feature_names.extend(['accel_dominant_freq', 'accel_dominant_power', 
                                'accel_low_band_energy', 'accel_mid_band_energy'])   
    if len(gyro_cols) >= 3 and len(gyro_mag) > 1:
        try:
            nperseg = min(len(gyro_mag), 16)
            freqs, psd = welch(gyro_mag, fs=sampling_rate, nperseg=nperseg)
            if len(psd) > 0:
                dominant_freq_idx = np.argmax(psd)
                dominant_freq = freqs[dominant_freq_idx]
                dominant_power = psd[dominant_freq_idx]
                
                # Frequency band energy
                low_band_mask = (freqs >= 0.2) & (freqs <= 1.5)
                mid_band_mask = (freqs > 1.5) & (freqs <= 4.0)
                low_band_energy = np.sum(psd[low_band_mask]) if np.any(low_band_mask) else 0
                mid_band_energy = np.sum(psd[mid_band_mask]) if np.any(mid_band_mask) else 0
                
                features.extend([dominant_freq, dominant_power, low_band_energy, mid_band_energy])
                feature_names.extend(['gyro_dominant_freq', 'gyro_dominant_power', 
                                    'gyro_low_band_energy', 'gyro_mid_band_energy'])
            else:
                features.extend([0, 0, 0, 0])
                feature_names.extend(['gyro_dominant_freq', 'gyro_dominant_power', 
                                    'gyro_low_band_energy', 'gyro_mid_band_energy'])
        except:
            features.extend([0, 0, 0, 0])
            feature_names.extend(['gyro_dominant_freq', 'gyro_dominant_power', 
                                'gyro_low_band_energy', 'gyro_mid_band_energy'])

    # ========== Directional and Motion Features ==========  
    # 5. Short-window gyroscope integration magnitude (racket rotation proxy)
    if len(gyro_cols) >= 3 and len(gyro_mag) > 2:
        window_size = max(3, min(5, len(gyro_mag) // 2))  # Short window
        gyro_integrated = []
        for i in range(0, len(gyro_mag) - window_size + 1, window_size):
            segment = gyro_mag[i:i + window_size]
            integrated = np.trapezoid(np.abs(segment)) / sampling_rate
            gyro_integrated.append(integrated)
        
        if gyro_integrated:
            features.extend([np.mean(gyro_integrated), np.std(gyro_integrated)])
            feature_names.extend(['gyro_rotation_mean', 'gyro_rotation_std'])
        else:
            features.extend([0, 0])
            feature_names.extend(['gyro_rotation_mean', 'gyro_rotation_std'])
    else:
        features.extend([0, 0])
        feature_names.extend(['gyro_rotation_mean', 'gyro_rotation_std'])  
    # 6. Inter-axis correlations
    # Acceleration inter-axis correlations
    if len(accel_cols) >= 3:
        try:
            accel_data = window_data[accel_cols].values
            if accel_data.shape[0] > 1:  # Ensure sufficient data points
                accel_corr = np.corrcoef(accel_data.T)
                # Extract upper triangular correlation coefficients
                accel_corr_xy = accel_corr[0, 1] if not np.isnan(accel_corr[0, 1]) else 0
                accel_corr_xz = accel_corr[0, 2] if not np.isnan(accel_corr[0, 2]) else 0
                accel_corr_yz = accel_corr[1, 2] if not np.isnan(accel_corr[1, 2]) else 0
                features.extend([accel_corr_xy, accel_corr_xz, accel_corr_yz])
                feature_names.extend(['accel_corr_xy', 'accel_corr_xz', 'accel_corr_yz'])
            else:
                features.extend([0, 0, 0])
                feature_names.extend(['accel_corr_xy', 'accel_corr_xz', 'accel_corr_yz'])
        except:
            features.extend([0, 0, 0])
            feature_names.extend(['accel_corr_xy', 'accel_corr_xz', 'accel_corr_yz'])
    # Gyroscope inter-axis correlations
    if len(gyro_cols) >= 3:
        try:
            gyro_data = window_data[gyro_cols].values
            if gyro_data.shape[0] > 1:
                gyro_corr = np.corrcoef(gyro_data.T)
                gyro_corr_xy = gyro_corr[0, 1] if not np.isnan(gyro_corr[0, 1]) else 0
                gyro_corr_xz = gyro_corr[0, 2] if not np.isnan(gyro_corr[0, 2]) else 0
                gyro_corr_yz = gyro_corr[1, 2] if not np.isnan(gyro_corr[1, 2]) else 0
                features.extend([gyro_corr_xy, gyro_corr_xz, gyro_corr_yz])
                feature_names.extend(['gyro_corr_xy', 'gyro_corr_xz', 'gyro_corr_yz'])
            else:
                features.extend([0, 0, 0])
                feature_names.extend(['gyro_corr_xy', 'gyro_corr_xz', 'gyro_corr_yz'])
        except:
            features.extend([0, 0, 0])
            feature_names.extend(['gyro_corr_xy', 'gyro_corr_xz', 'gyro_corr_yz'])
    
    return np.array(features), feature_names


"""
------ shot threshold settings ------
"""
def add_fallback_features(df, sensor_cols, shot_segments, 
                               existing_features, target_shots, 
                               min_shot_duration, sampling_rate=10):
    features = existing_features.copy()
    
    # Find unused data intervals
    used_ranges = set()
    for start, end in shot_segments:
        used_ranges.update(range(start, end))
    
    unused_data = [i for i in range(len(df)) if i not in used_ranges]
    
    if unused_data and len(features) < target_shots:
        remaining_needed = target_shots - len(features)
        
        # Method 1: Select by motion intensity ranking
        if 'accel_magnitude' in df.columns:
            window_size = 20  # Fixed window size
            candidate_windows = []       
            for i in range(0, len(unused_data) - window_size + 1, window_size//2):
                window_indices = unused_data[i:i+window_size]
                if len(window_indices) >= min_shot_duration:
                    window_data = df.iloc[window_indices]
                    intensity_score = (window_data['accel_magnitude'].max() + 
                                     window_data.get('gyro_magnitude', pd.Series([0])).max())
                    candidate_windows.append((window_indices, intensity_score))           
            
            # Select windows with highest intensity
            candidate_windows.sort(key=lambda x: x[1], reverse=True)       
            for window_indices, _ in candidate_windows[:remaining_needed]:
                window = df.iloc[window_indices]
                window_features, _ = extract_comprehensive_features(window, sensor_cols, sampling_rate)
                features.append(window_features)
        
        # Method 2: Simple segmentation of remaining data
        if len(features) < target_shots:
            remaining_needed = target_shots - len(features)
            unused_segments = np.array_split(unused_data, remaining_needed)   
            for segment in unused_segments:
                if len(segment) >= min_shot_duration:
                    window = df.iloc[segment]
                    window_features, _ = extract_comprehensive_features(window, sensor_cols, sampling_rate)
                    features.append(window_features)
    
    return features
def extract_shot_features(df, sensor_cols, shot_segments, target_shots, min_shot_duration, sampling_rate=10):
    features = []
    feature_names = None
    
    # Extract features from detected segments
    for i, (start, end) in enumerate(shot_segments):
        if len(features) >= target_shots:
            break        
        shot_window = df.iloc[start:end]
        window_features, names = extract_comprehensive_features(shot_window, sensor_cols, sampling_rate)
        if feature_names is None:
            feature_names = names
        features.append(window_features)
    
    # If no segments detected, generate feature names first
    if feature_names is None:
        sample_window = df.iloc[:min(20, len(df))]
        _, feature_names = extract_comprehensive_features(sample_window, sensor_cols, sampling_rate) 
    
    # Fallback method: if detected segments are insufficient - no shot_label usage
    if len(features) < target_shots:
        features = add_fallback_features(
            df, sensor_cols, shot_segments, 
            features, target_shots, min_shot_duration, sampling_rate
        )   
    
    return features, feature_names

def threshold_windowing(df, sensor_cols, target_shots=30, min_shot_duration=8, max_shot_duration=60):
    """
    Threshold-based windowing method for feature extraction
    
    This function implements a unified threshold-based approach to detect badminton
    shots and extract features. It uses consistent thresholds across all shot types
    to avoid label leakage and ensures fair model evaluation.
    
    Args:
        df (pandas.DataFrame): Preprocessed sensor data
        sensor_cols (list): List of sensor column names
        target_shots (int): Target number of shots to extract (default: 30)
        min_shot_duration (int): Minimum duration for a valid shot in samples (default: 8)
        max_shot_duration (int): Maximum duration for a valid shot in samples (default: 60)
    
    Returns:
        tuple: (features, feature_names)
            - features (list): List of feature vectors for detected shots
            - feature_names (list): Corresponding feature names for the vectors
    """
    # 1. Unified parameters - independent of shot_label to prevent data leakage
    UNIFIED_ACCEL_THRESHOLD = 1.0  # Same threshold for all shot types
    UNIFIED_GYRO_THRESHOLD = 40.0  # Same threshold for all shot types
    # 2. Calculate sensor magnitudes
    accel_cols = [col for col in sensor_cols if 'Accel' in col]
    gyro_cols = [col for col in sensor_cols if 'Gyro' in col]
    df['accel_magnitude'] = np.sqrt(
        df[accel_cols[0]]**2 + df[accel_cols[1]]**2 + df[accel_cols[2]]**2
    )
    df['gyro_magnitude'] = np.sqrt(
        df[gyro_cols[0]]**2 + df[gyro_cols[1]]**2 + df[gyro_cols[2]]**2
    )
    # 3. Unified triggering strategy - label-independent
    accel_trigger = df['accel_magnitude'] > UNIFIED_ACCEL_THRESHOLD
    gyro_trigger = df['gyro_magnitude'] > UNIFIED_GYRO_THRESHOLD
    trigger_condition = accel_trigger | gyro_trigger
    # 4. Unified segment detection
    shot_segments = []
    in_shot = False
    shot_start = None 
    for i, is_active in enumerate(trigger_condition):
        if is_active and not in_shot:
            shot_start = i
            in_shot = True
        elif not is_active and in_shot:
            shot_end = i
            duration = shot_end - shot_start       
            # Unified validation conditions - no shot_label usage
            if min_shot_duration <= duration <= max_shot_duration:
                segment = df.iloc[shot_start:shot_end]
                peak_accel = segment['accel_magnitude'].max()
                peak_gyro = segment['gyro_magnitude'].max()               
                # Simple unified validation criteria
                if (peak_accel > UNIFIED_ACCEL_THRESHOLD * 0.7 or 
                    peak_gyro > UNIFIED_GYRO_THRESHOLD * 0.7):
                    shot_segments.append((shot_start, shot_end))
            
            in_shot = False
    # Handle the last segment
    if in_shot and shot_start is not None:
        shot_end = len(df)
        duration = shot_end - shot_start
        if min_shot_duration <= duration <= max_shot_duration:
            segment = df.iloc[shot_start:shot_end]
            peak_accel = segment['accel_magnitude'].max()
            peak_gyro = segment['gyro_magnitude'].max()
            if (peak_accel > UNIFIED_ACCEL_THRESHOLD * 0.7 or 
                peak_gyro > UNIFIED_GYRO_THRESHOLD * 0.7):
                shot_segments.append((shot_start, shot_end))
    # 5. Extract features - no label involvement
    features = []
    feature_names = None    
    for start, end in shot_segments:
        if len(features) >= target_shots:
            break
        shot_window = df.iloc[start:end]
        window_features, names = extract_comprehensive_features(
            shot_window, sensor_cols, sampling_rate=10
        )
        if feature_names is None:
            feature_names = names
        features.append(window_features)
    # 6. Fallback method: if detected segments are insufficient, use sliding window
    if len(features) < target_shots:
        remaining_needed = target_shots - len(features)
        window_size = 20  # Fixed window size
        step_size = window_size // 2     
        for i in range(0, len(df) - window_size + 1, step_size):
            if len(features) >= target_shots:
                break
            window = df.iloc[i:i+window_size]
            # Simple intensity filtering, no label usage
            if (window['accel_magnitude'].max() > UNIFIED_ACCEL_THRESHOLD * 0.5 or
                window['gyro_magnitude'].max() > UNIFIED_GYRO_THRESHOLD * 0.5):
                window_features, names = extract_comprehensive_features(
                    window, sensor_cols, sampling_rate=10
                )
                if feature_names is None:
                    feature_names = names
                features.append(window_features)
    return np.array(features), shot_segments, feature_names

"""
------ save all features ------
"""
def save_threshold_windowing(all_preprocessed, all_sensor_cols, target_shots):
    activities = ['clear', 'smash', 'drive', 'lift']
    all_features = {}
    all_labels = {}
    all_feature_names = None
    for activity in activities:
        # Do not pass shot_label to prevent data leakage
        features, shot_segments, feature_names = threshold_windowing(
            all_preprocessed[activity], all_sensor_cols, target_shots
        )
        # Add labels only after feature extraction
        labels = np.array([activity] * len(features))      
        all_features[activity] = features
        all_labels[activity] = labels
        if all_feature_names is None:
            all_feature_names = feature_names
    
    return all_features, all_labels, all_feature_names

def combine_features_data(all_features, all_labels, all_feature_names):
    activities = ['clear', 'smash', 'drive', 'lift']
    combined_data = []
    total_samples = 0
    activity_counts = {}
    for activity in activities:
        if activity in all_features and len(all_features[activity]) > 0:
            # Create DataFrame
            feature_df = pd.DataFrame(all_features[activity], columns=all_feature_names)
            feature_df['shot_type'] = activity
            
            combined_data.append(feature_df)
            activity_counts[activity] = len(feature_df)
            total_samples += len(feature_df)
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        output_file = 'A3_features_kyle.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"✓ Combined dataset saved: {len(combined_df)} total samples")
        return combined_df, total_samples, activity_counts
    else:
        print("⚠️ No data to combine")
        return pd.DataFrame(), 0, {}
if __name__ == "__main__":
    all_preprocessed, all_sensor_cols = preprocess_all_data()
    all_features, all_labels, all_feature_names = save_threshold_windowing(all_preprocessed, all_sensor_cols, target_shots=10)
    # Combine all features and labels into a single DataFrame
    combined_df, total_samples, activity_counts = combine_features_data(all_features, all_labels, all_feature_names)
    plt.figure(figsize=(8,5))
    sns.boxplot(data=combined_df, x='shot_type', y='jerk_y (g)_mean')
    # plt.title("Jerk Y (g) Mean by Shot Type")
    # plt.savefig("feature_box_jerkymean.png")
    plt.show()