from pyexpat import features
import pandas as pd
import numpy as np
from scipy import stats
from scipy import signal
from scipy.signal import savgol_filter,find_peaks, welch
from pathlib import Path

"""
------ preprocess data ------
"""
# Function to resample and clean timestamp data
def resample_clean_timestamp(df, target_hz=10):
    df = df.copy()
    
    # 计算采样间隔
    sampling_interval_ms = int(1000 / target_hz)  # 毫秒
    sampling_interval_s = 1.0 / target_hz         # 秒
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['Timestamp (s)'], unit='s')
    
    # Check time span before resampling
    time_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
    
    # If time span is unreasonable, use relative timestamps
    if time_span > 3600:  # More than 1 hour - likely timestamp issue
        # 使用target_hz参数计算时间间隔
        df['timestamp'] = pd.to_datetime(df.index * sampling_interval_s, unit='s')
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    # Normal processing for reasonable time spans
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.drop_duplicates(subset=['timestamp'], keep='first')
    
    # Only resample if time span is small
    if time_span < 300:  # Less than 5 minutes
        df = df.set_index('timestamp')
        
        # 使用target_hz参数动态计算重采样间隔
        resample_rule = f'{sampling_interval_ms}ms'
        df_resampled = df.resample(resample_rule).first()
        
        # Don't fill all gaps - only interpolate small ones
        df_resampled = df_resampled.interpolate(method='time', limit=3)
        df_resampled = df_resampled.dropna()  # Remove unfilled gaps
        df_resampled = df_resampled.reset_index()
        return df_resampled
    else:
        return df
# Function to remove sensor bias
def remove_sensor_bias(df):
    sensor_cols = [
        'Accel_X (g)', 'Accel_Y (g)', 'Accel_Z (g)',
        'Gyro_X (°/s)', 'Gyro_Y (°/s)', 'Gyro_Z (°/s)'
    ]
    actual_cols = []
    for col in sensor_cols:
        if col in df.columns:
            actual_cols.append(col)
        else:
            # 尝试其他编码格式
            alt_col = col.replace('°', '�')  # 处理编码问题
            if alt_col in df.columns:
                actual_cols.append(alt_col)
                df = df.rename(columns={alt_col: col})
    # 陀螺仪和加速度计：去除均值偏差
    for col in actual_cols:
        if 'Accel' in col or 'Gyro' in col:
            bias = df[col].mean()
            df[col] = df[col] - bias
            # print(f"Removed bias from {col}: {bias:.4f}")
    return df, actual_cols
# gravity compensation
def gravity_compensation(df, accel_cols, sampling_rate=10):
    """
    改进的重力补偿方法
    """
    df_compensated = df.copy()
    
    # 设计高通滤波器 - 移除0.3Hz以下的重力和缓慢漂移
    cutoff_freq = 0.3  # Hz - 羽毛球动作通常>0.5Hz
    nyquist = sampling_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    
    # 2阶Butterworth高通滤波器
    b, a = signal.butter(2, normalized_cutoff, btype='high')
    
    for col in accel_cols:
        if col in df.columns:
            # 零相位滤波 - 避免信号延迟
            df_compensated[col] = signal.filtfilt(b, a, df_compensated[col])
    
    return df_compensated
# noise reduction using Savitzky-Golay filter
def noise_reduction(df, sensor_cols):
    # Savitzky-Golay滤波器 - 保持峰值
    window_size = 5  # 较小窗口保持峰值
    poly_order = 2     
    for col in sensor_cols:
        df[col] = savgol_filter(df[col], window_size, poly_order)
    return df
# Main preprocessing function
def preprocess_data(csv_path):
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
    data_dir = Path('./dataset')
    all_preprocessed = {}
    all_sensor_cols = set()
    for activity in activities:
        csv_path = data_dir / f'{activity}_x30.csv'
        preprocessed_df, sensor_cols = preprocess_data(csv_path)
        preprocessed_df['shot_type'] = activity
        all_preprocessed[activity] = preprocessed_df
        all_sensor_cols.update(sensor_cols)
        output_path = data_dir / f'{activity}_x30_preprocessed.csv'
        preprocessed_df.to_csv(output_path, index=False)
    return all_preprocessed, all_sensor_cols

"""
-------- comprehensive feature extraction ------
"""
def extract_comprehensive_features(window_data, sensor_cols, sampling_rate=10):
    """
    提取全面的时域、频域和方向变化特征
    
    Parameters:
    - window_data: 窗口数据
    - sensor_cols: 传感器列名
    - sampling_rate: 采样率 (Hz)
    
    Returns:
    - features: 特征向量
    - feature_names: 特征名称列表
    """
    features = []
    feature_names = []

     # 分离传感器数据
    accel_cols = [col for col in sensor_cols if 'Accel' in col]
    gyro_cols = [col for col in sensor_cols if 'Gyro' in col]

    # ========== 时域特征 ==========
    # 1. 每轴的基本统计特征 (Mean, Std, MAD, Min, Max, Range, IQR)
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
    # 2. 合成幅值特征 |a| 和 |g|
    if len(accel_cols) >= 3:
        accel_mag = np.sqrt(
            window_data[accel_cols[0]]**2 + 
            window_data[accel_cols[1]]**2 + 
            window_data[accel_cols[2]]**2
        )
        features.extend([np.mean(accel_mag), np.std(accel_mag)])
        feature_names.extend(['accel_mag_mean', 'accel_mag_std'])
        
        # Peak count and amplitude for acceleration
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
    #3. Jerk近似 (加速度的一阶差分)
    for i, col in enumerate(accel_cols):
        if col in window_data.columns:
            data = window_data[col].values
            if len(data) > 1:
                jerk = np.diff(data)  # 一阶差分
                axis_name = col.split('_')[1].lower()
                features.extend([np.mean(np.abs(jerk)), np.std(jerk)])
                feature_names.extend([f'jerk_{axis_name}_mean', f'jerk_{axis_name}_std'])
            else:
                # 处理数据太短的情况
                axis_name = col.split('_')[1].lower()
                features.extend([0, 0])
                feature_names.extend([f'jerk_{axis_name}_mean', f'jerk_{axis_name}_std'])

    # ========== 频域特征 ==========
    # 4. 主要频率和功率 (加速度和陀螺仪幅值)
    if len(accel_cols) >= 3 and len(accel_mag) > 1:
        try:
            nperseg = min(len(accel_mag), 16)  # 适应短信号
            freqs, psd = welch(accel_mag, fs=sampling_rate, nperseg=nperseg)
            if len(psd) > 0:
                dominant_freq_idx = np.argmax(psd)
                dominant_freq = freqs[dominant_freq_idx]
                dominant_power = psd[dominant_freq_idx]
                
                # 频带能量 (0.2-1.5 Hz 和 1.5-4 Hz)
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
                
                # 频带能量
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

    # ========== 方向和变化特征 ==========  
    # 5. 短窗口陀螺仪积分幅值 (球拍旋转代理)
    if len(gyro_cols) >= 3 and len(gyro_mag) > 2:
        window_size = max(3, min(5, len(gyro_mag) // 2))  # 短窗口
        gyro_integrated = []
        for i in range(0, len(gyro_mag) - window_size + 1, window_size):
            segment = gyro_mag[i:i + window_size]
            integrated = np.trapz(np.abs(segment)) / sampling_rate  # 积分
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
    # 6. 轴间相关性
    # 加速度轴间相关性
    if len(accel_cols) >= 3:
        try:
            accel_data = window_data[accel_cols].values
            if accel_data.shape[0] > 1:  # 确保有足够的数据点
                accel_corr = np.corrcoef(accel_data.T)
                # 取上三角矩阵的相关系数
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
    # 陀螺仪轴间相关性
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
def get_shot_thresholds(shot_label):  # 使用了标签信息
    """
    根据动作类型返回合适的阈值
    """
    thresholds = {
        'clear': {'accel': 1.8, 'gyro': 70.0},   # Clear需要高加速度和旋转
        'smash': {'accel': 2.2, 'gyro': 80.0},   # Smash最强烈
        'drive': {'accel': 1.5, 'gyro': 60.0},   # Drive中等强度
        'lift': {'accel': 1.2, 'gyro': 40.0}     # Lift相对温和
    }
    return thresholds.get(shot_label, {'accel': 1.5, 'gyro': 60.0})
def add_fallback_features(df, sensor_cols, shot_segments, shot_label, 
                         existing_features, existing_labels, target_shots, 
                         min_shot_duration, sampling_rate=10):
    """
    备用特征提取方法
    """
    features = existing_features.copy()
    labels = existing_labels.copy()
    
    # 找到未使用的数据区间
    used_ranges = set()
    for start, end in shot_segments:
        used_ranges.update(range(start, end))
    
    unused_data = [i for i in range(len(df)) if i not in used_ranges]
    
    if unused_data and len(features) < target_shots:
        remaining_needed = target_shots - len(features)
        # 方法1: 按运动强度排序选择
        if 'accel_magnitude' in df.columns:
            window_size = 20  # 固定窗口大小
            candidate_windows = []       
            for i in range(0, len(unused_data) - window_size + 1, window_size//2):
                window_indices = unused_data[i:i+window_size]
                if len(window_indices) >= min_shot_duration:
                    window_data = df.iloc[window_indices]
                    intensity_score = (window_data['accel_magnitude'].max() + 
                                     window_data.get('gyro_magnitude', pd.Series([0])).max())
                    candidate_windows.append((window_indices, intensity_score))           
            # 选择强度最高的窗口
            candidate_windows.sort(key=lambda x: x[1], reverse=True)       
            for window_indices, _ in candidate_windows[:remaining_needed]:
                window = df.iloc[window_indices]
                # 使用全面特征提取（72种特征）
                window_features, _ = extract_comprehensive_features(window, sensor_cols, sampling_rate)
                features.append(window_features)
                labels.append(shot_label)
        
        # 方法2: 简单分割剩余数据
        if len(features) < target_shots:
            remaining_needed = target_shots - len(features)
            unused_segments = np.array_split(unused_data, remaining_needed)   
            for segment in unused_segments:
                if len(segment) >= min_shot_duration:
                    window = df.iloc[segment]
                    # 使用全面特征提取(72中特征)
                    window_features, _ = extract_comprehensive_features(window, sensor_cols, sampling_rate)
                    features.append(window_features)
                    labels.append(shot_label)
    return features, labels
def extract_shot_features(df, sensor_cols, shot_segments, shot_label, target_shots, min_shot_duration, sampling_rate=10):
    """
    从动作片段中提取特征
    """
    features, labels = [], []
    feature_names = None
    # 从检测到的片段提取特征
    for i, (start, end) in enumerate(shot_segments):
        if len(features) >= target_shots:
            break        
        shot_window = df.iloc[start:end]
        window_features, names = extract_comprehensive_features(shot_window, sensor_cols, sampling_rate=10)
        if feature_names is None:
            feature_names = names
        features.append(window_features)
        labels.append(shot_label)
    # 如果没有检测到片段，先生成特征名称
    if feature_names is None:
        # 使用前20行数据作为样本来生成特征名称
        sample_window = df.iloc[:min(20, len(df))]
        _, feature_names = extract_comprehensive_features(sample_window, sensor_cols, sampling_rate) 
    # 备用方法：如果检测到的片段不够
    if len(features) < target_shots:
        features, labels = add_fallback_features(
            df, sensor_cols, shot_segments, shot_label, 
            features, labels, target_shots, min_shot_duration, sampling_rate
        )   
    return features, labels, feature_names
def is_valid_shot_segment(df, start, end, shot_label, accel_threshold, gyro_threshold,
                         min_shot_duration, max_shot_duration):
    """
    验证动作片段是否有效
    """
    duration = end - start
    if not (min_shot_duration <= duration <= max_shot_duration):
        return False
    segment = df.iloc[start:end]
    peak_accel = segment['accel_magnitude'].max()
    peak_gyro = segment['gyro_magnitude'].max()
    # 根据动作类型设置不同的验证条件
    if shot_label == 'smash':
        return peak_accel > accel_threshold * 0.9
    elif shot_label == 'clear':
        return (peak_accel > accel_threshold * 0.8) or (peak_gyro > gyro_threshold * 0.7)
    elif shot_label == 'drive':
        return peak_accel > accel_threshold * 0.7
    elif shot_label == 'lift':
        return (peak_accel > accel_threshold * 0.6) or (peak_gyro > gyro_threshold * 0.6)
    else:
        return (peak_accel > accel_threshold * 0.7) or (peak_gyro > gyro_threshold * 0.7)
def get_trigger_strategy(df, shot_label, accel_threshold, gyro_threshold): # 每个类标签不同策略
    """
    根据动作类型选择触发策略
    """
    accel_trigger = df['accel_magnitude'] > accel_threshold
    gyro_trigger = df['gyro_magnitude'] > gyro_threshold
    
    if shot_label == 'smash':
        # Smash: 需要极高的加速度
        return accel_trigger | (df['accel_magnitude'] > accel_threshold * 0.8) 
    elif shot_label == 'clear':
        # Clear: 加速度和旋转都重要
        combined_intensity = df['accel_magnitude'] + (df['gyro_magnitude'] / 40.0)
        return combined_intensity > (accel_threshold * 1.1)
    elif shot_label == 'drive':
        # Drive: 快速但不一定高旋转
        return accel_trigger | (df['accel_magnitude'] > accel_threshold * 0.7)
    elif shot_label == 'lift':
        # Lift: 相对温和，任一传感器激活即可
        return accel_trigger | gyro_trigger
    
    else:
        # 默认策略：综合判断
        return accel_trigger | gyro_trigger
def detect_shot_segments(trigger_condition, df, shot_label, 
                         accel_threshold, gyro_threshold, 
                        min_shot_duration, max_shot_duration):
    """
    检测动作片段
    """
    shot_segments = []
    in_shot = False
    shot_start = None
    for i, is_active in enumerate(trigger_condition):
        if is_active and not in_shot:
            shot_start = i
            in_shot = True
        elif not is_active and in_shot:
            shot_end = i    
            # 验证片段质量
            if is_valid_shot_segment(df, shot_start, shot_end, shot_label, 
                                   accel_threshold, gyro_threshold, 
                                   min_shot_duration, max_shot_duration):
                shot_segments.append((shot_start, shot_end))
            in_shot = False
    
    # 处理最后一个片段
    if in_shot and shot_start is not None:
        shot_end = len(df)
        if is_valid_shot_segment(df, shot_start, shot_end, shot_label,
                               accel_threshold, gyro_threshold, 
                               min_shot_duration, max_shot_duration):
            shot_segments.append((shot_start, shot_end))
    return shot_segments


"""
------ shot threshold settings no label ------
"""
def add_fallback_features_no_label(df, sensor_cols, shot_segments, 
                               existing_features, target_shots, 
                               min_shot_duration, sampling_rate=10):
    """
    修复版本：完全不使用shot_label
    """
    features = existing_features.copy()
    
    # 找到未使用的数据区间
    used_ranges = set()
    for start, end in shot_segments:
        used_ranges.update(range(start, end))
    
    unused_data = [i for i in range(len(df)) if i not in used_ranges]
    
    if unused_data and len(features) < target_shots:
        remaining_needed = target_shots - len(features)
        
        # 方法1: 按运动强度排序选择
        if 'accel_magnitude' in df.columns:
            window_size = 20  # 固定窗口大小
            candidate_windows = []       
            for i in range(0, len(unused_data) - window_size + 1, window_size//2):
                window_indices = unused_data[i:i+window_size]
                if len(window_indices) >= min_shot_duration:
                    window_data = df.iloc[window_indices]
                    intensity_score = (window_data['accel_magnitude'].max() + 
                                     window_data.get('gyro_magnitude', pd.Series([0])).max())
                    candidate_windows.append((window_indices, intensity_score))           
            
            # 选择强度最高的窗口
            candidate_windows.sort(key=lambda x: x[1], reverse=True)       
            for window_indices, _ in candidate_windows[:remaining_needed]:
                window = df.iloc[window_indices]
                window_features, _ = extract_comprehensive_features(window, sensor_cols, sampling_rate)
                features.append(window_features)
        
        # 方法2: 简单分割剩余数据
        if len(features) < target_shots:
            remaining_needed = target_shots - len(features)
            unused_segments = np.array_split(unused_data, remaining_needed)   
            for segment in unused_segments:
                if len(segment) >= min_shot_duration:
                    window = df.iloc[segment]
                    window_features, _ = extract_comprehensive_features(window, sensor_cols, sampling_rate)
                    features.append(window_features)
    
    return features
def extract_shot_features_no_label(df, sensor_cols, shot_segments, target_shots, min_shot_duration, sampling_rate=10):
    """
    修复版本：从动作片段中提取特征，不使用标签
    """
    features = []
    feature_names = None
    
    # 从检测到的片段提取特征
    for i, (start, end) in enumerate(shot_segments):
        if len(features) >= target_shots:
            break        
        shot_window = df.iloc[start:end]
        window_features, names = extract_comprehensive_features(shot_window, sensor_cols, sampling_rate)
        if feature_names is None:
            feature_names = names
        features.append(window_features)
    
    # 如果没有检测到片段，先生成特征名称
    if feature_names is None:
        sample_window = df.iloc[:min(20, len(df))]
        _, feature_names = extract_comprehensive_features(sample_window, sensor_cols, sampling_rate) 
    
    # 备用方法：如果检测到的片段不够 - 不使用shot_label
    if len(features) < target_shots:
        features = add_fallback_features_no_label(
            df, sensor_cols, shot_segments, 
            features, target_shots, min_shot_duration, sampling_rate
        )   
    
    return features, feature_names

"""
------ threshold windowing feature extraction ------
"""
def univeral_threshold_windowing(df, sensor_cols, shot_label,
                                 accel_threshold=None, gyro_threshold=None, 
                                 min_shot_duration=8, max_shot_duration=60, 
                                 target_shots=30):
    """
    通用阈值windowing方法，适用于所有羽毛球动作类型
    
    Parameters:
    - shot_label: 动作类型 ('lift', 'drive', 'smash', 'clear')
    - accel_threshold: 加速度阈值，如果为None则自动设置
    - gyro_threshold: 陀螺仪阈值，如果为None则自动设置
    - min_shot_duration: 最小动作持续时间（样本数）
    - max_shot_duration: 最大动作持续时间（样本数）
    """
    # 1. 根据动作类型自动设置阈值
    if accel_threshold is None or gyro_threshold is None:
        thresholds = get_shot_thresholds(shot_label)
        accel_threshold = accel_threshold or thresholds['accel']
        gyro_threshold = gyro_threshold or thresholds['gyro']
    # 2. 计算传感器幅值
    accel_cols = [col for col in sensor_cols if 'Accel' in col]
    gyro_cols = [col for col in sensor_cols if 'Gyro' in col]
    df['accel_magnitude'] = np.sqrt(
        df[accel_cols[0]]**2 + 
        df[accel_cols[1]]**2 + 
        df[accel_cols[2]]**2
    )
    df['gyro_magnitude'] = np.sqrt(
        df[gyro_cols[0]]**2 + 
        df[gyro_cols[1]]**2 + 
        df[gyro_cols[2]]**2
    )
    # 3. 智能触发检测
    trigger_condition = get_trigger_strategy(df, shot_label, accel_threshold, gyro_threshold)
    # 4. 检测动作片段
    shot_segments = detect_shot_segments(
        trigger_condition, 
        df, 
        shot_label, 
        accel_threshold, 
        gyro_threshold,
        min_shot_duration, 
        max_shot_duration
    )
    # 5. 提取全面特征
    features, labels, feature_names = extract_shot_features(
        df, sensor_cols, shot_segments, shot_label, target_shots,
        min_shot_duration, sampling_rate=10
    )
    return np.array(features), np.array(labels), shot_segments, feature_names
def threshold_windowing(df, sensor_cols, target_shots=30, min_shot_duration=8, max_shot_duration=60):
    """
    基于阈值的windowing方法提取特征
    """
    # 1. 统一参数 - 不根据shot_label变化
    UNIFIED_ACCEL_THRESHOLD = 1.0  # 所有动作使用相同阈值
    UNIFIED_GYRO_THRESHOLD = 40.0  # 所有动作使用相同阈值
    # 2. 计算传感器幅值
    accel_cols = [col for col in sensor_cols if 'Accel' in col]
    gyro_cols = [col for col in sensor_cols if 'Gyro' in col]
    df['accel_magnitude'] = np.sqrt(
        df[accel_cols[0]]**2 + df[accel_cols[1]]**2 + df[accel_cols[2]]**2
    )
    df['gyro_magnitude'] = np.sqrt(
        df[gyro_cols[0]]**2 + df[gyro_cols[1]]**2 + df[gyro_cols[2]]**2
    )
    # 3. 统一触发策略 - 不依赖标签
    accel_trigger = df['accel_magnitude'] > UNIFIED_ACCEL_THRESHOLD
    gyro_trigger = df['gyro_magnitude'] > UNIFIED_GYRO_THRESHOLD
    trigger_condition = accel_trigger | gyro_trigger
    # 4. 统一片段检测
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
            # 统一验证条件 - 不使用shot_label
            if min_shot_duration <= duration <= max_shot_duration:
                segment = df.iloc[shot_start:shot_end]
                peak_accel = segment['accel_magnitude'].max()
                peak_gyro = segment['gyro_magnitude'].max()
                
                # 简单的统一验证标准
                if (peak_accel > UNIFIED_ACCEL_THRESHOLD * 0.7 or 
                    peak_gyro > UNIFIED_GYRO_THRESHOLD * 0.7):
                    shot_segments.append((shot_start, shot_end))
            
            in_shot = False
    # 处理最后一个片段
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
    # 5. 提取特征 - 不涉及标签
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
    # 6. 备用方法：如果检测到的片段不够，使用滑动窗口
    if len(features) < target_shots:
        remaining_needed = target_shots - len(features)
        window_size = 20  # 固定窗口大小
        step_size = window_size // 2     
        for i in range(0, len(df) - window_size + 1, step_size):
            if len(features) >= target_shots:
                break
            window = df.iloc[i:i+window_size]
            # 简单的强度筛选，不使用标签
            if (window['accel_magnitude'].max() > UNIFIED_ACCEL_THRESHOLD * 0.5 or
                window['gyro_magnitude'].max() > UNIFIED_GYRO_THRESHOLD * 0.5):
                window_features, names = extract_comprehensive_features(
                    window, sensor_cols, sampling_rate=10
                )
                if feature_names is None:
                    feature_names = names
                features.append(window_features)
    return np.array(features), shot_segments, feature_names

def save_treshold_windowing_features(all_preprocessed, all_sensor_cols):
    activities = ['clear', 'smash', 'drive', 'lift']
    data_dir = Path('./dataset')
    all_features = {}
    all_labels = {}
    all_feature_names = None
    for activity in activities:
        features, labels, shot_segments, feature_names = univeral_threshold_windowing(
            all_preprocessed[activity], all_sensor_cols, activity,
            min_shot_duration=8, max_shot_duration=60, target_shots=30
        )
        all_features[activity] = features
        all_labels[activity] = labels
        if all_feature_names is None:
            all_feature_names = feature_names
        # 保存特征到CSV
        feature_df = pd.DataFrame(features, columns=feature_names)
        feature_df['shot_type'] = labels
        output_path = data_dir / f'{activity}_x30_threshold_windowing_features.csv'
        feature_df.to_csv(output_path, index=False)
    return all_features, all_labels, all_feature_names
def save_threshold_windowing_no_label(all_preprocessed, all_sensor_cols):
    activities = ['clear', 'smash', 'drive', 'lift']
    data_dir = Path('./dataset')
    all_features = {}
    all_labels = {}
    all_feature_names = None
    for activity in activities:
        # 使用修复后的方法 - 不传递shot_label
        features, shot_segments, feature_names = threshold_windowing(
            all_preprocessed[activity], all_sensor_cols, target_shots=30
        )
        
        # 现在才添加标签
        labels = np.array([activity] * len(features))      
        all_features[activity] = features
        all_labels[activity] = labels
        if all_feature_names is None:
            all_feature_names = feature_names
        
        # 保存特征到CSV
        if len(features) > 0:
            feature_df = pd.DataFrame(features, columns=feature_names)
            feature_df['shot_type'] = labels
            output_path = data_dir / f'{activity}_x30_threshold_windowing_features_no_label.csv'
            feature_df.to_csv(output_path, index=False)
    
    return all_features, all_labels, all_feature_names

def combine_features_data():
    activities = ['clear', 'smash', 'drive', 'lift']
    data_dir = Path('./dataset')
    combined_data = []
    total_samples = 0
    activity_counts = {}
    for activity in activities:
        feature_path = data_dir / f'{activity}_x30_threshold_windowing_features.csv'
        try:
            feature_df = pd.read_csv(feature_path)
            combined_data.append(feature_df)
            activity_counts[activity] = len(feature_df)
            total_samples += len(feature_df)
        except Exception as e:
            activity_counts[activity] = 0
    if not combined_data:
        return pd.DataFrame(), 0, {}
    combined_df = pd.concat(combined_data, ignore_index=True)
    output_file = data_dir / 'combined_threshold_windowing_features.csv'
    combined_df.to_csv(output_file, index=False)
    return combined_df, total_samples, activity_counts
def combine_features_data_no_label():
    activities = ['clear', 'smash', 'drive', 'lift']
    data_dir = Path('./dataset')
    combined_data = []
    total_samples = 0
    activity_counts = {}
    for activity in activities:
        feature_path = data_dir / f'{activity}_x30_threshold_windowing_features_no_label.csv'
        try:
            feature_df = pd.read_csv(feature_path)
            combined_data.append(feature_df)
            activity_counts[activity] = len(feature_df)
            total_samples += len(feature_df)
        except Exception as e:
            activity_counts[activity] = 0
    if not combined_data:
        return pd.DataFrame(), 0, {}
    combined_df = pd.concat(combined_data, ignore_index=True)
    output_file = data_dir / 'combined_threshold_windowing_features_no_label.csv'
    combined_df.to_csv(output_file, index=False)
    return combined_df, total_samples, activity_counts
if __name__ == "__main__":
    all_preprocessed, all_sensor_cols = preprocess_all_data()
    all_features, all_labels, all_feature_names = save_threshold_windowing_no_label(all_preprocessed, all_sensor_cols)
    combined_df, total_samples, activity_counts = combine_features_data_no_label()
