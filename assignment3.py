import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

def load_data(file_path):
    """
    加载数据集
    """
    df = pd.read_csv(file_path)
    feature_columns = [col for col in df.columns if col != 'shot_type']
    X = df[feature_columns].values
    y = df['shot_type'].values
    # 检查数据质量
    missing_values = np.isnan(X).sum()
    infinite_values = np.isinf(X).sum()
    if missing_values > 0 or infinite_values > 0:
        print("  ⚠️  Warning: Data contains missing or infinite values!")
        # 处理无穷大和NaN值
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        print("  ✓ Replaced NaN and infinite values with 0")
    return X, y, feature_columns

def split_train_test_data(X, y, test_size=0.2, random_state=42):
    """
    划分训练集和测试集
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y)
    return X_train, X_test, y_train, y_test

def axis_perturbation(accel_data, gyro_data, angle_std=3.0):
    """
        小幅度旋转加速度计和陀螺仪向量，模拟传感器轻微错位
        
        Parameters:
        - accel_data: 加速度数据 (N, 3)
        - gyro_data: 陀螺仪数据 (N, 3)  
        - angle_std: 旋转角度标准差（度）
        
        Returns:
        - rotated_accel, rotated_gyro: 旋转后的数据
    """
    # 生成小幅随机旋转 (围绕重力方向的小扰动)
    angles = np.random.normal(0, np.radians(angle_std), 3)
    rotation = Rotation.from_euler('xyz', angles)
    rotation_matrix = rotation.as_matrix()
    # 应用旋转到加速度和陀螺仪数据
    rotated_accel = (rotation_matrix @ accel_data.T).T
    rotated_gyro = (rotation_matrix @ gyro_data.T).T
    return rotated_accel, rotated_gyro

def amplitude_scaling(data, scale_range=0.1):
    """
    每个通道的小幅度统一缩放
    
    Parameters:
    - data: 传感器数据 (N, channels)
    - scale_range: 缩放范围 (0.1 = ±10%)
    
    Returns:
    - scaled_data: 缩放后的数据
    """
    scales = 1 + np.random.uniform(-scale_range, scale_range, data.shape[1])
    return data * scales

def data_augumentation(original_features, original_labels, 
                       feature_names, augumentation_factor=3):
    """
    重构传感器数据并应用物理上合理的增强
    
    这个函数模拟从特征重构回传感器数据的过程，然后应用增强
    由于我们只有提取的特征，我们需要创造性地应用增强到特征空间
    """
    augmented_X = [original_features.copy()]  # 包含原始数据
    augmented_y = [original_labels.copy()]
    # 识别传感器相关的特征组
    accel_features = [i for i, name in enumerate(feature_names) if 'accel' in name.lower()]
    gyro_features = [i for i, name in enumerate(feature_names) if 'gyro' in name.lower()]
    for aug_idx in range(augumentation_factor):
        X_aug = original_features.copy()
        # 1. Gaussian Jittering - 向加速度计和陀螺仪特征添加小高斯噪声
        noise_level = 0.015 + np.random.uniform(0, 0.01)  # 1.5-2.5% 噪声
        for feature_group, group_name in [(accel_features, 'accel'), (gyro_features, 'gyro')]:
            if feature_group:
                for idx in feature_group:
                    feature_std = np.std(X_aug[:, idx])
                    if feature_std > 1e-8:
                        noise = np.random.normal(0, feature_std * noise_level, X_aug.shape[0])
                        X_aug[:, idx] += noise
        # 2. Time Warping - 通过特征值的时序相关调整模拟时间扭曲
        if np.random.random() < 0.5:  # 50% 概率
            warp_factor = 1 + np.random.uniform(-0.08, 0.08)  # ±8% 时间扭曲         
            # 对均值和标准差特征应用时间扭曲效果
            mean_features = [i for i, name in enumerate(feature_names) if 'mean' in name.lower()]
            std_features = [i for i, name in enumerate(feature_names) if 'std' in name.lower()]           
            for idx in mean_features:
                # 均值在时间扭曲下保持相对稳定，轻微调整
                X_aug[:, idx] *= (1 + (warp_factor - 1) * 0.1)         
            for idx in std_features:
                # 标准差随时间扭曲变化更明显
                X_aug[:, idx] *= warp_factor
        # 3. Window Slicing - 通过特征值微调模拟不同窗口切片
        if np.random.random() < 0.3:  # 30% 概率
            slice_noise_level = 0.02
            # 对范围和IQR特征应用窗口切片效果
            range_features = [i for i, name in enumerate(feature_names) if any(x in name.lower() for x in ['range', 'iqr', 'min', 'max'])]     
            for idx in range_features:
                feature_std = np.std(X_aug[:, idx])
                if feature_std > 1e-8:  # 确保标准差不为0
                    slice_noise = np.random.normal(0, feature_std * slice_noise_level, X_aug.shape[0])
                    X_aug[:, idx] += slice_noise
        # 4. Axis Perturbation - 模拟传感器轴向扰动对特征的影响
        if np.random.random() < 0.3:  # 30% 概率       
            # 对x, y, z轴相关的特征应用相关性扰动
            axis_groups = {
                'x': [i for i, name in enumerate(feature_names) if '_x ' in name.lower()],
                'y': [i for i, name in enumerate(feature_names) if '_y ' in name.lower()], 
                'z': [i for i, name in enumerate(feature_names) if '_z ' in name.lower()]
            }     
            perturbation_strength = 0.05  # 5% 扰动       
            # 在轴间应用小幅度的特征混合
            for axis1, features1 in axis_groups.items():
                for axis2, features2 in axis_groups.items():
                    if axis1 != axis2 and features1 and features2:
                        mix_ratio = np.random.uniform(-perturbation_strength, perturbation_strength)
                        for f1, f2 in zip(features1[:min(len(features1), len(features2))], 
                                         features2[:min(len(features1), len(features2))]):
                            temp = X_aug[:, f1].copy()
                            X_aug[:, f1] += mix_ratio * X_aug[:, f2]
                            X_aug[:, f2] += mix_ratio * temp
        # 5. Amplitude Scaling (使用空间特征)
        if np.random.random() < 0.4:  # 40% 概率
            scale_range = 0.06 + np.random.uniform(0, 0.04)  # 6-10% 缩放        
            # 按特征类型分组缩放
            feature_groups = {
                'magnitude': [i for i, name in enumerate(feature_names) if 'mag' in name.lower()],
                'frequency': [i for i, name in enumerate(feature_names) if any(x in name.lower() for x in ['freq', 'power', 'energy'])],
                'statistical': [i for i, name in enumerate(feature_names) if any(x in name.lower() for x in ['mean', 'std', 'mad'])],
                'correlation': [i for i, name in enumerate(feature_names) if 'corr' in name.lower()]
            }       
            for group_name, feature_indices in feature_groups.items():
                if feature_indices:
                    scale = 1 + np.random.uniform(-scale_range, scale_range)
                    for idx in feature_indices:
                        X_aug[:, idx] *= scale
        # 最终处理：确保没有异常值
        X_aug = np.nan_to_num(X_aug, nan=0.0, posinf=0.0, neginf=0.0)    
        augmented_X.append(X_aug)
        augmented_y.append(original_labels.copy())
    # 合并所有增强数据
    X_final = np.vstack(augmented_X)
    y_final = np.hstack(augmented_y)
    return X_final, y_final

def save_augmented_data(X_train_aug, y_train_aug, X_test, y_test, feature_names, output_dir="./dataset"):
    """
    保存数据增强后的训练集和原始测试集
    
    Parameters:
    - X_train_aug: 增强后的训练特征
    - y_train_aug: 增强后的训练标签
    - X_test: 原始测试特征
    - y_test: 原始测试标签
    - feature_names: 特征名称列表
    - output_dir: 输出目录
    
    Returns:
    - train_file: 训练数据文件路径
    - test_file: 测试数据文件路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("SAVING AUGMENTED DATASETS")
    print("="*60)
    
    # 保存增强后的训练数据
    train_df = pd.DataFrame(X_train_aug, columns=feature_names)
    train_df['shot_type'] = y_train_aug
    train_file = output_dir / 'augmented_training_data.csv'
    train_df.to_csv(train_file, index=False)
    
    print(f"✓ Augmented training data saved: {train_file}")
    print(f"  Shape: {train_df.shape}")
    print("  Class distribution:")
    train_class_counts = train_df['shot_type'].value_counts()
    for class_name, count in train_class_counts.items():
        percentage = count / len(train_df) * 100
        print(f"    {class_name}: {count} samples ({percentage:.1f}%)")
    
    # 保存原始测试数据
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df['shot_type'] = y_test
    test_file = output_dir / 'original_testing_data.csv'
    test_df.to_csv(test_file, index=False)
    
    print(f"\n✓ Original test data saved: {test_file}")
    print(f"  Shape: {test_df.shape}")
    print("  Class distribution:")
    test_class_counts = test_df['shot_type'].value_counts()
    for class_name, count in test_class_counts.items():
        percentage = count / len(test_df) * 100
        print(f"    {class_name}: {count} samples ({percentage:.1f}%)")
    
    # 保存为numpy格式（可选）
    npz_file = output_dir / 'augmented_dataset.npz'
    np.savez(npz_file,
             X_train=X_train_aug, y_train=y_train_aug,
             X_test=X_test, y_test=y_test,
             feature_names=np.array(feature_names))
    
    print(f"\n✓ NumPy format saved: {npz_file}")
    
    return train_file, test_file, npz_file

def train_random_forest(X_train, y_train, X_test, y_test, random_state=42):
    """
    训练随机森林分类器并进行超参数搜索
    缩小搜索空间以适应GridSearchCV
    
    Parameters:
    - X_train, y_train: 训练数据
    - X_test, y_test: 测试数据
    - random_state: 随机种子
    
    Returns:
    - best_rf: 最佳随机森林模型
    - rf_results: 结果字典
    """
    # 定义超参数搜索空间
    rf_param_grid = {
        'n_estimators': [200, 300, 500],           # 树的数量
        'max_depth': [None, 10, 20],                # 最大深度
        'min_samples_split': [2, 5],                # 分割所需最小样本数
        'max_features': ['sqrt', 'log2'],         # 最大特征数
        'min_samples_leaf': [1, 2, 4],                  # 叶节点最小样本数
    }
    # 创建随机森林分类器
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    rf_search = GridSearchCV(
        rf, rf_param_grid, 
        cv=5,                         # 减少交叉验证折数以加快速度
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0
    )
    rf_search.fit(X_train, y_train)
    # 获取最佳模型
    best_rf = rf_search.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred_rf, average='macro')
    weighted_f1 = f1_score(y_test, y_pred_rf, average='weighted')
    feature_importance = best_rf.feature_importances_
    rf_results = {
        'model': best_rf,
        'best_params': rf_search.best_params_,
        'cv_score': rf_search.best_score_,
        'test_macro_f1': macro_f1,
        'test_weighted_f1': weighted_f1,
        'predictions': y_pred_rf,
        'feature_importance': feature_importance,
        'classification_report': classification_report(y_test, y_pred_rf, output_dict=True)
    }
    return best_rf, rf_results
        
def train_gradient_boosting(X_train, y_train, X_test, y_test, random_state=42):
    """
    训练梯度提升分类器
    
    Parameters:
    - X_train, y_train: 训练数据
    - X_test, y_test: 测试数据
    - random_state: 随机种子
    
    Returns:
    - best_gb: 最佳梯度提升模型
    - gb_results: 结果字典
    """
    # 基础参数
    base_params = {
        'learning_rate': 0.05,
        'max_depth': 3,
        'max_iter': 400,
        'random_state': random_state,
        'early_stopping': True,
        'n_iter_no_change': 20,
        'validation_fraction': 0.2,
        'scoring': 'loss'
    }
    # 创建和训练基础模型
    gb_base = HistGradientBoostingClassifier(**base_params)
    gb_base.fit(X_train, y_train)
    gb_param_grid = {
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'max_leaf_nodes': [31, 50],
        'min_samples_leaf': [20, 50],
        'l2_regularization': [0, 0.1, 0.5]
    }
    # 创建用于搜索的模型（不使用早停，用固定迭代数）
    gb_search_model = HistGradientBoostingClassifier(
        max_iter=200,
        random_state=random_state
    )
    # 随机搜索
    gb_search = GridSearchCV(
        gb_search_model, gb_param_grid,
        cv=5,                         # 交叉验证折数
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0
    )
    gb_search.fit(X_train, y_train)
    # 使用最佳参数重新训练带早停的模型
    best_params = gb_search.best_params_.copy()
    best_params.update({
        'max_iter': 400,
        'early_stopping': True,
        'n_iter_no_change': 20,
        'validation_fraction': 0.2,
        'random_state': random_state
    })
    # 训练最终模型
    best_gb = HistGradientBoostingClassifier(**best_params)
    best_gb.fit(X_train, y_train)
    y_pred_gb = best_gb.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred_gb, average='macro')
    weighted_f1 = f1_score(y_test, y_pred_gb, average='weighted')
    gb_results = {
        'model': best_gb,
        'base_model': gb_base,
        'best_params': best_params,
        'cv_score': gb_search.best_score_,
        'test_macro_f1': macro_f1,
        'test_weighted_f1': weighted_f1,
        'predictions': y_pred_gb,
        'n_iterations': best_gb.n_iter_,
        'classification_report': classification_report(y_test, y_pred_gb, output_dict=True)
    }
    return best_gb, gb_results

def plot_confusion_matrices(y_test, rf_predictions, gb_predictions, class_names):
    """
    绘制混淆矩阵对比
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # Random Forest 混淆矩阵
    rf_cm = confusion_matrix(y_test, rf_predictions)
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Random Forest Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # Gradient Boosting 混淆矩阵
    gb_cm = confusion_matrix(y_test, gb_predictions)
    sns.heatmap(gb_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Gradient Boosting Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(feature_names, rf_importance, top_n=20):
    """
    绘制随机森林特征重要性
    """
    # 获取top N重要特征
    indices = np.argsort(rf_importance)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Feature Importance (Random Forest)')
    plt.bar(range(top_n), rf_importance[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()

def save_model_results(rf_model, gb_model, rf_results, gb_results, feature_names):
    """
    保存模型和结果
    """
# 保存结果摘要
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'random_forest': {
            'best_params': rf_results['best_params'],
            'cv_score': rf_results['cv_score'],
            'test_macro_f1': rf_results['test_macro_f1'],
            'test_weighted_f1': rf_results['test_weighted_f1']
        },
        'gradient_boosting': {
            'best_params': gb_results['best_params'],
            'cv_score': gb_results['cv_score'],
            'test_macro_f1': gb_results['test_macro_f1'],
            'test_weighted_f1': gb_results['test_weighted_f1'],
            'n_iterations': gb_results['n_iterations']
        },
        'feature_names': feature_names
    }
    print(f"✓ Results summary saved: {results_summary}")

def reason_speed_test(X_test, rf_model, gb_model):
    # 测试推理速度
    print(f"\nInference speed comparison (1000 predictions):")
    test_data = X_test[:1].repeat(1000, axis=0)
    start_time = time.time()
    rf_pred = rf_model.predict(test_data)
    rf_time = time.time() - start_time
    start_time = time.time()
    gb_pred = gb_model.predict(test_data)
    gb_time = time.time() - start_time

    print(f"Random Forest Inference Time: {rf_time:.4f} seconds")
    print(f"Gradient Boosting Inference Time: {gb_time:.4f} seconds")
    print(f"  Random Forest: {rf_time:.4f} seconds")
    print(f"  Gradient Boosting: {gb_time:.4f} seconds")
    print(f"  Speed ratio (RF/GB): {rf_time/gb_time:.2f}x")
# 添加到主函数中：
def emergency_leak_check(X, y, feature_names):
    """紧急标签泄漏检查"""
    print("\n=== 紧急标签泄漏检查 ===")
    
    # 检查每个特征是否完美分离类别
    from sklearn.tree import DecisionTreeClassifier
    perfect_features = []
    
    for i, feature_name in enumerate(feature_names):
        # 用单个特征训练最简单的树
        dt = DecisionTreeClassifier(max_depth=1, random_state=42)
        X_single = X[:, i].reshape(-1, 1)
        dt.fit(X_single, y)
        pred = dt.predict(X_single)
        accuracy = (pred == y).mean()
        
        if accuracy > 0.9:
            perfect_features.append((feature_name, accuracy))
            
        # 检查类别是否完全不重叠
        classes = np.unique(y)
        ranges = {}
        for cls in classes:
            cls_data = X[y == cls, i]
            ranges[cls] = (cls_data.min(), cls_data.max())
        
        # 检查重叠
        no_overlap = True
        for c1 in classes:
            for c2 in classes:
                if c1 != c2:
                    min1, max1 = ranges[c1]
                    min2, max2 = ranges[c2]
                    if not (max1 < min2 or max2 < min1):  # 有重叠
                        no_overlap = False
                        break
            if not no_overlap:
                break
                
        if no_overlap:
            print(f"⚠️ {feature_name}: 类别完全不重叠! 范围: {ranges}")
    
    print(f"\n完美分离特征 (单特征准确率>90%):")
    for feature, acc in sorted(perfect_features, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {feature}: {acc:.4f}")
    
    return perfect_features
def compare_raw_vs_features():
    if Path("./dataset/A3_feature.csv").exists():
        from sklearn.model_selection import cross_val_score  # 添加导入
        
        raw_data = pd.read_csv("./dataset/A3_feature.csv")
        
        # 根据CSV结构：列2-7是传感器数据 (Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z)
        X_raw = raw_data.iloc[:, 2:8].values  # 列索引2-7 (Accel_X到Gyro_Z)
        y_raw = raw_data['shot_type'].values
        
        print(f"\nRaw vs Features comparison:")
        print(f"Raw sensor data shape: {X_raw.shape}")
        print(f"Raw labels shape: {y_raw.shape}")
        
        # 检查数据一致性
        if len(X_raw) != len(y_raw):
            min_len = min(len(X_raw), len(y_raw))
            X_raw = X_raw[:min_len]
            y_raw = y_raw[:min_len]
            print(f"Adjusted to consistent length: {min_len}")
        
        # 测试原始传感器数据的分类难度
        lr_raw = LogisticRegression(max_iter=5000, random_state=42, solver='lbfgs')
        scores_raw = cross_val_score(lr_raw, X_raw, y_raw, cv=5, scoring='f1_macro')
        print(f"Raw sensor data (6 channels) LR F1: {scores_raw.mean():.4f} ± {scores_raw.std():.4f}")
        
        # 对比当前使用的特征数据 (使用原始的X, y，不是增强后的)
        lr_feat = LogisticRegression(max_iter=5000, random_state=42, solver='lbfgs')
        scores_feat = cross_val_score(lr_feat, X, y, cv=5, scoring='f1_macro')
        print(f"Engineered features ({len(feature_columns)} features) LR F1: {scores_feat.mean():.4f} ± {scores_feat.std():.4f}")
        
        # 显示特征工程的影响
        improvement = scores_feat.mean() - scores_raw.mean()
        print(f"Feature engineering improvement: +{improvement:.4f} F1 score")
        
    else:
        print("A3_feature.csv not found - skipping raw vs features comparison")

if __name__ == "__main__":

    # Step 1: 加载数据
    data_file = Path("./dataset/all_feature_no_label.csv")
    print(f"Loading data from: {data_file}")
    X, y, feature_columns = load_data(data_file)
    print(f"✓ Loaded {len(X)} samples with {len(feature_columns)} features")
    # 显示类别分布
    unique_classes, class_counts = np.unique(y, return_counts=True)
    perfect_features = emergency_leak_check(X, y, feature_columns)
    
    # Step 2: 分割训练测试数据
    print(f"\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = split_train_test_data(X, y)
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Test samples: {len(X_test)}")

    # Step 2.5: 数据复杂化处理
    # 1. 移除零方差特征
    from sklearn.feature_selection import VarianceThreshold
    variance_selector = VarianceThreshold(threshold=0.01)  # 移除方差<0.01的特征
    X_train_filtered = variance_selector.fit_transform(X_train)
    X_test_filtered = variance_selector.transform(X_test)
    # 更新特征列名
    selected_features = [feature_columns[i] for i in range(len(feature_columns)) 
                if variance_selector.variances_[i] >= 0.01]
    # 2. 添加噪声增加数据复杂性
    noise_level = 0.05  # 5% 噪声
    X_train_noisy = X_train_filtered + np.random.normal(0, noise_level * np.std(X_train_filtered, axis=0), X_train_filtered.shape)
    X_test_noisy = X_test_filtered + np.random.normal(0, noise_level * np.std(X_test_filtered, axis=0), X_test_filtered.shape)
    # 3. 使用更严格的数据分割（增加测试集比例）
    # 重新测试简单模型的性能
    simple_lr_test = LogisticRegression(max_iter=1000, random_state=42)
    simple_lr_test.fit(X_train_noisy, y_train)
    lr_score_enhanced = f1_score(y_test, simple_lr_test.predict(X_test_noisy), average='macro')
    print(f"Enhanced LR Macro-F1: {lr_score_enhanced:.4f}")
    # 更新训练数据为处理后的数据
    X_train = X_train_noisy
    X_test = X_test_noisy
    feature_columns = selected_features

    # Step 3: 数据增强
    print(f"\nApplying data augmentation...")
    X_train_aug, y_train_aug = data_augumentation(
        X_train, y_train, 
        feature_names=feature_columns,
        augumentation_factor=3
    )
    
    # Step 4 train Random Forest
    rf_model, rf_results = train_random_forest(
        X_train_aug, y_train_aug, X_test, y_test
    )

    # Step 5 train Gradient Boosting
    gb_model, gb_results = train_gradient_boosting(
        X_train_aug, y_train_aug, X_test, y_test
    )

    # 1. 检查基线模型
    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train_aug, y_train_aug)
    dummy_score = f1_score(y_test, dummy.predict(X_test), average='macro')
    print(f"Dummy classifier Macro-F1: {dummy_score:.4f}")
    # 2. 使用更简单的模型测试
    from sklearn.linear_model import LogisticRegression
    simple_lr = LogisticRegression(max_iter=1000, random_state=42)
    simple_lr.fit(X_train_aug, y_train_aug)
    lr_score = f1_score(y_test, simple_lr.predict(X_test), average='macro')
    print(f"Simple Logistic Regression Macro-F1: {lr_score:.4f}")
    # 3. 使用留一法验证
    from sklearn.model_selection import LeaveOneOut, cross_val_score
    loo = LeaveOneOut()
    loo_scores = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X, y, cv=loo, scoring='f1_macro'
    )
    print(f"Leave-One-Out CV Macro-F1: {loo_scores.mean():.4f} (+/- {loo_scores.std()*2:.4f})")
    # 4. 检查特征方差
    feature_vars = np.var(X, axis=0)
    print(f"Zero variance features: {np.sum(feature_vars < 1e-10)}")
    print(f"Low variance features (<0.01): {np.sum(feature_vars < 0.01)}")

    compare_raw_vs_features()
        
    # Step 6: 比较结果
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}")
    
    print(f"Random Forest:")
    print(f"  CV Macro-F1: {rf_results['cv_score']:.4f}")
    print(f"  Test Macro-F1: {rf_results['test_macro_f1']:.4f}")
    print(f"  Test Weighted-F1: {rf_results['test_weighted_f1']:.4f}")
    
    print(f"\nGradient Boosting:")
    print(f"  CV Macro-F1: {gb_results['cv_score']:.4f}")
    print(f"  Test Macro-F1: {gb_results['test_macro_f1']:.4f}")
    print(f"  Test Weighted-F1: {gb_results['test_weighted_f1']:.4f}")
    print(f"  Training iterations: {gb_results['n_iterations']}")

    print(f"\n{'='*70}")

    print("MODEL ANALYSIS")
    print(f"{'='*70}")

    print(f"Dataset characteristics:")
    print(f"  - Total samples: {len(X)} (small but high-quality)")
    print(f"  - Features after filtering: {len(feature_columns)}")
    print(f"  - Classes: 4 distinct badminton shots")
    print(f"  - Feature/Sample ratio: {len(feature_columns)/len(X):.2f}")

    print(f"\nModel complexity comparison:")
    print(f"Random Forest:")
    print(f"  - Trees: {rf_results['best_params']['n_estimators']}")
    print(f"  - Max depth: {rf_results['best_params']['max_depth']}")
    print(f"  - Training time: Fast")

    print(f"Gradient Boosting:")
    print(f"  - Learning rate: {gb_results['best_params']['learning_rate']}")
    print(f"  - Max depth: {gb_results['best_params']['max_depth']}")
    print(f"  - Iterations used: {gb_results['n_iterations']}/400 (early stopping)")

    reason_speed_test(X_test, rf_model, gb_model)
    # Step 7: 可视化结果
    plot_confusion_matrices(
        y_test, 
        rf_results['predictions'], 
        gb_results['predictions'], 
        unique_classes
    )
    plot_feature_importance(
        feature_columns, 
        rf_results['feature_importance']
    )

    # # Step 4: 保存数据集
    # train_file, test_file, npz_file = save_augmented_data(
    #     X_train_aug, y_train_aug, 
    #     X_test, y_test, 
    #     feature_columns,
    #     output_dir="./dataset"
    # )
    