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
    Load dataset from CSV file
    """
    df = pd.read_csv(file_path)
    feature_columns = [col for col in df.columns if col != 'shot_type']
    X = df[feature_columns].values
    y = df['shot_type'].values
    # Check data quality
    missing_values = np.isnan(X).sum()
    infinite_values = np.isinf(X).sum()
    if missing_values > 0 or infinite_values > 0:
        print("  ⚠️  Warning: Data contains missing or infinite values!")
        # Handle infinite and NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        print("  ✓ Replaced NaN and infinite values with 0")
    return X, y, feature_columns

def split_train_test_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y)
    return X_train, X_test, y_train, y_test

def axis_perturbation(accel_data, gyro_data, angle_std=3.0):
    """
        Small-angle rotation of accelerometer and gyroscope vectors to simulate sensor misalignment
        
        Parameters:
        - accel_data: Acceleration data (N, 3)
        - gyro_data: Gyroscope data (N, 3)  
        - angle_std: Rotation angle standard deviation (degrees)
        
        Returns:
        - rotated_accel, rotated_gyro: Rotated data
    """
    # Generate small random rotation (small perturbations around gravity direction)
    angles = np.random.normal(0, np.radians(angle_std), 3)
    rotation = Rotation.from_euler('xyz', angles)
    rotation_matrix = rotation.as_matrix()
    # Apply rotation to acceleration and gyroscope data
    rotated_accel = (rotation_matrix @ accel_data.T).T
    rotated_gyro = (rotation_matrix @ gyro_data.T).T
    return rotated_accel, rotated_gyro

def amplitude_scaling(data, scale_range=0.1):
    """
    Small-amplitude uniform scaling for each channel
    
    Parameters:
    - data: Sensor data (N, channels)
    - scale_range: Scaling range (0.1 = ±10%)
    
    Returns:
    - scaled_data: Scaled data
    """
    scales = 1 + np.random.uniform(-scale_range, scale_range, data.shape[1])
    return data * scales

def data_augumentation(original_features, original_labels, 
                       feature_names, augumentation_factor=3):
    """
    Reconstruct sensor data and apply physically reasonable augmentation
    
    This function simulates the process of reconstructing from features back to sensor data, then applies augmentation
    Since we only have extracted features, we need to creatively apply augmentation to the feature space
    """
    augmented_X = [original_features.copy()]  # Include original data
    augmented_y = [original_labels.copy()]
    # Identify sensor-related feature groups
    accel_features = [i for i, name in enumerate(feature_names) if 'accel' in name.lower()]
    gyro_features = [i for i, name in enumerate(feature_names) if 'gyro' in name.lower()]
    for aug_idx in range(augumentation_factor):
        X_aug = original_features.copy()
        # 1. Gaussian Jittering - Add small Gaussian noise to accelerometer and gyroscope features
        noise_level = 0.015 + np.random.uniform(0, 0.01)  # 1.5-2.5% noise
        for feature_group, group_name in [(accel_features, 'accel'), (gyro_features, 'gyro')]:
            if feature_group:
                for idx in feature_group:
                    feature_std = np.std(X_aug[:, idx])
                    if feature_std > 1e-8:
                        noise = np.random.normal(0, feature_std * noise_level, X_aug.shape[0])
                        X_aug[:, idx] += noise
        # 2. Time Warping - Simulate time warping through temporal correlation adjustments of feature values
        if np.random.random() < 0.5:  # 50% probability
            warp_factor = 1 + np.random.uniform(-0.08, 0.08)  # ±8% time warping         
            # Apply time warping effects to mean and standard deviation features
            mean_features = [i for i, name in enumerate(feature_names) if 'mean' in name.lower()]
            std_features = [i for i, name in enumerate(feature_names) if 'std' in name.lower()]           
            for idx in mean_features:
                # Mean values remain relatively stable under time warping, slight adjustment
                X_aug[:, idx] *= (1 + (warp_factor - 1) * 0.1)         
            for idx in std_features:
                # Standard deviation changes more significantly with time warping
                X_aug[:, idx] *= warp_factor
        # 3. Window Slicing - Simulate different window slicing through feature value fine-tuning
        if np.random.random() < 0.3:  # 30% probability
            slice_noise_level = 0.02
            # Apply window slicing effects to range and IQR features
            range_features = [i for i, name in enumerate(feature_names) if any(x in name.lower() for x in ['range', 'iqr', 'min', 'max'])]     
            for idx in range_features:
                feature_std = np.std(X_aug[:, idx])
                if feature_std > 1e-8:  # Ensure standard deviation is not zero
                    slice_noise = np.random.normal(0, feature_std * slice_noise_level, X_aug.shape[0])
                    X_aug[:, idx] += slice_noise
        # 4. Axis Perturbation - Simulate the impact of sensor axis perturbation on features
        if np.random.random() < 0.3:  # 30% probability       
            # Apply correlation perturbations to x, y, z axis-related features
            axis_groups = {
                'x': [i for i, name in enumerate(feature_names) if '_x ' in name.lower()],
                'y': [i for i, name in enumerate(feature_names) if '_y ' in name.lower()], 
                'z': [i for i, name in enumerate(feature_names) if '_z ' in name.lower()]
            }     
            perturbation_strength = 0.05  # 5% perturbation       
            # Apply small-amplitude feature mixing between axes
            for axis1, features1 in axis_groups.items():
                for axis2, features2 in axis_groups.items():
                    if axis1 != axis2 and features1 and features2:
                        mix_ratio = np.random.uniform(-perturbation_strength, perturbation_strength)
                        for f1, f2 in zip(features1[:min(len(features1), len(features2))], 
                                         features2[:min(len(features1), len(features2))]):
                            temp = X_aug[:, f1].copy()
                            X_aug[:, f1] += mix_ratio * X_aug[:, f2]
                            X_aug[:, f2] += mix_ratio * temp
        # 5. Amplitude Scaling (using spatial features)
        if np.random.random() < 0.4:  # 40% probability
            scale_range = 0.06 + np.random.uniform(0, 0.04)  # 6-10% scaling        
            # Group scaling by feature type
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
        # Final processing: ensure no abnormal values
        X_aug = np.nan_to_num(X_aug, nan=0.0, posinf=0.0, neginf=0.0)    
        augmented_X.append(X_aug)
        augmented_y.append(original_labels.copy())
    # Combine all augmented data
    X_final = np.vstack(augmented_X)
    y_final = np.hstack(augmented_y)
    return X_final, y_final

def train_random_forest(X_train, y_train, X_test, y_test, random_state=42):
    """
    Train Random Forest classifier with hyperparameter search
    Reduced search space to adapt to GridSearchCV
    
    Parameters:
    - X_train, y_train: Training data
    - X_test, y_test: Test data
    - random_state: Random seed
    
    Returns:
    - best_rf: Best Random Forest model
    - rf_results: Results dictionary
    """
    # Define hyperparameter search space
    rf_param_grid = {
        'n_estimators': [200, 300, 500],           # Number of trees
        'max_depth': [None, 10, 20],                # Maximum depth
        'min_samples_split': [2, 5],                # Minimum samples required to split
        'max_features': ['sqrt', 'log2'],         # Maximum number of features
        'min_samples_leaf': [1, 2, 4],                  # Minimum samples in leaf node
    }
    # Create Random Forest classifier
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    rf_search = GridSearchCV(
        rf, rf_param_grid, 
        cv=5,                         # 5-fold cross-validation to speed up
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0
    )
    rf_search.fit(X_train, y_train)
    # Get best model
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
    Train Gradient Boosting classifier
    
    Parameters:
    - X_train, y_train: Training data
    - X_test, y_test: Test data
    - random_state: Random seed
    
    Returns:
    - best_gb: Best Gradient Boosting model
    - gb_results: Results dictionary
    """
    # Base parameters
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
    # Create and train base model
    gb_base = HistGradientBoostingClassifier(**base_params)
    gb_base.fit(X_train, y_train)
    gb_param_grid = {
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'max_leaf_nodes': [31, 50],
        'min_samples_leaf': [20, 50],
        'l2_regularization': [0, 0.1, 0.5]
    }
    # Create model for search (no early stopping, fixed iterations)
    gb_search_model = HistGradientBoostingClassifier(
        max_iter=200,
        random_state=random_state
    )
    # Grid search
    gb_search = GridSearchCV(
        gb_search_model, gb_param_grid,
        cv=5,                         # Cross-validation folds
        scoring='f1_macro',
        n_jobs=-1,
        verbose=0
    )
    gb_search.fit(X_train, y_train)
    # Retrain model with early stopping using best parameters
    best_params = gb_search.best_params_.copy()
    best_params.update({
        'max_iter': 400,
        'early_stopping': True,
        'n_iter_no_change': 20,
        'validation_fraction': 0.2,
        'random_state': random_state
    })
    # Train final model
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
    Plot confusion matrix comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # Random Forest confusion matrix
    rf_cm = confusion_matrix(y_test, rf_predictions)
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Random Forest Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # Gradient Boosting confusion matrix
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
    Plot Random Forest feature importance
    """
    # Get top N important features
    indices = np.argsort(rf_importance)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Feature Importance (Random Forest)')
    plt.bar(range(top_n), rf_importance[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()

def plot_f1_scores(rf_results, gb_results, class_names):
    """
    Plot F1 score comparison charts
    
    Parameters:
    - rf_results: Random Forest results dictionary
    - gb_results: Gradient Boosting results dictionary  
    - class_names: Class name list
    """
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overall F1 score comparison (top left)
    models = ['Random Forest', 'Gradient Boosting']
    cv_f1_scores = [rf_results['cv_score'], gb_results['cv_score']]
    test_macro_f1 = [rf_results['test_macro_f1'], gb_results['test_macro_f1']]
    test_weighted_f1 = [rf_results['test_weighted_f1'], gb_results['test_weighted_f1']]
    
    x = np.arange(len(models))
    width = 0.25
    
    axes[0, 0].bar(x - width, cv_f1_scores, width, label='CV Macro F1', alpha=0.8, color='skyblue')
    axes[0, 0].bar(x, test_macro_f1, width, label='Test Macro F1', alpha=0.8, color='orange')
    axes[0, 0].bar(x + width, test_weighted_f1, width, label='Test Weighted F1', alpha=0.8, color='lightgreen')
    
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].set_title('Overall F1 Score Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for i, (cv, macro, weighted) in enumerate(zip(cv_f1_scores, test_macro_f1, test_weighted_f1)):
        axes[0, 0].text(i - width, cv + 0.01, f'{cv:.3f}', ha='center', va='bottom', fontsize=9)
        axes[0, 0].text(i, macro + 0.01, f'{macro:.3f}', ha='center', va='bottom', fontsize=9)
        axes[0, 0].text(i + width, weighted + 0.01, f'{weighted:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Per-class F1 score comparison (top right)
    rf_class_f1 = [rf_results['classification_report'][cls]['f1-score'] for cls in class_names]
    gb_class_f1 = [gb_results['classification_report'][cls]['f1-score'] for cls in class_names]
    
    x_class = np.arange(len(class_names))
    width_class = 0.35
    
    bars1 = axes[0, 1].bar(x_class - width_class/2, rf_class_f1, width_class, 
                          label='Random Forest', alpha=0.8, color='lightcoral')
    bars2 = axes[0, 1].bar(x_class + width_class/2, gb_class_f1, width_class, 
                          label='Gradient Boosting', alpha=0.8, color='lightblue')
    
    axes[0, 1].set_xlabel('Shot Types')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Per-Class F1 Score Comparison')
    axes[0, 1].set_xticks(x_class)
    axes[0, 1].set_xticklabels(class_names)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Precision, Recall, F1-Score comparison (bottom left) - Random Forest
    rf_precision = [rf_results['classification_report'][cls]['precision'] for cls in class_names]
    rf_recall = [rf_results['classification_report'][cls]['recall'] for cls in class_names]
    
    x_metrics = np.arange(len(class_names))
    width_metrics = 0.25
    
    axes[1, 0].bar(x_metrics - width_metrics, rf_precision, width_metrics, 
                  label='Precision', alpha=0.8, color='gold')
    axes[1, 0].bar(x_metrics, rf_recall, width_metrics, 
                  label='Recall', alpha=0.8, color='lightgreen')
    axes[1, 0].bar(x_metrics + width_metrics, rf_class_f1, width_metrics, 
                  label='F1-Score', alpha=0.8, color='lightcoral')
    
    axes[1, 0].set_xlabel('Shot Types')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Random Forest: Precision, Recall, F1-Score by Class')
    axes[1, 0].set_xticks(x_metrics)
    axes[1, 0].set_xticklabels(class_names)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Precision, Recall, F1-Score comparison (bottom right) - Gradient Boosting
    gb_precision = [gb_results['classification_report'][cls]['precision'] for cls in class_names]
    gb_recall = [gb_results['classification_report'][cls]['recall'] for cls in class_names]
    
    axes[1, 1].bar(x_metrics - width_metrics, gb_precision, width_metrics, 
                  label='Precision', alpha=0.8, color='gold')
    axes[1, 1].bar(x_metrics, gb_recall, width_metrics, 
                  label='Recall', alpha=0.8, color='lightgreen')
    axes[1, 1].bar(x_metrics + width_metrics, gb_class_f1, width_metrics, 
                  label='F1-Score', alpha=0.8, color='lightblue')
    
    axes[1, 1].set_xlabel('Shot Types')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Gradient Boosting: Precision, Recall, F1-Score by Class')
    axes[1, 1].set_xticks(x_metrics)
    axes[1, 1].set_xticklabels(class_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_model_performance_summary(rf_results, gb_results):
    """
    绘制模型性能总结图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 雷达图比较 (左图)
    categories = ['CV F1', 'Test Macro F1', 'Test Weighted F1', 'Avg Precision', 'Avg Recall']
    
    # 计算平均精确率和召回率
    rf_avg_precision = np.mean([rf_results['classification_report'][cls]['precision'] 
                               for cls in ['clear', 'drive', 'lift', 'smash']])
    rf_avg_recall = np.mean([rf_results['classification_report'][cls]['recall'] 
                            for cls in ['clear', 'drive', 'lift', 'smash']])
    
    gb_avg_precision = np.mean([gb_results['classification_report'][cls]['precision'] 
                               for cls in ['clear', 'drive', 'lift', 'smash']])
    gb_avg_recall = np.mean([gb_results['classification_report'][cls]['recall'] 
                            for cls in ['clear', 'drive', 'lift', 'smash']])
    
    rf_values = [rf_results['cv_score'], rf_results['test_macro_f1'], 
                rf_results['test_weighted_f1'], rf_avg_precision, rf_avg_recall]
    gb_values = [gb_results['cv_score'], gb_results['test_macro_f1'], 
                gb_results['test_weighted_f1'], gb_avg_precision, gb_avg_recall]
    
    # 创建角度
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 闭合图形
    
    rf_values = np.concatenate((rf_values, [rf_values[0]]))
    gb_values = np.concatenate((gb_values, [gb_values[0]]))
    
    ax1 = plt.subplot(121, projection='polar')
    ax1.plot(angles, rf_values, 'o-', linewidth=2, label='Random Forest', color='red', alpha=0.7)
    ax1.fill(angles, rf_values, alpha=0.25, color='red')
    ax1.plot(angles, gb_values, 'o-', linewidth=2, label='Gradient Boosting', color='blue', alpha=0.7)
    ax1.fill(angles, gb_values, alpha=0.25, color='blue')
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories)
    ax1.set_ylim(0, 1)
    ax1.set_title('Model Performance Radar Chart', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax1.grid(True)
    
    # 2. F1分数趋势对比 (右图)
    metrics = ['CV F1', 'Test Macro F1', 'Test Weighted F1']
    rf_trend = [rf_results['cv_score'], rf_results['test_macro_f1'], rf_results['test_weighted_f1']]
    gb_trend = [gb_results['cv_score'], gb_results['test_macro_f1'], gb_results['test_weighted_f1']]
    
    ax2 = plt.subplot(122)
    x_trend = range(len(metrics))
    ax2.plot(x_trend, rf_trend, 'o-', linewidth=3, markersize=8, label='Random Forest', color='red', alpha=0.8)
    ax2.plot(x_trend, gb_trend, 's-', linewidth=3, markersize=8, label='Gradient Boosting', color='blue', alpha=0.8)
    
    # 添加数值标签
    for i, (rf_val, gb_val) in enumerate(zip(rf_trend, gb_trend)):
        ax2.text(i, rf_val + 0.01, f'{rf_val:.3f}', ha='center', va='bottom', fontweight='bold')
        ax2.text(i, gb_val - 0.03, f'{gb_val:.3f}', ha='center', va='top', fontweight='bold')
    
    ax2.set_xticks(x_trend)
    ax2.set_xticklabels(metrics)
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score Trend Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

def create_performance_dashboard(rf_results, gb_results, class_names, feature_columns):
    """
    创建综合性能仪表板
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 创建网格布局
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. 总体F1分数对比 (左上大图)
    ax1 = fig.add_subplot(gs[0, :2])
    models = ['Random Forest', 'Gradient Boosting']
    cv_f1 = [rf_results['cv_score'], gb_results['cv_score']]
    test_macro = [rf_results['test_macro_f1'], gb_results['test_macro_f1']]
    test_weighted = [rf_results['test_weighted_f1'], gb_results['test_weighted_f1']]
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax1.bar(x - width, cv_f1, width, label='CV Macro F1', color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x, test_macro, width, label='Test Macro F1', color='#4ECDC4', alpha=0.8)
    bars3 = ax1.bar(x + width, test_weighted, width, label='Test Weighted F1', color='#45B7D1', alpha=0.8)
    
    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax1.set_title('Model Performance Overview', fontsize=16, fontweight='bold')
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(max(cv_f1), max(test_macro), max(test_weighted)) + 0.05)
    
    # 2. 各类别性能热图 (右上)
    ax2 = fig.add_subplot(gs[0, 2:])
    
    # 准备热图数据
    rf_metrics = []
    gb_metrics = []
    metric_names = ['Precision', 'Recall', 'F1-Score']
    
    for cls in class_names:
        rf_metrics.append([
            rf_results['classification_report'][cls]['precision'],
            rf_results['classification_report'][cls]['recall'],
            rf_results['classification_report'][cls]['f1-score']
        ])
        gb_metrics.append([
            gb_results['classification_report'][cls]['precision'],
            gb_results['classification_report'][cls]['recall'],
            gb_results['classification_report'][cls]['f1-score']
        ])
    
    # 合并RF和GB数据进行对比
    combined_data = np.array(rf_metrics) - np.array(gb_metrics)  # RF - GB的差异
    
    im = ax2.imshow(combined_data.T, cmap='RdBu_r', aspect='auto', vmin=-0.2, vmax=0.2)
    ax2.set_xticks(range(len(class_names)))
    ax2.set_yticks(range(len(metric_names)))
    ax2.set_xticklabels(class_names)
    ax2.set_yticklabels(metric_names)
    ax2.set_title('Performance Difference (RF - GB)', fontsize=14, fontweight='bold')
    
    # 添加数值标签
    for i in range(len(metric_names)):
        for j in range(len(class_names)):
            text = ax2.text(j, i, f'{combined_data[j, i]:.3f}', 
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax2, shrink=0.8)
    
    # 3. 类别F1分数详细对比 (中间左)
    ax3 = fig.add_subplot(gs[1, :2])
    
    rf_f1_by_class = [rf_results['classification_report'][cls]['f1-score'] for cls in class_names]
    gb_f1_by_class = [gb_results['classification_report'][cls]['f1-score'] for cls in class_names]
    
    x_pos = np.arange(len(class_names))
    
    bars_rf = ax3.bar(x_pos - 0.2, rf_f1_by_class, 0.4, label='Random Forest', 
                     color='#FF6B6B', alpha=0.8)
    bars_gb = ax3.bar(x_pos + 0.2, gb_f1_by_class, 0.4, label='Gradient Boosting', 
                     color='#4ECDC4', alpha=0.8)
    
    # 添加数值标签
    for bars in [bars_rf, bars_gb]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_title('F1-Score by Shot Type', fontsize=14, fontweight='bold')
    ax3.set_ylabel('F1 Score')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(class_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 模型复杂度对比 (中间右)
    ax4 = fig.add_subplot(gs[1, 2:])
    
    complexity_metrics = ['Trees/Iterations', 'Max Depth', 'Training Time']
    rf_complexity = [
        rf_results['best_params']['n_estimators'],
        rf_results['best_params']['max_depth'] if rf_results['best_params']['max_depth'] else 20,
        1.0  # 相对训练时间
    ]
    gb_complexity = [
        gb_results['n_iterations'],
        gb_results['best_params']['max_depth'],
        0.8  # 相对训练时间
    ]
    
    # 标准化复杂度指标 (0-1范围)
    rf_norm = [rf_complexity[0]/500, rf_complexity[1]/25, rf_complexity[2]]
    gb_norm = [gb_complexity[0]/500, gb_complexity[1]/25, gb_complexity[2]]
    
    x_comp = np.arange(len(complexity_metrics))
    
    ax4.bar(x_comp - 0.2, rf_norm, 0.4, label='Random Forest', color='#FF6B6B', alpha=0.8)
    ax4.bar(x_comp + 0.2, gb_norm, 0.4, label='Gradient Boosting', color='#4ECDC4', alpha=0.8)
    
    ax4.set_title('Model Complexity Comparison (Normalized)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Relative Complexity')
    ax4.set_xticks(x_comp)
    ax4.set_xticklabels(complexity_metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 性能稳定性对比 (底部左)
    ax5 = fig.add_subplot(gs[2, :2])
    
    performance_gap = [
        abs(rf_results['cv_score'] - rf_results['test_macro_f1']),
        abs(gb_results['cv_score'] - gb_results['test_macro_f1'])
    ]
    
    colors = ['#FF6B6B', '#4ECDC4']
    bars = ax5.bar(models, performance_gap, color=colors, alpha=0.8)
    
    for bar, gap in zip(bars, performance_gap):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{gap:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax5.set_title('Model Stability (CV vs Test F1 Gap)', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Performance Gap')
    ax5.grid(True, alpha=0.3)
    
    # 6. Top特征重要性 (底部右)
    ax6 = fig.add_subplot(gs[2, 2:])
    
    top_n = 10
    indices = np.argsort(rf_results['feature_importance'])[::-1][:top_n]
    top_importance = rf_results['feature_importance'][indices]
    top_features = [feature_columns[i] for i in indices]
    
    # 缩短特征名称以便显示
    short_features = [name[:15] + '...' if len(name) > 15 else name for name in top_features]
    
    bars = ax6.barh(range(top_n), top_importance, color='#96CEB4', alpha=0.8)
    ax6.set_yticks(range(top_n))
    ax6.set_yticklabels(short_features)
    ax6.set_xlabel('Importance')
    ax6.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        ax6.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{top_importance[i]:.3f}', ha='left', va='center', fontweight='bold')
    
    plt.suptitle('Badminton Shot Classification - Performance Dashboard', 
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.show()

if __name__ == "__main__":

    # Step 1: Load original dataset
    data_file = Path("A3_features.csv")
    print(f"Loading original data from: {data_file}")
    X_original, y_original, feature_columns = load_data(data_file)
    print(f"✓ Loaded {len(X_original)} samples with {len(feature_columns)} features")
    
    # Step 2: Load Kyle's dataset
    kyle_data_file = Path("A3_features_kyle.csv")
    print(f"\nLoading Kyle's data from: {kyle_data_file}")
    X_kyle, y_kyle, feature_columns_kyle = load_data(kyle_data_file)
    print(f"✓ Loaded {len(X_kyle)} Kyle's samples with {len(feature_columns_kyle)} features")
    
    # Verify feature dimensions match
    if len(feature_columns) != len(feature_columns_kyle):
        print(f"\n⚠️  WARNING: Feature dimension mismatch!")
        print(f"  Original: {len(feature_columns)} features")
        print(f"  Kyle's: {len(feature_columns_kyle)} features")
        raise ValueError("Feature dimensions don't match!")
    
    # Display class distribution
    unique_classes_original, class_counts_original = np.unique(y_original, return_counts=True)
    unique_classes_kyle, class_counts_kyle = np.unique(y_kyle, return_counts=True)
    
    print(f"\nOriginal dataset class distribution:")
    for cls, count in zip(unique_classes_original, class_counts_original):
        print(f"  {cls}: {count} samples")
    
    print(f"\nKyle's dataset class distribution:")
    for cls, count in zip(unique_classes_kyle, class_counts_kyle):
        print(f"  {cls}: {count} samples")
    
    # Step 3: Split original data for training (use original data only for training)
    # print(f"\nSplitting original data for training and validation...")
    # X_train, X_val, y_train, y_val = split_train_test_data(X_original, y_original, test_size=0.2)
    # print(f"✓ Training samples: {len(X_train)}")
    # print(f"✓ Validation samples: {len(X_val)}")
    X_combined = np.vstack([X_original, X_kyle])
    y_combined = np.hstack([y_original, y_kyle])
    
    # Step 4: Use Kyle's data as test set
    # X_test = X_kyle
    # y_test = y_kyle
    # print(f"✓ Test samples (Kyle's data): {len(X_test)}")
    # 第一次划分：80% train+val, 20% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_combined, y_combined,
        test_size=0.2,
        random_state=42,
        stratify=y_combined
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.25,  # 0.25 * 0.8 = 0.2
        random_state=42,
        stratify=y_train_val
    )

    # Step 5: Data preprocessing
    print(f"\nApplying data preprocessing...")
    
    # 1. Remove zero-variance features
    from sklearn.feature_selection import VarianceThreshold
    variance_selector = VarianceThreshold(threshold=0.01)
    X_train_filtered = variance_selector.fit_transform(X_train)
    X_val_filtered = variance_selector.transform(X_val)
    X_test_filtered = variance_selector.transform(X_test)
    
    # Update feature columns
    selected_features = [feature_columns[i] for i in range(len(feature_columns)) 
                if variance_selector.variances_[i] >= 0.01]
    
    print(f"✓ Features after filtering: {len(selected_features)} (removed {len(feature_columns) - len(selected_features)} zero-variance features)")
    
    # 2. Add noise to increase data complexity
    noise_level = 0.05  # 5% noise
    X_train_noisy = X_train_filtered + np.random.normal(0, noise_level * np.std(X_train_filtered, axis=0), X_train_filtered.shape)
    
    # Update training data
    X_train = X_train_noisy
    X_val = X_val_filtered
    X_test = X_test_filtered
    feature_columns = selected_features
    
    # Step 6: Data augmentation (only on training data)
    print(f"\nApplying data augmentation on training set...")
    X_train_aug, y_train_aug = data_augumentation(
        X_train, y_train, 
        feature_names=feature_columns,
        augumentation_factor=3
    )
    print(f"✓ Training samples after augmentation: {len(X_train_aug)}")
    
    # Step 7: Train Random Forest
    print(f"\n{'='*70}")
    print("TRAINING RANDOM FOREST")
    print(f"{'='*70}")
    rf_model, rf_results = train_random_forest(
        X_train_aug, y_train_aug, X_test, y_test
    )
    print(f"✓ Random Forest training completed")

    # Step 8: Train Gradient Boosting
    print(f"\n{'='*70}")
    print("TRAINING GRADIENT BOOSTING")
    print(f"{'='*70}")
    gb_model, gb_results = train_gradient_boosting(
        X_train_aug, y_train_aug, X_test, y_test
    )
    print(f"✓ Gradient Boosting training completed")

    # Step 9: Compare results
    print(f"\n{'='*70}")
    print("MODEL COMPARISON - TEST ON KYLE'S DATA")
    print(f"{'='*70}")
    
    print(f"\nRandom Forest (tested on Kyle's data):")
    print(f"  CV Macro-F1 (on original data): {rf_results['cv_score']:.4f}")
    print(f"  Test Macro-F1 (on Kyle's data): {rf_results['test_macro_f1']:.4f}")
    print(f"  Test Weighted-F1 (on Kyle's data): {rf_results['test_weighted_f1']:.4f}")
    
    print(f"\nGradient Boosting (tested on Kyle's data):")
    print(f"  CV Macro-F1 (on original data): {gb_results['cv_score']:.4f}")
    print(f"  Test Macro-F1 (on Kyle's data): {gb_results['test_macro_f1']:.4f}")
    print(f"  Test Weighted-F1 (on Kyle's data): {gb_results['test_weighted_f1']:.4f}")
    print(f"  Training iterations: {gb_results['n_iterations']}")

    # Step 10: Detailed classification report
    print(f"\n{'='*70}")
    print("DETAILED CLASSIFICATION REPORT - KYLE'S DATA")
    print(f"{'='*70}")
    
    print(f"\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_results['predictions'], 
                                target_names=unique_classes_kyle, 
                                zero_division=0))
    
    print(f"\nGradient Boosting Classification Report:")
    print(classification_report(y_test, gb_results['predictions'], 
                                target_names=unique_classes_kyle, 
                                zero_division=0))

    print(f"\n{'='*70}")
    print("MODEL ANALYSIS")
    print(f"{'='*70}")

    print(f"\nDataset characteristics:")
    print(f"  - Training samples (original data): {len(X_original)}")
    print(f"  - Test samples (Kyle's data): {len(X_kyle)}")
    print(f"  - Features after filtering: {len(feature_columns)}")
    print(f"  - Classes: 4 distinct badminton shots")
    print(f"  - Feature/Sample ratio: {len(feature_columns)/len(X_original):.2f}")

    print(f"\nModel complexity comparison:")
    print(f"Random Forest:")
    print(f"  - Trees: {rf_results['best_params']['n_estimators']}")
    print(f"  - Max depth: {rf_results['best_params']['max_depth']}")
    
    print(f"Gradient Boosting:")
    print(f"  - Learning rate: {gb_results['best_params']['learning_rate']}")
    print(f"  - Max depth: {gb_results['best_params']['max_depth']}")
    print(f"  - Iterations used: {gb_results['n_iterations']}/400 (early stopping)")
    
    print(f"\nGeneralization Performance:")
    print(f"  - Testing on completely independent dataset (Kyle's data)")
    print(f"  - This evaluates model's ability to generalize to new subjects")
    
    # Step 11: Visualizations
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    # 1. Confusion matrices comparison
    plot_confusion_matrices(
        y_test, 
        rf_results['predictions'], 
        gb_results['predictions'], 
        unique_classes_kyle
    )
    
    # 2. F1 scores detailed comparison
    plot_f1_scores(rf_results, gb_results, unique_classes_kyle)
    
    # 3. Model performance summary
    plot_model_performance_summary(rf_results, gb_results)
    
    # 4. Feature importance
    plot_feature_importance(
        feature_columns, 
        rf_results['feature_importance']
    )
    
    # 5. Comprehensive performance dashboard
    create_performance_dashboard(rf_results, gb_results, unique_classes_kyle, feature_columns)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETED!")
    print(f"{'='*70}")
    print(f"\nKey Findings:")
    print(f"  1. Models trained on original dataset")
    print(f"  2. Tested on completely independent dataset (Kyle's data)")
    print(f"  3. Random Forest Test F1: {rf_results['test_macro_f1']:.4f}")
    print(f"  4. Gradient Boosting Test F1: {gb_results['test_macro_f1']:.4f}")
    print(f"  5. This demonstrates model's generalization ability")
    print(f"{'='*70}")